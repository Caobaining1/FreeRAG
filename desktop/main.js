'use strict';

/**
 * freerag desktop shell — Electron main process.
 *
 * Owns the window and bridges the renderer to the Go kernel over JSON-RPC 2.0 /
 * NDJSON on the kernel's stdio (docs/plan.md §3).
 *
 * The renderer never touches child_process: it calls window.freerag.rpc(...),
 * which arrives here as ipcMain.handle('rpc'); this process frames the request,
 * matches responses to requests by id, and resolves the promise.
 */

const { app, BrowserWindow, dialog, ipcMain, nativeTheme } = require('electron');
const { spawn, spawnSync } = require('node:child_process');
const os = require('node:os');
const path = require('node:path');
const fs = require('node:fs');
const readline = require('node:readline');

/**
 * How long a request may stay unanswered, per method.
 *
 * Two numbers, because "slow" and "stuck" are different things. `idle` is how
 * long the kernel may say *nothing at all*; `total` is the hard ceiling for a
 * call that keeps reporting progress but never finishes.
 *
 * A flat 60 s timeout is the wrong shape for these two calls. Measured: `ask`
 * takes ~113 s end to end, and the first 60–70 s of that is the model when
 * deciding what to retrieve — during which the kernel emits nothing, because
 * the agent loop only reports a step once it completes. So a call that is
 * working perfectly looks identical to a hang for exactly as long as the
 * timeout allows. `index` is slower still, and equally silent while the sidecar
 * parses a long document.
 */
const TIMEOUTS = {
  ask: { idle: 300000, total: 1800000 },
  index: { idle: 600000, total: 3600000 },
};
const DEFAULT_TIMEOUT = { idle: 60000, total: 60000 };

// Kernel log lines are buffered so a window opened after the kernel started can
// still show them: the kernel is spawned before the window in app.whenReady.
const logBuffer = [];
const MAX_LOG_BUFFER = 500;

/**
 * Locates everything the app needs to run, in a checkout or in an install.
 *
 * These layouts are genuinely different and both must work: in a checkout the
 * code sits beside the models and a venv; installed, the code lives inside the
 * asar while the rest is under `resources`. Getting this wrong does not fail
 * loudly — the kernel falls back to its own repo-relative defaults and either
 * cannot find a model or, worse, cannot write its cache and silently stops
 * caching. So every path is resolved here and handed over explicitly.
 */
function resolveRuntime() {
  const installed = app.isPackaged;
  const resources = process.resourcesPath;
  const dev = path.resolve(__dirname, '..');

  return {
    installed,
    bin: installed ? path.join(resources, 'bin') : path.join(dev, 'bin'),
    models: installed ? path.join(resources, 'models') : path.join(dev, 'models'),
    sidecar: installed ? path.join(resources, 'sidecar') : path.join(dev, 'sidecar'),
    qdrant: installed ? path.join(resources, 'bin', 'qdrant') : path.join(dev, '.toolchain', 'qdrant', 'qdrant'),
    // Installed, this is the interpreter vendored by fetch-python-runtime.sh,
    // which carries pymupdf and onnxruntime with it. In a checkout it is the
    // project venv. Neither is optional: `python3` on PATH on a clean machine
    // has none of the sidecar's dependencies.
    python: installed
      ? bundledPython(path.join(resources, 'python'))
      : path.join(dev, '.venv314', 'bin', 'python'),
  };
}

/** Path to the interpreter inside a vendored python-build-standalone tree. */
function bundledPython(root) {
  // POSIX layouts put it in bin/, Windows puts python.exe at the top level.
  return process.platform === 'win32'
    ? path.join(root, 'python.exe')
    : path.join(root, 'bin', 'python3');
}

/** Absolute path of the Go kernel binary for this platform. */
function kernelBinary(runtime) {
  const name = process.platform === 'win32' ? 'freerag.exe' : 'freerag';
  return path.join(runtime.bin, name);
}

/**
 * Reads the repository's `.env`, if there is one.
 *
 * The kernel reads real environment variables, and the documented workflow is
 * `set -a && . ./.env && set +a` before `npm start`. That works, and it fails
 * silently the one time it is forgotten: the kernel logs "dense retrieval
 * disabled", `hybrid_search` degrades to keyword-only, and the only symptom is
 * worse answers. Loading the file here removes the step.
 *
 * `process.env` wins over the file, so an explicitly exported value (a shell
 * export, a CI override) is never shadowed. Only the source tree has this file;
 * an installed bundle does not, which is why a missing file is not an error.
 */
function loadDotEnv(directory) {
  const values = {};
  let raw;
  try {
    raw = fs.readFileSync(path.join(directory, '.env'), 'utf8');
  } catch {
    return values;
  }
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (trimmed === '' || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    // Strip one layer of matching quotes, the way dotenv does.
    if (value.length >= 2 && (value[0] === '"' || value[0] === "'") && value[value.length - 1] === value[0]) {
      value = value.slice(1, -1);
    }
    if (key) values[key] = value;
  }
  return values;
}

/**
 * Environment handed to the kernel, which forwards the paths it cares about to
 * the Python sidecar.
 *
 * Writable locations go under `userData`, never beside the executable: an
 * installed app bundle is read-only, and a cache that cannot be written is
 * silently a cache that does not exist — so the app would look like it works
 * while re-parsing every document from scratch.
 */
/**
 * Settings the user can change from the UI, and the values in force before they
 * change anything.
 *
 * The app owns these rather than the kernel, because prefs.json lives in
 * userData and the kernel is never told where that is. The kernel receives them
 * as environment at spawn, and again over `settings_set` whenever they change —
 * so a change applies to everything the kernel reads per call without a restart.
 */
const SETTINGS_DEFAULTS = {
  // '' follows the language of the question; 'zh' or 'en' forces it.
  answerLanguage: '',
  vision: true,
  // Which describer handles figures: 'general' is the sidecar's vision model,
  // 'chart' hands statistical charts to Laya-Chart (chartvlm/server.py), which
  // transcribes them into a data table instead of describing them.
  //
  // Not a third state of `vision` on purpose: the two are mutually exclusive and
  // 'general' has to stay the default, because it describes a figure BEFORE the
  // figure/caption merge while the chart path runs after the parse.
  visionBackend: 'general',
  visionModel: 'qwen2.5vl:3b',
  // Measured on the reference machine (docs/plan.md §5.4.1): 4 workers gave
  // 2.75x over 1, 155 is what the model actually writes for a figure, and the
  // longest side caps the image tokens that cost prefill.
  visionWorkers: 4,
  visionMaxTokens: 155,
  visionMaxSide: 768,
};

/** Reads the settings, ignoring anything whose type is not the expected one. */
function readSettings() {
  const stored = readPrefs().data || {};
  const settings = { ...SETTINGS_DEFAULTS };
  for (const key of Object.keys(SETTINGS_DEFAULTS)) {
    if (stored[key] !== null && typeof stored[key] === typeof SETTINGS_DEFAULTS[key]) {
      settings[key] = stored[key];
    }
  }
  return settings;
}

/**
 * Validates a partial settings update, returning either the merged result or the
 * reasons it was refused.
 *
 * Refused rather than coerced, matching the theme below: a value that is stored,
 * survives, and is then quietly ignored is a choice the user made that did
 * nothing — and the kernel refuses the same values, so accepting them here would
 * only move the lie.
 */
function validateSettings(incoming) {
  const settings = readSettings();
  const errors = [];

  if ('answerLanguage' in incoming) {
    if (['', 'zh', 'en'].includes(incoming.answerLanguage)) {
      settings.answerLanguage = incoming.answerLanguage;
    } else {
      errors.push(`unknown answer language: ${JSON.stringify(incoming.answerLanguage)}`);
    }
  }
  if ('vision' in incoming) settings.vision = Boolean(incoming.vision);
  if ('visionBackend' in incoming) {
    if (['general', 'chart'].includes(incoming.visionBackend)) {
      settings.visionBackend = incoming.visionBackend;
    } else {
      errors.push(`unknown vision backend: ${JSON.stringify(incoming.visionBackend)}`);
    }
  }
  if ('visionModel' in incoming) {
    const model = String(incoming.visionModel || '').trim();
    if (!model) errors.push('vision model must not be empty');
    else settings.visionModel = model;
  }
  for (const [key, minimum] of [['visionWorkers', 1], ['visionMaxTokens', 1], ['visionMaxSide', 64]]) {
    if (!(key in incoming)) continue;
    const value = Number(incoming[key]);
    if (!Number.isInteger(value) || value < minimum) {
      errors.push(`${key} must be a whole number >= ${minimum}`);
    } else {
      settings[key] = value;
    }
  }
  return errors.length ? { errors } : { settings };
}

/** The kernel's spelling of the same settings, which is snake_case on the wire. */
function kernelSettingsPayload(settings) {
  return {
    answer_language: settings.answerLanguage,
    vision: settings.vision,
    vision_model: settings.visionModel,
    vision_workers: settings.visionWorkers,
    vision_max_tokens: settings.visionMaxTokens,
    vision_max_side: settings.visionMaxSide,
  };
}

function kernelEnv(runtime, qdrantReady, hardware) {
  const userData = app.getPath('userData');
  const env = {
    ...process.env,
    // Secrets from the source tree's `.env` (dense retrieval's key lives there).
    // Kept after process.env so an explicit export still wins.
    ...loadDotEnv(path.join(__dirname, '..')),
    FREERAG_DATA: path.join(userData, 'data', 'index.json'),
    FREERAG_CACHE_DIR: path.join(userData, 'cache', 'parse'),
    FREERAG_LAYOUT_MODEL: path.join(runtime.models, 'deepdoc', 'layout.onnx'),
    FREERAG_TSR_MODEL: path.join(runtime.models, 'deepdoc', 'tsr.onnx'),
    FREERAG_LAYA_DIR: path.join(runtime.models, 'laya-onnx'),
    FREERAG_PARSE_SIDECAR: path.join(runtime.sidecar, 'parse_server.py'),
    // Total memory, from the scan at the top of whenReady. Handed over because
    // the kernel is the process that would load several GB of vision weights
    // into it, and a model that does not fit does not fail — it swaps, and the
    // machine stops responding while indexing runs.
    FREERAG_HW_MEMORY_BYTES: String(hardware?.memoryBytes || 0),
  };
  // What the user chose last time. Environment rather than an RPC because these
  // have to be in force before the first parse, and because the sidecar reads
  // the shared ones (FREERAG_VLM_*) itself.
  const settings = readSettings();
  env.FREERAG_ANSWER_LANGUAGE = settings.answerLanguage;
  // 'on' is deliberately not 'go': the kernel's own vision path is opt-in and
  // separate from the sidecar's, and turning it on here would describe every
  // figure twice. Only 'chart' switches the figure path, because that backend
  // has no sidecar equivalent — it transcribes rather than describes.
  env.FREERAG_VLM = !settings.vision
    ? 'off'
    : settings.visionBackend === 'chart'
      ? 'chart'
      : 'on';
  // The model name is what the parse-time captioner keys off (sidecar/vlm.py
  // takes it as `vlm_model` on the parse request), so switching vision OFF has
  // to clear it — an empty name is how that path is disabled.
  env.FREERAG_VLM_MODEL = settings.vision ? settings.visionModel : '';
  env.FREERAG_VLM_CONCURRENCY = String(settings.visionWorkers);
  env.FREERAG_VLM_MAX_TOKENS = String(settings.visionMaxTokens);
  env.FREERAG_VLM_MAX_SIDE = String(settings.visionMaxSide);
  // Told about Qdrant only once something is actually answering there. The
  // kernel decides between the remote index and the in-process scan at startup,
  // so this is the only moment the choice can be made — omitting it silently
  // runs the whole session on the slow path, which looks like the index simply
  // not helping rather than like a misconfiguration.
  //
  // Not set when Qdrant is absent: the kernel would then log a connection
  // failure on every launch, and a warning that always fires is one nobody
  // reads.
  if (qdrantReady) env.FREERAG_QDRANT_URL = QDRANT_URL;
  // Told explicitly, never left to PATH discovery. "python3" on a machine that
  // never ran this project is the system interpreter, which has none of the
  // sidecar's dependencies — and that failure shows up as a parse error on the
  // first document rather than as a startup error, so it reads as "the app is
  // broken" instead of "a dependency is missing". Only set when the file is
  // actually there: a wrong-but-present path would be worse than none, since
  // the kernel reports the absence but cannot check a promise.
  if (fs.existsSync(runtime.python)) {
    env.FREERAG_PYTHON = runtime.python;
  }
  return env;
}

const QDRANT_URL = 'http://127.0.0.1:6333';
// Generous on purpose: a cold start unpacks its storage, and giving up early
// costs the whole session's ANN path for the sake of a couple of seconds.
const QDRANT_READY_TIMEOUT_MS = 20000;

/** Resolves true once something answers on the Qdrant URL. */
async function qdrantAnswers(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1500) });
      if (response.ok) return true;
    } catch {
      // Not listening yet, which is the expected answer for the first seconds.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

/**
 * Makes sure a Qdrant is answering before returning.
 *
 * Must be awaited before the kernel starts: the kernel decides *at startup*
 * whether dense retrieval is remote or in-process, so starting it first means
 * silently running the whole session on the slow path — which looks like the
 * ANN index simply not helping.
 *
 * Optional by design: without it the kernel ranks dense results with the
 * in-process scan. It is managed here rather than by the kernel because the
 * kernel is deliberately unaware of process management — it only speaks to a
 * URL.
 */
async function startQdrant(runtime, onLog) {
  if (await qdrantAnswers(QDRANT_URL, 1200)) {
    onLog(`qdrant already running at ${QDRANT_URL}`);
    return { child: null, ready: true };
  }

  const binary = runtime.qdrant;
  if (!fs.existsSync(binary)) {
    onLog(`qdrant not bundled (${binary}); dense retrieval will run in-process`);
    return { child: null, ready: false };
  }

  const storage = path.join(app.getPath('userData'), 'qdrant');
  // Not optional: the child is spawned with this as its working directory, and
  // spawn fails with ENOENT when it does not exist yet — which on a first run
  // is always.
  fs.mkdirSync(storage, { recursive: true });

  const child = spawn(binary, [], {
    stdio: ['ignore', 'pipe', 'pipe'],
    cwd: storage,
    env: { ...process.env, QDRANT__STORAGE__STORAGE_PATH: storage },
  });
  child.stderr.on('data', (data) => onLog(`qdrant: ${String(data).trim()}`));
  child.on('error', (error) => onLog(`qdrant failed to start: ${error.message}`));

  if (await qdrantAnswers(QDRANT_URL, QDRANT_READY_TIMEOUT_MS)) {
    onLog(`qdrant ready at ${QDRANT_URL}`);
    return { child, ready: true };
  }
  onLog('qdrant did not become ready; dense retrieval will run in-process');
  return { child, ready: false };
}

/**
 * KernelClient spawns the Go kernel and speaks NDJSON JSON-RPC to it.
 *
 * Responses are matched by id; stderr lines are forwarded as logs (stdout is
 * the protocol stream only — see cmd/freerag/main.go).
 */
class KernelClient {
  constructor({ binPath, env, onLog, onExit, onEvent }) {
    this.binPath = binPath;
    this.env = env || process.env;
    this.onLog = onLog || (() => {});
    this.onExit = onExit || (() => {});
    // onEvent receives kernel-initiated messages (progress and the like).
    this.onEvent = onEvent || (() => {});
    this.nextId = 1;
    this.pending = new Map();
    this.proc = null;
    this.ready = false;
    this.sweepTimer = null;
  }

  start() {
    if (!fs.existsSync(this.binPath)) {
      this.onLog(
        `kernel binary not found: ${this.binPath}\n` +
          'build it first:  go build -o bin/freerag ./cmd/freerag'
      );
      return false;
    }

    this.proc = spawn(this.binPath, [], { stdio: ['pipe', 'pipe', 'pipe'], env: this.env });

    readline.createInterface({ input: this.proc.stdout }).on('line', (line) => {
      const text = line.trim();
      if (!text) return;
      let msg;
      try {
        msg = JSON.parse(text);
      } catch {
        this.onLog(`kernel sent non-JSON on stdout: ${text.slice(0, 400)}`);
        return;
      }
      this._settle(msg);
    });

    readline.createInterface({ input: this.proc.stderr }).on('line', (line) => {
      if (line.trim()) this.onLog(line);
    });

    this.proc.on('error', (err) => this.onLog(`kernel process error: ${err.message}`));
    this.proc.on('exit', (code, signal) => {
      this.ready = false;
      this.onLog(`kernel exited (code=${code} signal=${signal})`);
      this.stopSweep();
      for (const [, pending] of this.pending) {
        pending.reject(new Error('kernel exited before responding'));
      }
      this.pending.clear();
      this.onExit();
    });

    // One clock for all requests rather than a timer each: idle time is a
    // property of the connection, so a single sweep is both simpler and exact.
    // unref'd so a pending sweep never keeps Electron alive on its own.
    if (!this.sweepTimer) {
      this.sweepTimer = setInterval(() => this._sweep(), 1000);
      if (this.sweepTimer.unref) this.sweepTimer.unref();
    }

    this.ready = true;
    this.onLog(`kernel started: ${this.binPath}`);
    return true;
  }

  /** Routes one inbound message: a response if it has an id, an event if not. */
  _settle(msg) {
    // Anything arriving from the kernel proves it is alive and working, so it
    // resets the idle clock for every request still in flight. This is what
    // makes the timeout mean "no progress" rather than "slow": a call that
    // reports a step every 30 s can run for half an hour, while one that goes
    // quiet is caught.
    this._touch();

    // A JSON-RPC notification carries no id because it was never requested.
    // Looking it up in `pending` would find nothing and drop it silently,
    // which is exactly how progress would go missing.
    if (msg.id === undefined || msg.id === null) {
      this.onEvent(msg);
      return;
    }

    const pending = this.pending.get(msg.id);
    if (!pending) return;
    this.pending.delete(msg.id);

    if (msg.error) {
      const err = new Error(msg.error.message || 'kernel error');
      err.code = msg.error.code;
      err.data = msg.error.data;
      pending.reject(err);
    } else {
      pending.resolve(msg.result);
    }
  }

  /** Sends one request and resolves with its result. */
  call(method, params, limits) {
    if (!this.ready || !this.proc) {
      return Promise.reject(new Error('kernel is not running'));
    }
    const budget = limits || TIMEOUTS[method] || DEFAULT_TIMEOUT;
    const id = this.nextId++;
    const request = { jsonrpc: '2.0', id, method };
    if (params !== undefined && params !== null) request.params = params;

    return new Promise((resolve, reject) => {
      const now = Date.now();
      this.pending.set(id, {
        method,
        resolve,
        reject,
        idle: budget.idle,
        total: budget.total,
        startedAt: now,
        lastAt: now,
      });
      this.proc.stdin.write(JSON.stringify(request) + '\n', (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  /** Marks the kernel as alive right now, for every in-flight request. */
  _touch() {
    const now = Date.now();
    for (const [, pending] of this.pending) pending.lastAt = now;
  }

  /** Fails requests that have gone quiet for too long, or run too long overall. */
  _sweep() {
    const now = Date.now();
    for (const [id, pending] of this.pending) {
      const quiet = now - pending.lastAt;
      // Idle time is reported first: when both limits are exceeded it is the
      // silence that explains the failure, not the total.
      let reason = null;
      if (quiet > pending.idle) {
        reason = `no progress for ${Math.round(quiet / 1000)}s`;
      } else if (now - pending.startedAt > pending.total) {
        reason = `still running after ${Math.round(pending.total / 1000)}s`;
      }
      if (!reason) continue;

      this.pending.delete(id);
      pending.reject(new Error(`rpc ${reason}: ${pending.method}`));
    }
  }

  /** Stops the timeout sweep. Safe to call more than once. */
  stopSweep() {
    if (this.sweepTimer) clearInterval(this.sweepTimer);
    this.sweepTimer = null;
  }

  stop() {
    this.stopSweep();
    if (this.proc && !this.proc.killed) this.proc.kill();
  }
}

let win = null;
let kernel = null;
let qdrant = null;

function broadcast(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

function createWindow() {
  const { theme } = resolveTheme();
  win = new BrowserWindow({
    width: 1080,
    height: 760,
    minWidth: 860,
    minHeight: 560,
    // Painted before any content exists. Paired with the preload setting
    // data-theme, this is what keeps a light-theme start from flashing a black
    // window — the stylesheet's own default is the dark palette.
    backgroundColor: THEME_BACKGROUND[theme] || THEME_BACKGROUND[DEFAULT_THEME],
    title: 'freerag',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      // Handed to the preload rather than fetched over IPC: IPC is asynchronous,
      // and a theme applied after the first paint is a theme applied too late.
      additionalArguments: [`--freerag-theme=${theme}`],
    },
  });
  // The hash is passed through so a view can be linked to, and so the
  // screenshot affordance below can capture either one for the docs.
  const initialView = process.env.FREERAG_SCREENSHOT_VIEW;
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'), {
    hash: initialView || undefined,
  });

  // Dev / CI affordance: FREERAG_SCREENSHOT=<path> captures the rendered window
  // once it has painted, then exits. It captures the app page only (never the
  // desktop), so UI screenshots for docs can be produced without a screen
  // recording permission prompt.
  const screenshotPath = process.env.FREERAG_SCREENSHOT;
  if (screenshotPath) {
    win.webContents.once('did-finish-load', () => {
      setTimeout(async () => {
        try {
          const image = await win.webContents.capturePage();
          fs.writeFileSync(screenshotPath, image.toPNG());
          console.log(`screenshot written: ${screenshotPath}`);
        } catch (err) {
          console.error(`screenshot failed: ${err.message}`);
        } finally {
          app.quit();
        }
      }, 2500);
    });
  }

  win.on('closed', () => {
    win = null;
  });
}

/** One place for log lines, so any child process's output is handled alike. */
function emitLog(line) {
  logBuffer.push(line);
  if (logBuffer.length > MAX_LOG_BUFFER) logBuffer.shift();
  broadcast('kernel-log', line);

  // Also echoed to the terminal. Without this, launching from a shell shows
  // nothing at all — so a misconfiguration the kernel reports loudly (dense
  // index disabled, sidecar missing, model not found) is only visible if you
  // are already looking at the in-app log panel, which is exactly the state in
  // which you would not be looking for it.
  console.log(line);
}

/**
 * Path of the chat history file.
 *
 * Under userData rather than beside the executable: an installed app bundle is
 * read-only, and a history that cannot be written is silently no history at all.
 */
function chatsPath() {
  return path.join(app.getPath('userData'), 'chats.json');
}

/** Reads the chat history, reporting an empty one when there is none. */
function readChats() {
  try {
    const parsed = JSON.parse(fs.readFileSync(chatsPath(), 'utf8'));
    // Guarded rather than trusted: the shape is written by this app, but a
    // file that lost its sessions, or that someone edited, would otherwise
    // fail somewhere deeper with a message about undefined.
    if (!parsed || !Array.isArray(parsed.sessions)) {
      return { ok: true, data: { sessions: [] } };
    }
    return { ok: true, data: parsed };
  } catch (error) {
    // Absent is the normal first run. Corrupt is not: it is reported so the
    // file can be recovered, rather than silently replaced with an empty
    // history that looks like the conversations were never there.
    if (error.code === 'ENOENT') return { ok: true, data: { sessions: [] } };
    return { ok: false, error: error.message };
  }
}

/**
 * Writes the chat history atomically.
 *
 * Temporary file plus rename: this is the only copy of every conversation, and
 * a crash partway through an in-place write would leave a truncated file that
 * parses as nothing.
 */
function writeChats(payload) {
  try {
    const target = chatsPath();
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const temporary = `${target}.tmp`;
    fs.writeFileSync(temporary, JSON.stringify(payload, null, 2), 'utf8');
    fs.renameSync(temporary, target);
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

/**
 * Path of the preferences file.
 *
 * Beside the chat history and under userData for the same reason: an installed
 * bundle is read-only, and a preference that cannot be written is silently no
 * preference at all.
 */
function prefsPath() {
  return path.join(app.getPath('userData'), 'prefs.json');
}

/** The two themes, and the colour each paints the window before content exists. */
const THEME_BACKGROUND = { dark: '#0f1115', light: '#eef1f5' };
const THEMES = Object.keys(THEME_BACKGROUND);
const DEFAULT_THEME = 'dark';

/** Reads the preferences, reporting none when there are none. */
function readPrefs() {
  try {
    const parsed = JSON.parse(fs.readFileSync(prefsPath(), 'utf8'));
    if (!parsed || typeof parsed !== 'object') return { ok: true, data: {} };
    return { ok: true, data: parsed };
  } catch (error) {
    if (error.code === 'ENOENT') return { ok: true, data: {} };
    return { ok: false, error: error.message };
  }
}

/** Writes the preferences atomically, the way the chat history is written. */
function writePrefs(payload) {
  try {
    const target = prefsPath();
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const temporary = `${target}.tmp`;
    fs.writeFileSync(temporary, JSON.stringify(payload, null, 2), 'utf8');
    fs.renameSync(temporary, target);
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

/**
 * What this machine is, scanned once and cached.
 *
 * Scanned rather than read off `process.platform` because the two disagree in
 * exactly the cases that matter: a Windows box with no NVIDIA driver, and one
 * whose driver does not match the CUDA runtime our onnxruntime was built
 * against, look identical from Node. Asking the driver is the only way to
 * tell them apart — and the difference decides whether "auto" resolves to
 * CoreML/CUDA or silently to CPU.
 *
 * Cached because it cannot change without the hardware changing, and because
 * the scan shells out: once per install, not once per launch.
 */
const HARDWARE_SCHEMA = 1;

function hardwarePath() {
  return path.join(app.getPath('userData'), 'hardware.json');
}

/** Asks the NVIDIA driver what it has, or null when there is nothing to ask. */
function nvidiaGpu() {
  try {
    const run = spawnSync('nvidia-smi',
      ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
      { encoding: 'utf8', timeout: 4000, windowsHide: true });
    if (run.status !== 0 || !run.stdout) return null;
    const [name, memory] = String(run.stdout).trim().split('\n')[0].split(',');
    // nvidia-smi reports MiB; everything downstream counts bytes.
    return {
      vendor: 'nvidia',
      backend: 'cuda',
      name: (name || '').trim(),
      memoryBytes: Number(memory || 0) * 1024 * 1024,
    };
  } catch {
    return null;
  }
}

function scanHardware() {
  const cpus = os.cpus();
  // "metal" rather than "Apple Silicon": it is the name the runtimes ask for,
  // and an Intel Mac with a supported GPU answers the same way.
  const accelerator = nvidiaGpu() || (process.platform === 'darwin'
    ? { vendor: 'apple', backend: 'metal', name: cpus[0]?.model || '', memoryBytes: 0 }
    : { vendor: '', backend: 'cpu', name: '', memoryBytes: 0 });
  return {
    schemaVersion: HARDWARE_SCHEMA,
    scannedAt: new Date().toISOString(),
    platform: process.platform,
    arch: process.arch,
    cpu: { model: cpus[0]?.model || '', cores: cpus.length },
    memoryBytes: os.totalmem(),
    accelerator,
  };
}

/** Writes the scan atomically, the way the preferences are written. */
function writeHardware(payload) {
  try {
    const target = hardwarePath();
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const temporary = `${target}.tmp`;
    fs.writeFileSync(temporary, JSON.stringify(payload, null, 2), 'utf8');
    fs.renameSync(temporary, target);
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

/**
 * The cached scan, scanning on first launch or after the schema changes.
 *
 * Called for its result but also for its side effect: the first launch is when
 * this has to happen, and there is nothing to wait for.
 */
function readHardware() {
  try {
    const parsed = JSON.parse(fs.readFileSync(hardwarePath(), 'utf8'));
    if (parsed && parsed.schemaVersion === HARDWARE_SCHEMA) return parsed;
  } catch {
    // Absent, unreadable, or written by an older schema: all scan again.
  }
  const scanned = scanHardware();
  writeHardware(scanned);
  return scanned;
}

/**
 * Resolves the theme to open in.
 *
 * Three sources, in the order that makes each one useful:
 *
 *   1. FREERAG_THEME — an override, so a screenshot or a test can pin the theme
 *      whatever this machine's user last chose. It writes nothing back, so a
 *      dev session cannot silently change the stored preference.
 *   2. the saved preference — what the user picked, and by definition the only
 *      source that survives a restart.
 *   3. the OS — the default for someone who has never chosen. Opening a
 *      light-mode machine in a black window is a choice made for them, and the
 *      wrong one.
 *
 * The result is handed to the renderer rather than left to a CSS media query,
 * so there is exactly one place that decides and the UI's own toggle can
 * disagree with the OS without the stylesheet second-guessing it.
 */
function resolveTheme() {
  const forced = String(process.env.FREERAG_THEME || '').toLowerCase();
  if (THEMES.includes(forced)) return { theme: forced, source: 'env' };

  const saved = readPrefs();
  const stored = saved.ok ? String(saved.data.theme || '').toLowerCase() : '';
  if (THEMES.includes(stored)) return { theme: stored, source: 'prefs' };

  return { theme: nativeTheme.shouldUseDarkColors ? 'dark' : 'light', source: 'system' };
}

app.whenReady().then(async () => {
  const runtime = resolveRuntime();

  // Scanned at launch, on the first one and on any launch whose cached scan is
  // from an older schema — not when the settings dialog happens to be opened.
  // The scan is what decides which inference backends are worth asking for, and
  // a first document parsed before anyone opened settings would otherwise be
  // the thing that discovers it.
  const hardware = readHardware();
  // Said out loud, because it is the one line that explains a machine running
  // slower than the user expects: "cpu" here means every model will, too.
  emitLog(`hardware: ${hardware.accelerator?.backend || 'cpu'} · ` +
    `${hardware.cpu?.cores || '?'} cores · ${(hardware.memoryBytes / 1024 ** 3).toFixed(0)} GB`);

  // Awaited on purpose: the kernel decides at startup whether dense retrieval
  // is remote or in-process, so it must not start until Qdrant answers.
  const service = await startQdrant(runtime, emitLog);
  qdrant = service.child;

  kernel = new KernelClient({
    binPath: kernelBinary(runtime),
    env: kernelEnv(runtime, service.ready, hardware),
    onLog: emitLog,
    onExit: () => broadcast('kernel-status', { running: false }),
    onEvent: (msg) => broadcast('kernel-event', msg),
  });
  const started = kernel.start();

  // The renderer cannot open a native dialog (nodeIntegration is off), and a
  // dropped file only yields a real path through webUtils. Both requests land
  // here, in the process that is allowed to answer them.
  ipcMain.handle('pick-files', async () => {
    const picked = await dialog.showOpenDialog(win, {
      title: '选择要索引的文档',
      // The sidecar decides what it can parse (sidecar/documents.py); this list
      // only spares the user from picking an archive and getting an error. Keep
      // the two in step.
      filters: [
        { name: '文档', extensions: ['pdf', 'docx', 'doc', 'rtf', 'txt', 'md'] },
        { name: '全部文件', extensions: ['*'] },
      ],
      properties: ['openFile', 'multiSelections'],
    });
    return picked.canceled ? [] : picked.filePaths;
  });

  ipcMain.handle('rpc', async (_event, method, params) => {
    try {
      return { ok: true, result: await kernel.call(method, params) };
    } catch (err) {
      return { ok: false, error: { message: err.message, code: err.code, data: err.data } };
    }
  });

  ipcMain.handle('kernel-status', () => ({
    running: kernel.ready,
    binary: kernel.binPath,
  }));

  ipcMain.handle('kernel-log-history', () => logBuffer.slice());

  // Chat history is the shell's own data rather than the kernel's. It is a list
  // of questions and answers that nothing in the retrieval pipeline reads, so
  // keeping it here means one writer for the file and no RPC surface for data
  // the kernel never touches.
  ipcMain.handle('chat-load', () => readChats());
  ipcMain.handle('chat-save', (_event, payload) => writeChats(payload));

  // Preferences live here for the same reason the chat history does: nothing in
  // the retrieval pipeline reads which theme is on, so the kernel has no
  // business knowing about it.
  // The scan is reported as two things, and the difference is the point: what
  // the machine has, and what the models actually got. A machine can offer an
  // accelerator and still run on CPU because its runtime lacks the provider,
  // and that is silent everywhere except here.
  ipcMain.handle('hardware-load', () => readHardware());
  ipcMain.handle('hardware-rescan', () => {
    const scanned = scanHardware();
    writeHardware(scanned);
    return scanned;
  });

  ipcMain.handle('prefs-load', () => {
    const resolved = resolveTheme();
    return {
      ok: true,
      data: {
        theme: resolved.theme,
        source: resolved.source,
        themes: THEMES,
        settings: readSettings(),
        defaults: SETTINGS_DEFAULTS,
      },
    };
  });

  ipcMain.handle('prefs-save', async (_event, payload) => {
    const incoming = payload || {};
    const checked = validateSettings(incoming);
    if (checked.errors) {
      return { ok: false, error: checked.errors.join('; ') };
    }

    const next = { ...(readPrefs().data || {}), ...incoming };
    // Refused rather than coerced. A theme name this build does not have would
    // be written, survive, and then be silently ignored at the next start — a
    // choice the user made that quietly did nothing.
    if (next.theme !== undefined && !THEMES.includes(next.theme)) {
      return { ok: false, error: `unknown theme: ${next.theme}` };
    }
    // The validated values are what land in the file, so a rejected number cannot
    // get in through a field this build does not know about.
    Object.assign(next, checked.settings);

    const written = writePrefs(next);
    if (!written.ok) return written;

    // Pushed to the running kernel so the change applies now rather than at the
    // next launch. Reported on failure instead of hidden: the file and the
    // running app would otherwise disagree, and the UI would look correct.
    if (kernel) {
      try {
        await kernel.call('settings_set', kernelSettingsPayload(checked.settings));
      } catch (error) {
        return { ok: true, data: { ...next, settings: checked.settings, kernel_error: error.message } };
      }
    }
    return { ok: true, data: { ...next, settings: checked.settings } };
  });

  createWindow();
  broadcast('kernel-status', { running: started, binary: kernel.binPath });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
}).catch((error) => {
  // An async whenReady has no other place to report a failure: without this the
  // window simply never appears and the app looks hung.
  console.error('startup failed:', error);
  dialog.showErrorBox('freerag 启动失败', String(error && error.message ? error.message : error));
  app.quit();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (kernel) kernel.stop();
  // Left running, Qdrant would keep holding its port and the storage lock, so
  // the next launch finds a database it cannot open.
  if (qdrant && !qdrant.killed) qdrant.kill();
});
