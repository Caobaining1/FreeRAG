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

const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
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
 * Environment handed to the kernel, which forwards the paths it cares about to
 * the Python sidecar.
 *
 * Writable locations go under `userData`, never beside the executable: an
 * installed app bundle is read-only, and a cache that cannot be written is
 * silently a cache that does not exist — so the app would look like it works
 * while re-parsing every document from scratch.
 */
function kernelEnv(runtime, qdrantReady) {
  const userData = app.getPath('userData');
  const env = {
    ...process.env,
    FREERAG_DATA: path.join(userData, 'data', 'index.json'),
    FREERAG_CACHE_DIR: path.join(userData, 'cache', 'parse'),
    FREERAG_LAYOUT_MODEL: path.join(runtime.models, 'deepdoc', 'layout.onnx'),
    FREERAG_TSR_MODEL: path.join(runtime.models, 'deepdoc', 'tsr.onnx'),
    FREERAG_LAYA_DIR: path.join(runtime.models, 'laya-onnx'),
    FREERAG_PARSE_SIDECAR: path.join(runtime.sidecar, 'parse_server.py'),
  };
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
  win = new BrowserWindow({
    width: 1080,
    height: 760,
    minWidth: 860,
    minHeight: 560,
    backgroundColor: '#0f1115',
    title: 'freerag',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));

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

app.whenReady().then(async () => {
  const runtime = resolveRuntime();

  // Awaited on purpose: the kernel decides at startup whether dense retrieval
  // is remote or in-process, so it must not start until Qdrant answers.
  const service = await startQdrant(runtime, emitLog);
  qdrant = service.child;

  kernel = new KernelClient({
    binPath: kernelBinary(runtime),
    env: kernelEnv(runtime, service.ready),
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
      filters: [{ name: 'PDF', extensions: ['pdf'] }],
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
