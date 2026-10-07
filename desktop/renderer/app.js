'use strict';

/**
 * freerag renderer.
 *
 * Two views over one kernel. The knowledge-base view owns documents; the chat
 * view owns sessions, and a session is bound to one knowledge base for its
 * whole life — switching it later would silently reinterpret everything already
 * asked in it.
 *
 * Session and conversation history is stored by the shell (desktop/main.js,
 * chats.json), not by the kernel: it is a list of questions and answers that
 * nothing in the retrieval pipeline reads.
 */

const api = window.freerag;
const $ = (id) => document.getElementById(id);

const state = {
  view: 'kb',
  // Where the current theme came from: the user, FREERAG_THEME, or the OS. Held
  // so the toggle can say so — an inherited default presented as a choice is a
  // setting the user thinks they made.
  themeSource: '',
  // Whether each two-screen view is showing its second screen. The chat view
  // guards a SESSION, the knowledge view a BASE — and both ask rather than
  // assume, because the thing being chosen is what everything after it is
  // read against.
  inSession: false,
  inKB: false,
  // Knowledge bases, as the kernel lists them. `selectedKB` is an id.
  kbs: [],
  selectedKB: '',
  documents: [],

  // The chunk inspector, for the document currently open. One state object
  // rather than several: everything here is derived from "which document, which
  // page, which chunk", and separating them is how a box ends up drawn over the
  // page the reader has already left.
  inspectDoc: '',
  inspectChunks: [],
  // Per-page chunk counts, so the arrows know which pages exist before one is
  // chosen. Counting client-side avoids a round trip per page turn.
  inspectPages: [],
  inspectPage: 1,
  inspectSelected: '',
  // The rendered page's size IN POINTS — the space bbox is measured in, so these
  // two are what every highlight is divided by.
  inspectMeta: null,

  // Session and conversation history, persisted by the shell.
  sessions: [],
  selectedSession: '',
  selectedConversation: '',

  // Which of the right pane's two modes is showing.
  // `mode` rather than a boolean because a third is coming (quote verification,
  // once a passage can be checked against the section that produced it), and
  // string-of-four-states is not something to build twice.
  mode: 'chunks',

  // The corpus directory tree: its shape as the kernel reports it, and nothing
  // else. Retrieval over the tree happens in 问答, so there is no result to
  // keep here — the tree is the artefact, not a query window onto it.
  dirStatus: null,

  busy: false,
  startedAt: 0,
  timer: null,

  // The assistant message currently being streamed into, and the nodes its
  // text and trace are written to. Held so events can be appended without
  // re-rendering the transcript each time — against a detailed answer that
  // would be hundreds of re-renders.
  pending: null,
  pendingEl: null,
  pendingTraceEl: null,
  pendingTraceCountEl: null,
};

// ---------------------------------------------------------------------------
// theme
//
// The palettes live in styles.css; this is only the switch. Two themes, one
// attribute: every colour is a custom property under `:root`, so applying a
// theme is a single assignment and no component has to know a theme exists.

const THEMES = ['dark', 'light'];

/** The theme currently applied to the document. */
function currentTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

/**
 * Applies a theme without persisting it.
 *
 * `data-theme` rather than a class: it is a fact about the document, and the
 * light palette is written as `:root[data-theme='light']` — a selector that
 * says what it means rather than where it came from.
 *
 * An unknown name falls back to dark rather than being written through. A theme
 * that is applied but has no palette leaves every custom property undefined,
 * which renders as an unstyled page rather than as the wrong colours.
 */
function applyTheme(name) {
  const next = THEMES.includes(name) ? name : 'dark';
  document.documentElement.dataset.theme = next;
  renderThemeToggle();
  return next;
}

/** Applies a theme and remembers it. */
async function setTheme(name) {
  // An explicit choice is its own source: whatever the theme was inherited from
  // before is no longer what is on screen, and saying so would be a lie the
  // toggle then repeats.
  state.themeSource = '';
  const applied = applyTheme(name);
  const reply = await api.savePrefs({ theme: applied });
  if (reply && reply.ok === false) {
    // Reported rather than swallowed. A preference that failed to save looks
    // exactly like one that saved, right up until the next restart — which is
    // the moment the user finds out the toggle does not stick.
    notice(`主题已切换，但没能记住：${reply.error}`, 'warn');
  }
  return applied;
}

/** Switches to the other theme. */
function toggleTheme() {
  return setTheme(currentTheme() === 'light' ? 'dark' : 'light');
}

/** Paints the toggle for the theme it would switch TO, not the current one. */
function renderThemeToggle() {
  const button = $('theme-toggle');
  if (!button) return;

  const dark = currentTheme() === 'dark';
  const where = { env: '（被 FREERAG_THEME 固定）', system: '（跟随系统）' }[state.themeSource];
  button.textContent = dark ? '☀' : '☾';
  button.title = `${dark ? '切换到白昼主题' : '切换到黑夜主题'}${where || ''}`;
  button.setAttribute('aria-label', button.title);
}

/**
 * The theme interface.
 *
 * On `window` rather than kept private to this file, so that it is a named
 * surface: the topbar toggle, anything added later, and the DevTools console
 * all go through the same four calls, and the rules that matter — validate the
 * name before applying it, persist a change, report a write that failed — are
 * stated once instead of being re-derived by each caller.
 */
window.freeragTheme = {
  list: () => THEMES.slice(),
  current: currentTheme,
  set: setTheme,
  toggle: toggleTheme,
};

// ---------------------------------------------------------------------------
// plumbing

async function rpc(method, params) {
  const reply = await api.rpc(method, params);
  if (!reply.ok) {
    const error = new Error((reply.error && reply.error.message) || '未知错误');
    error.data = reply.error && reply.error.data;
    throw error;
  }
  return reply.result;
}

/**
 * Escapes text before it reaches innerHTML.
 *
 * Everything shown here is untrusted in principle: answers come from a language
 * model that has read the document, and a document may contain markup. Building
 * DOM nodes would be safer still, but escaping at the one place that writes HTML
 * is enough and much shorter.
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function show(id, visible) {
  $(id).classList.toggle('hidden', !visible);
}

function notice(text, kind) {
  const node = $('notice');
  node.textContent = text;
  node.className = 'notice' + (kind ? ' ' + kind : '');
  node.classList.toggle('hidden', !text);
}

function clearNotice() {
  notice('', '');
}

function setBusy(busy, label) {
  state.busy = busy;
  $('ask').disabled = busy;
  $('doc-add').disabled = busy;
  show('progress', busy);

  if (busy) {
    state.startedAt = Date.now();
    $('progress-log').textContent = '';
    $('progress-stage').textContent = label || '处理中…';
    state.timer = setInterval(() => {
      $('progress-elapsed').textContent = elapsed() + 's';
    }, 200);
  } else if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function elapsed() {
  return ((Date.now() - state.startedAt) / 1000).toFixed(1);
}

/**
 * Shows the prompt dialog and resolves with { name, kb } — or null if cancelled.
 *
 * `options` turns the knowledge-base picker on, which is what makes "a session
 * needs a knowledge base before it exists" a step in the flow rather than a
 * validation error afterwards.
 */
function askForInput({ title, placeholder, value, options }) {
  return new Promise((resolve) => {
    const dialog = $('prompt-dialog');
    $('prompt-title').textContent = title;
    $('prompt-input').value = value || '';
    $('prompt-input').placeholder = placeholder || '';

    const selectRow = $('prompt-select-row');
    const select = $('prompt-select');
    select.innerHTML = '';
    if (options && options.length) {
      options.forEach((option) => {
        const node = document.createElement('option');
        node.value = option.value;
        node.textContent = option.label;
        select.appendChild(node);
      });
      show('prompt-select-row', true);
    } else {
      show('prompt-select-row', false);
    }

    dialog.addEventListener(
      'close',
      () => {
        resolve(
          dialog.returnValue === 'ok'
            ? { name: $('prompt-input').value.trim(), kb: select.value }
            : null
        );
      },
      { once: true }
    );

    dialog.showModal();
    $('prompt-input').focus();
    $('prompt-input').select();
  });
}

// Enter in the name field must mean 确定. A form's implicit submission fires
// the FIRST submit button in tree order, which here is 取消 — so without this
// the very keystroke a user expects to confirm the new base/session/dialog
// closes the prompt as cancelled. Submitting the 确定 button explicitly also
// sets the dialog's returnValue to "ok", which is what askForInput reads.
$('prompt-input').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  $('prompt-form').requestSubmit($('prompt-ok'));
});

// ---------------------------------------------------------------------------
// health

async function refreshHealth() {
  try {
    // The index half of the health report belongs to a knowledge base, so the
    // one in view is named. Reporting the default base instead would show
    // figures for something the user is not looking at.
    const status = await rpc('status', state.selectedKB ? { kb: state.selectedKB } : {});
    renderHealth(status);
  } catch (error) {
    $('health').innerHTML = '<span class="chip bad">内核不可用</span>';
  }
}

function renderHealth(status) {
  const chips = [];

  const generator = status.generator || {};
  if (!generator.configured) {
    chips.push(['bad', '生成模型未配置']);
  } else if (generator.checked && generator.reachable === false) {
    chips.push(['bad', 'Ollama 不可达']);
  } else {
    chips.push(['ok', 'Ollama']);
  }

  const dense = status.dense_index || {};
  if (dense.backend) {
    chips.push(['ok', 'ANN: ' + dense.backend]);
  } else {
    chips.push(['warn', 'ANN: 进程内扫描']);
  }

  const embedding = status.embedding || {};
  if (embedding.enabled) {
    // Marked as unverified rather than healthy: probing a hosted embedder costs
    // a request on every poll, so this is configuration, not a health check.
    chips.push(['warn', '嵌入: 托管（未探测）']);
  } else {
    chips.push(['warn', '嵌入未配置']);
  }

  const sidecar = status.sidecar || {};
  if (!sidecar.configured) chips.push(['bad', '解析 sidecar 未找到']);

  $('health').innerHTML = chips
    .map(([kind, text]) => `<span class="chip ${kind}">${escapeHtml(text)}</span>`)
    .join('');
}

// ---------------------------------------------------------------------------
// views

function switchView(view) {
  state.view = view;
  $('tab-kb').classList.toggle('active', view === 'kb');
  $('tab-chat').classList.toggle('active', view === 'chat');
  $('view-kb').classList.toggle('hidden', view !== 'kb');
  $('view-chat').classList.toggle('hidden', view !== 'chat');

  // Entering either view always starts at its picker. Which session or base you
  // were in last time is not what you asked for now, and resuming one silently
  // is how a question gets typed into the wrong session — or a file added to the
  // wrong index.
  if (view === 'chat') leaveSession();
  if (view === 'kb') leaveKB();
  clearNotice();
}

// ---------------------------------------------------------------------------
// knowledge bases

async function refreshKBs() {
  const payload = await rpc('kb.list');
  state.kbs = payload.bases || [];

  // Keep the selection when it still exists, otherwise fall to the first. There
  // is always at least one base — the kernel refuses to delete the last — so
  // this never has to represent "none".
  if (!state.kbs.some((base) => base.id === state.selectedKB)) {
    state.selectedKB = state.kbs.length ? state.kbs[0].id : '';
  }
  renderKBs();
}

function renderKBs() {
  const list = $('kb-list');
  list.innerHTML = '';
  show('kb-empty', state.kbs.length === 0);

  state.kbs.forEach((base) => {
    const card = document.createElement('li');
    card.className = 'session-card';
    card.innerHTML =
      `<div class="session-card-main">` +
      `<span class="session-name">${escapeHtml(base.name)}</span>` +
      `<span class="list-meta">${base.doc_count || 0} 篇文档 · ${base.chunk_count || 0} chunks</span>` +
      `</div>` +
      `<span class="session-kb">${escapeHtml(shortDate(base.created_at))}</span>`;
    card.addEventListener('click', () => enterKB(base.id));
    list.appendChild(card);
  });
}

/** shortDate renders a timestamp as a date, or "" when there is none. */
function shortDate(value) {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleDateString();
}

/** Switches the knowledge view to its second screen. */
async function enterKB(id) {
  if (!state.kbs.some((base) => base.id === id)) return;

  state.inKB = true;
  state.selectedKB = id;
  // The inspector belongs to the base it was opened in. Carrying it across
  // would show one base's chunks under another base's name until the next fetch
  // landed — and nothing would say which of the two was on screen.
  resetInspector();
  // Belongs to the base being left: another base's directory tree would be
  // drawn over this base's name.
  state.dirStatus = null;
  // Another base's tree would open the same node ids, which exist there too and
  // mean something else — a left-over expansion state is a tree that appears to
  // have been explored when it has not.
  dirOpen.clear();
  dirOpen.add('__root__');
  state.mode = 'chunks';
  renderDirStatus();
  applyMode();
  show('kb-picker', false);
  show('kb-detail', true);

  await refreshDocuments();
  await refreshHealth();
  clearNotice();
}

/** Switches the knowledge view back to the picker. */
function leaveKB() {
  state.inKB = false;
  show('kb-picker', true);
  show('kb-detail', false);
  // Dropped rather than kept: the list belongs to a base that is no longer on
  // screen, and leaving it rendered would attribute one base's files to another.
  state.documents = [];
  renderDocumentList();
  renderKBs();
}

/** Applies a base summary returned by another call, without refetching. */
function patchKB(summary) {
  if (!summary || !summary.id) return;
  const index = state.kbs.findIndex((base) => base.id === summary.id);
  if (index >= 0) state.kbs[index] = { ...state.kbs[index], ...summary };
  else state.kbs.push(summary);
  renderKBs();
}

// ---------------------------------------------------------------------------
// the chunk inspector
//
// A chunk and the region it was cut from are one fact told twice: the page is
// what the reader can check against the original, the text is what retrieval
// actually sees. The box drawn over the page is the only thing that ties them
// together, so the single piece of state that matters is which chunk is
// selected — page, highlight and list all follow from it.

function resetInspector() {
  state.inspectDoc = '';
  state.inspectChunks = [];
  state.inspectPages = [];
  state.inspectPage = 1;
  state.inspectSelected = '';
  state.inspectMeta = null;
  applyMode();
}

/** Opens one document: all its chunks, and the first page they came from. */
async function openDocument(docId) {
  if (!state.selectedKB) return;
  if (state.inspectDoc === docId) {
    // Already open — so this click can only mean "show me that one again", which
    // is what "switch back to the chunks pane" is, not a no-op.
    setMode('chunks');
    return;
  }
  clearNotice();

  state.inspectDoc = docId;
  state.inspectSelected = '';

  let payload;
  try {
    payload = await rpc('chunks', { kb: state.selectedKB, doc_id: docId });
  } catch (error) {
    notice('无法读取分块：' + error.message, 'bad');
    state.inspectDoc = '';
    return;
  }

  // Every chunk of the document in ONE reply, so turning a page or toggling the
  // filter never waits on the kernel. What a page turn needs is a rendered page,
  // not another list.
  state.inspectChunks = payload.chunks || [];
  state.inspectPages = payload.pages || [];
  state.inspectPage = state.inspectPages.length ? state.inspectPages[0].page : 1;

  // Choosing a document while the search panel is up means the user wants to
  // look INSIDE it, so the pane follows rather than staying on a list of hits.
  state.mode = 'chunks';
  applyMode();
  renderDocumentList();
  renderChunkList();
  await loadPage(state.inspectPage);
}

/** Renders one page of the document's original file. */
async function loadPage(page) {
  state.inspectPage = page;
  $('page-label').textContent = `第 ${page} 页 · 载入中…`;
  $('origin-status').textContent = '';
  $('page-boxes').innerHTML = '';
  $('page-prev').disabled = true;
  $('page-next').disabled = true;

  let payload;
  try {
    payload = await rpc('page', { kb: state.selectedKB, doc_id: state.inspectDoc, page });
  } catch (error) {
    // The file may have moved since it was indexed, so this is a normal outcome
    // rather than a crash — and it is SAID, because an empty frame would read as
    // a rendering bug instead of a missing file.
    state.inspectMeta = null;
    $('page-image').removeAttribute('src');
    $('page-label').textContent = `第 ${page} 页`;
    $('origin-status').textContent = `原文无法显示：${error.message}`;
    renderBoxes();
    renderChunkList();
    return;
  }

  state.inspectMeta = payload;
  $('page-image').src = 'data:image/png;base64,' + payload.image;
  $('page-label').textContent = `第 ${payload.page} / ${payload.pages} 页 · ${payload.width_pt}×${payload.height_pt} pt`;
  $('origin-status').textContent = '';
  $('page-prev').disabled = payload.page <= 1;
  $('page-next').disabled = payload.page >= payload.pages;

  renderBoxes();
  renderChunkList();
}

/** Selects one chunk: highlight it on the right, show its origin on the left. */
async function selectChunk(chunkId) {
  state.inspectSelected = chunkId;
  const chunk = state.inspectChunks.find((item) => item.chunk_id === chunkId);

  renderChunkList();
  const node = $(`chunk-${chunkId}`);
  if (node) node.scrollIntoView({ block: 'nearest' });

  // The page turn is part of the selection, not a separate step. A chunk on
  // page 7 that highlights nothing — because the left pane is still showing
  // page 1 — is exactly the failure this screen exists to prevent.
  if (chunk && chunk.page_num && chunk.page_num !== state.inspectPage) {
    await loadPage(chunk.page_num);
    const box = $(`box-${chunkId}`);
    if (box) box.scrollIntoView({ block: 'center', behavior: 'smooth' });
    return;
  }

  renderBoxes();
  const box = $(`box-${chunkId}`);
  if (box) box.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

/**
 * Draws the boxes for the page on screen.
 *
 * The selected chunk only, by default: thirty overlapping rectangles is not a
 * locator, it is a heat map. The checkbox reveals the rest, which is how you see
 * that the chunks really do tile the page rather than leave gaps.
 *
 * One box per chunk, because one chunk is one rectangle on the page: the merge
 * only folds blocks that sit in the same column one below another (see
 * sidecar/chunking.py `_stacked`). Without that rule a chunk could join the
 * bottom of one column to the top of the next, and its box would cover the whole
 * corner between them.
 */
function renderBoxes() {
  const overlay = $('page-boxes');
  overlay.innerHTML = '';

  const meta = state.inspectMeta;
  if (!meta || !meta.width_pt || !meta.height_pt) return;

  const showAll = $('show-all').checked;
  state.inspectChunks
    .filter((chunk) => chunk.page_num === state.inspectPage)
    .forEach((chunk) => {
      if (!Array.isArray(chunk.bbox) || chunk.bbox.length !== 4) return;
      const selected = chunk.chunk_id === state.inspectSelected;
      if (!selected && !showAll) return;

      const [x0, y0, x1, y1] = chunk.bbox;
      const box = document.createElement('div');
      box.className = 'box' + (selected ? ' selected' : '');
      box.id = `box-${chunk.chunk_id}`;
      // Percentages of the page's size IN POINTS, which is the space bbox is
      // measured in. Percentages rather than pixels so the overlay follows the
      // image when the pane is resized — no listener, no arithmetic on resize.
      box.style.left = `${(x0 / meta.width_pt) * 100}%`;
      box.style.top = `${(y0 / meta.height_pt) * 100}%`;
      box.style.width = `${((x1 - x0) / meta.width_pt) * 100}%`;
      box.style.height = `${((y1 - y0) / meta.height_pt) * 100}%`;
      box.title = `${chunk.chunk_id} · ${chunk.block_type || '?'} · p.${chunk.page_num}`;
      box.addEventListener('click', () => selectChunk(chunk.chunk_id));
      overlay.appendChild(box);
    });
}

/**
 * Renders the chunks of the open document.
 *
 * "Only this page" is a VIEW filter over data already in hand, never a fetch:
 * that is what lets the reader turn a page and immediately see which chunks
 * share it, without waiting on the kernel twice.
 */
function renderChunkList() {
  const list = $('chunks');
  list.innerHTML = '';

  const onlyPage = $('only-page').checked;
  const rows = onlyPage
    ? state.inspectChunks.filter((chunk) => chunk.page_num === state.inspectPage)
    : state.inspectChunks;

  const doc = state.documents.find((item) => item.doc_id === state.inspectDoc);
  const name = (doc && (doc.source_file || doc.doc_id)) || state.inspectDoc;
  $('chunk-summary').textContent = onlyPage
    ? `${name} · 本页 ${rows.length} / 全文 ${state.inspectChunks.length} 块`
    : `${name} · 共 ${state.inspectChunks.length} 块`;

  if (!rows.length) {
    const empty = document.createElement('li');
    empty.className = 'muted';
    empty.style.padding = '8px';
    empty.textContent = onlyPage ? '这一页没有分块。' : '这个文档没有分块。';
    list.appendChild(empty);
    return;
  }

  rows.forEach((chunk) => {
    const item = document.createElement('li');
    item.className = 'chunk' + (chunk.chunk_id === state.inspectSelected ? ' selected' : '');
    item.id = `chunk-${chunk.chunk_id}`;
    const boxed = Array.isArray(chunk.bbox) && chunk.bbox.length === 4;
    item.innerHTML =
      `<div class="chunk-head">` +
      `<span class="chunk-id">${escapeHtml(chunk.chunk_id)}</span>` +
      `<span class="chunk-type">${escapeHtml(chunk.block_type || '?')}</span>` +
      `<span>p.${chunk.page_num}</span>` +
      `<span>${chunk.chars} 字</span>` +
      (boxed ? '' : '<span class="chunk-nobox">无位置</span>') +
      `</div>` +
      `<p class="chunk-text"></p>`;
    // textContent, not innerHTML: this text was read out of a PDF, and a
    // document is free to contain markup.
    item.querySelector('.chunk-text').textContent = chunk.text || '';
    item.addEventListener('click', () => selectChunk(chunk.chunk_id));
    list.appendChild(item);
  });
}

// ---------------------------------------------------------------------------
// the corpus directory tree
//
// The right pane's second mode, and it does two things only: build the tree,
// and show it.
//
// Retrieval over the tree is deliberately NOT offered here. A directory tree is
// not something a reader has a question about — it is the shape a question gets
// routed over, so asking belongs to 问答, where questions are asked, and what
// this screen owes the reader is the shape itself: which folders the corpus was
// split into, how many documents each holds, and which of those folders the
// system invented rather than found.
//
// Drawn rather than listed for the same reason: the split is a claim about the
// corpus, and a claim that can only be read as numbers is a claim nobody can
// check. A folder called "ChatGPT Everything / Mass breach of privacy TikTok"
// is either a good grouping or a bad one, and no count can tell you which.

// The whole tree is fetched, not the first few levels: opening a node has to be
// instant, and a click that waits on the kernel is a tree that feels broken.
// 48 folders and their file lists is a few tens of KB.
const DIR_DEPTH = 8;

// Which folders are open, by node id. A Set rather than a flag on the node
// objects: the tree arrives fresh from the kernel on every refresh, so state
// kept on it would be thrown away each time.
const dirOpen = new Set(['__root__']);

/** The directory tree's shape, for the knowledge base currently open. */
async function refreshDirStatus() {
  if (!state.selectedKB) return;
  // Said while it is happening: a 70-document corpus takes a moment to walk,
  // and a panel that says nothing during that moment looks finished.
  $('dir-status').textContent = '读取目录树…';
  try {
    state.dirStatus = await rpc('fs.status', { kb: state.selectedKB, depth: DIR_DEPTH });
  } catch (error) {
    // No tree yet is a normal state, not a failure: the panel says so and
    // offers the rebuild. Everything else on this screen works without one.
    state.dirStatus = null;
  }
  renderDirStatus();
}

function renderDirStatus() {
  const status = state.dirStatus;
  const label = $('dir-status');
  if (!status) {
    label.textContent = state.selectedKB ? '目录树尚未建立' : '';
    label.className = 'bad-text';
    show('dir-stale', false);
    $('dir-ask-hint').textContent = '';
    renderDirTreeEmpty(state.selectedKB ? '这个知识库还没有目录树。' : '');
    return;
  }
  const parts = [
    `${status.documents} 篇`,
    `${status.nodes} 个目录`,
    `${status.leaves} 个叶子`,
    `深 ${status.depth}`,
  ];
  // "真实目录" and "内容聚类" are not the same kind of object and are not
  // equally trustworthy: one is the user's own filing, the other is a grouping
  // this system invented. The tree mixes them freely, so the mix is reported.
  const from = { path: '来自真实目录', cluster: '内容聚类而来', mixed: '目录与聚类混合' }[status.from];
  if (from) parts.push(from);
  if (status.decider) parts.push(`决策者 ${status.decider}`);
  label.textContent = parts.join(' · ');
  label.className = 'muted';
  // Indexing does not rebuild the directory tree, so a base that just gained
  // documents is routed over a tree that does not contain them. Said here
  // rather than silently: the alternative is a route that cannot possibly
  // recall the document the user just added, with no sign of why.
  const stale = $('dir-stale');
  const behind = state.documents.length && status.documents !== state.documents.length
    ? state.documents.length - status.documents
    : 0;
  stale.textContent = behind > 0 ? `${behind} 篇新文档未纳入目录树` : '';
  show('dir-stale', behind > 0);
  // Where the tree is actually used. Stated on the screen that builds it,
  // because otherwise the reader is left to guess whether this artefact does
  // anything — and a tree that turns out to be decorative is a tree nobody
  // builds twice.
  $('dir-ask-hint').textContent = '问答会用这棵树检索';
  renderDirTree();
}

/**
 * The tree area's own empty state.
 *
 * Written into the tree's own box rather than only into the status line: an
 * empty panel next to a line of muted text is read as "the tree failed to
 * render", not as "there is no tree yet", and the difference decides whether
 * the reader presses 重建目录树 or reports a bug.
 */
function renderDirTreeEmpty(text) {
  const host = $('dir-tree');
  host.innerHTML = '';
  if (!text) return;
  const box = document.createElement('div');
  box.className = 'map map-empty';
  const line = document.createElement('div');
  line.className = 'map-empty-text';
  line.textContent = text;
  box.appendChild(line);
  if (state.selectedKB) {
    const button = document.createElement('button');
    button.className = 'primary';
    button.type = 'button';
    button.textContent = '重建目录树';
    button.addEventListener('click', rebuildDirs);
    box.appendChild(button);
  }
  host.appendChild(box);
}

/**
 * Draws the corpus as a tree that opens left to right.
 *
 * Left to right, not top to bottom: the route descends this tree one level per
 * decision, and a descent reads as depth — which is horizontal distance here,
 * the same way it is in a file browser. Every node is a card; opening one
 * reveals the folders and the files directly under it, to its right.
 *
 * Only what has been opened is drawn. The whole tree is 48 folders deep three
 * levels; drawing all of it at once is a picture of nothing, and the reader
 * is looking for one branch, not for all of them.
 */
function renderDirTree() {
  const host = $('dir-tree');
  host.innerHTML = '';
  const tree = (state.dirStatus && state.dirStatus.tree) || [];
  if (!tree.length) {
    renderDirTreeEmpty(state.selectedKB ? '这个知识库还没有目录树。' : '');
    return;
  }

  // Widest folder in the tree, so a card's bar compares it with the rest of
  // the corpus rather than with itself.
  let widest = 1;
  const scan = (nodes) => nodes.forEach((node) => {
    widest = Math.max(widest, node.docs || 0);
    if (node.kids) scan(node.kids);
  });
  scan(tree);

  const root = dirTreeNode(
    {
      nid: '__root__',
      name: '语料库',
      from: state.dirStatus.from,
      children: tree.length,
      docs: state.dirStatus.documents || 0,
      own_docs: 0,
      files: [],
      kids: tree,
    },
    widest,
  );
  root.classList.add('tl-root');
  host.appendChild(root);
}

/** One folder: its card, and — when open — its children to the right. */
function dirTreeNode(node, widest) {
  const item = document.createElement('div');
  item.className = 'tl-node';
  const open = dirOpen.has(node.nid);
  const kids = node.kids || [];
  const files = node.files || [];

  const card = document.createElement('div');
  card.className = 'tl-card';
  if (open) card.classList.add('open');
  card.title = node.name || '';

  // Whether it opens further, and which way: a folder with nothing under it
  // must not look clickable.
  const mark = document.createElement('span');
  mark.className = 'tl-mark';
  mark.textContent = !kids.length && !files.length ? '·' : open ? '−' : '+';

  const name = document.createElement('span');
  name.className = 'tl-name';
  name.textContent = node.name || node.nid;

  const meta = document.createElement('span');
  meta.className = 'tl-meta';
  const parts = [`${node.docs || 0} 篇`];
  if (kids.length) parts.push(`${kids.length} 个子目录`);
  if (files.length && (node.own_docs || 0) > files.length) {
    // The kernel names the first 40 files in a folder; the count stays exact,
    // so a folder of 200 says "200 篇" and lists what it can.
    parts.push(`列出前 ${files.length} 个`);
  }
  meta.textContent = parts.join(' · ');

  card.append(mark, name, meta);

  const bar = document.createElement('span');
  bar.className = 'tl-bar';
  const fill = document.createElement('i');
  // Capped at the track: the root holds every document in the corpus, so it is
  // several times wider than the widest folder — and a bar overflowing its card
  // is a bar that cannot be read.
  const share = Math.min(100, Math.max(3, ((node.docs || 0) / widest) * 100));
  fill.style.width = share + '%';
  bar.appendChild(fill);
  card.appendChild(bar);

  // A folder this system invented is marked as one: trusting a cluster as much
  // as the user's own directory is the mistake the `from` field exists to make
  // impossible.
  if (node.from === 'cluster') {
    const tag = document.createElement('span');
    tag.className = 'tl-tag';
    tag.textContent = '聚类';
    card.appendChild(tag);
  }

  card.addEventListener('click', (event) => {
    event.stopPropagation();
    if (!kids.length && !files.length) return;
    if (dirOpen.has(node.nid)) dirOpen.delete(node.nid);
    else dirOpen.add(node.nid);
    renderDirTree();
  });

  item.appendChild(card);
  if (open && (kids.length || files.length)) {
    const column = document.createElement('div');
    column.className = 'tl-children';
    kids.forEach((kid) => column.appendChild(dirTreeNode(kid, widest)));
    files.forEach((file) => column.appendChild(dirTreeFile(file)));
    item.appendChild(column);
  }
  return item;
}

/** One document under a folder. Clicking it opens the document itself. */
function dirTreeFile(file) {
  const item = document.createElement('div');
  item.className = 'tl-node';
  const card = document.createElement('div');
  card.className = 'tl-file';
  card.title = file.name || '';
  const mark = document.createElement('span');
  mark.className = 'tl-mark';
  mark.textContent = '◦';
  const name = document.createElement('span');
  name.className = 'tl-name';
  name.textContent = file.name || file.id;
  card.append(mark, name);
  card.addEventListener('click', (event) => {
    event.stopPropagation();
    if (file.id) openDocument(file.id);
  });
  item.appendChild(card);
  return item;
}

/** (Re)builds the corpus directory tree. */
async function rebuildDirs() {
  if (!state.selectedKB) return;
  clearNotice();
  setBusy(true, '建目录树…');
  try {
    const reply = await rpc('fs.index', { kb: state.selectedKB });
    // Rebuilt, so the old node ids are gone: keeping which of them were open
    // would silently re-open folders that no longer exist, or worse, different
    // ones that happen to reuse the id.
    dirOpen.clear();
    dirOpen.add('__root__');
    await refreshDirStatus();
    notice(`已把 ${reply.documents} 篇文档聚成 ${reply.nodes} 个目录（来源 ${reply.from}）`, 'ok');
  } catch (error) {
    notice('建目录树失败：' + error.message, 'bad');
  } finally {
    setBusy(false);
  }
}


/** Right-pane mode: 'chunks' shows one document, 'search' and 'dir' ask the corpus. */
function applyMode() {
  $('mode-chunks').classList.toggle('active', state.mode === 'chunks');
  $('mode-dir').classList.toggle('active', state.mode === 'dir');
  // One place decides all three: which of them is visible follows from the mode
  // and from whether a document is open, so a later change cannot leave two on
  // screen at once.
  const inspecting = state.mode === 'chunks' && Boolean(state.inspectDoc);
  show('inspect-body', inspecting);
  show('inspect-empty', state.mode === 'chunks' && !inspecting);
  show('dir-body', state.mode === 'dir');
}

function setMode(mode) {
  state.mode = mode;
  applyMode();
  clearNotice();
  // The tree is loaded with the mode rather than on some later action, because
  // the tree IS what this mode shows — there is nothing else on the screen to
  // prompt a fetch.
  if (mode === 'dir') refreshDirStatus();
}

async function createKB() {
  const answer = await askForInput({ title: '新建知识库', placeholder: '例如：论文库' });
  if (!answer || !answer.name) return;

  try {
    const base = await rpc('kb.create', { name: answer.name });
    await refreshKBs();
    // Entered straight away: creating a base is itself the answer to "which one
    // do you want", so landing back on the picker would ask the same question
    // twice.
    await enterKB(base.id);
    notice(`已创建「${base.name}」`, 'ok');
  } catch (error) {
    notice('新建失败：' + error.message, 'bad');
  }
}

async function renameKB() {
  const current = state.kbs.find((base) => base.id === state.selectedKB);
  if (!current) return;

  const answer = await askForInput({
    title: '重命名知识库',
    value: current.name,
  });
  if (!answer || !answer.name) return;

  try {
    const base = await rpc('kb.rename', { id: current.id, name: answer.name });
    // The id is unchanged, which is the point: renaming must not move the index
    // directory or the vectors.
    patchKB(base);
    notice(`已重命名为「${base.name}」`, 'ok');
  } catch (error) {
    notice('重命名失败：' + error.message, 'bad');
  }
}

async function deleteKB() {
  const current = state.kbs.find((base) => base.id === state.selectedKB);
  if (!current) return;

  const used = state.sessions.filter((session) => session.kb === current.id);
  const warning = used.length
    ? `\n\n有 ${used.length} 个会话在使用它，它们的提问将无法再检索到内容。`
    : '';
  if (!window.confirm(`删除知识库「${current.name}」？\n\n它的文档索引与向量都会被删除，无法撤销。${warning}`)) {
    return;
  }

  try {
    const reply = await rpc('kb.delete', { id: current.id });
    state.selectedKB = '';
    // The base that was open is gone, so the picker is the only screen that can
    // be shown next — and it must not be asked to render a base that no longer
    // exists.
    leaveKB();
    await refreshKBs();
    await refreshHealth();
    const leaked = reply.vectors_error || reply.files_error;
    notice(
      leaked
        ? `已删除「${reply.name}」，但残留清理失败：${leaked}`
        : `已删除「${reply.name}」`,
      leaked ? 'warn' : 'ok'
    );
  } catch (error) {
    notice('删除失败：' + error.message, 'bad');
  }
}

// ---------------------------------------------------------------------------
// documents (of the selected knowledge base)

async function refreshDocuments() {
  if (!state.selectedKB) return;

  let payload;
  try {
    payload = await rpc('documents', { kb: state.selectedKB });
  } catch (error) {
    notice('无法读取文档列表：' + error.message, 'bad');
    return;
  }

  $('kb-title').textContent = (payload.kb && payload.kb.name) || '知识库';
  state.documents = payload.documents || [];
  // The registry's cached counts for this base are refreshed by `documents`,
  // and the reply carries the result — so the list on the left is corrected
  // without a second call.
  if (payload.kb) patchKB(payload.kb);

  $('kb-summary').textContent =
    `${state.documents.length} 篇文档 · ${payload.indexed || 0} chunks · ` +
    `${payload.embedded || 0} 已嵌入`;
  renderDocumentList();

  // A document removed while it was open must not stay open: the inspector
  // would go on drawing chunks whose document is no longer in the base.
  if (state.inspectDoc && !state.documents.some((doc) => doc.doc_id === state.inspectDoc)) {
    resetInspector();
  }
}

function renderDocumentList() {
  const list = $('documents');
  list.innerHTML = '';

  state.documents.forEach((doc) => {
    const item = document.createElement('li');
    item.className = 'document' + (doc.doc_id === state.inspectDoc ? ' active' : '');
    const missing = doc.chunks_present !== doc.chunk_count;
    item.innerHTML =
      `<span class="doc-name">${escapeHtml(doc.source_file || doc.doc_id)}</span>` +
      `<span class="doc-meta muted">${doc.page_count} 页 · ${doc.chunks_present}/${doc.chunk_count} chunks` +
      `${missing ? ' · <span class="bad-text">不完整</span>' : ''}</span>`;
    // Clicking a document opens its chunks beside the page they came from.
    item.addEventListener('click', () => openDocument(doc.doc_id));

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'ghost danger';
    remove.textContent = '移除';
    remove.addEventListener('click', (event) => {
      // The button sits inside the clickable row; without this, removing a
      // document would also open it, on a document that is about to be gone.
      event.stopPropagation();
      forgetDocument(doc);
    });
    item.appendChild(remove);
    list.appendChild(item);
  });
}

async function addDocuments() {
  if (!state.selectedKB) return;
  const paths = await api.pickFiles();
  if (paths.length) await indexPaths(paths);
}

async function indexPaths(paths) {
  clearNotice();
  setBusy(true, '索引中…');
  const results = [];

  try {
    // One call for the batch, not one per file. The kernel runs it as a
    // background job and reports each document through `progress`, which is
    // also what lets 取消 reach it: the kernel answers requests one at a time,
    // so a synchronous batch could never be interrupted.
    const started = await rpc('index_batch', { paths, kb: state.selectedKB });
    const outcomes = await waitForBatch(started.job);
    results.push(...outcomes);
    await refreshDocuments();
    await refreshHealth();
    // The corpus directory tree is NOT rebuilt by indexing — a document joins
    // it only when 重建目录树 runs. Refreshed anyway so the line can say the
    // tree is behind the corpus rather than quietly describing the old one.
    refreshDirStatus();
    notice(indexSummary(results), 'ok');
  } catch (error) {
    notice('索引失败：' + error.message, 'bad');
  } finally {
    batch = null;
    $('progress-cancel').classList.add('hidden');
    setBusy(false);
  }
}

// The running batch, so the event handler can complete it and 取消 can name it.
let batch = null;

function waitForBatch(job) {
  return new Promise((resolve) => {
    batch = { job, resolve };
    $('progress-cancel').classList.remove('hidden');
  });
}

$('progress-cancel').addEventListener('click', async () => {
  if (!batch) return;
  $('progress-cancel').disabled = true;
  try {
    // Documents already indexed stay indexed; this stops the rest.
    await rpc('index_cancel', { job: batch.job });
  } catch (error) {
    notice('取消失败：' + error.message, 'bad');
  } finally {
    $('progress-cancel').disabled = false;
  }
});

function indexSummary(results) {
  const skipped = results.filter((item) => item.skipped).length;
  const added = results.reduce((total, item) => total + (item.added || 0), 0);
  const removed = results.reduce((total, item) => total + (item.removed || 0), 0);

  const parts = [`${results.length} 个文件：+${added} chunks`];
  if (skipped) parts.push(`${skipped} 个内容未变已跳过`);
  if (removed) parts.push(`${removed} 个旧 chunks 被替换`);
  return parts.join('，');
}

async function forgetDocument(doc) {
  const name = doc.source_file || doc.doc_id;
  if (!window.confirm(`从知识库中移除「${name}」？`)) return;

  clearNotice();
  try {
    const reply = await rpc('forget', { md5: doc.md5, kb: state.selectedKB });
    await refreshDocuments();
    // It left the corpus, so the directory tree is now describing a corpus that
    // does not contain it — until 重建目录树 runs. Refreshed so the panel can
    // say so rather than quietly count a document that is gone.
    refreshDirStatus();
    notice(`已移除 ${name}（${reply.removed} chunks）`, 'ok');
  } catch (error) {
    notice('移除失败：' + error.message, 'bad');
  }
}

// ---------------------------------------------------------------------------
// sessions and conversations
//
// Both live in chats.json, written by the shell. Saved after every change that
// a user would be annoyed to lose.

async function loadChats() {
  const reply = await api.loadChats();
  if (!reply.ok) {
    // Reported rather than replaced: a corrupt history is recoverable by hand,
    // and quietly starting empty would look like the conversations never
    // existed.
    notice('无法读取会话记录：' + reply.error + '（本次不会覆盖该文件）', 'bad');
    state.sessions = [];
    renderSessions();
    return;
  }
  state.sessions = reply.data.sessions || [];
  // Nothing is selected on load. The chat view opens on the picker, so choosing
  // a session here would be a decision the user is about to make anyway — and
  // for someone who never opens the chat view, one made for nothing.
  renderSessions();
}

async function saveChats() {
  const reply = await api.saveChats({ sessions: state.sessions });
  if (!reply.ok) notice('会话记录保存失败：' + reply.error, 'bad');
}

function currentSession() {
  return state.sessions.find((session) => session.id === state.selectedSession) || null;
}

function currentConversation() {
  const session = currentSession();
  if (!session) return null;
  return session.conversations.find((item) => item.id === state.selectedConversation) || null;
}

function newID() {
  // randomUUID is present under file://, which Chromium treats as a secure
  // context; the fallback is here because these ids only have to be unique
  // within one file, and a missing helper should not be what breaks creating a
  // session.
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return window.crypto.randomUUID();
  }
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function renderSessions() {
  const list = $('session-list');
  list.innerHTML = '';
  show('session-empty', state.sessions.length === 0);

  state.sessions.forEach((session) => {
    const base = state.kbs.find((item) => item.id === session.kb);
    const conversations = session.conversations || [];
    const messages = conversations.reduce((total, item) => total + item.messages.length, 0);

    const card = document.createElement('li');
    card.className = 'session-card';
    card.innerHTML =
      `<div class="session-card-main">` +
      `<span class="session-name">${escapeHtml(session.name)}</span>` +
      `<span class="list-meta">${conversations.length} 个对话 · ${messages} 条消息</span>` +
      `</div>` +
      `<span class="session-kb">${escapeHtml(base ? base.name : '知识库已删除')}</span>`;

    card.addEventListener('click', () => enterSession(session.id));

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'ghost danger tiny';
    remove.textContent = '×';
    remove.title = '删除会话';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteSession(session.id);
    });
    card.appendChild(remove);

    list.appendChild(card);
  });
}

/** Switches the chat view to its session screen. */
function enterSession(id) {
  const session = state.sessions.find((item) => item.id === id);
  if (!session) return;

  state.inSession = true;
  state.selectedSession = id;
  state.selectedConversation = session.conversations.length
    ? session.conversations[0].id
    : '';

  show('chat-picker', false);
  show('chat-session', true);
  renderSessionHeader();
  renderConversations();
  renderMessages();
  clearNotice();
}

/** Switches the chat view back to the session picker. */
function leaveSession() {
  state.inSession = false;
  show('chat-picker', true);
  show('chat-session', false);
  renderSessions();
}

function renderSessionHeader() {
  const session = currentSession();
  if (!session) return;

  const base = state.kbs.find((item) => item.id === session.kb);
  $('session-title').textContent = session.name;
  // The binding is shown, not offered: a session's knowledge base is fixed at
  // creation, and a control here would imply it can be changed.
  $('session-kb').textContent = base ? `知识库：${base.name}` : '知识库已删除';
}

function renderConversations() {
  const session = currentSession();
  $('conversation-add').disabled = !session;

  const list = $('conversation-list');
  list.innerHTML = '';
  if (!session) return;

  session.conversations.forEach((conversation) => {
    const item = document.createElement('li');
    item.className =
      'list-item' + (conversation.id === state.selectedConversation ? ' active' : '');
    item.innerHTML =
      `<span class="list-name">${escapeHtml(conversation.name)}</span>` +
      `<span class="list-meta">${conversation.messages.length} 条</span>`;
    item.addEventListener('click', () => selectConversation(conversation.id));

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'ghost danger tiny';
    remove.textContent = '×';
    remove.title = '删除对话';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });
    item.appendChild(remove);
    list.appendChild(item);
  });
}

async function createSession() {
  if (!state.kbs.length) {
    notice('请先在「知识库」里新建一个知识库。', 'warn');
    return;
  }

  const answer = await askForInput({
    title: '新建会话',
    placeholder: '例如：毕业论文',
    options: state.kbs.map((base) => ({ value: base.id, label: base.name })),
  });
  if (!answer || !answer.name) return;

  const session = {
    id: newID(),
    name: answer.name,
    // Bound at creation and never changed. Letting it move later would
    // reinterpret every answer already in the session, because the passages
    // they cite would have come from a different index.
    kb: answer.kb || state.kbs[0].id,
    created_at: new Date().toISOString(),
    conversations: [],
  };
  state.sessions.push(session);

  const conversation = {
    id: newID(),
    name: '对话 1',
    created_at: new Date().toISOString(),
    messages: [],
  };
  session.conversations.push(conversation);

  // Entered straight away: creating a session is itself the answer to "which
  // one do you want", so returning to the picker would ask the same question
  // twice.
  enterSession(session.id);
  await saveChats();
}

async function deleteSession(id) {
  const session = state.sessions.find((item) => item.id === id);
  if (!session) return;

  const conversations = (session.conversations || []).length;
  const warning = conversations
    ? `\n\n它的 ${conversations} 个对话与全部消息都会删除，无法撤销。`
    : '';
  if (!window.confirm(`删除会话「${session.name}」？${warning}`)) return;

  state.sessions = state.sessions.filter((item) => item.id !== id);

  // Never left pointing at the deleted session. The picker is the only screen
  // that can be shown next, and it must not try to render a session that is
  // gone.
  if (state.selectedSession === id) {
    state.selectedSession = '';
    state.selectedConversation = '';
    leaveSession();
  }
  renderSessions();
  await saveChats();
}

async function createConversation() {
  const session = currentSession();
  if (!session) return;

  const answer = await askForInput({
    title: '新建对话',
    placeholder: `对话 ${session.conversations.length + 1}`,
  });
  if (!answer) return;

  const conversation = {
    id: newID(),
    name: answer.name || `对话 ${session.conversations.length + 1}`,
    created_at: new Date().toISOString(),
    messages: [],
  };
  session.conversations.push(conversation);
  state.selectedConversation = conversation.id;

  renderConversations();
  renderMessages();
  await saveChats();
}

async function deleteConversation(id) {
  const session = currentSession();
  if (!session) return;
  if (session.conversations.length === 1) {
    notice('会话至少要保留一个对话。', 'warn');
    return;
  }
  if (!window.confirm('删除这个对话及其全部消息？')) return;

  session.conversations = session.conversations.filter((item) => item.id !== id);
  if (state.selectedConversation === id) {
    state.selectedConversation = session.conversations[0].id;
  }
  renderConversations();
  renderMessages();
  await saveChats();
}

function selectConversation(id) {
  state.selectedConversation = id;
  renderConversations();
  renderMessages();
  clearNotice();
}

// ---------------------------------------------------------------------------
// messages

function renderMessages() {
  const session = currentSession();
  const conversation = currentConversation();

  $('chat-title').textContent = conversation ? conversation.name : '对话';

  const base = session ? state.kbs.find((item) => item.id === session.kb) : null;

  show('ask-form', Boolean(conversation && base));
  show('chat-empty', !conversation);

  const list = $('messages');
  list.innerHTML = '';
  if (!conversation) return;

  // A session whose knowledge base is gone can still be read, but not asked in:
  // the answer would be retrieved from nothing, and that reads as the question
  // being bad rather than the index being missing.
  if (!base) {
    const warning = document.createElement('p');
    warning.className = 'muted center';
    warning.textContent = '这个会话的知识库已被删除，无法继续提问。历史仍可翻阅。';
    list.appendChild(warning);
  }

  conversation.messages.forEach((message, index) => {
    renderMessage(list, message, index);
  });
  list.scrollTop = list.scrollHeight;

  $('ask').disabled = state.busy || !base;
}

function renderMessage(list, message, index) {
  const item = document.createElement('div');
  item.className = `message ${message.role}`;

  if (message.role === 'user') {
    item.innerHTML = `<div class="bubble">${escapeHtml(message.text)}</div>`;
    list.appendChild(item);
    return;
  }

  const meta = message.meta || {};
  const evidence = message.evidence || [];
  const trace = message.trace || [];

  const head = meta.verdict
    ? `<div class="answer-head">` +
      `<span class="badge ${meta.verdict === 'SUFFICIENT' ? 'ok' : 'warn'}">${escapeHtml(meta.verdict)}</span>` +
      `<span class="muted">${meta.rounds || '?'} 轮 · ${evidence.length} 条证据` +
      `${meta.queries ? ` · ${meta.queries} 次查询` : ''}` +
      `${meta.seconds ? ` · ${meta.seconds}s` : ''}</span></div>`
    : '';

  // Open while the answer is being written, collapsed once it is done. It has
  // to be open while it is the only sign of progress; afterwards it should not
  // push the answer itself off the screen.
  const showTrace = Boolean(message.streaming || trace.length);
  const traceBox = showTrace
    ? `<details class="trace" id="msg-${index}-trace-box"${message.streaming ? ' open' : ''}>` +
      `<summary>检索与推理 <span class="muted" id="msg-${index}-trace-count">${trace.length} 步</span></summary>` +
      `<pre id="msg-${index}-trace">${escapeHtml(trace.join('\n')) || '…'}</pre>` +
      `</details>`
    : '';

  const details = evidence.length
    ? `<details class="evidence"><summary>证据 <span class="muted">${evidence.length} 条</span></summary>` +
      `<ol>${evidence
        .map((hit, position) => {
          const chunk = hit.chunk || {};
          const sources = (hit.sources || []).join(', ') || '—';
          return (
            `<li id="msg-${index}-evidence-${position + 1}">` +
            `<div class="evidence-head"><span class="evidence-index">${position + 1}</span>` +
            `<span class="muted">${escapeHtml(chunk.doc_id || '')} p.${chunk.page_num ?? '?'}` +
            `${chunk.block_type ? ' · ' + escapeHtml(chunk.block_type) : ''} · ${escapeHtml(sources)}</span></div>` +
            `<div class="evidence-text">${escapeHtml(chunk.text || '')}</div></li>`
          );
        })
        .join('')}</ol></details>`
    : '';

  item.innerHTML =
    head +
    `<div class="draft" id="msg-${index}-text">${linkCitations(message.text, index, evidence.length)}</div>` +
    traceBox +
    details;
  list.appendChild(item);
}

/**
 * Turns [n] into a chip that scrolls to the matching evidence.
 *
 * The text is escaped first: the answer is model output that has read the
 * document, so it is not markup, and treating it as markup would let a document
 * inject script into the app.
 */
function linkCitations(text, messageIndex, evidenceCount) {
  return escapeHtml(text || '')
    .replace(/\[(\d+)\]/g, (match, digits) => {
      const index = Number(digits);
      if (index < 1 || index > evidenceCount) return match;
      return `<button class="cite" type="button" data-target="msg-${messageIndex}-evidence-${index}">${index}</button>`;
    })
    .replace(/\n/g, '<br />');
}

document.addEventListener('click', (event) => {
  const chip = event.target.closest('.cite');
  if (!chip) return;
  const target = $(chip.dataset.target);
  if (!target) return;
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  target.classList.add('flash');
  setTimeout(() => target.classList.remove('flash'), 900);
});

// ---------------------------------------------------------------------------
// asking

$('ask-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || state.busy) return;
  await askQuestion(question);
});

$('question').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('ask-form').requestSubmit();
  }
});

async function askQuestion(question) {
  const session = currentSession();
  const conversation = currentConversation();
  if (!session || !conversation) return;

  clearNotice();
  $('question').value = '';

  const started = Date.now();

  conversation.messages.push({ role: 'user', text: question });
  const answer = {
    role: 'assistant',
    text: '',
    // Not persisted state: it only decides whether the trace starts expanded.
    streaming: true,
    trace: [],
    meta: null,
    evidence: [],
  };
  conversation.messages.push(answer);
  state.pending = answer;

  // First question names the conversation: an untitled list of "对话 1..N" is
  // unusable a day later, and the user's own words are the best label available
  // without asking for one.
  if (conversation.messages.length === 2 && /^对话 \d+$/.test(conversation.name)) {
    conversation.name = question.length > 24 ? question.slice(0, 24) + '…' : question;
  }

  renderMessages();
  renderConversations();

  // Held rather than looked up per event: the answer changes hundreds of times
  // per reply, and the trace a few dozen.
  const position = conversation.messages.length - 1;
  state.pendingEl = $(`msg-${position}-text`);
  state.pendingTraceEl = $(`msg-${position}-trace`);
  state.pendingTraceCountEl = $(`msg-${position}-trace-count`);

  setBusy(true, '检索与推理…');
  try {
    const result = await rpc('ask', { question, kb: session.kb });
    answer.evidence = result.evidence || [];
    // `result.answer` is the agent.Result json tag — the field this reply has to
    // be read from. It was renamed from `draft` when the per-round draft was
    // removed, and this line kept reading the old name: the streamed text was
    // replaced by the placeholder the moment the reply landed, which is exactly
    // what "SSE 答完后突然变成（无草稿）" was. cmd/freerag's
    // TestAskReplyCarriesTheAnswerFieldUnderTheNameTheRendererReads pins it now.
    if (!result.answer && answer.evidence.length === 0) {
      // A run that never retrieved anything has no answer by design: the loop
      // does not write one over an empty pool, because there is nothing to
      // write from and an answer reading "the evidence does not answer this" is
      // not an answer. Saying so is the UI's job — and the wording matters,
      // because "not found" is a fact about the SEARCH, not about the corpus.
      answer.text = '未检索到任何内容。这不代表知识库里没有——检索未命中，换个问法再试，或先在知识库页确认文档已索引。';
    } else {
      answer.text = result.answer || '（无回答）';
    }
    // The trace in the reply replaces whatever arrived as events. The two
    // should agree; if they ever do not, the one stored with the answer is the
    // one that describes it, since that is what the kernel actually ran.
    if (Array.isArray(result.trace) && result.trace.length) {
      answer.trace = result.trace;
    }
    answer.meta = {
      verdict: result.verdict,
      rounds: result.rounds,
      queries: (result.queries || []).length,
      mode: result.mode,
      seconds: ((Date.now() - started) / 1000).toFixed(1),
    };
  } catch (error) {
    answer.text = '提问失败：' + error.message;
  } finally {
    answer.streaming = false;
    state.pending = null;
    state.pendingEl = null;
    state.pendingTraceEl = null;
    state.pendingTraceCountEl = null;
    setBusy(false);
    // Re-rendered once the stream is over, which is also what collapses the
    // trace and turns citations into links: during streaming the text is plain,
    // because a fragment boundary can fall inside a citation marker.
    renderMessages();
    await saveChats();
  }
}

// ---------------------------------------------------------------------------
// progress events

const STAGE_LABELS = {
  hash: '计算内容指纹',
  skipped: '内容未变，跳过解析',
  parse: '解析文档（版面检测 + 表格结构）',
  parsed: '解析完成',
  stored: '写入索引',
  persisted: '已保存',
  'index-batch': '批量索引（解析串行，嵌入并行）',
  'index-batch-done': '批量索引完成',
  agent: '检索与推理',
  thinking: '正在决定检索动作（首次模型调用较慢）',
  answer: '正在生成答案',
};

api.onEvent((msg) => {
  if (!msg || msg.method !== 'progress') return;
  const params = msg.params || {};
  const stage = params.stage || '';

  $('progress-stage').textContent = STAGE_LABELS[stage] || stage || '处理中…';

  // Ends the wait that indexPaths started. The outcomes arrive here rather than
  // as the RPC's return value, because that call returned as soon as the job
  // was accepted — which is exactly what leaves the kernel free to answer 取消.
  if (stage === 'index-batch-done') {
    const pending = batch;
    batch = null;
    if (pending) pending.resolve(params.outcomes || []);
    return;
  }

  // The answer and the trace that produced it belong to the conversation, not
  // to the strip. They are what the question produced, and they have to still
  // be there after it has scrolled past — the strip is a status line, and the
  // next thing that happens clears it.
  if (stage === 'answer') {
    appendAnswer(params.delta || '');
    return;
  }
  if (stage === 'thinking' || stage === 'agent') {
    appendTrace(stage === 'thinking' ? STAGE_LABELS.thinking : params.line || '');
    return;
  }

  // Everything else is the indexing flow, which has no conversation to belong
  // to and does belong in the strip.
  const line = params.line || describe(params);
  if (line) {
    const log = $('progress-log');
    log.textContent += (log.textContent ? '\n' : '') + line;
    log.scrollTop = log.scrollHeight;
  }
});

/**
 * Appends one streamed fragment of the answer.
 *
 * Written with textContent, not innerHTML, and that is not only the usual
 * untrusted-model-output rule. The answer arrives in fragments, so a tag or a
 * citation marker could be split across two deltas with neither half looking
 * like anything on its own — escaping each fragment separately would not catch
 * that, and only assembling the whole answer first would. Setting textContent
 * sidesteps the question; renderMessages() replaces this with the linked
 * version once the call returns.
 */
function appendAnswer(delta) {
  if (!delta || !state.pending || !state.pendingEl) return;

  // Appending is always right here, because every fragment belongs to the same
  // text: only the FINAL answer is reported, and it is the only text the loop
  // asks a model to write — the per-round draft the checker used to judge is
  // gone (see agent.kbinfo). So what is on screen is never something that is
  // about to be retracted.
  state.pending.text += delta;
  state.pendingEl.textContent = state.pending.text;
}

/**
 * Appends one line to the trace of the message being generated.
 *
 * Rendered into the conversation rather than the progress strip: which tools
 * ran and what they returned is part of how the answer was reached, so it
 * belongs beside the answer and outlives the request that produced it.
 */
function appendTrace(line) {
  if (!line || !state.pending || !state.pendingTraceEl) return;
  if (!state.pending.trace) state.pending.trace = [];
  state.pending.trace.push(line);

  state.pendingTraceEl.textContent = state.pending.trace.join('\n');
  // Kept pinned to the newest line: a log that stops moving is the thing this
  // whole channel exists to avoid.
  state.pendingTraceEl.scrollTop = state.pendingTraceEl.scrollHeight;
  if (state.pendingTraceCountEl) {
    state.pendingTraceCountEl.textContent = `${state.pending.trace.length} 步`;
  }
}

/** Renders a progress payload that has no pre-formatted line. */
function describe(params) {
  const parts = [];
  if (params.done && params.total) parts.push(`${params.done}/${params.total}`);
  if (params.file) parts.push(params.file);
  if (params.error) parts.push('失败：' + params.error);
  if (params.pages) parts.push(params.pages + ' 页');
  if (params.blocks) parts.push(params.blocks + ' 块');
  if (params.chunks) parts.push(params.chunks + ' chunks');
  if (params.layout) parts.push('版面=' + params.layout);
  if (params.added) parts.push('+' + params.added);
  if (params.removed) parts.push('-' + params.removed);
  if (params.chunk_count) parts.push(params.chunk_count + ' chunks');
  return parts.join(' · ');
}

// ---------------------------------------------------------------------------
// kernel liveness and logs

function renderKernelStatus(status) {
  $('kernel-line').textContent = status.running ? '内核：运行中' : '内核：已停止';
}

// Subscribing alone is not enough: the kernel is started before the window
// exists, so its first status broadcast happens before this script runs and
// would be missed — leaving the bar saying "connecting" forever on a kernel
// that is fine. boot() therefore also asks for the current status.
api.onStatus((status) => {
  renderKernelStatus(status);
});

api.onLog((line) => {
  $('log-line').textContent = line.length > 160 ? line.slice(0, 160) + '…' : line;
});

// ---------------------------------------------------------------------------
// wiring

$('tab-kb').addEventListener('click', () => switchView('kb'));
$('tab-chat').addEventListener('click', () => switchView('chat'));
$('theme-toggle').addEventListener('click', toggleTheme);

$('kb-add').addEventListener('click', createKB);
$('kb-back').addEventListener('click', () => {
  leaveKB();
  clearNotice();
});
$('kb-rename').addEventListener('click', renameKB);
$('kb-delete').addEventListener('click', deleteKB);
$('doc-add').addEventListener('click', addDocuments);

$('mode-chunks').addEventListener('click', () => setMode('chunks'));
$('mode-dir').addEventListener('click', () => setMode('dir'));
$('dir-index').addEventListener('click', rebuildDirs);

// The inspector's controls. A page turn is a fetch (the image has to be
// rendered), while the two filters are pure re-renders over chunks already in
// hand — so neither waits on the kernel.
$('page-prev').addEventListener('click', () => {
  if (state.inspectPage > 1) loadPage(state.inspectPage - 1);
});
$('page-next').addEventListener('click', () => {
  const pages = (state.inspectMeta && state.inspectMeta.pages) || 1;
  if (state.inspectPage < pages) loadPage(state.inspectPage + 1);
});
$('show-all').addEventListener('change', renderBoxes);
$('only-page').addEventListener('change', renderChunkList);

$('session-add').addEventListener('click', createSession);
$('session-back').addEventListener('click', () => {
  leaveSession();
  clearNotice();
});
$('conversation-add').addEventListener('click', createConversation);

const dropzone = $('dropzone');
dropzone.addEventListener('click', addDocuments);
dropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropzone.classList.add('over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
dropzone.addEventListener('drop', async (event) => {
  event.preventDefault();
  dropzone.classList.remove('over');
  // The renderer cannot read a dropped file's path (nodeIntegration is off);
  // webUtils in the preload is the only way to resolve it.
  const paths = Array.from(event.dataTransfer.files).map((file) => api.pathForFile(file));
  if (paths.length) await indexPaths(paths);
});

// ---------------------------------------------------------------------------
// settings

/** "CoreMLExecutionProvider;CPUExecutionProvider" -> "CoreML 加速". */
function shortProvider(provider) {
  if (!provider) return '未加载';
  const name = String(provider).split(';')[0].replace(/ExecutionProvider$/, '');
  return { CoreML: 'CoreML 加速', CUDA: 'CUDA 加速', Dml: 'DirectML 加速',
    CPU: 'CPU（本机未获得加速）' }[name] || name;
}

/**
 * One line for the settings dialog: what this machine offers, and what the
 * models actually got.
 *
 * Both halves, because they are allowed to disagree and the disagreement is
 * invisible otherwise — the scan can report a GPU while every model runs on
 * CPU, because the runtime bundled with this install has no provider for it.
 */
function describeHardware(hardware, reported) {
  if (!hardware) return '未能检测本机配置。';
  const accelerator = hardware.accelerator || {};
  const offered = accelerator.vendor === 'apple'
    ? 'Apple 芯片（Metal）'
    : accelerator.vendor === 'nvidia'
      ? `NVIDIA ${accelerator.name || ''}（CUDA）`.trim()
      : '未检测到加速器';
  const parts = [
    offered,
    `${(hardware.memoryBytes / 1024 ** 3).toFixed(0)} GB 内存`,
    `${hardware.cpu?.cores || '?'} 核 CPU`,
  ];

  const lines = [parts.join(' · ')];
  // The second half is the kernel's, and it is the one that matters: sessions
  // report their own provider, so this is measured rather than predicted.
  const deepdoc = reported?.result?.deepdoc;
  if (deepdoc?.warmed) {
    lines.push(`版面与表格：${shortProvider(deepdoc.layout?.provider)}`);
  } else {
    lines.push('版面与表格：尚未预热');
  }
  if (reported?.result?.chart?.enabled) lines.push('图表转译：已启用');
  return lines.join('；');
}

/** Fills the hardware row, asking both the shell and the kernel. */
async function loadHardware() {
  const node = $('settings-hardware');
  if (!node) return;
  node.textContent = '检测中…';
  try {
    const [hardware, reported] = await Promise.all([
      api.hardware(),
      // The kernel may still be warming up; a missing report is said out loud
      // rather than filled in with a guess.
      api.rpc('hardware').catch(() => null),
    ]);
    node.textContent = describeHardware(hardware, reported);
  } catch (error) {
    node.textContent = `检测失败：${error.message}`;
  }
}

function fillSettingsForm(settings) {
  $('settings-language').value = settings.answerLanguage || '';
  $('settings-vision').checked = Boolean(settings.vision);
  $('settings-vision-backend').value = settings.visionBackend || 'general';
  $('settings-vision-model').value = settings.visionModel || '';
  $('settings-vision-workers').value = settings.visionWorkers;
  $('settings-vision-tokens').value = settings.visionMaxTokens;
  $('settings-vision-side').value = settings.visionMaxSide;
  syncVisionForm();
  // The one field that is not live. The caption gate is sized when the stage
  // first runs (cmd/freerag/vision.go), and resizing a semaphore in flight is
  // not something to guess at — so this says so instead of looking applied.
  // Two of these are read when the kernel is spawned, not when it is told —
  // FREERAG_VLM names the backend and the gate is sized on first use. Saying so
  // beats a dialog that looks applied and is not.
  $('settings-note').textContent = '「图表识别方式」与「描述并发」在下次启动后生效，其余立即生效。';
  loadHardware();
}

/**
 * Greys out what the chosen backend does not read.
 *
 * Under Laya-Chart the model name, the worker count and the token ceiling are
 * all ignored: the model is the chart service's own, it decodes one chart at a
 * time, and its budget is fixed at 768 — below which the transcribed tables come
 * back half-written. Leaving them editable would let a user store values that
 * survive, apply, and change nothing.
 */
function syncVisionForm() {
  const chart = $('settings-vision-backend').value === 'chart';
  for (const id of ['settings-vision-model', 'settings-vision-workers', 'settings-vision-tokens']) {
    $(id).disabled = chart;
  }
  $('settings-vision-model-note').textContent = chart
    ? '图表模式不读取这三项：模型与预算由图表服务自己决定。'
    : '';
}

function readSettingsForm() {
  return {
    answerLanguage: $('settings-language').value,
    vision: $('settings-vision').checked,
    visionBackend: $('settings-vision-backend').value,
    visionModel: $('settings-vision-model').value,
    visionWorkers: Number($('settings-vision-workers').value),
    visionMaxTokens: Number($('settings-vision-tokens').value),
    visionMaxSide: Number($('settings-vision-side').value),
  };
}

async function openSettings() {
  try {
    const reply = await api.loadPrefs();
    if (!reply || !reply.ok) {
      notice('读取设置失败：' + ((reply && reply.error) || '未知错误'), 'bad');
      return;
    }
    fillSettingsForm((reply.data && reply.data.settings) || {});
    $('settings-dialog').showModal();
  } catch (error) {
    notice('读取设置失败：' + error.message, 'bad');
  }
}

$('settings-open').addEventListener('click', () => { openSettings(); });

// Kept in step as the choice changes, not only when the dialog opens: which of
// these fields the backend reads is the point of showing it, and a greyed-out
// field the user just typed into is worse than no hint at all.
$('settings-vision-backend').addEventListener('change', () => { syncVisionForm(); });

// Re-scanning replaces the cache rather than amending it: hardware does not
// change while the app is running, so a rescan is for the case where the first
// one was wrong (a driver installed since, a scan that raced a boot).
$('settings-hardware-rescan').addEventListener('click', async () => {
  const node = $('settings-hardware');
  node.textContent = '重新扫描中…';
  try {
    const hardware = await api.hardwareRescan();
    const reported = await api.rpc('hardware').catch(() => null);
    node.textContent = describeHardware(hardware, reported);
  } catch (error) {
    node.textContent = `扫描失败：${error.message}`;
  }
});

// Enter must save, not cancel: implicit submission activates the first submit
// button in tree order, which here is 取消 — the same trap as the prompt dialog.
$('settings-form').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || event.target.tagName === 'BUTTON') return;
  event.preventDefault();
  $('settings-form').requestSubmit($('settings-save'));
});

$('settings-dialog').addEventListener('close', async () => {
  if ($('settings-dialog').returnValue !== 'ok') return;
  try {
    const reply = await api.savePrefs(readSettingsForm());
    if (!reply || !reply.ok) {
      notice('保存设置失败：' + ((reply && reply.error) || '未知错误'), 'bad');
      return;
    }
    // The file was written but the running kernel refused the values. Reported
    // rather than swallowed: otherwise the dialog, the file and the running app
    // would each say something different and only one of them would be right.
    if (reply.data && reply.data.kernel_error) {
      notice('已写入设置文件，但内核未接受：' + reply.data.kernel_error, 'bad');
      return;
    }
    notice('设置已保存', 'ok');
  } catch (error) {
    notice('保存设置失败：' + error.message, 'bad');
  }
});

async function boot() {
  // First, because it is the only boot step that changes what the window looks
  // like. The preload has already applied the theme from the command line; this
  // re-reads it so the renderer does not depend on that having worked, and so
  // the toggle can name where the theme came from.
  try {
    const prefs = await api.loadPrefs();
    if (prefs && prefs.ok && prefs.data) {
      state.themeSource = prefs.data.source || '';
      applyTheme(prefs.data.theme);
    }
  } catch (error) {
    // Falling through leaves whatever the preload applied — the theme the window
    // was opened in. The right answer, reached by the other path.
  }

  try {
    renderKernelStatus(await api.status());
  } catch (error) {
    /* the status bar is a convenience; failing to read it must not block boot */
  }

  try {
    await refreshKBs();
    await refreshHealth();
    await loadChats();
    await refreshDocuments();
  } catch (error) {
    notice('无法连接内核：' + error.message, 'bad');
  }

  // The hash is honoured so a view can be linked to, and so the screenshot
  // affordance in main.js can capture either one.
  switchView(location.hash === '#chat' ? 'chat' : 'kb');

}

boot();
