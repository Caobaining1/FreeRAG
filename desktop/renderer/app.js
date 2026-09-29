'use strict';

/**
 * freerag renderer.
 *
 * Three jobs: show what is indexed, let the user add documents, and answer
 * questions with the evidence visible. Progress arrives as kernel events while
 * an RPC is still outstanding — without that a 30 s index or a 3 min answer is
 * a frozen window.
 */

const api = window.freerag;
const $ = (id) => document.getElementById(id);

const state = {
  documents: [],
  evidence: [],
  busy: false,
  startedAt: 0,
  timer: null,
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
 * Everything shown here is untrusted in principle: the draft comes from a
 * language model that has read the document, and a document may contain markup.
 * Building DOM nodes would be safer still, but escaping at the one place that
 * writes HTML is enough and much shorter.
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
}

function clearNotice() {
  $('notice').className = 'notice hidden';
}

function setBusy(busy, label) {
  state.busy = busy;
  $('ask').disabled = busy;
  $('add').disabled = busy;
  show('progress', busy);

  if (busy) {
    state.startedAt = Date.now();
    $('progress-stage').textContent = label || '处理中…';
    $('progress-log').textContent = '';
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

// ---------------------------------------------------------------------------
// progress events

const STAGE_LABELS = {
  hash: '计算内容指纹',
  skipped: '内容未变，跳过解析',
  parse: '解析文档（版面检测 + 表格结构）',
  parsed: '解析完成',
  stored: '写入索引',
  persisted: '已保存',
  agent: '检索与推理',
  thinking: '正在决定检索动作（首次模型调用较慢）',
};

api.onEvent((msg) => {
  if (!msg || msg.method !== 'progress') return;
  const params = msg.params || {};
  const stage = params.stage || '';
  $('progress-stage').textContent = STAGE_LABELS[stage] || stage || '处理中…';

  const line = params.line || describe(params);
  if (line) {
    const log = $('progress-log');
    log.textContent += (log.textContent ? '\n' : '') + line;
    log.scrollTop = log.scrollHeight;
  }
});

/** Renders a progress payload that has no pre-formatted line. */
function describe(params) {
  const parts = [];
  if (params.file) parts.push(params.file);
  if (params.pages) parts.push(params.pages + ' 页');
  if (params.chunks) parts.push(params.chunks + ' chunks');
  if (params.layout) parts.push('版面=' + params.layout);
  if (params.added) parts.push('+' + params.added);
  if (params.removed) parts.push('-' + params.removed);
  if (params.chunk_count) parts.push(params.chunk_count + ' chunks');
  return parts.join(' · ');
}

// ---------------------------------------------------------------------------
// health

async function refreshHealth() {
  try {
    const status = await rpc('status');
    renderHealth(status);
  } catch (error) {
    $('health').innerHTML =
      '<span class="chip bad">内核不可用</span>';
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
// documents

async function refreshDocuments() {
  const payload = await rpc('documents');
  state.documents = payload.documents || [];
  renderDocuments(payload);

  const hasContent = state.documents.length > 0;
  show('empty', !hasContent);
  $('empty-text').textContent = hasContent
    ? ''
    : '还没有索引任何文档。把 PDF 拖进左侧，或点「添加」。';
}

function renderDocuments(payload) {
  const list = $('documents');
  list.innerHTML = '';

  for (const doc of state.documents) {
    const item = document.createElement('li');
    item.className = 'doc';
    if (doc.chunks_present !== doc.chunk_count) item.classList.add('stale');

    const title = doc.source_file || doc.doc_id;
    item.innerHTML =
      `<div class="doc-title" title="${escapeHtml(doc.path || title)}">${escapeHtml(title)}</div>` +
      `<div class="doc-meta">${escapeHtml(summaryLine(doc))}</div>`;

    const forget = document.createElement('button');
    forget.className = 'doc-forget';
    forget.type = 'button';
    forget.textContent = '移除';
    forget.title = '从索引中删除这篇文档';
    forget.addEventListener('click', () => forgetDocument(doc));
    item.appendChild(forget);

    list.appendChild(item);
  }

  $('index-summary').textContent = payload.indexed
    ? `${payload.indexed} chunks · ${payload.embedded} 已嵌入`
    : '索引为空';
}

function summaryLine(doc) {
  const parts = [];
  if (doc.page_count) parts.push(doc.page_count + ' 页');
  parts.push(doc.chunk_count + ' chunks');
  if (doc.indexed_at) {
    const when = new Date(doc.indexed_at);
    if (!isNaN(when)) parts.push(when.toLocaleString());
  }
  // A document whose chunks went missing is worth saying out loud rather than
  // showing a chunk count that no longer exists.
  if (doc.chunks_present !== doc.chunk_count) {
    parts.push(`⚠ 索引中只剩 ${doc.chunks_present}`);
  }
  return parts.join(' · ');
}

async function addDocuments() {
  const paths = await api.pickFiles();
  if (!paths || paths.length === 0) return;
  await indexPaths(paths);
}

async function indexPaths(paths) {
  if (state.busy) return;
  clearNotice();
  setBusy(true, '准备索引…');

  const results = [];
  try {
    for (const path of paths) {
      // One RPC per document, each reporting its own progress: indexing them in
      // one call would leave the UI unable to say which file is being worked on.
      results.push(await rpc('index', { path }));
    }
    await refreshDocuments();
    await refreshHealth();
    notice(indexSummary(results), 'ok');
  } catch (error) {
    notice('索引失败：' + error.message, 'bad');
  } finally {
    setBusy(false);
  }
}

function indexSummary(results) {
  if (results.length === 0) return '没有可索引的文件。';
  const skipped = results.filter((r) => r.skipped).length;
  const added = results.reduce((sum, r) => sum + (r.added || 0), 0);
  const removed = results.reduce((sum, r) => sum + (r.removed || 0), 0);

  const parts = [];
  if (added) parts.push(`新增 ${added} chunks`);
  if (removed) parts.push(`替换 ${removed} 个旧 chunks`);
  if (skipped) parts.push(`${skipped} 篇内容未变已跳过`);
  return parts.length ? parts.join('，') : '没有变化。';
}

async function forgetDocument(doc) {
  const name = doc.source_file || doc.doc_id;
  if (!window.confirm(`从索引中移除「${name}」？`)) return;

  clearNotice();
  try {
    const reply = await rpc('forget', { md5: doc.md5 });
    await refreshDocuments();
    notice(`已移除 ${name}（${reply.removed} chunks）`, 'ok');
  } catch (error) {
    notice('移除失败：' + error.message, 'bad');
  }
}

// ---------------------------------------------------------------------------
// drag and drop

const dropzone = $('dropzone');

['dragenter', 'dragover'].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add('over');
  })
);

['dragleave', 'drop'].forEach((type) =>
  dropzone.addEventListener(type, () => dropzone.classList.remove('over'))
);

dropzone.addEventListener('drop', async (event) => {
  event.preventDefault();
  // The renderer cannot read a path off a dropped File, so each one is handed
  // back to the preload to be resolved (Electron removed File.path).
  const paths = Array.from(event.dataTransfer.files)
    .map((file) => api.pathForFile(file))
    .filter(Boolean);

  if (paths.length === 0) {
    notice('无法读取拖入文件的路径。', 'bad');
    return;
  }
  await indexPaths(paths);
});

// ---------------------------------------------------------------------------
// ask

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
  clearNotice();
  show('answer', false);
  setBusy(true, '检索与推理…');

  try {
    const result = await rpc('ask', { question });
    renderAnswer(result);
  } catch (error) {
    notice('提问失败：' + error.message, 'bad');
  } finally {
    setBusy(false);
  }
}

function renderAnswer(result) {
  state.evidence = result.evidence || [];

  $('verdict').textContent = result.verdict || '—';
  $('verdict').className =
    'badge ' + (result.verdict === 'SUFFICIENT' ? 'ok' : 'warn');
  $('answer-meta').textContent =
    `${result.rounds} 轮 · ${state.evidence.length} 条证据 · ` +
    `${((result.queries || []).length)} 次查询 · ${result.mode || ''}`;

  $('draft').innerHTML = linkCitations(result.draft || '（无草稿）');
  $('evidence-count').textContent = state.evidence.length + ' 条';
  renderEvidence();

  show('answer', true);
  show('empty', false);
}

/**
 * Turns [n] into a chip that scrolls to the matching evidence.
 *
 * The text is escaped first: the draft is model output that has read the
 * document, so it is not markup, and treating it as markup would let a document
 * inject script into the app.
 */
function linkCitations(text) {
  return escapeHtml(text)
    .replace(/\[(\d+)\]/g, (match, digits) => {
      const index = Number(digits);
      if (index < 1 || index > state.evidence.length) return match;
      return `<button class="cite" type="button" data-index="${index}">${index}</button>`;
    })
    .replace(/\n/g, '<br />');
}

function renderEvidence() {
  const list = $('evidence');
  list.innerHTML = '';

  state.evidence.forEach((hit, position) => {
    const index = position + 1;
    const chunk = hit.chunk || {};
    const item = document.createElement('li');
    item.id = 'evidence-' + index;
    item.className = 'evidence-item';

    const sources = (hit.sources || []).join(', ') || '—';
    item.innerHTML =
      `<div class="evidence-head">` +
      `<span class="evidence-index">${index}</span>` +
      `<span class="muted">${escapeHtml(chunk.doc_id || '')} p.${chunk.page_num ?? '?'}` +
      `${chunk.block_type ? ' · ' + escapeHtml(chunk.block_type) : ''}` +
      ` · ${escapeHtml(sources)}</span></div>` +
      `<div class="evidence-text">${escapeHtml(chunk.text || '')}</div>`;

    list.appendChild(item);
  });
}

document.addEventListener('click', (event) => {
  const chip = event.target.closest('.cite');
  if (!chip) return;
  const target = $('evidence-' + chip.dataset.index);
  if (!target) return;
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  target.classList.add('flash');
  setTimeout(() => target.classList.remove('flash'), 900);
});

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
  if (status.running) refreshHealth();
});

api.onLog((line) => {
  $('log-line').textContent = line.length > 160 ? line.slice(0, 160) + '…' : line;
});

// ---------------------------------------------------------------------------
// boot

async function boot() {
  try {
    const history = await api.logHistory();
    if (history && history.length) {
      $('log-line').textContent = history[history.length - 1];
    }
  } catch (error) {
    /* the log is a convenience; failing to load it must not block the UI */
  }

  $('add').addEventListener('click', addDocuments);

  try {
    renderKernelStatus(await api.status());
    await refreshHealth();
    await refreshDocuments();
  } catch (error) {
    notice('无法连接内核：' + error.message, 'bad');
  }
}

boot();
