# freerag

本地优先（Local-First）的桌面端 Agentic RAG 系统。全部模型在安装过程中下载到本地，用户无需手动安装任何东西。

- **桌面端**：Electron（UI 壳层）
- **内核**：Go（业务逻辑、编排、并发）
- **解析**：Python Sidecar（PyMuPDF；PP-DocLayout / TSR 为后续接入点）
- **推理**：llama.cpp Sidecar（Laya 路由/充分性检查 + 生成 LLM + BGE-M3 嵌入）— 未接入
- **存储**：内存分块索引 + 手写 BM25 / 余弦 / RRF 融合，落盘为**单个 JSON 文件**。
  **没有数据库**：`go.mod` 零依赖，SQLite + sqlite-vec 只是 §3 的目标形态，尚未实现

> 完整设计见 [`docs/plan.md`](docs/plan.md)。

## 状态

| 模块 | 状态 | 说明 |
| :--- | :--- | :--- |
| Go 内核 + JSON-RPC 2.0 over stdio | ✅ 已验收 | `internal/ipc` |
| 解析 Sidecar（PyMuPDF 管线） | ✅ 已验收 | 版面块检测、表格转 Markdown、按块分块 |
| 版面分析（PP-DocLayout ONNX） | ✅ 已接入 | `sidecar/layout_onnx.py`，加速器优先；可视化 `scripts/visualize_layout.py` |
| 分块与超短块合并（§5.3 / §5.3.1） | ✅ 已验收 | 类型豁免 → 合并 → 软挂靠 → 丢弃 |
| Go ↔ Python Sidecar 桥接 | ✅ 已验收 | `internal/sidecar` + `internal/parser` |
| 分块索引与 **混合检索** | ✅ 已验收 | `internal/store`：BM25（CJK 二元切分）+ 密集 ANN（Qdrant）+ RRF 融合 |
| 密集检索（BGE-M3） | ✅ 已接入 | `internal/embed`，OpenAI 兼容接口；无 key 时自动降级为纯关键词 |
| **密集 ANN 索引（Qdrant）** | ✅ 已接入 | `internal/dense`；实测 recall@10 = **1.000**；不可达时响亮告警并回退精确扫描 |
| Agentic Loop（medium 模式，§6） | ✅ 已验收 | `internal/agent`；模型可插拔，无模型时降级 |
| **SCA 用 Laya（§6.6）** | ✅ 已验收 | `sidecar/laya.py` + `internal/agent/checker.go`；只喂 draft，~104 ms/次 |
| **解析缓存** | ✅ 已验收 | `sidecar/cache.py`；重复解析 21.5s → 0.18s |
| **模型存储去重** | ✅ 已验收 | `scripts/setup-ollama.sh` 硬链接；7.1 GB → 4.4 GB |
| 检索工具面（§6.7） | ✅ 已验收 | `hybrid_search` / `grep_search` / `list_chunks` / `metadata_search` |
| **模型驱动的工具选择** | ✅ 已验收 | 模型在 4 个工具里自选（§6.1 `ActionMaxTurns=8` 仍是预算上限）；未知工具名剔除、失败回退确定性计划 |
| **Electron 桌面端（产品形态）** | ✅ 可用 | `desktop/`：文档列表 + 拖放/选择文件 + 提问 + 引用跳转 + 实时进度 |
| **进度通知（JSON-RPC notification）** | ✅ 已接入 | `internal/ipc` 的 `Server.Notify`；`index` / `ask` 逐阶段上报 |
| **文档管理 RPC** | ✅ 已接入 | `documents` / `forget` / `status` |
| **安装器（electron-builder）** | ✅ 含随包 Python 运行时，安装即用 | `scripts/build-installer.sh`、`scripts/fetch-python-runtime.sh`；产出约 772 MB，见「打包安装器」 |
| **模型下载（本地化）** | ✅ 已下载 | 4.39 GB，见「本地模型」 |
| **生成 LLM 接入（Qwen3-4B）** | ✅ 已接入 | 经 Ollama / llama.cpp；不可达时降级为抽取式 draft |
| **TSR 表格结构（`tsr.onnx`）** | ✅ 已接入 | `sidecar/tsr_onnx.py` + `sidecar/table_grid.py`；真实论文表格转 Markdown **1/10 → 9/10** |
| Laya 充分性检查 | ⚠️ 模型就位 | ONNX 版可经 onnxruntime 调用；提示构造待完成 |
| BGE-M3 向量检索 | ✅ 已接入（托管接口） | 与 BM25 经 RRF 融合；本地 ONNX 版待做 |
| 安装器 | ❌ 未实现 | plan.md Phase 3 |

## 目录结构

```
freerag/
├── cmd/freerag/           # Go 内核入口（stdio JSON-RPC 服务）
├── internal/
│   ├── ipc/               # JSON-RPC 2.0 over stdio（NDJSON）传输层
│   ├── sidecar/           # 子进程 JSON-RPC 客户端
│   ├── parser/            # 解析服务（定位并调用 Python Sidecar）
│   ├── store/             # 分块索引 + BM25 检索
│   └── agent/             # Agentic Loop（medium 模式）
├── desktop/               # Electron 桌面端
│   ├── main.js            # 主进程：拉起 Go 内核 + RPC 桥接
│   ├── preload.js         # contextBridge 安全桥接
│   └── renderer/          # 界面
├── sidecar/               # Python 解析 Sidecar
│   ├── parse_server.py    # JSON-RPC 入口
│   ├── pipeline.py        # PDF → 版面块 → chunk
│   ├── layout.py          # 版面块检测（PyMuPDF 提供者）
│   ├── chunking.py        # 分块与超短块合并
│   └── tests/             # 单元 + 集成测试
├── scripts/               # install-go.sh / acceptance.sh / 验收断言 / 版面可视化
└── docs/plan.md           # 开发计划（设计依据）
```

## 快速开始

### 1. 依赖

```bash
# Go 工具链（仓库本地，不污染系统）
scripts/install-go.sh                 # 默认 1.27.1，装到 .toolchain/go

# Python 环境（解析 Sidecar 需要 PyMuPDF）
python3 -m venv .venv314              # 需要 Python >= 3.10（PyMuPDF 为 cp310-abi3）
.venv314/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

### 2. 构建与运行

```bash
export PATH="$PWD/.toolchain/go/bin:$PATH"
go build -o bin/freerag ./cmd/freerag

# 交互：每行一个 JSON-RPC 请求
echo '{"jsonrpc":"2.0","id":1,"method":"version"}' | ./bin/freerag
```

### 3. 桌面端

```bash
cd desktop
npm install
npm start
```

Electron 二进制若未随 `npm install` 下载（国内网络），用镜像补装：

```bash
cd desktop
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
  npm install --save-dev electron --registry=https://registry.npmmirror.com
```

## 本地模型

```bash
scripts/download-models.sh          # 全部下载到 models/（默认 all）
scripts/setup-ollama.sh             # 解包 Ollama 运行时并导入 GGUF
```

| 分组 | 文件 | 大小 | 来源 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `deepdoc` | `layout.onnx` | 72.2 MB | `InfiniFlow/deepdoc` | 版面检测（10 类 YOLOv10） |
| `deepdoc` | `tsr.onnx` | 11.7 MB | 同上 | 表格结构识别 |
| `deepdoc` | `det.onnx` / `rec.onnx` / `ocr.res` | 14.8 MB | 同上 | OCR 检测/识别（扫描件兜底） |
| `laya-onnx` | `laya.onnx` + `.data` + tokenizer | 1.61 GB | `receptron/laya-onnx` | 路由 / 充分性检查 |
| `laya-gguf` | `laya_english_ud_q4_k_m.gguf` | 400.5 MB | `mys/laya-GGUF` | ⚠️ 见下方说明 |
| `qwen3-4b` | `Qwen3-4B-Q4_K_M.gguf` | 2.33 GB | `Qwen/Qwen3-4B-GGUF` | 生成 LLM |

下载清单（含 sha256）写在 `models/manifest.json`。

**BGE-M3 按要求不下载**，嵌入检索暂用 BM25。

> ⚠️ **Laya 的 GGUF 不能用**：`mys/laya-GGUF` 是 **ggmlc** 格式（非 llama.cpp），
> Ollama 加载会报 `unknown model architecture: 'ggmlc'`，且 Laya 是单次编码打分模型、
> 不生成 token。可运行的是 **ONNX 版**（`models/laya-onnx`，经 onnxruntime）。
> 该 GGUF 仅为完整性保留。

**网络说明**：`huggingface.co`、`github.com`、`go.dev` 在本网络不可达，
脚本已改用 `hf-mirror.com` / `gh-proxy.com` / `golang.google.cn`；可用
`HF_ENDPOINT` 覆盖。Ollama 官方下载地址会跳转到 GitHub，故同样走代理。

## RPC 方法

内核从 **stdin 读、向 stdout 写** NDJSON 格式的 JSON-RPC 2.0 消息；日志全部走 **stderr**。

| 方法 | 参数 | 说明 |
| :--- | :--- | :--- |
| `ping` | — | 存活检查 |
| `version` | — | 内核版本、已索引 chunk 数、索引路径、Sidecar 配置 |
| `parse` | `path`, `profile?`, `max_chars?`, `max_pages?` | 解析文档并返回 chunk（不入库） |
| `index` | 同上 + `force?` | 解析并写入本地索引；**内容未变的文档自动跳过**（见「增量索引」），返回 `skipped` / `added` / `removed` / `indexed_total` |
| `search` | `query`, `limit?` | BM25 关键词检索 |
| `ask` | `question` | 运行 medium 模式 Agentic Loop，返回 draft / verdict / evidence / trace |
| `tools` | — | 返回当前模式的检索工具面（4 个工具的 JSON schema） |
| `tool` | `name`, `arguments?` | 直接执行一次工具调用，不经过模型 |
| `documents` | — | 列出已索引文档（md5 / 文件名 / 页数 / chunk 数 / 索引时间），供界面显示 |
| `forget` | `md5` 或 `doc_id` | 移除一篇文档的 chunks、清单与向量 |
| `status` | — | 各子系统健康状态（sidecar / 嵌入 / ANN / 生成模型），界面启动时调用 |

**进度通知**：`index` 与 `ask` 会先发若干条**无 `id` 的 JSON-RPC 通知**，同一条流上，然后才是响应：

```json
{"jsonrpc":"2.0","method":"progress","params":{"stage":"parse","file":"paper.pdf"}}
{"jsonrpc":"2.0","method":"progress","params":{"stage":"parsed","pages":38,"chunks":203,"layout":"pp-doclayout"}}
{"jsonrpc":"2.0","id":1,"result":{"added":203,"indexed_total":203}}
```

`stage` 取值：`hash` / `skipped` / `parse` / `parsed` / `stored` / `persisted` / `agent`。
**任何客户端都必须按 `id` 是否为 null 区分**：没有 `id` 的是事件，不是"忘了对应的响应"。
（`index` 约 30 s、`ask` 可达 280 s —— 没有这个通道，界面就只有一个永不变化的转圈。）

示例：

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"index","params":{"path":"/tmp/a.pdf"}}' | ./bin/freerag
echo '{"jsonrpc":"2.0","id":2,"method":"ask","params":{"question":"文档讲了什么"}}' | ./bin/freerag
```

索引默认落在 `<root>/data/index.json`，可用环境变量 `FREERAG_DATA` 覆盖。注意
`FREERAG_DATA` 是**索引文件路径**（不是目录）——传目录会让 `Save` 失败并只在 stderr 留一条 warning。

Sidecar 可用 `FREERAG_PYTHON` / `FREERAG_PARSE_SIDECAR` 覆盖。

## 增量索引（按内容 MD5）

`index` 在**解析之前**先算文件的 MD5，命中已索引记录就整段跳过——既省掉版面检测，也省掉嵌入调用（这一项按 token 计费）：

```bash
# 首次：27.9s（解析 + 嵌入 203 chunks）
# 再次：0.12s，skipped=true          ← 快 234×
echo '{"method":"index","params":{"path":"paper.pdf"}}' | ./bin/freerag
```

跳过的判据有三条，**全部满足**才跳过：

| 判据 | 为什么需要 |
| :--- | :--- |
| 内容 MD5 匹配 | 文档身份是**内容**，不是路径，也不是 mtime |
| 分块规则指纹一致 | 否则会继续返回按旧规则切出来的 chunk |
| 该文档的 chunk 仍在索引里 | 索引被重建过（或部分丢失）就该重做 |

**为什么不用 mtime**：`cp -p` / rsync / 恢复备份都会保留时间戳。实测把 38 页论文换成 5 页版本、并把 mtime **精确保留到纳秒**，基于 mtime 的判断会说「未改动」，而内容哈希正确识别并替换：

```
skipped=False  added=42  removed=28   28 stale chunk(s) replaced, 42 added
```

**改动过的文档是替换，不是叠加**：chunk id 由位置派生（如 `c00059`），改了内容的文件在同一位置产生同样的 id；若不去掉旧块，新块会被当成重复而丢弃，**旧文本就静默留存了**。所以检测到改动时先 `RemoveDoc` 再写入。

| 参数 | 说明 |
| :--- | :--- |
| `force: true` | 无视跳过判断，强制重新解析索引 |

清单（`documents`）与 chunk **存在同一个索引文件里**。分开存放的话，索引被清空后清单仍会宣称文档已索引，而它其实不在索引里——跳过就变成了永久性的空洞。

> 跳过的粒度是**文档级**（整篇内容相同才跳过）。TUIrag 用的是 chunk 级去重，代价是必须在切块之后才能判断，因而**省不掉解析开销**；文档级才能在解析前判断。
> 连带好处：同一 PDF 重复提交时，跳过的判断不需要 sidecar 在场（代码里 sidecar 检查放在跳过判断之后）。

生成模型相关：

| 环境变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `FREERAG_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 服务地址 |
| `FREERAG_MODEL` | `freerag-qwen3` | 生成模型名 |
| `FREERAG_NUM_CTX` | `8192` | 上下文窗口。**必须显式设置**：Ollama 默认 4096，小于 Agentic 循环拼出的证据块，超长 prompt 会被拒绝（HTTP 400）而不是截断 |
| `FREERAG_THINK` | 空（**关闭**） | Qwen3 的推理模式。见下方「推理模式为什么默认关闭」 |
| `FREERAG_KEEP_ALIVE` | `30m` | 模型在显存中的保留时长。Ollama 自己的默认是 5 分钟，短于一个人读完答案再问下一个的间隔 |
| `FREERAG_LAYOUT_THREADS` | `1` | 版面模型的 CPU 线程数（增多会更慢） |
| `FREERAG_LAYA_DIR` | `models/laya-onnx` | Laya ONNX 目录 |
| `FREERAG_LAYA_THREADS` | `1` | Laya 的 CPU 线程数。**实测 1 线程 104ms vs CoreML 434ms**，与版面模型相反 |
| `FREERAG_CACHE_DIR` | `cache/parse` | 解析缓存目录 |

密集检索（可选，见下方"嵌入模型"）：

| 环境变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `FREERAG_EMBED_PROVIDER` | 空（关闭） | `siliconflow` 走托管 BGE-M3；空 = 纯关键词检索 |
| `FREERAG_SILICONFLOW_KEY` | — | `siliconflow` 时必填 |

## 问答延迟：推理模式与预热

启动后第一次提问曾经要 **113 秒**。逐段实测（Apple M5 / 17GB / 4B Q4_K_M **全部在 GPU 上**）后，原因与预想不同：

| 阶段 | 实测 | 占比 |
| :--- | ---: | :--- |
| 模型加载（冷） | **0.5 s** | 微不足道 |
| prompt 处理 | 465 tok/s | 很少 |
| **生成** | **6–8 tok/s** | **全部成本** |
| ↳ 其中 thinking | 246 token 里 384 字符推理 | **没人读** |

所以**"启动时预加载模型"救不了这个** —— 加载本来就只有半秒。真正的问题是模型在把整个 token 预算花在推理上。

**`think: false` 的效果**（同一台机器、同一个问题）：

| 配置 | 生成 token | 耗时 | 答案内容 |
| :--- | ---: | ---: | :--- |
| 推理模式开（原行为） | 246 | 29.7 s | 有 |
| **`think: false`** | **14** | **1.9 s** | 有 |
| 推理开 + `num_predict=96` | 96 | 13.2 s | **0 字符（答案被吃光）** |

最后一行说明这不只是速度问题：**推理与答案共用 `num_predict` 预算**，预算被推理吃光就会返回空答案。因此 `FREERAG_THINK` **默认关闭**，`=1` 可恢复。

端到端（同一份索引、同一个问题）：

```
启动后第一次提问   113s → 15.4s
```

**启动预热**（内核在后台发一个与 tool-planning 同前缀的请求）：

```
model warm-up done in 5.326s
```

它刻意复用 tool-planning 的**系统提示与工具定义** —— Ollama 缓存的是提示词**前缀**，换个提示词预热就只省了那 0.5 s 的加载，白付一次模型调用。预热**不阻塞启动**，失败只记日志：没有 Ollama 时循环本来就退化为抽取式草稿，预热失败只是同一情况的轻微不便。

> 上述数字来自本机实测，机器相关。在更慢的机器上 `think: false` 的相对收益会更大（生成更慢，而浪费的 token 不变）。
>
> `FREERAG_KEEP_ALIVE` 默认 `30m`：卸载模型会**连提示词缓存一起丢掉**，而工具定义与系统提示是每一轮首个请求的主体，重算这个前缀比重新加载权重更贵。

## 验收

```bash
scripts/acceptance.sh
```

覆盖：构建（gofmt/vet/build）、Go 测试、Python 测试、真实 PDF 的端到端解析、
`index → search → ask` 全链路、检索工具面、以及跨进程的协议错误码。

## 版面分析可视化

把 PP-DocLayout 的检测框画在原页上，输出逐页 PNG + 一份自包含的 HTML 报告：

```bash
.venv314/bin/python scripts/visualize_layout.py /path/to/paper.pdf --pages 1-6
# 产物：artifacts/layout/<pdf名>/{page-NN.png, page-NN-boxes.png, report.html}
```

实测（Apple M 系列）：CoreML **0.17s/页**，纯 CPU 1.9s/页（1 线程）～7.8s/页。
无加速器时可用 `FREERAG_LAYOUT_THREADS` 覆盖线程数。用 `--pages` 只跑需要的页。

## 表格还原

一个 `Table` 区域按**精度降序**尝试三条路，最后一条总能返回内容：

| 顺序 | 方法 | 适用 |
| :--- | :--- | :--- |
| 1 | PyMuPDF `find_tables()` | 有完整框线的表格——直接读框线，**精确** |
| 2 | **TSR**（`tsr.onnx`） | 没有框线的表格。TSR 直接给出**行 / 列 / 表头**框，网格是它们的交集 |
| 3 | 区域纯文本 | 兜底 |

顺序不能颠倒：实测在规整的 3×3 测试表上，TSR 反而会报出真实行之间的**幻影空行**，比 `find_tables()` 差。
TSR 的价值是补上第 1 条完全看不见的那些表。

**实测（38 页真实论文，10 个 `Table` 块）**：

| | 转成 Markdown |
| :--- | ---: |
| 只有 `find_tables()` | **1 / 10** |
| 加上 TSR | **9 / 10** |

代价是全篇解析 27.9s → 30.2s（10 个表格合计多约 2s）。剩下的 1 个是**正确回退**——TSR 未找到 ≥2 行 × ≥2 列，退回纯文本。

**单元格文字始终来自 PDF 文字层**（从不用 OCR），所以扫描件的表格仍需 OCR 兜底。

**已知残留**：单元格文字可能串到相邻列（`[169]` 被切开）、无表头时表头行会重复成数据行、表头框分数低时易漏检。
未做：孤立行列剔除、跨页合并、colspan/rowspan 还原。

> ⚠️ **改了表格逻辑必须 bump `sidecar/cache.py` 的 `CACHE_VERSION`。**
> 解析缓存按 `路径 + 大小 + mtime` 命中，它不会自己知道流水线变了。
> 接入 TSR 时就踩了这个坑：版本没 bump，重新索引只花 5.7s（而非 30s），拿到的仍是 TSR 之前的旧结果，
> 于是**表格覆盖率看起来一点没变**——差点据此得出"TSR 没用"的错误结论。

## 嵌入模型（密集检索）

`hybrid_search` = BM25 + 向量，用 **RRF**（reciprocal rank fusion，k=60）融合。向量来自
BGE-M3（1024 维）。

**当前配置为托管接口（SiliconFlow）**，理由是省掉 2.3 GB 权重下载；`internal/embed.Embedder`
接口把本地实现留成可替换项，而不是重写检索层。

```bash
cp .env.example .env          # 填 FREERAG_SILICONFLOW_KEY，.env 已被 gitignore
set -a && . ./.env && set +a
```

> ⚠️ **这与 docs/plan.md §1「完全本地化、安装即用」的目标冲突**：托管嵌入需要联网、按量计费，
> 并且**会把文档分块原文发往第三方**。面向私有文档发布时必须换成本地 provider
> （`FREERAG_EMBED_PROVIDER=local`，尚未实现）。开发阶段用它换"免下载"是合理的，发布不行。

**实测：混合检索比纯关键词多召回什么**

```bash
.venv314/bin/python scripts/compare_retrieval.py paper.pdf --build
```

| 探针 | BM25 | 混合 | 结论 |
| :--- | ---: | ---: | :--- |
| 词面重叠 | 5 | 5 | 各有 2 条独占，融合在仲裁分歧 |
| 同义改写 | 5 | 5 | 各有 1 条独占 |
| **中文查询打英文论文** | **0** | **5** | **稠密全揽**：CJK 二元组与英文零匹配 |
| 标识符 `DAPO` | 5 | 5 | 关键词强项，二者一致 |

无嵌入配置时一切正常工作，只是 `hybrid_search` 退化为纯 BM25，并在 Note 里说明原因。

## 向量索引（Qdrant）

密集检索默认走 **Qdrant**。**只有这一条腿被委托出去**：BM25、`grep_search`、RRF 融合全部留在
`internal/store`（纯 Go）——它们是精确的、不需要服务，也是 Qdrant 挂掉时检索仍然可用的原因。

```bash
# 官方原生二进制，不需要 Docker（按平台换文件名：aarch64/x86_64 + apple-darwin/linux/windows）
curl -sL https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-aarch64-apple-darwin.tar.gz | tar xz -C /usr/local/bin

QDRANT__STORAGE__STORAGE_PATH=./data/qdrant qdrant &
set -a && . ./.env && set +a
export FREERAG_QDRANT_URL=http://127.0.0.1:6333   # 不设 = 用内置精确扫描
```

| 环境变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `FREERAG_QDRANT_URL` | 空（关闭） | 空 = 密集检索走内置精确扫描 |
| `FREERAG_QDRANT_COLLECTION` | `freerag` | 集合名 |
| `FREERAG_QDRANT_M` | `32` | HNSW 每个节点的邻居上限 |
| `FREERAG_QDRANT_EF_CONSTRUCT` | `100` | 建图时的候选宽度 |
| `FREERAG_QDRANT_EF_SEARCH` | `128` | 查询时的候选宽度 |

**实测（20k × 1024 维）**：空闲常驻 **47 MB**、20k 集合 **203 MB**、灌库 4.6 s、
单次查询 **1.6–3.2 ms**（同数据内置扫描 60 ms）、`recall@10` **1.000**。

> 同一批数据上 `coder/hnsw` 的 `recall@10` 只有 **0.215**。所以「纯 Go ANN 不行」是**库的实现问题**，
> 不是 ANN 本身。完整对比与这批数据的适用范围见 docs/plan.md §4。

**降级行为**（都是响亮而非静默的）：

| 情形 | 行为 |
| :--- | :--- |
| 启动时不可达 | stderr 打 `warning: dense index disabled: ...`，退回精确扫描 |
| 运行中请求失败 | 记日志并回退，`hybrid_search` 仍返回能算出的结果而不是报错 |
| 集合维度与嵌入模型不符 | **拒绝启用密集索引**（不同模型的向量数值可比、语义无关） |

`version` 报告当前实际生效的后端，不必猜：

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"version"}' | ./bin/freerag 2>/dev/null \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['dense_index'])"
# {'backend': 'qdrant', 'vectors': 203}
```

> 小语料下若点数低于 Qdrant 的建索引阈值，它会**自动做精确检索**——小库下这反而最快，
> 因此无需特判，小库的 `recall@10` 天然是 1.0。

## Agentic 工具选择

medium 模式的循环（§6.1）把**选哪些工具**交给生成模型：每轮先问一次「为这个问题该调哪些工具」，
模型从 §6.7 的 4 个工具里自选；内核执行后把结果并入证据池，再由 Laya 判充分性。

```
[Action Session] the model chose 1 call(s): metadata_search(block_type=Table page=3)
[Tool] metadata_search(block_type=Table page=3) -> +1 passage(s). 1 chunk(s) matched the filter
[RAGAgent] Round 1: +1 passage(s); pool now 1.
[SCA] Round 1 verdict=SUFFICIENT (missing=0).
```

回退是内建的，而且**每条都写进 trace**，不会静默降级：

| 情形 | 行为 |
| :--- | :--- |
| 没有模型（Ollama 不可达） | 确定性计划：每个 query 一条 `hybrid_search` + 缺失词的最多 2 条 `grep_search` |
| 模型调用失败 | 同上，记 `tool planning failed` |
| 模型没要求任何工具 | 同上，记 `chose no tool` |
| 模型点了 §6.7 之外的工具名 | 剔除，记 `outside the surface`；若一个都不剩则用确定性计划 |

> 调用次数仍受 `ActionMaxTurns = 8` 约束。`draft` 与 SCA 是**独立的模型调用**——§6.6 要求 Laya 只看 draft，
> 把选工具混进起草轮会让「充分性」到底在判什么变得含糊。

## 充分性检查（Laya）

Agentic 循环需要判断"这份 draft 是否足以回答该问题"。这一步用 **Laya** ——一个非生成式的
类型化决策模型：给它一个问题、一组带名字的选项和一段文本，它给每个选项打分，取 argmax。
不生成文本，所以既没有幻觉面也没有 prompt 注入面。

按 §6.6，**只把 draft 喂给它**，不喂原始片段：判断对象是 draft 本身，groundedness 留给生成模型。
序列模板（每个选项前插一个 `[MASK]`，取其位置向量）由 `sidecar/laya.py` 按 checkpoint 规范复现。

```bash
scripts/download-models.sh laya       # ~1.7 GB，若尚未下载
```

实测（真实论文）：

| 问题 | 旧的词重叠启发式 | Laya |
| :--- | :--- | :--- |
| "How does GRPO differ from PPO?" | **2 轮 280s** | **1 轮 76s** |

差别来自判准：启发式只看"问题的词是否出现在证据里"，误判首轮 draft 不足，于是多跑一整轮
（检索 + 生成 + 检查）。Laya 判对了就省掉整轮 —— 这也是 §8 结论"延迟由轮数决定"的直接体现。

> ⚠️ pip wheel 注意：Laya 的 provider 结论**与版面模型相反** —— CPU 单线程 104ms 完胜
> CoreML 434ms（小文本编码器被每次调用的图上传开销拖累）。可用 `FREERAG_LAYA_THREADS` 覆盖。

## 全链路测试

在真实文档上跑通 **解析 → 分块 → 索引 → 4 个检索工具 → Agentic 问答**，
逐阶段输出耗时，并把原始响应写到 `artifacts/e2e/`：

```bash
.venv314/bin/python scripts/e2e_pipeline.py /path/to/paper.pdf
.venv314/bin/python scripts/e2e_pipeline.py paper.pdf --no-ask        # 只验证检索链路
.venv314/bin/python scripts/e2e_pipeline.py paper.pdf --question "..."  # 自定义问题
```

## 桌面端界面

`desktop/` 是可用的文档问答界面，不再是 RPC 调试台：

| 区域 | 内容 |
| :--- | :--- |
| 顶栏 | 健康状态芯片：Ollama / ANN 后端 / 嵌入配置 / sidecar |
| 左栏 | 文档列表（页数、chunk 数、索引时间）；支持**拖入 PDF** 或「添加」选择文件；悬停可「移除」 |
| 主区 | 提问框（回车发送）、进度（实时阶段 + 内核 trace）、答案（引用 `[n]` 可点击跳转到对应证据） |

```bash
go build -o bin/freerag ./cmd/freerag     # 界面需要内核
cd desktop && npm start

# 截图（开发/CI 用，不触发录屏权限弹窗）：
FREERAG_SCREENSHOT=/tmp/ui.png npm start
```

> **草稿是转义后才插入 DOM 的。** 它来自一个读过文档的语言模型，而文档本身也可能含标记；
> 按 HTML 处理等于允许文档往应用里注入脚本。

界面在**订阅之外还会主动拉一次 `status`**：内核早于窗口启动，它的首次状态广播发生在页面脚本加载之前，
只订阅会让状态栏永远停在"连接中"。

**超时是"闲置超时"，不是"总耗时上限"**（`desktop/main.js`）：

| 方法 | 允许完全静默 | 硬上限 |
| :--- | ---: | ---: |
| `ask` | 5 分钟 | 30 分钟 |
| `index` | 10 分钟 | 60 分钟 |
| 其他 | 60 秒 | 60 秒 |

**内核来的任何消息都会重置所有在途请求的闲置时钟。** 这对 `ask` 是必需的：它的第一次模型调用
（决定检索什么）曾经要 60–70 秒且**期间零输出**，而固定 60 秒超时正好卡在这段空白里 ——
一个完全正常的调用因此看起来和挂死一样。现在内核在调模型**之前**就先发一条 `thinking` 进度。

## 打包安装器

```bash
scripts/build-installer.sh --dir    # 未打包的 .app，用于验证安装态路径解析
scripts/build-installer.sh          # 出 dmg / nsis / AppImage
```

产物在 `dist/`。**安装态与源码树是两套布局**，`desktop/main.js` 的 `resolveRuntime()`
统一它们，并把可写路径经环境变量下发给内核：

| 变量 | 安装态取值 | 为什么必须显式下发 |
| :--- | :--- | :--- |
| `FREERAG_DATA` | `<userData>/data/index.json` | 安装后的 app bundle **只读**，写在旁边必然失败 |
| `FREERAG_CACHE_DIR` | `<userData>/cache/parse` | 同上；而且缓存写失败是**静默的**（`cache.py` 吞掉 `OSError`），界面看起来正常却每次全量重解析 |
| `FREERAG_LAYOUT_MODEL` / `FREERAG_TSR_MODEL` | `<resources>/models/deepdoc/*.onnx` | 模型随包分发 |
| `FREERAG_PARSE_SIDECAR` | `<resources>/sidecar/parse_server.py` | sidecar 随包分发 |
| `FREERAG_PYTHON` | `<resources>/python/bin/python3` | **绝不留待 PATH 探测**，见下 |

**体积约 770 MB**：Electron ~250 MB + **Python 运行时 273 MB** + `qdrant` 80 MB +
`deepdoc` 模型 108 MB + 内核 10 MB + 应用外壳。

## 随包的 Python 运行时

```bash
scripts/fetch-python-runtime.sh           # 下载 + 装依赖 + 裁剪（约 4 分钟）
scripts/fetch-python-runtime.sh --force   # requirements.txt 变了之后重建
```

构建进 `desktop/vendor/python`（约 **258 MB**，gitignore 掉 —— 它是产物，不是源码）：一份
**python-build-standalone 的 CPython 3.14** 加上 `pymupdf` / `onnxruntime` / `numpy` / `tokenizers`。

**为什么必须随包带**：干净的机器上没有任何一个依赖。缺了它，应用能启动、能回答索引里已有的文档，
**但解析不了任何新文档** —— 而那正是任何人打开它的第一件事。它的失败形态也最容易被误读：
不是启动报错，而是**第一次解析文档时**才失败，看起来像"应用坏了"而不是"缺依赖"。

**为什么用 python-build-standalone 而不是系统 Python**：它被设计成可重定位（`sys.prefix` 跟着可执行文件走，
脚本第 5 步会验证这一点），不依赖 Xcode 命令行工具、Homebrew 或 pyenv，可以整个塞进 app bundle，
不碰用户自己装的那些东西。

裁剪掉的东西（每一项都对照 sidecar 实际会执行的 import 核过）：

| 裁掉 | 省 | 为什么安全 |
| :--- | ---: | :--- |
| `huggingface_hub` + `hf_xet` | 15 MB | 只有 `from_pretrained` 才会用到；`sidecar/laya.py:192` 用的是 `Tokenizer.from_file` 读本地 `tokenizer.json` |
| `pip` / `setuptools` / `wheel` | 11 MB | 运行时不需要装任何东西（用 `pip uninstall` 卸，不是 `rm`） |
| `include` / `ensurepip` / `idlelib` / `tkinter` / `pydoc_data` 等 | 7 MB | 解释器家具，本项目用不到 |

> **`__pycache__` 是刻意保留的（约 44 MB），不是疏漏。** 安装后的 app bundle **只读**，
> Python 再也无法写回缓存：实测带缓存导入依赖 **0.19s**，不带则**每次启动都要 1.16s**。
> 用 44 MB 换回每次启动的一秒，对天天要开的应用是划算的。缓存也不会过期失效 ——
> pyc 记录源文件的 size 与 mtime，之后被改过的源码照样会重新编译。

> **下载会走镜像。** GitHub 的 release CDN 在部分网络下不可达，而且失败形态是**静默卡住**而非报错。
> 脚本按 直连 → `gh-proxy.com` → `ghproxy.net` 依次重试，每次都有超时上界；
> 可用 `GITHUB_MIRROR=<prefix>` 指定。实测本机直连**从未成功**，走镜像约 170 秒。
> 下载好的 tarball 缓存在 `desktop/vendor/python-runtime.tar.gz`，重建时直接复用。

**三样东西刻意不打包**（由 `scripts/download-models.sh` 另外取）：

| 不打包 | 原因 |
| :--- | :--- |
| Qwen3-4B（~2.4 GB） | 由 Ollama 管理 |
| Laya ONNX（`laya.onnx.data` **1.6 GB**） | 会使下载量翻两倍多，而缺了它只退化为词重叠启发式 |

> ⚠️ **macOS 未签名**，首次启动会被 Gatekeeper 拦。右键 → 打开，或
> `xattr -dr com.apple.quarantine /Applications/freerag.app`。

> ⚠️ **构建需要镜像**：electron-builder 会去 GitHub 取自己的 Electron 发行包。网络不通时它会
> **挂满 10 分钟、然后只报一句 `Timeout awaiting 'request'`**，完全看不出原因。
> `build-installer.sh` 已默认走 npmmirror，可用 `ELECTRON_MIRROR` 覆盖。

## 开发

- 协议：JSON-RPC 2.0，一行一条消息（NDJSON）。
- 边界：Electron 只管 UI；Go 管业务与并发；解析/推理各为独立进程（故障隔离）。
- 测试：`go test ./...` 与 `.venv314/bin/python -m unittest discover -s sidecar/tests`。

## 参考

- `ragflow/internal/rag/agentic-rag/` —— Agentic Loop 对齐 medium 模式的来源
- `TUIrag/layout_chunker.py` —— 版面检测 + PyMuPDF 串行解析管线
- `ragflow/deepdoc/` —— TSR 表格结构识别与 `construct_table`
