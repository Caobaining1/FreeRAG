# 桌面端 Agentic RAG 系统 — 开发计划（Plan）

> 依据：`deepseek_txt_20260928_b9912d.txt`（桌面端 Agentic RAG 系统开发对话记录）
> 补充：解析管线参考 `TUIrag/layout_chunker.py`（PP-DocLayout + PyMuPDF 串行管道），表格解析借鉴 RAGFlow `deepdoc`（TSR 表格结构识别 + `construct_table`）。
> 说明：对话末尾"客户端/开发端划分与版本分发"一节在「采用以下目录结构：」处中断，相关条目在 §11 标注为**待补充**。

## 0. 状态与剩余工作

> **核对时间：2026-09-29**。§9 的勾选已逐条对照代码更新，不再反映计划时的假设。
> 验收基线：`scripts/acceptance.sh` 12/12、Go 全测试通过、Python 96 项通过、真实 38 页论文全链路通过。

### 0.1 剩余工作（按优先级）

**P0 — 不解决就偏离项目目标**

| # | 事项 | 为什么现在必须做 | 位置 |
| :--- | :--- | :--- | :--- |
| 1 | **存储层：关键词索引与增量落盘** | **密集那一半已由 Qdrant 接手**（实测 recall@10 = 1.000，选型见 §4）。剩下的是关键词侧：BM25 词索引与 chunk 文本都在内存、`Save` 仍是 O(全库) 重写。实测 200k chunks：Save **10.3s/次**、内存 1343MB。**是否真要引入 SQLite FTS5，取决于目标语料规模——取舍分析见 §0.3.2** | `internal/store/`（BM25 / grep / RRF） |
> **P0 只剩一项。** 原 **P0-2（打包 Python 运行时）已于 2026-09-29 完成**：
> 随包带 python-build-standalone 的 CPython 3.14 加 sidecar 的四个依赖
> （`scripts/fetch-python-runtime.sh`，裁到约 258 MB），`desktop/main.js` 改成**显式下发
> `FREERAG_PYTHON`** 而不再依赖 PATH 探测。验收用 `env -i` 的干净环境跑打包件解析真实 38 页 PDF：
> **成功，38 页 / 287 块 / 203 chunks，版面引擎 `pp-doclayout`**。打包体积 498 MB → **772 MB**。
>
> 原来的 P0-1（本地嵌入）已由决策定为「始终走 SiliconFlow 托管 API」，
> 理由、代价与缓解见 **§0.3.1**；它从待办中移除，但**它带来的风险没有被解决，只是被接受了**。

**P1 — 功能缺口**

| # | 事项 | 现状 | 位置 |
| :--- | :--- | :--- | :--- |
| 3 | **OCR 兜底（扫描件）** | `det.onnx` + `rec.onnx` + `ocr.res` 已下载未接入；**现在扫描件完全无法解析** | `sidecar/` |
| 4 | **双栏处理** | 依赖版面模型的阅读顺序，未显式处理 x 间隙。缓存部分**已完成** | `sidecar/layout.py` |

> **TSR 缺口的实证与修复（2026-09-29）**：模型驱动工具选择上线后，agentic 路径第一次能触达 `metadata_search`。
> 问「Which tables appear on page 3?」时，模型正确选择了 `metadata_search(block_type=Table, page=3)`，
> 工具也正确返回了 page 3 的对比表——**但 draft 仍回答"没有提到任何表格"**，因为该块内容是被拍平的
> `Singh et al. [169] / Agentic RAG / ✗ ✗ ✓ ✓ ✓`，没有任何表格标记可供识别。
> 即：**检索正确、证据正确、答案仍然错，瓶颈在表格结构而非检索。**
>
> **接入 TSR 后，同一次检索、同一个问题**，draft 变为
> 「The table on page 3 lists several studies, including Singh et al. [169], Liang et al. [108]…」。
> 表格转 Markdown 的比例 **1/10 → 9/10**（见 §0.2）；全篇解析 27.9s → **30.2s**（10 个表格合计多约 2s）。
> 剩余 1 个（page 15）是**正确的回退**：TSR 未找到 ≥2 行 × ≥2 列，退回纯文本。

**P2 — 体验与工程**

| # | 事项 | 说明 |
| :--- | :--- | :--- |
| 6 | **Windows/Linux 的版面加速方案** | 实测 CoreML 0.17s/页 vs CPU 1.9–7.8s/页，而 **CoreML 仅 Apple**。Windows 需 DirectML、Linux 需 CUDA/OpenVINO，且 pip wheel 不同（`onnxruntime-directml` / `-gpu`）。需实测 |
| 7 | **批量解析队列与并行度** | 索引是"一次调用处理一篇"，无队列、无阶段重叠、无并行（§9 Phase 2/5） |
| 8 | **Phase 6 分发方案** | 原对话在此截断，客户端/开发端划分、auto-update 均未定（§11） |
| 9 | **Tier 1 逐页检测缓存（§5.6）** | 策略已定、**未实现**。收益实测预估：只改分块规则 14.2s → **0.6s**，改 1 页 14.2s → **~1.0s**。落地前需先解决打包后的缓存目录可写性与静默失败（§5.6 注意 3） |
| 10 | **去掉 Go 侧的向量副本** | Qdrant 接手后，`store.vectors` 与 JSON 里的 base64 向量已成冗余（@200k 约 **819MB 内存 + 落盘**各一份）。当前特意保留是为了"Qdrant 挂了能回退 + 能重灌"。要动它必须同时给出重灌路径，见 §0.3.2 的 #1 |

### 0.2 已完成（经代码核实）

| 模块 | 状态 | 位置 |
| :--- | :--- | :--- |
| 解析管线（版面 → 按块取字 → 分块） | ✅ 真实 PDF 38 页验收 | `sidecar/pipeline.py` |
| 版面分析 PP-DocLayout ONNX | ✅ 加速器优先，实测 CoreML 0.17s/页 | `sidecar/layout_onnx.py` |
| 版面可视化报告 | ✅ 逐页 PNG + 自包含 HTML | `scripts/visualize_layout.py` |
| 按块取字（PyMuPDF 文字层优先） | ✅ | `sidecar/layout.py` |
| **表格结构识别（TSR）** | ✅ 真实论文表格转 Markdown **1/10 → 9/10**；行列/表头由 TSR 给出，单元格取字走文字层。**已知残留**：单元格文字串列（`[169]` 被切到相邻列）、无表头时表头行重复成数据行、表头框分数仅 0.22 时易漏检。**未做**：孤立行列剔除 / 跨页合并 / colspan-rowspan 还原 | `sidecar/tsr_onnx.py`、`sidecar/table_grid.py` |
| 分块 + 超短块合并/挂靠/丢弃 | ✅ | `sidecar/chunking.py` |
| **混合检索：BM25 + 向量 + RRF** | ✅ 实测中文查询打英文论文：BM25 0 命中 → 混合 5 命中 | `internal/store/` |
| 嵌入（BGE-M3，1024 维） | ✅ **托管接口，已决定长期如此**（见 §0.3.1） | `internal/embed/` |
| 4 个检索工具 | ✅ 全部有测试 | `internal/agent/tools.go` |
| **模型驱动的工具选择** | ✅ 模型在 4 个工具里自选；未知工具名剔除、失败回退确定性计划。实测选对 `metadata_search(block_type=Table, page=3)` | `internal/agent/loop.go`、`internal/agent/ollama.go` |
| **进度通知** | ✅ `ipc.Server.Notify`（无 id 的 JSON-RPC 通知）；`index` / `ask` 逐阶段上报：hash / skipped / parse / parsed / stored / persisted / agent | `internal/ipc/jsonrpc.go` |
| **文档管理 RPC** | ✅ `documents`（含 md5/页数/索引时间/实际 chunk 数）、`forget`、`status`（各子系统健康 + 哪些没探测） | `cmd/freerag/main.go` |
| **桌面端界面（产品形态）** | ✅ 文档列表 + 拖放/选择文件 + 提问 + 引用 `[n]` 跳转 + 实时进度；草稿转义后进 DOM | `desktop/renderer/` |
| **安装器（electron-builder）** | ✅ **772 MB**，含随包 Python 运行时，安装即用；路径全部落到 userData、Qdrant 随包自启；干净环境实测可解析新文档 | `scripts/build-installer.sh`、`scripts/fetch-python-runtime.sh` |
| Agentic 循环（medium） | ✅ 3 轮 SCA + 重写；含提取式兜底 | `internal/agent/loop.go` |
| **SCA 用 Laya（§6.6）** | ✅ 类型化决策，只喂 draft；约 104ms/次 | `sidecar/laya.py`、`internal/agent/checker.go` |
| 生成 LLM（Qwen3-4B via Ollama） | ✅ 含 `num_ctx` 与 prompt 预算护栏 | `internal/agent/ollama.go` |
| **问答延迟（推理 token）** | ✅ `FREERAG_THINK` 默认关闭（单次调用 29.7s → **1.9s**）；后台预热；`FREERAG_KEEP_ALIVE=30m`。端到端首次提问 **113s → 15.4s**；顺带修掉"推理吃光预算返回空答案" | `internal/agent/ollama.go`、`cmd/freerag/main.go` |
| **桌面端闲置超时** | ✅ `ask` 5 分钟静默 / 30 分钟上限；内核任何消息重置时钟；内核在调模型前先发 `thinking` 进度（原固定 60s 超时正好卡在首次模型调用的静默期） | `desktop/main.js` |
| 上下文预算（4K–8K） | ✅ 显式 `num_ctx` + 按 token 换算的字符预算 | `internal/agent/loop.go` |
| **解析缓存（sidecar）** | ✅ 实测重复解析 21.5s → **0.18s**。键含**路径 + 大小 + mtime**，因此 `cp -p` / rsync / 恢复备份保留时间戳时可能假命中 | `sidecar/cache.py` |
| **增量索引（按内容 MD5）** | ✅ 重复 `index` **27.9s → 0.12s**（粗算快 234×）；内容变更自动替换旧块；MD5 清单与 chunk 同文件持久化 | `internal/store/doc.go`、`cmd/freerag/main.go` |
| JSON-RPC over stdio | ✅ 含跨进程错误码 | `internal/ipc/` |
| 模型下载与 manifest | ⚠️ 脚本可用，未与安装器集成 | `scripts/download-models.sh` |
| **模型存储去重** | ✅ 硬链接，实测 7.1 GB → **4.4 GB** | `scripts/setup-ollama.sh` |
| 全链路与对比测试脚本 | ✅ | `scripts/e2e_pipeline.py`、`scripts/compare_retrieval.py` |

**Laya 接入的实测收益**（这是本轮最大的一项）：

| 问题 | 旧（词重叠启发式） | 新（Laya） |
| :--- | :--- | :--- |
| "What is RL-based agentic search?" | 1 轮，70.9s | 1 轮，66.7s |
| "Along which three dimensions…" | 1 轮，58.3s | 1 轮，75.1s |
| "How does GRPO differ from PPO?" | **2 轮，280.2s** | **1 轮，76.1s** |

第 3 题之所以差 3.7 倍：词重叠启发式只看"问题的词是否出现在证据里"，误判首轮 draft 不足，
于是又跑了一整轮（检索 + 生成 + 检查）。Laya 判对了，省掉整轮。**三题答案均正确。**

Laya 的 provider 结论也值得记：**CPU 单线程 104ms 完胜 CoreML 434ms**（与版面模型相反——
视觉大模型受益于 CoreML，小文本编码器则被每次调用的图上传开销拖累）。

**一句话概括**：解析、检索、问答三段链路全部打通并经过真实文档验收；SCA 已换成真实模型、工具选择已交给模型。
**"完全本地"这一条已明确放弃**（嵌入固定走托管，见 §0.3.1）；剩下的是存储层能否撑住语料规模（§0.3.2）。

---

### 0.3 已决定的技术取舍（含接受的代价）

#### 0.3.1 嵌入模型始终使用 SiliconFlow 托管 API

**决定（2026-09-29）**：嵌入固定走托管接口（`FREERAG_EMBED_PROVIDER=siliconflow`，BGE-M3，1024 维），
**不实现本地嵌入 provider**，本地 ONNX 版从计划中移除。

这条决定**与 §1 的目标直接冲突**，所以代价必须显式记录——是"接受"，不是"以后再说"：

| 代价 | 具体影响 |
| :--- | :--- |
| **破坏「完全本地化」** | 嵌入成为唯一仍在运行期依赖外部服务的环节 |
| **文档分块原文发往第三方** | 每次 `index` 都上传分块文本。**私有/敏感文档不适用**——这对"本地优先"的产品定位是硬伤 |
| **需要联网且按量计费** | 离线不可用；大批量索引有成本 |
| **索引不可移植** | 换 provider / 换模型会让已有向量失效。`store.SetEmbedder` 会按**宽度**拒绝混用，但**同宽度下的语义漂移无法被检测** |

**已做的缓解**（均已在代码里）：

- `Embedder` 接口保留（`internal/embed/embed.go`）——本地实现仍是"加一个实现"的距离，不是重写检索层；
- 嵌入失败**不会**让索引失败：chunks 仍以纯关键词入库（`indexChunks` 的降级路径），`version` 的 `embedding` 字段如实报告 `enabled` / `vectors`；
- 没有 key 时 `hybrid_search` 退化为纯 BM25，并在 Note 里说明原因（不会假装成混合检索）。

**若要改回本地**：需要 BGE-M3 权重（约 2.3GB，当前 `models/` 中没有）。**这是一次下载决策，不是架构改动。**

#### 0.3.2 Qdrant 之后，SQLite FTS5 还需要吗

**结论：密集那一半已不需要；关键词那一半取决于目标语料规模，而这个问题至今没有答案。**

原 P0-2 想解决的问题有三块，Qdrant 只解决了其中一块：

| # | 问题 | @200k 实测 | Qdrant 解决了？ |
| :--- | :--- | :--- | :--- |
| 1 | 向量存在 Go 内存 + JSON（base64，5.5KB/chunk） | 约 819MB | ✅ **可以省掉**——Qdrant 已持有向量，Go 侧那份只是回退与重灌来源 |
| 2 | BM25 词索引在内存（`terms` / `lengths` / `df`） | 与 #1 共同构成 1343MB | ❌ **没有**——Qdrant 不持有文本与词索引 |
| 3 | `Save` 全量重写（O(全库)，加一篇就重写一遍） | 10.3s/次 | ❌ **没有**——JSON 落盘与 Qdrant 无关 |

**#1 是白拿的收益，值得单独做。** **#2/#3 才是"要不要引入 SQLite"的实质**，而它的触发条件是**语料规模**：
§8 已记录 20k chunks 时手写层够用（查询 46ms、内存 124MB、Save 664ms），**引入数据库是净风险**。
目标规模（几十上百篇 vs 上千篇）**仍未确认**。

**Qdrant 的存在还多出第三种选择**，比引入 SQLite 更省：

| 方案 | 做法 | 代价 |
| :--- | :--- | :--- |
| A. SQLite FTS5 | 词索引与文本下沉 SQLite，`bm25()` 内置 | 引入**第二个存储**，需与 Qdrant 双写同步；FTS5 的 CJK 分词要自行处理 |
| B. **BM25 下沉为 Qdrant sparse vector** | 用现有 `Terms()` 生成稀疏向量，Qdrant 原生支持 sparse 与服务端 RRF | 需自己算 BM25 权重（或接受 Qdrant `idf` modifier 的简化版）；**但不必引入第二个存储** |
| C. 维持现状 | 不动 | @200k 时 1343MB / Save 10.3s |

**推荐顺序**：先做 #1（去掉 Go 侧向量副本，见 §0.1 P2-10），再等目标规模确认后在 A 与 B 之间选。
**规模确认前不要动 #2/#3**——那正是 §8 已经判过的净风险。

---

## 1. 项目目标

构建一个**完全本地化、安装即用**的桌面端 Agentic RAG 系统：

- 桌面端用 **Electron** 实现 UI。
- 后端与核心逻辑用 **Go** 实现。
- 所有模型与 `llama.cpp` 在**安装 exe 过程中自动下载到本地**，用户无需手动下载任何东西。
- 覆盖**传统 parser 阶段 + query 阶段**，包含**版面分析、文本提取、Agentic Loop**。
- Agentic Loop 中使用 **Laya** 作为 route（路由）与 sufficient check（充分性检查）模型。
- 面向文本 QA 问答领域。

> ⚠️ **例外（2026-09-29 决定）**：**嵌入不走本地**，固定使用 SiliconFlow 托管的 BGE-M3。
> 因此"完全本地化"应读作"**除嵌入外**全部本地"，且**私有 / 敏感文档场景不适用**。
> 理由、代价与已做的缓解见 **§0.3.1**。

---

## 2. 硬约束

| 约束 | 数值 | 影响 |
| :--- | :--- | :--- |
| 系统内存 | **8GB RAM** | 决定嵌入模型、向量库、进程常驻内存策略 |
| 显存 | **8GB VRAM** | 决定能否并行、模型是否必须 4-bit 量化、能否同时驻留 |
| GPU 显存瓶颈 | 主要矛盾 | OCR / TSR / LLM / Laya 同时常驻易超 8GB → **必须动态加载** |
| 单篇解析耗时（电子版 PDF） | **秒级**（文字层直读，CPU） | OCR 仅兜底，见 §5 |
| 单篇解析耗时（扫描件） | 约 **45–60 秒/篇**（1MB、约 20 页） | 主要耗在 GPU OCR（约 3 秒/页） |
| 并行 parse 上限 | **建议 2，极限 4** | 由显存与 KV Cache 线性增长决定 |

**结论**：GPU 显存是唯一硬瓶颈；CPU 侧（版面分析、文字提取、嵌入）应尽量卸载到 CPU；**OCR 从"必经阶段"降级为"扫描件兜底"**（见 §5）。

---

## 3. 总体架构（本地优先 Local-First，无需云原生）

> 明确结论：**不需要 Kubernetes / 服务网格 / 微服务等云原生架构**，采用分层本地架构。

```
┌─────────────────────────────────────────────┐
│  Electron 壳层（UI 层）                       │
│  职责：界面渲染、交互、托盘/菜单等系统原生功能   │
└───────────────┬─────────────────────────────┘
                │  JSON-RPC 2.0 over Stdio (NDJSON)
┌───────────────▼─────────────────────────────┐
│  Go 内核（业务逻辑与编排层）                   │
│  - 文件调度与解析编排（批量队列）               │
│  - Agentic Loop 编排（路由 / 充分性检查 / 工具） │
│  - 向量检索接口                               │
│  - 上下文组装（Prompt）                        │
│  优势：静态二进制、常驻 <80MB、并发模型友好      │
└───────┬───────────────────────┬─────────────┘
        │ JSON-RPC over Stdio   │ 本地调用
┌───────▼──────────────┐  ┌─────▼─────────────┐
│ 解析 Sidecar（Python） │  │ 本地推理服务（Sidecar）│
│ PP-DocLayout / PyMuPDF│  │ Laya / LLM / Embed  │
│ TSR / OCR(兜底)       │  │ 独立进程，故障隔离    │
└──────────────────────┘  └─────┬───────────────┘
        ┌────────────────────────┘
┌───────▼───────────────┐
│ 数据持久层              │
│ SQLite(FTS5) 全文      │
│ + coder/hnsw 向量索引   │
└───────────────────────┘
```

**核心原则**：关注点分离——Electron 只管 UI，Go 管业务与并发，解析与推理各为独立 Sidecar；任一层崩溃不拖垮整体。

---

## 4. 模型选型（8GB 约束）

| 组件 | 推荐模型 | 参数量/量化 | 预估占用 | 运行设备 |
| :--- | :--- | :--- | :--- | :--- |
| 版面分析 | **RAGFlow `layout.onnx`**（PP-DocLayout 系，YOLOv10，10 类） | 75MB，输入 1024×1024 | ~75MB 权重 | **加速器优先**；实测 CoreML **0.17s/页**，纯 CPU 1.9s/页（1 线程）～7.8s/页 |
| 文本提取（主） | **PyMuPDF / MuPDF 文字层** | 无模型 | CPU 侧 ~0 | **CPU**（毫秒级） |
| 文本提取（兜底） | **GLM-OCR**（备选 EasyOCR / Tesseract） | 0.9B (FP16/INT4) | ~2GB VRAM（临时） | GPU（**仅扫描件**时动态加载） |
| 表格结构识别 TSR | **RAGFlow `tsr.onnx`**（TableStructureRecognizer，6 类标签） | 小型 ONNX 检测模型 | <0.5GB | **CPU**（可 GPU 加速） |
| 路由 / 充分性检查 | **Laya（Router 模式）** | 322M–421M | ~1GB VRAM | GPU（常驻） |
| 生成 LLM | **Qwen3-4B**（备选 Llama 3.1 8B / Agents-A1-4B-Fable） | 4B (Q4_K_M) | ~3–4GB VRAM | GPU（动态加载） |
| 嵌入 | **BGE-M3**（备选 nomic-embed-text / Qwen3-Embedding-0.6B / bge-small-*） | 568M（XLM-RoBERTa-large），1024 维 | ~0.6GB RAM（量化更少） | **CPU**（可选 GPU） |

> **实现说明（2026-09-29）**：检索层已按本节实现为 **BM25 + 向量 + RRF 融合**，但向量目前来自
> **托管 BGE-M3（SiliconFlow）**，本地 ONNX 版本尚未接入。这是开发期的取舍（省 2.3GB 下载），
> **与 §1「完全本地化」冲突**——见 §10 新增风险项。
| 向量索引 | **Qdrant**（外部进程，原生二进制） | ANN，recall@10 实测 **1.000** | 47 MB 空闲；203 MB @20k×1024 | **CPU** |
| ↳ 回退 | 手写精确扫描（内置于 `internal/store`） | Qdrant 不可达时自动接管 | 1343 MB @200k | **CPU** |

**向量存储的实测选型（2026-09-29，在 8GB 机器上实测，非文档推断）**

原定方案是 **SQLite + `sqlite-vec`**，实测**纯 Go 下不成立**——`sqlite-vec` 是 C 扩展，
`modernc.org/sqlite` 加载不了（`vec_version()` / `vec0` 均报错）。随后实测的**纯 Go** 候选全部被否决：

| 方案 | 查询 @200k | 构建 @200k | 启动加载 | 内存 | 结论 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 手写暴力扫描 | 740 ms | 增量≈0 | 5.9 s | 1343 MB | ⚠️ 精确，但随规模线性劣化 |
| `chromem-go` v0.7.0 | 143 ms | 19 s | **20.4 s** | ~1617 MB | ❌ 查询实为线性扫描；并发灌数据**丢 5/200000 条** |
| `coder/hnsw` v0.6.1 | **0.35 ms** | 3m43s | 2.2 s | 1289 MB | ❌ **召回率不达标**，见下 |

> ### ⚠️ 重要修正：问题不是「ANN 不行」，而是「纯 Go 的 ANN 库不行」
>
> 上一轮的结论写得过宽。原记录是「ANN 经召回率验证不达标，故不引入 ANN」，
> 但**被验证的只有 `coder/hnsw` 一个实现**。用**完全相同的数据与指标**复测 Qdrant，结论反转：
>
> | 索引 | self-hit | recall@1 | recall@10 | 单次搜索 |
> | :--- | ---: | ---: | ---: | ---: |
> | `coder/hnsw`（M=64，其最优参数） | 76.5% | 0.300 | **0.215** | 218 µs |
> | **Qdrant**（M=32, ef=100） | **98.5%** | **1.000** | **1.000** | 1.7 ms |
>
> 数据完全相同（20k 向量、seed 7、同样 20 个真实问句嵌入），
> 所以**低召回是库的实现问题，不是数据的性质**。
> 佐证：Qdrant 单次 1.7 ms vs 暴力扫描 60 ms——它确实在走 ANN 索引，而召回反而满分。
>
> `coder/hnsw` 的失效特征（图内向量查自己只中 72%，EfSearch 从 20 提到 200 反而变差 146→139）
> 是**图构造缺陷**的特征，调参无法解决，也不该推广成「ANN 不可用」。
>
> **结论：采纳 ANN，但用 Qdrant，不用纯 Go 内嵌库。**

> ### 关于这批数据的诚实说明（结论的适用范围）
>
> 203 个真实向量来自论文，但**20k 规模是把真实向量加高斯扰动复制出来的（σ=0.02）**，
> 是**模拟语料而非真实语料**：每个真实向量周围有约 100 个近重复点。
> 这让「精确 top-10」本身带有任意性（近重复点之间语义等价），因此
> **这批 recall 数字既可能偏悲观，也不代表真实 20k 语料的质量**。
> 它适合做**索引之间的横向对比**（数据同、指标同），不适合当作绝对质量结论。
> **待办：用真实多文档语料（≥100 篇）重测。**

**Qdrant 实测（v1.19.1，macOS arm64 原生二进制，非 Docker）**

| 项 | 实测 |
| :--- | :--- |
| 空闲常驻 | **47 MB** |
| 20k × 1024 维集合 | **203 MB**（裸向量 80 MB） |
| 灌 20k | 4.6 s |
| 建 HNSW | 0.25–1.3 s |
| 单次查询 | **1.6–3.2 ms**（同数据暴力扫描 60 ms） |
| 原生混合 | dense + sparse + **服务端 RRF 融合**，排序实测正确 |
| 小语料行为 | 203 点低于建索引阈值时**自动做精确检索**——小库下反而最优，无需特判 |

选它的另一个理由是**能力对齐**：稠密与稀疏向量同库、服务端 RRF、官方 Go 客户端（REST + gRPC），
且是**单文件原生二进制**，macOS / Linux / Windows 可直接分发，不需要 JVM 或 Docker。

**同时否决的候选**：

| 候选 | 否决理由 |
| :--- | :--- |
| **Milvus** | 部署需 etcd + MinIO + 消息队列，桌面端不可接受；Milvus Lite 仅 Python 侧 |
| **Chroma** | Python 服务；混合检索弱（无原生稀疏向量），BM25 要自己补 |
| **Elasticsearch / OpenSearch** | JVM，常驻数 GB，8GB 预算下不可行 |

**集成架构（2026-09-29 已实现）**

只把**密集这一条腿**委托给 Qdrant，其余全部留在 Go 内：

| 组件 | 位置 | 为什么留在本地 |
| :--- | :--- | :--- |
| BM25 + CJK 二元切分 | `internal/store`（纯 Go） | 精确，且**不需要任何服务** |
| grep（字面 / 正则） | `internal/store`（纯 Go） | Qdrant 没有正则全文能力 |
| **密集 ANN** | **`internal/dense`（Qdrant）** | 唯一 O(N)、唯一真正需要索引的腿 |
| RRF 融合 | `internal/store`（纯 Go） | 两路都在本地拼装，结果可审计（`Hit.Sources`） |

接口为 `store.DenseIndex`（`Upsert` / `Search` / `Len`），**回退是内置且响亮的行为**：

- Qdrant **启动时不可达** → 打 `warning: dense index disabled: ...`，退回精确扫描；
- **运行中失败** → 记日志并回退，`hybrid_search` 仍返回能算出来的结果，而不是报错；
- **维度不匹配** → 拒绝（不同模型的向量数值可比、语义无关，混用会返回自信的胡话）。

**Laya 版本选择**：
- `laya`（421M, ModernBERT-large）：英语首选，上下文 512 tokens。
- `laya-multilingual`（322M, mmBERT-base）：多语言首选，支持 100+ 语言，上下文 1024 tokens，速度为英语版 2 倍。
- `laya-typed-decisions`（421M）：复杂类型化决策工作流。
- 使用内置 **Router 模式**自动检测语言并分派，单次前向 ~33ms（批量 ~7.2ms/问题）。
- 特点：非自回归"System 1 决策引擎"，输出 `choice`/`score`/`noul`，**不生成文本 → 无幻觉风险、延迟极低**。

**核心逻辑**：版面分析、文字提取、TSR、嵌入全部下沉 CPU；GPU 只常驻极轻量 Laya，按需动态加载 LLM；**OCR 仅在检测到无文字层的扫描件时才加载**。

**版面模型的实测修正（重要）**：

| 结论 | 数据 |
| :--- | :--- |
| 原计划"PP-DocLayout-S，14.5ms/页"**过于乐观** | 实际权重 75MB、输入 1024×1024，属 L 级而非 S 级 |
| **必须优先使用加速器** | CoreML **0.17s/页** vs 纯 CPU 1.9～7.8s/页（同页、同输入） |
| CoreML 结果可信 | 与 CPU 输出**逐框一致**（9/9 区域完全相同），非近似 |
| CPU 上线程越多越慢 | 1/2/3/4 线程 = 1.9/3.7/5.4/7.4s/页（算子不并行，线程只增竞争） |
| 输出质量 | 段落级语义块；同一份 4 页 PDF：**31 chunks**（PP-DocLayout）vs **123 chunks**（PyMuPDF 行级碎片） |

结论：**版面检测走加速器（CoreML/DML/CUDA），无加速器时必须限制 CPU 线程数为 1**；Windows/Linux 需验证 DirectML/OpenVINO 或改用更小的版面模型。

---

## 5. 解析层（Parser）设计

### 5.1 总体管线（TUIrag 风格：PP-DocLayout + PyMuPDF）

参考 `TUIrag/layout_chunker.py::LayoutAwarePDFChunker.chunk_pdf` 的串行管道，**核心是用 PyMuPDF 直读 PDF 内嵌文字层（CPU），而非逐页 OCR**：

```
PDF
 → ① 打开 / 按需渲染页面位图(dpi≈150)      [仅版面检测与裁剪需要]
 → ② 版面检测 PP-DocLayout-S(CPU)          → 区域块:
                                             Title / Text / Table / Figure / Equation / Caption
 → ③ 按块类型分派内容提取：
      Title / Text  → PyMuPDF get_text("words", clip=bbox)   ← 主路径，CPU，毫秒级
      Table         → TSR 结构识别 + PyMuPDF 文字层按 cell 归位（见 5.2）
      Figure        → 裁剪保存 +（可选）OCR 读图中文字
      Equation      → PyMuPDF words；为空则 OCR 兜底
      文字层为空（扫描件）→ OCR 兜底（GLM-OCR / EasyOCR / Tesseract）
 → ④ 语义块组装（携带 page_num / block_type / bbox / font_size / parent_section / source_file）
 → ⑤ 分块：**每个 bbox 块直接作为一个 chunk**（策略 A，不合并，见 5.3）
 → ⑥ 富元数据注入 → 入库
```

**关键点**：电子版 PDF 全程 CPU、无需 OCR；**只有扫描件/图片文字才触发 GPU OCR**。这让 §2 中"60s/篇"不再是普遍路径。

**双栏处理**：`_extract_text` 依据 word 的 x 坐标间隙（`max_gap > avg_gap*5` 且 `> 区域宽*0.15`）判断左右栏，分别按 y 分组成行再拼接。

### 5.2 表格解析（借鉴 RAGFlow `deepdoc`）

RAGFlow 的表格能力远强于 TUIrag 的"bbox 内按 y/x 排序取字"。其实现（`deepdoc/vision/table_structure_recognizer.py` + `deepdoc/parser/pdf_parser.py::_table_transformer_job`）：

1. **裁剪表格图**：版面检测标出 table 区域后，从高 DPI 页面图裁剪。
2. **方向校正**：`_evaluate_table_orientation` 尝试 0/90/180/270，用 OCR 置信度综合评分选最佳角度，必要时对旋转图重跑 OCR。
3. **TSR 表格结构识别**：专用 ONNX 模型（`tsr.onnx`）输出 6 类组件：
   `table` / `table column` / `table row` / `table column header` / `table projected row header` / `table spanning cell`。
4. **结构与文字对齐**：TSR box 映射回页面坐标；为每个文字 box 打标签
   `R`(行) / `C`(列) / `H`(表头) / `SP`(跨格)，配 `R_top/R_bott`、`C_left/C_right`、`H_*` 边界，
   用 `Recognizer.find_overlapped_with_threshold` 与 `find_horizontally_tightest_fit` 完成匹配。
5. **`construct_table` 重建表格**：
   - 抽出 caption（`is_caption`：`[图表]+[ 0-9:：]{2,}`、`Fig.\d+`、`Figure \d+`、`Table \d+`）。
   - 单元格文本按正则分类 `blockType`（Dt 日期 / Nu 数字 / Ca 代码 / En 英文 / NE 混合 / Sg 单字 / Tx / Lx / Nr / Ot）。
   - 行排序 `sort_R_firstly`、列排序 `sort_C_firstly`（跨页用 `sort_X_firstly`）。
   - 建 `tbl[行][列]` 网格，再做 **"remove single in column / row"** 剔除孤立行列。
   - **表头检测** `hdset`（多数 H 规则，`h/cnt > 0.5`）。
   - `__cal_spans` 由 `SP` 框计算 **colspan / rowspan**。
   - 输出 **HTML 表格**（`__html_table`：`<caption>` + `<th>`/`<td>` + span）或 **描述性文本**（`__desc_table`：`表头：值; ...`，并附 `——来自"<caption>"`）。
6. **列扫描 / 跨页**：`_concat_downward`、`表格跨页合并`（`crosspage` 时改按 x 排序）。

**适配到本项目的做法**（结合 5.1 的 PyMuPDF 主路径）：

| 步骤 | 电子版 PDF（主路径） | 扫描件（兜底路径） |
| :--- | :--- | :--- |
| 结构 | **TSR ONNX** 出行/列/表头/跨格 | 同左 |
| 文字来源 | **PyMuPDF 文字层**：word/span 按 bbox 归入 TSR 单元格 | OCR box |
| 方向校正 | 一般不需要（文字层带坐标） | 需要（RAGFlow 的 4 角度评估） |
| 输出 | HTML + 描述性文本双份 | 同左 |

- **TSR 模型沿用 RAGFlow `tsr.onnx`**（`InfiniFlow/deepdoc`），体积小，可随安装包下载，CPU 可跑。
- 重建逻辑（行列排序、孤立行列剔除、表头判定、`__cal_spans`、HTML 生成）**直接移植 RAGFlow `construct_table` 的算法**，与模型无关。
- 入库同时保留 **HTML**（保结构）与 **描述性文本**（供纯文本检索 / embedding / LLM 阅读）。

### 5.3 分块策略：每个 PP-DocLayout bbox 块 = 一个 chunk

**不做合并/重切**（放弃 TUIrag 策略 B）：PP-DocLayout 检出的**每个 bbox 块直接作为一个 chunk**（对应 TUIrag `_build_chunks_strategy_a`）。理由：BGE-M3 窗口 **8,192 token（≈ 12k 字符，见 §5.5）** 足以容纳单个版面块，无需再合并或按固定字符数重切。

- **块 → chunk 一一对应**：`{text: 该块提取内容, metadata: {page_num, block_type, bbox, font_size, parent_section, source_file}}`。
- **上下文靠元数据、不靠合并**：TUIrag 用"Title 归组 Section"补章节上下文；这里改为在 metadata 保留 `parent_section`（最近一个 Title），检索/Prompt 组装时按需拼接，不牺牲块粒度。
- **Table / Figure 与 Caption 不合并**：Caption 作为独立 chunk，或在 metadata 加 `caption` 字段引用最近的 Caption 文本（只挂引用，不改块边界）。
- **必要保护（三项）**：
  1. **超限拆分**：单块超过 BGE-M3 窗口（§5.5）时，按句子边界拆成多块，**不截断**（避免丢内容）。
  2. **超短块合并/挂靠（必做，见 §5.3.1）**：不允许 <N 字符的碎块单独入库。
  3. **噪音过滤**：同一文本出现 ≥3 次的页眉/页脚、纯页码/装饰符号（`[0-9  •一—-]+`）直接丢弃。
- **缓存**：`layout cache` + `chunk cache`，key = `md5(path|backend|dpi|threshold)`（无 strategy），按 `file_mtime` 失效。
- **内存管理**：每页处理完 `del pix` + `gc.collect()`，避免页面位图累积。
- **暂不引入父子块索引（parent-child）**：当前为**单层扁平索引**，每个 bbox 块直接入库（不借鉴 TUIrag 的 HierarchicalNodeParser / 子块检索→父块回溯）。若后续实际出现召回不足，再评估是否加。

#### 5.3.1 超短块合并算法（借 RAGFlow 判据 + TUIrag 缓冲）

**短块定义**：文本 < 40 中文字符 / < 20 英文词（或 < 32 tokens）。

**处理优先级**（前一步成功即止）：

1. **类型豁免**：`layout_type ∈ {Title, Table, Figure, Equation}` 的块不参与合并/丢弃（RAGFlow `usefull()` 的 layout_type 分支，`pdf_parser.py:1550-1557`）——标题天然短但重要。
2. **软挂靠（首选）**：短块挂到相邻主块，**只写 metadata 的 `attached_to`，不改块边界**（贴合"每个 bbox 块 = 一个 chunk"）。条件需同时满足：
   - 同一 `parent_section`；
   - 垂直相邻：与上/下块间隔 < `mean_height × 1.5`（RAGFlow `_naive_vertical_merge:1047`）；
   - x 重叠 ≥ `min(width_a, width_b) × 0.3`（RAGFlow `:1051-1054`）；
   - 上一块结尾非终止标点（`。？！?` / 英文 `.!?`）——RAGFlow `feats:1064-1065`。
3. **缓冲合并（次选）**：连续同类型短块（`Text`/`Equation`）在同一 `parent_section` + 同页时，按 TUIrag `_flush_text_buffer` 合并为一个块。
4. **丢弃（兜底）**：以上都失败时按 RAGFlow `__filterout_scraps` 的判据（`:1543-1601`）：
   - **保留**：宽度 > 页宽/3，或 高度 > `mean_height`（`usefull()`）；
   - **丢弃**：平均宽度占比 `mw/pw < 0.35` 且 `mw ≤ 200`——页眉/页脚/页码类。
5. **合并后不截断**：合并结果若超 BGE-M3 窗口（§5.5），按句边界拆分（`_split_at_sentences`）。

**为什么不照搬 RAGFlow 的垂直合并**：`_naive_vertical_merge`（`:1013`）依赖 `mean_height` / `mean_width` 等版面统计与大量中英标点特征，移植成本高；且 `_concat_downward`（`:1117`）在 `__call__` 路径**已被 early return 禁用**（只做排序，`:1119`），`_merge_with_same_bullet`（`:1267`）亦无调用者——说明其跨行拼接曾有稳定性问题。故**只取其碎片判据，不取其合并实现**。

### 5.4 语言与实现边界

| TUIrag 组件 | Go 等价物 | 结论 |
| :--- | :--- | :--- |
| PyMuPDF (fitz) | `go-fitz`（MuPDF 绑定） | 需先 PoC 验证是否暴露 **word-level bbox / clip** 提取；否则 CGO 直连 MuPDF |
| pdfplumber | 无可靠等价物 | **不采用**，保留 PyMuPDF |
| PaddleX PP-DocLayout | ONNX Runtime 或 Python Sidecar | 走 ONNX 或 sidecar |
| TSR / `construct_table` | 纯算法可移植 + ONNX 推理 | 算法用 Go 重写，模型走 ONNX |

**推荐**：解析层整体保留 **Python Sidecar**（PyMuPDF + PP-DocLayout + TSR），Go 内核通过 JSON-RPC 调用（复用 TUIrag 与 RAGFlow 的成熟实现，风险最低）；若追求全 Go，则按上表逐项替换，并先做 `go-fitz` 能力 PoC。

### 5.5 分块上限：按 BGE-M3 最大 token 换算字符数

- **模型上限**：BGE-M3 最大输入 **8,192 tokens**（XLM-RoBERTa tokenizer，含 `[CLS]`/`[SEP]`，可用正文 ≈ 8,190）。
- **换算为字符数**（sentencepiece tokenizer，`chars/token` 随语种变化，下表为经验值）：

| 语种/文本 | 经验 chars/token | 8,192 tokens ≈ 字符数 | 说明 |
| :--- | :--- | :--- | :--- |
| 中文为主 | ≈ 1.0–1.5 | **≈ 8,000–12,000 字符** | Han 字符多为 1 token |
| 英文为主 | ≈ 3.5–4.0 | **≈ 28,000–32,000 字符** | 常见词 ~4 chars/token |
| 中英混排 | ≈ 2.0 | **≈ 12,000–16,000 字符** | 按文档语言分布取值 |

- **本项目默认硬上限**：取中英混排保守值 **≈ 12,000 字符**（纯中文文档取 ≈ 8,000，纯英文取 ≈ 30,000）。
- **上限 = 超限拆分阈值**（不是目标块大小）：既然每个 bbox 块直接作为一个 chunk（§5.3），块大小由版面结构决定、通常远小于上限；8,192-token 换算值仅用于**单块超限时拆分**。个别异常大的版面块（如整页正文）拆到该阈值以内即可。注：块过长会稀释 embedding 语义、降低召回精度，必要时可对超大块主动降低阈值。
- **精确做法**：用 BGE-M3 自带 tokenizer **按 token 计数**（不要用字符估算）；字符换算仅用于无 tokenizer 的快速回退。
- **同步影响**：§5.3 的"超限拆分"与"超短块合并"阈值、检索 `top_k` 与上下文组装预算都应据此对齐。

### 5.6 解析缓存分层（策略已定，**Tier 1 未实现**）

> 记录时间：2026-09-29。下面大部分是**设计**而非实现，落地状态见每层标注。

**为什么值得做**：把一次 38 页论文的解析拆开实测——

| 阶段 | 38 页耗时 | 每页 | 占比 |
| :--- | ---: | ---: | ---: |
| 渲染（150 DPI） | 0.61s | 0.016s | 4.0% |
| **检测 + 取字** | **14.20s** | **0.374s** | **95.7%** |
| 合并 + 分块 | **0.022s** | 0.0006s | **0.15%** |

这组数字直接定了方案：**检测占 96%，而合并几乎免费**——所以合并永远全文档重跑，不必也不能缓存。

**三层结构**

```
Tier 0  文档内容 MD5    → 未改动整篇跳过                ✅ 已实现
Tier 1  逐页检测缓存     → 未改动的页跳过检测             ❌ 未实现
Tier 2  合并 + 分块      → 永远全文档重跑（0.022s/38页）   ✅ 已实现
```

**缓存单位必须是「页」，不能是 chunk**：检测逐页独立（`pipeline.py:101-108` 在页循环内调用），但合并**会跨页**（`chunking.py:197` `cur.page_num == prev.page_num + 1: return True`）。因此可缓存的只有「逐页检测结果」，最终 chunk 跨越页边界、无法按页缓存。

**缓存键：哈希渲染后的位图，不要图省事**

| 可选键 | 成本 | 正确性 |
| :--- | :--- | :--- |
| 路径 + mtime | ~0 | ❌ 已知会假命中（`cp -p` / rsync / 恢复备份保留时间戳） |
| 页面文本 + 几何 | ~1ms | ❌ **会漏失效**：文字相同但图片/图表变了，检测框就不同 |
| **渲染位图哈希** | **~16ms** | ✅ **可证明正确**：像素相同 + 推理确定性 ⟹ 检测结果相同 |

渲染只占 4%，用这点成本买「可证明正确」是划算的；而且命中时这笔渲染无论如何都省不掉（除非接受上面两种有洞的键）。

```
key = md5(位图)
    + 检测器身份   ← 模型文件哈希 + 执行 provider（CoreML / CPU / DML）
    + 检测参数     ← dpi, score_threshold, NMS_IOU, CROSS_CLASS_COVER_RATIO
    ✗ 不含 profile / max_chars / max_pages
```

包含 provider 的理由：实测 CoreML 与 CPU 在一页 9 个区域上**逐框一致**，但那是单页单次结果，不能当作跨版本保证——不同 provider 的浮点行为可以不同，而检测一旦不同缓存就是错的。

**排除分块参数是主要收益来源**：检测与分块规则无关，而现状 `cache.py:49-58` 把 `profile` / `max_chars` 和 `layout` 塞进同一个键、存的又是最终 chunk——等于改分块规则就把检测结果一起作废，方向正好反了。

**预期收益**

| 场景 | 现在 | 加 Tier 1 后 |
| :--- | ---: | ---: |
| 只改分块规则（38 页，检测全复用） | 14.2s | **0.6s** |
| 改 1 页 | 14.2s | **~1.0s** |
| 内容相同、路径不同 | 已跳过 | 不变（Tier 0 已挡） |

附带好处：键是内容寻址的，**同一页出现在不同文档里会自动命中**（同一张图、同一段模板页、同一篇论文的两个版本），跨文档去重不需要额外代码。

**落地时必须注意**

1. **确定性假设**：像素相同 ⟹ 检测相同，前提是推理确定。CoreML 在极少数图上可能有非确定性 kernel（**未实测**）。需要一条**自检降级路径**：命中的条目里保留检测结果摘要，抽样重算比对，发现不一致就不使用缓存。
2. **绝不缓存位图**：150 DPI 一页约 6.3MB，38 页就是 240MB。只存检测结果 + 位图哈希（每页几 KB）。
3. **打包后路径（会静默失效）**：`cache.py:23` 的默认目录是 `<repo>/cache/parse`，打包后通常不可写；而 `cache.py:104` 把写失败**静默吞掉**（`except OSError: return`）——表现是"功能看起来在、实际每次全量重跑"。落地前必须先由桌面端经 `FREERAG_CACHE_DIR` 下发 per-user 可写目录，并让写失败可告警。
4. **升级失效**：桌面端 auto-update 后，键里的模型哈希 + provider 会让旧条目自然失配，**不需要额外清理逻辑**；反之若键只含路径/mtime，升级后会拿旧模型的检测结果去跑新版本。

---

## 6. Agentic Loop 设计（初步对齐 RAGFlow medium 模式）

最新 RAGFlow（Go 版，`internal/rag/agentic-rag/`）把模式行为集中在一张表里（`runtime/config.go:91` `THINKING_MODES`），分 low / medium / high / ultra 四档。**本项目初期只做 medium 一档**，其余按需扩展。

### 6.1 medium 规格（`runtime/config.go:98-103`）

| 参数 | medium | 含义 |
| :--- | :--- | :--- |
| Agentic | true | 走 action session 工具循环 |
| EnableSCA | true | 开启充分性检查 |
| SCAMaxRounds | 3 | 最多 3 轮审查（枚举类问题内部 bound 到 2） |
| UseFanout | **false** | **无 planner、无 prefetch** |
| ActionMaxTurns | 8 | 单个 session 最多 8 轮工具调用 |
| SnippetsPerQuery | 6 | 单查询最多读 6 条命中 |
| Tools | **4 个** | `hybrid_search` / `grep_search` / `list_chunks` / `metadata_search`（本项目定制，见 §6.7） |

### 6.2 图结构（`BuildAgenticGraph`，`agentic_rag_graph.go:1490`）

```
START → formalize_question
formalize_question ─(medium: UseFanout=false)────────────→ rag_agent
planner → prefetch → rag_agent        ← 仅 high/ultra
rag_agent  [= ragAgentNode → draftNode → scaNode，计 3 次 node visit]
   ├─ SUFFICIENT ───────────────────────────────→ formalize_answer → END
   └─ INSUFFICIENT / 未定 → query_rewrite ──────→ rag_agent（下一轮）
stop → END                            ← visit 预算耗尽（记 graph 失败）
```

- medium **跳过 planner / prefetch**，`formalize_question` 直接进 `rag_agent`（`:1615-1620`）。
- `rag_agent` 一个节点内串起三件事：一轮 slot research pass → 生成 draft → SCA 审核（`:1580-1586`）。
- 循环由 SCA 判定 + **本轮证据增量**（`st.LastRoundNew`，`:586`）共同决定：不够且确有增长，才重写查询再来一轮。

### 6.3 与本项目 Laya 的对接点（关键差异）

RAGFlow 用**同一个生成模型**做路由与 SCA；本项目把这两处**无生成的决策**交给 Laya：

| RAGFlow 决策点 | 本项目替换 | 说明 |
| :--- | :--- | :--- |
| `routeSCA`（要不要再审 / 要不要再来一轮） | **Laya `choice`** | 非自回归、~33ms、无幻觉 |
| SCA 的 sufficient 判定 | **Laya `score`/`choice`** | 只判"够不够"，不生成文本 |
| `query_rewrite`（缺口 → 新查询） | 生成 LLM（Qwen3-4B） | 需要生成 |
| `draftNode`（中间 draft） | 生成 LLM | 需要生成 |
| action session 工具循环 | 生成 LLM + 工具 | 需要生成与工具协议 |

> **关键约束**：Laya 上下文仅 512 / 1024 tokens，无法像 RAGFlow 的生成式 SCA 那样读 48k 字符的 evidence。**故 SCA 输入只喂 draft**（见 §6.6），不做 evidence anchor 校验。

### 6.4 预算与守卫（照搬要点）

- **无整图 wall clock**：靠各节点超时（`PassTimeoutS` / `SCATimeoutS` / `RewriteTimeoutS`）+ `MinRoundHeadroomS` 路由守卫兜底（`:1506-1511`、`:538`）。
- **visit 预算**：`graphRecursionLimit(true, maxLoops)`，`maxLoops=3`（`:1760`）；一轮研究 = 3 次 visit。
- **枚举类问题**：`CoverageOf(slotTable).Ok()` 时把 SCA 轮数 bound 到 2（`:1642-1644`），避免对同一清单反复重写。
- **判决必须三分**：`SUFFICIENT` / `INSUFFICIENT` / `UNKNOWN`——"审核没跑成" ≠ "证据不足"，混同会误报 partial（`:57-64`）。

### 6.5 本项目初期简化

- **slot 表**：medium 仍由 `RunSlotResearchPass` 内部的 `initialize_state` 拆 slot（`st.SlotTable`，`:564`）。本项目可先**单 slot（整问一槽）**跑通循环，再启用多 slot 分解。
- **工具面固定为 4 个**：`hybrid_search` / `grep_search` / `list_chunks` / `metadata_search`（见 §6.7）。RAGFlow 的 `navigate_tree` / `navigate_structure` / `calculate` / `graph_explore` 需要编译出的目录树 / 知识图谱等额外结构，**不引入**。
- **不做 web_search**：纯本地场景，无须该工具，也就不存在"无 provider 时是否广告"的问题。

### 6.6 Laya 输入策略：只喂 draft

**决策**：充分性检查（SCA）**只把生成的 draft 喂给 Laya**，不喂原始片段、不喂各 slot 的 evidence。

- **输入** = `[question, draft]`。draft 由生成 LLM 产出（`draftNode`），本身已是"答案级"凝练，长度可控。
- **输出** = Laya 的类型化判决（`choice` / `score`）：`SUFFICIENT` / `INSUFFICIENT`（+ 置信度）。
- **超窗口兜底**：draft 若仍超 Laya 窗口（512 / 1024 tokens），按句截断到窗口内（保头 + 保尾），或先由生成 LLM 再压一层。

**为什么可行**：判"这份 draft 是否足以回答 question"是对 **draft 自身**的判断，不需要重回原始证据。RAGFlow 的生成式 SCA 兼做 groundedness 校验（故需 evidence anchor）；本项目把 groundedness 交给生成 LLM 在 draft 阶段负责，Laya 只管充分性这一个决策。

**代价与缓解**：
- Laya 无法核验 draft 是否被证据支持，存在"draft 看似完整实为幻觉 → 误判 SUFFICIENT"的风险。
- 缓解：draft 生成 prompt 强制带引用标记（`[n]`）；可选把**引用标记密度/覆盖度**作为弱特征一并喂给 Laya（仍是短输入），作为充分性的补充信号。

### 6.7 工具面：4 个检索工具（本项目定制）

初期固定 4 个工具，覆盖"语义召回 / 精确匹配 / 结构浏览 / 字段过滤"四种检索意图。

| 工具 | 用途 | 参数 | 返回 |
| :--- | :--- | :--- | :--- |
| `hybrid_search` | 语义 + 关键词混合召回，主力检索 | `query`, `k?` | top-k chunk（score / doc / page / block_type） |
| `grep_search` | 字面 / 正则精确匹配：术语、编号、专有名词、代号 | `pattern`, `regex?`, `k?` | 命中行及所属 chunk |
| `list_chunks` | 按 `doc_id`（+`page`）顺序枚举，用于浏览结构与回读原文 | `doc_id?`, `page?`, `offset?`, `limit?` | chunk 列表（分页） |
| `metadata_search` | 按 metadata 字段过滤（不依赖全文匹配） | `doc_id?`, `block_type?`, `page?`, `source_file?`, `limit?` | 匹配的 chunk 列表 |

设计要点：

- **互补而非重叠**：`hybrid_search` 管"意思像"，`grep_search` 管"字面有"，`metadata_search` 管"字段是"，`list_chunks` 管"这篇/这页有什么"。
- **`grep_search` 的独有价值**：BM25 / 向量对编号（`GB/T 1234`）、代号（`Qwen3-4B`）、罕见专名召回差，字面匹配能补上。
- **`list_chunks` 是安全网**：配合 §6.6"只喂 draft"，模型失去回看原文的能力；`list_chunks` 让它按页回读（对应 RAGFlow 的无损 `Kbinfos` 证据池）。
- **预算**：4 个工具的调用合计仍受 `ActionMaxTurns = 8` 约束（§6.1）。
- **职责边界**：工具只负责取证据；`draft` 由生成 LLM 产出、`verdict` 由 Laya 给（§6.3 / §6.6）。

---

## 7. 关键技术决策

| 决策点 | 结论 | 理由 |
| :--- | :--- | :--- |
| 是否用云原生 | **否** | 单机本地场景，云原生解决的是分布式/弹性问题，强行引入只增复杂度 |
| 后端语言 | **Go** | 吞吐量显著提升、常驻内存从 ≥500MB 降至 <80MB；但**不提升准确性**，端到端延迟受 GPU 瓶颈制约基本持平 |
| 解析层语言 | **Python Sidecar（推荐）** | 复用 PyMuPDF / PP-DocLayout / TSR 成熟实现；Go 全量重写需先 PoC `go-fitz` |
| 文本提取策略 | **PyMuPDF 文字层优先，OCR 兜底** | 电子版 PDF 免除 GPU OCR，是 8GB 约束下最大优化 |
| 表格解析 | **TSR 结构 + 文字层归位 + RAGFlow `construct_table`** | 保住行列/表头/跨格，避免按 y/x 排序丢失结构 |
| 分块策略 | **每个 PP-DocLayout bbox 块 = 一个 chunk（不合并）** | BGE-M3 窗口足够大，块级粒度保留完整语义与结构 |
| Agentic Loop | **对齐 RAGFlow medium**（无 planner/fanout，SCA ≤ 3 轮，session ≤ 8 轮） | 初期最小可用；high/ultra 后续按需 |
| 路由 / 充分性检查 | **Laya 类型化决策（不生成文本），SCA 只喂 draft（§6.6）** | 省一次大模型推理，~33ms、无幻觉；draft 长度可控，适配 Laya 小窗口 |
| 模型推理 | 外部独立进程（llama.cpp / Ollama） | 故障隔离，避免把推理编进 Go 二进制 |
| GPU 管理 | **动态模型切换**（加载/卸载） | 模型常驻总和易超 8GB，必须按序加载卸载 |
| 上下文长度 | 限制 **4K–8K tokens**（默认取 8192） | 8GB 显存下超 4096 tokens 即可能 OOM |

> **实测修正（2026-09-29）**：Ollama 的默认窗口是 **4096**，而 Agentic 循环第 2 轮的证据块实测会到 **7647 tokens**，
> 服务端直接返回 **HTTP 400**（不是截断），循环静默降级成抽取式 draft。因此：
> 1. `num_ctx` **必须显式下发**（`FREERAG_NUM_CTX`，默认 8192）；
> 2. prompt 侧同时要有**预算护栏**（`agent.PromptCharBudget`，按 3 字符/token 保守换算，超出则截断并声明省略条数），
>    否则换更大的模型只是把崩溃点往后推。
| 进程通信 | **JSON-RPC 2.0 over Stdio（NDJSON）** | 无需端口、安全、实现简单、开销低 |
| 全文存储 | **SQLite + FTS5**（单文件） | 纯 Go 驱动实测支持 FTS5，`bm25()` 内置；`sqlite-vec` 实测**不支持**（C 扩展） |
| 向量检索 | **Qdrant**（稠密 ANN；不可达时回退精确扫描） | 与 `coder/hnsw` 同数据同指标实测：后者 recall@10 仅 **0.215**，Qdrant **1.000**，单次 1.7 ms（暴力扫描 60 ms）。纯 Go 内嵌 ANN 未过闸门，外部引擎通过 |
| 批量策略 | **分批流水线**（每批 5–10 文件，阶段重叠） | 无法同时处理 100 篇，只能靠流水线提升吞吐 |

---

## 8. 性能预估

### 单篇解析（1MB ≈ 20 页）

**电子版 PDF（文字层可用，主路径）**：
- 版面检测：**实测 0.2–0.7s/页**（含渲染+letterbox+后处理+取字；CoreML）→ 20 页约 **4–14s**；纯 CPU 为 1.9–7.8s/页
- PyMuPDF 文字提取（CPU）：亚秒级
- TSR（仅含表格的页）：每表 < 数十 ms
- **合计 ≈ 秒级**（GPU 不参与）

**端到端实测（2026-09-29，38 页 arXiv 论文 / 203 chunks）**：

| 阶段 | 实测 | 备注 |
| :--- | :--- | :--- |
| parse（38 页 → 287 块 → 203 chunks） | **21.5s** | 含 PP-DocLayout 推理 |
| index（再解析 + 嵌入 + 落盘） | **22–26s** | 嵌入 203 chunks 约 +4s（托管接口） |
| `hybrid_search` | **0.15–0.17s** | 含一次网络嵌入调用 |
| `grep_search` / `list_chunks` / `metadata_search` | **0.01–0.03s** | 内存扫描，**不可扩展**（见 §0.1 P0-1 与 §0.3.2） |
| Agentic 问答（1 轮） | **15.4s**（修正前 113s） | 见下方修正：主因是推理 token，不是轮数 |
| Agentic 问答（2 轮） | **128–280s**（未复测） | 轮次翻倍则耗时翻倍，**轮数是延迟的次变量** |
| 索引文件 | **1344 KB** | 无向量时 257 KB（向量占 ~1.1 MB） |

> **结论修正（2026-09-29 晚，推翻上面那句"完全由轮数决定"）**：逐段实测后，
> **单轮内最大的成本是 Qwen3 的推理（thinking）token** —— 它进的是 `message.thinking`，
> 这份流水线里没有任何东西读它，却和答案共用 `num_predict` 预算（Apple M5 / 4B Q4_K_M / 全 GPU）：
>
> | 配置 | 生成 token | 单次调用 |
> | :--- | ---: | ---: |
> | 推理开（原行为） | 246 | 29.7s |
> | **`think: false`** | **14** | **1.9s** |
> | 推理开 + `num_predict=96` | 96 | 13.2s，**答案 0 字符** |
>
> 而**模型加载只有 0.5s**、prompt 处理 465 tok/s —— 所以"启动时预加载模型"几乎不解决问题。
> 关闭推理后端到端 **113s → 15.4s**。已实现：`FREERAG_THINK`（默认关）、
> 后台预热（复用 tool-planning 前缀以命中 Ollama 的提示词缓存）、`FREERAG_KEEP_ALIVE=30m`。
> **优化方向变为两条并行**：减少无用的推理 token（已做），**以及**减少轮数（未做）。
>
> 另：第三行同时是一个**正确性**问题，不只是速度 —— 预算被推理吃光会返回空答案。

**扫描件（无文字层，兜底路径）**：
- 版面分析（CPU）：≈ 0.3s
- OCR（GPU）：≈ 60s（3s/页）
- 模型动态切换开销：数秒～数十秒
- **合计 ≈ 45–60s**

### 批量 100 篇（约 2000 页）
- **电子版为主时**：主要成本在 CPU 解析与入库，可接近线性并发（受 RAM 限制）
- 扫描件为主时：串行 OCR 约 100 分钟；优化后（≈5s/页 + 批处理 + 流水线重叠）**约 2.5–3.5 小时**

### 并行度
| 策略 | 并行数 | 风险 |
| :--- | :--- | :--- |
| 保守（推荐） | **2** | 稳定，`OLLAMA_NUM_PARALLEL=2` 起点 |
| 激进 | **4** | 需 INT4 量化 + 精细 `--gpu-memory-utilization`，易 OOM |
| 极限（不推荐） | > 4 | 极大概率 OOM |

> 注：若解析以电子版为主（无 GPU OCR），解析阶段的并行主要由 RAM 与 CPU 核数决定，可高于表格中的 OCR 并行上限。

### Go 化预期收益
| 维度 | 效果 |
| :--- | :--- |
| 吞吐量 | ⬆️ 显著提升（批量总时间缩短） |
| 端到端延迟 | ➡️ 基本持平（瓶颈在 GPU） |
| 内存占用 | ⬇️ 大幅降低（≥500MB → <80MB） |
| 显存占用 | ➡️ 不变 |
| 准确性 | ➡️ 无直接影响 |
| 开发效率 | ⚠️ 可能下降（Go AI 生态弱于 Python） |

---

## 9. 分阶段实施计划

### Phase 0 — 技术可行性验证（PoC）
- [x] 验证 Go 侧 JSON-RPC over Stdio 通信骨架。→ `internal/ipc`，已验收。
- [x] ~~验证 `sqlite-vec` 在 8GB RAM 下的检索延迟~~ → **已作废并得出结论**：纯 Go 驱动加载不了 C 扩展（实测 `vec_version()`/`vec0` 均失败）。改用 **`coder/hnsw`** 做向量索引、**SQLite FTS5** 做全文，实测数据见 §4。
- [x] **用真实嵌入验 HNSW 召回率** → **已执行，`coder/hnsw` 未通过**。真实 BGE-M3 向量（203 个）+ 真实问句向量（20 个）实测：
      `recall@1` 0.30–0.90、**`recall@10` 各参数下均 ≤ 0.43**；对照实验（图内向量查自己）默认参数只命中 **146/203**。
      结论：**不采纳 `coder/hnsw`**。
- [x] ~~据此「不引入 ANN」~~ → **该结论已于 2026-09-29 被推翻并修正**：用**完全相同的数据与指标**复测 **Qdrant**，
      `recall@10` **1.000**、recall@1 **1.000**、单次 1.7 ms。**问题出在纯 Go 内嵌库的实现缺陷，不是 ANN 本身。**
      已接入 `internal/dense`（Qdrant），精确扫描保留为回退与降级路径。详见 §4。
- ~~[ ] 验证 `llama.cpp` 的 Go 绑定（`yzma` / `gollama.cpp`）能加载 Laya GGUF~~ → **已作废**：推理改为 Ollama HTTP（§7），不引入 Go 绑定。
- ~~[ ] PoC `go-fitz` 的 word-level bbox 提取能力~~ → **已作废**：解析层已定为 Python Sidecar（§7），并已实现。
- ~~[ ] 小规模验证 AGGO / agenticgokit~~ → **已作废**：Agentic Loop 已自研（§6）。

### Phase 1 — 解析层（Parser）
- [x] 集成 **PP-DocLayout ONNX**（`layout.onnx`）做版面分析：已实现并实测（`sidecar/layout_onnx.py`，加速器优先）；可视化见 `scripts/visualize_layout.py`。
- [x] 实现 **PyMuPDF 文字层优先**的按块内容提取（Title/Text/Equation），OCR 仅兜底。→ `layout.py::text_in_bbox`；表格走 `find_tables()` 转 Markdown。
- [x] 实现**解析流水线**（§5.1 的 ①→⑥）。→ `sidecar/pipeline.py`，38 页真实论文验收。
- [x] 实现**分块**：每个 PP-DocLayout bbox 块直接作为一个 chunk（策略 A，不合并）。→ `sidecar/chunking.py`。
- [x] 实现**超短块合并/挂靠**（§5.3.1）与噪音页眉页脚过滤、超 BGE-M3 窗口的按句拆分。→ 已实现；`Caption`/`Reference` 已加入豁免类型。
- [ ] **表格解析**：集成 TSR（`tsr.onnx`）+ 移植 RAGFlow `construct_table`（行列排序、孤立行列剔除、表头判定、colspan/rowspan、HTML/描述文本双输出）。→ **已完成（2026-09-29）**：`sidecar/tsr_onnx.py` + `sidecar/table_grid.py`；表格转 Markdown **1/10 → 9/10**。注：输出目前只走 Markdown 一份，HTML/描述文本未做。
- [~] 实现**双栏处理**与 **layout/chunk 缓存**。→ **缓存已完成**（`sidecar/cache.py`，实测 21.5s → 0.18s）；**双栏仍未显式处理**，见 §0.1 P1-4。
- [ ] 集成 GLM-OCR（GPU，动态加载）作为扫描件兜底；评估 INT4 量化版。→ 见 §0.1 P1-3；**扫描件目前完全不可用**。
- [ ] 实现模型**动态加载/卸载**调度。

### Phase 2 — Go 内核
- [x] 工具面（4 个，见 §6.7）：`hybrid_search` / `grep_search` / `list_chunks` / `metadata_search`。→ 全部实现并有测试。
- [x] 向量检索接口封装。→ **BM25 + BGE-M3 向量 + RRF 融合**（`internal/store`）。注意：表格目前只入库 Markdown 一份，**"HTML + 描述文本双份索引"未做**（依赖 TSR）。
- [x] 上下文组装（Prompt 准备，控制 4K–8K tokens）。→ 显式 `num_ctx` + `PromptCharBudget` 字符预算护栏。
- [~] Agentic Loop 编排（详见 §6）：`formalize_question → rag_agent(session → draft → sca) ⇄ query_rewrite`。→ 循环已实现并验收；**SCA 已换成真实 Laya**（`sidecar/laya.py`，实测第 3 题 2 轮 280s → 1 轮 76s）。**工具选择已改为模型驱动**（2026-09-29）：模型在 4 个工具里自选，未知工具名被剔除并回退确定性计划（`internal/agent/loop.go` 的 `planTools`）。
- [ ] 文件调度与批量解析队列（分批 + 阶段重叠）。→ 见 §0.1 P2-7。

### Phase 3 — 本地推理服务 + 模型自动下载
- [x] 推理服务作为 Sidecar 独立进程（llama.cpp / Ollama）。→ Ollama 跑 Qwen3-4B，已接入。
- [~] 从镜像源（如 `hf-mirror.com`）自动下载 GGUF / ONNX 模型。→ `scripts/download-models.sh` 可用，但**是手动脚本，不是安装流程**。→ **安装器已完成**（见 §0.2），但**仍未随包集成模型下载**。已下载 4.24 GB；嵌入模型**未下载**（走托管接口）。
- [~] Go 内核维护**模型注册表（Manifest）**：名称、路径、哈希；缺失/损坏时触发重下。→ `models/manifest.json` 由脚本生成，但**内核不读取、不校验、不重下**。
- [ ] 安装程序（NSIS / Inno Setup / Electron Builder）承担模型下载与初始化。→ **未开始**。
- [x] 减少磁盘重复：同一 GGUF 存了两份。→ 已用硬链接去重（Ollama 的 blob 是内容寻址的，链接对其不可见），实测 **7.1 GB → 4.4 GB**。

### Phase 4 — Electron UI 层
- [x] 与 Go 内核建立类型安全的 IPC（JSON-RPC over Stdio）。→ `desktop/main.js` + `preload.js`。
- [~] UI 渲染、交互、托盘/菜单等系统原生功能。→ 目前只是 **RPC 调试台**（方法下拉 + 参数框 + 响应面板），→ **已完成（2026-09-29）**：文档列表、拖放/选择文件、引用跳转、实时进度。

### Phase 5 — 性能调优与稳定性
- [ ] 并行度调优（从 2 起步，评估是否到 4；电子版为主时可放宽）。
- [ ] 减少 GPU 模型切换开销。
- [ ] 在显存限制内最大化 OCR 批处理效率。
- [ ] 监控 `nvidia-smi`，建立 OOM 回退机制。
- [ ] **跨平台加速器实测**：CoreML 实测 0.17s/页 vs 纯 CPU 1.9–7.8s/页，而 CoreML 仅 Apple；Windows 需 DirectML、Linux 需 CUDA/OpenVINO，且 pip wheel 不同（§0.1 P2-6）。

### Phase 6 — 客户端/开发端划分与版本分发
> 对话在「采用以下目录结构：」处中断，以下为**待补充**：
- [ ] 代码仓库目录结构划分（参考 SiYuan / Reasonix 做法）。
- [ ] 开发端（核心代码与构建）与客户端（分发与运行）的职责边界。
- [ ] 自动化构建与版本分发机制。
- [ ] 应用内更新（auto-update）方案。

---

**图例**：`[x]` 已完成并验收 · `[~]` 部分完成（说明见行尾）· `[ ]` 未开始 · `~~删除线~~` 已作废。

**实施中新发现、原计划没有的工作**（详见 §0.1）：混合检索（BM25+向量+RRF，**已完成**）、
解析缓存（**已完成**）、模型存储去重（**已完成**）、Laya 接入 SCA（**已完成**）、
`layout_provider` 可观测性（**已完成**）、页眉页脚过滤（**已完成**，但对该 PDF 零效果）、
模型驱动的工具选择（**已完成**）、增量索引（**已完成**）。
存储层落地：**密集一半已完成**（Qdrant），关键词与落盘的部分见 §0.3.2。
本地嵌入 provider 已由决策取消（§0.3.1）。

---

## 10. 风险与对策

| 风险 | 影响 | 对策 |
| :--- | :--- | :--- |
| 显存 OOM | 应用崩溃 | 动态加载、上下文限 4K–8K、INT4 量化、OCR 仅兜底、并行数保守取 2 |
| 模型下载失败/损坏 | 安装即用目标破功 | Manifest + 哈希校验 + 断点重下 + 镜像源 |
| Go AI 生态短板 | 开发效率下降 | 解析层保留 Python Sidecar（PyMuPDF/PP-DocLayout/TSR），Go 只做编排 |
| `go-fitz` 能力不足 | 全 Go 方案受阻 | 先 PoC；不可行则退化为 Python Sidecar |
| TSR 漏检 / 单元格串列 | 表格还原错误 | **已部分缓解**：TSR 给出结构后表格转 Markdown 达 9/10。**未做**：RAGFlow 的孤立行列剔除、跨页合并、colspan/rowspan 还原；实测残留问题是**单元格文字串列**（`[169]` 被切到相邻列）与**无表头时表头行重复**——见 §0.2 的 TSR 行。结构不足 2×2 时回退纯文本 |
| 推理进程崩溃 | 影响 UI | 解析与推理各为独立 Sidecar，故障隔离 |
| 批量耗时过长 | 用户体验 | 分批流水线 + 阶段重叠 + 缓存（layout/chunk）+ 批处理调优 |
| **嵌入走托管接口（已决定长期如此）** | **① 文档分块原文发往第三方（隐私）② 需联网、按量计费、有速率限制 ③ 与 §1「完全本地化」冲突——该目标已因此修正** | **已接受（§0.3.1）**。缓解：`Embedder` 接口保留、切换成本压到一个文件；嵌入失败不阻断索引（降级为纯关键词）；`version` 如实上报。**私有文档场景不适用，须在文档与 UI 中明示** |
| **向量存 JSON，随语料线性膨胀** | 实测 200k chunks：内存 1343MB（向量本身 819MB）、**Save 10.3s/次**（O(全库)，加一篇文档就重写整个索引） | **密集已改由 Qdrant 承担**（§4，`recall@10` 实测 1.000），Go 侧副本可去（§0.1 P2-10）。关键词索引与增量落盘的取舍见 §0.3.2 |
| **ANN 会静默漏掉最近的邻居** | 实测 `coder/hnsw` 在真实 BGE-M3 向量上 `recall@10` ≤ 0.43；M=16 时图内向量查自己只命中 146/203。漏掉的邻居没有任何下游信号能发现 | **已按此闸门换库**：同数据同指标下 **Qdrant `recall@10` = 1.000**（§4）。准入标准保持"真实嵌入上 `recall@10 ≥ 0.95`"，**每次换库都要重测，不采信速度指标** |
| **纯 Go 向量索引生态不成熟** | 实测 `chromem-go` 并发写入丢数据（5/200000）；`coder/hnsw` 图质量不达标 | 改用**外部引擎 Qdrant**（原生二进制，无需 Docker/JVM）。**结论修正：问题在纯 Go 内嵌库的实现，不在 ANN 本身**（§4） |

---

## 11. 待确认问题

1. **目标文档类型**（扫描件 / 电子版 PDF / 表格密集型）决定主路径与兜底路径的占比，进而决定并行度与耗时。
2. **查询复杂度**（单跳 vs 多跳）影响 Laya 决策链设计。
3. **客户端/开发端划分 + 版本分发**方案（对话未完成，需补充）。
4. ~~Agentic Loop 走自研还是 AGGO / agenticgokit~~ → **已定：自研**（对齐 RAGFlow medium，见 §6），已实现并验收。
5. ~~解析层最终形态：Python Sidecar 还是全 Go~~ → **已定：Python Sidecar**（§7），已实现。
6. TSR 输出默认走 **HTML** 还是 **描述性文本**（或双份入库）。→ 仍未决；当前表格只入 Markdown 一份，待 TSR 接入时一并确定。
7. ~~嵌入模型最终形态~~ → **已定（2026-09-29）：长期使用 SiliconFlow 托管 API**，不实现本地 provider。代价与缓解见 §0.3.1。
8. ~~向量存储最终形态~~ → **已定（实测）**：密集检索交 **Qdrant**（`recall@10` = 1.000）。**原"不采纳 ANN"的结论已作废**——问题在纯 Go 库的实现，不在 ANN 本身（§4）。关键词侧待定，见 §0.3.2。
9. **目标语料规模**：几十上百篇（≤20k chunks）还是上千篇（200k+）？这决定要不要把关键词索引也下沉——实测 20k 时手写层（查询 46ms、内存 124MB、Save 664ms）够用，引入第二个存储是净风险。**仍待确认，且这是 §0.3.2 的唯一前置**。
10. ~~HNSW 召回率~~ → **已测**：`coder/hnsw` 未达标（≤0.43），**Qdrant 达标（1.000）并已采纳**。见 §4。

---

## 12. 一句话结论

用 **Electron 壳层 + Go 业务内核 + 解析/推理 Sidecar + 单文件存储（SQLite FTS5 全文 + 精确向量扫描）** 的本地优先架构；
解析采用 **PP-DocLayout + PyMuPDF 文字层优先 + TSR 表格结构识别（借鉴 RAGFlow）** 的串行管线，
**OCR 仅在扫描件时兜底**；Agentic Loop 初步对齐 **RAGFlow medium**（Laya 做路由与充分性检查、LLM 做生成）；GPU 只常驻 Laya、按需动态加载 LLM，
在 8GB 内存/显存内实现完整 Agentic RAG；**无需云原生**，精力应集中在 Go 侧并发调度、显存动态控制与稳定的自动模型下载机制。

> **实施后的结论修正（2026-09-29）**
>
> 架构判断成立：Go 内核 + Python 解析 Sidecar + Ollama 推理的组合已经在 38 页真实论文上跑通解析到问答的完整链路。
> 三点修正：
> 1. **延时不来自解析或检索，全部来自生成 LLM 与 SCA 轮数**。解析 21.5s、检索 0.15s，而问答 50–280s，
>    且随轮数线性增长。优化重心应放在**减少轮数**（更准的 SCA + 更准的首轮检索），而不是解析/检索。
> 2. **检索必须做混合**。实测中文查询打英文论文：纯 BM25 **0 命中**，BM25+向量+RRF **5 命中**。
>    单靠关键词在跨语言与非词面匹配场景下不可用。
> 3. ~~**"完全本地化"还差三件事**：本地嵌入、SQLite 落盘、Laya 接入~~ → **两条已不成立，修正如下**：
>    Laya 已接入（§0.2）；**本地嵌入已由决策取消、改为长期托管**（§0.3.1），故"完全本地"这一目标不再成立；
>    真正剩下的只有存储层能否撑住语料规模（§0.3.2 / §0.1 P0-1）。
