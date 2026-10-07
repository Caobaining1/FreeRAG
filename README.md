# freerag

本地优先（Local-First）的桌面端 Agentic RAG 系统。全部模型在安装过程中下载到本地，用户无需手动安装任何东西。

- **桌面端**：Electron（UI 壳层）
- **内核**：Go（业务逻辑、编排、并发）
- **解析**：Python Sidecar（PyMuPDF；PP-DocLayout / TSR 为后续接入点）
- **推理**：llama.cpp Sidecar（Laya 路由/充分性检查 + 生成 LLM + BGE-M3 嵌入）— 未接入
- **存储**：内存分块索引 + 手写 BM25 / 余弦 / RRF 融合，落盘为**单个 JSON 文件**。
  **没有数据库**：SQLite + sqlite-vec 只是 §3 的目标形态，尚未实现
- **编排**：顶层问答流程用 **eino**（`compose.Graph`）——见「问答编排（eino）」。
  这是内核目前**唯一的第三方依赖**，构建需走镜像（见下方「依赖」）

> 完整设计见 [`docs/plan.md`](docs/plan.md)。

## 状态

| 模块 | 状态 | 说明 |
| :--- | :--- | :--- |
| Go 内核 + JSON-RPC 2.0 over stdio | ✅ 已验收 | `internal/ipc` |
| 解析 Sidecar（PyMuPDF 管线） | ✅ 已验收 | 版面块检测、表格转 Markdown、按块分块 |
| 版面分析（PP-DocLayout ONNX） | ✅ 已接入 | `sidecar/layout_onnx.py`，加速器优先；可视化 `scripts/visualize_layout.py` |
| 分块合并（§5.3 / §5.3.1） | ✅ 已验收 | TUIrag 策略 B：标题归组 Section → 段落缓冲 → 表/图合并 Caption → 碎片丢弃 |
| Go ↔ Python Sidecar 桥接 | ✅ 已验收 | `internal/sidecar` + `internal/parser` |
| 分块索引与 **混合检索** | ✅ 已验收 | `internal/store`：BM25（CJK 二元切分）+ 密集 ANN（Qdrant）+ RRF 融合 |
| 密集检索（BGE-M3） | ✅ 已接入 | `internal/embed`，OpenAI 兼容接口；无 key 时自动降级为纯关键词 |
| **密集 ANN 索引（Qdrant）** | ✅ 已接入 | `internal/dense`；实测 recall@10 = **1.000**；不可达时响亮告警并回退精确扫描 |
| Agentic Loop（medium 模式，§6） | ✅ 已验收 | `internal/agent`；模型可插拔，无模型时降级 |
| **SCA 用 Laya（§6.6）** | ✅ 已验收 | `sidecar/laya.py` + `internal/agent/checker.go`；只喂 draft，~104 ms/次 |
| **解析缓存** | ✅ 已验收 | `sidecar/cache.py`；重复解析 21.5s → 0.18s |
| **模型存储去重** | ✅ 已验收 | `scripts/setup-ollama.sh` 硬链接；7.1 GB → 4.4 GB |
| 检索工具面（§6.7） | ✅ 已验收 | `hybrid_search` / `grep_search` / `list_chunks` / `metadata_search` |
| **每轮工具选择（Laya）** | ✅ 已接入 | 每轮把「问题 + 证据摘要 + 已试调用 + missing」发给 Laya，它在**确定性候选调用集**里选下一个（~100 ms）；Laya 不可用时回退模型规划 → 确定性计划。见「问答编排（eino）」 |
| **REFRAG 式证据压缩** | ✅ 已接入（**默认关闭**） | 只展开排名最前的段落，其余以 gist 进入提示词：受控 A/B 实测 **TTFT 4.01×**（25.8s → 6.4s），池越深压缩比越高（池 20 → 3.50×，且 20/20 段全保留）。见「证据压缩（REFRAG 式）」 |
| **问答编排（eino）** | ✅ 已接入 | `route`（Laya 判简单/复杂）→ 简单：单次 agentic pass；复杂：`decompose` → 并发 `fanout` → `synthesize` |
| **Electron 桌面端（产品形态）** | ✅ 可用 | `desktop/`：文档列表 + 拖放/选择文件 + 提问 + 引用跳转 + 实时进度 |
| **进度通知（JSON-RPC notification）** | ✅ 已接入 | `internal/ipc` 的 `Server.Notify`；`index` / `ask` 逐阶段上报 |
| **文档管理 RPC** | ✅ 已接入 | `documents` / `forget` / `status` |
| **分块检查器（原文 ↔ chunk 对应）** | ✅ 已接入 | `chunks` / `page`：左侧渲染原文某页并画 bbox 高亮，右侧列分块，两边点击互相定位。见「桌面端界面」 |
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
│   ├── chunking.py        # 分块合并（TUIrag 策略 B）
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
export GOPROXY=https://goproxy.cn,direct   # eino 依赖需走镜像，见「依赖」
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

**凡是作用于索引的方法都接受一个可选 `kb`**（知识库 id）。**省略即第一个知识库** —— 这既让改造前的脚本与 CLI 继续可用，也意味着"忘了传"的后果是**答案变窄**，而不是串到别的知识库去。每个此类响应都会回带 `kb`，所以调用方看得出实际作用于哪一个。

| 方法 | 参数 | 说明 |
| :--- | :--- | :--- |
| `ping` | — | 存活检查 |
| `version` | — | 内核版本、数据目录、知识库数量、Sidecar / 嵌入 / 生成模型配置。**不含索引数字**（那些属于某个知识库，见 `status`） |
| `parse` | `path`, `profile?`, `max_chars?`, `max_pages?` | 解析文档并返回 chunk（不入库，不涉及知识库） |
| `index` | 同上 + `force?`, `kb?` | 解析并写入该知识库；**内容未变的文档自动跳过**（见「增量索引」），返回 `skipped` / `added` / `removed` / `indexed_total` |
| `search` | `query`, `limit?`, `kb?` | BM25 关键词检索 |
| `ask` | `question`, `kb?` | 运行 medium 模式 Agentic Loop，返回 draft / verdict / evidence / trace |
| `tools` | — | 返回当前模式的检索工具面（4 个工具的 JSON schema） |
| `tool` | `name`, `arguments?`, `kb?` | 直接执行一次工具调用，不经过模型 |
| `documents` | `kb?` | 列出该知识库的已索引文档（md5 / 文件名 / 页数 / chunk 数 / 索引时间） |
| `chunks` | `doc_id`, `kb?`, `page?`, `offset?`, `limit?` | 某文档的分块：chunk_id / 页码 / 块类型 / **bbox（页内 PDF 点坐标）** / 字数 / 文本；外加**每页块数**（供翻页箭头）与总数。`page` 只过滤列表，不过滤计数 |
| `page` | `doc_id`, `page`, `kb?`, `dpi?` | 原文某页渲染成 PNG（base64）+ **页尺寸（pt）**。bbox 就是按 pt 度量的，前端拿它换算高亮框 |
| `forget` | `md5` 或 `doc_id`, `kb?` | 移除一篇文档的 chunks、清单与向量 |
| `status` | `kb?` | 各子系统健康状态（sidecar / 嵌入 / ANN / 生成模型）；索引部分属于指定的知识库 |
| `kb.list` | — | 列出知识库（id / 名称 / 文档数 / chunk 数）。**不打开任何一个** |
| `kb.create` | `name` | 新建知识库（只写注册表，空库不产生索引） |
| `kb.rename` | `id`, `name` | 改名。**id 不变** —— 索引目录与 Qdrant 集合都以 id 命名，改名不该搬动数据 |
| `kb.delete` | `id` | 删除知识库：登记 + 向量 + 索引目录。**最后一个不可删** |

**知识库的隔离是结构性的**：每个库有自己的 `index.json` 和自己的 Qdrant 集合（`freerag_<id>`）。共用一个集合需要每次查询都带 payload 过滤，而**第一个忘记带的调用点**就会返回另一个库的段落 —— 那种泄露的结果看起来仍然合理，下游没有任何东西能发现。

数据布局（`FREERAG_DATA` 覆盖目录，`FREERAG_KB_REGISTRY` / `FREERAG_KB_ROOT` 可单独覆盖）：

```
data/
  kbs.json              注册表（列表 + 缓存的文档数）
  kbs/<id>/index.json   每个知识库自己的索引
  index.json            改造前的单索引文件，仅作为迁移来源保留
```

**升级不会看起来像丢了数据**：首次启动时，若注册表不存在而旧的 `index.json` 存在，它会被**复制**（不是移动）成第一个知识库「默认知识库」。原文件保留在原地 —— 万一迁移的判断有误，还有东西可以回退。

**进度通知**：`index` 与 `ask` 会先发若干条**无 `id` 的 JSON-RPC 通知**，同一条流上，然后才是响应：

```json
{"jsonrpc":"2.0","method":"progress","params":{"stage":"parse","file":"paper.pdf"}}
{"jsonrpc":"2.0","method":"progress","params":{"stage":"draft","delta":"这门课的"}}
{"jsonrpc":"2.0","id":1,"result":{"added":203,"indexed_total":203}}
```

`stage` 取值：`hash` / `skipped` / `parse` / `parsed` / `stored` / `persisted` / `agent` / `thinking` / `draft`。
**任何客户端都必须按 `id` 是否为 null 区分**：没有 `id` 的是事件，不是"忘了对应的响应"。
（`index` 约 30 s、`ask` 可达 280 s —— 没有这个通道，界面就只有一个永不变化的转圈。）

三个 stage 属于**某一次提问**，界面把它们渲染在那条消息里，而不是全局状态条上：

- `thinking` —— 在第一次模型调用**之前**发出，那是最长的静默段（60–70 s）。它同时喂给外壳的闲置超时（内核来的任何消息都会重置该时钟）。
- `agent` —— 每一步的工具调用与结果，`params.line` 是可直接显示的一行。
- `draft` —— **答案本身**的分片。界面在流结束后**整体替换**为带引用链接的版本：分片边界可能落在一个引用标记或标签中间，所以流式期间只能按纯文本写入。

其余的 stage 都属于**索引流程**，没有对话可归属，仍然进全局状态条。

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
| `FREERAG_GENERATION_TEMPERATURE` | `0.2` | 采样温度。**`0` 就是贪心**（会真的下发）；负值才表示"用 Ollama 的默认"。评测必须设 `0`，否则同题两次会改写不同的查询 |
| `FREERAG_NUM_CTX` | `8192` | 上下文窗口。**必须显式设置**：Ollama 默认 4096，小于 Agentic 循环拼出的证据块，超长 prompt 会被拒绝（HTTP 400）而不是截断 |
| `FREERAG_THINK` | 空（**关闭**） | Qwen3 的推理模式。见下方「推理模式为什么默认关闭」 |
| `FREERAG_KEEP_ALIVE` | `30m` | 模型在显存中的保留时长。Ollama 自己的默认是 5 分钟，短于一个人读完答案再问下一个的间隔 |
| `FREERAG_VLM_MODEL` | 空（**关闭**） | **解析期**视觉模型名（如 `qwen2.5vl:3b`，**必须非推理**）：给 `Figure` 区域生成文字描述。空 = 不做。见下方「解析期视觉（图表 → 文字）」 |
| `FREERAG_LAYOUT_THREADS` | `1` | 版面模型的 CPU 线程数（增多会更慢） |
| `FREERAG_LAYA_DIR` | `models/laya-onnx` | Laya ONNX 目录 |
| `FREERAG_LAYA_THREADS` | `1` | Laya 的 CPU 线程数。**实测 1 线程 104ms vs CoreML 434ms**，与版面模型相反 |
| `FREERAG_CACHE_DIR` | `cache/parse` | 解析缓存目录 |

## 解析期视觉（图表 → 文字）

`Figure` 区域**没有文字层**，所以它本来只能靠 caption 被检索到。设 `FREERAG_VLM_MODEL=qwen2.5vl:3b`
后，解析时会把每个 Figure 按 bbox 裁剪成 PNG 交给该视觉模型，把描述写进块正文，于是图也能像
普通段落一样被检索（实现见 `sidecar/vlm.py`）。

### ⚠️ 必须用**非推理**视觉模型

实测（Ollama 0.34.4，同一张流程图裁剪，`temperature=0`）：

| 模型 | dpi | 耗时 | `thinking` | 描述正文 |
| :--- | ---: | ---: | ---: | :--- |
| **`qwen2.5vl:3b`** | 150 | **25.7s** | 0 字 | 104 字 ✅ |
| **`qwen2.5vl:3b`** | 100 | **21.8s** | 0 字 | 85 字 ✅ |
| `qwen3-vl:4b` | 100 | 55.6s | 466 字 | **0 字（空）** ❌ |
| `qwen3-vl:4b`（`bench_vlm.py` 全量 11 张） | — | **213s/张** | 大量 | 慢且不稳定 |

**`qwen3-vl:4b` 不可用**：Ollama 忽略 `"think": false`，它永远推理；推理与正文共用
`num_predict`，预算被推理吃光后**正文为空** —— 与生成模型踩过的坑完全同一个
（见「问答延迟」一节）。`sidecar/vlm.py` 因此会在收到「只有 reasoning、没有 content」时
**打 warning**，而不是静默写一个空 chunk。

> 每张图约 22s、每篇最多 12 张，所以一篇带 7 张图的论文首次索引约 **+2.5 分钟**；
> 结果进解析缓存（缓存键含模型名），重复索引为 0。

**已知残留（实测）**：提示词要求「不要引用任何具体数值」，但 `qwen2.5vl:3b` 仍会写
（实测一条描述里出现 `κ = 0`、`H(x) = 0.52`、`A3`）。更大的 `qwen3-vl:4b` 指令遵循更好，
但它在本机不可用（见上表）。目前**不做事后过滤**：数字清洗的边界（`A3` 是分类标签还是数值？
`Figure 1` 呢？）不适合用正则一刀切，宁可在描述前保留原样式。真要保证「索引里没有数值」，
正确做法是加一道可配置的后处理，而不是指望模型遵守。

**只在解析阶段用，问答阶段绝不用。** 生成模型与一个 ~3.5 GB 的视觉模型不能同时驻留——本机
16 GB 且已实测在换页（§「问答编排」）。因此：

| 行为 | 说明 |
| :--- | :--- |
| **索引 ⟂ 问答 互斥** | 每个知识库一把 `RWMutex`：索引独占（`index` / `index_batch` 全程），问答共享（可多个并发）。索引进行中发起的提问会**等待**，反之亦然 |
| **索引任务结束即卸载** | `index` 完成、或 `index_batch` 整个作业结束后，内核用 `keep_alive: 0` 让 Ollama 立刻卸载该视觉模型（`agent.UnloadModel`） |
| 失败不致命 | 裁剪失败 / 调用失败 / 超时的**单个图**保持原样，不影响整篇解析；卸载失败只记日志（模型会按自己的 keep-alive 过期） |
| 每个文档最多 12 张图 | `MIN_FIGURE_POINTS`/`DEFAULT_MAX_FIGURES`，避免一个异常文件把解析变成上百次模型调用 |

> ⚠️ 改了 `vlm.py` 的提示词或裁剪逻辑**必须 bump `sidecar/cache.py` 的 `CACHE_VERSION`**，
> 并且注意视觉模型名已纳入缓存键——换模型会产生不同的 caption，缓存不能跨模型命中。
> 既有的索引里 Figure 块没有描述，需要 `force: true` 重新索引才能补上。

密集检索（可选，见下方"嵌入模型"）：

| 环境变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `FREERAG_EMBED_PROVIDER` | 空（关闭） | `siliconflow` 走托管 BGE-M3；空 = 纯关键词检索 |
| `FREERAG_SILICONFLOW_KEY` | — | `siliconflow` 时必填 |

### 统计图表专用后端（Laya-Chart 1.825B）

通用视觉模型被明确要求**不要写数值**（`sidecar/vlm.py` 的提示词），理由见上一节：读错的数字进索引后
永远没人看得见。这对流程图是对的，对柱状图是错的——数值被抹掉后，「2020 年营收是多少」无从命中。

`FREERAG_VLM=chart` 换成图表专用模型：Laya-Chart 把图转译成 **Markdown 数据表 + 中英双语分析**，
实测数字幻觉率 NHR **0.0095**（写出的数字里每千个有 9.5 个无法由它自己刚写的表推出）。数字策略因此
**按部分相反**：

| 产出 | 数值 | 原因 |
| :--- | :--- | :--- |
| 转译出的**表格** | ✅ 保留 | 它就是本后端存在的理由，是「X 年的值是多少」的答案来源 |
| **分析**文字 | ❌ 剥离（复用 `stripNumbers`） | 幻觉集中在叙述里；分析只负责回答「这张图说明了什么」 |

模型是自定义三件套（SigLIP2 + 投影 + Qwen3），需要 torch，且权重 7.4 GB 要常驻，所以它是**独立进程**，
不是 sidecar 模块（sidecar 只有 pymupdf + onnxruntime）：

```bash
pip install -r chartvlm/requirements.txt

# 两个 backbone 首次会从 HuggingFace 拉 ~4 GB；国内建议先用 ModelScope 拉到本地再指过去：
python -c "from modelscope import snapshot_download; \
print(snapshot_download('Qwen/Qwen3-1.7B')); \
print(snapshot_download('google/siglip2-base-patch16-512'))"

python chartvlm/server.py --root /path/to/laya-chart --port 8731 \
    --llm    ~/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master \
    --vision ~/.cache/modelscope/models/google--siglip2-base-patch16-512/snapshots/master \
    --preload                        # 不加则在首张图时再加载
curl -s localhost:8731/health        # {"ok":true,"ready":true,...}
```

> **`transformers` 必须锁 `5.4.0`（已写在 requirements 里，不要升级）**。SigLIP2 的视觉参数名在版本间变过：
> ≤5.0 拿到的 config 没有 `hidden_size`（投影层建不起来）；**≥5.5 把 `vision_model.` 这一层去掉了，208 个视觉
> 参数全部 miss** —— 而 `load_state_dict(strict=False)` **不会报错**，视觉塔会以随机权重跑，输出格式完美、
> 内容全是编的。判据是加载日志里的 `missing=` / `unexpected=` 必须**都不出现**（5.4.0 实测为 0）。

装好依赖后**不需要手工起服务**——内核自己拉起并回收它：

```bash
FREERAG_VLM=chart ./freerag          # 或桌面端：设置 → 图表识别方式 → Laya-Chart
```

```
figure captions: laya-chart-1.7b at http://127.0.0.1:59396 (concurrency 1, longest side 768px)
                 (started by this process, port 59396)
```

端口是启动时现选的空闲端口，不是固定 8731：两个内核（比如开发版和桌面版）不该撞在同一个端口上，
然后让没抢到的那个悄悄去用对方加载的模型。服务的 stderr 会转发进内核日志，权重在**首次转译**时才加载
（不是启动时），关掉内核时进程一起退出、7.4 GB 归还。

| 环境变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `FREERAG_VLM` | 空（关闭） | `chart` = 本后端；`go` = Go 侧通用视觉模型；空 = 只用解析期的 sidecar |
| `FREERAG_CHARTVLM_ROOT` | 空（**必填**） | Laya-Chart 解压目录（含 `src/` `scripts/` `configs/` `ckpt_chart_1_7b_r5/`）。没设就不启动，并明确报错——不是等到第一张图才失败 |
| `FREERAG_CHART_ENDPOINT` | 空 | 设了就**不自管**，连这个已有服务（适合 GPU 机器上常驻一个）。不设则内核自己拉起 |
| `FREERAG_CHART_CONCURRENCY` | `1` | 并发数。**默认 1**（Ollama 路径是 4）：一次转译是 1.8B 模型解码最多 768 token，都在同一个 Python 进程里，第二张图是排队而不是并行 |
| `FREERAG_CHART_MODEL` | `laya-chart-1.7b` | 只写进 chunk 的 `figure_summary_model` 作溯源。**刻意不共用 `FREERAG_VLM_MODEL`**：那个是 Ollama 标签，用它会把 Laya-Chart 干的活记成别的模型干的 |
| `FREERAG_CHART_PYTHON` / `_SERVER` | 自动探测 | 解释器默认用仓库根的 `.venv-chartvlm/bin/python`（torch 只装在那里） |

> 桌面端把 `FREERAG_VLM` 写成 `on`/`off`（驱动 sidecar 的解析期视觉），**只有选了 Laya-Chart 才写 `chart`**。
> 选中后「视觉模型 / 描述并发 / 描述长度」三项会置灰：这三项在图表模式下不读取，留着可编辑等于让用户存一组
> 会保存、会生效、但什么也不改的值。

**入库前的护栏**（都是实测确认的失败模式，不是理论风险）：

- **输出被截断就整张丢弃**。模型训练时的 `max_len` 装不下完整目标，它有「写一半就停」的倾向；
  半张表比没有表更危险——它看起来是完整的，缺的行谁也看不出来。此时 chunk 记 `chart_truncated=true`
  ，正文保持原样。
- **密集图会转译不全**。图像 token 固定 640，20 条系列的图只能读到约 8 条；`chart_series > 8` 时
  标 `chart_low_confidence=true` 供下游降权。
- **`chart_type` 不可信**：20 万条真实训练样本的图型标签是随机填的，不要用它做过滤或路由。
- `chart_nhr` / `chart_rows` / `chart_series` 一并写进 metadata，检索命中后可判断可信度。

缓存与通用路径分开（`.chart.json`）：一次转译的**质检标志**不只是文字，无法从正文反推，所以必须一起缓存。

**实测**（Apple M5 / MPS / fp32 / 850×600 的 ChartQA 图，端到端含 HTTP）：

| 项 | 结果 |
| :--- | :--- |
| 单图耗时 | **105–107 s**（CUDA 上报告值 13.9 s，CPU/MPS 慢一个量级，批量入库建议夜间跑） |
| 转译结果 | 13 行数据表，`truncated=false`、`schema_ok=true`、**NHR=0.0** |
| 数值准确性 | Lamb 103.7 / Corn 103.1 → 差值 0.6，该图 ChartQA 标注答案 0.57 ✅ |
| `chart_type` | 判成 `scatter`（实际是柱状图）—— 印证上文「图型不可信」，不要用它做过滤 |

> 一个已知代价：分析文字经 `stripNumbers` 后句子会残缺（`The lowest point is  at Rice.`）。这是**故意的**
> ——它换掉的是「误读的数值被当成证据永久检索」。表格已提供精确数值，分析只负责「这张图说明了什么」。
> 若你的场景更看重分析可读性，改 `vision_chart.go` 的 `chartText()` 让它跳过 `stripNumbers` 即可。

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
关推理前  113s（首次提问）
关推理后  15.4s   （答案 147 字符，"two to four sentences"）
改详尽后  83s     （答案 646 字符，4.4× 长）
```

**答案长度完全由 `draftSystemPrompt` 决定**，而且只有它决定 —— `num_predict` 是上限（实测最多用到 296），`MaxDraftChars` 是截断（从未触发），两者都没有真正塑造过答案。同一问题、同一证据、只换系统提示：147 → 315（去掉长度指令）→ 537（要求详尽）字符。

> ⚠️ **但"更长"当前有一部分是空话，原因不在 prompt。** 实测那次详尽回答引用的 6 条证据**全部短于 120 字符，其中 5 条就是标题**（`一、考核目的` 6 字符、`重点考核：` 5 字符、`三、交付要求` 6 字符）。
> 模型被要求写出 646 字符的详尽答案，而手头只有约 135 字符的实质内容 —— **它只能靠重复凑**。
>
> 根因是**分块**：索引里 1302 个 chunk 中有 **164 个 `Title` 块，平均 27 字符**，而标题逐字包含查询词，所以检索会**优先**返回它们。这不是调 prompt 能修的，需要让标题并入其后的正文块（见 §0.1）。

**流式输出**：`ask` 期间答案以增量事件逐步推送，前端边收边渲染（写完再换成带引用跳转的版本）。

```
t=  6s   draft 增量= 11 个   累计  18 字符
t= 18s   draft 增量= 86 个   累计 161 字符
t= 42s   draft 增量=237 个   累计 431 字符
t= 66s   draft 增量=360 个   累计 646 字符   ← 结束
```

> 流式下**不能对每个分块单独剥离 `<think>`**：Ollama 按自己的边界切分，`<thi` 和 `nk>推理` 可能落在不同块里，逐块正则两半都不匹配，用户的答案前面就会被打印出模型的私有推理。`thinkFilter` 因此会**少量回退**：只压住可能是标签开头的几个字符，普通文本照常立即输出（实测 8 个用例，含跨块、三分块、未闭合、`a < b` 这类普通尖括号）。

**启动预热**（内核在后台发一个与 tool-planning 同前缀的请求）：

```
model warm-up done in 5.326s
```

它刻意复用 tool-planning 的**系统提示与工具定义** —— Ollama 缓存的是提示词**前缀**，换个提示词预热就只省了那 0.5 s 的加载，白付一次模型调用。预热**不阻塞启动**，失败只记日志：没有 Ollama 时循环本来就退化为抽取式草稿，预热失败只是同一情况的轻微不便。

> 上述数字来自本机实测，机器相关。在更慢的机器上 `think: false` 的相对收益会更大（生成更慢，而浪费的 token 不变）。
>
> `FREERAG_KEEP_ALIVE` 默认 `30m`：卸载模型会**连提示词缓存一起丢掉**，而工具定义与系统提示是每一轮首个请求的主体，重算这个前缀比重新加载权重更贵。

## 证据压缩（REFRAG 式）与流式合并

**默认都关闭**，且关闭时提示词与改造前**逐字节相同**——没有实测的提示词改动和回归无法区分。

| 环境变量 | 默认 | 说明 |
| :--- | :--- | :--- |
| `FREERAG_REFRAG` | 空（**关闭**） | 打开**选择性证据压缩**。只把排名最前的段落展开成原文，其余以 gist 进入提示词 |
| `FREERAG_REFRAG_EXPAND_TOP` | `4` | 展开前几段；其余压缩 |
| `FREERAG_REFRAG_GIST_CHARS` | `240` | 每段保留多少字符（< 40 视为配置错误，回落默认） |
| `FREERAG_STREAM_COALESCE_MS` | `0`（**关闭**） | 答案分片的合并窗口；首帧永不等待 |
| `FREERAG_STREAM_COALESCE_RUNES` | `0` | 合并帧的字符上限，到顶即发 |

**实测（受控 A/B：同一问题、同一证据池、同一生成器，只换渲染方式；预热 + 交替两遍取最快）**：

```
未压缩  证据 13468 字符   TTFT 25.83s   整次调用 80.56s   答案 961 字符
压缩后  证据  6818 字符   TTFT  6.44s   整次调用 46.09s   答案 887 字符
```

**TTFT 4.01×，且不是靠把答案写短换来的。** 证据块字符数随池深变化：
池 6 → 1.52×、池 10 → 1.95×、池 20 → **3.50×**；池 20 时未压缩的块已被预算截断（14/20 段进入），
压缩后 **20/20 段全在**——所以它同时买到"更少 token"和"更长上下文"。

**流式合并默认关闭，因为它在结构上帮不上忙**：生成 ~4.5–6.4 tok/s，分片间隔约 **220 ms**，
比 120 ms 的合并窗口还慢——绝大多数分片等不到同伴，只能自己成帧。
实测（同题 2 臂 2 遍）：**帧 227 → 214（−5.7%），延迟无差别**。
一帧的成本是微秒级，**没有东西可摊薄**。开关留着，条件写在 `internal/ipc/coalescer.go`：
生成端比窗口快一个数量级时它才是收益。完整数据与"为什么不搬 REFRAG 的 encoder / RL 策略 / 预计算"
见 [`docs/performance.md`](docs/performance.md) §10。

> ⚠️ **`FREERAG_GENERATION_TEMPERATURE=0` 在此之前从未生效**（2026-10-05 修复）。
> `OllamaModel` 原先只在 `temperature > 0` 时才下发，于是"0"等于什么都不发，实际跑在 Ollama 默认的
> **0.8** 上——同一道题、同一个知识库，两次运行的证据池在 44k↔72k 字符之间跳。
> 现在 `0` 就是贪心，负值才表示"用服务端默认"。**任何"温度 0 下的对比"结论都需要按这条重看。**

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
set -a && . ./.env && set +a  # 直接跑内核时需要；桌面端 npm start 会自动加载
```

> 桌面端（`desktop/main.js` 的 `loadDotEnv`）启动时会读**仓库根目录**的 `.env` 并下发给内核，
> 所以 `npm start` 不必手动 source。`process.env` 优先于文件，显式 export 不会被覆盖；
> 内核单独运行时仍按上面两行 source。忘了这一步的失败形态是**静默**的——内核只打一行
> `dense retrieval disabled`，`hybrid_search` 退化成纯关键词，症状只是答案变差。

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

## 问答编排（eino）

`ask` 的顶层流程是一张 **eino `compose.Graph`**（`internal/agent/flow.go`）。它取代了原来「直接跑一个 agentic 循环」的入口：

```
START → route ─(simple)→ rag ──────────────────────────────→ END
              └(complex)→ decompose → fanout → synthesize → END
```

- **route**：把问题交给 Laya 判**简单 / 复杂**（`Decide` 的二选一，~100 ms）。Laya 不可用时退化为确定性启发式（`internal/agent/router.go`）。
- **简单** → `rag`：直接跑一次 `Loop.Run`，只有一次 agentic pass。
- **复杂** → `decompose`：生成模型把问题拆成至多 4 个**互相独立**的子问题（JSON 数组；解析失败则退回原问题）。
- **fanout**：每个子问题**并发**跑一次 agentic 循环，各自独立检索。
- **synthesize**：生成模型把子答案合并成最终答案，**流式**输出。子循环的答案不流式——`fanout` 给每个子循环一份 `OnDraftDelta = nil` 的副本，否则会把「答案的一部分」当成整个答案发布出去。

> eino 的 `compose.Graph` **不支持环**，也没有 `Map` 原语：所以「轮次循环」仍留在循环内部、作为图里的一个节点；动态 N 路 fan-out 用节点内并发（子问题数来自 `decompose`，没有静态的 `Parallel` 分支可用）。

### 每轮由 Laya 决定下一个工具调用

**Laya 是非生成式决策模型 —— 它只能对给定选项打分取 argmax，不能生成工具参数。** 所以「让它决定工具调用」实现为：循环每轮确定性地构造一组**可执行的候选调用**（`internal/agent/toolchoice.go`），Laya 只负责选一个：

| 候选 | 参数来自 |
| :--- | :--- |
| `search("…")` | 原问题、最近一次 rewrite 的 query |
| `grep("…")` | SCA 报告的 missing 词（最多 2 个） |
| `read("doc.pdf")` | 证据池里已出现的 doc_id（最多 2 个） |
| `stop` | 证据已足够，不再检索 |

收益正是替换掉原来每轮 ~40 s 的「工具规划」模型调用（≈100 ms），而且**候选集不可能包含编造的值**——模型把 `author_zhao_hui` 当 doc_id 那类事故在候选集里根本不存在。已试过的调用会被剔除，因此不会重复。

回退顺序：**Laya 选工具 → 模型规划（`toolPlanSystemPrompt`）→ 确定性 `roundCalls`**。工具提示按 RAGFlow 的 `action_run.md` 精简过，只保留 freerag 真实存在的 4 个工具（去掉了 `navigate_*` / `graph_explore` / `calculate` / `web_search`）。

### 依赖

引入 eino 打破了原先「`go.mod` 零依赖、离线可构建」的状态：现在多出约 27 个间接依赖，`go build` 需要能访问模块代理。本网络下 `proxy.golang.org` 不可达，走镜像：

```bash
export GOPROXY=https://goproxy.cn,direct
go build -o bin/freerag ./cmd/freerag
```

## Agentic 工具选择

medium 模式的循环（§6.1）把**选哪些工具**交给生成模型：每轮先问一次「为这个问题该调哪些工具」，
模型从 §6.7 的 4 个工具里自选；内核执行后把结果并入证据池，再由 Laya 判充分性。

```
[Action Session] the model chose 1 call(s): metadata_search(filters=[key:indexed_at op:start with value:2026-09-29] logic:and)
[Tool] metadata_search(...) -> +3 passage(s). 3 chunk(s) from 1 document(s) matched indexed_at start with 2026-09-29 and
[RAGAgent] Round 1: +3 passage(s); pool now 3.
[SCA] Round 1 verdict=SUFFICIENT (missing=0).
```

### 循环的四条不变式

这四条都**不是"提示模型做对"，而是循环自己保证**：

| 不变式 | 为什么不能交给模型或 checker |
| :--- | :--- |
| **空证据池永不判充分** | checker 被问的是"这份草稿有没有回答问题"，而"证据里没有"的草稿**确实回答了**——实测 Laya 给 **0.93** 置信度的 sufficient。它没答错，是**问错了**："看没看过语料"这个事实住在 `len(evidence)` 里，而 §6.6 刻意不让 checker 看到证据。守卫因此放在 `Loop.Run`（evidence 在作用域内），空池直接判 `INSUFFICIENT` 且**不咨询 checker** |
| **"已试过什么"进下一轮提示词** | 系统提示词早就写着"不要重复已做过的调用"，但**从没说过那些调用是什么**。证据为空时这个遗漏正好致命：那时提示词里没有别的线索，"刚返回 0 的那个调用"就是最可能的下一个动作 |
| **完全相同的重复调用被拒绝** | 实测：提示词把该调用列出来并标注 `returned nothing`，模型第 3 轮**照样重复**。一次 run 内索引不变，重复不可能带来新信息——所以由 loop 在**预算检查之前**拒掉它 |
| **答案只在循环结束后写一次** | 每轮的草稿是给 checker **判**的**提案**，可能被否决、要求再来一轮。边写边发到答案区，就是"答案改主意"：第一轮没检索到就流出"证据无法回答"，下一轮把它换掉。所以草稿**不流式**（`draft`），循环结束后 `answer` 写一次并流式输出 |

> `CoverageChecker`（无模型时的备用件）一直有第一条的守卫；`LayaChecker`（实际在跑的）没有。守卫现在在 `Loop.Run` 里，两条路径都覆盖。

**答案与草稿的分工**（`internal/agent/loop.go`）：

```
循环：检索 → draft（给 checker 判，不输出）→ 判定
        ↓ 判定不足则重来；充分或轮数用尽则退出
循环结束 → answer（写一次，流式输出）
```

实测事件顺序（`test1` 库，3 轮问答）：

```
Round 1  metadata_search -> +0     [Draft] skipped: no evidence to draft from.
         [SCA] Round 1 verdict=INSUFFICIENT (no evidence to judge; the checker is not asked)
Round 2  hybrid_search -> +6       [SCA] Round 2 verdict=INSUFFICIENT (missing=6)
Round 3  metadata_search（重复）→ 被拒    [SCA] Round 3 verdict=SUFFICIENT
         ↓ 循环结束
         >>>>>> 答案开始流出 <<<<<<
```

流出的 **733 字符 == `Result.Draft` 的 733 字符** —— 屏幕上那段文字就是最终答案，不流式输出任何会被撤回的东西。

**实测代价**（`test1` 库，3 轮问答，含计时 trace）：

| 阶段 | 耗时 |
| :--- | ---: |
| Round 2 的判定草稿 | 37.1 s |
| Round 3 的判定草稿 | 38.8 s |
| 循环结束后的答案 | 41.4 s |
| **总计** | **164 s** |

改造前这 3 轮只需 **2 份**生成（最后一轮的草稿**兼**答案），约 76 s 生成；
新结构是 **3 份**，多一次完整生成（**+41.4 s，约 +35%**）。
1 轮问答（最常见情形）则是 1 份变 2 份，**接近翻倍**。

这一份不是重复劳动——判定草稿写于某一轮的证据快照、目的是"能否回答问题"；答案写于**定稿证据池**、目的是"给人读"。
但**要把 N 份降到 1 份，正确做法是让 checker 直接判证据而不是判草稿**（§6.6 的改动），而不是砍掉写答案这一步。

实测（`test1` 库，原问题「赵慧为作者的论文有哪些，讲了啥」）：

```
改前   rounds=1  evidence=0  verdict=SUFFICIENT（假的：空池被当成了充分）
改后   rounds=2  evidence=6  verdict=SUFFICIENT（真实判定：非空池上 Laya 判充分）
```

改后的草稿会列出论文与共同作者，但**检索质量仍有残留**：中文问「赵慧」而语料是英文（署名 `Hui Zhao`），
`hybrid_search` 返回的 6 条里混有标题页碎片，草稿会把 `Introduction` 当成论文标题。
那是**跨语言召回质量**问题，与循环结构无关。

### `metadata_search`：只有两个字段

[`internal/store/metafilter.go`](internal/store/metafilter.go) 里**硬编码**两个可过滤字段，`key` 的 enum 也只列它们：

| 字段 | 含义 | 取值来源 |
| :--- | :--- | :--- |
| `doc_id` | 文档 id（源文件名） | **chunk 自身的 DocID**，不是清单 —— 有 chunk 却没清单记录的文档也必须能过滤到 |
| `indexed_at` | 装入时间，`YYYY-MM-DD HH:MM:SS` | 清单的 `DocumentRecord.IndexedAt`；缺失则留空 |

算子沿用 RAGFlow 的一整套（`=` `≠` `>` `<` `≥` `≤` `in` `not in` `contains` `not contains` `start with` `end with` `empty` `not empty`），语义逐条对齐它的内存实现。

**时间必须用 `start with`**，这是 RAGFlow 对自身时间字段的规则，我们原样照搬：

```
indexed_at 存的是 "2026-09-29 14:26:01"，所以
  要一天 → op "start with" + "2026-09-29"   ✅
  用 "="  + "2026-09-29"                     ❌ 永远匹配不上（存的是完整时间戳）
  用 "="  + "2026-09-29 14:26:01"            ✅ 但要求秒级精确
```

**未知字段与"合法空结果"是两句不同的话。** 这一条是那次事故的直接修复：

| 情形 | 返回 |
| :--- | :--- |
| 字段不存在（`author`） | 拒绝，并列出真实字段与可用取值：`metadata field(s) [author] do not exist ... It has exactly two: doc_id, indexed_at. Available: ...` |
| 算子不存在 | 拒绝，并列出全部算子 + `start with` 的时间提示 |
| 字段与算子都对，但没匹配到 | 明确说明"这是关于**过滤器**的陈述，不是关于语料的"，并把真实 doc_id 列出来 |

规划提示词里还有一段 **AVAILABLE METADATA**，列出两个字段**实际存在的取值**（值 + 覆盖文档数），
结尾一句取自 RAGFlow：*"Values NOT listed here do not exist — never invent one."*
—— 因为枚举字段名挡不住编造**取值**，而那次事故编的正是取值。

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

`desktop/` 有两个视图，共用一条顶栏（品牌 / 视图切换 / 健康芯片）：

**知识库视图** —— 管理"能问什么"，分两级，与问答视图同形。

*知识库选择页*（**每次进入知识库页都落在这里**）：

| 区域 | 内容 |
| :--- | :--- |
| 顶部 | 「新建知识库」 |
| 列表 | 知识库卡片：名称、文档数与 chunk 数、创建日期 |

*知识库内*：

| 区域 | 内容 |
| :--- | :--- |
| 左栏 | 「← 返回」、「重命名」/「删除」、库名与统计、**拖入 PDF** / 「添加文档」、文档列表（页数、chunk 数；点选即打开分块） |
| 主区 | **分块检查器**（见下） |

**分块检查器** —— 一块 chunk 和它来自的那片原文，是同一个事实的两种说法：

| 区域 | 内容 |
| :--- | :--- |
| 左 | **原文页面**：sidecar 用 PyMuPDF 渲染成 PNG，上面按 bbox 叠加高亮框。翻页箭头 + 「显示本页全部框」 |
| 右 | 该文档的**分块列表**：chunk_id / 块类型 / 页码 / 字数 / 文本（选中那块的文本展开，其余折到 3 行）。「只看本页」 |

**两边点击互相定位**：点右侧某块 → 左侧自动翻到它所在的页并高亮它的框；点左侧的框 → 右侧选中并滚到对应卡片。**翻页是选择的一部分，不是单独一步** —— 一个在第 7 页的 chunk 却在第 1 页上高亮不出任何东西，正是这个界面要防止的事。

**为什么原文在 sidecar 里渲染而不是直接嵌 PDF 阅读器**：bbox 是按 **PDF 点**（`page.rect` 空间、左上原点）度量的，所以页尺寸（pt）随图一起返回，前端只做百分比换算 —— **一个坐标系**，Electron 侧永远不用打开 PDF。框用百分比而非像素，缩放窗口时框跟着图走，不需要任何 resize 监听。见 `sidecar/parse_server.py` 的 `method_render` 与 `desktop/renderer/app.js` 的 `renderBoxes`。

**整篇分块一次取回**：`chunks` 返回该文档的全部分块，所以翻页与「只看本页」都是**本地重渲染**，不等内核 —— 翻页真正需要的是一张渲染好的页图，而不是另一份列表。

**文件被移走时会明说**：原文路径来自索引时的清单记录。文件不在了就显示「原文无法显示：…」，而不是把高亮框压在空白页上 —— 后者看起来像数据坏了，而不是文件没了。

**进入知识库页也一定落在选择页**，理由与问答视图相同：上次在哪个库，不是"现在要往哪儿放文件"的答案。

## 主题（白昼 / 黑夜）

顶栏右侧有切换按钮。组件从不写颜色，只写角色（`--panel` 表示"抬升于页面之上的面"），十六进制值只出现在 `styles.css` 的两处调色板里。

**浅色不是深色反相。** 反相会得到 `#3fb950` 的绿，在白底上是 1.9:1 对比度——当文字读不了、当细线看不见。两套配色的实际取值都是对着它所在的面选的。

**唯一不随主题变的一组颜色**是原文页上的高亮框。它压在白纸上，跟着调色板走会在深色主题里被调成"对近黑背景可见"、然后在白纸上消失。所以 `--box-*` 只在 `:root` 定义一次、浅色主题不覆盖。

**主题从哪来**（`main.js` 的 `resolveTheme`）：`FREERAG_THEME` 覆盖 → `prefs.json` 里存的选择 → 操作系统。环境变量排第一是因为截图与测试要能钉住主题而不管机器上选了什么，**且它不写回文件**。最后一档兜底是"从没选过的人"——把浅色机器开成黑窗口，是替他们做的一个错决定。

**界面调用的接口**是 `window.freeragTheme`（`list` / `current` / `set` / `toggle`）。按钮和 DevTools 控制台走同一条 API，"校验名字 → 应用 → 持久化 → 写失败要报错"只写一遍。

**防闪白**分两半：窗口 `backgroundColor` 按主题设，preload 从 `process.argv` 读主题并在 `<html>` 出现时立刻写上 `data-theme`。走 IPC 就晚了——它

**问答视图** —— 管理"问过什么"，分两级。

*会话选择页*（**每次进入问答页都落在这里**）：

| 区域 | 内容 |
| :--- | :--- |
| 顶部 | 「新建会话」 |
| 列表 | 会话卡片：名称、对话数与消息数、它绑定的知识库、悬停可删 |

*会话内*：

| 区域 | 内容 |
| :--- | :--- |
| 左栏 | 「← 返回」、会话名与绑定知识库、「新建对话」、该会话的对话历史 |
| 右栏 | 消息记录与提问框（回车发送，Shift+回车换行） |

**为什么进入时一定落在选择页**：上次你待在哪个会话，不是"现在想问什么"的答案 —— 静默恢复它，正是问题被打进错会话的方式。会话内**刻意不再显示会话列表**：那会给出两条改变"当前在哪个会话"的路径，而返回键已经是其中一条。

**每次提问的工具调用轨迹显示在这条消息里，不在全局状态条上。** 它回答的是"这个答案是怎么来的"，那是读完答案之后的问题 —— 放在对话里，它会留在原地、跟着会话一起保存；放在状态条上，下一件事发生时就没了。生成过程中它是**展开**的（此时它是唯一的进度信号），答完自动**折叠**成「检索与推理 N 步」，不把答案挤出屏幕。消息头另外记录该次提问的耗时。

**会话在创建时绑定一个知识库，之后不可更改。** 允许更改等于让同一个会话里的历史答案被重新解读 —— 它们引用的段落来自另一个索引。知识库被删除后，会话仍可翻阅，但不能再提问（提问会从空索引里检索，看起来像提问本身失败）。

**会话与对话历史由外壳（`desktop/main.js` → `chats.json`）保存，不在内核里。** 它是一串问答记录，检索链路的任何环节都不读它；放在外壳意味着这份文件只有一个写入方，也不必为内核从不接触的数据新增 RPC。写入用临时文件 + rename —— 这是全部对话的唯一副本，写到一半崩溃会留下一个解析不出任何东西的文件。历史损坏时**报告而不是覆盖**：手工还能救回来，静默清空则看起来像对话从未存在过。

```bash
go build -o bin/freerag ./cmd/freerag     # 界面需要内核
cd desktop && npm start

# 截图（开发/CI 用，不触发录屏权限弹窗）：
FREERAG_SCREENSHOT=/tmp/ui.png npm start
FREERAG_SCREENSHOT=/tmp/chat.png FREERAG_SCREENSHOT_VIEW=chat npm start
```

> **答案在流式期间是按纯文本写入的，只有流结束后才转成 HTML。** 它来自一个读过文档的语言模型，而文档本身也可能含标记；按 HTML 处理等于允许文档往应用里注入脚本。流式还多一层：分片边界可能落在一个标签或引用标记中间，**逐片转义也拦不住** —— 只有把整个答案拼起来才知道。所以流式用 `textContent`，结束后再由 `renderMessages()` 整体替换。

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
