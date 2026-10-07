# fs 通道实验：步骤与命令

> 从零到出数字的全部命令。所有路径以仓库根目录 `/Users/cbn/workspace/freerag` 为基准。
> 方案说明见 [`README.md`](README.md)。
> **新机器必读 §1.1（装依赖）与 §1.4（数据集路径）**；本机已齐全，可直接跳到 §0。

---

## 0. 一键跑（已经建好 KB 的情况）

```bash
cd /Users/cbn/workspace/freerag
bash experiments/fs-doctree/run_all.sh
```

它做四件事：构建 Go 内核 → `fs.index` 建树 → `fs.status` 打印树形 → 跑 5 个通道的评测 → 结果写到 `eval/runs/doc-recall.json`。约 3–5 分钟。

想连 KB 一起从 PDF 重建（约 15–25 分钟，只在第一次或换语料时需要）：

```bash
bash experiments/fs-doctree/run_all.sh --reindex
```

---

## 1. 环境

### 1.1 依赖安装（新机器必做；本机已齐全）

| 依赖 | 本机状态 | 新机器 | 体积 |
| :--- | :--- | :--- | :--- |
| Go 工具链 | 有（1.22.12） | 装（或 `scripts/install-go.sh` → `.toolchain/go`） | — |
| Go 模块 | 已在 `/Users/cbn/go/pkg/mod` | 首次构建自动拉（无 `vendor/`） | — |
| `.venv314` | 有（onnxruntime / pymupdf 齐全） | 装 | 225M |
| `models/` | 有 | **下载** | laya-onnx 1.2G + deepdoc 99M |
| MultiHop-RAG PDFs | 有，101 篇 | **自备**（见 §1.4） | — |
| KB `eval/data-v12` | 已建 | `--reindex` 重建（§2 步骤 2） | — |

```bash
cd /Users/cbn/workspace/freerag

# 1) Go 工具链 + 模块（无 vendor/，GOPROXY 走镜像更快）
scripts/install-go.sh                          # 装到 .toolchain/go，默认 1.27.1
export PATH="$PWD/.toolchain/go/bin:$PATH"
export GOPROXY=https://goproxy.cn,direct       # eino 依赖需走镜像
go mod download

# 2) Python 环境（sidecar 解析 PDF + laya 推理）
python3 -m venv .venv314                       # 需要 >= 3.10，PyMuPDF 是 cp310-abi3
.venv314/bin/python -m pip install \
  -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 3) 模型。models/ 被 gitignore，不在仓库里，必须自己下
bash scripts/download-models.sh                # all；也可只下 laya 或 deepdoc
#   默认走 hf-mirror.com（huggingface.co 在这网络不可达），覆盖：
#   HF_ENDPOINT=https://huggingface.co bash scripts/download-models.sh laya
```

只跑 fs 通道的话：**laya 必须下**（方向判定靠它），**deepdoc 建库时要**（解析 PDF 用）。

**不需要下载的两样**：

- **ragas** —— `ragas_eval.py:400` 是函数内 lazy import，只有 `score` 子命令用；本实验走 `index` + `eval_doc_recall.py`，纯 stdlib。
- **向量 / 嵌入** —— fs 通道不依赖嵌入。本机建库时托管嵌入大部失败（84/731 有向量），不影响任何 fs 数字。

装完自检（四项都要有输出/为真）：

```bash
go version
.venv314/bin/python -c "import onnxruntime, pymupdf; print('venv ok')"
ls models/laya-onnx                       # 至少 laya.onnx / laya_config.json / tokenizer/
ls /Users/cbn/workspace/benchmarks/MultiHop-RAG/pdfs | wc -l   # 101
```

> ⚠️ `.venv314` **未被 `.gitignore` 覆盖**（规则只写了 `.venv/` 和 `venv/`），也未被 git 跟踪。别用 `git add -A`，否则 225M 环境会被提交。

缺 laya 也能跑：树照建，路由自动降级为关键词，trace 里 `decider: keyword`、`reason: keyword`。**这时 `fs` 与 `fs-keyword` 数字会一致**，不要误读成"laya 和关键词一样好"。

### 1.2 依赖清单与检查

| 依赖 | 用途 | 检查 |
| :--- | :--- | :--- |
| Go ≥ 1.22 | 内核 `cmd/freerag` | `go version` |
| Python 3.12+（`.venv314`） | sidecar，内含 laya-32k | `.venv314/bin/python -c "import onnxruntime; print('ok')"` |
| `models/laya-onnx/` | 方向判定模型 | `ls models/laya-onnx` |
| MultiHop-RAG PDFs | 语料 | `ls /Users/cbn/workspace/benchmarks/MultiHop-RAG/pdfs \| wc -l` → 101 |

### 1.3 需要的文件（都在仓库里，不用另外下载）

| 文件 | 作用 |
| :--- | :--- |
| `eval/kb_sources.json` | 101 篇 PDF 的文件名清单（建库用） |
| `eval/dev.json` | 32 道题（其中 **24 道有证据**，8 道 null 题会被自动剔除） |
| `eval/test.json` | 结论定稿后在这里复测（**不要拿它调参**） |
| `scripts/ragas_eval.py` | `Kernel` 类：起内核、发 JSON-RPC |
| `scripts/eval_tree_recall.py` | 提供 `basename` / `norm` / `fact_found` 三个判定函数 |
| `scripts/eval_doc_recall.py` | 评测主脚本（本目录 `eval/` 下是其副本） |

### 1.4 ⚠️ 换机器必须改这一处

`eval/kb_sources.json` 里的文件名会拼到硬编码目录上。查看当前根路径：

```bash
grep -n "MultiHop\|benchmarks\|pdfs" scripts/ragas_eval.py | head
```

把 PDF 放到对应目录，或改脚本里的根目录常量，保证 101 篇都能找到。否则建库会静默少文档，召回数字全线偏低而看不出原因。

---

## 2. 分步命令

### 步骤 1 — 构建内核

```bash
cd /Users/cbn/workspace/freerag
CGO_ENABLED=0 go build -o bin/freerag ./cmd/freerag
```

改动 `internal/dirtree/` 或 `cmd/freerag/dirtree.go` 后必须重新构建，内核是长驻进程，脚本每次新起一个，所以**改完代码直接重跑脚本即可**。

### 步骤 2 — 建 KB（PDF → chunks，只做一次）

```bash
python3 scripts/ragas_eval.py index --kb eval/data-v12 --sources eval/kb_sources.json --force
```

预期：101 docs / 731 chunks。

> **已知缺陷**：只有约 84/731 个 chunk 拿到了向量（托管嵌入在建库时大部失败）。本方案**不依赖向量**，所以不影响 fs 通道；但会影响任何语义相似度类的结构评估指标。

### 步骤 3 — 建目录树

```bash
python3 - <<'PY'
import sys, json, pathlib
sys.path.insert(0, 'scripts')
from ragas_eval import Kernel
k = Kernel(pathlib.Path('eval/data-v12'))
kb = open('eval/data-v12/kb_id.txt').read().strip()
print(json.dumps(k.call('fs.index', {'kb': kb}), ensure_ascii=False))
k.close()
PY
```

输出示例：`{"documents": 101, "nodes": 71, "from": "cluster", ...}`。

可调形状（**评测时必须固定**：`branch=6 / leaf_size=4`）：

```python
k.call('fs.index', {'kb': kb, 'branch': 6, 'leaf_size': 4})
```

`from` 字段会说明树从哪来：`path`（用了真实目录）/ `cluster`（全靠聚类）/ `mixed`。本语料 101 篇在同一个文件夹里，所以恒为 `cluster`。

### 步骤 4 — 看树长什么样

```bash
python3 - <<'PY'
import sys, json, pathlib
sys.path.insert(0, 'scripts')
from ragas_eval import Kernel
k = Kernel(pathlib.Path('eval/data-v12'))
kb = open('eval/data-v12/kb_id.txt').read().strip()
s = k.call('fs.status', {'kb': kb})
print('docs', s['documents'], 'nodes', s['nodes'], 'leaves', s['leaves'],
      'depth', s['depth'], 'from', s['from'], 'decider', s['decider'])
for t in s['top']:
    print('  -', t['name'][:60], '| children', t['children'], 'docs', t['docs'])
k.close()
PY
```

`decider` 必须是 `laya` 才算模型路由生效。

### 步骤 5 — 单条查询，看 trace

```bash
python3 - <<'PY'
import sys, json, pathlib
sys.path.insert(0, 'scripts')
from ragas_eval import Kernel
k = Kernel(pathlib.Path('eval/data-v12'))
kb = open('eval/data-v12/kb_id.txt').read().strip()
row = [r for r in json.load(open('eval/dev.json')) if r.get('evidence_pdfs')][0]
r = k.call('fs.search', {'query': row['question'], 'kb': kb, 'limit': 5, 'beam': 2})
print('gold:', row['evidence_pdfs'])
for st in r['trace']['steps']:
    print('step %d  from=%s  reason=%s  p=%.3f' % (st['level'], st['from_name'][:30], st['reason'], st['probability']))
    for o in st['options']:
        print('     %-44s %.3f' % (o['name'][:44], o['score']))
for h in r['hits']:
    print('HIT  %-40s source=%s  %s' % (h['doc_id'][:40], h['source'], ' > '.join(h['path'])[:60]))
k.close()
PY
```

trace 里要看的三件事：`reason` 是不是 `laya`、每层 `kept` 有哪几个、命中 `source` 是 `route` 还是 `keyword`。

### 步骤 6 — 跑全量评测

```bash
python3 scripts/eval_doc_recall.py \
  --kb eval/data-v12 --build -k 8 --cuts 3,5,8 \
  --channels keyword-doc,tree-doc,fs,fs-keyword,fs-pure \
  --out eval/runs/doc-recall.json
```

结果落 `eval/runs/doc-recall.json`，末尾打印汇总表。24 题 × 5 通道，约 1–2 分钟。

---

## 3. 通道与参数

| 通道 | 含义 | 用来看什么 |
| :--- | :--- | :--- |
| `keyword-doc` | BM25 聚合成文档（非树基线） | 天花板 |
| `tree-doc` | 旧章节树通道聚合成文档 | 另一个非目录树对照 |
| `fs` | laya 路由 + 关键词兜底网 | **本方案** |
| `fs-keyword` | 同一棵树，`no_laya` 关模型 | **树本身的质量**（与模型无关） |
| `fs-pure` | 同一棵树，`no_fallback` 关兜底 | **路由本身的真实水平** |
| `fs-keyword-pure` | 关键词路由 + 关兜底 | **模型消失了还剩多少**（关键对照） |
| `fs-candN` | `candidate_k=N`：只把关键词最强的 N 个目录给 laya 看 | 选项集大小的影响 |
| `fs-brN` | `branches=N`：根层选 N 个方向，各走到自己的叶子 | **加宽**（§5.1 里唯一赚到的） |
| `fs-beamN` | `beam=N`：每层都保留 N 个方向 | 另一种加宽，实测更差 |
| `fs-slotsN` | `slots=N`：拆成 N 个子问题（槽表），每槽各搜一遍树 | deep research 式扇出 |

**`-kw`（不用模型）和 `-pure`（关兜底网）可以叠加在任何一条后面**，也可以互相叠加（`fs-br4-kw-pure`）。没有这两个后缀的数字同时包含模型、兜底网和机制三件事，可以被读成支持其中任何一个的结论——不要那么跑。

`-kw` 尤其不能省：Beam=2 时 `cand2` 意味着两个候选都会被打开，laya 只影响先后顺序——收益可能全来自"缩减"而模型在蹭功劳。

`fs.search` 参数：

| 参数 | 默认 | 说明 |
| :--- | :--- | :--- |
| `limit` | 5 | 召回文档数 |
| `beam` | 2 | 每层保留几个方向 |
| `max_steps` | 8 | 决策次数上限（也是延迟上限） |
| `min_probability` | 0 | 低于此置信度改关键词排序 |
| `no_fallback` | false | 关兜底网 |
| `fallback_limit` | 3 | 兜底网预留槽位 |
| `no_laya` | false | 关模型，纯关键词路由 |
| `candidate_k` | 0 | 只把关键词最强的 N 个子目录给模型看；被筛掉的降级到队尾而非丢弃 |
| `branches` | 1 | 根层选 N 个方向，各自走到自己的叶子；每分支独立步数预算，结果按名次轮转合并 |
| `beam` | 2 | 每层保留 N 个方向（与 `branches` 不是同一个旋钮，见 README §2.5） |
| `slots` | 0 | 拆成 N 个子问题后每槽各搜一遍树。需要可达的 generator，否则跳过并在 `trace.notes` 说明 |
| `explain` | false | **纯输出**：附加每个目录/文档的命中词、一句话理由、证据段落、未开启分支的原因 |

---

## 4. 基线数字（应能复现）

有 laya 时：

| 通道 | doc@3 | doc@5 | doc@8 | fact@8 | complete@8 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| keyword-doc | 0.712 | 0.872 | **0.938** | 0.986 | 20/24 |
| tree-doc | 0.642 | 0.764 | 0.785 | 0.861 | 12/24 |
| fs | 0.035 | 0.056 | 0.743 | 0.868 | 10/24 |
| fs-keyword | 0.427 | 0.646 | 0.795 | 0.924 | 12/24 |
| fs-pure | 0.035 | 0.056 | 0.090 | 0.184 | 0/24 |

（同一份数字的原始文件：[`results/baseline-doc-recall.json`](results/baseline-doc-recall.json)。注意 `fs` 在 @3/@5 几乎为 0 而 @8 突增——不是路由在低位失效，是**兜底网的 3 个预留槽位排在末尾**，只有把 k 放到 8 才把它们包含进来。）

已验证：`run_all.sh` 在 2026-10-05 完整跑通，五个通道数字与上表逐一吻合，全程约 2 分钟。

### 4.1 看一次可解释输出

`explain` 会给出"为什么开这 4 个分支""为什么召回这篇文档""文档里哪句话命中"。任何通道名后面加 `-explain` 即可：

```bash
python3 scripts/eval_doc_recall.py --kb eval/data-v12 -k 8 --cuts 8 \
  --channels fs-br4-kw-explain --out eval/runs/explain-demo.json
```

或者单条查询直接看 trace（更直观）：

```bash
python3 - <<'PY'
import sys, json, pathlib
sys.path.insert(0, 'scripts')
from ragas_eval import Kernel
row = [r for r in json.load(open('eval/dev.json')) if r.get('evidence_pdfs')][0]
k = Kernel(pathlib.Path('eval/data-v12'))
kb = open('eval/data-v12/kb_id.txt').read().strip()
r = k.call('fs.search', {'query': row['question'], 'limit': 8, 'kb': kb,
                         'branches': 4, 'no_laya': True, 'explain': True})
step = r['trace']['steps'][0]
for o in step['options']:
    mark = '开' if o['nid'] in step['kept'] else '  '
    print(' %s %-52s %6.1f  %s' % (mark, o['name'][:52], o['score'],
          '、'.join('%s(%.1f)' % (t['term'], t['score']) for t in (o.get('matched') or []))))
for h in r['hits']:
    print('\n •', h['name'], '\n  ', h.get('why'))
    if h.get('evidence'):
        print('   证据:', h['evidence'][:150].replace('\n', ' '))
k.close()
PY
```

**已验证 `explain` 不改变召回**：`fs-br4-kw` 与 `fs-br4-kw-explain` 在 24 题上 24/24 文档列表完全相同，doc@8 都是 0.8542。改代码后这条要重测——解释层一旦参与决策，上面所有的消融数字就都作废了。

### 4.0 加宽与槽表（doc recall@8，24 题）

| 通道 | doc@8 | 全部找齐 | 中位耗时 |
| :--- | :--- | :--- | :--- |
| keyword-doc（BM25，参照） | **0.938** | 20/24 | 0.01s |
| fs-br4-kw | **0.854** | 15/24 | 0.16s |
| fs-br6-kw | 0.781 | 13/24 | 0.06s |
| fs-slots2-br4-kw | 0.781 | 12/24 | 9.5s |
| fs-br4（laya 选分支） | 0.771 | 11/24 | 1.7s |
| fs-beam4-kw | 0.747 | 12/24 | 0.16s |
| fs-slots4-kw | 0.733 | 13/24 | 12.6s |
| fs-slots3-kw | 0.701 | 13/24 | 11.6s |
| fs-br4-kw-pure | 0.563 | 6/24 | 0.16s |
| fs-br4-pure | 0.125 | 0/24 | 1.7s |
| fs-slots3（laya） | 0.083 | 0/24 | 16.8s |

原始文件：`results/ablation-branches-slots.json`。复现：

```bash
python3 scripts/eval_doc_recall.py --kb eval/data-v12 -k 8 --cuts 8 \
  --channels fs-br4-kw,fs-br4,fs-beam4-kw,fs-slots3-kw,fs-slots3-br4-kw \
  --out eval/runs/branches-slots.json
```

读法见 README §4。**没有任何一条打过 0.938**，这是这套机制降级为辅助通道的依据。

### 4.0b 结构容量（先跑这个，它决定值不值得继续）

```bash
python3 experiments/fs-doctree/eval/oracle_capacity.py \
  --out experiments/fs-doctree/results/oracle-capacity.json
```

| 指标 | 值 |
| :--- | :--- |
| gold 文档在某个叶子里 | 1.000 |
| **单路径上限（只进 1 个叶子）** | **0.510** |
| 多分支上限（≤2 个叶子） | 0.868 |
| 多分支上限（≤4 个叶子） | 1.000 |

单路径 0.510 低于 BM25 的 0.938 —— **单路径下降已被证伪，别再往里投**。多分支容量 1.000，剩下的空间是真的。读法见 README §4.1。

### 4.1 纯路由口径的对照（都关兜底网，doc recall@8）

| 通道 | doc@8 | 全部找齐 |
| :--- | :--- | :--- |
| fs-pure（laya 看全部目录） | 0.090 | 0/24 |
| fs-cand2-pure | 0.601 | 6/24 |
| fs-cand3-pure | 0.236 | 1/24 |
| fs-cand4-pure | 0.194 | 2/24 |
| **fs-keyword-pure（不用模型）** | **0.802** | **11/24** |
| fs-candN-kw-pure（缩减但不用模型） | 0.802 | 11/24 |

结论与读法见 README §4.1。**laya 在这套目录名上每一次判定都是净损失**，缩减候选只是让它的破坏范围变小。

复现：

```bash
python3 scripts/eval_doc_recall.py --kb eval/data-v12 -k 8 --cuts 8 \
  --channels fs-pure,fs-keyword-pure,fs-cand2-pure,fs-cand2-kw-pure,\
fs-cand3-pure,fs-cand3-kw-pure --out eval/runs/cand-ab.json
```

数字对不上时，按顺序查：

1. `fs.status` 的 `decider` 是不是 `laya`（不是则 `fs` ≡ `fs-keyword`）；
2. `documents` 是不是 101（少于 101 说明建库漏了文件，回 §1.4）；
3. 树是不是被重建过（`from` / `nodes` 变了就得所有通道一起重跑，不能只重跑一个）。

---

## 4.9 桌面端：知识库 →「文档路由」

`desktop/renderer` 右栏第三个模式就是这条通道，界面上三块：

| 区域 | 内容 |
| :--- | :--- |
| **路由图** | 每层决策逐个画出：每个候选目录一条分数条，**开**的高亮、**没开**的灰着仍在；条下是该目录的命中词（chip 的填充宽度 = 这个词在分数里的占比）；被开启的目录下方嵌套它的下一层 |
| **命中文档** | 卡片：`由第 N 个分支召回；命中 3/28 个查询词（llm、kids、amazon），BM25 32.2；目录 语料库 › …` + 证据段落 + 命中词。点卡片跳到「分块」模式看原文 |
| **目录树** | 整个语料的目录形状，每层带文档数条；`聚类` 标签标出这个目录是系统聚出来的、不是用户真实目录 |

通道下拉 = `fs` / `fs-only`（关兜底网）/ `fs-keyword`（关模型，纯关键词排序）；分支下拉 = 1/2/4/6（`branches`）；「解释命中词」= `explain`，默认开。

两个与命令行不同的地方：

- **目录树不会随入库自动重建**。新文档要按「重建目录树」才进树；没按时状态行会挂 `N 篇新文档未纳入目录树` 的黄标——否则路由会在一个不包含刚入库文档的树上跑，且没有任何迹象。
- 界面固定 `limit=8`、`fs.status depth=3`（目录树只画三层，再往下是两条截断标题拼的文件夹名，在这个宽度上是噪声）。

---

## 5. 改代码后怎么验证

| 改了什么 | 要重跑什么 |
| :--- | :--- |
| `internal/dirtree/dirtree.go`（建树） | 步骤 1 → 3 → 6（**必须重建树**） |
| `internal/dirtree/route.go`（路由） | 步骤 1 → 6（树不变，可不重建） |
| `cmd/freerag/dirtree.go`（RPC） | 步骤 1 → 6 |

最小回归（快，30 秒）：

```bash
CGO_ENABLED=0 go build ./... && \
python3 scripts/eval_doc_recall.py --kb eval/data-v12 -k 8 --cuts 8 \
  --channels fs,fs-keyword,fs-pure --out /tmp/quick.json
```

`fs-pure` 是诚实的指标：它涨了才是路由真的变好，`fs` 涨了可能只是兜底网兜住了。

---

## 6. 常见坑

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| `fs` 与 `fs-keyword` 数字完全一样 | laya 没起来 | 看步骤 4 的 `decider`；查 `models/laya-onnx` 与 `.venv314` |
| `fs.search` 报 "no directory tree" | 没建树 | 跑步骤 3 |
| 召回全是 ` India to be one of worlds...` 这类无关文档 | 路由走偏，`fs-pure` 本来就只有 0.09 | 看 trace 每层的 `kept`，确认是不是根层就选错 |
| `hits` 里 `source` 几乎全是 `keyword` | 兜底网在兜底 | 正常，0.743 就是这么来的；要看路由真实水平跑 `fs-pure` |
| 建树很慢 / 卡死 | 老版本有 nid 分配冲突导致死循环 | 拉最新 `internal/dirtree`（已修） |
| 只有 84/731 有向量 | 建库时托管嵌入失败 | 不影响本方案（不依赖向量） |
| `download-models.sh` 卡住 / 404 | 默认走 `hf-mirror.com`，镜像可能缺文件 | 换 `HF_ENDPOINT=https://huggingface.co`（需能直连）；只缺 laya 就 `bash scripts/download-models.sh laya` |
| `ModuleNotFoundError: onnxruntime` | venv 没装或用了系统 python | 用 `.venv314/bin/python`，按 §1.1 重装 |
| `go: ... cannot find module` | 首次构建没拉模块 / 网络不通 | `export GOPROXY=https://goproxy.cn,direct && go mod download` |
