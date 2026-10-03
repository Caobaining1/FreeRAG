# 从零到一：给你的项目搭一套 RSI 自我改进系统

> 本文以 `freerag` 项目的真实实现为蓝本（`docs/plan.md §13` 设计、`scripts/rsi_*.py` 与 `scripts/ragas_eval.py` 落地、`docs/rsi-ledger.md` 记录），
> 提炼成一份**可移植**的工程教程。你不需要是 RAG 项目也能照做——把"被改进对象"换成你自己的 prompt、阈值或配置文件即可。
>
> 阅读顺序：先读 §0 弄清 RSI 的边界，再按 Phase A→F 一步步搭。每一节都给出 freerag 的**具体文件、函数、行号**，方便你对照抄。

---

## 0. 先读这一节：RSI 到底在做什么（以及不做什么）

### 0.1 是什么

**一个有界自我改进循环（Bounded Self-Improvement Loop）**：

> 在**固定的评测集**上，系统用 LLM 提出对**自身 prompt / 工具描述 / 数值阈值**的修改，
> 用评分决定**接受或回滚**，并把每次尝试（含失败）写进**账本**。
> 它的"自我"，体现在**修改的对象是它自己的指令，而不是模型权重**。

对照 `docs/plan.md:1088-1098` 的原文定义。

### 0.2 不是什么（诚实边界）

| 不是 | 说明 |
| :--- | :--- |
| 权重训练 | 不改任何模型参数，改的是文本与常量 |
| 架构自改 | 不加工具、不改控制流、不改契约 |
| 开放式自复制 | 是"有界"的，单步改进规模受**评测分辨率**限制 |
| 一步到位的自动调参 | 一次评测在此机器上要 1–2 小时，循环只能小步走 |

### 0.3 "递归"落在哪里

**递归出现在 Phase E**：前四阶段（C/D）改进的是"系统"本身；
Phase E 改进的是"**提出修改的那套规则**"——即用账本里"哪类提案真的涨分"的记录，
反过来重写**提案 prompt** 本身。**在此之前，都只是自动化调参，不叫 RSI。**

对应代码：`scripts/rsi_propose.py:69-71` 的注释——

```python
# Phase D's genome: the text half. Each entry names the Go file and the constant
# whose raw string literal is the prompt. Phase E will add "the prompt that
# writes these proposals" as an entry here — that is the recursion.
```

### 0.4 一条铁律

> **自我改进最容易失败的方式，是学会讨好尺子（reward hacking）。**
> 所以：能"为了提高分数"而被反向优化的东西（拒答行为、空池判定、引用锚定），
> 必须写成**不可改的守卫**，而不是适应度的一部分。（`plan.md:1133-1135`）

---

## 1. 核心心智模型：适应度 + 基因型 + 守卫 + 账本

搭 RSI 前，先在白纸上写下这四样东西。

### 1.1 适应度函数（fitness）

`plan.md:1100` 给出的公式：

```
fitness = 质量 − λ·成本 − M·违规
```

- **质量**：RAGAS 四指标（`faithfulness` / `answer_relevancy` / `context_precision` / `context_recall`）的**宏平均**。
- **成本**：端到端墙钟（秒/题）+ 生成 token。不加此项，循环会拿无限延时换质量。
- **违规 M**：违反任一不变量 ⇒ **直接否决**，即使质量上升。

**关键实现选择**（`rsi_sweep.py:132-139`）：把"质量"单独作为标量判决，
把**成本和守卫分离开来单独检查**，而不是折成一个数。原因写在注释里：

> 折成一个数，会让"大幅的质量提升"为一处守卫违规买单——而这正是 §13.1 明令禁止的交易。

### 1.2 基因型（genome）：什么可以被改

- **必须声明（declared），不能靠正则乱扫源码。** `rsi_sweep.py:9-12`：
  > 基因组必须被声明，因为 §13.2 同时列出了"绝不能改"的东西。
  > 一个遍历源码树的正则，迟早会改到守卫、指标里的阈值、或嵌入维度——而失败会伪装成改进。
- freerag 的基因组分两半：
  - **数值旋钮**：`KNobs` 表（`rsi_sweep.py:73-123`）
  - **文本 prompt**：`PROMPTS` 表（`rsi_propose.py:72-104`）

### 1.3 守卫（guards）：绝不能改

`plan.md:1124-1131` 列出的六条硬守卫：

1. **评测本身**——数据集、拆分、指标定义、判分模型，循环只能读。
2. **"空池从不通融"**——池为空即 INSUFFICIENT 的分支。
3. **引用与锚定**——成员必须锚定到 passage，答案只能引用证据里的 `[n]`。
4. **`DecisionKind` 契约**——路由/工具选择=`choice`、SCA=`noul`。
5. **工具面**——仍是 4 个，不允许加工具绕过检索困难。
6. **拒答行为**——拒答率是守卫，不是优化目标。

### 1.4 账本（ledger）

`docs/rsi-ledger.md`，每次提案一行：`日期 | 对象 | 修改 | 前 | 后 | 成本 | 结果 | 理由`。
**被否决的同样入账**——因为"哪些方向试过且无效"正是 Phase E 的输入（`rsi_sweep.py:5-7`）。

---

## 2. 系统总览：组件与数据流

```
┌───────────────────────────────────────────────────────────────────┐
│  被改进对象（你的项目代码）                                          │
│    数值常量  +  prompt 文本字面量                                    │
│    freerag: internal/agent/{queries,loop,synthesize,...}.go        │
└───────────────┬───────────────────────────────────────────────────┘
                │ 读/写（声明式基因组）
┌───────────────▼───────────────────────────────────────────────────┐
│  改进引擎                                                          │
│    rsi_sweep.py    Phase C：数值旋钮坐标下降                        │
│    rsi_propose.py  Phase D：LLM 改 prompt（提案≠生成器）            │
│    rsi_loop.py     Phase C+D 两阶段无人值守驱动器                    │
└───────────────┬───────────────────────────────────────────────────┘
                │ 调用
┌───────────────▼───────────────────────────────────────────────────┐
│  评测（适应度）  ragas_eval.py  freeze/index/run/score              │
│    产出 scorecard：quality_macro + 拒答率 + violations + git        │
└───────────────┬───────────────────────────────────────────────────┘
                │ 归约
┌───────────────▼───────────────────────────────────────────────────┐
│  基线/账本  rsi_baseline.py  +  docs/rsi-ledger.md                 │
└───────────────────────────────────────────────────────────────────┘
```

**一句话控制流**：`rsi_loop.py` 是总控 → 调 `rsi_sweep`/`rsi_propose` 生成候选（改源码）→ 重建 → 调 `ragas_eval.py` 跑题打分 → `decide()` 依"质量宏平均 + 拒答守卫 + 不变量"三条件接受或回滚 → 记账本 → 接受则提交、并把候选 scorecard 立为新基线。

---

## Phase A：打造可复现的评测地基

**这是全套系统里最重要、也最不能偷懒的一步。** 没有可复现的分数，后面全是猜。

### A.1 四阶段脚手架：为什么必须拆开

`ragas_eval.py:2-15` 的文档字符串说清了理由：

```
freeze   questions + the PDFs they need   -> eval/{dev,test}.json, eval/kb_sources.json
index    those PDFs through the real pipeline -> a dedicated eval KB
run      `ask` per question               -> eval/runs/<run>.jsonl
score    ragas metrics + guards           -> eval/runs/<run>.scorecard.json
```

> 一次评测 1–2 小时，**打分绝不能要求重新提问**。
> 这也正是 Phase C/D 能成立的前提：一个候选改动通过"重跑 run + score"来测量，而这两步是**分别计价**的。

要点：
- **每阶段写一个文件，供下一阶段读**（可续跑、可重打分）。
- **`run` 支持 resume**：`ragas_eval.py:256-265` 读出已答题目的 id 并跳过——
  因为内存压力会把单题从 ~140s 拖到 507s，唯一的应对是停掉重来，没有 resume 就会丢掉整轮。

### A.2 freeze：冻结分片（split）

`stage_freeze`（`ragas_eval.py:146`）做三件事：

1. **按 `question_type` 分层、按 id 排序**取题（保证可复现，`ragas_eval.py:163-185`）。
2. **null 题（不可回答）不带 PDF 过滤**直接进分片：
   > 它们按构造就不可回答，必须留在分片里，因为**拒答率**是守卫指标之一——
   > 而没有样本的守卫不是守卫。（`ragas_eval.py:170-178`）
3. **冻结 KB 源清单** `kb_sources.json`，附版本号，随代码提交。

真值来源**不自己标注**：直接复用 benchmark 的 `answer` 与 `evidence_list[].fact`（后者作为 RAGAS 的 reference context）。

### A.3 index：用真实 pipeline 建评测 KB

`stage_index`（`ragas_eval.py:208`）**不复用生产 KB**（内容随机、无真值），而是按 `evidence_pdfs`
反查出题所需的文章集合，建一个**固定、可重建**的 KB。
`--force` 参数用于重建：内容指纹不变时默认会跳过文件，导致"嵌入服务宕机后留下无向量的 chunk、混合检索静默退化为纯关键词"（`ragas_eval.py:222-227`）。

### A.4 run：驱动内核取 answer + contexts

`stage_run`（`ragas_eval.py:243`）通过 JSON-RPC 逐题调 `ask`，把每题写成一行 JSONL：

```python
record = {
    "id", "question", "type", "ground_truth", "reference_contexts",
    "answer", "contexts",          # 生成答案 + 检索到的证据片段
    "verdict", "rounds", "enumeration",
    "latency_s", "trace", "error", # 成本与可诊断性
}
```

`Kernel` 类（`ragas_eval.py:69`）用 stdio JSON-RPC 拉起内核子进程，一个进程服务整阶段。

**关键：测量时必须冻结采样。** `ragas_eval.py:78-85`：

```python
env.setdefault("FREERAG_GENERATION_TEMPERATURE", "0")
```

> 生成器出厂温度是 0.2（答案读起来更有变化）——而这点变化**比循环提出的任何改动都大**：
> 同一份有效代码的三次 dev-loop 跑出 0.3672 / 0.3735 / 0.4465，**跨度 0.079**，
> 循环据此接受过一个 +0.079 的**惰性改动**。比较无法穿透这个噪声。

### A.5 score：RAGAS 打分 + 守卫

`stage_score`（`ragas_eval.py:329`）三个关键设计：

**(1) 判分模型绝不能是生成器。**

```python
"judge_is_generator": args.judge_model == os.environ.get("FREERAG_GENERATOR_MODEL", "freerag-qwen3"),
```

`ragas_eval.py:460-465`：
> 一个 LLM 给自己的文本打分是有偏的。绝对分数会偏乐观。
> RSI 循环只需要**相对变化**，而冻结判分器 + 冻结评测集仍能测出相对变化——但这个 flag 不是 False 时，是相信任何 delta 之前**第一件要检查的事**。

**(2) `answer_relevancy` 需要嵌入端点，若没有就显式移除并记录**，不要让它静默变 NaN 被平均掉（`ragas_eval.py:372-387`）。

**(3) 指标是针对"判分器被告知的 contexts"算的**——截断会系统性压低所有指标（`ragas_eval.py:582-586` 实测：截到 5×1200 时 faithfulness 0.60→0.27）。所以要截断就必须**对所有 run 用同一套截断参数并写进 scorecard**。

**scorecard 结构**（`ragas_eval.py:455-481`）：

```json
{
  "run": "...", "kb": "eval-kb-v1",
  "judge": "...", "judge_base": "...", "judge_is_generator": false,
  "metrics_requested": [...], "n_scored": 24, "n_failed": 0,
  "metrics": {"faithfulness": ..., "context_precision": ..., "context_recall": ...},
  "judge_nan_samples": {...},          // ★ 必须看：NaN 会被从均值里丢掉，样本变少→更容易虚高
  "quality_macro": ...,                // 四指标宏平均，适应度的"质量"项
  "refusal_rate_on_null": ..., "n_null": ...,
  "violations": [...],                 // 不变量违规
  "git": "fa75917"
}
```

逐题分数另存 `<run>.scores.json`（`ragas_eval.py:486`）——**均值说不出"哪几题被改动影响"，而账本正需要这个**。

---

## Phase B：建立基线 + 标定噪声带

### B.1 基线 scorecard

跑一次完整 `run` + `score`，得到 `eval/runs/baseline.jsonl` / `.scorecard.json`。
这就是后面所有比较的锚点。

### B.2 为什么必须先测噪声带

**这是 freerag 花了最惨痛代价换来的教训**（`docs/rsi-ledger.md:21-41`）：

`defaultToolLimit` 被扫，"提升" quality_macro **+0.079** 并被接受。
但事后发现它是**死代码**（被 `DefaultLimit` 遮蔽，见 §C.3），改动**不可能产生任何效果**。
那个 +0.079 是纯粹的**运行间噪声**：

| 运行 | 有效代码 | quality_macro |
| :--- | :--- | ---: |
| `noise-a` | 未改动 | 0.3735 |
| `sweep-defaultToolLimit-10` | **等价于未改动** | **0.4465** |

差 **+0.073** ⇒ 真实运行间噪声**至少 0.073**。

**结论**：在 10 题（9 题 dev-loop）上，只有 **≥0.10~0.15** 的变化才可能被分辨。
阈值 0.05 在 dev-loop 上其实偏松，但配合"全 dev 复核"（见 §D/§E）能兜住。

### B.3 怎么测噪声带

**拿同一份代码跑两次**，取分数差。用 `rsi_baseline.py` 做归约（见下）。

### B.4 rsi_baseline.py：把 run 归约到可比子集

**问题**：Phase D 在 `eval/dev-loop.json`（10 题，~28 min）上测候选，但出厂基线是**全 dev（32 题）**。
直接比就是**在比样本量而非系统**（`rsi_baseline.py:4-8`）：

```python
# reduce_run: 只保留 split 中出现的题的逐题分数，重新求宏平均
wanted = {row["id"] for row in json.loads(split_path.read_text(...))}
pick = [(r, s) for r, s in zip(answerable, scores) if r["id"] in wanted]
```

**比较点必须是同一批题。** 这也是"噪声带"的测法：把两次相同代码的 run 都归约到同一子集再相减。

产出 `eval/runs/devloop-baseline.scorecard.json`，它就是 Phase C/D 的**比较基准**（`rsi_sweep.py:43-46`）。

---

## Phase C：数值旋钮坐标下降

**先做 C 再做 D 的理由**（`plan.md:1175-1176`）：数值扫描验证的是**整条测量链路**（噪声、方差、成本），
而它失败起来**比文本修改便宜得多**。

### C.1 声明式基因组 `KNobs`

`rsi_sweep.py:73-123`，每个旋钮声明：`file` + `pattern` + `current` + `candidates` + `why`（可读的"为什么它重要"）：

```python
KNobs = {
  "maxSearchQueries": {
    "file": "internal/agent/queries.go",
    "pattern": r"(?m)^const maxSearchQueries = (\d+)$",
    "current": 3, "candidates": [2, 4, 5],
    "why": "queries per rewrite: fewer 表示更窄的首次召回，more 表示每轮多一次检索调用",
  },
  "SCAMaxRounds": {
    "file": "internal/agent/loop.go",
    "pattern": r"(?m)^\t\tSCAMaxRounds: (\d+),$",
    "current": 3, "candidates": [2, 4],
    "why": "rounds per simple question: the whole cost curve is linear in this",
  },
  # ...
}
```

### C.2 精确匹配，不许模糊

`apply_knob`（`rsi_sweep.py:217-241`）：

```python
matches = re.findall(pattern, original)
if len(matches) != 1:
    raise SystemExit(f"refusing to edit {name}: pattern matched {len(matches)} time(s), want exactly 1")
```

> 一个"匹配 0 次"的静默 no-op 会被记成"无变化"；匹配 2 次说明周边代码变了。**两种都拒绝，而不是猜。**

### C.3 `also_update`：同步更新测试 pin

有些旋钮被回归测试钉住（如 `internal/agent/loop_test.go` 钉 `SnippetsPerQuery = 6`）。
**扫这个旋钮必须同步改 pin**，否则否决后回滚只回滚旋钮，留下一个断言"被否决的值"的失败测试（`rsi_sweep.py:108-121`）：

```python
"also_update": [{"file": "internal/agent/loop_test.go",
                 "pattern": r"(spec\.SnippetsPerQuery != )(\d+)",
                 "replacement": r"\g<1>{value}"}],
```

> 测试 pin 的职责是"抓意外漂移"，不是"禁止有实测依据的改动"——所以扫描时**更新 pin**，测试仍在守护其它所有字段。

### C.4 `INERT_KNOBS`：死旋钮表

`rsi_sweep.py:55-59`。记录"看起来可改、实际不可能生效"的旋钮，**脚本拒绝执行**：

```python
INERT_KNOBS = {
  "defaultToolLimit":
      "shadowed by Toolbox.DefaultLimit, which every production construction sets "
      "from Spec.SnippetsPerQuery (loop.go:754, cmd/freerag/kb.go:132)",
}
```

> 教训：一个旋钮被扫之前必须**先证明它是活的**，这张表就是记录这个的地方。

### C.5 应用 / 回滚 / 重建 / 测量

`_TOUCHED`（`rsi_sweep.py:202`）记录每个被改写文件的原文，**回滚是精确的、最新的先还原**：

```python
def _write(path, text):
    _TOUCHED.append((path, path.read_text(encoding="utf-8")))  # 先存原文
    path.write_text(text, encoding="utf-8")

def restore():
    while _TOUCHED:
        path, original = _TOUCHED.pop()
        path.write_text(original, encoding="utf-8")
```

一轮迭代（`run_one`, `rsi_sweep.py:278-308`）：

```
apply_knob(改源码) → rebuild(go build) → measure(run+score) → accepts(判决)
   ├─ 接受：文件保留 → 候选 scorecard 覆盖 BASELINE_SCORECARD
   └─ 否决：restore() → rebuild() 精确回滚
```

### C.6 成本

单次迭代 ≈ 10 题 × ~140s 问答 + ~5 min 判分 ≈ **28 分钟**（`rsi_sweep.py:19-22`）。
这就是循环用 dev-loop(10) 而非全 dev(32) 的原因。

---

## Phase D：让系统改自己的 prompt（真正的自我改进）

### D.1 声明式文本基因组 `PROMPTS`

`rsi_propose.py:72-104`，每个 prompt 声明所在 Go 文件与用途：

```python
PROMPTS = {
  "queryRewriteSystemPrompt": {"file": "internal/agent/queries.go",
                               "why": "turns the question into round 1's queries; feeds every retrieval"},
  "synthesizeSystemPrompt":   {"file": "internal/agent/synthesize.go",
                               "why": "writes the final answer of the complex path from the merged pool"},
  # ...
}
```

`synthesizeSystemPrompt` 曾**缺席**这张表，代价巨大（`rsi_propose.py:87-93`）：
复杂路径的最终答案由它写（24 道可回答 dev 题里有 19 道），而循环花了三轮改那些"输出根本不被复杂路径使用"的 prompt。

### D.2 关键：给提案模型看的是"可归因的失败样本"

`worst_cases`（`rsi_propose.py:146-225`）**不是简单取分数最低的题**，而是先做归因：

1. **算证据覆盖率**：用 ground truth 的 evidence 句子，对检索到的 passages 做 **5-gram 覆盖率**（复用 `evidence_check.py` 的 `text_coverage`）。
2. **剔除检索失败样本**：覆盖率 < `RETRIEVAL_FLOOR = 0.15` 的题，**prompt 再怎么改也救不了**。
   > 全局最差的题被检索失败主导，把它们喂给提案器，它**正确地拒绝了三次**，而 prompt 杠杆从未被用在它真正能撬动的题上。（`rsi_propose.py:165-171`）
3. **只保留"证据已检索到、答案仍答错"的题**（覆盖率 ≥ `PROMPTABLE_FLOOR = 0.5`），这才是 prompt 的锅。
4. **按证据覆盖率给 passages 排序**，展示"承载证据的段落"而不是"检索返回的前三段"：
   > 池子被污染后，检索顺序不是相关性顺序——坏 run 的前几段是垃圾，证据藏在深处。（`rsi_propose.py:201-211`）

### D.3 提案 ≠ 生成器

`rsi_propose.py:15-22`：

> 提案器跑在**托管模型**上，**不是生成器**：一个模型改另一个模型的 prompt、又给它打分，正是 §13.1 警告的自偏好环。

`propose()`（`rsi_propose.py:228-291`）用**纯 urllib**走 OpenAI 兼容 HTTP，`temperature=0`，**故意不引 SDK**：
> 这只需要一次 chat completion，引 SDK 只会把 RSI 循环绑死在评测 venv 上。

**提案器的系统提示 `PROPOSER_SYSTEM`**（`rsi_propose.py:106-128`）——这是 Phase E 递归要改的对象，值得逐条背下来：

```text
You improve ONE prompt in a retrieval-augmented answering system.
...Propose the SMALLEST edit to that prompt that would plausibly fix those cases.
Rules:
- Change as little as possible. One clause or one rule, not a rewrite. 大改动无法归因。
- Keep every existing rule that is not contradicted by the failures. 规则都是为特定失败存在的。
- Do not add facts about the corpus, do not name specific answers, and do not mention the evaluation.
- 每个 case 都说明了检索到的段落**确实包含**期望证据……材料就在写作者眼前，回复却错过了它。
- 只有"这个 prompt 怎么改都不可能影响结果"时才设 no_edit: true——不是因为难，不是因为检索不完美。
- The prompt is a raw Go string literal: no backticks, no unescaped double quotes inside it.
Reply with JSON only: {"no_edit": false, "rationale": "...", "new_prompt": "..."}
```

### D.4 提案先落盘，再应用

`rsi_propose.py:314-336`：

> 提案在**应用前**写入 `rsi-<prompt>.proposal.json`，已有测量时**复用记录中的提案**而不是重新提案
> （提案模型即使 `temperature=0` 也非确定）。否则重新运行会重新提案，判决就被记在"从未被跑过的文本"上——
> 第一次 synthesis 判决曾被记成 "+205 chars"，而实际被测的是 +99 字符的版本。
> **一份不描述实验的记录比没有记录更糟，因为它被信任。**

### D.5 回滚用 git，不用文本撤销

`rsi_propose.py:20-22`：

> prompt 编辑是多行字符串字面量，**部分回滚**会把源码留在"既不是旧 prompt、也不是新 prompt"的状态。

所以 D 阶段一律 `git checkout -- <file>`（`rsi_propose.py:354, 360, 389, 393`）。

### D.6 应用 / 重建 / 测量 / 判决

`run_once`（`rsi_propose.py:307-394`）流程：

```
read_prompt → worst_cases → propose → 写 proposal.json
  → 应用(替换字符串字面量) → go build
      ├─ 失败：git checkout 回滚
      └─ 成功：measure(run+score) → accepts()
            ├─ 接受：保留
            └─ 否决：git checkout + rebuild
```

---

## 判决规则：整套系统的心脏

判决逻辑在 `rsi_sweep.py:163-194`（`accepts`）和 `rsi_loop.py:99-116`（`decide`），**两处必须一致**（注释明说"共享同一规则，好让两个阶段不会漂移"）。

### 规则：三条件 + 一条测量完整性

```python
def accepts(candidate, baseline, noise) -> (bool, reason):
    # 0. 测量完整性：两次 run 必须在同一批题上打分
    incomplete = measurement_is_whole(candidate, baseline)
    if incomplete: return False, incomplete

    # 1. 提升必须超过噪声带
    if after < before + noise:
        return False, f"提升 {after-before:+.3f} 未超过噪声带 {noise:.3f}"

    # 2. 拒答率不得下降（守卫）
    if n_null >= MIN_NULL_FOR_GUARD and refusal_after < refusal_before:
        return False, "拒答率下降（守卫项，不允许用来换分）"

    # 3. 不得新增不变量违规
    if new_violations:
        return False, f"新增不变量违规：{...}"

    return True, f"提升 {after-before:+.3f} ≥ 噪声带 {noise:.3f}，守卫无回归"
```

### 为什么需要 `measurement_is_whole`

`rsi_sweep.py:144-160`：

> 一次判分调用失败 ⇒ 该指标为 NaN ⇒ 样本被从均值里**丢掉**。
> 于是"判分失败更多"的 run，宏平均是在一个**更小、更简单**的样本集上算的，可能因此**更高**。
> 实测：候选有 3 个 NaN `context_precision`，基线有 1 个，宏平均动了 +0.054——从这两个数**根本不可判定**。
> 任何越界都**拒绝判决**，而不是缩小样本。

### `MIN_NULL_FOR_GUARD = 4`

`rsi_sweep.py:61-64`：null 题少于 4 道时，拒答率不可比（dev-loop 只带 1 道，取值只能是 0.0 或 1.0）。
低于阈值时，守卫**记为"未测"而非"通过"**——这是**有记录的弱化，不是静默的**。

---

## Phase E：递归——改进"提出修改的规则"

**这一步才让 RSI 配得上"递归"二字。**

机制极简：**把提案 prompt 本身也加入 `PROMPTS` 表**。

`rsi_propose.py:69-71` 已埋好钩子：

```python
# Phase E will add "the prompt that writes these proposals" as an entry here — that is the recursion.
```

具体做法：

1. **账本积累信号**：`docs/rsi-ledger.md` 里每条记录的"结果 + 理由"。
   统计"哪类提案（改哪条规则、加约束 / 删约束 / 举例）真的涨分"。
2. **用这些统计反向重写 `PROPOSER_SYSTEM`**（`rsi_propose.py:106`）。
3. 把 `PROPOSER_SYSTEM` 作为一个新条目加入基因组，用同一套 `propose → 应用 → 测量 → 判决` 流程去改进它。
4. **度量递归效果**：提案命中率（提案被接受的比例）的前后对比（`plan.md:1172`）。

**注意**：Phase E 的信噪比更低，迭代更贵，务必先把 C/D 跑稳、账本积累到几十条记录再开。

---

## Phase F：守卫回归自动化

`plan.md:1173`：**一次"为涨分而违规"的提案被自动否决的实例**，是这一阶段的交付。

守卫分两类：

**(1) 能在评分时机械检测的**（写进 `ragas_eval.py` 的 `detect_violations`, `ragas_eval.py:516-529`）：

```python
# "An empty pool is never sufficient":
if not record.get("contexts") and len(record.get("answer", "")) > 200:
    violations.append(f"{record['id']}: answered at length from an empty pool")
```

> 故意少而机械；需要判断力的（引用锚定、锚定）由 Go 测试套件覆盖——
> **循环在获准打分任何东西之前，会先跑测试套件作为闸门。**（`ragas_eval.py:519-522`）

**(2) 语义化的守卫**：拒答率原用正则匹配措辞，`refusal_check.py` 改为让判分模型**语义判定**更强的拒答——
正则太窄会漏判，太宽会误判（`plan.md:1235-1236`）。

**判决时**：候选出现了基线没有的 violation ⇒ 直接否决（见上面 `accepts` 第 3 条）。

---

## 无人值守循环：状态机（`rsi_loop.py`）

### 两阶段：筛选 ≠ 接受

`rsi_loop.py:1-24` 的文档字符串是整套系统最关键的设计说明：

```
Two stages per candidate:
  screen   dev-loop (10 questions) at temperature 0, ~45 min. 便宜，只决定"值不值得跑贵的那阶段"。
           筛过 ≠ 接受。
  decide   FULL dev split (32 questions) at SHIPPED temperature 0.2, ~3h. 这才是判决，
           也是唯一能测出拒答守卫的地方。
```

**为什么必须两阶段**（`docs/rsi-ledger.md:62-66`）：
`SnippetsPerQuery 6→10` 在 10 题 dev-loop 上 +0.060 被接受，
但全 dev 32 题 @0.2° 复核后 **−0.057，且拒答率 0.75→0.625（守卫失败）**，回滚。

### 阈值

```python
SCREEN_DELTA   = 0.05   # dev-loop 实测噪声带 0.0271，取 0.05 ≈ 两个带宽
VALIDATE_DELTA = 0.05   # 全 dev 的噪声带没测过（再跑一次要 3h），故意保守
SHIPPED_TEMPERATURE = "0.2"
```

### 状态机三条铁律

`rsi_loop.py:14-20`：

> - **工作树永远等于"已接受"的状态**：被否决的候选在下一个候选应用前回滚，被接受的立即提交。
> - **两个比较 scorecard 永远描述"已接受"的状态**，接受时用候选自己的测量重写它们。
> - **每个结果都进账本**，包括跑失败的。

### 启动前置条件

```python
if not tree_is_clean():
    log("refusing to start: the working tree is not clean, so a revert would not be exact")
    return 1
```

> 工作区不干净就拒绝启动——因为"回滚"将不再精确。（`rsi_loop.py:248-250`）

### 接受时的动作

```python
shutil.copy(screen_scorecard, BASELINE_SCORECARD)      # 新的比较点
shutil.copy(validate_scorecard, baseline.scorecard.json)
commit(f"Accept {name} {current} -> {value} ...", [改的文件, ledger])
```

### 异常隔离

单个候选抛异常**绝不能终止整轮**（`rsi_loop.py:262-266`）：

```python
except Exception as exc:
    append_ledger([... 跳过 ... f"{type(exc).__name__}: {str(exc)[:100]}"])
    subprocess.run(["git", "checkout", "--", "."])   # 清理
```

---

## 血泪教训清单（直接抄，能救你几天时间）

| # | 教训 | freerag 出处 |
| :--- | :--- | :--- |
| 1 | **先测噪声带**。同代码跑两次的差值，就是你系统的分辨率下限。 | `rsi-ledger.md:29-41` |
| 2 | **测量必须冻结采样**（温度 0）。出厂温度的变化宽度可能大于任何提案的效果。 | `ragas_eval.py:78-85` |
| 3 | **单阶段不够**。筛选便宜但会误接受，必须用全量 + 出厂温度复核。 | `rsi_loop.py:4-12` |
| 4 | **判分器 ≠ 生成器**。并在 scorecard 里写 `judge_is_generator`，判决前先看它。 | `ragas_eval.py:460-465` |
| 5 | **看 `judge_nan_samples`**。NaN 被丢出均值，样本变少会让分数虚高。 | `rsi_sweep.py:144-160` |
| 6 | **基因组必须声明**，不能用正则乱扫源码，否则会改到守卫。 | `rsi_sweep.py:9-12` |
| 7 | **模式必须精确匹配一次**。0 次是静默 no-op，2 次是代码漂移，都拒绝。 | `rsi_sweep.py:224-225` |
| 8 | **死旋钮表**。旋钮被扫前先证明它是活的；记录证明过的死旋钮，脚本拒绝执行。 | `rsi_sweep.py:55-59` |
| 9 | **扫旋钮要同步更新测试 pin**，否则否决后留下"断言被否决值"的红树。 | `rsi_sweep.py:108-121` |
| 10 | **提案先落盘再应用**。提案模型非确定，否则判决记在没跑过的文本上。 | `rsi_propose.py:314-336` |
| 11 | **喂给提案器的失败样本要先归因**，剔除检索失败——否则它只会正确地拒绝。 | `rsi_propose.py:165-195` |
| 12 | **展示承载证据的段落，而非前几段**。池子污染后检索顺序≠相关顺序。 | `rsi_propose.py:201-211` |
| 13 | **prompt 回滚用 git checkout**，多行字面量的部分撤销会留下第三种状态。 | `rsi_propose.py:20-22` |
| 14 | **错误基线之所以没被发现，是因为两种算法判决恰好相同**——判决正确不会暴露错误的比较。 | `rsi-ledger.md:15-17` |
| 15 | **测量工具的默认值会伪装成能力**。上下文截断曾系统性压低四指标 0.20。 | `plan.md:1593` / `ragas_eval.py:582-586` |
| 16 | **守卫要语义化**。正则判拒答太窄会漏、太宽会误。 | `plan.md:1626` |
| 17 | **工作树不干净就拒绝启动**，否则回滚不精确。 | `rsi_loop.py:248-250` |
| 18 | **报告"未测"而不是"通过"**。null 题 < 4 道时守卫记 unmeasured。 | `rsi_sweep.py:61-64` |
| 19 | **拒绝判决而不是缩小样本**。测量不完整时绝不接受。 | `rsi_sweep.py:152-160` |
| 20 | **一个候选的异常不能终止整轮**。 | `rsi_loop.py:262-266` |

---

## 移植到你自己项目的检查清单

### 第 0 步：判断你适不适合做 RSI

- [ ] 你的项目瓶颈是**"决策质量"**（prompt、策略、阈值），而不是解析/IO 速度？
- [ ] 这些决策的载体**全是文本与常量**（可以在不训练权重的前提下改）？
- [ ] 你有**带真值的评测集**，能算出可复现的分数？
- [ ] 你能接受**单次迭代几十分钟到几小时**的成本？

四条都满足，才值得做。（对应 `plan.md:1084-1086`）

### 最小可用骨架（MVP）

照着 freerag 的四个脚本各写一个精简版：

```
your-project/
├── eval/
│   ├── dev.json / dev-loop.json        # 冻结的评测分片（含 null 题守卫）
│   ├── runs/                           # <run>.jsonl / .scorecard.json / .scores.json
│   └── kb_sources.json                 # 冻结的语料清单（如有 KB）
├── scripts/
│   ├── eval_runner.py     # freeze/index/run/score 四阶段   ← 抄 ragas_eval.py
│   ├── rsi_baseline.py    # 归约到可比子集                  ← 抄 rsi_baseline.py
│   ├── rsi_sweep.py       # 数值旋钮：KNobs + apply/restore/rebuild/measure/accepts
│   ├── rsi_propose.py     # 文本 prompt：PROMPTS + worst_cases + propose
│   └── rsi_loop.py        # 两阶段无人值守驱动器              ← 抄 rsi_loop.py
└── docs/
    └── rsi-ledger.md      # 账本
```

### 落地顺序（严格按此序）

1. **A**：搭好四阶段评测，**先能出一条可复现的 scorecard**。
2. **B**：跑基线；**跑两次测噪声带**；用 baseline 归约脚本定出比较点。
3. **C**：声明 2–3 个**活**旋钮（先证明它们生效），跑坐标下降，拿到**一次被接受的改动**。
4. **D**：把 1–2 个 **"证据已在眼前、答案仍错"** 的 prompt 列入基因组，让 LLM 提案，跑通"提案→重建→测量→判决→回滚"。
5. **F**：把不变量写成可机械检测的 `detect_violations`，让判决自动否决违规。
6. **E**：账本攒够几十条后，**再把提案 prompt 加入基因组**，开始递归。

### 阈值怎么定

- `noise`（接受阈值）：**测出来的**，不是拍的。等于"同代码两次运行的分差"（保守取 2 倍带宽）。
- `MIN_NULL_FOR_GUARD`：你的守卫样本少于此数就记"未测"。
- 筛选阈值 ≥ 噪声带；判决阈值**故意保守**（freerag 用 0.05，因为全 dev 噪声带没测过）。

---

## 附：核心文件速查（freerag）

| 作用 | 路径 |
| :--- | :--- |
| 设计与原则（§13） | `docs/plan.md:1082` 起 |
| 账本（含血泪勘误） | `docs/rsi-ledger.md` |
| 评测四阶段脚手架 | `scripts/ragas_eval.py` |
| 基线归约 | `scripts/rsi_baseline.py` |
| 数值旋钮坐标下降 | `scripts/rsi_sweep.py` |
| LLM 改自身 prompt | `scripts/rsi_propose.py` |
| 无人值守两阶段循环 | `scripts/rsi_loop.py` |
| 失败归因（检索侧 vs 生成侧） | `scripts/evidence_check.py` |
| 语义拒答守卫 | `scripts/refusal_check.py` |
| 被改进的 prompt 所在 | `internal/agent/{queries,loop,synthesize,decompose,enumerate}.go` |

---

## 一句话总结

**RSI = 一个冻结的尺子 + 一个声明式的基因组 + 一条"超噪声带且守卫不回归"的判决规则 + 一本连失败都记的账本 + 一个能把工作树始终拉回"已接受"状态的循环。**

尺子（评测）是可复现的，基因组是声明式的，判决是保守的，账本是完整的，回滚是精确的——这五条里任何一条松了，循环就会开始**讨好尺子**，而那是它失败的唯一方式。
