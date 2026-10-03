# RSI 账本（docs/plan.md §13.6）

每次提案一行；**被否决的同样入账**——"哪些方向试过且无效"是 Phase E 的输入。

| 日期 | 对象 | 修改 | 前 | 后 | 成本 | 结果 | 理由 |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| 2026-10-02 | `queryRewriteSystemPrompt` | +140 chars: The failures are all questions comparing claims across two named articles, and t | 0.4734 | 0.4041 | 48 min | 否决 | 提升 -0.069 未超过实测噪声带 0.050 |

> **更正（2026-10-02，第一次迭代那行）**：该行"前"列记的 `0.4734` 是**全 dev（24 题）**的宏平均，
> 而候选测在 **dev-loop（9 题）**上——`run_once` 里残留了一处硬编码路径，判决与记账都用了错的那一侧。
> **正确的前值 = `0.3672`**（`devloop-baseline.scorecard.json`）。
> **判决不变**：正确的 Δ = **+0.0369** < 阈值 0.05，仍是**否决**。
> 已修（改读 `BASELINE_SCORECARD`）。
>
> 这次事故值得记下：**错误基线之所以没被发现，是因为两种算法给出的判决恰好相同**
> （−0.069 与 +0.037 都低于阈值）。若候选的提升落在 0.037~0.069 之间，它就会被误判——
> 而"判决正确"这个结果本身不会暴露它。
| 2026-10-02 | `rewriteSystemPrompt` | — | — | — | — | 无提案 | The failures are due to retrieval returning irrelevant documents and the downstream answer generator failing to synthesi |
| 2026-10-02 | ~~`defaultToolLimit`~~ | ~~→ 10~~ | ~~0.3672~~ | ~~0.4465~~ | 55 min | **无效（误接受）** | 该旋钮是死代码，见下方勘误 |

> **勘误（2026-10-02，第二次迭代那行——原判"接受"，现判"无效"）**
>
> **该改动不可能产生任何效果。** `defaultToolLimit` 是 `Toolbox.limit()` 的**末位兜底**：
> 参数 → `DefaultLimit` → 常量。而两条生产路径**都设了** `DefaultLimit`：
> `loop.go:754`（`l.spec().SnippetsPerQuery`）与 `cmd/freerag/kb.go:132`（`Medium().SnippetsPerQuery`），
> 都等于 6。因此 `return defaultToolLimit` **不可达**，改它等于没改。
> 常量已还原为 6。
>
> **那个 +0.079 是什么**：同一份有效代码、同为独立 dev-loop 跑两次：
>
> | 运行 | 有效代码 | `quality_macro` |
> | :--- | :--- | ---: |
> | `noise-a`（样本#2） | 未改动 | 0.3735 |
> | `sweep-defaultToolLimit-10` | **行为等价于未改动** | **0.4465** |
>
> 差 **+0.073**，即**真实的运行间噪声至少 0.073**，而 §13.16 记的 0.0063 只是两次样本的巧合，
> **低估了约 12 倍**。
>
> **因此两次判决都要作废**：本次"接受"（+0.079）与被否决的 D #1（+0.037）**都在噪声内**。
> 在 9 题 dev-loop 上，只有 **≥0.10~0.15** 的变化才可能被分辨。已据此改脚本（禁止扫死旋钮）
> 与会改噪声带估计（`docs/plan.md` §13.17）。
| 2026-10-02 | `SnippetsPerQuery` | → 10 | 0.3691 | 0.4293 | 49 min | **接受** | 提升 +0.060 ≥ 噪声带 0.050，守卫无回归（拒答守卫未测：只有 1 道 null 题） |
<<<<<<< HEAD
| 2026-10-02 | `decomposeSystemPrompt` | — | — | — | — | 无提案 | The retrieved passages in the failing cases do not contain the expected source articles (e.g., 'The Age', 'Fortune', 'Th |
| 2026-10-02 | `rewriteSystemPrompt` | — | — | — | — | 无提案 | The retrieved passages in the failing cases are mostly irrelevant to the questions (e.g., sports, antitrust, unrelated n |
| 2026-10-02 | `synthesizeSystemPrompt` | — | — | — | — | 无提案 | The retrieved passages in the failing cases are mostly irrelevant to the questions (e.g., cricket schedules, stadium cha |
| 2026-10-02 | `synthesizeSystemPrompt` | — | — | — | — | 无提案 | The failures are not caused by the prompt: the retrieved passages do not contain the expected evidence (context_sample s |
| 2026-10-02 | `synthesizeSystemPrompt` | — | — | — | — | 无提案 | The failures are dominated by retrieval problems (context_precision and context_recall are 0 or near 0, and the context  |
| 2026-10-02 | `synthesizeSystemPrompt` | +99 chars: The failures show the model answers the sub-question's implied comparison instea | 0.4293 | 0.3745 | 0 min | 否决 | 提升 -0.055 未超过实测噪声带 0.050 |

> **勘误（2026-10-02，`synthesizeSystemPrompt` 那行）**：该行原先记的描述是 **"+205 chars"**，
> 那是**重跑时新生成的提案**，而**实际被测的是 +99 字符的版本**
> （"Answer the original question's exact wording and intent, not the sub-questions' framing
> or any implied yes/no."）。**数值与判决无误**（0.4293 → 0.3745，否决 ✓ 干净测量：NaN 0/0）。
> 已修描述，并在机制上堵住：提案在**应用前**写入 `rsi-<prompt>.proposal.json`，
> 已有测量时**复用记录中的提案**而不是重新提案（提案模型即使 temperature=0 也非确定）。
>
> **判定**：这条 prompt 改动**确实让结果变差**（四项指标全降或持平），
> 说明"要求答原问题"这个方向不对——**加规则不等于改善了行为**。
> 这是第一次在**干净的测量**下被否决的 prompt 改动，也是基因组扩展后第一次真实 D 迭代。

> **勘误（2026-10-02，`SnippetsPerQuery` 那行）**：该行原判 **"接受"**，**复核后回滚**。
> 全 dev 32 题 @0.2°：`quality_macro` **0.4734 → 0.4165（−0.057）**，
> 且**拒答率 0.75 → 0.625**（守卫失败）。dev-loop 的 +0.060 不可复现。
> 见 `docs/plan.md` §13.21：**dev-loop 接受只是筛选，判决需全 dev @0.2°**。
> 回滚提交 `37e44b2`。
=======
>>>>>>> parent of a005ab6 (Refuse grep probes built from words that match the whole corpus)
| 2026-10-03 | `maxSearchQueries` → 2 | 筛选 | 0.3691 | 0.3576 | 止 | 筛选: 提升 -0.0115 未超过阈值 0.050 |
| 2026-10-03 | `maxSearchQueries` → 4 | 筛选 | 0.3691 | 0.4558 | 过 | 筛选: +0.0867 ≥ 0.050（拒答守卫未测：1 道 null 题） |
| 2026-10-03 | `maxSearchQueries` → 4 | 判决 | 0.4734 | 0.4753 | 否决 | 判决: 提升 +0.0019 未超过阈值 0.050 |
| 2026-10-03 | `maxSearchQueries` → 5 | 筛选 | 0.3691 | 0.3671 | 止 | 筛选: 提升 -0.0020 未超过阈值 0.050 |
| 2026-10-03 | `enumMaxRounds` → 3 | 筛选 | 0.3691 | 0.4091 | 止 | 筛选: 提升 +0.0400 未超过阈值 0.050 |
| 2026-10-03 | `DefaultMaxStateChars` → 4000 | 筛选 | 0.3691 | 0.3994 | 止 | 筛选: 提升 +0.0303 未超过阈值 0.050 |
| 2026-10-03 | `DefaultMaxStateChars` → 6000 | 筛选 | 0.3691 | 0.3968 | 止 | 筛选: 提升 +0.0277 未超过阈值 0.050 |

> **决定（2026-10-03，用户明示）**：以下两处**按用户决定发布**，尽管守卫与判决给出了相反结论。
>
> | 改动 | 循环的结论 | 用户决定 | 已知代价（未变） |
> | :--- | :--- | :--- | :--- |
> | `greppableTerms`（拒绝无区分度 grep） | 拒答守卫拦下 | **发布** | 语义拒答率 0.875→0.625；q0150、q0213 开始编造答案 |
> | `Spec.SnippetsPerQuery 6→10` | 全 dev 判决 −0.057、拒答 0.75→0.625，已回滚 | **发布** | 充分性更易被判"够了"→ 少拒答；dev-loop 的 +0.060 未复现 |
>
> **两点如实记录**：① 这两处**从未一起测过** ✗——各自单独测时都在不可接受的方向上；
> ② 二者的作用方向都指向"更愿意回答"，叠加后对**不可答题的拒答行为**的影响大于任一单独测得的幅度 ✗。
> 若日后要评估，应直接测**这两处合并后**的全 dev @0.2°，而不是分别测。
| 2026-10-03 | `maxSearchQueries` → 2 | 0.6813 | 0.2959 | 否决 | measurement incomplete: answer_relevancy lost 17 sample(s) to judge failures against the baseline's 0; re-score before deciding |

> **撤回（2026-10-03）**：本轮**没有任何**"两处发布的 +0.2079"这类结论 ✗ ——
> 那份测量（`ship-baseline`）是在**生成器（Ollama）不可用**的状态下跑的 ✗：
> 24/24 答案恰好 6000 字、每题 7 秒 = 内核退回的**摘录式草稿** ✗。
> **两处发布改动的真实效果至今未测** ✓（待生成器恢备后重测 ✓）。
> 详见 `docs/plan.md` §13.27。
