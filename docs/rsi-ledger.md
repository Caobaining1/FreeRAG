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
