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
| 2026-10-02 | `defaultToolLimit` | → 10 | 0.3672 | 0.4465 | 55 min | **接受** | 提升 +0.079 ≥ 噪声带 0.050，守卫无回归（拒答守卫未测：只有 1 道 null 题） |
