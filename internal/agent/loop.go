package agent

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"

	"freerag/internal/store"
)

// Spec is the mode table row this loop runs (docs/plan.md §6.1).
type Spec struct {
	Label            string
	EnableSCA        bool
	SCAMaxRounds     int
	UseFanout        bool
	ActionMaxTurns   int
	SnippetsPerQuery int
	Tools            []string
}

// Medium is the medium-mode spec: agentic with SCA review, but no planner and
// no prefetch fan-out (docs/plan.md §6.2).
func Medium() Spec {
	return Spec{
		Label:            "medium",
		EnableSCA:        true,
		SCAMaxRounds:     3,
		UseFanout:        false,
		ActionMaxTurns:   8,
		SnippetsPerQuery: 6,
		// The four retrieval tools of §6.7: nothing that needs a compiled
		// structure (no navigate_*, no graph_explore) and no web search.
		Tools: ToolNames(),
	}
}

// DefaultMaxDraftChars caps the draft handed to the checker.
const DefaultMaxDraftChars = 6000

// DefaultContextTokens is the context window the kernel asks the generator for.
//
// docs/plan.md §5 budgets 4K–8K; the top of that range is the useful default
// because the draft prompt carries the evidence block.
const DefaultContextTokens = 8192

// draftPromptReserveTokens covers the system prompt, the question, and the room
// the answer itself needs.
const draftPromptReserveTokens = 768

// charsPerToken converts tokens to characters for the prompt budget.
//
// Deliberately conservative: overestimating the token count only wastes a
// little room, while underestimating produces an oversized prompt that the
// server rejects outright. English prose runs nearer 4, technical text with
// formulas and identifiers lower.
const charsPerToken = 3

// PromptCharBudget returns how many characters of evidence fit in a context
// window alongside the system prompt and the answer.
func PromptCharBudget(contextTokens int) int {
	if contextTokens <= 0 {
		contextTokens = DefaultContextTokens
	}
	usable := contextTokens - draftPromptReserveTokens
	if usable < 512 {
		usable = 512
	}
	return usable * charsPerToken
}

// Result is one loop run's outcome.
type Result struct {
	Question string      `json:"question"`
	Mode     string      `json:"mode"`
	Rounds   int         `json:"rounds"`
	Draft    string      `json:"draft"`
	Verdict  Verdict     `json:"verdict"`
	Missing  []string    `json:"missing,omitempty"`
	Evidence []store.Hit `json:"evidence"`
	Queries  []string    `json:"queries"`
	Trace    []string    `json:"trace"`
}

// EvidenceIDs returns the chunk identities behind the answer, for citations.
func (r *Result) EvidenceIDs() []string {
	ids := make([]string, 0, len(r.Evidence))
	for _, hit := range r.Evidence {
		ids = append(ids, hit.Chunk.ChunkID)
	}
	return ids
}

// Loop is the medium-mode research loop.
//
// Model may be nil: without a generating model the loop still runs, drafting
// extractively and rewriting from the checker's missing terms. That keeps the
// graph exercised in tests and in a model-less desktop install.
type Loop struct {
	Store *store.Store
	// Tools executes the retrieval tool calls; nil builds a default Toolbox
	// over Store (docs/plan.md §6.7).
	Tools   *Toolbox
	Model   Model
	Checker Checker
	Spec    Spec
	Logger  *log.Logger

	// MaxDraftChars caps the draft; <=0 selects DefaultMaxDraftChars.
	MaxDraftChars int
	// MaxPromptChars caps the evidence block sent to the model; <=0 selects
	// PromptCharBudget(DefaultContextTokens). Keep it consistent with the
	// generator's context window — an oversized prompt is rejected, not
	// truncated, by Ollama.
	MaxPromptChars int

	// OnStep, when set, receives every trace line as it is produced.
	//
	// The loop runs for minutes (measured 50–280s), so a caller that shows
	// progress needs to see it advancing: a spinner that never changes is
	// indistinguishable from a hang, and this is the only signal that changes
	// during a round.
	OnStep func(line string)
}

// step appends one trace line and reports it when OnStep is wired.
//
// Every trace append goes through here. Notifying at each call site instead
// would work until someone adds a step and forgets — and a missing progress
// event is invisible in a way a missing line in the result is not.
func (l *Loop) step(trace *[]string, line string) {
	*trace = append(*trace, line)
	if l.OnStep != nil {
		l.OnStep(line)
	}
}

func (l *Loop) spec() Spec {
	if l.Spec.SCAMaxRounds <= 0 && l.Spec.ActionMaxTurns <= 0 {
		return Medium()
	}
	return l.Spec
}

func (l *Loop) checker() Checker {
	if l.Checker != nil {
		return l.Checker
	}
	return CoverageChecker{}
}

func (l *Loop) logf(format string, args ...any) {
	if l.Logger != nil {
		l.Logger.Printf(format, args...)
	}
}

func (l *Loop) maxDraftChars() int {
	if l.MaxDraftChars > 0 {
		return l.MaxDraftChars
	}
	return DefaultMaxDraftChars
}

func (l *Loop) maxPromptChars() int {
	if l.MaxPromptChars > 0 {
		return l.MaxPromptChars
	}
	return PromptCharBudget(DefaultContextTokens)
}

// Run executes the loop: gather evidence, draft, check, and rewrite while the
// verdict stays unsatisfied and rounds remain.
func (l *Loop) Run(ctx context.Context, question string) (*Result, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return nil, fmt.Errorf("agent: question is required")
	}
	if l.Store == nil {
		return nil, fmt.Errorf("agent: store is required")
	}

	spec := l.spec()
	checker := l.checker()

	result := &Result{Question: question, Mode: spec.Label}
	if !spec.EnableSCA {
		spec.SCAMaxRounds = 1
	}

	queries := []string{question}
	seen := map[string]bool{}
	var missing []string

	for round := 1; round <= spec.SCAMaxRounds; round++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		result.Rounds = round
		result.Queries = queries

		calls := l.planTools(ctx, question, queries, missing, result.Evidence, &result.Trace)
		added := l.runTools(ctx, calls, &result.Evidence, seen, &result.Trace)
		l.step(&result.Trace, fmt.Sprintf(
			"[RAGAgent] Round %d: +%d passage(s); pool now %d.",
			round, added, len(result.Evidence)))

		result.Draft = l.draft(ctx, question, result.Evidence)

		verdict, gaps := checker.Check(ctx, question, result.Draft, result.Evidence)
		result.Verdict = verdict
		result.Missing = gaps
		missing = gaps
		l.step(&result.Trace, fmt.Sprintf(
			"[SCA] Round %d verdict=%s (missing=%d).", round, verdict, len(gaps)))

		if verdict == VerdictSufficient {
			return result, nil
		}
		if round == spec.SCAMaxRounds {
			break
		}

		next := l.rewrite(ctx, question, missing)
		if len(next) == 0 {
			l.step(&result.Trace, "[QueryRewriter] no new angle; stopping.")
			break
		}
		queries = next
	}

	return result, nil
}

// maxGrepLegs caps how many literal probes one round may issue for the terms
// the checker reported missing. Grep exists for exactly that case — a missing
// term is usually an identifier the keyword index could not rank — but it must
// not crowd out the hybrid leg.
const maxGrepLegs = 2

// roundCalls builds one round's tool calls: a hybrid search per query, plus a
// literal probe per missing term (docs/plan.md §6.7).
func roundCalls(queries []string, missing []string) []ToolCall {
	calls := make([]ToolCall, 0, len(queries)+maxGrepLegs)

	for _, query := range queries {
		query = strings.TrimSpace(query)
		if query == "" {
			continue
		}
		calls = append(calls, ToolCall{
			Name:      ToolHybridSearch,
			Arguments: map[string]any{"query": query},
		})
	}

	for index, term := range missing {
		if index >= maxGrepLegs {
			break
		}
		term = strings.TrimSpace(term)
		if term == "" {
			continue
		}
		calls = append(calls, ToolCall{
			Name:      ToolGrepSearch,
			Arguments: map[string]any{"pattern": term},
		})
	}
	return calls
}

// toolPlanSystemPrompt asks the model to choose retrieval tools rather than to
// answer.
//
// Tool selection is a separate call from the draft on purpose: docs/plan.md §6.6
// keeps the checker reading only the draft, so folding tool choice into the
// drafting turn would blur what the sufficiency verdict is actually about.
const toolPlanSystemPrompt = `You choose retrieval tools for a question about a private document set.

Pick the calls most likely to find what is still missing.

- Call a tool only when it adds information. Return no calls when the evidence
  already listed is enough.
- hybrid_search finds passages that mean the same thing; grep_search finds exact
  strings such as identifiers, codes and rare proper nouns; list_chunks reads a
  document or a page back in order; metadata_search filters by metadata only.
- Do not repeat a call that was already made.`

// renderToolPlanPrompt describes what is already in hand, so the model does not
// ask for it twice.
//
// The evidence is summarised rather than quoted: the model is choosing tools,
// and the full text would crowd out the window needed for that choice.
func renderToolPlanPrompt(question string, evidence []store.Hit, missing []string, budget int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n\n", question)
	fmt.Fprintf(&b, "Already retrieved: %d passage(s).\n", len(evidence))

	// A sample is enough to avoid repeats, and it keeps the prompt small.
	for i, hit := range evidence {
		if i >= 8 {
			fmt.Fprintf(&b, "- ... and %d more\n", len(evidence)-i)
			break
		}
		fmt.Fprintf(&b, "- %s p.%d %s\n", hit.Chunk.DocID, hit.Chunk.PageNum,
			truncateRunes(collapse(hit.Chunk.Text), 80))
	}

	if len(missing) > 0 {
		fmt.Fprintf(&b, "\nReported missing: %s\n", strings.Join(missing, ", "))
	}
	fmt.Fprintf(&b, "\nChoose at most %d call(s).", budget)
	return b.String()
}

// planTools asks the model which retrieval tools to run this round.
//
// Every failure path falls back to the deterministic roundCalls — no model, a
// failed call, or a model that asks for nothing usable. A round that retrieves
// nothing would stall the loop, and the deterministic plan is exactly the
// behaviour that shipped before tool selection existed, so degrading to it is a
// return to a known-good state rather than a new risk.
func (l *Loop) planTools(
	ctx context.Context,
	question string,
	queries, missing []string,
	evidence []store.Hit,
	trace *[]string,
) []ToolCall {
	fallback := roundCalls(queries, missing)
	if l.Model == nil {
		return fallback
	}

	reply, err := l.Model.Complete(ctx, []Message{
		{Role: RoleSystem, Content: toolPlanSystemPrompt},
		{Role: RoleUser, Content: renderToolPlanPrompt(question, evidence, missing, l.spec().ActionMaxTurns)},
	}, ToolSpecs())
	if err != nil {
		l.logf("[Action Session] tool planning failed (%v); using the deterministic plan", err)
		return fallback
	}
	if reply == nil || len(reply.ToolCalls) == 0 {
		l.step(trace, "[Action Session] the model chose no tool; using the deterministic plan.")
		return fallback
	}

	calls := make([]ToolCall, 0, len(reply.ToolCalls))
	dropped := 0
	for _, call := range reply.ToolCalls {
		if !KnownTool(call.Name) {
			dropped++
			continue
		}
		calls = append(calls, call)
	}
	if len(calls) == 0 {
		l.step(trace, fmt.Sprintf(
			"[Action Session] the model chose %d tool(s) outside the surface; using the deterministic plan.",
			dropped))
		return fallback
	}

	rendered := make([]string, 0, len(calls))
	for _, call := range calls {
		rendered = append(rendered, fmt.Sprintf("%s(%s)", call.Name, renderArgs(call.Arguments)))
	}
	l.step(trace, fmt.Sprintf("[Action Session] the model chose %d call(s): %s",
		len(calls), strings.Join(rendered, ", ")))
	if dropped > 0 {
		l.step(trace, fmt.Sprintf(
			"[Action Session] dropped %d call(s) outside the surface.", dropped))
	}
	return calls
}

// toolbox returns the configured toolbox, or a default one over the store.
func (l *Loop) toolbox() *Toolbox {
	if l.Tools != nil {
		return l.Tools
	}
	return &Toolbox{Store: l.Store, DefaultLimit: l.spec().SnippetsPerQuery}
}

// runTools executes one round's tool calls, merging new hits into evidence.
//
// The number of calls is capped by ActionMaxTurns, mirroring the session's tool
// budget (docs/plan.md §6.1). A failing tool is recorded and skipped: one bad
// call must not abandon the round.
func (l *Loop) runTools(ctx context.Context, calls []ToolCall, evidence *[]store.Hit, seen map[string]bool, trace *[]string) int {
	spec := l.spec()
	toolbox := l.toolbox()
	added := 0
	issued := 0

	for _, call := range calls {
		if issued >= spec.ActionMaxTurns {
			l.step(trace, fmt.Sprintf(
				"[Action Session] tool budget spent (%d call(s)); stopping retrieval.", issued))
			break
		}
		issued++

		result, err := toolbox.Execute(ctx, call)
		if err != nil {
			l.step(trace, fmt.Sprintf("[Tool] %s failed: %v", call.Name, err))
			continue
		}

		fresh := 0
		for _, hit := range result.Hits {
			key := hit.Chunk.DocID + "\x00" + hit.Chunk.ChunkID
			if seen[key] {
				continue
			}
			seen[key] = true
			*evidence = append(*evidence, hit)
			fresh++
		}
		added += fresh
		l.step(trace, fmt.Sprintf(
			"[Tool] %s(%s) -> +%d passage(s). %s",
			call.Name, renderArgs(call.Arguments), fresh, result.Note))
	}
	return added
}

// renderArgs renders tool arguments compactly for the trace, sorted so the line
// is stable across runs.
func renderArgs(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", key, args[key]))
	}
	return strings.Join(parts, " ")
}

// Warm loads the generator and primes the prompt prefix the next real call will
// reuse.
//
// The request deliberately mirrors the tool-planning call — same system prompt,
// same tool specs — because Ollama caches the prompt *prefix*: warming with a
// different prompt would pay the model load and nothing else. Measured, that
// load is 0.5 s; the prompt cache is the part worth having, since the tool
// definitions and system prompt are the bulk of the first request of every
// round.
//
// Callers invoke this in the background: its whole purpose is to move work off
// the first question, so blocking startup on it would defeat it.
func (l *Loop) Warm(ctx context.Context) error {
	if l.Model == nil {
		return fmt.Errorf("agent: no model configured")
	}
	_, err := l.Model.Complete(ctx, []Message{
		{Role: RoleSystem, Content: toolPlanSystemPrompt},
		{Role: RoleUser, Content: "Warm-up request. Return no tool calls."},
	}, ToolSpecs())
	return err
}

// draft produces the intermediate draft the checker reviews.
//
// With a model it is a generation call; without one — or when that call fails —
// it degrades to an extractive draft, because the loop must keep making
// progress rather than stall on an unavailable model.
func (l *Loop) draft(ctx context.Context, question string, evidence []store.Hit) string {
	if l.Model != nil {
		reply, err := l.Model.Complete(ctx, []Message{
			{Role: RoleSystem, Content: draftSystemPrompt},
			{Role: RoleUser, Content: renderEvidence(question, evidence, l.maxPromptChars())},
		}, nil)
		if err != nil {
			l.logf("[Draft] model call failed (%v); falling back to an extractive draft", err)
		} else if reply != nil && strings.TrimSpace(reply.Content) != "" {
			return truncateRunes(strings.TrimSpace(reply.Content), l.maxDraftChars())
		}
	}
	return truncateRunes(extractiveDraft(evidence), l.maxDraftChars())
}

// rewrite asks the model for one new query, falling back to the missing terms
// the checker reported. Returning nil stops the loop.
func (l *Loop) rewrite(ctx context.Context, question string, missing []string) []string {
	if l.Model != nil {
		var prompt strings.Builder
		fmt.Fprintf(&prompt, "Question: %s\n", question)
		if len(missing) > 0 {
			fmt.Fprintf(&prompt, "Missing information: %s\n", strings.Join(missing, ", "))
		}
		reply, err := l.Model.Complete(ctx, []Message{
			{Role: RoleSystem, Content: rewriteSystemPrompt},
			{Role: RoleUser, Content: prompt.String()},
		}, nil)
		if err != nil {
			l.logf("[QueryRewriter] model call failed (%v); using the missing terms", err)
		} else if reply != nil {
			if query := strings.TrimSpace(reply.Content); query != "" {
				return []string{query}
			}
		}
	}

	if len(missing) == 0 {
		return nil
	}
	// Deterministic fallback: re-ask with the terms the evidence lacked.
	return []string{question + " " + strings.Join(missing, " ")}
}

// renderEvidence renders the numbered evidence block for the draft prompt.
//
// maxChars bounds the block so the prompt fits the generator's context window.
// Passages are added in rank order until the budget runs out; the last one is
// trimmed mid-text rather than dropped, and the omission is stated so the model
// knows the evidence was cut instead of silently reasoning over a partial list.
func renderEvidence(question string, evidence []store.Hit, maxChars int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n\nEvidence:\n", question)

	kept := 0
	for i, hit := range evidence {
		header := fmt.Sprintf("[%d] (%s p.%d) ", i+1, hit.Chunk.DocID, hit.Chunk.PageNum)
		text := collapse(strings.TrimSpace(hit.Chunk.Text))

		if maxChars > 0 {
			room := maxChars - b.Len() - len(header) - 1
			if room < 120 {
				// Too little room left for a usable passage; stop here rather
				// than emit a stream of stubs.
				break
			}
			if len(text) > room {
				text = truncateRunes(text, room) + "…"
			}
		}
		fmt.Fprintf(&b, "%s%s\n", header, text)
		kept++
	}

	if kept < len(evidence) {
		fmt.Fprintf(&b, "\n(%d of %d passage(s) omitted to fit the context window.)\n",
			len(evidence)-kept, len(evidence))
	}
	return b.String()
}

// extractiveDraft is the model-free draft: the top passages, quoted and cited.
func extractiveDraft(evidence []store.Hit) string {
	const maxPassages = 6
	if len(evidence) == 0 {
		return ""
	}
	var lines []string
	for i, hit := range evidence {
		if i >= maxPassages {
			break
		}
		lines = append(lines, fmt.Sprintf("[%d] %s", i+1, collapse(strings.TrimSpace(hit.Chunk.Text))))
	}
	return strings.Join(lines, "\n")
}

// collapse squeezes whitespace so a passage renders on one line.
func collapse(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

// truncateRunes cuts text to at most n runes.
func truncateRunes(text string, n int) string {
	if n <= 0 {
		return text
	}
	runes := []rune(text)
	if len(runes) <= n {
		return text
	}
	return string(runes[:n])
}

// Prompts. The draft prompt demands [n] citations: with the checker reading the
// draft only (docs/plan.md §6.6), citation markers are the one signal that ties
// the draft back to the evidence.
const draftSystemPrompt = `You answer the user's question using ONLY the numbered evidence passages provided.
Write a short answer of two to four sentences. Cite the passages you used as [n].
If the evidence does not answer the question, state what is missing instead of guessing.
Never invent facts.`

const rewriteSystemPrompt = `You write ONE search query that would find the information still missing.
Output the query text only — no quotes, no explanation, no surrounding punctuation.`
