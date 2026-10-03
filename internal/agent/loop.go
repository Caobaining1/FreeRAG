package agent

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"time"

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
	// AnswerLanguage forces the language the answer is written in: "zh", "en",
	// or "" to follow the question. A setting rather than a property of the
	// prompt, because a user asking a Chinese question of an English corpus
	// wants the answer in the language they asked in — and one asking in
	// English about Chinese sources wants the same.
	AnswerLanguage string
}

// Medium is the medium-mode spec: agentic with SCA review, but no planner and
// no prefetch fan-out (docs/plan.md §6.2).
func Medium() Spec {
	return Spec{
		Label:        "medium",
		EnableSCA:    true,
		SCAMaxRounds: 3,
		UseFanout:    false,
		// A per-TURN allowance, not a run total: the planner is asked to issue
		// every independent call in one turn (see toolPlanSystemPrompt), because
		// the calls inside a turn are independent and running five costs the same
		// as running one. RAGFlow's tool agents do the same — several tool calls
		// per step, bounded by an explicit round count — and its `max_rounds` is
		// the counterpart of SCAMaxRounds here. 12 leaves room for a multi-part
		// question (a search per part, a grep per identifier) without letting a
		// single turn pull in so much evidence that the answer prompt stops
		// fitting; the ceiling that matters is still SCAMaxRounds × this.
		ActionMaxTurns:   12,
		SnippetsPerQuery: 10,
		// The four retrieval tools of §6.7: nothing that needs a compiled
		// structure (no navigate_*, no graph_explore) and no web search.
		Tools: ToolNames(),
	}
}

// DefaultExtractiveAnswerChars caps the answer handed to the checker.
const DefaultExtractiveAnswerChars = 6000

// DefaultContextTokens is the context window the kernel asks the generator for.
//
// docs/plan.md §5 budgets 4K–8K; the top of that range is the useful default
// because the answer prompt carries the evidence block.
const DefaultContextTokens = 8192

// answerPromptReserveTokens covers the system prompt, the question, and the room
// the answer itself needs.
//
// Raised from 768 when the answer prompt started asking for a thorough answer:
// the reserve has to hold the answer, and the generator's num_predict is what
// bounds it. Reserving less than the answer can grow to does not truncate the
// answer — it overflows the window, and Ollama rejects an oversized prompt with
// HTTP 400 rather than trimming it.
const answerPromptReserveTokens = 1280

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
	usable := contextTokens - answerPromptReserveTokens
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
	Answer   string      `json:"answer"`
	Verdict  Verdict     `json:"verdict"`
	Missing  []string    `json:"missing,omitempty"`
	Evidence []store.Hit `json:"evidence"`
	Queries  []string    `json:"queries"`
	// Route records the flow's simple/complex classification. Empty when the
	// loop was run directly (no flow), which is what the tests and the
	// deterministic `ask` path do.
	Route Route `json:"route,omitempty"`
	// SubQuestions is non-empty only on the complex path: it is the decompose
	// result the sub-loops were run over, kept so the run can be explained.
	SubQuestions []string `json:"sub_questions,omitempty"`
	// NotSufficient names the sub-questions that never reached SUFFICIENT on
	// their own pool. Only the complex path fills it: there, Verdict is SUFFICIENT
	// iff this is empty, and each sub-question was judged on its own evidence
	// rather than on the merged pool (see Flow.fanoutNode).
	NotSufficient []string `json:"not_sufficient,omitempty"`
	// Enumeration reports that the rewrite declared the answer to be a SET of
	// named elements, which switches the run to the enumeration strategy
	// (see coverage.go). ItemKind is what the elements are, and Members are the
	// ones the run could anchor to a passage.
	Enumeration bool     `json:"enumeration,omitempty"`
	ItemKind    string   `json:"item_kind,omitempty"`
	Members     []Member `json:"members,omitempty"`
	// MetadataScope is the document set a metadata_search narrowed this run to,
	// empty when no filter was applied, and MetadataFilter is that filter in the
	// words the model used. Both are reported because a narrow search that found
	// nothing is not the same statement as a corpus that has nothing.
	MetadataScope  []string `json:"metadata_scope,omitempty"`
	MetadataFilter string   `json:"metadata_filter,omitempty"`
	Trace          []string `json:"trace"`
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
// Model may be nil: without a generating model the loop still runs, answering
// extractively and rewriting from the checker's missing terms. That keeps the
// graph exercised in tests and in a model-less desktop install.
type Loop struct {
	Store *store.Store
	// Tools executes the retrieval tool calls; nil builds a default Toolbox
	// over Store (docs/plan.md §6.7).
	Tools   *Toolbox
	Model   Model
	Checker Checker
	// Chooser decides the next retrieval step each round. When set it takes
	// precedence over Model's tool planning: Laya scores a deterministic
	// candidate set in ~100 ms, where the planning model call costs tens of
	// seconds and can name values the index does not hold.
	Chooser ToolChooser
	Spec    Spec
	Logger  *log.Logger

	// ExtractiveAnswerChars caps the answer; <=0 selects DefaultExtractiveAnswerChars.
	ExtractiveAnswerChars int
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

	// OnAnswerDelta, when set, receives the final answer as it is written.
	//
	// Kept separate from OnStep because it carries the answer itself rather than
	// a trace line, and the two are rendered differently: trace lines go to a
	// log, this goes where the answer will appear.
	//
	// ONLY the answer is reported here, and it is the only text the loop asks a
	// model to write: the per-round draft the checker used to review is gone (see
	// kbinfo). So a caller receives exactly one text per question, that text is
	// the one in Result.Answer, and it is never replaced — which is what lets a
	// renderer append fragments without having to ask whether it is showing
	// something that will be retracted.
	OnAnswerDelta func(delta string)
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

func (l *Loop) maxExtractiveAnswerChars() int {
	if l.ExtractiveAnswerChars > 0 {
		return l.ExtractiveAnswerChars
	}
	return DefaultExtractiveAnswerChars
}

func (l *Loop) maxPromptChars() int {
	if l.MaxPromptChars > 0 {
		return l.MaxPromptChars
	}
	return PromptCharBudget(DefaultContextTokens)
}

// Run executes the loop and writes the answer: gather evidence, check, rewrite
// while the verdict stays unsatisfied and rounds remain, then write ONE answer
// from the pool that was gathered. The half that writes nothing is
// RunRetrieval.
func (l *Loop) Run(ctx context.Context, question string) (*Result, error) {
	result, err := l.RunRetrieval(ctx, question)
	if err != nil {
		// A cancelled or failed run returns its partial result with no answer
		// written, which is the behaviour the loop has always had.
		return result, err
	}
	if result == nil {
		return nil, fmt.Errorf("agent: loop produced no result")
	}
	l.writeAnswer(ctx, question, result)
	return result, nil
}

// RunRetrieval gathers evidence and judges it, and writes nothing.
//
// This is the complex path's entry point. A sub-question exists to be SEARCHED
// and judged, and the single answer is written once by the synthesis from the
// merged pool — so a sub-loop having its own answer written was N full
// generations whose only reader was the synthesis prompt. Measured on the
// reference machine an answer costs about a minute (6.4 tok/s), which made this
// the largest avoidable cost on the complex path.
func (l *Loop) RunRetrieval(ctx context.Context, question string) (*Result, error) {
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

	// Round 1 searches the REWRITTEN queries, never the question as asked (see
	// RewriteQueries). The question stays the question for everything else — the
	// checker, the answer prompt, and the rewrite that follows a reported gap —
	// because those are about meaning rather than about matching an index.
	plan := l.searchPlan(ctx, question, &result.Trace)
	queries := plan.Queries
	// Enumeration. The gate reads the DECLARATION (a set-shaped slot with an
	// element kind and the deed's words) and never the question's wording, so a
	// question that merely looks like a list pays nothing for this.
	coverage := CoverageOf(plan.Slots)
	rounds := spec.SCAMaxRounds
	if coverage.Ok() {
		queries, rounds = l.enterEnumeration(coverage, queries, rounds, &result.Trace)
		result.Enumeration, result.ItemKind = true, coverage.ItemKind
	}
	// Everything retrieval returns accumulates here, and this is what the
	// sufficiency decision reads (see kbinfo).
	info := newKBInfo()
	var missing []string
	// Every call this run has already made, with what it returned. The planner
	// system prompt has always said "do not repeat a call that was already
	// made", but the prompt never said WHICH calls those were — an omission that
	// only bites when the evidence pool is empty, because then there is nothing
	// else in the prompt to steer by either.
	var attempts []attempt

	// Round 1's calls, when the declaration passed the gate: one recall per
	// operand, all of them in this one turn. Built here rather than through
	// planTools because the operand list IS the declaration — it is not a decision,
	// and RAGFlow issues the whole recall list itself (see OperandCalls).
	var enumerationCalls []ToolCall
	if coverage.Ok() {
		enumerationCalls = OperandCalls(coverage.Operands(), spec.ActionMaxTurns)
	}

	for round := 1; round <= rounds; round++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		result.Rounds = round
		result.Queries = queries

		var calls []ToolCall
		switch {
		case round == 1 && len(enumerationCalls) > 0:
			l.step(&result.Trace, fmt.Sprintf(
				"[Enumerate] recalling all %d operand(s) in this turn.", len(enumerationCalls)))
			calls = enumerationCalls
		default:
			calls = l.planTools(ctx, question, queries, missing, info.pool(), attempts, &result.Trace)
		}
		added := l.runTools(ctx, calls, info, &result.Trace, &attempts)
		l.step(&result.Trace, fmt.Sprintf(
			"[RAGAgent] Round %d: +%d passage(s); kbinfo now holds %d.",
			round, added, info.len()))

		// No answer is written, in any round.
		//
		// The loop used to generate one per round so the checker had something
		// to judge: a full generation — measured at 60 s on the reference
		// machine — whose only reader was that decision, and which the user
		// never saw, because the published text is written once at the end by
		// answer. The checker now reads kbinfo, so the answer had no reader left.
		empty := info.empty()

		// An empty pool is never sufficient. The guard stays even though the
		// checker now reads the pool itself, because the failure it prevents was
		// measured, not imagined.
		//
		// Back when the checker judged a answer, a first tool call that matched
		// nothing produced the answer "the evidence does not answer this", which
		// Laya scored SUFFICIENT at 0.93 — correctly, since that answer does
		// answer the question it was asked. Nothing asked whether the corpus had
		// been looked at.
		//
		// Asking about the pool makes sufficiency much harder to mistake, but
		// "much harder" is not an invariant. Without this the loop can return
		// after one round holding zero passages, and the answer reads as a
		// finding about the corpus ("there are no papers by this author") when it
		// is a report of a lookup that never happened.
		var (
			verdict Verdict
			gaps    []string
		)
		if empty {
			verdict, gaps = VerdictInsufficient, uniqueTerms(question)
			l.step(&result.Trace, fmt.Sprintf(
				"[SCA] Round %d verdict=INSUFFICIENT (no evidence to judge; the checker is not asked).", round))
		} else {
			verdict, gaps = checker.Check(ctx, question, info)
			l.step(&result.Trace, fmt.Sprintf(
				"[SCA] Round %d verdict=%s (missing=%d).", round, verdict, len(gaps)))
		}
		result.Verdict = verdict
		result.Missing = gaps
		missing = gaps

		if verdict == VerdictSufficient {
			break
		}
		if round == rounds {
			break
		}

		next := l.rewrite(ctx, question, missing)
		if len(next) == 0 {
			l.step(&result.Trace, "[QueryRewriter] no new angle; stopping.")
			break
		}
		queries = next
	}

	result.Evidence = info.pool()
	// The enumeration's members are resolved once, over the pool the run settled
	// on. RAGFlow resolves coverage in its formalize-answer node for the same
	// reason: a member list is a reading of the FINAL evidence, not of evidence a
	// later round may supersede.
	if coverage.Ok() && len(result.Evidence) > 0 {
		result.Members = l.extractMembers(ctx, question, result.Evidence, &result.Trace)
	}
	// The scope travels on the result for two reasons: the answer prompt has to
	// say that the search was narrowed (a "not found" inside a filter is not a
	// statement about the corpus), and the reply has to show what the run was
	// limited to.
	result.MetadataScope = info.scopeIDs()
	result.MetadataFilter = info.scopeNote
	return result, nil
}

// writeAnswer is the only place a loop asks a model to write the answer.
//
// It used to be the tail of Run, which meant every sub-loop on the complex path
// wrote one too (see RunRetrieval). There is no per-round answer to contrast
// this with: the text written here is the text published, which is what makes it
// final by construction — nothing is reported while the loop may still change
// its mind, so there is nothing to retract. It is also where OnAnswerDelta
// fires, so a caller sees exactly one text per Run.
func (l *Loop) writeAnswer(ctx context.Context, question string, result *Result) {
	l.step(&result.Trace, "[Answer] writing…")
	started := time.Now()
	result.Answer = l.answer(ctx, question, result.Evidence, result.MetadataFilter,
		RenderMemberRecord(question, result.Members))
	if result.Answer == "" {
		l.step(&result.Trace, "[Answer] not written: the run found no evidence to answer from.")
		return
	}
	l.step(&result.Trace, fmt.Sprintf("[Answer] written in %s (%d char(s)).",
		time.Since(started).Round(100*time.Millisecond), len([]rune(result.Answer))))
}

// attempt is one tool call this run has already made, and what it returned.
//
// Recorded so the next round's planner can see it. A round that retrieved
// nothing is exactly when the model most needs to know what was tried — the
// evidence sample it would otherwise learn from is empty — and repeating the
// same empty call is otherwise the single most likely next action.
type attempt struct {
	Call  ToolCall
	Added int
}

// findAttempt reports whether this exact call has already been made this run.
func findAttempt(attempts []attempt, call ToolCall) (attempt, bool) {
	for _, previous := range attempts {
		if sameToolCall(previous.Call, call) {
			return previous, true
		}
	}
	return attempt{}, false
}

// sameToolCall compares a name and its arguments.
//
// Arguments are compared through fmt.Sprint rather than with ==, because the
// same logical call arrives with different concrete types: a number is float64
// when it came from JSON and int when a test built the map by hand, and those
// two spellings must count as one call.
func sameToolCall(a, b ToolCall) bool {
	if a.Name != b.Name || len(a.Arguments) != len(b.Arguments) {
		return false
	}
	for key, value := range a.Arguments {
		other, ok := b.Arguments[key]
		if !ok || fmt.Sprint(other) != fmt.Sprint(value) {
			return false
		}
	}
	return true
}

// renderAttempts renders the calls already made, most recent first.
func renderAttempts(attempts []attempt) string {
	if len(attempts) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\nAlready tried this run — do NOT repeat these:\n")
	// Most recent first: it is the one the model is about to duplicate.
	for i := len(attempts) - 1; i >= 0; i-- {
		item := attempts[i]
		fmt.Fprintf(&b, "- %s(%s) -> +%d passage(s)", item.Call.Name, renderArgs(item.Call.Arguments), item.Added)
		if item.Added == 0 {
			b.WriteString("  ← returned nothing; change the field, the value, or the tool")
		}
		b.WriteString("\n")
	}
	return b.String()
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
// Tool selection is a separate call from the answer on purpose: the checker
// judges the pool, and a call that also produced prose would blur what the
// verdict is about — the passages, not what someone wrote about them.
//
// Modelled on RAGFlow's action_run.md playbook (docs/plan.md's reference
// implementation), cut down to what freerag actually has: the four tools below
// and nothing else. RAGFlow's playbook also covers navigate_tree,
// navigate_structure, graph_explore, calculate and web_search — naming a tool
// the kernel does not answer is how a model spends a round on a call that is
// silently dropped, so those are gone rather than merely discouraged.
//
// This prompt is the fallback path: the live path asks Laya to choose among
// fully-formed candidate calls (toolchoice.go), which cannot invent a value.
const toolPlanSystemPrompt = `You choose retrieval calls for a question about a private document set.

The tools are the ones defined in the API schema for this call — read their WHEN TO
CALL / DO NOT CALL / ARGUMENTS / IF IT FAILS sections, which are the only guide for
choosing between them. Never name a tool that is not defined there.

How to act:
- metadata_search NARROWS THE WHOLE RUN. The documents it selects become the scope for every later call, so once it has run, do not repeat the filter and do not pass the document again: hybrid_search and grep_search are already limited to those documents. Use it first when the question names or dates documents, then search inside them.
- A later metadata_search REPLACES the scope, and clear=true lifts it. If you are finding nothing and you set a scope earlier, either widen it or clear it — the evidence list is empty because of the filter, not because the corpus has nothing.
- Issue every independent call in ONE turn. A turn spent on five calls costs the same as a turn spent on one. Independent means: one per spelling of a name (a keyword index matches literally, so "Apple Inc." and "AAPL" are two searches), one to read one document whole (list_chunks) as against one that searches across documents, and one per missing term. Two rewordings of one spelling are NOT independent — that is one call written twice, and the second is refused as a repeat.
- Call a tool only when it adds information. Return no calls when the evidence already listed is enough.
- Never repeat a call that was already made: the index does not change during a run, so it cannot return anything new. Change the field, the value or the tool.`

// renderToolPlanPrompt describes what is already in hand, so the model does not
// ask for it twice.
//
// The evidence is summarised rather than quoted: the model is choosing tools,
// and the full text would crowd out the window needed for that choice.
//
// metadataCatalog is the AVAILABLE METADATA block (see renderMetadataCatalog).
// It rides here rather than in the system prompt because it is per-index while
// the system prompt is fixed — and it must not change the prompt PREFIX, which
// is what the warm-up request primes and Ollama caches.
//
// attempts are the calls already made this run. They matter most in the case
// where the evidence sample below is EMPTY: with nothing retrieved there is
// nothing to steer by, and the model's most likely next action is to repeat the
// call that just returned nothing.
func renderToolPlanPrompt(
	question string,
	queries []string,
	evidence []store.Hit,
	attempts []attempt,
	missing []string,
	budget int,
	metadataCatalog string,
) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n", question)
	if len(queries) > 0 {
		// Spelled out because the candidate list is built from these, and a
		// planner that searches the question text on its own is reaching for a
		// query the rewriter already replaced (see RewriteQueries).
		b.WriteString("\nSearch queries retrieval runs (the question itself is not one of them):\n")
		for _, query := range queries {
			fmt.Fprintf(&b, "- %s\n", query)
		}
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "Already retrieved: %d passage(s).\n", len(evidence))

	if metadataCatalog != "" {
		fmt.Fprintf(&b, "\n%s\n", metadataCatalog)
	}

	// A sample is enough to avoid repeats, and it keeps the prompt small.
	for i, hit := range evidence {
		if i >= 8 {
			fmt.Fprintf(&b, "- ... and %d more\n", len(evidence)-i)
			break
		}
		fmt.Fprintf(&b, "- %s p.%d %s\n", hit.Chunk.DocID, hit.Chunk.PageNum,
			truncateRunes(collapse(hit.Chunk.Text), 80))
	}

	if rendered := renderAttempts(attempts); rendered != "" {
		b.WriteString(rendered)
	}

	if len(missing) > 0 {
		fmt.Fprintf(&b, "\nReported missing: %s\n", strings.Join(missing, ", "))
	}
	// The number is a per-turn allowance, not a total: saying "at most N" without
	// saying "this turn" reads as a cap on the whole run and makes a small model
	// hold calls back for no reason.
	fmt.Fprintf(&b, "\nThis turn runs up to %d call(s): issue every independent call you need now, "+
		"in this one turn, rather than one per turn.", budget)
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
	attempts []attempt,
	trace *[]string,
) []ToolCall {
	fallback := roundCalls(queries, missing)

	// Laya first when configured: it chooses among fully-formed candidate calls,
	// so it is fast (~100 ms) and cannot invent a value the index does not hold.
	// A failure here falls through to the model plan rather than to the
	// deterministic fallback, because the model plan is still a better answer
	// than the fixed roundCalls when it is available.
	if l.Chooser != nil {
		candidates := buildCandidates(queries, missing, evidence, attempts)
		if len(candidates) == 0 {
			l.step(trace, "[Action Session] no untried candidate call remains; stopping retrieval.")
			return nil
		}
		index, err := l.Chooser.Choose(ctx, question, candidates, evidence,
			renderAttempts(attempts), missing)
		switch {
		case err != nil:
			// Reported in the TRACE, not only in the log. A chooser that fails on
			// every round is exactly the kind of thing that must be visible where
			// a user looks, and this one was not: the bug it hid was that two of
			// Laya's three uses were dead, and it took a full run's transcript to
			// notice that every round had said "falling back to the model plan".
			l.step(trace, fmt.Sprintf(
				"[Action Session] tool choice unavailable (%v); using the model plan.", err))
			l.logf("[Action Session] tool choice failed (%v); falling back to the model plan", err)
		case index >= 0 && index < len(candidates):
			chosen := candidates[index]
			l.step(trace, fmt.Sprintf("[Action Session] chose %s", chosen.Label))
			return []ToolCall{chosen.Call}
		default:
			l.step(trace, "[Action Session] the chooser stopped: no further retrieval.")
			return nil
		}
	}

	if l.Model == nil {
		return fallback
	}

	reply, err := l.Model.Complete(ctx, []Message{
		{Role: RoleSystem, Content: toolPlanSystemPrompt},
		{Role: RoleUser, Content: renderToolPlanPrompt(question, queries, evidence, attempts, missing,
			l.spec().ActionMaxTurns, renderMetadataCatalog(l.Store, maxMetadataHintValues))},
	}, ToolSpecs())
	if err != nil {
		l.logf("[Action Session] tool planning failed (%v); using the deterministic plan", err)
		return fallback
	}
	if reply == nil || len(reply.ToolCalls) == 0 {
		l.step(trace, "[Action Session] the model chose no tool; using the deterministic plan.")
		return fallback
	}

	// Calls already tried this run are dropped HERE rather than only in runTools,
	// so that a plan consisting entirely of repeats becomes the deterministic
	// plan instead of an empty round. Measured on a real complex question: a
	// generating model re-issued the same fabricated metadata filter in all three
	// rounds of a sub-question, each one refused as a repeat, so the sub-question
	// finished with an empty pool while the hybrid search that would have worked
	// was never run. The repeat guard kept the calls from being wasted; it could
	// not make the round retrieve anything.
	calls := make([]ToolCall, 0, len(reply.ToolCalls))
	dropped, repeated := 0, 0
	for _, call := range reply.ToolCalls {
		if !KnownTool(call.Name) {
			dropped++
			continue
		}
		if _, tried := findAttempt(attempts, call); tried {
			repeated++
			// Traced HERE because the call no longer reaches runTools, which is
			// where this refusal used to be reported. Dropping it quietly would
			// make "the model asked for something it had already tried"
			// invisible — and that is the fact that explains a round which
			// retrieved nothing.
			l.step(trace, fmt.Sprintf(
				"[Action Session] %s(%s) was already tried this run; not repeating it.",
				call.Name, renderArgs(call.Arguments)))
			continue
		}
		calls = append(calls, call)
	}
	if len(calls) == 0 {
		if repeated > 0 {
			l.step(trace, fmt.Sprintf(
				"[Action Session] every one of the %d call(s) the model chose was already tried; using the deterministic plan.",
				repeated))
		} else {
			l.step(trace, fmt.Sprintf(
				"[Action Session] the model chose %d tool(s) outside the surface; using the deterministic plan.",
				dropped))
		}
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
//
// Every call is also recorded as an attempt, INCLUDING the ones that failed or
// matched nothing. Those are the ones the next round most needs to avoid, and
// they are exactly the ones a hits-only record would omit.
func (l *Loop) runTools(
	ctx context.Context,
	calls []ToolCall,
	info *kbinfo,
	trace *[]string,
	attempts *[]attempt,
) int {
	spec := l.spec()
	toolbox := l.toolbox()
	added := 0
	issued := 0

	for _, call := range calls {
		// An identical call cannot add information: the index does not change
		// during a run, so a call that returned nothing returns nothing again.
		//
		// The planner is told this (renderAttempts) and a small model does not
		// always comply — a measured run repeated, in round 3, the exact call
		// that had already returned nothing in round 1, with the call listed and
		// marked "returned nothing" in the prompt it was answering. Refusing it
		// here makes the saving a property of the loop rather than of the
		// model's instruction-following, and it is checked BEFORE the budget so
		// a repeat cannot spend a turn that would otherwise retrieve something.
		if previous, repeated := findAttempt(*attempts, call); repeated {
			note := fmt.Sprintf("+%d passage(s)", previous.Added)
			if previous.Added == 0 {
				note = "it returned nothing"
			}
			l.step(trace, fmt.Sprintf(
				"[Action Session] %s(%s) was already tried this run (%s); not repeating it.",
				call.Name, renderArgs(call.Arguments), note))
			continue
		}

		if issued >= spec.ActionMaxTurns {
			l.step(trace, fmt.Sprintf(
				"[Action Session] tool budget spent (%d call(s)); stopping retrieval.", issued))
			break
		}
		issued++

		// The scope goes INTO the search, not onto its results: a top-k taken
		// globally and then sifted can come back empty while in-scope passages
		// exist (see store.Filter).
		result, err := toolbox.ExecuteScoped(ctx, call, info.filter())
		if err != nil {
			*attempts = append(*attempts, attempt{Call: call})
			l.step(trace, fmt.Sprintf("[Tool] %s failed: %v", call.Name, err))
			continue
		}

		// The scope is adopted BEFORE the hits are added: metadata_search both
		// selects the documents and returns their chunks, and the documents it
		// just selected must be inside the scope they establish.
		if result.Scope != nil {
			if result.Scope.Clear {
				info.clearScope()
			} else {
				info.setScope(result.Scope.DocIDs, result.Scope.Note)
			}
			l.step(trace, fmt.Sprintf("[Scope] %s.", result.Scope.Note))
		}

		outside := info.outOfScope
		fresh := info.add(result.Hits)
		added += fresh
		*attempts = append(*attempts, attempt{Call: call, Added: fresh})
		scopeNote := ""
		if info.outOfScope > outside {
			// Reported, not silent: a call that found passages and contributed
			// fewer ones is something the trace should say out loud, or the
			// filter looks like a corpus with holes in it.
			scopeNote = fmt.Sprintf(" %d passage(s) from outside the scope were ignored.",
				info.outOfScope-outside)
		}
		l.step(trace, fmt.Sprintf(
			"[Tool] %s(%s) -> +%d passage(s). %s%s",
			call.Name, renderArgs(call.Arguments), fresh, result.Note, scopeNote))
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

// generate performs one answer-writing call, reporting the text as it is
// written when onDelta is set.
//
// With no model — or when the call fails, or returns nothing — it degrades to an
// extractive answer, because the loop must keep making progress rather than stall
// on an unavailable model.
func (l *Loop) generate(
	ctx context.Context,
	question string,
	evidence []store.Hit,
	onDelta func(string),
	scopeNote string,
	memberRecord string,
) string {
	if l.Model != nil {
		messages := []Message{
			{Role: RoleSystem, Content: answerSystemPrompt + answerLanguageDirective(l.spec().AnswerLanguage)},
			{Role: RoleUser, Content: scopePrefix(scopeNote) + renderEvidence(question, evidence, l.maxPromptChars(), memberRecord)},
		}

		var (
			reply *Reply
			err   error
		)
		if streaming, ok := l.Model.(StreamingModel); ok && onDelta != nil {
			reply, err = streaming.CompleteStream(ctx, messages, nil, onDelta)
		} else {
			reply, err = l.Model.Complete(ctx, messages, nil)
		}

		if err != nil {
			l.logf("[Answer] model call failed (%v); falling back to an extractive answer", err)
		} else if reply != nil && strings.TrimSpace(reply.Content) != "" {
			return truncateRunes(strings.TrimSpace(reply.Content), l.maxExtractiveAnswerChars())
		}
	}
	return truncateRunes(extractiveAnswer(evidence), l.maxExtractiveAnswerChars())
}

// (There is no answer() any more. It wrote the per-round text the checker judged,
// and the bug its own comment warned about — a rejected answer streamed into the
// answer area and then replaced — is unreachable now by construction, because
// the only text the loop asks a model to write is the final answer. See kbinfo.)

// answer writes the answer the user reads, reporting it as it is written.
//
// Called ONCE, after the loop has settled which evidence to use. That is the
// whole point of separating it from answer: the rounds decide WHAT to answer
// from — each round's answer is a proposal the checker accepts or rejects — and
// the answer is then written once, against the settled pool. Generating it
// per round instead both publishes prose written against evidence the loop had
// not finished gathering, and puts the reader in front of a text that is
// replaced under them.
//
// The order is also what makes the streamed text final by construction: nothing
// is published while the loop is still deciding, so there is nothing to retract.
func (l *Loop) answer(
	ctx context.Context,
	question string,
	evidence []store.Hit,
	scopeNote string,
	memberRecord string,
) string {
	if len(evidence) == 0 {
		// Nothing to answer from. The caller reports that, in its own words: a
		// model handed an empty evidence block writes a sentence that reads
		// like a finding about the corpus ("there is no information about this
		// author") when it is a report of a lookup that never happened.
		return ""
	}
	// nil when nothing is listening, so the model is not made to call through a
	// closure that goes nowhere.
	var onDelta func(string)
	if l.OnAnswerDelta != nil {
		onDelta = l.OnAnswerDelta
	}
	return l.generate(ctx, question, evidence, onDelta, scopeNote, memberRecord)
}

// scopePrefix tells the generator that the search was narrowed, so that "the
// evidence does not say" is not written as "the documents do not say".
//
// The distinction is the same one metadata_search's empty-result note draws for
// the model, moved one step later: the model wrote that note while it could
// still act on it, and by the time an answer is written only the wording is
// left. A run filtered to one document that finds nothing there has learned
// about that document, not about the corpus.
func scopePrefix(scopeNote string) string {
	trimmed := strings.TrimSpace(scopeNote)
	if trimmed == "" {
		return ""
	}
	return fmt.Sprintf(
		"Search scope: this run was RESTRICTED to a subset of the corpus — %s. "+
			"If the evidence does not answer the question, say it was not found within this scope and "+
			"name the scope, rather than stating that the documents do not contain it.\n\n",
		trimmed)
}

// searchQueries establishes the query set retrieval starts from.
//
// Reported rather than silent: "the queries changed" is the part of a run a
// reader cannot otherwise see, and when the rewriter declines (no model, a
// failed call, an unusable reply) that is worth knowing before wondering why the
// first round retrieved nothing.
func (l *Loop) searchPlan(ctx context.Context, question string, trace *[]string) QueryPlan {
	plan := PlanQueries(ctx, l.Model, question, maxSearchQueries)
	if !plan.Rewritten {
		l.step(trace, "[QueryRewriter] no usable rewrite; searching the question as asked.")
		return plan
	}
	l.step(trace, fmt.Sprintf("[QueryRewriter] searching %s", joinQuoted(plan.Queries)))
	return plan
}

// enterEnumeration applies the strategy's two decisions to a run whose
// declaration passed the gate.
//
// The recall list becomes the declared operands, and the round budget is bounded
// to two. Both are RAGFlow's:
//
//   - enumeration does not go through query rewriting at all — its recall runs
//     off Coverage.Operands before any rewrite happens (coverage_enumerate.go), so
//     the rewrites a generic run would issue are exactly the near-duplicates this
//     strategy exists to avoid;
//   - the bound is `CoverageOf(table).Ok() && rounds > 2 → 2`
//     (agentic_rag_graph.go:1642), with its reason: once a set is being
//     enumerated, further rounds re-ask the same list rather than find different
//     things.
func (l *Loop) enterEnumeration(cov Coverage, queries []string, rounds int, trace *[]string) ([]string, int) {
	if operands := cov.Operands(); len(operands) > 0 {
		l.step(trace, fmt.Sprintf("[Enumerate] the answer is a set of %s: recalling %d operand(s) — %s",
			cov.ItemKind, len(operands), joinQuoted(operands)))
		queries = operands
	} else {
		l.step(trace, fmt.Sprintf("[Enumerate] the answer is a set of %s; no operand was declared, so the queries stand.", cov.ItemKind))
	}
	if rounds > enumMaxRounds {
		l.step(trace, fmt.Sprintf("[Enumerate] rounds bounded %d -> %d.", rounds, enumMaxRounds))
		rounds = enumMaxRounds
	}
	return queries, rounds
}

// extractMembers runs the one extraction call the enumeration makes.
func (l *Loop) extractMembers(ctx context.Context, question string, evidence []store.Hit, trace *[]string) []Member {
	if l.Model == nil {
		return nil
	}
	started := time.Now()
	members := ExtractMembers(ctx, l.Model, question, evidence, PromptCharBudget(DefaultContextTokens))
	l.step(trace, fmt.Sprintf("[Enumerate] %d member(s) anchored out of %d passage(s) in %s.",
		len(members), len(evidence), time.Since(started).Round(100*time.Millisecond)))
	return members
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

// renderEvidence renders the numbered evidence block for the answer prompt.
//
// maxChars bounds the block so the prompt fits the generator's context window.
// Passages are added in rank order until the budget runs out; the last one is
// trimmed mid-text rather than dropped, and the omission is stated so the model
// knows the evidence was cut instead of silently reasoning over a partial list.
func renderEvidence(question string, evidence []store.Hit, maxChars int, extra string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n\n", question)

	// The enumeration record sits between the question and the passages: it says
	// what the list already holds, which is what the writer needs before reading
	// the evidence the list is drawn from. It counts against the same budget,
	// because it consumes the same context.
	if block := strings.TrimSpace(extra); block != "" {
		b.WriteString(block)
		b.WriteString("\n")
	}
	b.WriteString("Evidence:\n")

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

// extractiveAnswer is the model-free answer: the top passages, quoted and cited.
func extractiveAnswer(evidence []store.Hit) string {
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

// Prompts. The answer prompt demands [n] citations: with the checker reading the
// answer only (docs/plan.md §6.6), citation markers are the one signal that ties
// the answer back to the evidence.
//
// The length of the answer is decided here and nowhere else. Measured on the
// reference machine, one question and one evidence set produced 147 characters
// under the previous "two to four sentences" instruction and 537 under this one.
// num_predict is a ceiling that was never reached (296 tokens at most) and
// ExtractiveAnswerChars a clamp that never engaged, so neither actually shaped the
// answer — only this text did.
//
// Everything after the first line is a constraint rather than a style note, and
// asking for more makes each one load-bearing: a longer answer needs more
// citations, and the failure mode of "be thorough" is a paragraph of confident
// invention with a citation bolted on. The last line exists because that already
// happened once — against a document whose headings are separate chunks, a
// longer answer cited the heading "重点考核：" as though it were content.
const answerSystemPrompt = `You answer the user's question using ONLY the numbered evidence passages provided.
Answer as thoroughly and completely as the evidence allows: cover every point it
supports, and organise a long answer so it stays readable.
Cite the passages you used as [n], placing each marker after the claim it supports.
If the evidence does not answer the question, state what is missing instead of guessing.
Never invent facts. Never cite a passage that does not support the sentence it is attached to.`

const rewriteSystemPrompt = `You write ONE search query that would find the information still missing.
Output the query text only — no quotes, no explanation, no surrounding punctuation.`

// answerLanguageDirective tells the model which language to answer in.
//
// A directive rather than a separate prompt: the evidence block is in whatever
// language the corpus is, and a model left to itself mirrors it — so a Chinese
// question over English papers came back in English. Identifiers are called out
// explicitly because translating a model name or a quoted term makes the answer
// unverifiable against the passage it cites.
func answerLanguageDirective(language string) string {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "zh", "cn", "chinese", "中文", "简体中文":
		return "\nWrite the answer in Chinese (简体中文), even when the evidence is in another language. " +
			"Leave identifiers, model names, numbers and quoted phrases in their original form."
	case "en", "english", "英文":
		return "\nWrite the answer in English, even when the evidence is in another language. " +
			"Leave identifiers, model names, numbers and quoted phrases in their original form."
	}
	return ""
}
