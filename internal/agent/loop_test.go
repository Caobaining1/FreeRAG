package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"freerag/internal/store"
)

// scriptedModel replays canned replies so the graph's control flow can be
// verified without a model download.
//
// Replies are routed by what the turn is for, not by call order: a turn that
// offers tools is a tool-planning turn, everything else is a text turn (answer or
// rewrite). Routing by order alone would make every flow test depend on how many
// planning calls the loop happens to make, which is not what they assert.
type scriptedModel struct {
	mu      sync.Mutex
	replies []*Reply
	calls   int
	seen    [][]Message

	// planReplies are handed out on tool-planning turns, in order. When they run
	// out the model answers with no calls, which makes the loop use its
	// deterministic plan — the behaviour the flow tests were written against.
	planReplies []*Reply
	planCalls   int
	// planned records the tool specs offered on each planning turn.
	planned [][]ToolSpec

	// rewriteReplies are handed out on query-rewrite turns, in order. When they
	// run out the model answers with nothing — and RewriteQueries then falls
	// back to the question itself, which is the behaviour every test here was
	// written against. So a test sees a rewrite only if it asks for one, and a
	// rewrite turn never consumes a reply meant for the answer.
	rewriteReplies []*Reply
	rewriteCalls   int
	// rewriteTopics records which questions the rewriter was asked about.
	rewriteTopics []string
}

// rewriteQuestion reports whether this turn is the query rewrite, and what it
// was asked about.
//
// Matched on the system prompt, the way planReplies are matched on the presence
// of tools: routing by call order would make every test depend on how many
// turns happen to precede it.
func rewriteQuestion(messages []Message) (string, bool) {
	const marker = "You turn a user's question into the search queries"
	for _, message := range messages {
		if message.Role != RoleSystem || !strings.Contains(message.Content, marker) {
			continue
		}
		for _, candidate := range messages {
			if candidate.Role == RoleUser {
				return strings.TrimSpace(strings.TrimPrefix(candidate.Content, "Question:")), true
			}
		}
		return "", true
	}
	return "", false
}

func (m *scriptedModel) Complete(_ context.Context, messages []Message, tools []ToolSpec) (*Reply, error) {
	// Guarded because the complex path runs the sub-loops concurrently against
	// one model: the reply counter is per-model, not per-goroutine.
	m.mu.Lock()
	defer m.mu.Unlock()

	m.seen = append(m.seen, messages)

	if question, isRewrite := rewriteQuestion(messages); isRewrite {
		m.rewriteCalls++
		m.rewriteTopics = append(m.rewriteTopics, question)
		if m.rewriteCalls-1 < len(m.rewriteReplies) {
			return m.rewriteReplies[m.rewriteCalls-1], nil
		}
		// An empty reply makes RewriteQueries fall back to the question, so this
		// turn is invisible to a test that did not script one.
		return &Reply{}, nil
	}

	if len(tools) > 0 {
		m.planned = append(m.planned, tools)
		if m.planCalls < len(m.planReplies) {
			reply := m.planReplies[m.planCalls]
			m.planCalls++
			return reply, nil
		}
		return &Reply{}, nil
	}

	if m.calls >= len(m.replies) {
		return &Reply{}, nil
	}
	reply := m.replies[m.calls]
	m.calls++
	return reply, nil
}

func newStore(t *testing.T, chunks ...store.Chunk) *store.Store {
	t.Helper()
	s := store.New()
	s.Add(chunks)
	return s
}

func TestWarmPrimesTheToolPlanPrefix(t *testing.T) {
	model := &scriptedModel{}
	loop := &Loop{Model: model}

	if err := loop.Warm(context.Background()); err != nil {
		t.Fatalf("Warm: %v", err)
	}

	// The warm-up has to reuse the tool-planning system prompt and the tool
	// specs. Ollama caches the prompt *prefix*, so a warm-up written any other
	// way would load the weights and prime nothing — the caller would pay for
	// a model call and still not get the cache hit it was after.
	if len(model.seen) != 1 {
		t.Fatalf("calls = %d, want exactly 1", len(model.seen))
	}
	if len(model.seen[0]) == 0 || model.seen[0][0].Content != toolPlanSystemPrompt {
		t.Fatalf("messages = %#v, want the tool-planning system prompt first", model.seen[0])
	}
	if len(model.planned) != 1 || len(model.planned[0]) == 0 {
		t.Fatalf("tool specs = %#v, want them forwarded", model.planned)
	}
}

func TestWarmWithoutAModelReportsWhy(t *testing.T) {
	// Not a no-op: a silent success here would let the kernel log
	// "model warm-up done" for a warm-up that never happened.
	if err := (&Loop{}).Warm(context.Background()); err == nil {
		t.Fatal("warming a loop with no model must be an error")
	}
}

// streamingScriptedModel is scriptedModel with the streaming capability, so the
// loop can be exercised on the path that reports the answer as it is written.
type streamingScriptedModel struct {
	scriptedModel
	// deltas records what the loop was told, in order.
	deltas []string
}

func (m *streamingScriptedModel) CompleteStream(
	_ context.Context, messages []Message, tools []ToolSpec, onDelta func(string),
) (*Reply, error) {
	reply, err := m.Complete(context.Background(), messages, tools)
	if err != nil || reply == nil {
		return reply, err
	}

	// Split in halves rather than delivered whole, so the test also covers a
	// delta boundary landing inside the answer. Against a real server that is
	// the normal case, and delivering the answer in one piece would hide any
	// assumption that it arrives complete.
	runes := []rune(reply.Content)
	half := len(runes) / 2
	for _, piece := range []string{string(runes[:half]), string(runes[half:])} {
		if piece == "" || onDelta == nil {
			continue
		}
		m.deltas = append(m.deltas, piece)
		onDelta(piece)
	}
	return reply, nil
}

func TestAnswerReportsDeltasAsTheAnswerIsWritten(t *testing.T) {
	model := &streamingScriptedModel{scriptedModel: scriptedModel{
		replies: []*Reply{{Content: "A longer answer, drafted in parts [1]."}},
	}}
	loop := &Loop{Model: model}

	var got []string
	loop.OnAnswerDelta = func(delta string) { got = append(got, delta) }

	answer := loop.answer(context.Background(), "what?",
		[]store.Hit{{Chunk: store.Chunk{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"}}}, "", "")

	// The deltas must assemble into exactly the answer the call returns. A
	// renderer shows the deltas while the loop reports the return value, so any
	// divergence between the two is silently wrong rather than visibly broken.
	if joined := strings.Join(got, ""); joined != answer {
		t.Fatalf("deltas join to %q but the answer is %q", joined, answer)
	}
	if len(got) < 2 {
		t.Fatalf("deltas = %#v, want the answer reported in more than one piece", got)
	}
}

// TestIntermediateDraftIsNotStreamed pins the separation that removes the flip.
//
// The answer the checker reviews is a PROPOSAL: the loop may reject it and ask
// for another round. Publishing it as it is written put a text in the answer
// area that the next round replaced — a first round that found nothing streamed
// "the evidence does not answer this", and the reader watched the answer change
// its mind. The answer must still be produced (the checker needs it); it must
// simply not be reported.

// TestAnswerIsWrittenOnceAfterTheLoop pins the shape the two tests above add up
// to: the rounds answer for the checker, and the text the user reads is written
// once, at the end, from the settled evidence.
//
// Round 1 here retrieves nothing, so it drafts nothing; round 2 retrieves and
// drafts; then the answer. Exactly one text is ever streamed, and it is the one
// in Result.Answer.
func TestAnswerIsWrittenOnceAfterTheLoop(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "alpha.pdf", PageNum: 1, Text: "alpha passage"})
	model := &streamingScriptedModel{scriptedModel: scriptedModel{
		planReplies: []*Reply{
			{ToolCalls: []ToolCall{{Name: ToolMetadataSearch, Arguments: metadataFilterArgs(
				metadataCondition("doc_id", store.OpEqual, "does-not-exist.pdf"))}}},
		},
		// The rewriter's query, then the answer. No per-round answer any more:
		// the only text this loop asks a model to write is the answer.
		replies: []*Reply{
			{Content: "alpha"},
			{Content: "the answer the reader gets [1]"},
		},
	}}

	var streamed []string
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}}
	loop.OnAnswerDelta = func(delta string) { streamed = append(streamed, delta) }

	result, err := loop.Run(context.Background(), "what does alpha say?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Answer != "the answer the reader gets [1]" {
		t.Fatalf("answer = %q, want the answer written after the loop", result.Answer)
	}
	if joined := strings.Join(streamed, ""); joined != result.Answer {
		t.Fatalf("streamed %q, want exactly the answer %q", joined, result.Answer)
	}
	if !strings.Contains(traceText(result.Trace), "Round 2") {
		t.Fatalf("the run must have taken two rounds to exercise this:\n%s", traceText(result.Trace))
	}
}

func TestMediumSpecMatchesPlan(t *testing.T) {
	spec := Medium()
	if spec.Label != "medium" {
		t.Fatalf("label = %q", spec.Label)
	}
	if spec.SCAMaxRounds != 3 {
		t.Fatalf("SCAMaxRounds = %d, want 3", spec.SCAMaxRounds)
	}
	if spec.ActionMaxTurns != 12 {
		t.Fatalf("ActionMaxTurns = %d, want 12 (per-turn allowance, see Medium)", spec.ActionMaxTurns)
	}
	if spec.SnippetsPerQuery != 6 {
		t.Fatalf("SnippetsPerQuery = %d, want 6", spec.SnippetsPerQuery)
	}
	if spec.UseFanout {
		t.Fatal("medium must not fan out (no planner / prefetch)")
	}
	if !spec.EnableSCA {
		t.Fatal("medium must enable SCA")
	}
	for _, tool := range ToolNames() {
		if !contains(spec.Tools, tool) {
			t.Fatalf("tools %v missing %q", spec.Tools, tool)
		}
	}
	if len(spec.Tools) != 4 {
		t.Fatalf("tools = %v, want exactly the four of docs/plan.md §6.7", spec.Tools)
	}
	for _, forbidden := range []string{"graph_explore", "navigate_tree", "navigate_structure", "web_search", "calculate"} {
		if contains(spec.Tools, forbidden) {
			t.Fatalf("tool %q must not be in the medium surface", forbidden)
		}
	}
}

func contains(list []string, want string) bool {
	for _, item := range list {
		if item == want {
			return true
		}
	}
	return false
}

func TestRunStopsOnFirstSufficientRound(t *testing.T) {
	s := newStore(t, store.Chunk{
		ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Text",
		Text: "Sufficiency checking decides whether the answer answers the question.",
	})
	loop := &Loop{Store: s}

	result, err := loop.Run(context.Background(), "sufficiency answer")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Verdict != VerdictSufficient {
		t.Fatalf("verdict = %s, want SUFFICIENT", result.Verdict)
	}
	if result.Rounds != 1 {
		t.Fatalf("rounds = %d, want 1", result.Rounds)
	}
	if len(result.Evidence) == 0 {
		t.Fatal("no evidence was gathered")
	}
	if result.Answer == "" {
		t.Fatal("answer is empty")
	}
	if len(result.EvidenceIDs()) != len(result.Evidence) {
		t.Fatal("evidence ids do not match the evidence")
	}
}

// scriptedChecker forces a given verdict per round, so the loop's rewrite
// wiring is tested independently of how coverage happens to score.
type scriptedChecker struct {
	verdicts []Verdict
	missing  [][]string

	// mu guards the fields below: on the complex path every sub-question runs
	// its own loop and they share this checker, so Check is called concurrently.
	mu    sync.Mutex
	calls int
	// questions and poolSizes record what was judged on each call, which is how
	// a test asserts WHICH pool a verdict was taken over — the complex path's
	// invariant that no verdict is ever taken over the merged pool lives here.
	questions []string
	poolSizes []int
}

func (c *scriptedChecker) Check(_ context.Context, question string, info *kbinfo) (Verdict, []string) {
	c.mu.Lock()
	index := c.calls
	c.calls++
	c.questions = append(c.questions, question)
	c.poolSizes = append(c.poolSizes, info.len())
	c.mu.Unlock()

	if index < len(c.verdicts) {
		return c.verdicts[index], c.missing[index]
	}
	return VerdictSufficient, nil
}

func traceText(trace []string) string { return strings.Join(trace, "\n") }

func TestPlanToolsUsesTheModelsChoice(t *testing.T) {
	model := &scriptedModel{planReplies: []*Reply{{ToolCalls: []ToolCall{
		{Name: ToolMetadataSearch, Arguments: metadataFilterArgs(metadataCondition("doc_id", store.OpEqual, "a.pdf"))},
	}}}}
	loop := &Loop{Store: newStore(t), Model: model, Spec: Medium()}

	var trace []string
	calls := loop.planTools(context.Background(), "which tables?", []string{"which tables?"}, nil, nil, nil, &trace)

	if len(calls) != 1 || calls[0].Name != ToolMetadataSearch {
		t.Fatalf("calls = %#v, want the tool the model chose", calls)
	}
	// The model has to see the surface it is choosing from.
	if len(model.planned) != 1 || len(model.planned[0]) != len(ToolNames()) {
		t.Fatalf("offered %#v", model.planned)
	}
	if !strings.Contains(traceText(trace), "the model chose 1 call(s)") {
		t.Fatalf("trace = %v", trace)
	}
}

// TestPlanToolsPromptCarriesTheAvailableMetadata pins the half of the fix a
// schema cannot carry.
//
// The `key` enum says which FIELDS exist. Only the values say which VALUES do,
// and inventing a value — `author_zhao_hui` passed as a doc_id — is what turned
// an index holding four of an author's papers into the sentence "there are no
// papers by this author".
func TestPlanToolsPromptCarriesTheAvailableMetadata(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "2403.03558.pdf", PageNum: 1, BlockType: "Text", Text: "alpha"},
	)
	model := &scriptedModel{planReplies: []*Reply{{ToolCalls: []ToolCall{
		{Name: ToolHybridSearch, Arguments: map[string]any{"query": "q"}},
	}}}}
	loop := &Loop{Store: s, Model: model, Spec: Medium()}

	var trace []string
	loop.planTools(context.Background(), "q", []string{"q"}, nil, nil, nil, &trace)

	if len(model.seen) == 0 || len(model.seen[0]) < 2 {
		t.Fatal("the planning turn never ran")
	}
	prompt := model.seen[0][1].Content
	for _, want := range []string{
		"AVAILABLE METADATA",
		store.FieldDocID,
		store.FieldIndexedAt,
		// A real value, so the model can copy one instead of composing one.
		"2403.03558.pdf",
		// And the prohibition, stated rather than implied.
		"never invent one",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("planning prompt must contain %q:\n%s", want, prompt)
		}
	}
}

// TestPlanToolsPromptOmitsMetadataForAnEmptyIndex: with no documents there are no
// values to copy, and a field list with nothing under it is an invitation to
// compose one — the very failure the block exists to prevent.
func TestPlanToolsPromptOmitsMetadataForAnEmptyIndex(t *testing.T) {
	model := &scriptedModel{planReplies: []*Reply{{ToolCalls: []ToolCall{
		{Name: ToolHybridSearch, Arguments: map[string]any{"query": "q"}},
	}}}}
	loop := &Loop{Store: newStore(t), Model: model, Spec: Medium()}

	var trace []string
	loop.planTools(context.Background(), "q", []string{"q"}, nil, nil, nil, &trace)

	if len(model.seen) == 0 || len(model.seen[0]) < 2 {
		t.Fatal("the planning turn never ran")
	}
	if prompt := model.seen[0][1].Content; strings.Contains(prompt, "AVAILABLE METADATA") {
		t.Fatalf("an empty index must advertise nothing:\n%s", prompt)
	}
}

func TestPlanToolsDropsCallsOutsideTheSurface(t *testing.T) {
	// A model asked to choose tools can name anything. Neither of these is
	// answered by the kernel.
	model := &scriptedModel{planReplies: []*Reply{{ToolCalls: []ToolCall{
		{Name: "graph_explore"},
		{Name: "web_search"},
	}}}}
	loop := &Loop{Store: newStore(t), Model: model, Spec: Medium()}

	var trace []string
	calls := loop.planTools(context.Background(), "q", []string{"q"}, nil, nil, nil, &trace)

	// Nothing usable came back, so the deterministic plan must take over — the
	// alternative is a round that retrieves nothing at all.
	if len(calls) != 1 || calls[0].Name != ToolHybridSearch {
		t.Fatalf("calls = %#v, want the deterministic fallback", calls)
	}
	if !strings.Contains(traceText(trace), "outside the surface") {
		t.Fatalf("trace = %v", trace)
	}
}

func TestPlanToolsKeepsKnownCallsAndReportsTheRest(t *testing.T) {
	model := &scriptedModel{planReplies: []*Reply{{ToolCalls: []ToolCall{
		{Name: ToolGrepSearch, Arguments: map[string]any{"pattern": "DAPO"}},
		{Name: "navigate_tree"},
	}}}}
	loop := &Loop{Store: newStore(t), Model: model, Spec: Medium()}

	var trace []string
	calls := loop.planTools(context.Background(), "q", []string{"q"}, nil, nil, nil, &trace)

	if len(calls) != 1 || calls[0].Name != ToolGrepSearch {
		t.Fatalf("calls = %#v, want only the known call", calls)
	}
	// One bad name must not discard the good call alongside it.
	if !strings.Contains(traceText(trace), "dropped 1 call(s) outside the surface") {
		t.Fatalf("trace = %v", trace)
	}
}

func TestPlanToolsFallsBackWithoutAModel(t *testing.T) {
	loop := &Loop{Store: newStore(t), Spec: Medium()}

	var trace []string
	calls := loop.planTools(context.Background(), "q", []string{"alpha"}, []string{"GB/T 1234"}, nil, nil, &trace)

	if len(calls) != 2 || calls[0].Name != ToolHybridSearch || calls[1].Name != ToolGrepSearch {
		t.Fatalf("calls = %#v, want the deterministic plan", calls)
	}
	// No model means no planning turn, so the trace must not claim one happened.
	if strings.Contains(traceText(trace), "the model") {
		t.Fatalf("trace = %v", trace)
	}
}

func TestPlanToolsFallsBackWhenTheModelDeclines(t *testing.T) {
	// No plan replies: the scripted model answers a planning turn with no calls.
	loop := &Loop{Store: newStore(t), Model: &scriptedModel{}, Spec: Medium()}

	var trace []string
	calls := loop.planTools(context.Background(), "q", []string{"alpha"}, nil, nil, nil, &trace)

	if len(calls) != 1 || calls[0].Name != ToolHybridSearch {
		t.Fatalf("calls = %#v, want the deterministic fallback", calls)
	}
	if !strings.Contains(traceText(trace), "chose no tool") {
		t.Fatalf("trace = %v", trace)
	}
}

func TestRunUsesTheModelsToolChoice(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Text", Text: "alpha passage"},
		store.Chunk{ChunkID: "c1", DocID: "b.pdf", PageNum: 2, BlockType: "Table", Text: "beta table"},
	)
	model := &scriptedModel{
		planReplies: []*Reply{{ToolCalls: []ToolCall{
			{Name: ToolMetadataSearch, Arguments: metadataFilterArgs(
				metadataCondition("doc_id", store.OpEqual, "b.pdf"))},
		}}},
		replies: []*Reply{{Content: "The table says beta [1]."}},
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{}}

	result, err := loop.Run(context.Background(), "which tables are there?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The metadata leg selects documents, and only the filtered one's chunks may
	// reach the evidence pool — a filter that also dragged in a.pdf would make
	// the tool a slower spelling of "list everything".
	if len(result.Evidence) != 1 || result.Evidence[0].Chunk.ChunkID != "c1" {
		t.Fatalf("evidence = %#v, want the table the model asked for", result.Evidence)
	}
	if !strings.Contains(traceText(result.Trace), "metadata_search") {
		t.Fatalf("trace = %v", result.Trace)
	}
}

// TestRunNeverAcceptsAnEmptyPoolAsSufficient is the regression for the failure
// that made an index holding five of an author's papers answer "there are no
// papers by this author".
//
// The checker below says SUFFICIENT on every call, which is not a straw man: the
// real one does. Measured, Laya returns sufficient at 0.93 confidence for a
// answer that says "the evidence does not answer this" — correctly, because that
// answer IS a coherent answer to the question the checker is asked. Whether the
// corpus was actually looked at is a fact the checker is never shown, so the
// loop has to hold it.
//
// A run that consults the checker on an empty pool stops after one round.
func TestRunNeverAcceptsAnEmptyPoolAsSufficient(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "alpha.pdf", PageNum: 1, Text: "alpha passage"})
	model := &scriptedModel{
		planReplies: []*Reply{
			{ToolCalls: []ToolCall{{Name: ToolMetadataSearch, Arguments: metadataFilterArgs(
				metadataCondition("doc_id", store.OpEqual, "does-not-exist.pdf"))}}},
		},
		// Round 1 writes NO answer (the pool is empty), so these are consumed as
		// the rewriter's query and then round 2's answer — one generation fewer
		// than the rounds would otherwise cost.
		replies: []*Reply{{Content: "alpha"}, {Content: "alpha says so [1]."}},
	}
	checker := &scriptedChecker{verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}
	loop := &Loop{Store: s, Model: model, Checker: checker}

	result, err := loop.Run(context.Background(), "what does alpha say?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Rounds != 2 {
		t.Fatalf("rounds = %d, want 2 — an empty round 1 must be retried, not answered", result.Rounds)
	}
	if len(result.Evidence) == 0 {
		t.Fatalf("evidence = %#v, want the second round's passage", result.Evidence)
	}
	if checker.calls != 1 {
		t.Fatalf("checker calls = %d, want 1 — it must not be asked to judge an empty pool", checker.calls)
	}
	if !strings.Contains(traceText(result.Trace), "no evidence to judge") {
		t.Fatalf("trace must say why the checker was skipped:\n%s", traceText(result.Trace))
	}
}

// TestRunWritesNoDraftOverAnEmptyPool pins the other half of that fix, and it is
// the half the user sees.
//
// The answer exists to be judged, and an empty pool is not judged — so generating
// one costs a generation call (tens of seconds) and does nothing except get
// STREAMED to the UI as though it were the answer. A answer reading "the evidence
// does not answer this" appears where the answer goes, the next round replaces
// it, and the reader watches the answer change its mind.
func TestRunWritesNoDraftOverAnEmptyPool(t *testing.T) {
	model := &streamingScriptedModel{scriptedModel: scriptedModel{
		replies: []*Reply{{Content: "the evidence does not answer this"}},
	}}
	loop := &Loop{Store: newStore(t), Model: model, Spec: Spec{SCAMaxRounds: 1}}
	loop.OnAnswerDelta = func(delta string) {
		t.Fatalf("a answer was streamed over an empty pool: %q", delta)
	}

	result, err := loop.Run(context.Background(), "anything")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Answer != "" {
		t.Fatalf("answer = %q, want none over an empty pool", result.Answer)
	}
	if result.Verdict != VerdictInsufficient {
		t.Fatalf("verdict = %s, want INSUFFICIENT", result.Verdict)
	}
	if !strings.Contains(traceText(result.Trace), "no evidence to judge") {
		t.Fatalf("trace must record why no verdict was asked for:\n%s", traceText(result.Trace))
	}
	// The generator must not have been called at all, which is the saving.
	if model.calls != 0 || len(model.deltas) != 0 {
		t.Fatalf("model calls = %d, deltas = %d, want none", model.calls, len(model.deltas))
	}
}

// TestRunTellsTheNextRoundWhatWasAlreadyTried: the planner system prompt has
// always said "do not repeat a call that was already made", and the prompt never
// said which calls those were.
//
// The omission only bites when the evidence pool is empty — and that is exactly
// when the model most needs it, because the evidence sample it would otherwise
// learn from is empty too, leaving the call that just returned nothing as the
// single most likely next action.
func TestRunTellsTheNextRoundWhatWasAlreadyTried(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "alpha.pdf", PageNum: 1, Text: "alpha passage"})
	model := &scriptedModel{
		planReplies: []*Reply{
			{ToolCalls: []ToolCall{{Name: ToolMetadataSearch, Arguments: metadataFilterArgs(
				metadataCondition("doc_id", store.OpEqual, "does-not-exist.pdf"))}}},
		},
		// Round 1 writes no answer, so the first entry is the rewriter's query.
		replies: []*Reply{{Content: "alpha"}, {Content: "done [1]"}},
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}}

	if _, err := loop.Run(context.Background(), "what does alpha say?"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	var second string
	planningTurns := 0
	for _, messages := range model.seen {
		if len(messages) < 2 || messages[0].Content != toolPlanSystemPrompt {
			continue
		}
		planningTurns++
		if planningTurns == 2 {
			second = messages[1].Content
		}
	}
	if planningTurns != 2 {
		t.Fatalf("planning turns = %d, want 2", planningTurns)
	}

	for _, want := range []string{"Already tried this run", "does-not-exist.pdf", "returned nothing"} {
		if !strings.Contains(second, want) {
			t.Fatalf("round 2's planning prompt must contain %q:\n%s", want, second)
		}
	}
}

// TestRunRefusesToRepeatAnIdenticalCall: the prompt says not to repeat a call,
// and a measured run repeated one anyway — the exact call that had already
// returned nothing, with that call listed and marked "returned nothing" in the
// prompt it was answering.
//
// A repeat cannot add information, because the index does not change within a
// run, so the loop refuses it instead of paying for it. That makes the saving a
// property of the loop rather than of the model's instruction-following.
func TestRunRefusesToRepeatAnIdenticalCall(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "alpha.pdf", PageNum: 1, Text: "alpha passage"})
	// The same map instance twice, so the two calls are identical by
	// construction rather than by coincidence.
	empty := metadataFilterArgs(metadataCondition("doc_id", store.OpEqual, "does-not-exist.pdf"))

	model := &scriptedModel{
		planReplies: []*Reply{
			{ToolCalls: []ToolCall{{Name: ToolMetadataSearch, Arguments: empty}}},
			{ToolCalls: []ToolCall{
				{Name: ToolMetadataSearch, Arguments: empty},
				{Name: ToolHybridSearch, Arguments: map[string]any{"query": "alpha"}},
			}},
		},
		replies: []*Reply{{Content: "r1"}, {Content: "alpha"}, {Content: "alpha [1]"}},
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}}

	result, err := loop.Run(context.Background(), "what does alpha say?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The call beside it must still run: refusing the repeat may not refuse the
	// round.
	if len(result.Evidence) == 0 {
		t.Fatal("the non-repeated call in the same round must still run")
	}
	trace := traceText(result.Trace)
	if !strings.Contains(trace, "not repeating it") {
		t.Fatalf("the refusal must be in the trace:\n%s", trace)
	}
	if got := strings.Count(trace, "[Tool] metadata_search"); got != 1 {
		t.Fatalf("metadata_search ran %d time(s), want 1:\n%s", got, trace)
	}
}

func TestRunRewritesThenStops(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Text",
			Text: "Sufficiency checking decides whether the answer answers the question."},
		store.Chunk{ChunkID: "c1", DocID: "a.pdf", PageNum: 2, BlockType: "Text",
			Text: "Quokka wombat narwhal appear only in this second passage."},
	)

	model := &scriptedModel{replies: []*Reply{
		{Content: "quokka wombat narwhal"}, // round 1 rewrite
	}}
	checker := &scriptedChecker{
		verdicts: []Verdict{VerdictInsufficient, VerdictSufficient},
		missing:  [][]string{{"quokka", "wombat", "narwhal"}, nil},
	}

	loop := &Loop{Store: s, Model: model, Checker: checker}
	result, err := loop.Run(context.Background(), "sufficiency answer quokka wombat narwhal")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Rounds != 2 {
		t.Fatalf("rounds = %d, want 2 (rewrite should have run once)", result.Rounds)
	}
	if result.Verdict != VerdictSufficient {
		t.Fatalf("verdict = %s, want SUFFICIENT after the rewrite", result.Verdict)
	}
	if len(result.Queries) != 1 || result.Queries[0] != "quokka wombat narwhal" {
		t.Fatalf("round 2 queries = %v", result.Queries)
	}
	if !strings.Contains(result.Answer, "Quokka") {
		t.Fatalf("answer did not come from the model: %q", result.Answer)
	}
	if model.calls != 1 {
		// One generation for the whole run, and it is the rewriter's: the planner
		// falls back with no scripted plan, and the answer that used to be the
		// second call no longer exists. This is the saving the change is for.
		t.Fatalf("model calls = %d, want 1 (no per-round answer)", model.calls)
	}
	if len(result.Evidence) != 2 {
		t.Fatalf("evidence = %d passages, want 2", len(result.Evidence))
	}
}

func TestRunStopsAtSCAMaxRoundsWithoutEvidence(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", Text: "unrelated content about gardening"})

	loop := &Loop{Store: s}
	result, err := loop.Run(context.Background(), "quokka wombat narwhal")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Verdict == VerdictSufficient {
		t.Fatal("verdict should not be SUFFICIENT")
	}
	if result.Rounds != Medium().SCAMaxRounds {
		t.Fatalf("rounds = %d, want %d", result.Rounds, Medium().SCAMaxRounds)
	}
}

// The planner has to be told that a turn can carry several calls, and that the
// number is per turn. Without it a small model issues one call per turn and
// burns a round on each; measured, the prompt it answered said only "choose at
// most N calls", which reads as a cap on the whole run.
func TestToolPlanPromptAsksForBatchedCalls(t *testing.T) {
	if !strings.Contains(toolPlanSystemPrompt, "ONE turn") {
		t.Fatal("the planner is not told to batch independent calls into one turn")
	}
	for _, phrase := range []string{"per spelling", "read one document"} {
		if !strings.Contains(toolPlanSystemPrompt, phrase) {
			t.Fatalf("the planner is not told what makes calls independent: missing %q", phrase)
		}
	}

	rendered := renderToolPlanPrompt("q", []string{"rewritten query"}, nil, nil, nil, 12, "")
	if !strings.Contains(rendered, "up to 12 call(s)") {
		t.Fatalf("the budget is not described as a per-turn allowance:\n%s", rendered)
	}
	if !strings.Contains(rendered, "in this one turn") {
		t.Fatalf("the budget line does not say the calls share one turn:\n%s", rendered)
	}
}

func TestRunToolsRespectsActionMaxTurns(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", Text: "alpha passage"},
		store.Chunk{ChunkID: "c1", DocID: "b.pdf", Text: "alpha second passage"},
	)
	loop := &Loop{Store: s, Spec: Spec{SCAMaxRounds: 1, ActionMaxTurns: 2, SnippetsPerQuery: 5}}

	var trace []string

	// More calls than the tool budget allows, and all DISTINCT: an identical
	// call is refused before the budget is consulted (a repeat that ate a turn
	// would spend it on a call the loop already knows the answer to), so five
	// identical calls would never reach the budget at all.
	queries := []string{"alpha one", "alpha two", "alpha three", "alpha four", "alpha five"}
	calls := make([]ToolCall, 0, len(queries))
	for _, query := range queries {
		calls = append(calls, ToolCall{
			Name:      ToolHybridSearch,
			Arguments: map[string]any{"query": query},
		})
	}
	var attempts []attempt
	info := newKBInfo()
	loop.runTools(context.Background(), calls, info, &trace, &attempts)
	evidence := info.pool()

	// Two calls at most, and duplicate hits are merged across calls.
	if len(evidence) != 2 {
		t.Fatalf("evidence = %d, want 2 unique passages", len(evidence))
	}

	stopped := false
	for _, line := range trace {
		if strings.Contains(line, "tool budget spent") {
			stopped = true
		}
	}
	if !stopped {
		t.Fatalf("trace should record the budget stop: %v", trace)
	}
}

func TestRoundCallsAddsGrepLegsForMissingTerms(t *testing.T) {
	calls := roundCalls([]string{"first query"}, []string{"GB/T 1234", "Qwen3-4B", "third term"})

	// One hybrid leg plus at most maxGrepLegs literal probes.
	if len(calls) != 1+maxGrepLegs {
		t.Fatalf("calls = %d, want %d", len(calls), 1+maxGrepLegs)
	}
	if calls[0].Name != ToolHybridSearch || calls[0].Arguments["query"] != "first query" {
		t.Fatalf("first call = %#v", calls[0])
	}
	if calls[1].Name != ToolGrepSearch || calls[1].Arguments["pattern"] != "GB/T 1234" {
		t.Fatalf("second call = %#v", calls[1])
	}
	if calls[2].Name != ToolGrepSearch || calls[2].Arguments["pattern"] != "Qwen3-4B" {
		t.Fatalf("third call = %#v", calls[2])
	}
}

func TestRoundCallsSkipsBlankQueries(t *testing.T) {
	if calls := roundCalls([]string{"  ", ""}, nil); len(calls) != 0 {
		t.Fatalf("calls = %#v, want none", calls)
	}
}

func TestDraftFallsBackWhenModelFails(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", Text: "fallback evidence text"})
	loop := &Loop{Store: s, Model: failingModel{}}

	result, err := loop.Run(context.Background(), "fallback")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(result.Answer, "fallback evidence text") {
		t.Fatalf("expected an extractive answer, got %q", result.Answer)
	}
}

type failingModel struct{}

func (failingModel) Complete(context.Context, []Message, []ToolSpec) (*Reply, error) {
	return nil, context.DeadlineExceeded
}

func TestRunValidatesInput(t *testing.T) {
	if _, err := (&Loop{Store: store.New()}).Run(context.Background(), "   "); err == nil {
		t.Fatal("expected an error for a blank question")
	}
	if _, err := (&Loop{}).Run(context.Background(), "hello"); err == nil {
		t.Fatal("expected an error when no store is configured")
	}
}

func TestCoverageCheckerVerdicts(t *testing.T) {
	checker := CoverageChecker{}
	ctx := context.Background()

	if verdict, _ := checker.Check(ctx, "anything", newKBInfo()); verdict != VerdictInsufficient {
		t.Fatalf("no evidence -> %s, want INSUFFICIENT", verdict)
	}

	evidence := []store.Hit{{Chunk: store.Chunk{ChunkID: "c0", Text: "sufficiency answer answer"}}}

	// One of three question terms is present: 0.33 < the 0.6 threshold.
	verdict, missing := checker.Check(ctx, "sufficiency quokka wombat", infoOf(evidence...))
	if verdict != VerdictInsufficient {
		t.Fatalf("partial coverage -> %s, want INSUFFICIENT", verdict)
	}
	if len(missing) != 2 || missing[0] != "quokka" || missing[1] != "wombat" {
		t.Fatalf("missing = %v, want [quokka wombat]", missing)
	}

	if verdict, _ := checker.Check(ctx, "sufficiency", infoOf(evidence...)); verdict != VerdictSufficient {
		t.Fatalf("full coverage -> %s, want SUFFICIENT", verdict)
	}
}
