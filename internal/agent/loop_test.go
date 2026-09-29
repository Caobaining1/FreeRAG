package agent

import (
	"context"
	"strings"
	"testing"

	"freerag/internal/store"
)

// scriptedModel replays canned replies so the graph's control flow can be
// verified without a model download.
//
// Replies are routed by what the turn is for, not by call order: a turn that
// offers tools is a tool-planning turn, everything else is a text turn (draft or
// rewrite). Routing by order alone would make every flow test depend on how many
// planning calls the loop happens to make, which is not what they assert.
type scriptedModel struct {
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
}

func (m *scriptedModel) Complete(_ context.Context, messages []Message, tools []ToolSpec) (*Reply, error) {
	m.seen = append(m.seen, messages)

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

func TestMediumSpecMatchesPlan(t *testing.T) {
	spec := Medium()
	if spec.Label != "medium" {
		t.Fatalf("label = %q", spec.Label)
	}
	if spec.SCAMaxRounds != 3 {
		t.Fatalf("SCAMaxRounds = %d, want 3", spec.SCAMaxRounds)
	}
	if spec.ActionMaxTurns != 8 {
		t.Fatalf("ActionMaxTurns = %d, want 8", spec.ActionMaxTurns)
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
		Text: "Sufficiency checking decides whether the draft answers the question.",
	})
	loop := &Loop{Store: s}

	result, err := loop.Run(context.Background(), "sufficiency draft")
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
	if result.Draft == "" {
		t.Fatal("draft is empty")
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
	calls    int
}

func (c *scriptedChecker) Check(context.Context, string, string, []store.Hit) (Verdict, []string) {
	index := c.calls
	c.calls++
	if index < len(c.verdicts) {
		return c.verdicts[index], c.missing[index]
	}
	return VerdictSufficient, nil
}

func traceText(trace []string) string { return strings.Join(trace, "\n") }

func TestPlanToolsUsesTheModelsChoice(t *testing.T) {
	model := &scriptedModel{planReplies: []*Reply{{ToolCalls: []ToolCall{
		{Name: ToolMetadataSearch, Arguments: map[string]any{"block_type": "Table"}},
	}}}}
	loop := &Loop{Store: newStore(t), Model: model, Spec: Medium()}

	var trace []string
	calls := loop.planTools(context.Background(), "which tables?", []string{"which tables?"}, nil, nil, &trace)

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

func TestPlanToolsDropsCallsOutsideTheSurface(t *testing.T) {
	// A model asked to choose tools can name anything. Neither of these is
	// answered by the kernel.
	model := &scriptedModel{planReplies: []*Reply{{ToolCalls: []ToolCall{
		{Name: "graph_explore"},
		{Name: "web_search"},
	}}}}
	loop := &Loop{Store: newStore(t), Model: model, Spec: Medium()}

	var trace []string
	calls := loop.planTools(context.Background(), "q", []string{"q"}, nil, nil, &trace)

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
	calls := loop.planTools(context.Background(), "q", []string{"q"}, nil, nil, &trace)

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
	calls := loop.planTools(context.Background(), "q", []string{"alpha"}, []string{"GB/T 1234"}, nil, &trace)

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
	calls := loop.planTools(context.Background(), "q", []string{"alpha"}, nil, nil, &trace)

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
		store.Chunk{ChunkID: "c1", DocID: "a.pdf", PageNum: 2, BlockType: "Table", Text: "beta table"},
	)
	model := &scriptedModel{
		planReplies: []*Reply{{ToolCalls: []ToolCall{
			{Name: ToolMetadataSearch, Arguments: map[string]any{"block_type": "Table"}},
		}}},
		replies: []*Reply{{Content: "The table says beta [1]."}},
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{}}

	result, err := loop.Run(context.Background(), "which tables are there?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// metadata_search is the point of this change: before it, the loop could
	// only ever issue hybrid_search and grep_search, so this passage was
	// unreachable through the agentic path.
	if len(result.Evidence) != 1 || result.Evidence[0].Chunk.ChunkID != "c1" {
		t.Fatalf("evidence = %#v, want the table the model asked for", result.Evidence)
	}
	if !strings.Contains(traceText(result.Trace), "metadata_search") {
		t.Fatalf("trace = %v", result.Trace)
	}
}

func TestRunRewritesThenStops(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Text",
			Text: "Sufficiency checking decides whether the draft answers the question."},
		store.Chunk{ChunkID: "c1", DocID: "a.pdf", PageNum: 2, BlockType: "Text",
			Text: "Quokka wombat narwhal appear only in this second passage."},
	)

	model := &scriptedModel{replies: []*Reply{
		{Content: "The draft discusses sufficiency."},      // round 1 draft
		{Content: "quokka wombat narwhal"},                 // round 1 rewrite
		{Content: "Quokka, wombat and narwhal are found."}, // round 2 draft
	}}
	checker := &scriptedChecker{
		verdicts: []Verdict{VerdictInsufficient, VerdictSufficient},
		missing:  [][]string{{"quokka", "wombat", "narwhal"}, nil},
	}

	loop := &Loop{Store: s, Model: model, Checker: checker}
	result, err := loop.Run(context.Background(), "sufficiency draft quokka wombat narwhal")
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
	if !strings.Contains(result.Draft, "Quokka") {
		t.Fatalf("draft did not come from the model: %q", result.Draft)
	}
	if model.calls != 3 {
		t.Fatalf("model calls = %d, want 3", model.calls)
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

func TestRunToolsRespectsActionMaxTurns(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", Text: "alpha passage"},
		store.Chunk{ChunkID: "c1", DocID: "b.pdf", Text: "alpha second passage"},
	)
	loop := &Loop{Store: s, Spec: Spec{SCAMaxRounds: 1, ActionMaxTurns: 2, SnippetsPerQuery: 5}}

	var evidence []store.Hit
	var trace []string
	seen := map[string]bool{}

	// More calls than the tool budget allows.
	calls := make([]ToolCall, 0, 5)
	for i := 0; i < 5; i++ {
		calls = append(calls, ToolCall{
			Name:      ToolHybridSearch,
			Arguments: map[string]any{"query": "alpha"},
		})
	}
	loop.runTools(context.Background(), calls, &evidence, seen, &trace)

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
	if !strings.Contains(result.Draft, "fallback evidence text") {
		t.Fatalf("expected an extractive draft, got %q", result.Draft)
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

	if verdict, _ := checker.Check(ctx, "anything", "", nil); verdict != VerdictInsufficient {
		t.Fatalf("no evidence -> %s, want INSUFFICIENT", verdict)
	}

	evidence := []store.Hit{{Chunk: store.Chunk{ChunkID: "c0", Text: "sufficiency draft answer"}}}
	if verdict, _ := checker.Check(ctx, "sufficiency draft", "", evidence); verdict != VerdictUnknown {
		t.Fatalf("empty draft -> %s, want UNKNOWN", verdict)
	}

	// One of three question terms is present: 0.33 < the 0.6 threshold.
	verdict, missing := checker.Check(ctx, "sufficiency quokka wombat", "a draft", evidence)
	if verdict != VerdictInsufficient {
		t.Fatalf("partial coverage -> %s, want INSUFFICIENT", verdict)
	}
	if len(missing) != 2 || missing[0] != "quokka" || missing[1] != "wombat" {
		t.Fatalf("missing = %v, want [quokka wombat]", missing)
	}

	if verdict, _ := checker.Check(ctx, "sufficiency", "a draft", evidence); verdict != VerdictSufficient {
		t.Fatalf("full coverage -> %s, want SUFFICIENT", verdict)
	}
}
