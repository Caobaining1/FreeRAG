package agent

import (
	"context"
	"strings"
	"testing"

	"freerag/internal/store"
)

// ---- routing ----

func TestRouterHeuristicClassifies(t *testing.T) {
	router := Router{} // no Laya: the heuristic path, i.e. a machine without the checkpoint
	cases := []struct {
		question string
		want     Route
	}{
		{"What is alpha?", RouteSimple},
		{"赵慧为作者的论文有哪几篇，分别介绍了什么", RouteComplex},
		{"Compare GRPO and PPO", RouteComplex},
		{"Does the paper mention gamma? And what about delta?", RouteComplex},
	}
	for _, tc := range cases {
		if got := router.Route(context.Background(), tc.question); got != tc.want {
			t.Errorf("Route(%q) = %s, want %s", tc.question, got, tc.want)
		}
	}
}

func TestRouterTrustsLaya(t *testing.T) {
	recorder := &recordingDecide{choice: complexOption + ": several parts", probability: 0.9}
	router := Router{Decide: recorder.decide}
	if got := router.Route(context.Background(), "What is alpha?"); got != RouteComplex {
		t.Fatalf("Route = %s, want complex (Laya said so)", got)
	}
}

func TestRouterIgnoresACoinFlip(t *testing.T) {
	// A low-confidence routing answer is discarded for the heuristic: being
	// misrouted only changes the shape of the search, so the safe default is the
	// one the machine can compute.
	recorder := &recordingDecide{choice: complexOption + ": maybe", probability: 0.2}
	router := Router{Decide: recorder.decide}
	if got := router.Route(context.Background(), "What is alpha?"); got != RouteSimple {
		t.Fatalf("Route = %s, want simple (the low-confidence answer is ignored)", got)
	}
}

// ---- decomposition ----

func TestDecomposeParsesAJSONArray(t *testing.T) {
	model := &scriptedModel{replies: []*Reply{{Content: `["first part", "second part"]`}}}
	subs := Decompose(context.Background(), model, "a big question", 4)
	if len(subs) != 2 || subs[0] != "first part" || subs[1] != "second part" {
		t.Fatalf("subs = %#v", subs)
	}
}

func TestDecomposeParsesABulletedList(t *testing.T) {
	model := &scriptedModel{replies: []*Reply{{Content: "1. first part\n2. second part"}}}
	subs := Decompose(context.Background(), model, "a big question", 4)
	if len(subs) != 2 || subs[0] != "first part" || subs[1] != "second part" {
		t.Fatalf("subs = %#v", subs)
	}
}

func TestDecomposeFallsBackToTheQuestion(t *testing.T) {
	// Prose that is neither JSON nor a list must NOT become a sub-question: the
	// original question is a better thing to search for than the model's
	// commentary about it.
	model := &scriptedModel{replies: []*Reply{{Content: "I cannot split this."}}}
	subs := Decompose(context.Background(), model, "the real question", 4)
	if len(subs) != 1 || subs[0] != "the real question" {
		t.Fatalf("subs = %#v, want the original question", subs)
	}
}

func TestDecomposeWithoutAModelReturnsTheQuestion(t *testing.T) {
	subs := Decompose(context.Background(), nil, "the real question", 4)
	if len(subs) != 1 || subs[0] != "the real question" {
		t.Fatalf("subs = %#v", subs)
	}
}

// ---- candidate building ----

func TestBuildCandidatesSkipsTriedCalls(t *testing.T) {
	tried := ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "q"}}
	candidates := buildCandidates([]string{"q"}, []string{"term"}, nil, []attempt{{Call: tried}})

	for _, candidate := range candidates {
		if sameToolCall(candidate.Call, tried) {
			t.Fatal("a call already tried this run was offered again")
		}
	}
	found := false
	for _, candidate := range candidates {
		if candidate.Call.Name == ToolGrepSearch {
			found = true
		}
	}
	if !found {
		t.Fatal("the missing-term grep was dropped")
	}
}

func TestBuildCandidatesOffersDocumentsFromThePool(t *testing.T) {
	evidence := []store.Hit{
		{Chunk: store.Chunk{ChunkID: "c0", DocID: "a.pdf"}},
		{Chunk: store.Chunk{ChunkID: "c1", DocID: "b.pdf"}},
	}
	candidates := buildCandidates([]string{"q"}, nil, evidence, nil)
	var listed []string
	for _, candidate := range candidates {
		if candidate.Call.Name == ToolListChunks {
			listed = append(listed, candidate.Call.Arguments["doc_id"].(string))
		}
	}
	if len(listed) != 2 || listed[0] != "a.pdf" || listed[1] != "b.pdf" {
		t.Fatalf("listed = %#v", listed)
	}
}

// A probe for a word that matches the whole corpus is not a probe.
//
// The checker reports the terms an answer is missing, and on the MultiHop-RAG dev split
// it reported "the" — which became grep("the"). Six unrelated questions were then
// answered from pools whose first passage was the same sports article, the query the
// rewriter had produced was never searched at all, and those questions scored 0.11-0.23
// with 100% of their evidence sitting in the index (docs/plan.md §13.19).
func TestBuildCandidatesRefusesUndiscriminatingGreps(t *testing.T) {
	missing := []string{"the", "of", "it", "AI", "information", "Anthropic"}
	candidates := buildCandidates([]string{"a real query"}, missing, nil, nil)

	var patterns []string
	for _, candidate := range candidates {
		if candidate.Call.Name == ToolGrepSearch {
			patterns = append(patterns, candidate.Call.Arguments["pattern"].(string))
		}
	}
	if len(patterns) != 1 || patterns[0] != "Anthropic" {
		t.Fatalf("grep patterns = %#v, want only the discriminating term", patterns)
	}
}

// The search candidates are untouched by the filter: a run whose only missing terms are
// function words still has its queries to fall back on.
func TestBuildCandidatesKeepsSearchesWhenEveryTermIsRefused(t *testing.T) {
	candidates := buildCandidates([]string{"the query the rewriter produced"}, []string{"the", "and"}, nil, nil)

	if len(candidates) != 1 || candidates[0].Call.Name != ToolHybridSearch {
		t.Fatalf("candidates = %#v, want exactly the one search", candidates)
	}
}

// ---- Laya tool choice ----

func TestLayaToolChooserPicksACandidate(t *testing.T) {
	candidates := []Candidate{
		{Call: ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "q"}}, Label: `search("q")`, Detail: "d"},
		{Call: ToolCall{Name: ToolGrepSearch, Arguments: map[string]any{"pattern": "t"}}, Label: `grep("t")`, Detail: "d"},
	}
	recorder := &recordingDecide{choice: candidates[1].Label + ": desc", probability: 0.8}
	chooser := LayaToolChooser{Decide: recorder.decide}

	index, err := chooser.Choose(context.Background(), "q", candidates, nil, "", nil)
	if err != nil {
		t.Fatalf("Choose: %v", err)
	}
	if index != 1 {
		t.Fatalf("index = %d, want 1", index)
	}
	if _, ok := recorder.criteria[stopOption]; !ok {
		t.Fatal("the stop option was not offered to Laya")
	}
}

func TestLayaToolChooserCanStop(t *testing.T) {
	candidates := []Candidate{{Call: ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "q"}}, Label: `search("q")`}}
	recorder := &recordingDecide{choice: stopOption + ": done", probability: 0.8}
	chooser := LayaToolChooser{Decide: recorder.decide}

	// A pool, because stopping means "the evidence already answers the question"
	// and there has to be evidence for that to be sayable (see
	// TestLayaToolChooserRefusesToStopWithNothingRetrieved).
	index, err := chooser.Choose(context.Background(), "q", candidates, oneHit().pool(), "", nil)
	if err != nil {
		t.Fatalf("Choose: %v", err)
	}
	if index != -1 {
		t.Fatalf("index = %d, want -1 (stop)", index)
	}
}

// Stopping before anything has been retrieved is refused.
//
// "stop" is defined to Laya as "the evidence already answers the question", so
// choosing it over an empty pool is a contradiction — and obeying it ends the
// round with no retrieval at all. Measured on a real complex question after the
// question type was fixed: both sub-questions stopped on their first round, before
// any search had run, and the run answered from an empty pool.
//
// Refused as "could not decide", which the loop answers by planning a retrieval,
// rather than by picking a candidate the model did not choose.
func TestLayaToolChooserRefusesToStopWithNothingRetrieved(t *testing.T) {
	candidates := []Candidate{{Call: ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "q"}}, Label: `search("q")`}}
	recorder := &recordingDecide{choice: stopOption, probability: 0.9}
	chooser := LayaToolChooser{Decide: recorder.decide}

	index, err := chooser.Choose(context.Background(), "q", candidates, nil, "", nil)
	if err == nil {
		t.Fatalf("stopping over an empty pool was accepted (index=%d)", index)
	}
	if index != -1 {
		t.Fatalf("index = %d, want -1", index)
	}
	if !strings.Contains(err.Error(), "nothing retrieved") {
		t.Fatalf("the refusal does not say why: %v", err)
	}
}

func TestLayaToolChooserRejectsAnOffMenuChoice(t *testing.T) {
	candidates := []Candidate{{Call: ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "q"}}, Label: `search("q")`}}
	recorder := &recordingDecide{choice: "something else entirely", probability: 0.8}
	chooser := LayaToolChooser{Decide: recorder.decide}

	if _, err := chooser.Choose(context.Background(), "q", candidates, nil, "", nil); err == nil {
		t.Fatal("a choice outside the candidate set must be an error, not a silent pick")
	}
}

// ---- the eino flow ----

func TestNewFlowRequiresALoop(t *testing.T) {
	if _, err := NewFlow(FlowDeps{}); err == nil {
		t.Fatal("a flow without a loop must be an error")
	}
}

func TestFlowSimplePathRunsOneLoop(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"})
	model := &scriptedModel{replies: []*Reply{{Content: "the answer [1]"}}}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}, Spec: Medium()}

	flow, err := NewFlow(FlowDeps{Loop: loop, Model: model})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	result, err := flow.Run(context.Background(), "What is alpha?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Route != RouteSimple {
		t.Fatalf("route = %s, want simple", result.Route)
	}
	if len(result.SubQuestions) != 0 {
		t.Fatalf("sub-questions = %#v, want none on the simple path", result.SubQuestions)
	}
	if result.Answer != "the answer [1]" {
		t.Fatalf("answer = %q", result.Answer)
	}
}

func TestFlowComplexPathFansOutAndSynthesizes(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"})
	// Two replies, not three: a sub-loop retrieves and is judged, and no longer
	// writes an answer of its own (see Loop.RunRetrieval), so the text after
	// decompose is the synthesis directly.
	model := &scriptedModel{replies: []*Reply{
		{Content: `["What is alpha?"]`}, // decompose
		{Content: "merged answer [1]"},  // synthesize
	}}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}, Spec: Medium()}

	// Force the complex branch: the question is short enough that the heuristic
	// alone would call it simple.
	router := Router{Decide: (&recordingDecide{choice: complexOption + ": several", probability: 0.9}).decide}

	flow, err := NewFlow(FlowDeps{Loop: loop, Router: router, Model: model})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	result, err := flow.Run(context.Background(), "What is alpha?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Route != RouteComplex {
		t.Fatalf("route = %s, want complex", result.Route)
	}
	if len(result.SubQuestions) != 1 || result.SubQuestions[0] != "What is alpha?" {
		t.Fatalf("sub-questions = %#v", result.SubQuestions)
	}
	if result.Answer != "merged answer [1]" {
		t.Fatalf("answer = %q, want the synthesis", result.Answer)
	}
	if len(result.Evidence) == 0 {
		t.Fatal("the fan-out gathered no evidence")
	}
}

// The sub-loops must not publish their own answers: only the synthesis does.
func TestFlowComplexPathDoesNotStreamSubAnswers(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"})
	model := &streamingScriptedModel{scriptedModel: scriptedModel{replies: []*Reply{
		{Content: `["What is alpha?"]`},
		{Content: "merged answer [1]"}, // the synthesis: the one thing published
	}}}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}, Spec: Medium()}
	router := Router{Decide: (&recordingDecide{choice: complexOption + ": several", probability: 0.9}).decide}

	var streamed []string
	flow, err := NewFlow(FlowDeps{
		Loop: loop, Router: router, Model: model,
		OnAnswerDelta: func(delta string) { streamed = append(streamed, delta) },
	})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	if _, err := flow.Run(context.Background(), "What is alpha?"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	joined := strings.Join(streamed, "")
	if joined != "merged answer [1]" {
		t.Fatalf("streamed %q, want exactly the synthesis and nothing else", joined)
	}
	if strings.Contains(joined, "sub answer") {
		t.Fatal("a sub-loop's answer was published as the answer")
	}
}
