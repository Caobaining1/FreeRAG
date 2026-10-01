package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"freerag/internal/store"
)

// The complex path's contract, in three parts:
//
//  1. every sub-question is judged on its OWN pool — never on the union;
//  2. the pools are merged only after every sub-loop has stopped;
//  3. the merge obeys kbinfo's dedup rules, and the answer is written from it.
//
// Sub-loop 1 and sub-loop 2 are given the SAME text, so nothing here depends on
// which goroutine finishes first. That matters: an order-based reply script
// would hand the synthesis text to a rewrite as soon as two loops share one
// model, and the test would fail intermittently rather than meaningfully. Hence
// routedModel below, which answers by what the turn is for.

// routedModel answers by the purpose of the turn, read off the system prompt,
// instead of by call order. The phrases are the openings of
// decomposeSystemPrompt, synthesizeSystemPrompt, answerSystemPrompt and
// rewriteSystemPrompt.
//
// It also counts which writer was asked, which is how the tests below assert
// that a sub-loop never writes an answer. The two writers are told apart by
// their system prompt: "You write ONE answer" is the synthesis, "You answer the
// user's question using ONLY" is the loop's own writer.
type routedModel struct {
	sub       string
	final     string
	decompose string

	mu        sync.Mutex
	answers   int
	syntheses int
	rewrites  int
}

func (m *routedModel) Complete(_ context.Context, messages []Message, tools []ToolSpec) (*Reply, error) {
	if len(tools) > 0 {
		// A planning turn: no calls means the loop uses its deterministic plan,
		// which is what keeps these tests about orchestration.
		return &Reply{}, nil
	}
	for _, message := range messages {
		if message.Role != RoleSystem {
			continue
		}
		switch {
		case strings.Contains(message.Content, "You break a multi-part question"):
			return &Reply{Content: m.decompose}, nil
		case strings.Contains(message.Content, "You write ONE answer"):
			m.count(func() { m.syntheses++ })
			return &Reply{Content: m.final}, nil
		case strings.Contains(message.Content, "You answer the user's question using ONLY"):
			m.count(func() { m.answers++ })
			return &Reply{Content: m.sub}, nil
		case strings.Contains(message.Content, "You write ONE search query"):
			m.count(func() { m.rewrites++ })
			// An empty rewrite stops the loop after the round it came from,
			// which is what pins the SCA call count in the insufficient test.
			return &Reply{Content: ""}, nil
		}
		break
	}
	return &Reply{Content: m.sub}, nil
}

func (m *routedModel) count(f func()) {
	m.mu.Lock()
	defer m.mu.Unlock()
	f()
}

// writers reports how many times each writer was asked to write.
func (m *routedModel) writers() (answers, syntheses, rewrites int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.answers, m.syntheses, m.rewrites
}

func complexRouter() Router {
	return Router{Decide: (&recordingDecide{choice: complexOption + ": several", probability: 0.9}).decide}
}

// A verdict must be taken per sub-question, over that sub-question's own pool.
// Nothing may ask Laya about the merged pool: a single INSUFFICIENT over the
// union cannot say which sub-question is thin, so the loop would have nothing to
// rewrite, and "is this enough for the whole compound question" is not a
// decision the checkpoint was trained on.
func TestFlowComplexPathJudgesEachSubQuestionOnItsOwnPool(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"},
		store.Chunk{ChunkID: "c1", DocID: "b.pdf", PageNum: 1, Text: "alpha passage about alpha"},
	)
	model := &routedModel{
		decompose: `["What is alpha?", "What is alpha?"]`,
		sub:       "sub answer [1]",
		final:     "merged answer [1]",
	}
	checker := &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient, VerdictSufficient},
		missing:  [][]string{nil, nil},
	}
	loop := &Loop{Store: s, Model: model, Checker: checker, Spec: Medium()}

	flow, err := NewFlow(FlowDeps{Loop: loop, Router: complexRouter(), Model: model})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	const question = "a big compound question about alpha"
	result, err := flow.Run(context.Background(), question)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// One round each, so exactly two verdicts. A third call would be a
	// judgement over the merged pool.
	if checker.calls != 2 {
		t.Fatalf("the checker was called %d time(s), want 2 (one per sub-question); questions=%q",
			checker.calls, checker.questions)
	}
	for _, judged := range checker.questions {
		if judged == question {
			t.Fatalf("the top-level question was passed to the checker: %q", checker.questions)
		}
		if judged != "What is alpha?" {
			t.Fatalf("unexpected question judged: %q", judged)
		}
	}
	for i, size := range checker.poolSizes {
		if size == 0 {
			t.Fatalf("verdict %d was asked about an empty pool", i)
		}
	}
	if result.Verdict != VerdictSufficient {
		t.Fatalf("verdict = %q, want sufficient (both sub-questions were)", result.Verdict)
	}
	if len(result.NotSufficient) != 0 {
		t.Fatalf("not-sufficient = %#v, want none", result.NotSufficient)
	}
	if result.Answer != "merged answer [1]" {
		t.Fatalf("answer = %q, want the synthesis", result.Answer)
	}
}

// The merge is kbinfo's, not an identity-only sweep: the same text twice inside
// one document collapses, while the same text in two documents is two citations
// and survives — the answer cites by index, so dropping the second would lose a
// link to a document that really does say it. Both sub-loops retrieve all three
// chunks, so every passage arrives in the union twice.
func TestFlowComplexPathMergesWithKBInfoDedup(t *testing.T) {
	shared := "alpha framework overview"
	s := newStore(t,
		// Same document, same text, two chunk ids: a caption, and the merged
		// figure block that absorbed it. Identity dedup keeps both; kbinfo one.
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: shared},
		store.Chunk{ChunkID: "c9", DocID: "a.pdf", PageNum: 1, Text: shared},
		store.Chunk{ChunkID: "c1", DocID: "b.pdf", PageNum: 1, Text: shared},
	)
	model := &routedModel{
		decompose: `["What is alpha?", "What is alpha?"]`,
		sub:       "sub answer [1]",
		final:     "merged answer [1]",
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient, VerdictSufficient},
		missing:  [][]string{nil, nil}}, Spec: Medium()}

	flow, err := NewFlow(FlowDeps{Loop: loop, Router: complexRouter(), Model: model})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	result, err := flow.Run(context.Background(), "a big compound question about alpha")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(result.Evidence) != 2 {
		t.Fatalf("merged pool holds %d passage(s), want 2 — one per document, with the twice-told "+
			"text inside a.pdf collapsed and the repeat across a.pdf/b.pdf kept; got %s",
			len(result.Evidence), describeHits(result.Evidence))
	}
	seen := map[string]int{}
	for _, hit := range result.Evidence {
		seen[hit.Chunk.DocID+"\x00"+hit.Chunk.ChunkID]++
	}
	for key, count := range seen {
		if count != 1 {
			t.Fatalf("the merge left a duplicate passage (%s x%d) in %s",
				key, count, describeHits(result.Evidence))
		}
	}
	docs := map[string]bool{}
	for _, hit := range result.Evidence {
		docs[hit.Chunk.DocID] = true
	}
	if !docs["a.pdf"] || !docs["b.pdf"] {
		t.Fatalf("the merge dropped a document: %s", describeHits(result.Evidence))
	}
}

// A sub-question that never reached SUFFICIENT is reported, and its evidence
// still joins the pool. The alternative to answering with thin evidence is not
// answering at all, and a search tool that returns nothing for a partly covered
// question is worse than one that says which part is thin.
func TestFlowComplexPathReportsASubQuestionThatStayedInsufficient(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"})
	model := &routedModel{
		decompose: `["What is alpha?", "What is alpha?"]`,
		sub:       "sub answer [1]",
		final:     "merged answer [1]",
	}
	// Every round of every sub-loop must be scripted: the helper falls back to
	// SUFFICIENT once its script runs out, and a rewrite with no model reply
	// still produces terms, so the sub-loops do use all three of Medium()'s
	// rounds. 3 rounds x 2 sub-questions.
	verdicts := make([]Verdict, 6)
	missing := make([][]string, 6)
	for i := range verdicts {
		verdicts[i] = VerdictInsufficient
		missing[i] = []string{"alpha"}
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: verdicts, missing: missing,
	}, Spec: Medium()}

	flow, err := NewFlow(FlowDeps{Loop: loop, Router: complexRouter(), Model: model})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	result, err := flow.Run(context.Background(), "a big compound question about alpha")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Verdict != VerdictInsufficient {
		t.Fatalf("verdict = %q, want insufficient", result.Verdict)
	}
	if len(result.NotSufficient) != 2 {
		t.Fatalf("not-sufficient = %#v, want both sub-questions", result.NotSufficient)
	}
	if len(result.Evidence) == 0 {
		t.Fatal("the evidence of an insufficient sub-question was dropped; that part of the answer would be lost")
	}
	if result.Answer != "merged answer [1]" {
		t.Fatalf("answer = %q, want an answer anyway", result.Answer)
	}
}

// On the complex path exactly ONE text is written: the synthesis. A sub-loop
// retrieves and is judged, and writing its own answer as well cost N full
// generations whose only reader was the synthesis prompt — about a minute each
// on the reference machine, and text the merged pool already contains.
func TestFlowComplexPathNeverWritesASubAnswer(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"})
	model := &routedModel{
		decompose: `["What is alpha?", "What is alpha?"]`,
		sub:       "sub answer [1]",
		final:     "merged answer [1]",
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient, VerdictSufficient},
		missing:  [][]string{nil, nil}}, Spec: Medium()}

	flow, err := NewFlow(FlowDeps{Loop: loop, Router: complexRouter(), Model: model})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	result, err := flow.Run(context.Background(), "a big compound question about alpha")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	answers, syntheses, _ := model.writers()
	if answers != 0 {
		t.Fatalf("a sub-loop wrote %d answer(s); the complex path writes only the synthesis, "+
			"and each one costs a full generation", answers)
	}
	if syntheses != 1 {
		t.Fatalf("the synthesis was written %d time(s), want exactly 1", syntheses)
	}
	if result.Answer != "merged answer [1]" {
		t.Fatalf("answer = %q", result.Answer)
	}
	if len(result.Evidence) == 0 {
		t.Fatal("the sub-loops retrieved nothing to synthesise from")
	}
}

// The simple path has no synthesis — one loop, one answer — so the writer that
// the complex path must not use is exactly the one this path must use.
func TestFlowSimplePathWritesExactlyOneAnswer(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"})
	model := &routedModel{sub: "the answer [1]", final: "should not be written"}
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

	answers, syntheses, _ := model.writers()
	if answers != 1 || syntheses != 0 {
		t.Fatalf("writers: answers=%d syntheses=%d, want 1 and 0 on the simple path", answers, syntheses)
	}
	if result.Answer != "the answer [1]" {
		t.Fatalf("answer = %q", result.Answer)
	}
}

func describeHits(hits []store.Hit) string {
	rendered := ""
	for i, hit := range hits {
		if i > 0 {
			rendered += ", "
		}
		rendered += hit.Chunk.DocID + "/" + hit.Chunk.ChunkID
	}
	return "[" + rendered + "]"
}
