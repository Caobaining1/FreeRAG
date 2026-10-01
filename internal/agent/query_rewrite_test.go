package agent

import (
	"context"
	"strings"
	"testing"

	"freerag/internal/store"
)

// The two behaviours this file exists to pin, both asked for as requirements:
//
//  1. a simple question never reaches the decomposer;
//  2. the question does not enter the loop as asked — retrieval runs the
//     REWRITTEN queries, and the question itself is not one of them.
//
// They are separate assertions on purpose. The first is about the router's
// branch, the second is about what a round searches, and a change could break
// either without touching the other.

// ---- the rewrite itself ----

func TestRewriteQueriesParsesTheDocumentedShape(t *testing.T) {
	model := &scriptedModel{rewriteReplies: []*Reply{{Content: `{"queries": ["alpha beta", "gamma delta"]}`}}}

	queries, rewritten := RewriteQueries(context.Background(), model, "what is alpha?", maxSearchQueries)
	if !rewritten {
		t.Fatalf("the reply was not treated as a rewrite: %#v", queries)
	}
	if len(queries) != 2 || queries[0] != "alpha beta" || queries[1] != "gamma delta" {
		t.Fatalf("queries = %#v", queries)
	}
	if len(model.rewriteTopics) != 1 || model.rewriteTopics[0] != "what is alpha?" {
		t.Fatalf("the rewriter was asked about %#v", model.rewriteTopics)
	}
}

func TestRewriteQueriesAcceptsTheShapesAModelActuallySends(t *testing.T) {
	// A bare array, an object per entry with the query under another key, and a
	// fenced reply: all three carry usable queries, and discarding them would
	// send the run back to searching the raw question.
	cases := map[string]string{
		"bare array":    `["alpha retrievers"]`,
		"object entry":  `{"queries": [{"query": "alpha retrievers"}]}`,
		"question key":  `{"queries": [{"question": "alpha retrievers"}]}`,
		"fenced":        "```json\n{\"queries\": [\"alpha retrievers\"]}\n```",
		"bulleted list": "Here are the queries:\n- alpha retrievers\n- beta rerankers\n",
	}
	for name, reply := range cases {
		model := &scriptedModel{rewriteReplies: []*Reply{{Content: reply}}}
		queries, rewritten := RewriteQueries(context.Background(), model, "what is alpha?", maxSearchQueries)
		if !rewritten || len(queries) == 0 {
			t.Fatalf("%s: no query recovered from %q", name, reply)
		}
		if !strings.Contains(queries[0], "alpha") {
			t.Fatalf("%s: queries = %#v", name, queries)
		}
	}
}

func TestRewriteQueriesRejectsAnAnswerAndFallsBackToTheQuestion(t *testing.T) {
	// A rewrite that answered the question turns the retrieval that follows into
	// a search for its own guess. The guards are RAGFlow's (its fanout shape
	// guard rejects the same markers).
	for name, reply := range map[string]string{
		"cites a source": `{"queries": ["according to Wikipedia, alpha is a retriever"]}`,
		"links out":      `{"queries": ["see https://example.com/alpha"]}`,
		"prose":          `{"queries": ["` + strings.Repeat("alpha ", 40) + `"]}`,
	} {
		model := &scriptedModel{rewriteReplies: []*Reply{{Content: reply}}}
		queries, rewritten := RewriteQueries(context.Background(), model, "what is alpha?", maxSearchQueries)
		if rewritten {
			t.Fatalf("%s: %q was accepted as a search query: %#v", name, reply, queries)
		}
		if len(queries) != 1 || queries[0] != "what is alpha?" {
			t.Fatalf("%s: fallback = %#v, want the question itself", name, queries)
		}
	}
}

func TestRewriteQueriesWithoutAModelSearchesTheQuestion(t *testing.T) {
	queries, rewritten := RewriteQueries(context.Background(), nil, "  what is alpha? ", maxSearchQueries)
	if rewritten {
		t.Fatal("a rewrite was claimed with no model to do it")
	}
	if len(queries) != 1 || queries[0] != "what is alpha?" {
		t.Fatalf("queries = %#v", queries)
	}
}

// ---- goal 2: the loop searches the rewrite, not the question ----

func TestTheFirstSearchUsesTheRewrittenQueryNotTheQuestion(t *testing.T) {
	const question = "这两种动物各有多少只？"
	const rewritten = "zebra quokka marsupial census population"

	s := newStore(t, store.Chunk{
		ChunkID: "c0", DocID: "census.pdf", PageNum: 1,
		Text: "The zebra quokka marsupial census records the population of each.",
	})
	model := &scriptedModel{rewriteReplies: []*Reply{{Content: `{"queries": ["` + rewritten + `"]}`}}}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient}, missing: [][]string{nil}}, Spec: Medium()}

	result, err := loop.RunRetrieval(context.Background(), question)
	if err != nil {
		t.Fatalf("RunRetrieval: %v", err)
	}

	trace := traceText(result.Trace)
	if !strings.Contains(trace, "[QueryRewriter]") || !strings.Contains(trace, rewritten) {
		t.Fatalf("the trace does not report what retrieval searched:\n%s", trace)
	}

	// The assertion that matters: every search the round issued carried the
	// rewritten query, and none carried the question.
	searches := 0
	for _, line := range strings.Split(trace, "\n") {
		if !strings.Contains(line, ToolHybridSearch+"(") {
			continue
		}
		searches++
		if !strings.Contains(line, rewritten) {
			t.Fatalf("a search ran a query that is not the rewrite: %s", line)
		}
		if strings.Contains(line, question) {
			t.Fatalf("a search ran the question as asked: %s", line)
		}
	}
	if searches == 0 {
		t.Fatalf("no search was issued at all:\n%s", trace)
	}
	if len(result.Evidence) == 0 {
		t.Fatal("the rewritten query found nothing, so this proves the query was sent but not that it works")
	}
}

func TestASecondRoundSearchesTheGapRewriteNotTheQuestion(t *testing.T) {
	// Round 2's queries come from the gap rewrite. They must replace the round-1
	// set rather than being appended to it, or a run that failed on its first
	// angle keeps spending calls on that angle.
	//
	// The pool must be non-empty after round 1, or the loop skips the checker
	// over an empty pool (see loop.go's "an empty pool is never sufficient") and
	// the verdict script slides by one round.
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "the first angle passage"})
	model := &scriptedModel{
		rewriteReplies: []*Reply{{Content: `{"queries": ["first angle"]}`}},
		replies:        []*Reply{{Content: "second angle"}}, // the round-2 gap rewrite
	}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictInsufficient, VerdictSufficient},
		missing:  [][]string{{"term"}, nil}}, Spec: Medium()}

	result, err := loop.RunRetrieval(context.Background(), "original question")
	if err != nil {
		t.Fatalf("RunRetrieval: %v", err)
	}
	if result.Rounds != 2 {
		t.Fatalf("rounds = %d, want 2", result.Rounds)
	}

	trace := traceText(result.Trace)
	// Round 2's searches precede its own "[RAGAgent] Round 2" summary line, so
	// they are isolated by cutting after round 1's summary instead.
	index := strings.Index(trace, "[RAGAgent] Round 1")
	if index < 0 {
		t.Fatalf("no round 1 summary in the trace:\n%s", trace)
	}
	searches := 0
	for _, line := range strings.Split(trace[index:], "\n") {
		if !strings.Contains(line, ToolHybridSearch+"(") {
			continue
		}
		searches++
		if !strings.Contains(line, "second angle") {
			t.Fatalf("round 2 searched something other than the gap rewrite: %s", line)
		}
		if strings.Contains(line, "first angle") {
			t.Fatalf("round 2 repeated the round-1 angle: %s", line)
		}
	}
	if searches == 0 {
		t.Fatalf("round 2 issued no search:\n%s", trace[index:])
	}
}

// ---- goal 1: a simple question never reaches the decomposer ----

// decomposeMarker is the opening of decomposeSystemPrompt. Matched by content
// rather than by pointer because that is how the model sees it.
const decomposeMarker = "You break a multi-part question into the smallest set"

func TestASimpleQuestionNeverReachesTheDecomposer(t *testing.T) {
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage"})
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
		t.Fatalf("a simple question produced sub-questions: %#v", result.SubQuestions)
	}
	// The decisive check: the decomposer was never asked. Asserting only on
	// SubQuestions would pass if the flow decomposed and then discarded the
	// result.
	for _, messages := range model.seen {
		for _, message := range messages {
			if strings.Contains(message.Content, decomposeMarker) {
				t.Fatalf("the decomposer ran on a simple question:\n%s", message.Content)
			}
		}
	}
}

func TestAComplexQuestionDoesReachTheDecomposer(t *testing.T) {
	// The complement, so the test above cannot pass by the decomposer being
	// unreachable for some unrelated reason.
	s := newStore(t, store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage"})
	model := &scriptedModel{replies: []*Reply{
		{Content: `["part one", "part two"]`}, // decompose
		{Content: "the merged answer [1]"},    // synthesize
	}}
	loop := &Loop{Store: s, Model: model, Checker: &scriptedChecker{
		verdicts: []Verdict{VerdictSufficient, VerdictSufficient},
		missing:  [][]string{nil, nil}}, Spec: Medium()}

	flow, err := NewFlow(FlowDeps{Loop: loop, Router: complexRouter(), Model: model})
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	result, err := flow.Run(context.Background(), "Compare alpha and beta and what each explains.")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Route != RouteComplex {
		t.Fatalf("route = %s, want complex", result.Route)
	}
	if len(result.SubQuestions) != 2 {
		t.Fatalf("sub-questions = %#v, want 2", result.SubQuestions)
	}
	decomposed := false
	for _, messages := range model.seen {
		for _, message := range messages {
			if strings.Contains(message.Content, decomposeMarker) {
				decomposed = true
			}
		}
	}
	if !decomposed {
		t.Fatal("a complex question was never sent to the decomposer")
	}

	// And each sub-question is itself rewritten before it is searched, which is
	// the other half of the requirement.
	if model.rewriteCalls < len(result.SubQuestions) {
		t.Fatalf("the rewriter ran %d time(s) for %d sub-question(s); each sub-question must be "+
			"rewritten before it is searched", model.rewriteCalls, len(result.SubQuestions))
	}
}
