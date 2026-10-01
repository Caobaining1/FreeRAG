package agent

import (
	"context"
	"strings"
	"testing"

	"freerag/internal/store"
)

// The enumeration as the loop runs it: the declaration switches round 1's recall
// list to the operands, bounds the round budget to two, and resolves the members
// onto the result. The other half of the design is tested beside it — a
// declaration that does NOT pass the gate must change nothing at all, down to not
// making the extraction call.
//
// Note where each reply lives: the loop's model helper routes query-rewrite turns
// to `rewriteReplies` and everything else to `replies`, because a rewrite turn must
// not consume a reply meant for the answer.

func enumStore(t *testing.T) *store.Store {
	t.Helper()
	return newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "alpha.pdf", PageNum: 1, Text: "alpha was listed in the report"},
		store.Chunk{ChunkID: "c1", DocID: "beta.pdf", PageNum: 2, Text: "beta was also listed there"},
	)
}

// The rounds the enumeration allows, all ending INSUFFICIENT so the run spends its
// whole budget and the bound is observable.
func everInsufficient() *scriptedChecker {
	return &scriptedChecker{
		verdicts: []Verdict{VerdictInsufficient, VerdictInsufficient, VerdictInsufficient},
		missing:  [][]string{{"alpha"}, {"alpha"}, {"alpha"}},
	}
}

func TestRunEnumerationRecallsOperandsAndBoundsRounds(t *testing.T) {
	model := &scriptedModel{
		// Round 0's rewrite is the only turn that carries the round-0 prompt, and
		// it is where the declaration comes from.
		rewriteReplies: []*Reply{
			{Content: `{"queries":["generic query"],"slots":[{"type":"list","subject":"alpha|beta","scan":["listed"]}]}`},
		},
		replies: []*Reply{
			// The gap rewrite after round 1 (plain text, which is what l.rewrite
			// reads). The operand recall is round 1's and runs once.
			{Content: "second angle"},
			// The one extraction call, over the pool.
			{Content: `{"members":[{"name":"alpha","evidence":1},{"name":"beta","evidence":2}]}`},
		},
		// Only round 2 plans: round 1's calls are the operands, issued by the loop.
		planReplies: []*Reply{
			{ToolCalls: []ToolCall{{Name: ToolHybridSearch, Arguments: map[string]any{"query": "angle"}}}},
		},
	}
	loop := &Loop{Store: enumStore(t), Model: model, Checker: everInsufficient(), Spec: Medium()}

	result, err := loop.RunRetrieval(context.Background(), "which papers are listed?")
	if err != nil {
		t.Fatalf("RunRetrieval: %v", err)
	}
	trace := traceText(result.Trace)

	// 1. Round 1's recall list is the declared operands, one per operand: the
	//    actor's alternatives and the act words. The generic rewrite did not run
	//    for it — RAGFlow's rule, and the reason the strategy exists.
	if !strings.Contains(trace, "recalling all 3 operand(s) in this turn") {
		t.Fatalf("the operand recall must be reported:\n%s", trace)
	}
	// One call PER operand, all three of them: offering them to the tool chooser
	// instead would spend the round on one, which is exactly how a real run came to
	// search "acquisition" and return nothing while the actor was never searched.
	for _, operand := range []string{"alpha", "beta", "listed"} {
		if !strings.Contains(trace, "hybrid_search(query="+operand+")") {
			t.Fatalf("operand %q was not recalled:\n%s", operand, trace)
		}
	}
	// Round 1 did not go through the planner at all.
	if model.planCalls != 1 {
		t.Fatalf("the planner ran %d time(s); want 1 (round 2 only, round 1 is the operand recall)",
			model.planCalls)
	}

	// 2. The budget is bounded to two, from Medium()'s three.
	if result.Rounds != enumMaxRounds {
		t.Fatalf("Rounds = %d, want %d", result.Rounds, enumMaxRounds)
	}
	if !strings.Contains(trace, "rounds bounded 3 -> 2") {
		t.Fatalf("the bound must be reported:\n%s", trace)
	}

	// 3. The members are resolved onto the result, each anchored to its passage.
	if !result.Enumeration || result.ItemKind != "list" {
		t.Fatalf("Enumeration = %v / ItemKind = %q, want true / list", result.Enumeration, result.ItemKind)
	}
	if len(result.Members) != 2 {
		t.Fatalf("got %d member(s), want 2: %+v", len(result.Members), result.Members)
	}
	if result.Members[0].Name != "alpha" || result.Members[0].ChunkID != "c0" {
		t.Fatalf("member 0 = %+v, want alpha anchored to c0", result.Members[0])
	}
	for _, member := range result.Members {
		if member.ChunkID == "" || member.Quote == "" {
			t.Fatalf("member %+v is not anchored with a quote", member)
		}
	}
}

func TestRunWithoutAPassingDeclarationIsUnchanged(t *testing.T) {
	// The strategy's whole cost model rests on this: a declaration that does not
	// pass the gate buys nothing and spends nothing. The slot declares the actor
	// and the act words but a date-shaped answer, so the run must behave exactly as
	// a run with no declaration — the same rewritten queries, the full budget, and
	// not even the extraction call.
	model := &scriptedModel{
		rewriteReplies: []*Reply{
			{Content: `{"queries":["generic query"],"slots":[{"type":"date","subject":"alpha","scan":["listed"]}]}`},
		},
		replies: []*Reply{
			{Content: "second angle"},
			{Content: "third angle"},
		},
		planReplies: []*Reply{
			{ToolCalls: []ToolCall{{Name: ToolHybridSearch, Arguments: map[string]any{"query": "listed"}}}},
			{ToolCalls: []ToolCall{{Name: ToolHybridSearch, Arguments: map[string]any{"query": "listed"}}}},
			{ToolCalls: []ToolCall{{Name: ToolHybridSearch, Arguments: map[string]any{"query": "listed"}}}},
		},
	}
	loop := &Loop{Store: enumStore(t), Model: model, Checker: everInsufficient(), Spec: Medium()}

	result, err := loop.RunRetrieval(context.Background(), "when was alpha listed?")
	if err != nil {
		t.Fatalf("RunRetrieval: %v", err)
	}

	if result.Enumeration {
		t.Fatal("a date-shaped slot declared an enumeration")
	}
	if strings.Join(result.Queries, ",") != "third angle" {
		t.Fatalf("Queries = %v, want the rewritten queries untouched", result.Queries)
	}
	if result.Rounds != Medium().SCAMaxRounds {
		t.Fatalf("Rounds = %d, want the unbounded %d", result.Rounds, Medium().SCAMaxRounds)
	}
	if trace := traceText(result.Trace); strings.Contains(trace, "[Enumerate]") {
		t.Fatalf("the enumeration reported itself on a run that must not use it:\n%s", trace)
	}
	if len(result.Members) != 0 {
		t.Fatalf("members were extracted without an enumeration: %+v", result.Members)
	}
	// The sharpest form of "pays nothing": the extraction call never happened.
	// `calls` counts this helper's replies, which are the two gap rewrites and
	// nothing else — an extraction would have made it three.
	if model.calls != 2 {
		t.Fatalf("the model took %d reply(s) besides the declaration; want 2 (one gap rewrite per round, "+
			"and no extraction without an enumeration)", model.calls)
	}
}
