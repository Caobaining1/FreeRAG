package agent

import (
	"context"
	"testing"
)

// The bug this file exists for: one shared decider, one fixed question type.
//
// The kernel's closure fixed qtype at "noul" for the sufficiency check, and the
// same closure answered the router and the tool chooser. Laya was therefore asked
// a yes/no question in all three cases, and two of the three callers were dead:
//
//   - the tool chooser's options became {false, true} instead of the candidate
//     calls, so it failed EVERY round with "chose an option outside the candidate
//     set", and the loop answered that by silently using the generating model's
//     own plan — the very path whose invented doc_id filters the chooser exists to
//     prevent. Measured on a real complex question: three rounds of a
//     sub-question spent on one fabricated metadata filter, ending with an empty
//     pool;
//   - the router's answer never matched "complex"/"simple", so every route came
//     from the heuristic while the model's opinion was discarded;
//   - only the sufficiency check, which wants a boolean, worked.
//
// Every caller degrades to a plausible fallback, so nothing failed loudly: the
// symptom was slower, vaguer retrieval and a trace that said "using the model
// plan" once per round. These tests assert the QUESTION TYPE each caller asks,
// which is the thing that was wrong and the thing no test covered — the existing
// ones all drove a recorder that ignored the kind entirely.

func TestEachCallerAsksItsOwnQuestionType(t *testing.T) {
	t.Run("the sufficiency check asks a boolean question", func(t *testing.T) {
		recorder := &recordingDecide{choice: "true", probability: 0.9}
		checker := LayaChecker{Decide: recorder.decide}

		checker.Check(context.Background(), "what does alpha say?", oneHit())

		if recorder.kind != DecisionNoul {
			t.Fatalf("the checker asked %q; sufficiency is the boolean question this checkpoint "+
				"was trained on (DEPLOY_CONTRACT.md §10.2)", recorder.kind)
		}
		if len(recorder.criteria) != 2 || recorder.criteria["true"] == "" || recorder.criteria["false"] == "" {
			t.Fatalf("the boolean options are not the {true, false} pair: %#v", recorder.criteria)
		}
	})

	t.Run("the router asks a choice question", func(t *testing.T) {
		recorder := &recordingDecide{choice: complexOption + ": several parts", probability: 0.9}
		router := Router{Decide: recorder.decide}

		if route := router.Route(context.Background(), "compare alpha with beta"); route != RouteComplex {
			t.Fatalf("route = %s, want complex", route)
		}
		if recorder.kind != DecisionChoice {
			t.Fatalf("the router asked %q; it picks between named options, and a boolean question "+
				"means its answer can never match one", recorder.kind)
		}
	})

	t.Run("the tool chooser asks a choice question", func(t *testing.T) {
		recorder := &recordingDecide{choice: `search("alpha")`, probability: 0.9}
		chooser := LayaToolChooser{Decide: recorder.decide}

		if _, err := chooser.Choose(context.Background(), "what does alpha say?",
			[]Candidate{{Call: ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "alpha"}},
				Label: `search("alpha")`, Detail: "semantic + keyword retrieval"}},
			nil, "", nil); err != nil {
			t.Fatalf("Choose: %v", err)
		}

		if recorder.kind != DecisionChoice {
			t.Fatalf("the tool chooser asked %q; its options are the candidate calls, and asking a "+
				"boolean question makes the option set {false, true} — which is how it came to fail "+
				"on every round of every run", recorder.kind)
		}
		if _, boolean := recorder.criteria["false"]; boolean {
			t.Fatalf("the chooser offered a boolean pair as its options: %#v", recorder.criteria)
		}
		if _, offered := recorder.criteria[`search("alpha")`]; !offered {
			t.Fatalf("the chooser did not offer the candidate by its label: %#v", recorder.criteria)
		}
	})
}

// A boolean reply is an ERROR, not a candidate.
//
// The old wiring made this the normal case, so the tempting "fix" is to map
// "false" onto the first candidate and carry on. That would run an arbitrary call
// and report it as a decision — the failure would move from visible (a fallback
// the trace names) to invisible (a retrieval nobody asked for).
func TestABooleanReplyIsNotMappedOntoACandidate(t *testing.T) {
	chooser := LayaToolChooser{Decide: (&recordingDecide{choice: "false", probability: 0.99}).decide}

	index, err := chooser.Choose(context.Background(), "what does alpha say?",
		[]Candidate{
			{Call: ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "alpha"}}, Label: `search("alpha")`},
			{Call: ToolCall{Name: ToolGrepSearch, Arguments: map[string]any{"pattern": "alpha"}}, Label: `grep("alpha")`},
		}, nil, "", nil)

	if err == nil {
		t.Fatalf("a boolean answer selected candidate %d; it must not select anything", index)
	}
	if index != -1 {
		t.Fatalf("index = %d on a boolean answer, want -1", index)
	}
}
