package agent

import "context"

// DecisionKind is which of Laya's question types is being asked.
//
// It is a parameter, and not something the decider infers, because the decider is
// SHARED: one closure answers the router, the sufficiency checker and the tool
// chooser. When the type was fixed once — for the sufficiency check — every
// caller was asked a yes/no question, and the consequences were silent:
//
//   - the tool chooser's options were {false, true} instead of the candidate
//     calls, so it failed every round with "chose an option outside the candidate
//     set" and the loop fell back to the generating model's own plan, which then
//     invented a metadata_search whose doc_id was a phrase from the question;
//   - the router's answer never matched "complex"/"simple", so every route came
//     from the heuristic while the model's opinion was discarded;
//   - only the sufficiency check was asked what it wanted, and it was the only
//     one that worked.
//
// Two of the three uses of the model were dead. Nothing failed loudly: each
// caller degrades to a plausible fallback, so the symptom was slower, vaguer
// retrieval rather than an error. Measured on a real complex question: three of
// the sub-question's rounds were spent on one fabricated metadata filter.
//
// The SHAPE matters, not just the labels. noul renders a bare boolean pair, which
// is what the sufficiency check was trained against (DEPLOY_CONTRACT.md §10.2);
// choice renders "<label>: <description>" lines and scores the label positions.
// Asking a choice question as noul does not merely mislabel the options — it asks
// the model something it was never trained to answer.
type DecisionKind string

const (
	// DecisionNoul asks a yes/no question. The criteria must be the boolean pair.
	DecisionNoul DecisionKind = "noul"
	// DecisionChoice asks the model to pick one of the named options.
	DecisionChoice DecisionKind = "choice"
)

// DecideFunc runs one Laya typed decision.
//
// Laya is a non-generative decision model (sidecar/laya.py): given an
// instruction, a set of named options and some state, it scores every option
// and returns the argmax. It never writes free text, so it cannot be asked to
// produce a tool's arguments — only to choose among options the caller built.
//
// One named type rather than three identical anonymous ones: the sufficiency
// checker, the simple/complex router and the per-round tool chooser all call
// the very same sidecar method, and naming it is what lets the kernel wire
// them from a single closure. `kind` is what keeps those three callers from
// having to agree on one question type.
type DecideFunc func(
	ctx context.Context,
	kind DecisionKind,
	instructions string,
	criteria map[string]string,
	state string,
) (choice string, probability float64, err error)
