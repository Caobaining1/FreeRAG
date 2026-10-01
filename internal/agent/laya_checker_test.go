package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"freerag/internal/store"
)

// recordingDecide captures what the checker hands the model and returns a
// scripted answer.
type recordingDecide struct {
	choice      string
	probability float64
	err         error

	instructions string
	criteria     map[string]string
	state        string
	calls        int
	// kind is what the caller asked to be asked; the bug this records is a
	// shared decider answering every caller with one question type.
	kind DecisionKind
}

func (r *recordingDecide) decide(_ context.Context, kind DecisionKind, instructions string, criteria map[string]string, state string) (string, float64, error) {
	r.calls++
	r.kind = kind
	r.instructions = instructions
	r.criteria = criteria
	r.state = state
	return r.choice, r.probability, r.err
}

// oneHit is a pool holding a single passage: the smallest thing the checker can
// be asked about. Tests that are not about the pool's contents use it, because
// an empty pool is answered UNKNOWN without consulting the model at all.
func oneHit() *kbinfo {
	return infoOf(store.Hit{Chunk: store.Chunk{ChunkID: "c0", Text: "passage"}})
}

// infoOf builds the store the checker reads from these hits.
func infoOf(hits ...store.Hit) *kbinfo {
	info := newKBInfo()
	info.add(hits)
	return info
}

func layaChecker(r *recordingDecide) LayaChecker {
	return LayaChecker{Decide: r.decide}
}

func TestLayaCheckerAcceptsASufficientDraft(t *testing.T) {
	recorder := &recordingDecide{choice: "true", probability: 0.92}

	verdict, missing := layaChecker(recorder).Check(
		context.Background(), "What is the capital of France?", oneHit())

	if verdict != VerdictSufficient {
		t.Fatalf("verdict = %s, want SUFFICIENT", verdict)
	}
	if len(missing) != 0 {
		t.Fatalf("missing = %v, want none on a SUFFICIENT verdict", missing)
	}
	if recorder.calls != 1 {
		t.Fatalf("decisions = %d, want 1", recorder.calls)
	}
	if len(recorder.criteria) != 2 {
		t.Fatalf("criteria = %v, want the two noul keys", recorder.criteria)
	}
	// The head must be a rendered template carrying the question, not a bare
	// constant: the model was trained on this set, and a mismatch degrades
	// calibration without moving accuracy (DEPLOY_CONTRACT.md §3.3).
	if !strings.Contains(recorder.instructions, "What is the capital of France?") {
		t.Fatalf("instructions = %q, want the question rendered into a template",
			recorder.instructions)
	}
}

func TestLayaCheckerRejectsAnInsufficientDraft(t *testing.T) {
	recorder := &recordingDecide{choice: "false", probability: 0.97}

	question := "How does GRPO differ from PPO?"
	verdict, missing := layaChecker(recorder).Check(
		context.Background(), question, oneHit())

	if verdict != VerdictInsufficient {
		t.Fatalf("verdict = %s, want INSUFFICIENT", verdict)
	}
	// The gap hint must come from the question terms the answer lacks, because the
	// rewrite step consumes it to build the next query.
	if len(missing) == 0 {
		t.Fatal("an INSUFFICIENT verdict must carry a rewrite hint")
	}
	joined := strings.Join(missing, " ")
	for _, term := range []string{"grpo", "ppo"} {
		if !strings.Contains(joined, term) {
			t.Fatalf("missing = %v, want it to include %q", missing, term)
		}
	}
	// Terms the answer does contain must not be reported as missing.
	if strings.Contains(joined, "passages") {
		t.Fatalf("missing = %v, but the answer does contain \"passages\"", missing)
	}
}

func TestLayaCheckerDowngradesALowConfidenceSufficient(t *testing.T) {
	// A coin-flip "sufficient" is reported as UNKNOWN rather than acted on: the
	// loop then keeps looking instead of declaring victory.
	recorder := &recordingDecide{choice: "true", probability: 0.41}

	verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", oneHit())
	if verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN for a low-probability SUFFICIENT", verdict)
	}
}

func TestLayaCheckerThresholdIsConfigurable(t *testing.T) {
	recorder := &recordingDecide{choice: "true", probability: 0.41}

	checker := layaChecker(recorder)
	checker.MinProbability = 0.3
	if verdict, _ := checker.Check(context.Background(), "q?", oneHit()); verdict != VerdictSufficient {
		t.Fatalf("verdict = %s, want SUFFICIENT once the threshold is lowered", verdict)
	}
}

// A review that could not run is not the same claim as "the evidence is
// insufficient" (docs/plan.md §6.4); collapsing them mislabels the run.
func TestLayaCheckerReportsUnknownWhenTheModelFails(t *testing.T) {
	recorder := &recordingDecide{err: errors.New("sidecar unreachable")}

	verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", oneHit())
	if verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN when the model errors", verdict)
	}
}

func TestLayaCheckerReportsUnknownWithoutAModelOrAnEmptyPool(t *testing.T) {
	if verdict, _ := (LayaChecker{}).Check(context.Background(), "q?", oneHit()); verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN with no decider", verdict)
	}

	// An empty pool is not judged: there is nothing in it to judge, and asking
	// anyway would manufacture a verdict out of an empty state.
	recorder := &recordingDecide{choice: "true", probability: 0.99}
	if verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", newKBInfo()); verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN for an empty pool", verdict)
	}
	if recorder.calls != 0 {
		t.Fatal("an empty pool must not reach the model")
	}
}

func TestLayaCheckerReportsUnknownOnAnUnrecognisedChoice(t *testing.T) {
	recorder := &recordingDecide{choice: "maybe: unclear", probability: 0.7}
	if verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", oneHit()); verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN for an option we do not know", verdict)
	}
}

// Laya judges the evidence, not the answer.
//
// This test used to assert the opposite — "§6.6: Laya sees the answer only" —
// because the old window was 512 tokens. That is exactly the constraint the 32k
// context work removed: the model is now trained on "is this evidence enough to
// answer the question", so the passages are the in-distribution input and
// withholding them would be the defect (DEPLOY_CONTRACT.md §10.2).
func TestLayaCheckerSendsTheEvidence(t *testing.T) {
	recorder := &recordingDecide{choice: "true", probability: 0.9}

	evidence := []store.Hit{
		{Chunk: store.Chunk{ChunkID: "c0", Text: "SECRETPASSAGE one two three"}},
		{Chunk: store.Chunk{ChunkID: "c1", Text: "SECRETPASSAGE four five six"}},
	}
	layaChecker(recorder).Check(context.Background(), "What is X?", infoOf(evidence...))

	if !strings.Contains(recorder.state, "SECRETPASSAGE") {
		t.Fatalf("the state dropped the evidence: %q", recorder.state)
	}
	// The answer is not part of the trained task. It still drives the rewrite
	// hint through missingFromEvidence, which is a separate concern from the input.
	if strings.Contains(recorder.state, "X is Y.") {
		t.Fatalf("the answer leaked into the decision state: %q", recorder.state)
	}
	// The question reaches the model through the head, so it is not repeated in
	// the state — but it must be there.
	if !strings.Contains(recorder.instructions, "What is X?") {
		t.Fatalf("instructions = %q, want the question rendered in", recorder.instructions)
	}
}

func TestMissingFromEvidenceIsCapped(t *testing.T) {
	long := strings.Repeat("alpha beta gamma delta epsilon zeta eta theta iota kappa ", 5)
	missing := missingFromEvidence(long, infoOf(store.Hit{Chunk: store.Chunk{Text: "nothing in common"}}))
	if len(missing) > maxMissingTerms {
		t.Fatalf("missing = %d terms, want at most %d", len(missing), maxMissingTerms)
	}
}
