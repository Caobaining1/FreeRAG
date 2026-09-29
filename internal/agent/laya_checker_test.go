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
}

func (r *recordingDecide) decide(_ context.Context, instructions string, criteria map[string]string, state string) (string, float64, error) {
	r.calls++
	r.instructions = instructions
	r.criteria = criteria
	r.state = state
	return r.choice, r.probability, r.err
}

func layaChecker(r *recordingDecide) LayaChecker {
	return LayaChecker{Decide: r.decide}
}

func TestLayaCheckerAcceptsASufficientDraft(t *testing.T) {
	recorder := &recordingDecide{choice: "sufficient: the draft answers the question", probability: 0.92}

	verdict, missing := layaChecker(recorder).Check(
		context.Background(), "What is the capital of France?", "The capital of France is Paris.", nil)

	if verdict != VerdictSufficient {
		t.Fatalf("verdict = %s, want SUFFICIENT", verdict)
	}
	if len(missing) != 0 {
		t.Fatalf("missing = %v, want none on a SUFFICIENT verdict", missing)
	}
	if recorder.calls != 1 {
		t.Fatalf("decisions = %d, want 1", recorder.calls)
	}
	if recorder.instructions == "" || len(recorder.criteria) != 2 {
		t.Fatalf("the decision was not asked properly: %q %v", recorder.instructions, recorder.criteria)
	}
}

func TestLayaCheckerRejectsAnInsufficientDraft(t *testing.T) {
	recorder := &recordingDecide{choice: "insufficient: the draft does not answer", probability: 0.97}

	question := "How does GRPO differ from PPO?"
	verdict, missing := layaChecker(recorder).Check(
		context.Background(), question, "The passages discuss unrelated optimizers.", nil)

	if verdict != VerdictInsufficient {
		t.Fatalf("verdict = %s, want INSUFFICIENT", verdict)
	}
	// The gap hint must come from the question terms the draft lacks, because the
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
	// Terms the draft does contain must not be reported as missing.
	if strings.Contains(joined, "passages") {
		t.Fatalf("missing = %v, but the draft does contain \"passages\"", missing)
	}
}

func TestLayaCheckerDowngradesALowConfidenceSufficient(t *testing.T) {
	// A coin-flip "sufficient" is reported as UNKNOWN rather than acted on: the
	// loop then keeps looking instead of declaring victory.
	recorder := &recordingDecide{choice: "sufficient: answers it", probability: 0.41}

	verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", "some draft", nil)
	if verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN for a low-probability SUFFICIENT", verdict)
	}
}

func TestLayaCheckerThresholdIsConfigurable(t *testing.T) {
	recorder := &recordingDecide{choice: "sufficient: answers it", probability: 0.41}

	checker := layaChecker(recorder)
	checker.MinProbability = 0.3
	if verdict, _ := checker.Check(context.Background(), "q?", "draft", nil); verdict != VerdictSufficient {
		t.Fatalf("verdict = %s, want SUFFICIENT once the threshold is lowered", verdict)
	}
}

// A review that could not run is not the same claim as "the evidence is
// insufficient" (docs/plan.md §6.4); collapsing them mislabels the run.
func TestLayaCheckerReportsUnknownWhenTheModelFails(t *testing.T) {
	recorder := &recordingDecide{err: errors.New("sidecar unreachable")}

	verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", "draft", nil)
	if verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN when the model errors", verdict)
	}
}

func TestLayaCheckerReportsUnknownWithoutAModelOrDraft(t *testing.T) {
	if verdict, _ := (LayaChecker{}).Check(context.Background(), "q?", "draft", nil); verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN with no decider", verdict)
	}

	recorder := &recordingDecide{choice: "sufficient: x", probability: 0.99}
	if verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", "   ", nil); verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN for an empty draft", verdict)
	}
	if recorder.calls != 0 {
		t.Fatal("an empty draft must not reach the model")
	}
}

func TestLayaCheckerReportsUnknownOnAnUnrecognisedChoice(t *testing.T) {
	recorder := &recordingDecide{choice: "maybe: unclear", probability: 0.7}
	if verdict, _ := layaChecker(recorder).Check(context.Background(), "q?", "draft", nil); verdict != VerdictUnknown {
		t.Fatalf("verdict = %s, want UNKNOWN for an option we do not know", verdict)
	}
}

// §6.6: Laya sees the draft only. Feeding it the raw passages would blow its
// small window and move the groundedness question into the wrong model.
func TestLayaCheckerSendsOnlyTheQuestionAndDraft(t *testing.T) {
	recorder := &recordingDecide{choice: "sufficient: answers it", probability: 0.9}

	evidence := []store.Hit{
		{Chunk: store.Chunk{ChunkID: "c0", Text: "SECRETPASSAGE one two three"}},
		{Chunk: store.Chunk{ChunkID: "c1", Text: "SECRETPASSAGE four five six"}},
	}
	layaChecker(recorder).Check(context.Background(), "What is X?", "X is Y.", evidence)

	if strings.Contains(recorder.state, "SECRETPASSAGE") {
		t.Fatalf("the state leaked evidence into the decision: %q", recorder.state)
	}
	if !strings.Contains(recorder.state, "What is X?") || !strings.Contains(recorder.state, "X is Y.") {
		t.Fatalf("state = %q, want question and draft", recorder.state)
	}
}

func TestMissingFromDraftIsCapped(t *testing.T) {
	long := strings.Repeat("alpha beta gamma delta epsilon zeta eta theta iota kappa ", 5)
	missing := missingFromDraft(long, "nothing in common")
	if len(missing) > maxMissingTerms {
		t.Fatalf("missing = %d terms, want at most %d", len(missing), maxMissingTerms)
	}
}
