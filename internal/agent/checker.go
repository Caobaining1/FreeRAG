package agent

import (
	"context"
	"fmt"
	"strings"

	"freerag/internal/store"
)

// Verdict is the three-way sufficiency judgement.
//
// The three cases are deliberately distinct (docs/plan.md §6.4): a review that
// could not run ("UNKNOWN") is not the same claim as "the evidence is
// insufficient", and collapsing them mislabels a run that never asked.
type Verdict string

// Verdict values.
const (
	VerdictSufficient   Verdict = "SUFFICIENT"
	VerdictInsufficient Verdict = "INSUFFICIENT"
	VerdictUnknown      Verdict = "UNKNOWN"
)

// Checker decides whether a draft answers the question. It also returns the
// terms it considers missing, which drives the next query rewrite.
//
// Laya implements this in production (docs/plan.md §6.6: the checker sees the
// draft only). RuleChecker is the deterministic fallback used when no decision
// model is configured.
type Checker interface {
	Check(ctx context.Context, question, draft string, evidence []store.Hit) (Verdict, []string)
}

// CoverageChecker is the model-free sufficiency check.
//
// It asks one question: how much of the question's own vocabulary does the
// retrieved evidence actually contain? That is a weak signal, which is the
// point — it exists so the loop is runnable and testable without Laya, and it
// reports UNKNOWN rather than guessing when it has nothing to judge.
type CoverageChecker struct {
	// Threshold is the fraction of question terms that must appear in the
	// evidence for a SUFFICIENT verdict. Zero selects DefaultCoverage.
	Threshold float64
}

// DefaultCoverage is the fraction of question terms the evidence must cover.
const DefaultCoverage = 0.6

// Check implements Checker.
func (c CoverageChecker) Check(_ context.Context, question, draft string, evidence []store.Hit) (Verdict, []string) {
	if len(evidence) == 0 {
		terms := uniqueTerms(question)
		return VerdictInsufficient, terms
	}
	if strings.TrimSpace(draft) == "" {
		return VerdictUnknown, uniqueTerms(question)
	}

	threshold := c.Threshold
	if threshold <= 0 {
		threshold = DefaultCoverage
	}

	questionTerms := uniqueTerms(question)
	if len(questionTerms) == 0 {
		// Nothing to judge against: say so instead of inventing a verdict.
		return VerdictUnknown, nil
	}

	var corpus strings.Builder
	for _, hit := range evidence {
		corpus.WriteString(hit.Chunk.Text)
		corpus.WriteByte('\n')
	}
	evidenceTerms := map[string]bool{}
	for _, term := range store.Terms(corpus.String()) {
		evidenceTerms[term] = true
	}

	var missing []string
	covered := 0
	for _, term := range questionTerms {
		if evidenceTerms[term] {
			covered++
			continue
		}
		missing = append(missing, term)
	}

	if float64(covered)/float64(len(questionTerms)) >= threshold {
		return VerdictSufficient, nil
	}
	return VerdictInsufficient, missing
}

// LayaChecker decides sufficiency with the Laya typed-decision model.
//
// docs/plan.md §6.6: only the draft is shown to Laya, never the raw passages.
// Laya answers one question — does this draft answer the question — and
// groundedness stays the generator's job. That keeps the decision to a single
// short forward pass (measured ~100 ms on CPU) inside Laya's small window.
type LayaChecker struct {
	// Decide runs the model and returns the chosen option text plus its
	// probability.
	Decide func(ctx context.Context, instructions string, criteria map[string]string, state string) (string, float64, error)
	// MinProbability is the probability below which even a SUFFICIENT decision
	// is downgraded to UNKNOWN. <=0 selects DefaultLayaMinProbability.
	MinProbability float64
}

// DefaultLayaMinProbability guards against acting on a coin-flip decision.
//
// Measured on clear-cut cases the winning probability was 0.83–1.00; 0.6 leaves
// room for a genuinely borderline draft to be reported as UNKNOWN — the safe
// direction, because UNKNOWN keeps the loop looking rather than claiming done.
const DefaultLayaMinProbability = 0.6

// Option keys for the sufficiency decision.
//
// These are matched as prefixes of the returned option text, not by index: the
// criteria map is serialised in key order, so the index is not ours to rely on.
const (
	sufficientOption   = "sufficient"
	insufficientOption = "insufficient"
)

// maxMissingTerms caps the rewrite hint so one long question cannot flood the
// next query.
const maxMissingTerms = 8

var sufficiencyCriteria = map[string]string{
	sufficientOption:   "the draft answers the question using the evidence",
	insufficientOption: "the draft does not answer the question, or goes beyond the evidence",
}

const sufficiencyInstructions = "Does the draft answer the question?"

// Check implements Checker.
func (c LayaChecker) Check(ctx context.Context, question, draft string, _ []store.Hit) (Verdict, []string) {
	draft = strings.TrimSpace(draft)
	if draft == "" || c.Decide == nil {
		return VerdictUnknown, uniqueTerms(question)
	}

	choice, probability, err := c.Decide(ctx, sufficiencyInstructions, sufficiencyCriteria,
		renderDecisionState(question, draft))
	if err != nil {
		// A review that could not run must not be reported as "insufficient"
		// (§6.4): that would claim the evidence was judged and found lacking.
		return VerdictUnknown, uniqueTerms(question)
	}

	threshold := c.MinProbability
	if threshold <= 0 {
		threshold = DefaultLayaMinProbability
	}

	switch {
	case strings.HasPrefix(choice, sufficientOption):
		if probability < threshold {
			return VerdictUnknown, uniqueTerms(question)
		}
		return VerdictSufficient, nil
	case strings.HasPrefix(choice, insufficientOption):
		return VerdictInsufficient, missingFromDraft(question, draft)
	default:
		return VerdictUnknown, uniqueTerms(question)
	}
}

// renderDecisionState builds the block Laya judges.
func renderDecisionState(question, draft string) string {
	return fmt.Sprintf("Question: %s\nDraft: %s", question, draft)
}

// missingFromDraft reports the question's terms the draft does not contain.
//
// Laya returns a verdict, not a gap list, so the rewrite hint is derived here.
// The draft — not the evidence — is the right source: it is what the model
// judged, so a term missing from it is the honest reason the verdict was
// INSUFFICIENT.
func missingFromDraft(question, draft string) []string {
	present := map[string]bool{}
	for _, term := range store.Terms(draft) {
		present[term] = true
	}

	var missing []string
	for _, term := range uniqueTerms(question) {
		if present[term] {
			continue
		}
		if len(missing) >= maxMissingTerms {
			break
		}
		missing = append(missing, term)
	}
	return missing
}

// uniqueTerms returns the question's distinct terms in first-seen order.
func uniqueTerms(text string) []string {
	seen := map[string]bool{}
	var out []string
	for _, term := range store.Terms(text) {
		if seen[term] {
			continue
		}
		seen[term] = true
		out = append(out, term)
	}
	return out
}
