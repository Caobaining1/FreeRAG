package agent

import (
	"context"
	"fmt"
	"math/rand"
	"regexp"
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

// Checker judges whether the passages gathered so far are enough to answer the
// question. It also returns the terms it considers missing, which drives the
// next query rewrite.
//
// Laya implements this in production; CoverageChecker is the model-free
// fallback used when no decision model is configured.
//
// It reads the POOL, not a answer. The answer this used to receive was one full
// generation per round, written only to be judged and never shown to anyone (see
// kbinfo). Asking "is this pool enough" is also the question the deployed model
// was trained on, which "does this answer answer the question" was not.
type Checker interface {
	Check(ctx context.Context, question string, info *kbinfo) (Verdict, []string)
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
func (c CoverageChecker) Check(_ context.Context, question string, info *kbinfo) (Verdict, []string) {
	if info.empty() {
		terms := uniqueTerms(question)
		return VerdictInsufficient, terms
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
	for _, hit := range info.pool() {
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
// It is shown the raw passages — the whole kbinfo pool — and asked whether they
// answer the question, which is the task this checkpoint was trained on (its
// instruction templates are "given the evidence, is it sufficient to answer
// {q}?"). docs/plan.md §6.6's "the answer only" rule belonged to the 512-token
// model and is obsoleted by the 32k one (DEPLOY_CONTRACT.md §10.2). Groundedness
// stays the generator's job. One forward pass, measured ~0.48s on CPU with the
// 32k checkpoint.
type LayaChecker struct {
	// Decide runs the model and returns the chosen option text plus its
	// probability. Shared with the router and the tool chooser — one sidecar
	// method, one named type (see decide.go).
	Decide DecideFunc
	// MinProbability is the probability below which even a SUFFICIENT decision
	// is downgraded to UNKNOWN. <=0 selects DefaultLayaMinProbability.
	MinProbability float64
	// MaxStateChars caps the passage text handed to the model, in characters.
	// <=0 selects DefaultMaxStateChars. The pool grows with every round, so this
	// is what keeps both the prompt inside the model's window AND the decision
	// inside a tolerable wall time — the second binds first on CPU.
	MaxStateChars int
}

// maxStateChars is the configured cap, or the default.
func (c LayaChecker) maxStateChars() int {
	if c.MaxStateChars > 0 {
		return c.MaxStateChars
	}
	return DefaultMaxStateChars
}

// DefaultLayaMinProbability guards against acting on a coin-flip decision.
//
// Measured on clear-cut cases the winning probability was 0.83–1.00; 0.6 leaves
// room for a genuinely borderline case to be reported as UNKNOWN — the safe
// direction, because UNKNOWN keeps the loop looking rather than claiming done.
const DefaultLayaMinProbability = 0.6

// DefaultMaxStateChars caps the evidence block the checker reads.
//
// Sized against what a decision COSTS, not against the model's window. The
// window would allow far more — 32k tokens is roughly 100k characters — but on
// this machine the cost grows with the state while the accuracy does not.
//
// Measured on the 32k checkpoint, one decision each: 236 tokens 2.0s, 875
// tokens 8.9s, and 8.4k characters (about 2k tokens) a median of 14.6s over 12
// labelled cases. Past that the cost stops being predictable — 3811 tokens took
// 65s, 8k tokens 59s with an 11-76s spread as the machine began swapping, and
// 10.7k tokens did not finish in nine minutes. Accuracy over the same range did
// not improve: on the bundle's own labelled set 93.8% at ~370 tokens, 91.7% at
// ~2k, 4/4 at ~8k, with every positive case answered correctly at every length.
// A longer state there buys latency, not accuracy.
//
// 10000 characters is about 2.3k tokens — the token count the 14.6s figure was
// measured at, so a check that actually fills this budget costs about 15s on
// CPU, and a typical pool costs a fraction of that. A machine with an
// accelerator can raise it per checker (LayaChecker.MaxStateChars); on CPU this
// constant is the difference between a sufficiency check that fits inside a
// conversation and one that does not.
const DefaultMaxStateChars = 10000

// maxMissingTerms caps the rewrite hint so one long question cannot flood the
// next query.
const maxMissingTerms = 8

// Option keys for a `noul` sufficiency decision.
//
// A `noul` request renders its own options on the sidecar side — the literal
// text `"false"` then `"true"` — and ignores any criteria map that is sent
// (laya.py's render_options). So these are matched against the returned option
// text, and `criteria` is deliberately left nil: sending a choice-style map is
// what made the old call render `"sufficient: …"` instead of `"true"`, which the
// model had never seen.
const (
	sufficientOption   = "true"
	insufficientOption = "false"
)

// sufficiencyCriteria is sent on every request because the sidecar's request
// validation refuses an empty criteria map (parser.Service.Decide). A `noul`
// decision ignores it and renders "false"/"true" itself, so these descriptions
// are documentation rather than input — kept accurate to the task the model was
// trained on, because a `choice`-style request would actually use them.
var sufficiencyCriteria = map[string]string{
	sufficientOption:   "the evidence answers the question",
	insufficientOption: "the evidence does not answer the question",
}

// sufficiencyTemplates render a question into the head the model was trained on,
// keyed by instruction language.
//
// DEPLOY_CONTRACT.md §3.3 requires these to be the *same set* as training
// (laya_config.json → instruction_templates.noul). They are duplicated here so
// the checker works before that plumbing exists; a mismatch degrades calibration
// rather than breaking, which is why §3.3 calls it the nastier failure — so
// wiring them from the config is the follow-up, not an optional polish.
var sufficiencyTemplates = map[string][]string{
	"en": {
		"Does the provided evidence sufficiently answer the question: %s?",
		"Given the evidence above, is it possible to answer the question: %s?",
		"Is the evidence complete enough to answer: %s?",
		"Can the question be answered from the evidence alone: %s?",
	},
	"zh": {
		"给定的证据是否足以回答这个问题：%s？",
		"根据上述证据，能否回答这个问题：%s？",
		"证据是否完整到足以回答：%s？",
		"仅凭证据能否回答：%s？",
	},
}

// cjkThreshold and cjkRange mirror `instruction_lang_rule` in laya_config.json.
//
// The rule is written out rather than approximated: a router that is "close
// enough" drifts, and the drift is invisible. See languageOf.
const cjkThreshold = 0.2

var cjkRange = regexp.MustCompile(
	`[\x{3400}-\x{4dbf}\x{4e00}-\x{9fff}\x{f900}-\x{faff}\x{3040}-\x{30ff}\x{ac00}-\x{d7af}]`)

// languageOf reports which instruction language a question is written in.
//
// It measures the question *on its own*. Joining several fields and measuring
// the result lets a missing field's separator inflate the denominator, which
// flips borderline questions — measured on the training data: a 0.2105 question
// was padded to 0.1905 and routed to English by exactly that mistake, and 9 of
// 24493 samples ended up on the wrong template.
func languageOf(question string) string {
	runes := []rune(question)
	if len(runes) == 0 {
		return "en"
	}
	hits := len(cjkRange.FindAllString(question, -1))
	if float64(hits)/float64(len(runes)) > cjkThreshold {
		return "zh"
	}
	return "en"
}

// renderInstructions builds the head for a sufficiency question in the
// question's own language.
//
// It rotates over the template set rather than fixing one: training rotated, the
// fitted temperature describes that mixture, and pinning a single template would
// make the reported metrics describe a system nobody deploys
// (DEPLOY_CONTRACT.md §3.3).
func renderInstructions(question string) string {
	templates := sufficiencyTemplates[languageOf(question)]
	if len(templates) == 0 {
		templates = sufficiencyTemplates["en"]
	}
	return fmt.Sprintf(templates[rand.Intn(len(templates))], question)
}

// Check implements Checker.
func (c LayaChecker) Check(ctx context.Context, question string, info *kbinfo) (Verdict, []string) {
	// Nothing to judge is not a verdict, and not an invitation to guess (§6.4).
	if info.empty() || c.Decide == nil {
		return VerdictUnknown, uniqueTerms(question)
	}

	choice, probability, err := c.Decide(ctx, DecisionNoul, renderInstructions(question), sufficiencyCriteria,
		renderDecisionState(info.pool(), c.maxStateChars()))
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
		return VerdictInsufficient, missingFromEvidence(question, info)
	default:
		return VerdictUnknown, uniqueTerms(question)
	}
}

// renderDecisionState builds the block Laya judges.
//
// It carries the **evidence**, not a answer. The model was trained on "is this
// evidence enough to answer the question", so passages are the in-distribution
// input and a answer is not part of that task at all. §6.6's "the answer only"
// rule existed because the old window was 512 tokens — which is precisely the
// constraint the 32k context work removed (DEPLOY_CONTRACT.md §10.2) — and with
// the answer gone from the loop entirely (see kbinfo) the rule has nothing left
// to bind.
//
// The question reaches the model through the head: laya.py renders it into the
// instruction, so it is not repeated here.
func renderDecisionState(evidence []store.Hit, limit int) string {
	var b strings.Builder
	for _, hit := range evidence {
		text := strings.TrimSpace(hit.Chunk.Text)
		if text == "" {
			continue
		}
		separator := 0
		if b.Len() > 0 {
			separator = 2
		}
		// Whole passages only, dropped from the END. The pool grows with every
		// round (up to ActionMaxTurns calls each), so without this the state is
		// whatever the corpus happened to contain — and the model would be handed
		// a prompt past its window, which the sidecar rejects rather than trims.
		// Dropping the last-arrived passage is the honest cut: it is the one the
		// loop added most recently and the least corroborated. A passage that
		// alone exceeds the budget is truncated instead, so a single long
		// document can still be judged.
		if limit > 0 && b.Len()+separator+len(text) > limit {
			if b.Len() == 0 {
				return truncateRunes(text, limit)
			}
			break
		}
		if separator > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(text)
	}
	return b.String()
}

// missingFromEvidence reports the question's terms the pool does not contain.
//
// Laya returns a verdict, not a gap list, so the rewrite hint is derived here.
// The pool — not a answer — is the right source now: what a rewriter needs to
// know is what retrieval has not found yet, and the pool is the only thing that
// can say. (It used to read the answer, which was the thing the model judged;
// with the answer gone the honest source is the passages themselves.)
func missingFromEvidence(question string, info *kbinfo) []string {
	var corpus strings.Builder
	for _, hit := range info.pool() {
		corpus.WriteString(hit.Chunk.Text)
		corpus.WriteByte('\n')
	}

	present := map[string]bool{}
	for _, term := range store.Terms(corpus.String()) {
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
