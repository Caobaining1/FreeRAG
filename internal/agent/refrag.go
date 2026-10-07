package agent

import (
	"fmt"
	"strings"
	"time"

	"freerag/internal/store"
)

// REFRAG-style selective compression of the evidence block.
//
// The reference is "REFRAG: Rethinking RAG based Decoding" (arXiv:2509.01092).
// Its observation is the one that matters here: in a RAG prompt the retrieved
// passages dominate the input, only a small subset is relevant to the query,
// and the decoder therefore spends most of its prefill on tokens that do not
// change the answer. REFRAG's response is compress · sense · expand — every
// chunk enters the decoder as a cheap compressed representation, and a policy
// decides which few chunks are worth expanding back to their full tokens. That
// buys up to 30.85x TTFT at equal perplexity.
//
// What transfers to this project, and what does not (recorded so the next
// reader does not think this is REFRAG):
//
//   - COMPRESS — REFRAG compresses a chunk into ONE embedding in the decoder's
//     token space, and relies on continual pre-training to teach the decoder to
//     read it. That alignment is the part we cannot have: freerag runs a frozen
//     Qwen3-4B GGUF through Ollama, so no amount of prompt engineering produces
//     a decoder that understands a projected chunk embedding. The transferable
//     form of "compressed representation" for a frozen decoder is a textual
//     gist: the passage's own leading tokens, which is all a non-retrained
//     decoder can read. It is weaker than an encoder embedding and stronger
//     than dropping the passage.
//
//   - SENSE — REFRAG trains an RL policy that decides which chunks to expand.
//     We cannot train a policy either. The proxy used here is the retrieval
//     rank the pool already carries: the top RefragConfig.ExpandTop passages
//     are expanded, the rest stay compressed. That is a heuristic standing in
//     for a learned policy, and it is named as such rather than dressed up as
//     one.
//
//   - EXPAND — identical in effect: the selected passages are handed over in
//     full, in their original positions, keeping the input a single flat
//     autoregressive sequence with stable [n] citation markers.
//
//   - REUSE (REFRAG's "precomputable" advantage) does NOT transfer and is not
//     faked here: its chunk embedding costs an encoder forward pass, so
//     precomputing it is worth something. A gist is a rune slice, so there is
//     nothing to amortise. Recorded because "REFRAG precomputes its chunk
//     representations" sounds like a free win and is not, in this form.
//
// The systemic property REFRAG buys on the system side — attention and prefill
// that scale with the number of CHUNKS rather than the number of tokens in them
// — is exactly what the char ratio below measures.
//
// Default OFF (RefragConfig.Enabled): the same prompt text as before is the
// baseline any comparison has to be taken against, and an unmeasured prompt
// change is indistinguishable from a regression.
type RefragConfig struct {
	// Enabled turns selective compression on.
	Enabled bool
	// ExpandTop is how many of the highest-ranked passages are handed over in
	// full; the rest are compressed. 0 compresses every passage.
	ExpandTop int
	// GistChars is how many characters of a compressed passage are kept.
	GistChars int
}

// Defaults for RefragConfig. Chosen so a compressed passage still carries its
// topic sentence, and a pool of ten still fits well inside the 8192-token
// window without dropping anything.
const (
	DefaultRefragExpandTop = 4
	DefaultRefragGistChars = 240
	// minRefragGistChars refuses a gist so short it would carry no claim.
	minRefragGistChars = 40
	// minRefragShrink is the factor below which compressing a passage is not
	// worth the information loss: a passage shorter than 2x the gist would not
	// even halve.
	minRefragShrink = 2
)

// normalized applies the defaults and refuses nonsense values.
//
// A negative ExpandTop would otherwise mean "compress everything including the
// best passage", which is a silently worse prompt rather than an error.
func (c RefragConfig) normalized() RefragConfig {
	if !c.Enabled {
		return RefragConfig{}
	}
	out := c
	if out.ExpandTop < 0 {
		out.ExpandTop = 0
	}
	if out.GistChars < minRefragGistChars {
		out.GistChars = DefaultRefragGistChars
	}
	return out
}

// refragMarker labels a passage that was compressed.
//
// In the header rather than in the body, and spelled out rather than a symbol:
// the decoder reading this is a 4B model that was never trained on compressed
// chunks, so it has to be told in words that the text is an abbreviation. A
// marker that only said "…" would read as an ellipsis in the source document
// and invite the model to treat the passage as complete.
const refragMarker = "compressed"

// compressPassage returns the passage's gist with an explicit cut marker.
//
// The cut prefers a sentence boundary in the last part of the window, because
// the leading fragment of a passage is what names its subject; cutting mid-word
// leaves the model with a subject and no verb. When no boundary exists the cut
// falls back to whitespace, and only then to a hard rune cut.
func compressPassage(text string, gistChars int) string {
	runes := []rune(text)
	if gistChars <= 0 || len(runes) <= gistChars {
		return text
	}
	window := runes[:gistChars]
	cut := gistCut(window)
	return strings.TrimRight(string(window[:cut]), " \t") + " […]"
}

// gistCut picks where to cut the gist inside its window.
func gistCut(window []rune) int {
	// Sentence enders, both scripts. A CJK full stop is a sentence ender here
	// exactly as a period is: the corpus is routinely Chinese (see plan.md §5.5).
	const enders = ".!?;。！？；"
	limit := len(window) * 6 / 10
	if limit < 1 {
		limit = 1
	}
	for i := len(window) - 1; i >= limit; i-- {
		if strings.ContainsRune(enders, window[i]) {
			return i + 1
		}
	}
	for i := len(window) - 1; i >= limit; i-- {
		if window[i] == ' ' || window[i] == '\t' {
			return i
		}
	}
	return len(window)
}

// PromptStats reports what the answer prompt cost: its evidence block, before
// and after selective compression, and what the generator said the call cost.
//
// Reported rather than logged because it is the measurement: the whole claim of
// this change is a ratio, and a ratio nobody can read off a result is not
// evidence. Zero-valued when compression is off, so a result taken without it
// unambiguously says so.
type PromptStats struct {
	Passages         int `json:"passages"`
	Expanded         int `json:"expanded"`
	Compressed       int `json:"compressed"`
	EvidenceChars    int `json:"evidence_chars"`
	EvidenceCharsRaw int `json:"evidence_chars_raw"`
	// Usage is the generator's own accounting for this call. Nil when the
	// generator reports none.
	//
	// Here rather than in a channel of its own because it is the same
	// statement from the other end: the evidence block decides the prompt
	// tokens, and this is what the prompt tokens cost. Separating them would
	// let a report show a 2x smaller block and no call figures at all, which
	// is how a prompt change gets credited to the wrong half of the call.
	Usage *Usage `json:"usage,omitempty"`
}

// Ratio is the compression the block achieved, raw chars over used chars.
//
// 1.0 means nothing was compressed. Values below 1.0 are impossible and would
// mean the block grew, which is worth seeing rather than clamping.
func (s PromptStats) Ratio() float64 {
	if s.EvidenceChars <= 0 {
		return 1
	}
	return float64(s.EvidenceCharsRaw) / float64(s.EvidenceChars)
}

// TraceLine renders the block's cost, or "" when nothing was compressed.
//
// Empty when inactive on purpose: a line that fires on every run with "1.00x"
// trains the reader to skip the one line that matters.
func (s PromptStats) TraceLine() string {
	if s.Compressed == 0 {
		return ""
	}
	return fmt.Sprintf(
		"[Refrag] evidence %d chars -> %d chars (%.2fx) over %d passage(s): %d expanded, %d compressed.",
		s.EvidenceCharsRaw, s.EvidenceChars, s.Ratio(), s.Passages, s.Expanded, s.Compressed)
}

// TraceLines renders every line this call deserves, in order.
//
// A slice rather than one line because the two facts are independent: the
// compression line fires only when something shrank, the usage line fires
// whenever the generator reported one. The usage line always fires because it
// is the cost of the call, and the cost does not become uninteresting on a run
// that compressed nothing.
func (s PromptStats) TraceLines() []string {
	lines := make([]string, 0, 2)
	if line := s.TraceLine(); line != "" {
		lines = append(lines, line)
	}
	if s.Usage != nil {
		lines = append(lines, s.Usage.TraceLine())
	}
	return lines
}

// TraceLine renders one generator call's cost and, where both are known, the
// rates — a prompt token and an output token cost very different amounts here
// (prefill runs hundreds of tokens per second, generation about six), and only
// the split shows which half a change touched.
func (u Usage) TraceLine() string {
	line := fmt.Sprintf("[Usage] prompt %d tok", u.PromptTokens)
	if u.PromptNanos > 0 {
		line += fmt.Sprintf(" in %s", time.Duration(u.PromptNanos).Round(10*time.Millisecond))
		if u.PromptTokens > 0 {
			line += fmt.Sprintf(" (%.0f tok/s)", float64(u.PromptTokens)/u.promptSeconds())
		}
	}
	line += fmt.Sprintf(", output %d tok", u.OutputTokens)
	if u.OutputNanos > 0 {
		line += fmt.Sprintf(" in %s", time.Duration(u.OutputNanos).Round(10*time.Millisecond))
		if u.OutputTokens > 0 {
			line += fmt.Sprintf(" (%.1f tok/s)", float64(u.OutputTokens)/u.outputSeconds())
		}
	}
	return line + "."
}

func (u Usage) promptSeconds() float64 {
	if u.PromptNanos <= 0 {
		return 0
	}
	return float64(u.PromptNanos) / float64(time.Second)
}

func (u Usage) outputSeconds() float64 {
	if u.OutputNanos <= 0 {
		return 0
	}
	return float64(u.OutputNanos) / float64(time.Second)
}

// refragPlan is the sense step: which passages of a pool get expanded.
//
// Built once per render so the policy is applied in one place rather than at
// each call site. Passages are expanded when they are among the ExpandTop
// highest-ranked, or when compressing them would not actually shrink them.
func refragPlan(evidence []store.Hit, cfg RefragConfig) []bool {
	expand := make([]bool, len(evidence))
	if !cfg.Enabled {
		for i := range expand {
			expand[i] = true
		}
		return expand
	}
	for i, hit := range evidence {
		if i < cfg.ExpandTop {
			expand[i] = true
			continue
		}
		// A passage only a little longer than the gist loses information and
		// saves nothing, so it is expanded on size rather than on rank.
		text := collapse(strings.TrimSpace(hit.Chunk.Text))
		expand[i] = len([]rune(text)) < minRefragShrink*cfg.GistChars
	}
	return expand
}

// renderPassage applies the plan to one passage and updates the stats.
//
// rawChars is counted from the passage as it would have been rendered without
// compression, so the ratio compares like with like even when the budget trims
// the tail.
func renderPassage(hit store.Hit, index int, expanded bool, cfg RefragConfig, stats *PromptStats) (header, body string, rawChars int) {
	text := collapse(strings.TrimSpace(hit.Chunk.Text))
	rawChars = len([]rune(text))

	header = fmt.Sprintf("[%d] (%s p.%d) ", index+1, hit.Chunk.DocID, hit.Chunk.PageNum)
	if expanded {
		if stats != nil {
			stats.Expanded++
		}
		return header, text, rawChars
	}

	gist := compressPassage(text, cfg.GistChars)
	if stats != nil {
		stats.Compressed++
	}
	header = fmt.Sprintf("[%d] (%s p.%d | %s) ", index+1, hit.Chunk.DocID, hit.Chunk.PageNum, refragMarker)
	return header, gist, rawChars
}
