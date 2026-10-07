package agent

import (
	"context"
	"fmt"
	"strings"

	"freerag/internal/store"
)

// synthesizeSystemPrompt writes the one answer on the complex path.
//
// The evidence it is handed was gathered across the sub-questions, so this
// prompt names the parts to cover and then hands over the pool. It used to name
// the per-sub-question answers as the input to merge; those do not exist any
// more (see Loop.RunRetrieval), and what the wording still guards against is a
// model answering the sub-questions one by one instead of the question asked.
const synthesizeSystemPrompt = `You write ONE answer to the user's question.

The question was split into sub-questions and the evidence below was gathered
for them. Answer the ORIGINAL question in one well-organised reply:
- Cover every part the evidence supports, rather than the sub-questions one by one.
- Do not repeat yourself.
- Cite with the evidence's [n] markers, and only those.
- Use ONLY what the evidence contains. Never invent facts.
- If a part is not covered by the evidence, say what is missing instead of guessing.`

// synthesize writes the single answer of the complex path from the merged pool.
//
// With no model — or on a failed or empty call — it falls back to the evidence
// itself, because a run that retrieved evidence must not end with an empty
// answer just because the merge call failed.
//
// cfg is the REFRAG-style compression of refrag.go, applied to the merged pool
// exactly as the simple path applies it (see renderEvidenceWith). The two paths
// share the compression because they share the cost it targets: prefill over a
// pool of passages, most of which the answer does not use.
func synthesize(
	ctx context.Context,
	model Model,
	question string,
	subQuestions []string,
	evidence []store.Hit,
	members []Member,
	maxChars int,
	cfg RefragConfig,
	onDelta func(string),
) (string, PromptStats) {
	evidence = dedupeHits(evidence)

	if model != nil {
		prompt, stats := renderSynthesisPrompt(question, subQuestions, evidence,
			RenderMemberRecord(question, members), maxChars, cfg)
		messages := []Message{
			{Role: RoleSystem, Content: synthesizeSystemPrompt},
			{Role: RoleUser, Content: prompt},
		}

		var (
			reply *Reply
			err   error
		)
		if streaming, ok := model.(StreamingModel); ok && onDelta != nil {
			reply, err = streaming.CompleteStream(ctx, messages, nil, onDelta)
		} else {
			reply, err = model.Complete(ctx, messages, nil)
		}
		if err == nil && reply != nil && strings.TrimSpace(reply.Content) != "" {
			stats.Usage = reply.Usage
			return strings.TrimSpace(reply.Content), stats
		}
		return truncateRunes(fallbackSynthesis(evidence), DefaultExtractiveAnswerChars), stats
	}

	return truncateRunes(fallbackSynthesis(evidence), DefaultExtractiveAnswerChars), PromptStats{}
}

// fallbackSynthesis renders the merged evidence when no model wrote an answer.
//
// It used to join the sub-loops' own answers. There are none to join now, and
// rendering the pool is what a merge step with no model can honestly do: every
// passage is attributed, and nothing is asserted that the evidence does not say.
func fallbackSynthesis(evidence []store.Hit) string {
	return extractiveAnswer(evidence)
}

// renderSynthesisPrompt builds the merge call's user message.
//
// The compression is the same selective expand/compress as the simple path's
// (refrag.go), applied to the merged pool: the highest-ranked passages stay
// whole, the rest enter as a gist and keep their [n] markers. The merge pool is
// the one place the compression has the most to work with — it is the union of
// several sub-questions' retrievals, so most of it is there for a part of the
// question the final answer barely touches.
func renderSynthesisPrompt(
	question string,
	subQuestions []string,
	evidence []store.Hit,
	memberRecord string,
	maxChars int,
	cfg RefragConfig,
) (string, PromptStats) {
	cfg = cfg.normalized()
	stats := PromptStats{Passages: len(evidence)}
	expand := refragPlan(evidence, cfg)

	var b strings.Builder
	fmt.Fprintf(&b, "Original question: %s\n\n", question)

	if len(subQuestions) > 0 {
		b.WriteString("Parts it was split into (the evidence below was gathered for them):\n")
		for i, sub := range subQuestions {
			fmt.Fprintf(&b, "%d. %s\n", i+1, sub)
		}
		b.WriteString("\n")
	}

	// The enumeration's members sit between the question and the passages: what
	// the list already holds, before the evidence it is drawn from.
	if block := strings.TrimSpace(memberRecord); block != "" {
		b.WriteString(block)
		b.WriteString("\n")
	}

	b.WriteString("Evidence:\n")
	kept := 0
	for i, hit := range evidence {
		header, text, raw := renderPassage(hit, i, expand[i], cfg, &stats)
		stats.EvidenceCharsRaw += raw
		if maxChars > 0 {
			room := maxChars - b.Len() - len(header) - 1
			if room < 120 {
				break
			}
			if len(text) > room {
				text = truncateRunes(text, room) + "…"
			}
		}
		fmt.Fprintf(&b, "%s%s\n", header, text)
		// Per passage, not the builder's length: see renderEvidenceWith.
		stats.EvidenceChars += len(text)
		kept++
	}
	if kept < len(evidence) {
		fmt.Fprintf(&b, "\n(%d of %d passage(s) omitted to fit the context window.)\n",
			len(evidence)-kept, len(evidence))
	}
	return b.String(), stats
}

// dedupeHits removes repeated chunks, keeping first-seen order.
//
// Sub-questions are independent but not disjoint: two of them routinely
// retrieve the same passage, and handing the merge call the same paragraph
// three times invites it to treat it as three corroborating sources.
func dedupeHits(evidence []store.Hit) []store.Hit {
	seen := map[string]bool{}
	out := make([]store.Hit, 0, len(evidence))
	for _, hit := range evidence {
		key := hit.Chunk.DocID + "\x00" + hit.Chunk.ChunkID
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, hit)
	}
	return out
}
