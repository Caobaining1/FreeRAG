package agent

import (
	"context"
	"fmt"
	"strings"

	"freerag/internal/store"
)

// Candidate is one concrete retrieval step the loop could take next.
//
// Concrete, not a tool name: Laya decides by scoring option texts, and it
// cannot fill a tool's arguments (it does not generate). So the caller builds
// fully-formed calls — query, pattern or doc_id already bound — and Laya only
// picks which one to run.
type Candidate struct {
	Call ToolCall
	// Label is the option text Laya sees and returns. Unique across a candidate
	// set, because the choice comes back as text and is matched by prefix.
	Label string
	// Detail is the one-line description shown beside the label.
	Detail string
}

// ToolChooser picks the next retrieval step, or none.
//
// Implementations return an index into candidates, or -1 to stop. An error
// means "could not decide", which the loop answers by falling back to its
// deterministic plan — never by stalling.
type ToolChooser interface {
	Choose(
		ctx context.Context,
		question string,
		candidates []Candidate,
		evidence []store.Hit,
		attemptsSummary string,
		missing []string,
	) (int, error)
}

// option keys for the tool decision.
const stopOption = "stop"

const chooseInstructions = "Which single retrieval step should run next? Pick the option most likely to fill what is missing."

// LayaToolChooser decides the next tool call with Laya.
type LayaToolChooser struct {
	Decide DecideFunc
}

// Choose implements ToolChooser.
//
// The candidate set is built by the caller from what the loop already knows
// (the question, the rewritten queries, the checker's missing terms and the
// documents already seen), so every option is executable as written and no
// option can name a value the index does not hold. That is the whole point of
// choosing here rather than asking the generating model to invent a call: the
// failure this replaces was a model inventing a doc_id filter, and an option
// set cannot contain an invented value.
func (c LayaToolChooser) Choose(
	ctx context.Context,
	question string,
	candidates []Candidate,
	evidence []store.Hit,
	attemptsSummary string,
	missing []string,
) (int, error) {
	if c.Decide == nil {
		return -1, fmt.Errorf("tool choice: no decider configured")
	}
	if len(candidates) == 0 {
		return -1, nil
	}

	criteria := make(map[string]string, len(candidates)+1)
	labels := make([]string, len(candidates))
	for i, candidate := range candidates {
		criteria[candidate.Label] = candidate.Detail
		labels[i] = candidate.Label
	}
	criteria[stopOption] = "the evidence already answers the question; stop retrieving"

	choice, probability, err := c.Decide(ctx, DecisionChoice, chooseInstructions, criteria,
		renderChoiceState(question, evidence, attemptsSummary, missing))
	if err != nil {
		return -1, err
	}

	// stop first: "stop" is not a prefix of any generated label, but keeping the
	// order explicit means a future label like "stop after reading" cannot
	// shadow the real stop.
	if strings.HasPrefix(choice, stopOption) {
		// Refused when nothing has been retrieved yet, because the option means
		// "the evidence already answers the question" — with an empty pool that
		// is false by construction, and obeying it ends the round with no
		// retrieval at all.
		//
		// Measured on a real complex question after the question TYPE was fixed:
		// both sub-questions chose stop on their FIRST round, before any search
		// had run, and the run answered from an empty pool. Reported as "could
		// not decide" — which the loop answers by planning a retrieval — rather
		// than obeyed, and rather than silently picking a candidate the model did
		// not choose.
		if len(evidence) == 0 {
			return -1, fmt.Errorf(
				"tool choice: chose %q (p=%.2f) with nothing retrieved yet, which cannot be what it means",
				choice, probability)
		}
		return -1, nil
	}
	for i, label := range labels {
		if strings.HasPrefix(choice, label) {
			return i, nil
		}
	}
	return -1, fmt.Errorf("tool choice: Laya chose an option outside the candidate set: %q", choice)
}

// renderChoiceState is the block Laya scores the options against.
func renderChoiceState(question string, evidence []store.Hit, attemptsSummary string, missing []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n", question)
	fmt.Fprintf(&b, "Passages already retrieved: %d\n", len(evidence))
	for i, hit := range evidence {
		if i >= 6 {
			break
		}
		fmt.Fprintf(&b, "- %s p.%d %s\n", hit.Chunk.DocID, hit.Chunk.PageNum,
			truncateRunes(collapse(hit.Chunk.Text), 80))
	}
	if attemptsSummary != "" {
		b.WriteString(attemptsSummary)
	}
	if len(missing) > 0 {
		fmt.Fprintf(&b, "Reported missing: %s\n", strings.Join(missing, ", "))
	}
	return b.String()
}

// maxCandidateListChunks caps how many documents are offered as "read this
// document" options, so a large pool cannot pad the option set.
const maxCandidateListChunks = 2

// buildCandidates builds the executable options for one round.
//
// Every call is deterministic from state the loop already holds, and calls
// already tried this run are dropped — offering a step that is known to return
// nothing would spend a round re-confirming it.
func buildCandidates(
	queries []string,
	missing []string,
	evidence []store.Hit,
	attempts []attempt,
) []Candidate {
	var out []Candidate

	add := func(call ToolCall, label, detail string) {
		if previous, ok := findAttempt(attempts, call); ok {
			_ = previous
			return
		}
		for _, existing := range out {
			if sameToolCall(existing.Call, call) {
				return
			}
		}
		out = append(out, Candidate{Call: call, Label: label, Detail: detail})
	}

	// One search per query, and only per query.
	//
	// The question used to be prepended here, so every run searched the question
	// as asked on top of whatever the rewriter produced. That is no longer the
	// case: `queries` is what retrieval is for (see RewriteQueries), and it
	// always contains at least one entry because the rewriter falls back to the
	// question itself when it cannot do better.
	for _, query := range dedupeQueries(queries) {
		add(
			ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": query}},
			fmt.Sprintf("search(\"%s\")", truncateRunes(query, 60)),
			"semantic + keyword retrieval for this query",
		)
	}

	// One exact-match probe per missing term.
	for index, term := range missing {
		if index >= maxGrepLegs {
			break
		}
		term = strings.TrimSpace(term)
		if term == "" {
			continue
		}
		add(
			ToolCall{Name: ToolGrepSearch, Arguments: map[string]any{"pattern": term}},
			fmt.Sprintf("grep(\"%s\")", truncateRunes(term, 40)),
			"exact-string lookup for a term the answer is missing",
		)
	}

	// One "read this document" per document already in the pool.
	for _, docID := range evidenceDocIDs(evidence, maxCandidateListChunks) {
		add(
			ToolCall{Name: ToolListChunks, Arguments: map[string]any{"doc_id": docID}},
			fmt.Sprintf("read(\"%s\")", docID),
			"read this document's chunks in order",
		)
	}

	return out
}

// dedupeQueries trims and deduplicates the query set, keeping order.
func dedupeQueries(queries []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, query := range queries {
		query = strings.TrimSpace(query)
		if query == "" || seen[query] {
			continue
		}
		seen[query] = true
		out = append(out, query)
	}
	return out
}

// evidenceDocIDs returns up to limit distinct document ids from the pool,
// in first-seen order.
func evidenceDocIDs(evidence []store.Hit, limit int) []string {
	seen := map[string]bool{}
	var out []string
	for _, hit := range evidence {
		docID := hit.Chunk.DocID
		if docID == "" || seen[docID] {
			continue
		}
		seen[docID] = true
		out = append(out, docID)
		if len(out) >= limit {
			break
		}
	}
	return out
}
