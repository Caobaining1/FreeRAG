//go:build ignore

// 只读快照：权威副本在 internal/dirtree/route.go ，改动请改那里。
// 带 build tag 是因为快照位于 Go module 内，否则会被 go build ./... 编译。

package dirtree

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
)

// maxDepthGuard stops a descent that a cyclic or malformed tree would otherwise
// run forever. The real limit is the step budget; this one only exists so a
// corrupt dirs.json cannot hang the kernel.
const maxDepthGuard = 16

// Decider asks one typed decision: which of these named directions does the
// question point at.
//
// It is a function rather than a concrete model because the tree must be
// testable without one, and because the decision is the same shape Laya already
// answers elsewhere in the app (internal/agent.DecideFunc): named options, one
// choice, no generated text. Options are keyed by label and the answer comes
// back as text, so a caller matches by prefix — the convention the tool chooser
// already relies on.
type Decider func(ctx context.Context, instructions string, options map[string]string, state string) (Decision, error)

// Decision is one answered direction.
type Decision struct {
	Choice      string             `json:"choice"`
	Probability float64            `json:"probability"`
	Scores      map[string]float64 `json:"scores,omitempty"`
	// Reason is what answered: "laya" or, when the model was not asked,
	// "keyword". Recorded per decision, because a route that silently changed
	// ranker halfway down cannot be read as one route.
	Reason string `json:"reason"`
}

// Options are the knobs of one routed search.
type Options struct {
	// Limit is how many DOCUMENTS are recalled. There is no chunk limit because
	// no chunk is recalled.
	Limit int
	// Beam is how many directions survive one level.
	Beam int
	// MaxSteps caps the number of decisions, which is also the cap on latency:
	// every step is one model call.
	MaxSteps int
	// MinProbability is the confidence below which a decision is not trusted and
	// the level is ordered by keyword score instead. A typed model that is
	// evenly split between folders has not said anything, and following it
	// anyway is how a route ends up in the wrong half of the corpus.
	MinProbability float64
	// Branches is how many distinct leaf directories the route opens at once.
	// 1 is the single-path descent: one leaf, one cluster of documents.
	//
	// Measured on this corpus, a single path cannot hold a multi-hop answer no
	// matter how accurately it is routed: the evidence for one question sits in
	// 2.29 leaves on average, and the oracle — perfect routing, one leaf — is
	// 0.51 document recall against a BM25 baseline of 0.94. Four leaves is
	// 1.00. So the width of the descent, not the accuracy of the decisions, is
	// what decides this channel, and it is set here rather than left to Beam
	// because Beam and Branches are not the same knob: Beam keeps runners-up
	// behind a decision, Branches opens ends of the tree in parallel.
	//
	// Each branch gets Limit/Branches of the budget, so widening cannot simply
	// spend more — it spends the same slots in more places.
	Branches int
	// CandidateK narrows the options shown to the model to the K strongest
	// children by keyword score before it is asked. 0 asks about every child.
	//
	// Measured on this corpus, the root decision is chosen correctly 8 times in
	// 24 with six options and 24 in 24 by keyword alone: the failure is in the
	// option set, not in the model — the same model picks correctly when the
	// folders are named unambiguously. Six near-identical folder names produce
	// scores as flat as 1.60/1.45/1.21/1.00, where a guess and a decision are
	// the same thing. Narrowing turns six-way guessing into three-way judging.
	//
	// The narrowed-out children are demoted, never dropped: they follow the
	// model's ordering at their keyword score, so this can only change the
	// order in which branches are opened, not whether they exist.
	CandidateK int
	// Fallback adds the strongest documents by keyword score that the route did
	// not reach. It is on by default and is the reason a bad first decision
	// costs precision rather than the whole answer.
	Fallback bool
	// FallbackLimit is how many such documents join.
	FallbackLimit int
	// Explain attaches the terms behind every score: which query terms made a
	// folder win a level, which made a document win a leaf, and one sentence of
	// prose per recalled document.
	//
	// Off by default for two reasons. It is output for a person and no decision
	// reads it, so it is dead weight on the hot path; and a reader who sees
	// "matched 3 of 9 query terms" will treat it as a relevance judgment, which
	// it is not — it is a statement about term overlap, the instrument this
	// channel uses, and the reason the channel is explainable at all.
	Explain bool
}

// DefaultOptions are the P0 settings: two directions per level, at most eight
// decisions, and three documents of keyword safety net.
func DefaultOptions() Options {
	return Options{
		Limit:          5,
		Beam:           2,
		MaxSteps:       8,
		MinProbability: 0.0,
		Fallback:       true,
		FallbackLimit:  3,
	}
}

func (o Options) withDefaults() Options {
	if o.Limit <= 0 {
		o.Limit = 5
	}
	if o.Beam <= 0 {
		o.Beam = 2
	}
	if o.CandidateK < 0 {
		o.CandidateK = 0
	}
	if o.MaxSteps <= 0 {
		o.MaxSteps = 8
	}
	if o.FallbackLimit <= 0 {
		o.FallbackLimit = 3
	}
	return o
}

// Hit is one recalled document.
type Hit struct {
	DocID string   `json:"doc_id"`
	Name  string   `json:"name"`
	Path  []string `json:"path,omitempty"`
	Score float64  `json:"score"`
	// Source says how it was reached: "route", "keyword" or "both".
	Source string `json:"source"`
	// Slot is the sub-question that reached this document, when the search was
	// fanned out over a slot table. Empty for a single-question search, so a
	// caller can tell which of the two produced a hit.
	Slot string `json:"slot,omitempty"`
	// Route is the confidence of the decisions that led here, or 0 for a
	// document the keyword net added.
	Route float64 `json:"route_confidence"`
	// Matched is the query terms this document matched, strongest first.
	// Empty unless Options.Explain is set.
	Matched []TermHit `json:"matched,omitempty"`
	// Why is the same thing as one sentence a person can read. Empty unless
	// Options.Explain is set.
	Why string `json:"why,omitempty"`
	// Evidence is the passage inside the document that matched the query best.
	// Empty unless Options.Explain is set, and filled by the caller, not here:
	// the tree holds no passage text — a directory is document ids and names —
	// so the only thing that can supply this is the store the corpus lives in.
	Evidence string `json:"evidence,omitempty"`
}

// Step is one decision, kept so a route can be replayed.
type Step struct {
	Level       int           `json:"level"`
	From        string        `json:"from"`
	FromName    string        `json:"from_name"`
	Options     []OptionScore `json:"options"`
	Kept        []string      `json:"kept"`
	Choice      string        `json:"choice,omitempty"`
	Probability float64       `json:"probability,omitempty"`
	Reason      string        `json:"reason"`
}

// OptionScore is one direction's score at one level.
type OptionScore struct {
	NID   string  `json:"nid"`
	Name  string  `json:"name"`
	Score float64 `json:"score"`
	// Matched is the query terms that produced the score, strongest first.
	// Empty unless Options.Explain is set.
	//
	// A bare float is not an explanation. "39.72" tells a reader nothing about
	// whether the folder was chosen because it matched "amazon" or because it
	// matched "the"; these three terms do. Kept behind Explain because it is
	// output for a person, not input to any decision — routing never reads it.
	Matched []TermHit `json:"matched,omitempty"`
}

// TermHit is one query term's contribution to a score, strongest first.
type TermHit struct {
	Term  string  `json:"term"`
	Score float64 `json:"score"`
}

// ranker is everything one search knows about how the query scored against the
// corpus: a score per document, a score per directory, and the terms behind
// both.
//
// One struct rather than four maps passed around because the four are never
// used apart — a directory's score without the terms that produced it is the
// bare float that made this channel unexplainable.
type ranker struct {
	doc      map[string]float64
	node     map[string]float64
	docTerms map[string][]TermHit
	// nodeTerms is the attribution of the document that scored best inside the
	// directory. A directory has no terms of its own — its name is two
	// truncated titles — so the honest attribution is "the strongest document
	// in here matched these terms", not "this folder matched these terms".
	nodeTerms map[string][]TermHit
	// queryTerms is how many distinct terms the query had, so an explanation
	// can say "3 of 9" instead of "3".
	queryTerms int
}

func (r ranker) docScore(id string) float64      { return r.doc[id] }
func (r ranker) nodeScore(nid string) float64    { return r.node[nid] }
func (r ranker) termsOfDoc(id string) []TermHit  { return r.docTerms[id] }
func (r ranker) termsOfNode(nid string) []TermHit { return r.nodeTerms[nid] }

// Trace is the audit record of one search.
type Trace struct {
	Query     string   `json:"query"`
	Documents int      `json:"documents"`
	Decider   string   `json:"decider"`
	Steps     []Step   `json:"steps"`
	Notes     []string `json:"notes,omitempty"`
}

// Result is what one routed search returns.
type Result struct {
	Query string `json:"query"`
	Hits  []Hit  `json:"hits"`
	Trace Trace  `json:"trace"`
}

// Search recalls documents by choosing a direction at every level.
//
// The routing unit is the node and the recall unit is the document. Nothing in
// here ranks, selects or returns a chunk: the question is "which documents
// answer this", and the tree answers it by narrowing a directory at a time.
func Search(ctx context.Context, ix *Index, docs map[string]Doc, query string,
	decide Decider, opts Options) *Result {
	opts = opts.withDefaults()
	// Never a nil slice: the reply is JSON, and an empty result that serialises
	// as `null` reads as "no answer" on the other side of the wire rather than
	// "no documents".
	result := &Result{Query: query, Hits: []Hit{},
		Trace: Trace{Query: query, Documents: len(docs)}}

	if ix == nil || len(ix.Nodes) == 0 {
		result.Trace.Notes = append(result.Trace.Notes, "no directory tree; run fs.index first")
		return result
	}
	scores, docTerms := docScores(docs, query)
	subtree := subtreeDocs(ix)
	nodeScore := map[string]float64{}
	nodeTerms := map[string][]TermHit{}
	for nid, ids := range subtree {
		best, bestID := 0.0, ""
		for _, id := range ids {
			if value := scores[id]; value > best {
				best, bestID = value, id
			}
		}
		nodeScore[nid] = best
		if bestID != "" {
			nodeTerms[nid] = docTerms[bestID]
		}
	}
	r := ranker{doc: scores, node: nodeScore, docTerms: docTerms, nodeTerms: nodeTerms,
		queryTerms: len(TermsOf(query))}

	deciderName := "keyword"
	if decide != nil {
		deciderName = "laya"
	}
	result.Trace.Decider = deciderName

	if opts.Branches > 1 && len(ix.Nodes[ix.Root].Children) > 1 {
		return searchBranches(ctx, ix, docs, query, decide, opts, r, result)
	}

	open := []branch{{nid: ix.Root, confidence: 1}}
	collected := map[string]*Hit{}
	seqOf := map[string]int{}
	ordered := make([]string, 0, opts.Limit*2)
	steps := 0
	seq := 0

	// Best-first, and the priority is the path of beam ranks, not the depth.
	//
	// A level-by-level descent spends its whole step budget opening siblings and
	// can run out before it reaches ANY leaf, which costs the entire answer:
	// measured with branch=3 and MaxSteps=8, the route returned nothing at all
	// and everything the channel found came from the keyword net. Following the
	// chosen branch to a leaf first guarantees a leaf after MaxDepth decisions,
	// and only then spends what is left on the branches behind it.
	for len(open) > 0 && steps < opts.MaxSteps && len(ordered) < opts.Limit {
		sort.SliceStable(open, func(a, b int) bool { return shallower(&open[a], &open[b]) })
		current := open[0]
		open = open[1:]
		{
			node, ok := ix.Nodes[current.nid]
			if !ok {
				continue
			}
			if len(node.Children) == 0 {
				// A leaf answers with its documents, ranked by the document's
				// own keyword score — the order within a leaf is a term
				// question, and asking a model to sort four filenames it has
				// never seen is not a better answer than BM25.
				for _, id := range orderDocs(node.Docs, scores) {
					if _, seen := collected[id]; seen {
						continue
					}
					collected[id] = &Hit{
						DocID:  id,
						Name:   nameOf(id, docs[id]),
						Path:   ix.PathNames(current.nid),
						Score:  scores[id],
						Source: "route",
						Route:  current.confidence,
						}
						explainHit(collected[id], r, opts, "route")
						seqOf[id] = current.seq
						ordered = append(ordered, id)
						}
						continue
						}
						steps++
						step := Step{Level: steps - 1, From: current.nid, FromName: node.Name}
						kept, decision := chooseDirection(ctx, ix, node, query, decide, r, opts, opts.Beam, &step)
			result.Trace.Steps = append(result.Trace.Steps, step)
			for i, nid := range kept {
				confidence := current.confidence
				if decision.Reason == "laya" {
					// Multiplied, not added: a route is only as good as its
					// weakest turn, and two confident decisions followed by a
					// coin flip is a coin flip.
					confidence *= clamp01(decision.Probability)
				}
				// Opened in beam order, and each is given its own sequence
				// number: the branch Laya put first is the branch whose
				// documents are read first, whatever its depth.
				seq++
				open = append(open, branch{nid: nid, confidence: confidence,
					depth: current.depth + 1, seq: seq,
					ranks: append(append([]int{}, current.ranks...), i)})
			}
		}
		// A queue that grows without bound is a latency risk, not a recall one:
		// the branches behind the beam are the ones the model ranked last.
		if len(open) > 32 {
			open = open[:32]
		}
	}

	// Order: the route's own order. A leaf opened earlier — because the
	// decisions above it pointed at it — contributes before one opened later,
	// and within one leaf the documents are ordered by their own keyword score.
	//
	// Confidence is deliberately NOT the sort key. It is a product over levels,
	// so it falls with depth, and sorting by it prefers the SHORTEST route —
	// the least specific one — over the route the model actually chose.
	sort.SliceStable(ordered, func(a, b int) bool {
		left, right := seqOf[ordered[a]], seqOf[ordered[b]]
		if left != right {
			return left < right
		}
		return collected[ordered[a]].Score > collected[ordered[b]].Score
	})

	finish(result, ix, docs, r, opts, ordered, collected)
	return result
}

// explainHit fills in the terms and the one-line reason for one recalled
// document. Does nothing unless Options.Explain is set.
//
// Deliberately a statement about term overlap and nothing else. It does not say
// "relevant" — this channel has never claimed to know that, and a reader who
// is told a document matched 2 of 9 query terms can see the weakness for
// themselves, which is the point of explaining anything.
func explainHit(hit *Hit, r ranker, opts Options, how string) {
	if !opts.Explain {
		return
	}
	matched := r.termsOfDoc(hit.DocID)
	hit.Matched = matched
	terms := make([]string, 0, len(matched))
	for _, t := range matched {
		terms = append(terms, t.Term)
	}
	if len(terms) == 0 {
		hit.Why = fmt.Sprintf("由%s召回；未命中任何查询词（BM25 0）", how)
		return
	}
	path := ""
	if len(hit.Path) > 0 {
		path = "；目录 " + strings.Join(hit.Path, " › ")
	}
	hit.Why = fmt.Sprintf("由%s召回；命中 %d/%d 个查询词（%s），BM25 %.1f%s",
		how, len(terms), r.queryTerms, strings.Join(terms, "、"), hit.Score, path)
}

// finish applies the keyword safety net, cuts to the limit and writes the hits.
//
// Split out of Search because the widened descent (searchBranches) needs the
// same ending: the net's slots are reserved the same way whichever route
// produced the list, and two copies of this is two chances to fix a bug in one.
func finish(result *Result, ix *Index, docs map[string]Doc, r ranker,
	opts Options, ordered []string, collected map[string]*Hit) {
	if opts.Fallback {
		ranked := rankedDocs(ix.Docs, r.doc)
		added := 0
		net := make([]string, 0, opts.FallbackLimit)
		for _, id := range ranked {
			if added >= opts.FallbackLimit {
				break
			}
			if _, seen := collected[id]; seen {
				continue
			}
			collected[id] = &Hit{DocID: id, Name: nameOf(id, docs[id]),
				Score: r.doc[id], Source: "keyword"}
			explainHit(collected[id], r, opts, "关键词兜底网")
			net = append(net, id)
			added++
		}
		// The net's slots are reserved before the route's list is cut, not
		// appended after it. Appending first and truncating second — which is
		// what this did — throws the safety net away exactly when it is needed:
		// a full route means the net never survives to the reply, so the one
		// configuration where the route found eight documents is also the one
		// where nothing catches it being wrong.
		if len(ordered) > opts.Limit-len(net) && opts.Limit-len(net) >= 0 {
			ordered = ordered[:opts.Limit-len(net)]
		}
		ordered = append(ordered, net...)
		if added > 0 {
			result.Trace.Notes = append(result.Trace.Notes,
				"keyword fallback added documents the route did not reach")
		}
	}

	if len(ordered) > opts.Limit {
		ordered = ordered[:opts.Limit]
	}
	for _, id := range ordered {
		result.Hits = append(result.Hits, *collected[id])
	}
}

// searchBranches opens Branches leaf directories instead of one.
//
// One decision at the root picks the directions — the model's ranking if there
// is a model, keyword otherwise — and each direction is then walked to its own
// leaf. Every leaf contributes Limit/Branches documents, so the budget is the
// same as the single path's and only its distribution changes.
//
// This is the shape the oracle measurement asks for: the answer to a multi-hop
// question is spread over 2.29 leaves here, and no amount of routing accuracy
// recovers it from one leaf.
func searchBranches(ctx context.Context, ix *Index, docs map[string]Doc, query string,
	decide Decider, opts Options, r ranker, result *Result) *Result {
	root := ix.Nodes[ix.Root]
	// The one decision that picks the directions. Asked at the root rather than
	// once per branch: which branches to open is one question, and it is the
	// question the model is being paid to answer.
	step := Step{Level: 0, From: ix.Root, FromName: root.Name}
	kept, _ := chooseDirection(ctx, ix, root, query, decide, r, opts,
		opts.Branches, &step)
	result.Trace.Steps = append(result.Trace.Steps, step)
	if len(kept) == 0 {
		kept = append([]string{}, root.Children...)
		sort.SliceStable(kept, func(a, b int) bool {
			return r.nodeScore(kept[a]) > r.nodeScore(kept[b])
		})
	}
	if len(kept) > opts.Branches {
		kept = kept[:opts.Branches]
	}
	if opts.Explain && len(kept) < len(root.Children) {
		// Why these branches and not the others. Without it the reply shows a
		// route that opened four of six folders and no reason for the two it
		// left alone — which is the one question a reader of a four-branch
		// route actually has.
		chosen := map[string]bool{}
		for _, nid := range kept {
			chosen[nid] = true
		}
		var skipped []string
		for _, nid := range root.Children {
			if chosen[nid] {
				continue
			}
			skipped = append(skipped, fmt.Sprintf("%s (BM25 %.1f)",
				ix.Nodes[nid].Name, r.nodeScore(nid)))
		}
		if len(skipped) > 0 {
			result.Trace.Notes = append(result.Trace.Notes,
				"未开启的分支："+strings.Join(skipped, "；"))
		}
	}

	// Round-robin across branches, not a fixed quota each.
	//
	// A quota of Limit/Branches looks fair and is not: leaves here hold fewer
	// documents than that on average, so a fixed quota leaves slots unused, and
	// where a leaf holds more it throws the rest away. Measured with Branches=4
	// and Limit=8 — quota 2 each — the route alone scored 0.56 against the
	// single path's 0.80: widening managed to spend LESS of the budget than not
	// widening. Taking one document from every branch before a second from any
	// keeps the fair share and spends all of it.
	perBranch := make([][]string, 0, len(kept))
	collected := map[string]*Hit{}
	for i, start := range kept {
		// Each branch gets the full step budget. Shared across branches, it ran
		// out before the last branch reached a leaf — which is invisible in the
		// reply, because the empty branch simply contributes nothing and the
		// route looks like it chose to stop.
		steps := 1 // the root decision is shared
		leaf := descend(ctx, ix, start, query, decide, opts, r, &steps, &result.Trace)
		if leaf == "" {
			continue
		}
		node := ix.Nodes[leaf]
		var ids []string
		for _, id := range orderDocs(node.Docs, r.doc) {
			if _, seen := collected[id]; seen {
				continue
			}
			collected[id] = &Hit{DocID: id, Name: nameOf(id, docs[id]),
				Path: ix.PathNames(leaf), Score: r.doc[id], Source: "route"}
			explainHit(collected[id], r, opts, fmt.Sprintf("第 %d 个分支", i+1))
			ids = append(ids, id)
		}
		perBranch = append(perBranch, ids)
		result.Trace.Notes = append(result.Trace.Notes,
			fmt.Sprintf("branch %d -> %s (%d document(s))", i+1, node.Name, len(ids)))
	}

	finish(result, ix, docs, r, opts, interleave(perBranch, opts.Limit), collected)
	return result
}

// interleave merges per-branch rankings by rank: the first document of every
// branch before the second document of any.
func interleave(lists [][]string, limit int) []string {
	out := make([]string, 0, limit)
	for rank := 0; len(out) < limit; rank++ {
		progressed := false
		for _, list := range lists {
			if rank >= len(list) {
				continue
			}
			out = append(out, list[rank])
			progressed = true
			if len(out) >= limit {
				break
			}
		}
		if !progressed {
			break
		}
	}
	return out
}

// descend walks one direction to its leaf, recording every decision.
//
// Beam 1 within a branch: a branch is a commitment to a direction, and the
// width this channel needs is spread across branches (Branches), not inside
// one. Steps are shared across all branches so that widening the descent is a
// latency choice the caller makes with MaxSteps, not a free lunch.
func descend(ctx context.Context, ix *Index, start, query string, decide Decider,
	opts Options, r ranker, steps *int, trace *Trace) string {
	nid := start
	for depth := 0; depth <= maxDepthGuard; depth++ {
		node, ok := ix.Nodes[nid]
		if !ok {
			return ""
		}
		if len(node.Children) == 0 {
			return nid
		}
		if *steps >= opts.MaxSteps {
			trace.Notes = append(trace.Notes,
				"branch stopped at the step budget before reaching a leaf")
			return ""
		}
		*steps++
		step := Step{Level: *steps - 1, From: nid, FromName: node.Name}
		kept, _ := chooseDirection(ctx, ix, node, query, decide, r, opts, 1, &step)
		trace.Steps = append(trace.Steps, step)
		if len(kept) == 0 {
			return ""
		}
		nid = kept[0]
	}
	return ""
}

// chooseDirection asks which child to descend into, and returns them in order.
//
// The model picks one; the beam takes the next-best from the same decision's
// scores, so a second direction costs nothing extra and is what keeps a single
// newOptionScore builds one direction's score, with the terms behind it when
// the caller asked for an explanation.
//
// The terms come from the strongest document inside the directory, not from the
// directory itself: a folder here has no text of its own, only a name assembled
// from two truncated titles. Saying "this folder matched amazon" would be a
// claim about an object that cannot match anything.
func newOptionScore(ix *Index, r ranker, nid string, opts Options) OptionScore {
	out := OptionScore{NID: nid, Name: ix.Nodes[nid].Name, Score: r.nodeScore(nid)}
	if opts.Explain {
		out.Matched = r.termsOfNode(nid)
	}
	return out
}

// wrong turn from losing the answer. When the model is absent — or answered
// without confidence — the level is ordered by keyword score and the step says
// so, because a route that quietly swapped rankers is not auditable.
func chooseDirection(ctx context.Context, ix *Index, node *Node, query string,
	decide Decider, r ranker, opts Options, beam int, step *Step) ([]string, Decision) {
	children := make([]string, len(node.Children))
	copy(children, node.Children)

	byKeyword := func(nids []string) []string {
		out := make([]string, len(nids))
		copy(out, nids)
		sort.SliceStable(out, func(a, b int) bool {
			if r.nodeScore(out[a]) != r.nodeScore(out[b]) {
				return r.nodeScore(out[a]) > r.nodeScore(out[b])
			}
			return out[a] < out[b]
		})
		return out
	}

	// Narrowing: the model is asked about the K strongest children only. The
	// rest are not candidates for the decision but are not discarded either —
	// they are appended behind the model's ordering below.
	candidates := children
	var demoted []string
	if opts.CandidateK > 0 && len(children) > opts.CandidateK {
		ranked := byKeyword(children)
		candidates = ranked[:opts.CandidateK]
		demoted = ranked[opts.CandidateK:]
	}

	labels := map[string]string{}
	options := map[string]string{}
	for _, nid := range candidates {
		label := ix.Nodes[nid].Name
		if _, taken := labels[label]; taken {
			label = label + " (" + nid + ")"
		}
		labels[label] = nid
		options[label] = ix.Nodes[nid].Summary
	}

	decision := Decision{Reason: "keyword"}
	if decide != nil && len(options) >= 2 {
		answered, err := decide(ctx, renderRouteInstructions(query, node.NID), options,
			renderState(ix, node, len(subtreeOf(ix, node.NID))))
		if err == nil {
			decision = answered
			decision.Reason = "laya"
		}
	}

	scored := make([]OptionScore, 0, len(candidates))
	for _, nid := range candidates {
		scored = append(scored, newOptionScore(ix, r, nid, opts))
	}
	if decision.Reason == "laya" && len(decision.Scores) > 0 {
		for i := range scored {
			// Matched by prefix, the same convention the tool chooser uses: the
			// choice comes back as "<label>: <detail>".
			for label, value := range decision.Scores {
				if strings.HasPrefix(ix.Nodes[scored[i].NID].Name, label) {
					scored[i].Score = value
					break
				}
			}
		}
	}
	if decision.Reason == "laya" && opts.MinProbability > 0 && decision.Probability < opts.MinProbability {
		decision.Reason = "keyword"
		step.Reason = "keyword (decision below MinProbability)"
		scored = nil
		for _, nid := range byKeyword(children) {
			scored = append(scored, newOptionScore(ix, r, nid, opts))
		}
	} else {
		step.Reason = decision.Reason
	}
	step.Choice = decision.Choice
	step.Probability = decision.Probability
	sort.SliceStable(scored, func(a, b int) bool {
		if scored[a].Score != scored[b].Score {
			return scored[a].Score > scored[b].Score
		}
		return scored[a].NID < scored[b].NID
	})
	// Appended after the sort, so a demoted child can never outrank one the
	// model actually considered — it keeps its place in the queue, at the back.
	for _, nid := range demoted {
		scored = append(scored, newOptionScore(ix, r, nid, opts))
	}
	if len(demoted) > 0 && decision.Reason == "laya" {
		step.Reason = fmt.Sprintf("laya (narrowed to %d of %d by keyword)",
			len(candidates), len(children))
	}
	step.Options = scored
	if beam <= 0 {
		beam = opts.Beam
	}
	kept := scored
	if len(kept) > beam {
		kept = kept[:beam]
	}
	out := make([]string, 0, len(kept))
	for _, option := range kept {
		out = append(out, option.NID)
		step.Kept = append(step.Kept, option.NID)
	}
	return out, decision
}

// routeTemplates are the heads the routing question is asked with, in the
// question's own language.
//
// Several rather than one, and rotated, for the same reason the sufficiency
// checker rotates (internal/agent.renderInstructions): training rotated, the
// fitted temperature describes that mixture, and pinning one template would
// make every measured number describe a system nobody deploys.
//
// The question is rendered INTO the head rather than left in the state: Laya
// scores the options against the head first, and a folder choice asked without
// the question in it is a different question.
var routeTemplates = map[string][]string{
	"en": {
		"Which folder most likely contains the documents that answer this question: %s",
		"Given the question \"%s\", which of these folders should be searched?",
		"The question is: %s. Which folder holds the documents needed to answer it?",
		"To answer \"%s\", which folder is the right place to look?",
	},
	"zh": {
		"要回答这个问题：%s，应该查看下列哪个目录？",
		"问题：“%s”。哪个目录最可能包含回答它所需的文档？",
		"为了回答“%s”，应当进入哪个目录？",
	},
}

// branch is one node the descent has opened but not yet finished with.
type branch struct {
	nid        string
	confidence float64
	depth      int
	ranks      []int
	// seq is the order in which this branch was opened, and it is what orders
	// the result. Not the confidence: confidence is multiplied once per level,
	// so a deep branch always looks less confident than a shallow one, and
	// ordering by it hands every slot to the shortest route — which is the
	// least specific one. Measured: beam=2 scored 0.10 against beam=1's 0.66
	// on the same tree, entirely because of this, and a wider beam was worse
	// than a narrower one.
	seq int
}

// shallower orders two open branches: follow the better-ranked branch before
// the worse one, and a parent before its own children.
//
// Lexicographic on the beam ranks, shorter first on a tie — so (0) < (0,0) <
// (0,1) < (1): the branch Laya chose is walked all the way down before its
// runner-up is opened at all.
func shallower(a, b *branch) bool {
	for i := 0; i < len(a.ranks) && i < len(b.ranks); i++ {
		if a.ranks[i] != b.ranks[i] {
			return a.ranks[i] < b.ranks[i]
		}
	}
	if len(a.ranks) != len(b.ranks) {
		return len(a.ranks) < len(b.ranks)
	}
	return a.seq < b.seq
}

// renderRouteInstructions builds the head for one routing question.
//
// Rotated by a hash of the question and the node rather than at random: the
// mixture is what matters, and a route that cannot be replayed cannot be
// diffed against the one that failed.
func renderRouteInstructions(question, salt string) string {
	language := languageOf(question)
	templates := routeTemplates[language]
	if len(templates) == 0 {
		templates = routeTemplates["en"]
	}
	index := int(hashString(question+"|"+salt)) % len(templates)
	return fmt.Sprintf(templates[index], question)
}

// renderState is the block the decision is scored against: which folder this
// is, how much it holds, and what the route has already committed to.
func renderState(ix *Index, node *Node, size int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Current folder: %s (%d documents)\n", node.Name, size)
	if names := ix.PathNames(node.NID); len(names) > 0 {
		fmt.Fprintf(&b, "Folders chosen so far: %s\n", strings.Join(names, " > "))
	}
	fmt.Fprintf(&b, "Subfolders to choose from: %d", len(node.Children))
	return b.String()
}

// languageOf is the crude script test the checker uses: a decision model is
// calmer in the language the question was written in, and a Chinese question
// asked in an English head is a mismatch nobody would notice.
func languageOf(text string) string {
	runes := []rune(text)
	if len(runes) == 0 {
		return "en"
	}
	cjk := 0
	for _, r := range runes {
		if r >= 0x4e00 && r <= 0x9fff {
			cjk++
		}
	}
	if float64(cjk)/float64(len(runes)) > 0.15 {
		return "zh"
	}
	return "en"
}

func hashString(text string) uint32 {
	var value uint32 = 2166136261
	for _, r := range text {
		value = (value ^ uint32(r)) * 16777619
	}
	return value
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

// rankedDocs orders documents by score, dropping the ones with no term overlap.
//
// Only for the keyword net, where "matched nothing" is the whole criterion. A
// leaf the ROUTE reached must not be filtered this way — see orderDocs.
func rankedDocs(ids []string, scores map[string]float64) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		if scores[id] > 0 {
			out = append(out, id)
		}
	}
	sort.SliceStable(out, func(a, b int) bool {
		if scores[out[a]] != scores[out[b]] {
			return scores[out[a]] > scores[out[b]]
		}
		return out[a] < out[b]
	})
	return out
}

// orderDocs orders a leaf's documents by score, keeping all of them.
//
// The route already decided this leaf is where the answer is; dropping the
// documents in it that share no term with the query would re-decide that with a
// worse instrument — a leaf is small by construction, and the model's choice of
// folder is the more specific statement.
func orderDocs(ids []string, scores map[string]float64) []string {
	out := make([]string, len(ids))
	copy(out, ids)
	sort.SliceStable(out, func(a, b int) bool {
		if scores[out[a]] != scores[out[b]] {
			return scores[out[a]] > scores[out[b]]
		}
		return out[a] < out[b]
	})
	return out
}

// subtreeOf is how many documents sit under one node.
func subtreeOf(ix *Index, nid string) []string {
	return subtreeDocs(ix)[nid]
}

// subtreeDocs lists the documents under every node, once.
func subtreeDocs(ix *Index) map[string][]string {
	out := map[string][]string{}
	var walk func(nid string) []string
	walk = func(nid string) []string {
		if ids, ok := out[nid]; ok {
			return ids
		}
		node, ok := ix.Nodes[nid]
		if !ok {
			return nil
		}
		if len(node.Children) == 0 {
			out[nid] = append([]string{}, node.Docs...)
			return out[nid]
		}
		ids := make([]string, 0, 8)
		for _, child := range node.Children {
			ids = append(ids, walk(child)...)
		}
		out[nid] = ids
		return ids
	}
	for nid := range ix.Nodes {
		walk(nid)
	}
	return out
}

// docScores is BM25 where the document is the unit.
//
// Same arithmetic as the store's, applied one level up: term frequencies per
// document, idf over the corpus's documents, length normalised against the mean
// document. It is what orders a leaf's documents, what orders a level when no
// model answered, and what the keyword net adds — never what replaces a routed
// decision.
func docScores(docs map[string]Doc, query string) (map[string]float64, map[string][]TermHit) {
	terms := TermsOf(query)
	out := map[string]float64{}
	attribution := map[string][]TermHit{}
	if len(terms) == 0 || len(docs) == 0 {
		return out, attribution
	}
	df := map[string]int{}
	lengths := map[string]int{}
	for _, doc := range docs {
		for term := range doc.Terms {
			df[term]++
		}
		total := 0
		for _, count := range doc.Terms {
			total += count
		}
		lengths[doc.ID] = total
	}
	avgdl := 0.0
	for _, length := range lengths {
		avgdl += float64(length)
	}
	avgdl /= float64(len(docs))
	if avgdl == 0 {
		avgdl = 1
	}
	const k1, b = 1.2, 0.75
	total := float64(len(docs))
	for id, doc := range docs {
		score := 0.0
		dl := float64(lengths[id])
		perTerm := make([]TermHit, 0, len(terms))
		for term, qtf := range terms {
			freq, ok := doc.Terms[term]
			if !ok {
				continue
			}
			frequency := float64(df[term])
			idf := math.Log(1 + (total-frequency+0.5)/(frequency+0.5))
			denom := float64(freq) + k1*(1-b+b*dl/avgdl)
			contribution := idf * (float64(freq) * (k1 + 1) / denom) * float64(qtf)
			score += contribution
			perTerm = append(perTerm, TermHit{Term: term, Score: contribution})
		}
		out[id] = score
		if len(perTerm) > 0 {
			sort.SliceStable(perTerm, func(i, j int) bool {
				return perTerm[i].Score > perTerm[j].Score
			})
			if len(perTerm) > maxExplainTerms {
				perTerm = perTerm[:maxExplainTerms]
			}
			attribution[id] = perTerm
		}
	}
	return out, attribution
}

// maxExplainTerms is how many terms one score is explained by.
//
// Three, because a fourth term is almost always noise: BM25 contributions fall
// off fast once the discriminating terms are accounted for, and a reader given
// ten terms reads ten reasons where there is one.
const maxExplainTerms = 3
