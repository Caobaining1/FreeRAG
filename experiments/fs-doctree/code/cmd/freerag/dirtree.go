//go:build ignore

// 只读快照：权威副本在 cmd/freerag/dirtree.go ，改动请改那里。
// 带 build tag 是因为快照位于 Go module 内，否则会被 go build ./... 编译。

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"freerag/internal/agent"
	"freerag/internal/dirtree"
	"freerag/internal/ipc"
	"freerag/internal/parser"
)

// The document-level channel's kernel surface: a directory tree over the
// corpus, routed one decision at a time, recalling DOCUMENTS.
//
// Beside internal/tree rather than instead of it on purpose. The two answer
// different questions — "which document" and "where in it" — and a system that
// can only answer one of them is the reason this channel exists at all.

// dirPath is where a knowledge base keeps its directory tree: beside index.json.
func dirPath(live *kbRuntime) string {
	return filepath.Join(filepath.Dir(live.dataPath), "dirs.json")
}

// dirIndexFor opens a base's directory tree once per process.
//
// Unlike the section tree, this one is complete when it is built: it needs no
// attach step, because a directory holds document ids and names and no passage
// text ever lives in it.
func (k *kernel) dirIndexFor(live *kbRuntime) (*dirtree.Index, error) {
	live.dirsMu.Lock()
	defer live.dirsMu.Unlock()
	if live.dirs != nil {
		return live.dirs, nil
	}
	ix, err := dirtree.Open(dirPath(live))
	if err != nil {
		return nil, err
	}
	live.dirs = ix
	return ix, nil
}

// corpusDocs reads the corpus as the builder sees it: one entry per document,
// with the terms the tree is built from.
//
// Terms come from the stored text rather than from a second pass over the
// files, so building a tree never re-reads a document the index already holds.
func corpusDocs(live *kbRuntime) map[string]dirtree.Doc {
	out := map[string]dirtree.Doc{}
	for _, docID := range live.store.Documents() {
		doc := dirtree.Doc{ID: docID, Name: docID}
		if record, ok := live.store.DocumentByDocID(docID); ok {
			doc.Path = record.SourceFile
			if record.SourceFile != "" {
				doc.Name = filepath.Base(record.SourceFile)
			}
		}
		terms := map[string]int{}
		for _, chunk := range live.store.List(docID, 0, 0, 0) {
			for term, count := range dirtree.TermsOf(chunk.Text) {
				terms[term] += count
			}
		}
		doc.Terms = terms
		out[docID] = doc
	}
	return out
}

// routeDecider adapts the Laya sidecar to the tree's routing question.
//
// The agent's own decider is not reused: it returns one choice and one
// probability, and routing needs every option's score so that a beam can carry
// a second direction without paying for a second call. The sidecar already
// returns them (parser.Decision.Scores); this is the only place that reads them.
func (k *kernel) routeDecider() dirtree.Decider {
	if k.parse == nil || !k.layaReady {
		return nil
	}
	return func(ctx context.Context, instructions string, options map[string]string,
		state string) (dirtree.Decision, error) {
		decision, err := k.parse.Decide(ctx, parser.DecisionRequest{
			Instructions: instructions,
			Criteria:     options,
			State:        state,
			// A choice, not a noul: the question is "which folder", and noul
			// renders a bare boolean pair — asking it here would score the wrong
			// positions entirely (internal/agent.DecisionKind).
			QType: string(agent.DecisionChoice),
		}, 0)
		if err != nil {
			return dirtree.Decision{}, err
		}
		out := dirtree.Decision{Choice: decision.Choice, Probability: decision.Probability,
			Scores: map[string]float64{}}
		// The sidecar reports options as the RENDERED text ("<label>: <detail>"),
		// not as the label, and its order is the JSON map's — so a score cannot
		// be read by position and cannot be keyed by the rendered string either.
		// Matched back to the label we sent, which is the only key both sides
		// agree on. Getting this wrong is silent: the beam would then carry the
		// keyword order while the trace still said the model decided.
		for i, option := range decision.Options {
			if i >= len(decision.Scores) {
				break
			}
			for label, detail := range options {
				if option == label+": "+detail {
					out.Scores[label] = decision.Scores[i]
					break
				}
			}
		}
		return out, nil
	}
}

// handleDirIndex (re)builds the corpus directory tree.
func (k *kernel) handleDirIndex(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		KB       string `json:"kb"`
		Branch   int    `json:"branch"`
		LeafSize int    `json:"leaf_size"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	live.indexing.Lock()
	defer live.indexing.Unlock()

	docs := corpusDocs(live)
	list := make([]dirtree.Doc, 0, len(docs))
	for _, doc := range docs {
		list = append(list, doc)
	}
	opts := dirtree.DefaultBuildOptions()
	if params.Branch > 0 {
		opts.Branch = params.Branch
	}
	if params.LeafSize > 0 {
		opts.LeafSize = params.LeafSize
	}
	ix := dirtree.Build(list, opts)

	live.dirsMu.Lock()
	live.dirs = ix
	live.dirsMu.Unlock()

	reply := map[string]any{
		"documents": len(list),
		"nodes":     len(ix.Nodes),
		"from":      ix.From,
		"data_path": dirPath(live),
	}
	if err := ix.Save(dirPath(live)); err != nil {
		log.Printf("warning: could not persist the directory tree: %v", err)
		reply["save_error"] = err.Error()
	}
	return withKB(reply, live), nil
}

// handleDirStatus reports the shape of the tree, which is the only way to see
// whether it was built from the user's own folders or invented by clustering.
func (k *kernel) handleDirStatus(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		KB string `json:"kb"`
		// Depth is how many levels below the root the reply describes.
		// 0 (the default) reports the top level only, which is what a status
		// call is for. The desktop shell asks for more to draw the tree, and
		// drawing it is the only way to see whether the corpus was split the
		// way a reader would have split it — a folder called
		// "ChatGPT Everything / Mass breach of privacy TikTok" is a claim
		// about the corpus that no number in this reply can confirm or deny.
		Depth int `json:"depth"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	live.indexing.RLock()
	defer live.indexing.RUnlock()

	ix, err := k.dirIndexFor(live)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}
	// The same test `fs.search` applies, and for the same reason: Open() hands
	// back an EMPTY index (root named, no nodes) when the file is missing, so
	// "no tree yet" arrives here as a valid-looking index rather than as an
	// error. Asking for the shape of a tree that does not exist used to walk
	// straight into ix.Nodes[root].Children and take the whole kernel down —
	// which the desktop shell triggers just by opening the panel.
	if len(ix.Nodes) <= 1 {
		return nil, &ipc.Error{
			Code:    ipc.CodeInvalidParams,
			Message: "this knowledge base has no directory tree; run fs.index first",
		}
	}
	byFrom := map[string]int{}
	leaves, maxLevel := 0, 0
	for _, node := range ix.Nodes {
		byFrom[node.From]++
		if len(node.Children) == 0 {
			leaves++
		}
		if node.Level > maxLevel {
			maxLevel = node.Level
		}
	}
	top := []map[string]any{}
	for _, nid := range ix.Nodes[ix.Root].Children {
		node := ix.Nodes[nid]
		top = append(top, map[string]any{
			"nid": nid, "name": node.Name, "from": node.From,
			"children": len(node.Children), "docs": len(node.Docs),
			"summary": node.Summary,
		})
	}
	// The tree itself, cut at the depth asked for. Children are carried inside
	// their parent rather than as a flat list, because the shape is the point:
	// a flat list of 40 folders tells a reader nothing about which ones are
	// siblings of a decision the route made.
	var tree []map[string]any
	if params.Depth > 0 {
		tree = dirTree(ix, ix.Root, params.Depth, docNames(live))
	}
	return withKB(map[string]any{
		"documents": len(ix.Docs),
		"nodes":     len(ix.Nodes),
		"leaves":    leaves,
		"depth":     maxLevel,
		"from":      ix.From,
		"by_from":   byFrom,
		"top":       top,
		"tree":      tree,
		"decider":   k.deciderLabel(),
		"data_path": dirPath(live),
	}, live), nil
}

// dirTree renders the tree below one node, `levels` deep.
//
// `docs` is the count of documents in the WHOLE subtree, not the ones filed
// directly in the folder: a directory that holds four sub-folders holding three
// documents each is a branch of twelve, and reporting it as "0 documents"
// would make the picture of a corpus look like a picture of nothing.
// dirFileCap is how many documents one folder names in a status reply.
//
// A folder of 200 files is a scroll, and the shell asks for the whole tree at
// once so it can expand a node without a round trip — the count is kept exact
// (`own_docs`) so a truncated list still says how many there were.
const dirFileCap = 40

func dirTree(ix *dirtree.Index, nid string, levels int, names map[string]string) []map[string]any {
	if levels <= 0 {
		return nil
	}
	out := []map[string]any{}
	parent := ix.Nodes[nid]
	if parent == nil {
		return out
	}
	for _, childID := range parent.Children {
		node := ix.Nodes[childID]
		// A child id with no node behind it is a corrupt tree, and a corrupt
		// tree is a reason to draw a shorter one — not a reason for the kernel
		// to stop answering every other request on this knowledge base.
		if node == nil {
			continue
		}
		// The documents filed DIRECTLY in this folder, which is the list a
		// reader opens a folder for. Counts alone are not enough to browse:
		// "14 篇" under a folder tells nobody which 14.
		files := []map[string]any{}
		for i, docID := range node.Docs {
			if i >= dirFileCap {
				break
			}
			name := names[docID]
			if name == "" {
				name = docID
			}
			files = append(files, map[string]any{"id": docID, "name": name})
		}
		entry := map[string]any{
			"nid":      childID,
			"name":     node.Name,
			"from":     node.From,
			"summary":  node.Summary,
			"level":    node.Level,
			"children": len(node.Children),
			"docs":     len(subtreeDocsOf(ix, childID)),
			"own_docs": len(node.Docs),
			"files":    files,
		}
		if kids := dirTree(ix, childID, levels-1, names); len(kids) > 0 {
			entry["kids"] = kids
		}
		out = append(out, entry)
	}
	return out
}

// docNames is every indexed document's id mapped to the name it is shown under.
//
// Read from the store's records rather than from the terms (corpusDocs does
// that, and it walks every chunk in the base): a name is metadata, and paying
// a full pass over the text to answer it would make a status call expensive.
func docNames(live *kbRuntime) map[string]string {
	names := map[string]string{}
	for _, docID := range live.store.Documents() {
		name := docID
		if record, ok := live.store.DocumentByDocID(docID); ok && record.SourceFile != "" {
			name = filepath.Base(record.SourceFile)
		}
		names[docID] = name
	}
	return names
}

// subtreeDocsOf is every document filed under one node, at any depth.
func subtreeDocsOf(ix *dirtree.Index, nid string) []string {
	out := []string{}
	var walk func(string)
	walk = func(id string) {
		node := ix.Nodes[id]
		if node == nil {
			return
		}
		out = append(out, node.Docs...)
		for _, child := range node.Children {
			walk(child)
		}
	}
	walk(nid)
	return out
}

// handleDirSearch recalls documents through the directory tree.
func (k *kernel) handleDirSearch(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Query string `json:"query"`
		KB    string `json:"kb"`
		Limit int    `json:"limit"`
		Beam  int    `json:"beam"`
		Steps int    `json:"max_steps"`
		// MinProbability trusts nothing below this confidence and orders the
		// level by keyword score instead.
		MinProbability float64 `json:"min_probability"`
		// NoFallback drops the keyword safety net, so an evaluation can measure
		// what the route alone earns. Off by default.
		NoFallback    bool `json:"no_fallback"`
		FallbackLimit int  `json:"fallback_limit"`
		// NoLaya routes the same tree by keyword score. It exists so an
		// evaluation can separate "the tree is wrong" from "the model is
		// wrong" — without it, a bad recall number blames both at once.
		NoLaya bool `json:"no_laya"`
		// CandidateK narrows the options the model is shown to the K strongest
		// children by keyword score. 0 = show every child.
		CandidateK int `json:"candidate_k"`
		// Branches opens this many leaf directories instead of one. 1 = the
		// single-path descent. See dirtree.Options.Branches for why the width
		// of the descent, not the accuracy of its decisions, is the binding
		// constraint on this channel.
		Branches int `json:"branches"`
		// Slots fans the question out into at most this many independent
		// sub-questions and searches the tree once per slot. 0 = no fan-out.
		// Needs a reachable generator; without one it is silently skipped, and
		// the reply says so in trace.notes.
		Slots int `json:"slots"`
		// Explain attaches the terms behind every score and one readable
		// sentence per recalled document, plus the passage in it that matched
		// best. Output only: no decision reads it, and it does not change which
		// documents are recalled.
		Explain bool `json:"explain"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	params.Query = strings.TrimSpace(params.Query)
	if params.Query == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.query is required"}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	live.indexing.RLock()
	defer live.indexing.RUnlock()

	ix, err := k.dirIndexFor(live)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}
	if len(ix.Nodes) <= 1 {
		return nil, &ipc.Error{
			Code:    ipc.CodeInvalidParams,
			Message: "this knowledge base has no directory tree; run fs.index first",
		}
	}

	opts := dirtree.DefaultOptions()
	if params.Limit > 0 {
		opts.Limit = params.Limit
	}
	if params.Beam > 0 {
		opts.Beam = params.Beam
	}
	if params.Steps > 0 {
		opts.MaxSteps = params.Steps
	}
	if params.MinProbability > 0 {
		opts.MinProbability = params.MinProbability
	}
	if params.NoFallback {
		opts.Fallback = false
	}
	if params.FallbackLimit > 0 {
		opts.FallbackLimit = params.FallbackLimit
	}
	if params.CandidateK > 0 {
		opts.CandidateK = params.CandidateK
	}
	if params.Branches > 0 {
		opts.Branches = params.Branches
	}
	if params.Explain {
		opts.Explain = true
	}

	decider := k.routeDecider()
	if params.NoLaya {
		decider = nil
	}
	corpus := corpusDocs(live)

	// The slot table: split the question, search the tree once per part.
	// Skipped rather than failed when there is no generator — a fan-out over
	// one part is exactly the plain search, so the fallback is known-good.
	if params.Slots > 0 {
		if k.generator == nil {
			result := dirtree.Search(ctx, ix, corpus, params.Query, decider, opts)
			result.Trace.Notes = append(result.Trace.Notes,
				"slots requested but no generator is reachable; searched the question whole")
			return k.fsExplain(live, result, params.Query, params.Explain), nil
		}
		// A copy with thinking off, not the generator itself. Decomposition is
		// not a reasoning task — it is a split — and the thinking budget the
		// answer step needs turns a 1 s call into a 25 s one here, once per
		// question, which is the whole latency of the fan-out.
		planner := *k.generator
		noThink := false
		planner.Think = &noThink
		slots := agent.Decompose(ctx, &planner, params.Query, params.Slots)
		if len(slots) > 1 {
			return k.fsExplain(live, searchSlots(ctx, ix, corpus, params.Query, slots, decider, opts), params.Query, params.Explain), nil
		}
		result := dirtree.Search(ctx, ix, corpus, params.Query, decider, opts)
		result.Trace.Notes = append(result.Trace.Notes,
			"the question decomposed to one part; searched it whole")
		return k.fsExplain(live, result, params.Query, params.Explain), nil
	}

	result := dirtree.Search(ctx, ix, corpus, params.Query, decider, opts)
	return k.fsExplain(live, result, params.Query, params.Explain), nil
}

// explainEvidence attaches the best-matching passage to every recalled
// document.
//
// The strongest thing an explanation can show is the sentence the decision
// rests on, and this channel cannot produce it: the tree holds document ids and
// names, no text. So it is fetched here, from the store, after the recall is
// already decided — nothing in this function can change which documents were
// recalled, only what a reader is shown about them.
//
// Best passage = most distinct query terms, which is the same instrument the
// route used. A passage chosen by a different ranking than the one that
// recalled the document would be an explanation of something else.
func (k *kernel) explainEvidence(live *kbRuntime, result *dirtree.Result, query string) {
	terms := dirtree.TermsOf(query)
	if len(terms) == 0 {
		return
	}
	for i := range result.Hits {
		chunks := live.store.List(result.Hits[i].DocID, 0, 0, 0)
		best, bestScore := "", 0
		for _, chunk := range chunks {
			seen := map[string]bool{}
			for term := range dirtree.TermsOf(chunk.Text) {
				if terms[term] > 0 {
					seen[term] = true
				}
			}
			if len(seen) > bestScore {
				best, bestScore = chunk.Text, len(seen)
			}
		}
		if bestScore > 0 {
			result.Hits[i].Evidence = clipRunes(best, 240)
		}
	}
}

// clipRunes shortens a passage for display, keeping whole runes.
func clipRunes(text string, limit int) string {
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit]) + "…"
}

// fsExplain is fsReply with the explanation attached when it was asked for.
func (k *kernel) fsExplain(live *kbRuntime, result *dirtree.Result, query string,
	on bool) map[string]any {
	if on {
		k.explainEvidence(live, result, query)
	}
	return fsReply(result, live)
}

// fsReply is the wire shape of one fs.search reply.
func fsReply(result *dirtree.Result, live *kbRuntime) map[string]any {
	return withKB(map[string]any{
		"query":  result.Query,
		"hits":   result.Hits,
		"count":  len(result.Hits),
		"trace":  result.Trace,
		"chunks": 0, // the unit of recall here; stated so a caller cannot assume passages
	}, live)
}

// searchSlots searches the tree once per sub-question and merges the results.
//
// The merge is round-robin by rank, not concatenation: the first document of
// every slot comes before the second document of any. Concatenating hands the
// whole budget to slot 1 whenever it happens to fill it, which is the failure
// this channel already has in a different form — one path spending everything.
//
// Each slot searches with the full Limit and the merge keeps Limit. A slot that
// finds fewer documents than its share simply contributes fewer, and the space
// goes to the slots that found more.
func searchSlots(ctx context.Context, ix *dirtree.Index, corpus map[string]dirtree.Doc,
	query string, slots []string, decider dirtree.Decider, opts dirtree.Options) *dirtree.Result {
	lists := make([][]dirtree.Hit, 0, len(slots))
	trace := dirtree.Trace{Query: query, Documents: len(corpus)}
	if decider == nil {
		trace.Decider = "keyword"
	} else {
		trace.Decider = "laya"
	}
	for _, slot := range slots {
		one := dirtree.Search(ctx, ix, corpus, slot, decider, opts)
		for i := range one.Hits {
			one.Hits[i].Slot = slot
		}
		lists = append(lists, one.Hits)
		trace.Steps = append(trace.Steps, one.Trace.Steps...)
		trace.Notes = append(trace.Notes,
			fmt.Sprintf("slot %q -> %d document(s)", slot, len(one.Hits)))
	}
	return &dirtree.Result{Query: query, Hits: interleaveSlots(lists, opts.Limit), Trace: trace}
}

// interleaveSlots merges per-slot rankings round-robin, dropping repeats.
func interleaveSlots(lists [][]dirtree.Hit, limit int) []dirtree.Hit {
	out := make([]dirtree.Hit, 0, limit)
	seen := map[string]bool{}
	for rank := 0; len(out) < limit; rank++ {
		progressed := false
		for _, list := range lists {
			if rank >= len(list) {
				continue
			}
			hit := list[rank]
			if seen[hit.DocID] {
				continue
			}
			seen[hit.DocID] = true
			out = append(out, hit)
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

// deciderLabel names what will answer a routing decision, so a trace that says
// "keyword" is never mistaken for one the model made.
func (k *kernel) deciderLabel() string {
	if k.parse == nil || !k.layaReady {
		return "keyword"
	}
	return "laya"
}

// dirTreeOf is the section-tree analogue used when a document is forgotten: the
// directory tree names documents, so a deleted one has to leave it too, or a
// route would end at a document that no longer exists.
func (k *kernel) forgetDir(live *kbRuntime, docID string) {
	live.dirsMu.Lock()
	ix := live.dirs
	live.dirsMu.Unlock()
	if ix == nil {
		return
	}
	changed := false
	for _, node := range ix.Nodes {
		kept := make([]string, 0, len(node.Docs))
		for _, id := range node.Docs {
			if id == docID {
				changed = true
				continue
			}
			kept = append(kept, id)
		}
		node.Docs = kept
	}
	live.dirsMu.Lock()
	ix.Docs = removeString(ix.Docs, docID)
	live.dirsMu.Unlock()
	if !changed {
		return
	}
	if err := ix.Save(dirPath(live)); err != nil {
		log.Printf("warning: could not persist the directory tree: %v", err)
	}
}

func removeString(values []string, target string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == target {
			continue
		}
		out = append(out, value)
	}
	return out
}
