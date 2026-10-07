package tree

import (
	"sort"

	"freerag/internal/store"
)

// Hit is one returned passage, with the route that found it.
//
// Every field besides the chunk exists to make the hit answerable: `Path` names
// the sections above it, `Source` says which channel(s) produced it, and `Ranks`
// preserves the position it held in each of them so the fusion can be audited
// rather than believed.
type Hit struct {
	DocID   string         `json:"doc_id"`
	ChunkID string         `json:"chunk_id"`
	Chunk   store.Chunk    `json:"chunk"`
	Score   float64        `json:"score"`
	Source  string         `json:"source"` // tree | bm25 | both
	NodeID  string         `json:"nid,omitempty"`
	AID     string         `json:"aid,omitempty"`
	Page    int            `json:"page"`
	Path    []string       `json:"path,omitempty"`
	Ranks   map[string]int `json:"ranks,omitempty"`
}

// Result is what one tree search returns: the fused hits and the trace that
// explains them.
type Result struct {
	Query  string `json:"query"`
	Policy string `json:"policy"`
	Hits   []Hit  `json:"hits"`
	Trace  Trace  `json:"trace"`
}

// Trace is the run's audit record.
//
// It belongs to the answer rather than to a debug log: this channel's whole
// pitch is "you can check where it came from", and a trace nobody can read is
// not that.
type Trace struct {
	Query string `json:"query"`
	// Documents counts every document in the base while Routed counts the ones
	// the router descended into. The gap between them is what tells you whether
	// MaxDocs is costing recall.
	Documents  int         `json:"documents"`
	Routed     int         `json:"routed"`
	Untreed    []string    `json:"documents_without_tree,omitempty"`
	Unattached []string    `json:"documents_not_attached,omitempty"`
	Selected   []DocPick   `json:"selected,omitempty"`
	Routes     []*DocRoute `json:"routes,omitempty"`
	Fallback   int         `json:"fallback_hits"`
	Notes      []string    `json:"notes,omitempty"`
}

// DocPick records why a document made the candidate list.
//
// Both keys are normalised across this base's documents before anything compares
// them. A node route score and a BM25 score are different quantities, and picking
// a threshold on either would silently become picking one on the other as soon
// as the corpus changed size.
type DocPick struct {
	DocID   string  `json:"doc_id"`
	TreeKey float64 `json:"tree_key"`
	BM25Key float64 `json:"bm25_key"`
	Reason  string  `json:"reason"`
}

// Search runs one query through the tree channel and the BM25 safety net, then
// fuses them.
//
// The order of operations is deliberate:
//
//  1. rank every passage once, corpus-wide, with the store's own BM25 — that is
//     the baseline nothing here may fall below, and having the whole ranking up
//     front makes a later per-document lookup free;
//  2. choose the documents to route into, comparing normalised keys so neither
//     scale decides anything;
//  3. descend into them, collecting passages together with their section paths;
//  4. fuse the two rankings by reciprocal rank.
//
// Fusion is by RANK, not score: the tree's numbers are path qualities and BM25's
// are term weights, and averaging two different quantities produces a third that
// means nothing to whoever has to explain the result.
func Search(ix *Index, s *store.Store, query string, opts Options) *Result {
	opts = opts.withDefaults()
	result := &Result{Query: query, Policy: Policy, Trace: Trace{Query: query}}

	passages := s.SearchIn(query, 0, store.Filter{}) // 0 = every matching chunk
	scores := make(map[string]float64, len(passages))
	byChunk := make(map[string]store.Chunk, len(passages))
	bm25Order := make([]string, 0, len(passages))
	for _, hit := range passages {
		key := hit.Chunk.DocID + "\x00" + hit.Chunk.ChunkID
		scores[key] = hit.Score
		byChunk[key] = hit.Chunk
		bm25Order = append(bm25Order, key)
	}
	result.Trace.Fallback = len(bm25Order)

	docs := s.Documents()
	result.Trace.Documents = len(docs)

	// A document without a tree is still reachable through the fallback, which is
	// the reason the fallback is mandatory rather than a switch.
	treed := make([]string, 0, len(docs))
	for _, docID := range docs {
		switch {
		case ix.Tree(docID) == nil:
			result.Trace.Untreed = append(result.Trace.Untreed, docID)
		case !ix.Attached(docID):
			result.Trace.Unattached = append(result.Trace.Unattached, docID)
		default:
			treed = append(treed, docID)
		}
	}

	candidates := selectDocs(ix, treed, scores, query, opts)
	result.Trace.Selected = candidates

	// coords remembers, for every passage the tree reached, the section it came
	// from. It is populated during the descent because that is the only moment
	// the tree and its anchors are both in hand.
	coords := map[string]coord{}
	routes := make([]*DocRoute, 0, len(candidates))
	treeOrder := make([]string, 0, len(candidates)*opts.MaxAnchorsPerDoc)
	for _, pick := range candidates {
		docTree := ix.Tree(pick.DocID)
		if docTree == nil {
			continue
		}
		route := routeDoc(ix, pick.DocID, query, opts, func(chunkID string) float64 {
			return scores[pick.DocID+"\x00"+chunkID]
		})
		if route == nil {
			continue
		}
		routes = append(routes, route)
		nodesByID := map[string]RoutedNode{}
		for _, node := range route.Nodes {
			nodesByID[node.NID] = node
		}
		for _, aid := range route.Anchors {
			anchor, ok := docTree.Anchors[aid]
			if !ok {
				continue
			}
			node, ok := nodesByID[anchor.NID]
			if !ok {
				continue
			}
			for _, chunkID := range anchor.ChunkIDs {
				key := pick.DocID + "\x00" + chunkID
				if _, seen := coords[key]; seen {
					continue
				}
				coords[key] = coord{aid: aid, nid: anchor.NID, path: node.Titles}
				treeOrder = append(treeOrder, key)
			}
		}
	}
	result.Trace.Routed = len(routes)
	result.Trace.Routes = routes

	merged := map[string]*Hit{}
	order := make([]string, 0, len(treeOrder)+len(bm25Order))
	fuse := func(list []string, label string) {
		for rank, key := range list {
			hit, ok := merged[key]
			if !ok {
				chunk, have := byChunk[key]
				if !have {
					continue
				}
				hit = &Hit{
					DocID:   chunk.DocID,
					ChunkID: chunk.ChunkID,
					Chunk:   chunk,
					Page:    chunk.PageNum,
					Ranks:   map[string]int{},
				}
				merged[key] = hit
				order = append(order, key)
			}
			hit.Score += 1.0 / float64(opts.RRFK+rank+1)
			hit.Ranks[label] = rank + 1
			switch {
			case hit.Source == "":
				hit.Source = label
			case hit.Source != label:
				hit.Source = "both"
			}
		}
	}
	if opts.Fallback {
		fuse(bm25Order, "bm25")
	}
	fuse(treeOrder, "tree")

	hits := make([]Hit, 0, len(order))
	for _, key := range order {
		hit := merged[key]
		// Only a passage the tree actually reached may carry its coordinate. One
		// that arrived through BM25 keeps an empty path rather than borrowing a
		// neighbour's: claiming a section we did not arrive through is precisely
		// the kind of citation this channel exists to prevent.
		if found, ok := coords[key]; ok {
			hit.NodeID = found.nid
			hit.AID = found.aid
			hit.Path = found.path
		}
		hits = append(hits, *hit)
	}
	sort.SliceStable(hits, func(a, b int) bool {
		if hits[a].Score != hits[b].Score {
			return hits[a].Score > hits[b].Score
		}
		// A fused tie prefers the term match, the more explainable of the two.
		if rankOf(hits[a], "bm25") != rankOf(hits[b], "bm25") {
			return rankOf(hits[a], "bm25") < rankOf(hits[b], "bm25")
		}
		return hits[a].DocID+"\x00"+hits[a].ChunkID < hits[b].DocID+"\x00"+hits[b].ChunkID
	})
	if len(hits) > opts.Limit {
		hits = hits[:opts.Limit]
	}
	result.Hits = hits
	return result
}

// coord is one passage's place in the tree.
type coord struct {
	aid  string
	nid  string
	path []string
}

// rankOf returns a hit's 1-based rank in one channel, or a very large number when
// the channel never ranked it.
func rankOf(hit Hit, label string) int {
	if rank, ok := hit.Ranks[label]; ok {
		return rank
	}
	return int(^uint(0) >> 1)
}

// selectDocs picks the documents the router descends into.
//
// Both channels nominate. The tree nominates by how well this document's own
// section titles answer the query; BM25 nominates by its best passage. Each is
// scaled to [0,1] against the strongest document of that channel, and then the
// larger of the two decides — so neither scale, and neither channel alone, can
// keep a document out of the running. Everything else in the base stays on the
// fallback, which is what keeps this cap from costing recall.
func selectDocs(ix *Index, treed []string, scores map[string]float64, query string, opts Options) []DocPick {
	if len(treed) == 0 {
		return nil
	}

	raw := make([]DocPick, 0, len(treed))
	var maxTree, maxBM25 float64
	for _, docID := range treed {
		pick := DocPick{DocID: docID}
		if nodes := ix.nodes(docID); nodes != nil {
			tree := ix.Tree(docID)
			if children := tree.Nodes[tree.Root].Children; len(children) > 0 {
				childScores := nodes.score(query, children)
				for _, score := range childScores {
					if score > pick.TreeKey {
						pick.TreeKey = score
					}
				}
			} else {
				// A flat document — no sections — is scored on its root node, so
				// it can still be nominated instead of being skipped silently.
				pick.TreeKey = nodes.score(query, []string{tree.Root})[tree.Root]
			}
		}
		for key, score := range scores {
			if len(key) <= len(docID) || key[:len(docID)] != docID || key[len(docID)] != 0 {
				continue
			}
			if score > pick.BM25Key {
				pick.BM25Key = score
			}
		}
		if pick.TreeKey > maxTree {
			maxTree = pick.TreeKey
		}
		if pick.BM25Key > maxBM25 {
			maxBM25 = pick.BM25Key
		}
		raw = append(raw, pick)
	}

	keys := make([]DocPick, 0, len(raw))
	for _, pick := range raw {
		treeKey := 0.0
		if maxTree > 0 {
			treeKey = pick.TreeKey / maxTree
		}
		bm25Key := 0.0
		if maxBM25 > 0 {
			bm25Key = pick.BM25Key / maxBM25
		}
		switch {
		case treeKey > 0 && bm25Key > 0:
			pick.Reason = "both"
		case treeKey > 0:
			pick.Reason = "tree"
		case bm25Key > 0:
			pick.Reason = "bm25"
		default:
			continue
		}
		pick.TreeKey, pick.BM25Key = treeKey, bm25Key
		keys = append(keys, pick)
	}
	sort.SliceStable(keys, func(a, b int) bool {
		keyA := max(keys[a].TreeKey, keys[a].BM25Key)
		keyB := max(keys[b].TreeKey, keys[b].BM25Key)
		if keyA != keyB {
			return keyA > keyB
		}
		return keys[a].DocID < keys[b].DocID
	})
	if len(keys) > opts.MaxDocs {
		keys = keys[:opts.MaxDocs]
	}
	return keys
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
