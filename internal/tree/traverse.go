package tree

import "sort"

// Options are the knobs of one tree-routed search.
type Options struct {
	// Limit is the number of hits returned.
	Limit int
	// MaxDocs caps how many documents the router descends into. The rest are
	// left to the BM25 fallback, which is the safety net this cap relies on.
	MaxDocs int
	// BeamWidth is how many siblings survive each level inside one document.
	BeamWidth int
	// MaxAnchorsPerDoc caps the passages one document may contribute.
	MaxAnchorsPerDoc int
	// MaxDepth stops a descent longer than this, which also bounds a malformed
	// tree (a cycle cannot exist: every node moves away from the root).
	MaxDepth int
	// FallbackLimit is how many BM25 hits always join the result, whether or not
	// the tree found them.
	FallbackLimit int
	// Fallback turns the BM25 safety net off. Only the evaluation uses this, to
	// measure what the tree alone actually contributes; everything else leaves it
	// on, because without it a bad root costs the whole answer.
	Fallback bool
	// RRFK is the reciprocal-rank-fusion damping constant.
	RRFK int
}

// DefaultOptions are the P0 starting points: one document deeper than the plan's
// beam of 2 would cost latency for almost no recall at desktop corpus sizes.
func DefaultOptions() Options {
	return Options{
		Limit:            20,
		MaxDocs:          8,
		BeamWidth:        2,
		MaxAnchorsPerDoc: 6,
		MaxDepth:         8,
		FallbackLimit:    20,
		Fallback:         true,
		RRFK:             60,
	}
}

func (o Options) withDefaults() Options {
	if o.Limit <= 0 {
		o.Limit = 20
	}
	if o.MaxDocs <= 0 {
		o.MaxDocs = 8
	}
	if o.BeamWidth <= 0 {
		o.BeamWidth = 2
	}
	if o.MaxAnchorsPerDoc <= 0 {
		o.MaxAnchorsPerDoc = 6
	}
	if o.MaxDepth <= 0 {
		o.MaxDepth = 8
	}
	if o.FallbackLimit <= 0 {
		o.FallbackLimit = 20
	}
	if o.RRFK <= 0 {
		o.RRFK = 60
	}
	return o
}

// RoutedNode is one node the descent chose, with the route that reached it.
type RoutedNode struct {
	NID    string   `json:"nid"`
	Path   []string `json:"path"`
	Titles []string `json:"titles"`
	Score  float64  `json:"score"`
	// Terminal is false for a node the route passed through on its way down.
	// Its own passages are still evidence — an intro paragraph often carries the
	// sentence the subsection then qualifies.
	Terminal bool `json:"terminal"`
}

// DocRoute is what descending one document produced.
type DocRoute struct {
	DocID string `json:"doc_id"`
	// Score is the strongest path found in this document. It is comparable only
	// BETWEEN documents for the same query — it is a route quality measure, not
	// a relevance probability, and nothing downstream may read it as one.
	Score   float64      `json:"score"`
	Nodes   []RoutedNode `json:"nodes"`
	Anchors []string     `json:"anchors"`
	Steps   []StepTrace  `json:"steps"`
	Levels  string       `json:"levels_from"`
}

// StepTrace is one level of one document's descent: what was offered, what was
// kept, and what was cut.
//
// This is the audit trail that makes a vector-free channel verifiable: a wrong
// answer can be replayed to the exact sibling that was pruned, instead of being
// traced to a distance nobody can question.
type StepTrace struct {
	Level  int           `json:"level"`
	From   string        `json:"from"`
	Scores []OptionScore `json:"scores"`
	Kept   []string      `json:"kept"`
	Pruned []string      `json:"pruned"`
}

// OptionScore is one sibling's score at one level.
type OptionScore struct {
	NID   string  `json:"nid"`
	Title string  `json:"title"`
	Score float64 `json:"score"`
}

// depthDiscount damps a parent's score as the route goes deeper.
//
// Without it a long chain of mediocre matches outscores one strong section,
// because path scores otherwise only accumulate. 0.6 says a grandparent's
// contribution counts for 60% of what it was worth one level higher.
const depthDiscount = 0.6

// candidate is one open node during a descent.
type candidate struct {
	nid    string
	score  float64
	path   []string
	titles []string
	depth  int
}

// routeDoc descends one document's tree and returns the passages it reached.
//
// `passageScore` answers how much one chunk matches the query, and is supplied
// by the caller because ranking inside a node must use the corpus-wide keyword
// score: ranking node children and their passages on two different scales would
// make the order the caller sees depend on which scale happened to be larger.
func routeDoc(ix *Index, docID, query string, opts Options, passageScore func(chunkID string) float64) *DocRoute {
	nodes := ix.nodes(docID)
	tree := ix.Tree(docID)
	if nodes == nil || tree == nil {
		return nil
	}
	if _, ok := tree.Nodes[tree.Root]; !ok {
		return nil
	}
	opts = opts.withDefaults()

	rootTitle := tree.Nodes[tree.Root].Title
	open := []candidate{{nid: tree.Root, path: []string{tree.Root}, titles: []string{rootTitle}}}
	route := &DocRoute{DocID: docID, Levels: tree.LevelsFrom}

	var reached []RoutedNode
	for depth := 0; depth < opts.MaxDepth && len(open) > 0; depth++ {
		var next []candidate
		for _, current := range open {
			children := tree.Nodes[current.nid].Children
			if len(children) == 0 {
				reached = append(reached, RoutedNode{
					NID: current.nid, Path: current.path, Titles: current.titles,
					Score: current.score, Terminal: true,
				})
				continue
			}
			// An intermediate node's own passages are evidence too, so the node
			// is recorded whether or not the descent continues below it.
			if len(tree.Nodes[current.nid].Anchors) > 0 {
				reached = append(reached, RoutedNode{
					NID: current.nid, Path: current.path, Titles: current.titles,
					Score: current.score, Terminal: false,
				})
			}

			scores := nodes.score(query, children)
			ranked := make([]OptionScore, 0, len(children))
			for _, nid := range children {
				ranked = append(ranked, OptionScore{NID: nid, Title: tree.Nodes[nid].Title, Score: scores[nid]})
			}
			sort.SliceStable(ranked, func(a, b int) bool {
				if ranked[a].Score != ranked[b].Score {
					return ranked[a].Score > ranked[b].Score
				}
				return ranked[a].NID < ranked[b].NID // deterministic ties
			})

			kept := ranked
			if len(kept) > opts.BeamWidth {
				kept = kept[:opts.BeamWidth]
			}
			// A sibling that matched nothing still survives when it is the only
			// one: cutting everything below a node whose children scored zero
			// would lose that branch entirely, and "no keyword overlap" is not
			// evidence against a heading.
			if len(kept) > 1 {
				trimmed := kept[:1]
				for _, option := range kept[1:] {
					if option.Score > 0 {
						trimmed = append(trimmed, option)
					}
				}
				kept = trimmed
			}

			step := StepTrace{Level: depth, From: current.nid, Scores: ranked}
			for _, option := range kept {
				step.Kept = append(step.Kept, option.NID)
				path := append(append([]string{}, current.path...), option.NID)
				titles := append(append([]string{}, current.titles...), option.Title)
				next = append(next, candidate{
					nid:    option.NID,
					score:  current.score*depthDiscount + option.Score,
					path:   path,
					titles: titles,
					depth:  current.depth + 1,
				})
			}
			for _, option := range ranked[len(kept):] {
				step.Pruned = append(step.Pruned, option.NID)
			}
			route.Steps = append(route.Steps, step)
		}
		open = next
	}

	if len(reached) == 0 {
		return nil
	}
	route.Nodes = reached
	best := 0.0
	for _, node := range reached {
		if node.Score > best {
			best = node.Score
		}
	}
	route.Score = best
	route.Anchors = pickAnchors(tree, reached, opts.MaxAnchorsPerDoc, passageScore)
	return route
}

// pickAnchors chooses which passages a routed subtree contributes.
//
// Two rounds, and the order matters. The first takes each reached node's best
// passage, so no chosen section can be silently dropped by a global sort that
// happened to rank another branch higher — the multi-hop lesson from
// JevDeepResearch: merging first and truncating afterwards discards exactly what
// the weaker branch alone had found. The second spends whatever slots are left
// on the highest-scoring remaining passages.
func pickAnchors(tree *DocTree, reached []RoutedNode, limit int, passageScore func(chunkID string) float64) []string {
	scoreOf := func(aid string) float64 {
		anchor, ok := tree.Anchors[aid]
		if !ok {
			return 0
		}
		best := 0.0
		for _, chunkID := range anchor.ChunkIDs {
			if score := passageScore(chunkID); score > best {
				best = score
			}
		}
		return best
	}

	seen := map[string]bool{}
	out := make([]string, 0, limit)
	for _, node := range reached {
		var bestAID string
		bestScore := -1.0
		for _, aid := range tree.Nodes[node.NID].Anchors {
			if seen[aid] {
				continue
			}
			if score := scoreOf(aid); score > bestScore {
				bestAID, bestScore = aid, score
			}
		}
		if bestAID == "" {
			continue
		}
		seen[bestAID] = true
		out = append(out, bestAID)
		if len(out) >= limit {
			return out
		}
	}

	rest := make([]string, 0, 16)
	for _, node := range reached {
		for _, aid := range tree.Nodes[node.NID].Anchors {
			if seen[aid] {
				continue
			}
			seen[aid] = true
			rest = append(rest, aid)
		}
	}
	sort.SliceStable(rest, func(a, b int) bool {
		scoreA, scoreB := scoreOf(rest[a]), scoreOf(rest[b])
		if scoreA != scoreB {
			return scoreA > scoreB
		}
		return rest[a] < rest[b]
	})
	for _, aid := range rest {
		if len(out) >= limit {
			break
		}
		out = append(out, aid)
	}
	return out
}
