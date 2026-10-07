package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"freerag/internal/agent"
	"freerag/internal/dirtree"
	"freerag/internal/store"
)

// dirTreeChannel is the corpus directory tree, as a retrieval tool the agent
// can call.
//
// The tree itself lives behind the 目录树 screen, which builds it and draws it.
// What this file adds is the other half: a question asked in 问答 can be routed
// down that same tree. The two are the same artefact — a tree drawn and then
// ignored would be decoration, and the reason to build one is that a question
// can be answered by descending it.
//
// It holds the runtime rather than an id: the channel is wired into that base's
// own toolbox, so the two live and die together, and resolving an id here would
// mean calling kbFor from inside loadBase — which holds the registry lock and
// would deadlock on itself.
type dirTreeChannel struct {
	k    *kernel
	live *kbRuntime
}

// Has reports whether this knowledge base has a tree.
//
// Cheap by design — it is asked on every round of every question — so it reads
// the cached index and never reads the corpus. It is also asked fresh every
// time, which is what lets a tree built mid-session (fs.index, from the 目录树
// screen) start being used without reopening the base.
func (d *dirTreeChannel) Has() bool {
	d.live.indexing.RLock()
	defer d.live.indexing.RUnlock()
	ix, err := d.k.dirIndexFor(d.live)
	return err == nil && len(ix.Nodes) > 1
}

// Search routes one query down the tree.
//
// The corpus is read here, once per call: the tree ranks folders by the terms of
// the documents in them, so it cannot route a question without knowing those
// terms. That is the cost of this channel, and it is why the tree is built once
// (fs.index) rather than per question.
func (d *dirTreeChannel) Search(ctx context.Context, query string, limit int, explain bool) ([]store.Hit, string, error) {
	live := d.live
	live.indexing.RLock()
	defer live.indexing.RUnlock()

	ix, err := d.k.dirIndexFor(live)
	if err != nil {
		return nil, "", err
	}
	if len(ix.Nodes) <= 1 {
		// The same test fs.search applies, and for the same reason: Open()
		// answers a missing file with an EMPTY index, so "no tree" arrives here
		// looking like a tree with no folders in it.
		return nil, "", errors.New("this knowledge base has no directory tree")
	}

	opts := dirtree.DefaultOptions()
	if limit > 0 {
		opts.Limit = limit
	}
	if explain {
		opts.Explain = true
	}

	result := dirtree.Search(ctx, ix, corpusDocs(live), query, d.k.routeDecider(), opts)
	hits := dirTreePassages(live, result, query)

	note := fmt.Sprintf("%d document(s) for %q (directory tree route, %s)", len(hits), query, d.k.deciderLabel())
	if len(result.Hits) > 0 && len(result.Hits[0].Path) > 0 {
		// Which folder the route ended in, because that is the answer's shape:
		// a route that opened the right folder and a route that opened the wrong
		// one both return documents, and only the path tells them apart.
		note += "; first folder: " + strings.Join(result.Hits[0].Path, " › ")
	}
	return hits, note, nil
}

// dirTreePassages turns recalled documents into evidence.
//
// The tree answers in documents; the evidence pool holds passages, and an answer
// is cited by passage. So each document is carried by the passage inside it that
// matched the query on most distinct query terms — the same instrument the route
// used, because a passage picked by a different ranking would be an explanation
// of some other decision.
//
// The chunk is a real one from the store, not a synthesised container: its id is
// what a citation points at, and a citation that cannot be opened is not one.
func dirTreePassages(live *kbRuntime, result *dirtree.Result, query string) []store.Hit {
	terms := dirtree.TermsOf(query)
	out := make([]store.Hit, 0, len(result.Hits))
	for _, hit := range result.Hits {
		chunks := live.store.List(hit.DocID, 0, 0, 0)
		if len(chunks) == 0 {
			continue
		}
		best, bestScore := chunks[0], -1
		for _, chunk := range chunks {
			seen := 0
			for term := range dirtree.TermsOf(chunk.Text) {
				if terms[term] > 0 {
					seen++
				}
			}
			if seen > bestScore {
				best, bestScore = chunk, seen
			}
		}
		out = append(out, store.Hit{
			Chunk:   best,
			Score:   hit.Score,
			Sources: []string{"dirtree"},
		})
	}
	return out
}

// dirTreeFor wires the channel into one base's toolbox.
//
// Wired whether or not a tree exists yet, because whether one exists is decided
// at call time: building a tree does not reopen the base, so a channel that
// asked once at load time would keep answering "no tree" for the rest of the
// session.
func (k *kernel) dirTreeFor(live *kbRuntime) agent.DirTree {
	return &dirTreeChannel{k: k, live: live}
}
