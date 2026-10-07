package tree

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"freerag/internal/store"
)

// Index is every document tree in one knowledge base, plus the per-document
// scorer built over their node texts.
//
// Only the structure is persisted (tree.json, next to index.json). The scorer is
// rebuilt whenever a document's chunks are attached, because it is derived from
// text the store already owns: keeping a second copy on disk would need a second
// invalidation rule for every edit, and the first one missed would serve stale
// scoring for a chunk nobody changed.
type Index struct {
	mu   sync.RWMutex
	path string
	docs map[string]*DocTree
	live map[string]*docNodes // docID -> runtime scorer; absent means not attached
}

// docNodes scores one document's nodes against a query.
//
// It is BM25 with each node playing the part of a document: term frequencies per
// node, an inverse node frequency instead of an inverse document frequency, and
// length normalisation against the mean node. This is the whole reason a
// section-level router needs no model — the arithmetic is the same arithmetic
// the keyword index already trusts, just applied one level up.
type docNodes struct {
	tree    *DocTree
	terms   map[string]map[string]int // nid -> term -> count
	lengths map[string]int            // nid -> term count
	df      map[string]int            // term -> how many nodes contain it
	avgdl   float64
}

// Open reads an existing tree file, or starts empty when there is none.
//
// A file written by a different policy is discarded rather than migrated: the
// structure it describes was built by other rules, and guessing how it maps to
// these would put sections where they do not belong while everything downstream
// reports success.
func Open(path string) (*Index, error) {
	ix := &Index{path: path, docs: map[string]*DocTree{}, live: map[string]*docNodes{}}
	raw, err := os.ReadFile(path)
	switch {
	case err == nil:
		var payload file
		if jsonErr := json.Unmarshal(raw, &payload); jsonErr != nil {
			return nil, fmt.Errorf("tree: %s is not readable (%w); delete it to rebuild", path, jsonErr)
		}
		if payload.Policy != "" && payload.Policy != Policy {
			return ix, nil
		}
		for _, doc := range payload.Docs {
			if doc == nil || doc.DocID == "" {
				continue
			}
			ix.docs[doc.DocID] = doc
		}
		return ix, nil
	case !os.IsNotExist(err):
		return nil, fmt.Errorf("tree: read %s: %w", path, err)
	}
	return ix, nil
}

// Save writes the trees back to their file.
//
// Document trees are sorted by id so the file is byte-stable across runs: tree
// JSON is a build artefact, and a file that changes with no input changed makes
// every diff a lie.
func (ix *Index) Save() error {
	ix.mu.RLock()
	docs := make([]*DocTree, 0, len(ix.docs))
	for _, doc := range ix.docs {
		docs = append(docs, doc)
	}
	sort.Slice(docs, func(a, b int) bool { return docs[a].DocID < docs[b].DocID })
	payload := file{Version: FileVersion, Policy: Policy, Docs: docs}
	ix.mu.RUnlock()

	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("tree: marshal: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(ix.path), 0o755); err != nil {
		return fmt.Errorf("tree: %w", err)
	}
	return os.WriteFile(ix.path, raw, 0o644)
}

// Put installs (or replaces) one document's tree and builds its scorer.
//
// tree-less operation is impossible here on purpose: routing asks "which node",
// and without the node index the only answer available is a flat chunk score —
// the thing this channel exists to improve on. So Put takes the chunks and does
// both, and a caller cannot end up with a structure it cannot search.
func (ix *Index) Put(doc *DocTree, chunks []store.Chunk) {
	if doc == nil || doc.DocID == "" {
		return
	}
	textOf := make(map[string]string, len(chunks))
	for _, chunk := range chunks {
		textOf[chunk.ChunkID] = chunk.Text
	}
	nodes := newNodeIndex(doc, textOf)

	ix.mu.Lock()
	ix.docs[doc.DocID] = doc
	ix.live[doc.DocID] = nodes
	ix.mu.Unlock()
}

// BuildAndPut builds a document's tree from its chunks and installs it. It is
// the one call an indexing pass needs.
func (ix *Index) BuildAndPut(docID, sourceFile string, chunks []store.Chunk) *DocTree {
	ix.Put(Build(docID, sourceFile, chunks), chunks)
	return ix.Tree(docID)
}

// Remove drops a document's tree and its scorer.
func (ix *Index) Remove(docID string) {
	ix.mu.Lock()
	delete(ix.docs, docID)
	delete(ix.live, docID)
	ix.mu.Unlock()
}

// Tree returns one document's tree, or nil.
func (ix *Index) Tree(docID string) *DocTree {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.docs[docID]
}

// Docs lists the documents with a tree, in a stable order.
func (ix *Index) Docs() []string {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	out := make([]string, 0, len(ix.docs))
	for id := range ix.docs {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// Len is how many documents have a tree.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.docs)
}

// Attached reports whether a document's scorer is ready.
//
// Separate from Tree because the two diverge after a fresh process start: the
// structure loads from disk in one read, and each scorer is rebuilt only when
// that document's chunks are attached. Search skips what is not attached rather
// than failing, and says so in the trace.
func (ix *Index) Attached(docID string) bool {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	_, ok := ix.live[docID]
	return ok
}

func (ix *Index) nodes(docID string) *docNodes {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.live[docID]
}

// newNodeIndex computes every node's terms once, at attach time.
//
// Doing it here rather than per query is the difference between routing and brute
// force: the descent scores a handful of siblings per level against a table that
// was paid for when the document was indexed.
func newNodeIndex(doc *DocTree, textOf map[string]string) *docNodes {
	out := &docNodes{
		tree:    doc,
		terms:   make(map[string]map[string]int, len(doc.Nodes)),
		lengths: make(map[string]int, len(doc.Nodes)),
		df:      map[string]int{},
	}
	for nid := range doc.Nodes {
		tf := map[string]int{}
		total := 0
		for _, term := range store.Terms(nodeText(doc, nid, textOf)) {
			tf[term]++
			total++
		}
		out.terms[nid] = tf
		out.lengths[nid] = total
		for term := range tf {
			out.df[term]++
		}
	}
	count := len(doc.Nodes)
	if count > 0 {
		var total int
		for _, length := range out.lengths {
			total += length
		}
		out.avgdl = float64(total) / float64(count)
	}

	// Keywords are computed here rather than at build time because they need the
	// same term tables the routing does, and one pass answers both.
	for nid := range doc.Nodes {
		doc.Nodes[nid].Keywords = topTerms(out.terms[nid], out.df, count, keywordLimit)
	}
	return out
}

// keywordLimit is how many terms describe a node to a router.
const keywordLimit = 12

// BM25 parameters for node scoring — the store's own values, so a node score and
// a passage score are produced by the same arithmetic.
const (
	nodeK1 = 1.2
	nodeB  = 0.75
)

// score ranks the given nodes against a query, best-scoring highest.
//
// A node whose text contains none of the query terms scores exactly 0, not "not
// present": the caller sorts siblings, and a missing entry and a zero entry have
// to mean the same thing for that sort to be meaningful.
func (d *docNodes) score(query string, nids []string) map[string]float64 {
	out := make(map[string]float64, len(nids))
	terms := store.Terms(query)
	if len(terms) == 0 || len(d.tree.Nodes) == 0 {
		return out
	}

	counts := map[string]int{}
	for _, term := range terms {
		counts[term]++
	}
	total := len(d.terms)

	for _, nid := range nids {
		tf, ok := d.terms[nid]
		if !ok {
			continue
		}
		dl := float64(d.lengths[nid])
		score := 0.0
		for term, qtf := range counts {
			freq, ok := tf[term]
			if !ok {
				continue
			}
			frequency := d.df[term]
			idf := math.Log(1 + (float64(total)-float64(frequency)+0.5)/(float64(frequency)+0.5))
			denom := float64(freq) + nodeK1*(1-nodeB+nodeB*dl/avgdlOr(d))
			score += idf * (float64(freq) * (nodeK1 + 1) / denom) * float64(qtf)
		}
		out[nid] = score
	}
	return out
}

func avgdlOr(d *docNodes) float64 {
	if d.avgdl > 0 {
		return d.avgdl
	}
	return 1
}
