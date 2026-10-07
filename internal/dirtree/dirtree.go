// Package dirtree is the document-level retrieval channel: the corpus becomes a
// directory tree whose leaves are DOCUMENTS, and a query finds its evidence by
// choosing a direction at every level.
//
// It is deliberately a different thing from internal/tree, and the difference is
// the whole point:
//
//   - internal/tree builds a tree INSIDE one document (its sections) and returns
//     passages. Its unit of recall is the chunk.
//   - this package builds a tree OVER the corpus (directories of documents, like
//     a file system) and returns documents. Its unit of recall is the document,
//     and no chunk is ever ranked, selected or returned.
//
// The routing decision is a model decision, not an arithmetic one: at each node
// the caller's decider (Laya, sidecar/laya.py) is asked which of this node's
// children the question points at. That is what the structure exists for — a
// directory name is a claim about a group of documents, and "which claim does
// this question need" is a judgement, not a term overlap. When no decider is
// available the same tree is routed by document-level BM25, which is a fallback
// and says so in the trace.
package dirtree

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"freerag/internal/store"
)

// Policy names the build rules that produced a tree file, and is compared on
// load: a file written by other rules describes a different structure.
const Policy = "dirtree-v1"

// FileVersion is the on-disk layout version of dirs.json.
const FileVersion = 1

// Doc is one document as the builder sees it: an identity and its terms.
//
// Terms are counts, not text. The tree needs only enough of the document to say
// which other documents it belongs with, and building it from term counts keeps
// the whole corpus's text out of a second file.
type Doc struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Path  string         `json:"path,omitempty"`
	Terms map[string]int `json:"terms"`
}

// Node is one directory, or one leaf group of documents.
//
// Name and Summary are what a decider reads; Docs is what a leaf ANSWERS with.
// A node with neither children nor docs cannot exist — the builder never emits
// one — because a direction that leads nowhere is worse than no direction.
type Node struct {
	NID      string   `json:"nid"`
	Name     string   `json:"name"`
	Summary  string   `json:"summary,omitempty"`
	Keywords []string `json:"keywords,omitempty"`
	Level    int      `json:"level"`
	// From says how this node was made: "root", "path" (a real directory on
	// disk) or "cluster" (grouped by content). Mixed trees are normal and the
	// trace reports it, because "the user's own folder" and "a group we
	// invented" deserve different amounts of trust.
	From     string   `json:"from"`
	Parent   string   `json:"parent,omitempty"`
	Children []string `json:"children,omitempty"`
	Docs     []string `json:"docs,omitempty"`
}

// Index is one knowledge base's document directory tree.
type Index struct {
	Version int              `json:"version"`
	Policy  string           `json:"policy"`
	Root    string           `json:"root"`
	Nodes   map[string]*Node `json:"nodes"`
	Docs    []string         `json:"docs"`
	// From is what the whole tree was built from: "path", "cluster" or "mixed".
	From string `json:"from"`
}

// BuildOptions are the shape of the tree, not of a query.
type BuildOptions struct {
	// Branch is the widest a directory gets: how many children one split makes.
	Branch int
	// LeafSize is the most documents one leaf may hold before it is split again.
	// It is also what bounds a decider's final choice: a leaf is small enough
	// that "which of these documents" is a question worth asking.
	LeafSize int
	// MaxDepth stops a runaway split. A tree deeper than this is not more
	// precise, it is just more decisions that can each be wrong.
	MaxDepth int
	// KeywordLimit is how many terms name and describe a directory.
	KeywordLimit int
}

// DefaultBuildOptions are the P0 shape: six folders per level, four documents
// per leaf. Six is about what a typed decision can hold apart — Laya scores one
// position per option, and a dozen near-identical lines are not a choice — and
// four keeps a leaf's final decision honest.
func DefaultBuildOptions() BuildOptions {
	return BuildOptions{Branch: 6, LeafSize: 4, MaxDepth: 4, KeywordLimit: 8}
}

func (o BuildOptions) withDefaults() BuildOptions {
	if o.Branch <= 0 {
		o.Branch = 6
	}
	if o.LeafSize <= 0 {
		o.LeafSize = 4
	}
	if o.MaxDepth <= 0 {
		o.MaxDepth = 4
	}
	if o.KeywordLimit <= 0 {
		o.KeywordLimit = 8
	}
	return o
}

// Build turns a corpus into a directory tree.
//
// Two sources of structure, in the order they are trusted:
//
//  1. the file system the documents came from. A folder the user made is a claim
//     they wrote down, and ignoring it to invent our own would be arrogance with
//     a compute bill;
//  2. content, when the folders run out. Documents that share a folder are split
//     by their terms until a leaf is small enough to answer with.
//
// Documents whose paths say nothing (all in one folder) go straight to (2), so a
// flat corpus still gets a tree rather than a list.
func Build(docs []Doc, opts BuildOptions) *Index {
	opts = opts.withDefaults()
	ordered := make([]Doc, len(docs))
	copy(ordered, docs)
	sort.Slice(ordered, func(a, b int) bool { return ordered[a].ID < ordered[b].ID })

	ix := &Index{
		Version: FileVersion,
		Policy:  Policy,
		Root:    "d0",
		Nodes: map[string]*Node{
			"d0": {NID: "d0", Name: "语料库", Level: 0, From: "root"},
		},
	}
	for _, doc := range ordered {
		ix.Docs = append(ix.Docs, doc.ID)
	}

	idf := corpusIDF(ordered)
	next := 0
	usedPath := false
	usedCluster := false

	var grow func(parent string, level int, group []Doc, segs []string)
	grow = func(parent string, level int, group []Doc, segs []string) {
		if len(group) == 0 {
			return
		}
		// A path split wins while there are folders left to use, but only when
		// it actually divides: a directory holding every document is a level
		// that costs a decision and decides nothing.
		if len(segs) > 0 {
			byName := map[string][]Doc{}
			names := make([]string, 0, 4)
			for _, doc := range group {
				name := segment(doc, segs)
				if _, ok := byName[name]; !ok {
					names = append(names, name)
				}
				byName[name] = append(byName[name], doc)
			}
			if len(names) > 1 {
				usedPath = true
				sort.Strings(names)
				// Same reason as the cluster branch: ids are reserved for the
				// whole level before any child is descended into.
				base := next + 1
				next += len(names)
				for i, name := range names {
					nid := fmt.Sprintf("d%d", base+i)
					ix.Nodes[nid] = &Node{NID: nid, Name: name, Level: level + 1,
						From: "path", Parent: parent}
					ix.Nodes[parent].Children = append(ix.Nodes[parent].Children, nid)
					grow(nid, level+1, byName[name], segs[1:])
				}
				return
			}
		}
		if len(group) <= opts.LeafSize || level >= opts.MaxDepth {
			for _, doc := range group {
				ix.Nodes[parent].Docs = append(ix.Nodes[parent].Docs, doc.ID)
			}
			return
		}
		parts := split(group, idf, opts)
		if len(parts) <= 1 {
			for _, doc := range group {
				ix.Nodes[parent].Docs = append(ix.Nodes[parent].Docs, doc.ID)
			}
			return
		}
		usedCluster = true
		// Names come from the split itself: a cluster is described by the terms
		// that separate it from its siblings, which is exactly the distinction
		// the split was made on.
		labels := nameParts(parts, idf, opts.KeywordLimit)
		// Ids are reserved BEFORE recursing. `next` is the shared counter and a
		// child's own split advances it, so allocating inside the loop hands the
		// second sibling an id the first sibling's subtree already took — the
		// node is then overwritten, a parent ends up among its own children,
		// and every walk of the tree recurses forever.
		base := next + 1
		next += len(parts)
		for i, part := range parts {
			nid := fmt.Sprintf("d%d", base+i)
			ix.Nodes[nid] = &Node{NID: nid, Name: labels[i], Level: level + 1,
				From: "cluster", Parent: parent, Keywords: keywordsOf(part, idf, opts.KeywordLimit)}
			ix.Nodes[parent].Children = append(ix.Nodes[parent].Children, nid)
			grow(nid, level+1, part, nil)
		}
	}

	grow("d0", 0, ordered, pathSegments(ordered))
	describe(ix, ordered, idf, opts)
	switch {
	case usedPath && usedCluster:
		ix.From = "mixed"
	case usedPath:
		ix.From = "path"
	default:
		ix.From = "cluster"
	}
	return ix
}

// segment returns the directory name that owns this document at this depth.
//
// `segs` is the fixed depth of the paths being consumed; a document with fewer
// folders than the rest falls into the shared remainder rather than being
// dropped, which is the difference between a shallow file and a lost one.
func segment(doc Doc, segs []string) string {
	parts := splitPath(doc.Path)
	if len(parts) == 0 {
		return "_"
	}
	// A document with fewer folders than the level being consumed keeps its
	// own deepest folder: dropping it would lose the file, and inventing a
	// level for it would put it somewhere the user never put it.
	index := len(segs) - 1
	if index >= len(parts) {
		index = len(parts) - 1
	}
	if index < 0 {
		index = 0
	}
	return parts[index]
}

// pathSegments returns how many directory levels the corpus's real paths offer,
// as a slice whose length is that count (its contents are unused).
//
// Zero when the documents share a folder: a corpus in one directory has no
// directory structure to honour, and pretending otherwise would spend a level
// naming it.
func pathSegments(docs []Doc) []string {
	depths := map[int]bool{}
	dirs := map[string]bool{}
	for _, doc := range docs {
		parts := splitPath(doc.Path)
		if len(parts) > 0 {
			depths[len(parts)] = true
			dirs[strings.Join(parts, "/")] = true
		}
	}
	if len(dirs) < 2 {
		return nil
	}
	min := 1 << 30
	for depth := range depths {
		if depth < min {
			min = depth
		}
	}
	if min == 1<<30 {
		return nil
	}
	return make([]string, min)
}

func splitPath(path string) []string {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	dir := filepath.ToSlash(filepath.Dir(path))
	dir = strings.Trim(dir, "/")
	if dir == "" || dir == "." {
		return nil
	}
	return strings.Split(dir, "/")
}

// corpusIDF is the inverse document frequency over the corpus's documents.
func corpusIDF(docs []Doc) map[string]float64 {
	df := map[string]int{}
	for _, doc := range docs {
		for term := range doc.Terms {
			df[term]++
		}
	}
	total := float64(len(docs))
	idf := make(map[string]float64, len(df))
	for term, count := range df {
		idf[term] = math.Log(1 + (total-float64(count)+0.5)/(float64(count)+0.5))
	}
	return idf
}

// split groups documents by their terms, deterministically.
//
// k-means over cosine similarity on idf-weighted term vectors, seeded by taking
// every m-th document in id order rather than at random: the same corpus must
// produce the same tree on every machine, and a tree that moves between runs
// cannot be diffed, cached or trusted.
func split(group []Doc, idf map[string]float64, opts BuildOptions) [][]Doc {
	k := opts.Branch
	if k > len(group) {
		k = len(group)
	}
	if k < 2 {
		return nil
	}
	vectors := make([]map[string]float64, len(group))
	for i, doc := range group {
		vectors[i] = weighted(doc.Terms, idf)
	}
	step := len(group) / k
	if step < 1 {
		step = 1
	}
	centroids := make([]map[string]float64, 0, k)
	for i := 0; i < k; i++ {
		centroids = append(centroids, clone(vectors[min(i*step, len(group)-1)]))
	}

	assign := make([]int, len(group))
	for iteration := 0; iteration < 8; iteration++ {
		moved := false
		for i, vector := range vectors {
			best, bestScore := 0, -1.0
			for c, centroid := range centroids {
				if score := cosine(vector, centroid); score > bestScore {
					best, bestScore = c, score
				}
			}
			if assign[i] != best {
				assign[i] = best
				moved = true
			}
		}
		next := make([]map[string]float64, len(centroids))
		counts := make([]int, len(centroids))
		for i, vector := range vectors {
			if next[assign[i]] == nil {
				next[assign[i]] = map[string]float64{}
			}
			for term, value := range vector {
				next[assign[i]][term] += value
			}
			counts[assign[i]]++
		}
		for c := range next {
			if counts[c] == 0 {
				// An empty cluster keeps its seed rather than vanishing: a split
				// that loses a branch would silently shrink the corpus.
				next[c] = clone(centroids[c])
				continue
			}
			for term := range next[c] {
				next[c][term] /= float64(counts[c])
			}
		}
		centroids = next
		if !moved {
			break
		}
	}

	parts := make([][]Doc, len(centroids))
	for i, doc := range group {
		parts[assign[i]] = append(parts[assign[i]], doc)
	}
	out := make([][]Doc, 0, len(parts))
	for _, part := range parts {
		if len(part) > 0 {
			out = append(out, part)
		}
	}
	// One cluster means the split found no distinction at all. Answering with a
	// single child would make every query pay a decision that cannot change
	// anything, so the caller turns this group into a leaf instead.
	if len(out) < 2 {
		return nil
	}
	sort.SliceStable(out, func(a, b int) bool { return len(out[a]) > len(out[b]) })
	return out
}

func weighted(tf map[string]int, idf map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(tf))
	for term, count := range tf {
		out[term] = float64(count) * idf[term]
	}
	return out
}

func clone(in map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(in))
	for term, value := range in {
		out[term] = value
	}
	return out
}

func cosine(a, b map[string]float64) float64 {
	if len(a) > len(b) {
		a, b = b, a
	}
	var dot, na, nb float64
	for term, value := range a {
		if other, ok := b[term]; ok {
			dot += value * other
		}
		na += value * value
	}
	for _, value := range b {
		nb += value * value
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// nameParts names sibling clusters by the terms that separate them.
//
// tf-idf where a "document" is a cluster: a term in every sibling says nothing
// about which one to choose, so it cannot name any of them.
func nameParts(parts [][]Doc, idf map[string]float64, limit int) []string {
	tf := make([]map[string]int, len(parts))
	df := map[string]int{}
	for i, part := range parts {
		tf[i] = map[string]int{}
		for _, doc := range part {
			for term, count := range doc.Terms {
				tf[i][term] += count
			}
		}
		for term := range tf[i] {
			df[term]++
		}
	}
	total := len(parts)
	names := make([]string, len(parts))
	used := map[string]bool{}
	for i := range parts {
		type scored struct {
			term  string
			score float64
		}
		items := make([]scored, 0, len(tf[i]))
		for term, count := range tf[i] {
			freq := df[term]
			value := math.Log(1 + (float64(total)-float64(freq)+0.5)/(float64(freq)+0.5))
			items = append(items, scored{term: term, score: float64(count) * value})
		}
		sort.Slice(items, func(a, b int) bool {
			if items[a].score != items[b].score {
				return items[a].score > items[b].score
			}
			return items[a].term < items[b].term
		})
		terms := make([]string, 0, limit)
		for _, item := range items {
			if len(terms) >= limit {
				break
			}
			terms = append(terms, item.term)
		}
		// Named after the documents it actually holds, not after its terms.
		//
		// A folder is a claim a decider has to judge, and "bankman·ftx·alameda"
		// is not a claim anyone can judge — it is a bag of words. The two most
		// central documents' titles are: they say what the folder is about in
		// the words the documents themselves use, which is the only description
		// a model that has never seen the corpus can act on. The terms are kept
		// in Keywords, where they belong.
		name := strings.Join(docTitles(parts[i], idf, 2), " / ")
		if name == "" {
			name = strings.Join(terms, "·")
		}
		if name == "" {
			name = fmt.Sprintf("组%d", i+1)
		}
		// Two clusters can deserve the same name. Disambiguated rather than
		// left duplicated: a decider shown two identical options is being asked
		// to guess, and whichever it picks will look arbitrary in the trace.
		if used[name] {
			name = fmt.Sprintf("%s(%d)", name, i+1)
		}
		used[name] = true
		names[i] = name
	}
	return names
}

// docTitles names a group by its most representative documents.
//
// "Representative" is the documents carrying the most of the group's own
// distinctive terms: the ones a reader would pick to say what this folder is.
// Titles, not ids — a filename is the only human-written description most
// documents bring with them, and it is what a decider can actually read.
func docTitles(group []Doc, idf map[string]float64, limit int) []string {
	if len(group) == 0 || limit <= 0 {
		return nil
	}
	tf := map[string]int{}
	df := map[string]int{}
	per := make([]map[string]int, len(group))
	for i, doc := range group {
		per[i] = doc.Terms
		for term, count := range doc.Terms {
			tf[term] += count
		}
	}
	for i := range group {
		for term := range per[i] {
			df[term]++
		}
	}
	total := float64(len(group))
	type scored struct {
		index int
		score float64
	}
	items := make([]scored, 0, len(group))
	for i := range group {
		score := 0.0
		for term, count := range per[i] {
			freq := float64(df[term])
			value := math.Log(1 + (total-freq+0.5)/(freq+0.5))
			score += float64(count) * value * idf[term]
		}
		items = append(items, scored{index: i, score: score})
	}
	sort.Slice(items, func(a, b int) bool {
		if items[a].score != items[b].score {
			return items[a].score > items[b].score
		}
		return group[items[a].index].ID < group[items[b].index].ID
	})
	out := make([]string, 0, limit)
	for i, item := range items {
		if i >= limit {
			break
		}
		out = append(out, nameOf(group[item.index].ID, group[item.index]))
	}
	return out
}

func keywordsOf(group []Doc, idf map[string]float64, limit int) []string {
	tf := map[string]int{}
	for _, doc := range group {
		for term, count := range doc.Terms {
			tf[term] += count
		}
	}
	type scored struct {
		term  string
		score float64
	}
	items := make([]scored, 0, len(tf))
	for term, count := range tf {
		items = append(items, scored{term: term, score: float64(count) * idf[term]})
	}
	sort.Slice(items, func(a, b int) bool {
		if items[a].score != items[b].score {
			return items[a].score > items[b].score
		}
		return items[a].term < items[b].term
	})
	out := make([]string, 0, limit)
	for i, item := range items {
		if i >= limit {
			break
		}
		out = append(out, item.term)
	}
	return out
}

// describe fills in what a decider reads: how many documents a node holds and
// what they are about.
func describe(ix *Index, docs []Doc, idf map[string]float64, opts BuildOptions) {
	byID := map[string]Doc{}
	for _, doc := range docs {
		byID[doc.ID] = doc
	}
	nids := make([]string, 0, len(ix.Nodes))
	for nid := range ix.Nodes {
		nids = append(nids, nid)
	}
	sort.Strings(nids)

	var count func(nid string) int
	count = func(nid string) int {
		node, ok := ix.Nodes[nid]
		if !ok {
			return 0
		}
		if len(node.Children) == 0 {
			return len(node.Docs)
		}
		total := 0
		for _, child := range node.Children {
			total += count(child)
		}
		return total
	}
	for _, nid := range nids {
		node := ix.Nodes[nid]
		switch {
		case len(node.Children) > 0:
			names := make([]string, 0, len(node.Children))
			for _, child := range node.Children {
				names = append(names, ix.Nodes[child].Name)
			}
			node.Summary = fmt.Sprintf("%d 篇，%d 个子目录：%s", count(nid),
				len(node.Children), strings.Join(names, "、"))
		default:
			names := make([]string, 0, len(node.Docs))
			for _, id := range node.Docs {
				names = append(names, nameOf(id, byID[id]))
			}
			node.Summary = fmt.Sprintf("%d 篇：%s", len(node.Docs), strings.Join(names, "、"))
		}
	}
}

// nameOf renders one document for a human reader, which is what a decider is
// shown when it must choose between documents rather than between directories.
func nameOf(id string, doc Doc) string {
	name := strings.TrimSpace(doc.Name)
	if name == "" {
		name = id
	}
	if idx := strings.LastIndex(name, "."); idx > 0 {
		name = name[:idx]
	}
	// Corpora like MultiHop-RAG prefix every file with a hash
	// ("ccf3551800_A_scientist_..."), and a folder named after a hash tells a
	// decider nothing. The prefix is dropped, the rest is what the document is
	// called.
	if parts := strings.FieldsFunc(name, func(r rune) bool { return r == '_' || r == ' ' || r == '-' }); len(parts) > 1 {
		first := parts[0]
		if len(first) >= 6 && isHexLike(first) {
			name = name[len(first):]
		}
	}
	name = strings.ReplaceAll(name, "_", " ")
	name = strings.TrimSpace(name)
	if len([]rune(name)) > 60 {
		name = string([]rune(name)[:59]) + "…"
	}
	return name
}

// isHexLike reports whether a leading filename token is a corpus id rather than
// a word: eight-plus hex characters are a hash in every corpus we have seen,
// and no English title starts with one.
func isHexLike(token string) bool {
	if len(token) < 6 {
		return false
	}
	for _, r := range token {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'a' && r <= 'f':
		case r >= 'A' && r <= 'F':
		default:
			return false
		}
	}
	return true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ---- persistence ----

// Open reads a tree file, or starts empty when there is none. A file written
// under another policy is discarded rather than migrated.
func Open(path string) (*Index, error) {
	raw, err := os.ReadFile(path)
	switch {
	case err == nil:
		ix := &Index{}
		if jsonErr := json.Unmarshal(raw, ix); jsonErr != nil {
			return nil, fmt.Errorf("dirtree: %s is not readable (%w); delete it to rebuild", path, jsonErr)
		}
		if ix.Policy != "" && ix.Policy != Policy {
			return &Index{Version: FileVersion, Policy: Policy, Root: "d0",
				Nodes: map[string]*Node{}}, nil
		}
		if ix.Nodes == nil {
			ix.Nodes = map[string]*Node{}
		}
		return ix, nil
	case !os.IsNotExist(err):
		return nil, fmt.Errorf("dirtree: read %s: %w", path, err)
	}
	return &Index{Version: FileVersion, Policy: Policy, Root: "d0",
		Nodes: map[string]*Node{}}, nil
}

// Save writes the tree next to the index.
func (ix *Index) Save(path string) error {
	ix.Version = FileVersion
	ix.Policy = Policy
	raw, err := json.MarshalIndent(ix, "", "  ")
	if err != nil {
		return fmt.Errorf("dirtree: marshal: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("dirtree: %w", err)
	}
	return os.WriteFile(path, raw, 0o644)
}

// Path returns the directory names from the root down to nid, inclusive.
func (ix *Index) Path(nid string) []string {
	if _, ok := ix.Nodes[nid]; !ok {
		return nil
	}
	out := []string{nid}
	for current := nid; current != ""; {
		node, ok := ix.Nodes[current]
		if !ok || node.Parent == "" {
			break
		}
		out = append(out, node.Parent)
		current = node.Parent
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// PathNames is Path rendered as the names a reader sees.
func (ix *Index) PathNames(nid string) []string {
	path := ix.Path(nid)
	if path == nil {
		return nil
	}
	out := make([]string, 0, len(path))
	for _, nid := range path {
		out = append(out, ix.Nodes[nid].Name)
	}
	return out
}

// TermsOf counts one document's terms. Kept next to the builder because the
// tree's vocabulary is the store's vocabulary: routing, clustering and the
// keyword fallback all have to speak the same terms or the same document gets
// two different descriptions.
func TermsOf(text string) map[string]int {
	out := map[string]int{}
	for _, term := range store.Terms(text) {
		out[term]++
	}
	return out
}
