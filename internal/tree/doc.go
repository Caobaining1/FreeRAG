// Package tree is the vector-free retrieval channel: documents become trees of
// their own structure, and a query finds its evidence by routing down them.
//
// It sits beside internal/store rather than inside it, because the two answer
// different questions. The store answers "which passages match these terms",
// using a flat chunk list and BM25. This package answers "where in the document
// does the answer live", using the headings and reading order the parser has
// already recovered — no embeddings, no vector index, and no distance function
// that cannot be explained to anyone reading the result.
//
// Three properties are load-bearing and every test here asserts one of them:
//
//   - **Every chunk has a coordinate.** Each indexed chunk lands in exactly one
//     anchor, and each anchor hangs off exactly one node, so any passage the
//     channel returns can name the section path it came from. A hit without a
//     path is not evidence.
//   - **Structure is deterministic.** The tree is rebuilt from the same chunks
//     with the same rules, so it is byte-stable across runs and across machines.
//     Level inference may fall back to a flat tree — never to a guess.
//   - **Routing never lowers recall.** tree.Search always fuses with the store's
//     BM25 ranking, so the merged result cannot be worse than the keyword
//     baseline it is measured against.
package tree

// Policy names the build rules that produced a tree file. It is persisted with
// the tree and compared on load, so a tree built by an older version is rebuilt
// rather than served with silently different structure.
const Policy = "tree-v1"

// FileVersion is the on-disk layout version of tree.json.
const FileVersion = 1

// Node kinds.
const (
	KindDoc     = "doc"
	KindSection = "section"
)

// Node is one structural unit of a document: a chapter, a section, or the
// document itself.
//
// A node carries two things at once, and both are required: a **title** (what a
// router reads to decide where to descend) and **anchors** (the exact passages,
// named by their chunk ids, that justify keeping this node). A node with a title
// and no anchors is a table of contents; a node with anchors and no title is a
// haystack. Only the pair is searchable and verifiable.
type Node struct {
	NID      string   `json:"nid"`
	Title    string   `json:"title"`
	Level    int      `json:"level"`
	Kind     string   `json:"kind"`
	Page     int      `json:"page,omitempty"`
	Parent   string   `json:"parent,omitempty"`
	Children []string `json:"children,omitempty"`
	Anchors  []string `json:"anchors,omitempty"`
	Chars    int      `json:"chars"`
	// Summary replaces an LLM-generated abstract in this phase: the title plus
	// the opening of the node's first passage. Cheap, deterministic, and enough
	// for a router to compare siblings (P0 deliberately uses no summaries from a
	// model; see docs 无向量可信检索-实现方案.md §2.1).
	Summary string `json:"summary,omitempty"`
	// Keywords are the node's most distinctive terms, computed from its own
	// text. They are what later phases hand to an LLM as routing options, and
	// they are filled in when a document is indexed, not on load.
	Keywords []string `json:"keywords,omitempty"`
}

// Anchor is one passage attached to a node, named by the chunks it came from.
//
// It names rather than copies: the chunks live in the store, and duplicating
// every document's text into tree.json would double the index for no retrieval
// benefit. Quote verification later resolves the chunk ids and compares against
// the store's own copy, so there is exactly one version of every passage.
type Anchor struct {
	AID      string   `json:"aid"`
	NID      string   `json:"nid"`
	ChunkIDs []string `json:"chunk_ids"`
	Page     int      `json:"page"`
	Chars    int      `json:"chars"`
}

// DocTree is one document's structure.
type DocTree struct {
	DocID      string `json:"doc_id"`
	SourceFile string `json:"source_file,omitempty"`
	Version    string `json:"version,omitempty"`
	Policy     string `json:"policy"`
	// LevelsFrom says where this document's hierarchy came from: "font_size"
	// (a real heading hierarchy recovered from the PDF), "flat" (headings exist
	// but carry no level, so they are siblings), or "none" (no headings at all).
	// It is what stops a trace from claiming to have narrowed to a subsection
	// when the document never had subsections.
	LevelsFrom string             `json:"levels_from"`
	Root       string             `json:"root"`
	Nodes      map[string]*Node   `json:"nodes"`
	Anchors    map[string]*Anchor `json:"anchors"`
	ChunkCount int                `json:"chunk_count"`
}

// Path returns the node ids from the document root down to nid, inclusive. An
// unknown id yields an empty slice rather than a partial answer, because a route
// that silently stops halfway would look like a complete one.
func (t *DocTree) Path(nid string) []string {
	if t == nil || nid == "" {
		return nil
	}
	if _, ok := t.Nodes[nid]; !ok {
		return nil
	}
	out := []string{nid}
	for current := nid; current != ""; {
		node, ok := t.Nodes[current]
		if !ok || node.Parent == "" {
			break
		}
		out = append(out, node.Parent)
		current = node.Parent
	}
	// Reverse in place: built child-first, wanted root-first.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// PathTitles is Path rendered as titles — the human-readable form of the route,
// which is what a trace and a citation actually need.
func (t *DocTree) PathTitles(nid string) []string {
	path := t.Path(nid)
	if path == nil {
		return nil
	}
	out := make([]string, 0, len(path))
	for _, id := range path {
		out = append(out, t.Nodes[id].Title)
	}
	return out
}

// file is the persisted corpus form: every document's tree in one JSON file,
// written next to the knowledge base's index.
type file struct {
	Version int        `json:"version"`
	Policy  string     `json:"policy"`
	Docs    []*DocTree `json:"docs"`
}
