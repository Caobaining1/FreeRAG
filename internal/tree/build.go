package tree

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"freerag/internal/store"
)

// Build turns one document's chunks into a tree of its own structure.
//
// The input is what the parser emitted and the store indexed, in reading order:
// each chunk knows its block type, its page, and (for a PDF) the font size of
// the block that started it. That is enough to recover the document's shape
// without re-opening the file, which is why this package touches neither PDF nor
// models and can be built and tested on plain text.
//
// Two rules produce the whole hierarchy:
//
//  1. a heading chunk opens a node; everything after it belongs to that node
//     until another heading of the same or a shallower level appears;
//  2. every chunk becomes exactly one anchor of the node that is open when it
//     arrives — including the heading's own text, which otherwise disappears
//     into a title with nothing behind it.
//
// Rule 2 is what guarantees the "every chunk has a coordinate" property this
// package claims: there is no flat overflow pile where a chunk could end up
// without a path.
func Build(docID, sourceFile string, chunks []store.Chunk) *DocTree {
	title := strings.TrimSpace(sourceFile)
	if title == "" {
		title = docID
	}
	tree := &DocTree{
		DocID:      docID,
		SourceFile: sourceFile,
		Policy:     Policy,
		Root:       "n0",
		Nodes: map[string]*Node{
			"n0": {NID: "n0", Title: title, Level: 0, Kind: KindDoc},
		},
		Anchors: map[string]*Anchor{},
	}

	levels, levelsFrom := headingLevels(chunks)
	textOf := make(map[string]string, len(chunks))
	for _, chunk := range chunks {
		textOf[chunk.ChunkID] = chunk.Text
	}

	var (
		open     = []string{"n0"} // the chain of nodes accepting anchors, deepest last
		nextNode = 0
		anchorN  = 0
	)

	for i, chunk := range chunks {
		if strings.TrimSpace(chunk.Text) == "" {
			continue
		}
		if level, ok := levels[i]; ok {
			// Close every open node this heading does not sit under. A level is
			// only compared against other headings' levels, so it never matters
			// whether the numbers came from a font size rank or from the flat
			// fallback.
			for len(open) > 1 && tree.Nodes[open[len(open)-1]].Level >= level {
				open = open[:len(open)-1]
			}
			parent := open[len(open)-1]
			// A heading may jump two levels at once (a chapter straight to a
			// sub-subsection); clamping keeps the parent always shallower than
			// the child, which Path() relies on to terminate.
			if clamped := tree.Nodes[parent].Level + 1; level > clamped {
				level = clamped
			}
			nextNode++
			nid := fmt.Sprintf("n%d", nextNode)
			tree.Nodes[nid] = &Node{
				NID:    nid,
				Title:  headingTitle(chunk),
				Level:  level,
				Kind:   KindSection,
				Page:   chunk.PageNum,
				Parent: parent,
			}
			tree.Nodes[parent].Children = append(tree.Nodes[parent].Children, nid)
			open = append(open, nid)
		}

		owner := open[len(open)-1]
		anchorN++
		aid := fmt.Sprintf("a%04d", anchorN)
		tree.Anchors[aid] = &Anchor{
			AID:      aid,
			NID:      owner,
			ChunkIDs: []string{chunk.ChunkID},
			Page:     chunk.PageNum,
			Chars:    len(chunk.Text),
		}
		tree.Nodes[owner].Anchors = append(tree.Nodes[owner].Anchors, aid)
		tree.Nodes[owner].Chars += len(chunk.Text)
	}
	tree.ChunkCount = anchorN
	// Recorded, not logged: whether this document's hierarchy came from the
	// PDF's own font sizes or from the flat fallback decides how much of the
	// routing result to trust, and only the tree keeps that answer.
	tree.LevelsFrom = levelsFrom

	annotate(tree, textOf)
	return tree
}

// maxDepth caps how deep a level may go, so a font-size quirk in one heading
// cannot produce a 9-level tree from a 3-level document.
const maxDepth = 6

// headingLevels returns the level of every heading chunk, keyed by position.
//
// Where the levels come from decides whether the tree is real or flat:
//
//   - Declared first: a Markdown "### x" or a Word Heading3 arrives with the
//     depth the author wrote, and there is nothing to infer. Trusting it is not
//     a preference — it is the only way to read the author's own intent.
//   - PDF otherwise: every heading block carries the font size it was detected
//     with, and the distinct sizes in one document are its heading levels,
//     largest first. Ranking them turns "this line is big" into "this is a
//     chapter".
//   - Then flat: headings with no depth signal at all become level 1 siblings.
//     That is a FLAT tree, not a wrong one — routing still works on titles, it
//     just cannot descend — and LevelsFrom says so, because nothing downstream
//     could otherwise tell "descended three levels" from "there was nothing to
//     descend".
//
// Mixed signals fall back rather than blend: if any heading lacks the signal the
// others have, the whole document gets the next one down. Half of one hierarchy
// and half of another is worse than either.
func headingLevels(chunks []store.Chunk) (levels map[int]int, from string) {
	levels = map[int]int{}
	headings := make([]int, 0, 8)
	for i, chunk := range chunks {
		if isHeading(chunk) {
			headings = append(headings, i)
		}
	}
	if len(headings) == 0 {
		return levels, "none"
	}

	if declared := declaredLevels(chunks, headings); declared != nil {
		return declared, "declared"
	}

	sizes := make([]float64, 0, len(headings))
	seen := map[float64]bool{}
	for _, i := range headings {
		size := metaFloat(chunks[i].Metadata, "font_size")
		if size <= 0 {
			return flatLevels(headings), "flat"
		}
		if !seen[size] {
			seen[size] = true
			sizes = append(sizes, size)
		}
	}
	if len(sizes) < 2 {
		return flatLevels(headings), "flat"
	}

	sort.Sort(sort.Reverse(sort.Float64Slice(sizes)))
	rank := make(map[float64]int, len(sizes))
	for i, size := range sizes {
		if i >= maxDepth {
			// Everything below the cap shares the deepest level: the alternative
			// is dropping headings from the tree entirely.
			rank[size] = maxDepth
			continue
		}
		rank[size] = i + 1
	}
	for _, i := range headings {
		levels[i] = rank[metaFloat(chunks[i].Metadata, "font_size")]
	}
	return levels, "font_size"
}

// declaredLevels uses the depth each heading states about itself.
//
// Levels are renumbered rather than trusted as-is, because only the ORDER
// matters: a document whose first heading is "##" has no level 1, and feeding the
// raw numbers to a builder would leave everything else hanging under a gap. It
// returns nil when any heading is silent, so one unruly heading cannot produce a
// tree that is half right.
func declaredLevels(chunks []store.Chunk, headings []int) map[int]int {
	declared := make([]int, 0, len(headings))
	for _, i := range headings {
		level := int(metaFloat(chunks[i].Metadata, "level"))
		if level <= 0 {
			return nil
		}
		declared = append(declared, level)
	}

	distinct := make([]int, 0, len(declared))
	seen := map[int]bool{}
	for _, level := range declared {
		if !seen[level] {
			seen[level] = true
			distinct = append(distinct, level)
		}
	}
	if len(distinct) < 2 {
		// A real hierarchy needs two depths; one depth is a list, which the flat
		// case describes more honestly.
		return nil
	}
	sort.Ints(distinct)
	rank := make(map[int]int, len(distinct))
	for i, level := range distinct {
		if i >= maxDepth {
			rank[level] = maxDepth
			continue
		}
		rank[level] = i + 1
	}
	levels := make(map[int]int, len(headings))
	for j, i := range headings {
		levels[i] = rank[declared[j]]
	}
	return levels
}

// flatLevels puts every heading at level 1, directly under the document root.
func flatLevels(headings []int) map[int]int {
	levels := make(map[int]int, len(headings))
	for _, i := range headings {
		levels[i] = 1
	}
	return levels
}

// isHeading reports whether a chunk announces a section.
//
// "Section" is what the chunker calls a title that absorbed the body under it,
// and "Title" is one that did not — both are headings, and both open a node.
func isHeading(chunk store.Chunk) bool {
	switch chunk.BlockType {
	case "Title", "Section":
		return true
	}
	return false
}

// headingTitle extracts the heading line from a heading chunk.
//
// For "Title" that is the whole text. For "Section" the chunker folded the body
// in behind the heading, joined by two newlines, so the heading is the leading
// paragraph — but only when it reads like one. A section whose first block is
// already prose has no heading line to recover, and inventing one from the
// opening sentence would put a sentence in a title slot where nothing can tell
// it apart from a real heading.
func headingTitle(chunk store.Chunk) string {
	text := strings.TrimSpace(chunk.Text)
	if chunk.BlockType != "Section" {
		return truncateRunes(text, maxTitleRunes)
	}
	head := strings.TrimSpace(strings.SplitN(text, "\n\n", 2)[0])
	if head == "" || len([]rune(head)) > maxTitleRunes || strings.HasSuffix(head, "。") {
		return truncateRunes(text, maxTitleRunes)
	}
	return head
}

// maxTitleRunes is the longest line still believed to be a heading rather than
// a paragraph that merely sits at the top of one.
const maxTitleRunes = 120

// annotate fills in each node's summary text.
//
// The two Path* helpers and the summary are what a router reads; nothing here
// changes the structure. Summary is built after the whole tree exists because it
// needs the node's first passage, and that is only known once the last chunk has
// been attached.
func annotate(tree *DocTree, textOf map[string]string) {
	nids := make([]string, 0, len(tree.Nodes))
	for nid := range tree.Nodes {
		nids = append(nids, nid)
	}
	sort.Strings(nids)
	for _, nid := range nids {
		tree.Nodes[nid].Summary = summarize(tree, tree.Nodes[nid], textOf)
	}
}

// summarize is a node's human-readable description: its title plus the opening
// of the first passage it holds.
//
// It deliberately repeats no title text: a section's own leading line IS its
// title, and printing it twice would waste half of a router's budget on a string
// the caller already sees beside it.
func summarize(tree *DocTree, node *Node, textOf map[string]string) string {
	parts := []string{node.Title}
	for _, aid := range node.Anchors {
		anchor, ok := tree.Anchors[aid]
		if !ok || len(anchor.ChunkIDs) == 0 {
			continue
		}
		body, ok := textOf[anchor.ChunkIDs[0]]
		if !ok {
			continue
		}
		body = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(body), node.Title))
		body = firstSentence(body, summaryRunes)
		if body != "" {
			parts = append(parts, body)
		}
		break
	}
	if len(parts) == 1 {
		// A node with no text of its own is described by what it contains.
		subsections := make([]string, 0, len(node.Children))
		for _, child := range node.Children {
			subsections = append(subsections, tree.Nodes[child].Title)
		}
		if len(subsections) > 0 {
			parts = append(parts, "含："+strings.Join(subsections, "、"))
		}
	}
	return truncateRunes(strings.Join(parts, " — "), summaryRunes*2)
}

// summaryRunes is how much of a node's opening text a summary carries.
const summaryRunes = 160

// firstSentence returns the leading sentences of text, up to limit runes, cut on
// a sentence boundary rather than mid-word.
func firstSentence(text string, limit int) string {
	runes := []rune(text)
	if len(runes) <= limit {
		return strings.TrimSpace(text)
	}
	for i, r := range runes[:limit] {
		if r == '。' || r == '.' || r == '\n' {
			return strings.TrimSpace(string(runes[:i+1]))
		}
	}
	return strings.TrimSpace(string(runes[:limit])) + "…"
}

// truncateRunes shortens text to limit runes, never splitting a UTF-8 sequence.
func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit-1]) + "…"
}

// nodeText is what a router scores for one node: its own title, the titles it
// holds, and the passages attached directly to it.
//
// Descendants are NOT folded in. Doing so would make every ancestor contain the
// whole document, and then every section would score like the document does —
// which is exactly what the tree is there to prevent. A deep match is reached by
// descending, not by duplicating text upwards.
func nodeText(tree *DocTree, nid string, textOf map[string]string) string {
	node, ok := tree.Nodes[nid]
	if !ok {
		return ""
	}
	var b strings.Builder
	b.WriteString(node.Title)
	for _, child := range node.Children {
		if title := tree.Nodes[child].Title; title != "" {
			b.WriteString("\n")
			b.WriteString(title)
		}
	}
	for _, aid := range node.Anchors {
		anchor, ok := tree.Anchors[aid]
		if !ok {
			continue
		}
		for _, chunkID := range anchor.ChunkIDs {
			if text, ok := textOf[chunkID]; ok {
				b.WriteString("\n")
				b.WriteString(text)
			}
		}
	}
	return b.String()
}

// topTerms ranks a node's terms against the others in the same document.
//
// The scoring is tf-idf where a "document" is a node: a term in every node of
// the doc says nothing about which node to choose, which is why "员工手册" can
// lead no one anywhere while "年假" picks exactly one section.
func topTerms(tf map[string]int, df map[string]int, nodes, limit int) []string {
	if nodes <= 0 || limit <= 0 {
		return nil
	}
	type scored struct {
		term  string
		score float64
	}
	items := make([]scored, 0, len(tf))
	for term, count := range tf {
		freq := df[term]
		idf := math.Log(1 + (float64(nodes)-float64(freq)+0.5)/(float64(freq)+0.5))
		items = append(items, scored{term: term, score: float64(count) * idf})
	}
	sort.Slice(items, func(a, b int) bool {
		if items[a].score != items[b].score {
			return items[a].score > items[b].score
		}
		return items[a].term < items[b].term // deterministic ties
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

// metaFloat reads a numeric metadata field that survived a JSON round trip.
func metaFloat(meta map[string]any, key string) float64 {
	if meta == nil {
		return 0
	}
	switch value := meta[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	}
	return 0
}
