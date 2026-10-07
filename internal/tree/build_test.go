package tree

import (
	"strings"
	"testing"

	"freerag/internal/store"
)

// handbook is a synthetic parsed document: what the sidecar hands the kernel for
// a PDF whose headings came out of layout detection with their font sizes. The
// numbers are the three heading levels this fixture claims to have — 24 the
// title, 16 a chapter, 13 a section.
func handbook() []store.Chunk {
	type row struct {
		text string
		kind string
		size float64
		page int
	}
	rows := []row{
		{"员工手册 2025", "Title", 24, 1},
		{"第一章 总则\n\n本手册适用于全体正式员工，自发布之日起施行。", "Section", 16, 1},
		{"第二章 薪酬与福利\n\n薪酬由基本工资、绩效奖金与年终奖构成。", "Section", 16, 3},
		{"2.1 基本工资\n\n基本工资按月发放，每月十五日为发薪日。", "Section", 13, 3},
		{"2.2 年终奖\n\n年终奖根据年度考核结果发放，通常在次年一月发放。", "Section", 13, 5},
		{"第三章 考勤与假期\n\n员工享有法定节假日与带薪年休假。", "Section", 16, 8},
		{"3.1 年假\n\n入职满一年者，每年享有十天年假；满五年者享有十五天年假。", "Section", 13, 9},
		{"3.2 病假\n\n病假需要提供医院出具的诊断证明，超过三天按事假处理。", "Section", 13, 10},
	}
	chunks := make([]store.Chunk, 0, len(rows))
	for i, item := range rows {
		meta := map[string]any{"page_num": item.page, "block_type": item.kind, "font_size": item.size}
		chunks = append(chunks, store.Chunk{
			ChunkID:   string(rune('a'+i)) + "000",
			Text:      item.text,
			DocID:     "handbook.pdf",
			PageNum:   item.page,
			BlockType: item.kind,
			Metadata:  meta,
		})
	}
	return chunks
}

func policyDoc() []store.Chunk {
	return []store.Chunk{
		{ChunkID: "c000", Text: "报销管理办法", DocID: "policy.pdf", PageNum: 1, BlockType: "Title",
			Metadata: map[string]any{"font_size": 24.0}},
		{ChunkID: "c001", Text: "第一章 差旅报销\n\n差旅费在返回后五个工作日内提交报销单。", DocID: "policy.pdf", PageNum: 1,
			BlockType: "Section", Metadata: map[string]any{"font_size": 16.0}},
		{ChunkID: "c002", Text: "1.1 住宿标准\n\n住宿报销上限为每晚四百元，需提供发票。", DocID: "policy.pdf", PageNum: 2,
			BlockType: "Section", Metadata: map[string]any{"font_size": 13.0}},
	}
}

// chunkIDOf makes each chunk id unique across the fixture, cheaply.
func chunkIDOf(prefix string, i int) string {
	return prefix + string(rune('0'+i))
}

func TestBuildLevelsComeFromFontSizes(t *testing.T) {
	chunks := handbook()
	for i := range chunks {
		chunks[i].ChunkID = chunkIDOf("h", i)
	}
	tree := Build("handbook.pdf", "handbook.pdf", chunks)

	if tree.LevelsFrom != "font_size" {
		t.Fatalf("LevelsFrom = %q, want font_size", tree.LevelsFrom)
	}
	if tree.ChunkCount != len(chunks) {
		t.Fatalf("ChunkCount = %d, want %d: every chunk must land in exactly one anchor", tree.ChunkCount, len(chunks))
	}

	byTitle := map[string]string{}
	for nid, node := range tree.Nodes {
		byTitle[node.Title] = nid
	}
	handled, ok := byTitle["3.1 年假"]
	if !ok {
		t.Fatalf("no node titled 3.1 年假; nodes: %v", titleList(tree))
	}
	titles := tree.PathTitles(handled)
	want := []string{"handbook.pdf", "员工手册 2025", "第三章 考勤与假期", "3.1 年假"}
	if len(titles) != len(want) {
		t.Fatalf("path = %v, want %v", titles, want)
	}
	for i := range want {
		if titles[i] != want[i] {
			t.Fatalf("path[%d] = %q, want %q (full: %v)", i, titles[i], want[i], titles)
		}
	}
}

func titleList(tree *DocTree) []string {
	out := make([]string, 0, len(tree.Nodes))
	for nid := range tree.Nodes {
		out = append(out, nid+":"+tree.Nodes[nid].Title)
	}
	return out
}

// TestEveryChunkHasExactlyOneAnchor is the invariant the whole package rests on:
// a passage with no tree coordinate could still be retrieved, but it could never
// say where it came from — which is the one thing this channel promises.
func TestEveryChunkHasExactlyOneAnchor(t *testing.T) {
	chunks := handbook()
	for i := range chunks {
		chunks[i].ChunkID = chunkIDOf("x", i)
	}
	tree := Build("handbook.pdf", "handbook.pdf", chunks)

	seen := map[string]int{}
	for aid, anchor := range tree.Anchors {
		if len(anchor.ChunkIDs) != 1 {
			t.Errorf("anchor %s covers %d chunks, want 1", aid, len(anchor.ChunkIDs))
		}
		for _, id := range anchor.ChunkIDs {
			seen[id]++
		}
		if _, ok := tree.Nodes[anchor.NID]; !ok {
			t.Errorf("anchor %s hangs off unknown node %s", aid, anchor.NID)
		}
	}
	for _, chunk := range chunks {
		if seen[chunk.ChunkID] != 1 {
			t.Errorf("chunk %s appears in %d anchors, want exactly 1", chunk.ChunkID, seen[chunk.ChunkID])
		}
	}
}

// TestBuildFallsBackToFlatTree covers the case every non-PDF format hits today:
// Headings with no font size. The tree must still exist — titles are routable —
// but must not pretend to be a hierarchy, because nothing downstream can tell a
// guessed level from a real one except this field.
func TestBuildFallsBackToFlatTree(t *testing.T) {
	chunks := []store.Chunk{
		{ChunkID: "c0", Text: "服务协议", DocID: "terms.md", PageNum: 1, BlockType: "Title"},
		{ChunkID: "c1", Text: "一、适用范围", DocID: "terms.md", PageNum: 1, BlockType: "Title"},
		{ChunkID: "c2", Text: "本协议适用于所有注册用户。", DocID: "terms.md", PageNum: 1, BlockType: "Text"},
		{ChunkID: "c3", Text: "二、费用", DocID: "terms.md", PageNum: 2, BlockType: "Title"},
		{ChunkID: "c4", Text: "服务费用由平台按季度结算。", DocID: "terms.md", PageNum: 2, BlockType: "Text"},
	}
	tree := Build("terms.md", "terms.md", chunks)

	if tree.LevelsFrom != "flat" {
		t.Fatalf("LevelsFrom = %q, want flat", tree.LevelsFrom)
	}
	root := tree.Nodes[tree.Root]
	if len(root.Children) != 3 {
		t.Fatalf("flat tree has %d sections under the root, want 3 (%v)", len(root.Children), titleList(tree))
	}
	for _, nid := range root.Children {
		if tree.Nodes[nid].Level != 1 {
			t.Errorf("node %s at level %d, want 1", nid, tree.Nodes[nid].Level)
		}
	}
}

// TestBuildUsesDeclaredHeadingLevels covers the common non-PDF case: Markdown and
// Word files state their own depth, and the tree must honour it rather than guess
// from appearance. Two documents with identical titles but different depths —
// "3.1 年假" as a `###` versus a `#` — have to end up in different places.
func TestBuildUsesDeclaredHeadingLevels(t *testing.T) {
	rows := []struct {
		text  string
		level int
	}{
		{"员工手册", 1},
		{"第一章 总则", 2},
		{"本手册适用于全体正式员工。", 0},
		{"第二章 考勤与假期", 2},
		{"员工享有法定节假日与带薪年休假。", 0},
		{"2.1 年假", 3},
		{"入职满一年者，每年享有十天年假。", 0},
		{"2.2 病假", 3},
		{"病假需要提供医院出具的诊断证明。", 0},
	}
	chunks := make([]store.Chunk, 0, len(rows))
	for i, row := range rows {
		kind := "Text"
		if row.level > 0 {
			kind = "Title"
		}
		chunks = append(chunks, store.Chunk{
			ChunkID:   chunkIDOf("m", i),
			Text:      row.text,
			DocID:     "handbook.md",
			PageNum:   1,
			BlockType: kind,
			Metadata:  map[string]any{"level": row.level},
		})
	}
	tree := Build("handbook.md", "handbook.md", chunks)

	if tree.LevelsFrom != "declared" {
		t.Fatalf("LevelsFrom = %q, want declared", tree.LevelsFrom)
	}
	var target *Node
	for _, node := range tree.Nodes {
		if node.Title == "2.1 年假" {
			target = node
		}
	}
	if target == nil {
		t.Fatal("no 2.1 年假 node")
	}
	want := []string{"handbook.md", "员工手册", "第二章 考勤与假期", "2.1 年假"}
	got := tree.PathTitles(target.NID)
	if len(got) != len(want) {
		t.Fatalf("path = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("path = %v, want %v", got, want)
		}
	}
	// The first chapter has no children, so it must not have swallowed the
	// second one: 2.2 病假 belongs to chapter two, not to chapter one.
	var illness *Node
	for _, node := range tree.Nodes {
		if node.Title == "2.2 病假" {
			illness = node
		}
	}
	if illness == nil || illness.Parent == "" {
		t.Fatal("2.2 病假 lost its parent")
	}
	if parent := tree.Nodes[illness.Parent].Title; parent != "第二章 考勤与假期" {
		t.Fatalf("2.2 病假 sits under %q, want 第二章 考勤与假期", parent)
	}
}

// TestDeclaredLevelsNeedAgreement guards the fallback rule: one heading that does
// not know its depth sends the whole document flat, because a tree that is half
// right is worse than one that admits to being flat.
func TestDeclaredLevelsNeedAgreement(t *testing.T) {
	chunks := []store.Chunk{
		{ChunkID: "p0", Text: "标题", DocID: "d.md", BlockType: "Title", Metadata: map[string]any{"level": 1}},
		{ChunkID: "p1", Text: "章节", DocID: "d.md", BlockType: "Title", Metadata: map[string]any{}},
		{ChunkID: "p2", Text: "正文。", DocID: "d.md", BlockType: "Text", Metadata: map[string]any{}},
	}
	tree := Build("d.md", "d.md", chunks)
	if tree.LevelsFrom != "flat" {
		t.Fatalf("LevelsFrom = %q, want flat when one heading has no level", tree.LevelsFrom)
	}
}

// TestBuildNoHeadings covers a document with no structure at all: everything has
// to land somewhere searchable rather than in a void behind the root.
func TestBuildNoHeadings(t *testing.T) {
	chunks := []store.Chunk{
		{ChunkID: "c0", Text: "第一段，讲了一件事。", DocID: "note.txt", PageNum: 1, BlockType: "Text"},
		{ChunkID: "c1", Text: "第二段，讲了另一件事。", DocID: "note.txt", PageNum: 1, BlockType: "Text"},
	}
	tree := Build("note.txt", "note.txt", chunks)
	if tree.LevelsFrom != "none" {
		t.Fatalf("LevelsFrom = %q, want none", tree.LevelsFrom)
	}
	if len(tree.Nodes[tree.Root].Anchors) != 2 {
		t.Fatalf("root holds %d anchors, want 2", len(tree.Nodes[tree.Root].Anchors))
	}
}

// TestSummarySkipsTheRepeatedTitle guards a small thing that costs a router real
// budget: a Section's own leading line is its title, so repeating it in the
// summary spends tokens to say the same thing twice.
func TestSummarySkipsTheRepeatedTitle(t *testing.T) {
	chunks := handbook()
	for i := range chunks {
		chunks[i].ChunkID = chunkIDOf("s", i)
	}
	tree := Build("handbook.pdf", "handbook.pdf", chunks)

	var found *Node
	for _, node := range tree.Nodes {
		if node.Title == "3.1 年假" {
			found = node
		}
	}
	if found == nil {
		t.Fatal("no 3.1 年假 node")
	}
	if want := "3.1 年假"; len(found.Summary) <= len(want) {
		t.Fatalf("summary %q has no body", found.Summary)
	}
	if count := strings.Count(found.Summary, "3.1 年假"); count != 1 {
		t.Fatalf("title appears %d times in summary %q, want 1", count, found.Summary)
	}
}
