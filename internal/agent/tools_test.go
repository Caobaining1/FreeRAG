package agent

import (
	"context"
	"strings"
	"testing"

	"freerag/internal/store"
)

func toolStore(t *testing.T) *store.Store {
	t.Helper()
	s := store.New()
	s.Add([]store.Chunk{
		{ChunkID: "c0", DocID: "report.pdf", PageNum: 1, BlockType: "Title",
			Text:     "Freerag Acceptance Report",
			Metadata: map[string]any{"source_file": "report.pdf", "block_type": "Title", "page_num": float64(1)}},
		{ChunkID: "c1", DocID: "report.pdf", PageNum: 1, BlockType: "Text",
			Text:     "The standard is GB/T 1234 and the model is Qwen3-4B.\nSecond line mentions nothing notable.",
			Metadata: map[string]any{"source_file": "report.pdf", "block_type": "Text", "page_num": float64(1)}},
		{ChunkID: "c2", DocID: "report.pdf", PageNum: 2, BlockType: "Table",
			Text:     "| header | value |\n| --- | --- |\n| rows | 3 |",
			Metadata: map[string]any{"source_file": "report.pdf", "block_type": "Table", "page_num": float64(2)}},
		{ChunkID: "c3", DocID: "notes.pdf", PageNum: 1, BlockType: "Text",
			Text:     "Sufficiency checking decides whether the draft answers the question.",
			Metadata: map[string]any{"source_file": "notes.pdf", "block_type": "Text", "page_num": float64(1)}},
	})
	return s
}

func TestToolSpecsMatchTheSurface(t *testing.T) {
	specs := ToolSpecs()
	if len(specs) != 4 {
		t.Fatalf("specs = %d, want 4", len(specs))
	}
	names := ToolNames()
	for index, spec := range specs {
		if spec.Name != names[index] {
			t.Fatalf("spec %d = %q, want %q", index, spec.Name, names[index])
		}
		if strings.TrimSpace(spec.Description) == "" {
			t.Fatalf("%s has no description", spec.Name)
		}
		if spec.Parameters["type"] != "object" {
			t.Fatalf("%s parameters are not an object schema", spec.Name)
		}
	}
}

func TestHybridSearchTool(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	result, err := box.Execute(context.Background(), ToolCall{
		Name:      ToolHybridSearch,
		Arguments: map[string]any{"query": "sufficiency draft", "k": float64(2)},
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(result.Hits) == 0 {
		t.Fatal("expected hits")
	}
	if result.Hits[0].Chunk.ChunkID != "c3" {
		t.Fatalf("top hit = %s, want c3", result.Hits[0].Chunk.ChunkID)
	}
	if result.Hits[0].Score <= 0 {
		t.Fatalf("score = %f, want > 0", result.Hits[0].Score)
	}

	blank, err := box.Execute(context.Background(), ToolCall{Name: ToolHybridSearch, Arguments: map[string]any{"query": "   "}})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(blank.Hits) != 0 || !strings.Contains(blank.Note, "query is required") {
		t.Fatalf("blank query result = %#v", blank)
	}
}

func TestGrepSearchToolLiteral(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	result, err := box.Execute(context.Background(), ToolCall{Name: ToolGrepSearch, Arguments: map[string]any{"pattern": "GB/T 1234"}})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(result.Hits) != 1 || result.Hits[0].Chunk.ChunkID != "c1" {
		t.Fatalf("hits = %#v, want only c1", result.Hits)
	}
}

func TestGrepSearchToolFindsWhatKeywordSearchMisses(t *testing.T) {
	s := toolStore(t)
	box := &Toolbox{Store: s}

	// Terms() keeps only letters and digits, so a symbol-only pattern is
	// invisible to the keyword index but findable by a literal scan.
	if hits := s.Search("---", 5); len(hits) != 0 {
		t.Fatalf("keyword search unexpectedly found %d hit(s)", len(hits))
	}
	result, _ := box.Execute(context.Background(), ToolCall{Name: ToolGrepSearch, Arguments: map[string]any{"pattern": "---"}})
	if len(result.Hits) != 1 || result.Hits[0].Chunk.ChunkID != "c2" {
		t.Fatalf("grep hits = %#v, want only c2", result.Hits)
	}
	if !strings.Contains(result.Hits[0].Line, "---") {
		t.Fatalf("grep should report the matching line, got %q", result.Hits[0].Line)
	}
}

func TestGrepSearchToolRegexAndErrors(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	regex, err := box.Execute(context.Background(), ToolCall{
		Name:      ToolGrepSearch,
		Arguments: map[string]any{"pattern": `Qwen3-\d+B`, "regex": true},
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(regex.Hits) != 1 {
		t.Fatalf("regex hits = %d, want 1", len(regex.Hits))
	}

	// A bad pattern is the caller's mistake: reported, not fatal.
	bad, err := box.Execute(context.Background(), ToolCall{
		Name:      ToolGrepSearch,
		Arguments: map[string]any{"pattern": "([unclosed", "regex": true},
	})
	if err != nil {
		t.Fatalf("a bad regex must not fail the call: %v", err)
	}
	if len(bad.Hits) != 0 || !strings.Contains(bad.Note, "invalid regex") {
		t.Fatalf("bad regex result = %#v", bad)
	}
}

func TestListChunksTool(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	all, err := box.Execute(context.Background(), ToolCall{Name: ToolListChunks, Arguments: map[string]any{"limit": float64(10)}})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(all.Hits) != 4 {
		t.Fatalf("all hits = %d, want 4", len(all.Hits))
	}
	if !strings.Contains(all.Note, "report.pdf") || !strings.Contains(all.Note, "notes.pdf") {
		t.Fatalf("note should list the documents: %q", all.Note)
	}

	pageTwo, _ := box.Execute(context.Background(), ToolCall{
		Name:      ToolListChunks,
		Arguments: map[string]any{"doc_id": "report.pdf", "page": float64(2)},
	})
	if len(pageTwo.Hits) != 1 || pageTwo.Hits[0].Chunk.ChunkID != "c2" {
		t.Fatalf("page filter = %#v", pageTwo.Hits)
	}

	paged, _ := box.Execute(context.Background(), ToolCall{
		Name:      ToolListChunks,
		Arguments: map[string]any{"doc_id": "report.pdf", "offset": float64(1), "limit": float64(1)},
	})
	if len(paged.Hits) != 1 || paged.Hits[0].Chunk.ChunkID != "c1" {
		t.Fatalf("pagination = %#v", paged.Hits)
	}
}

func TestMetadataSearchTool(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	tables, err := box.Execute(context.Background(), ToolCall{
		Name:      ToolMetadataSearch,
		Arguments: map[string]any{"doc_id": "report.pdf", "block_type": "Table"},
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(tables.Hits) != 1 || tables.Hits[0].Chunk.ChunkID != "c2" {
		t.Fatalf("table filter = %#v", tables.Hits)
	}

	bySource, _ := box.Execute(context.Background(), ToolCall{
		Name:      ToolMetadataSearch,
		Arguments: map[string]any{"source_file": "notes.pdf"},
	})
	if len(bySource.Hits) != 1 || bySource.Hits[0].Chunk.ChunkID != "c3" {
		t.Fatalf("source_file filter = %#v", bySource.Hits)
	}

	// Structural matches carry no relevance score.
	if bySource.Hits[0].Score != 0 {
		t.Fatalf("score = %f, want 0 for a metadata match", bySource.Hits[0].Score)
	}
}

func TestToolboxRejectsUnknownToolAndMissingStore(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}
	if _, err := box.Execute(context.Background(), ToolCall{Name: "graph_explore"}); err == nil {
		t.Fatal("expected an error for a tool outside the surface")
	}
	if _, err := (&Toolbox{}).Execute(context.Background(), ToolCall{Name: ToolHybridSearch}); err == nil {
		t.Fatal("expected an error when no store is configured")
	}
}

func TestToolboxDefaultLimit(t *testing.T) {
	box := &Toolbox{Store: toolStore(t), DefaultLimit: 1}
	result, _ := box.Execute(context.Background(), ToolCall{Name: ToolListChunks, Arguments: map[string]any{}})
	if len(result.Hits) != 1 {
		t.Fatalf("hits = %d, want the default limit of 1", len(result.Hits))
	}
}
