package agent

import (
	"context"
	"strings"
	"testing"
	"time"

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
			Text:     "Sufficiency checking decides whether the answer answers the question.",
			Metadata: map[string]any{"source_file": "notes.pdf", "block_type": "Text", "page_num": float64(1)}},
	})
	return s
}

func TestToolSpecsMatchTheSurface(t *testing.T) {
	specs := ToolSpecs()
	if len(specs) != len(ToolNames()) {
		t.Fatalf("specs = %d, want one per tool (%d)", len(specs), len(ToolNames()))
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
		Arguments: map[string]any{"query": "sufficiency answer", "k": float64(2)},
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

// metadataFilterArgs builds the filters argument metadata_search takes.
func metadataFilterArgs(conditions ...map[string]any) map[string]any {
	items := make([]any, 0, len(conditions))
	for _, condition := range conditions {
		items = append(items, condition)
	}
	return map[string]any{"filters": items}
}

func metadataCondition(key, op string, value any) map[string]any {
	return map[string]any{"key": key, "op": op, "value": value}
}

func runMetadataSearch(t *testing.T, box *Toolbox, args map[string]any) ToolResult {
	t.Helper()
	result, err := box.Execute(context.Background(), ToolCall{Name: ToolMetadataSearch, Arguments: args})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	return result
}

func TestMetadataSearchFiltersByDocID(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	byDoc := runMetadataSearch(t, box, metadataFilterArgs(metadataCondition("doc_id", store.OpEqual, "notes.pdf")))
	if len(byDoc.Hits) != 1 || byDoc.Hits[0].Chunk.ChunkID != "c3" {
		t.Fatalf("doc_id filter = %#v", byDoc.Hits)
	}
	// Structural matches carry no relevance score.
	if byDoc.Hits[0].Score != 0 {
		t.Fatalf("score = %f, want 0 for a metadata match", byDoc.Hits[0].Score)
	}

	// A prefix, which is how a model asks about a document it half-remembers.
	prefixed := runMetadataSearch(t, box, metadataFilterArgs(metadataCondition("doc_id", store.OpStartWith, "rep")))
	if len(prefixed.Hits) != 3 {
		t.Fatalf("start with = %d hit(s), want the 3 chunks of report.pdf", len(prefixed.Hits))
	}

	// A list, for a model that wants two documents at once.
	listed := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition("doc_id", store.OpIn, []any{"notes.pdf", "report.pdf"})))
	if len(listed.Hits) != 4 {
		t.Fatalf("in = %d hit(s), want all 4", len(listed.Hits))
	}
}

// TestMetadataSearchTimeUsesStartWith pins the rule the tool description states,
// and which RAGFlow states for its own time fields: indexed_at holds a full
// timestamp, so a DAY is asked for with "start with" and the bare date. "="
// against a bare date can never match, and saying so in the schema is cheaper
// than a failed call the model reads as an empty corpus.
func TestMetadataSearchTimeUsesStartWith(t *testing.T) {
	s := toolStore(t)
	indexed := func(day int) time.Time {
		return time.Date(2026, 9, day, 14, 26, 1, 0, time.Local)
	}
	s.PutDocument(store.DocumentRecord{MD5: "m1", DocID: "report.pdf", IndexedAt: indexed(29)})
	s.PutDocument(store.DocumentRecord{MD5: "m2", DocID: "notes.pdf", IndexedAt: indexed(20)})
	box := &Toolbox{Store: s}

	byDay := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition("indexed_at", store.OpStartWith, "2026-09-29")))
	if len(byDay.Hits) != 3 {
		t.Fatalf("start with one day = %d hit(s), want report.pdf's 3 chunks", len(byDay.Hits))
	}

	// The negative half of the rule: equality with a bare date matches nothing,
	// because the stored value carries the time as well.
	equality := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition("indexed_at", store.OpEqual, "2026-09-29")))
	if len(equality.Hits) != 0 {
		t.Fatalf("= with a bare date = %d hit(s), want none — the stored value has a time", len(equality.Hits))
	}

	// The full timestamp is what "=" is for.
	exact := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition("indexed_at", store.OpEqual, "2026-09-29 14:26:01")))
	if len(exact.Hits) != 3 {
		t.Fatalf("= with the full timestamp = %d hit(s), want report.pdf's 3 chunks", len(exact.Hits))
	}
}

// TestMetadataSearchRejectsAnInventedField is the regression for the failure
// this surface was rebuilt to prevent: the model asked for a document by a
// fabricated key, the tool matched nothing, and the empty result was read — by
// the model and by the checker — as "the corpus has no such document".
//
// The fix is that the invented field is REFUSED, by name, with the fields that
// do exist listed alongside it.
func TestMetadataSearchRejectsAnInventedField(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	result := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition("author_zhao_hui", store.OpEqual, "Zhao Hui")))

	if len(result.Hits) != 0 {
		t.Fatalf("hits = %#v, want none", result.Hits)
	}
	for _, want := range []string{"author_zhao_hui", store.FieldDocID, store.FieldIndexedAt} {
		if !strings.Contains(result.Note, want) {
			t.Fatalf("note must name %q:\n%s", want, result.Note)
		}
	}
	// And it must not be mistakable for a filter that honestly matched nothing.
	if strings.Contains(result.Note, "matched the filter") {
		t.Fatalf("note reads as an empty corpus:\n%s", result.Note)
	}
}

func TestMetadataSearchOnZeroMatchesIsAStatementAboutTheFilter(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	result := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition("doc_id", store.OpEqual, "does-not-exist.pdf")))

	if len(result.Hits) != 0 {
		t.Fatalf("hits = %#v, want none", result.Hits)
	}
	// A real document id must be offered back, so the next attempt can use it.
	if !strings.Contains(result.Note, "report.pdf") {
		t.Fatalf("note must name the documents that do exist:\n%s", result.Note)
	}
	// The corpus is not empty, and the note may not imply that it is.
	if !strings.Contains(result.Note, "4 document(s)") && !strings.Contains(result.Note, "2 document(s)") {
		t.Fatalf("note must state the index is not empty:\n%s", result.Note)
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

// TestToolDescriptionsFollowThePortedTemplate pins the shape of the schemas
// ported from RAGFlow.
//
// RAGFlow's comment above its own specs is the reason this is a test and not a
// style preference: "The descriptions are the model's only guide for when to pick
// which tool, and rephrasing them changes routing behaviour." The section that
// matters most is IF IT FAILS — it is what tells a small model what a MISS means,
// so an exact-term probe that finds nothing in an enumeration is read as the
// answer rather than as a reason to rephrase.
func TestToolDescriptionsFollowThePortedTemplate(t *testing.T) {
	sections := []string{"WHEN TO CALL:", "DO NOT CALL:", "ARGUMENTS:", "OUTPUT:", "IF IT FAILS:"}

	specs := ToolSpecs()
	if len(specs) != len(ToolNames()) {
		t.Fatalf("%d spec(s) for %d tool(s)", len(specs), len(ToolNames()))
	}

	for _, spec := range specs {
		// The sections are read as a sequence — what this is for, when not to use
		// it, how to call it, what comes back, what a failure means — so they must
		// be present AND in order.
		last := -1
		for _, section := range sections {
			at := strings.Index(spec.Description, section)
			if at < 0 {
				t.Errorf("%s: the description has no %q section", spec.Name, section)
				continue
			}
			if at <= last {
				t.Errorf("%s: %q is out of order", spec.Name, section)
			}
			last = at
		}
		assertParametersDescribed(t, spec.Name, spec.Parameters)
	}
}

// assertParametersDescribed walks a JSON schema's properties, at every depth, and
// requires each one to say what it wants.
//
// A parameter with a type and no description is a parameter the model fills in by
// guessing, and a guessed doc_id or metadata value is indistinguishable from a
// real one until the search comes back empty.
func assertParametersDescribed(t *testing.T, tool string, schema map[string]any) {
	t.Helper()
	properties, _ := schema["properties"].(map[string]any)
	for name, raw := range properties {
		property, _ := raw.(map[string]any)
		if description, _ := property["description"].(string); strings.TrimSpace(description) == "" {
			t.Errorf("%s: parameter %q has no description", tool, name)
		}
		if nested, ok := property["items"].(map[string]any); ok {
			assertParametersDescribed(t, tool+"."+name+"[]", nested)
		}
		assertParametersDescribed(t, tool+"."+name, property)
	}
}
