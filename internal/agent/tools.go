package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"freerag/internal/embed"
	"freerag/internal/store"
)

// The four retrieval tools of docs/plan.md §6.7. They are complementary rather
// than overlapping: hybrid_search covers "means the same", grep_search covers
// "says exactly this", metadata_search covers "has these fields", and
// list_chunks covers "what is in this document/page".
const (
	ToolHybridSearch   = "hybrid_search"
	ToolGrepSearch     = "grep_search"
	ToolListChunks     = "list_chunks"
	ToolMetadataSearch = "metadata_search"
)

// ToolNames returns the tool surface in declaration order.
func ToolNames() []string {
	return []string{ToolHybridSearch, ToolGrepSearch, ToolListChunks, ToolMetadataSearch}
}

// KnownTool reports whether name is part of the tool surface.
//
// A model asked to choose tools can name anything at all. Checking here — before
// the call reaches the executor — is what lets the loop tell "the model picked
// nothing useful" apart from "the model picked something the kernel does not
// answer", and report the difference instead of silently retrieving less.
func KnownTool(name string) bool {
	for _, known := range ToolNames() {
		if name == known {
			return true
		}
	}
	return false
}

// ToolSpecs returns the four tool schemas (declaration order).
func ToolSpecs() []ToolSpec {
	return []ToolSpec{
		{
			Name: ToolHybridSearch,
			Description: "Semantic + keyword hybrid retrieval over the indexed chunks. " +
				"Use it first for most questions.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string", "description": "natural-language or keyword query"},
					"k":     map[string]any{"type": "integer", "description": "max results (default 6)"},
				},
				"required": []string{"query"},
			},
		},
		{
			Name: ToolGrepSearch,
			Description: "Exact literal or regular-expression match. " +
				"Use it for identifiers, codes and rare proper nouns that keyword search misses.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{"type": "string", "description": "literal text, or a regex when regex=true"},
					"regex":   map[string]any{"type": "boolean", "description": "treat pattern as a regular expression"},
					"k":       map[string]any{"type": "integer", "description": "max results (default 6)"},
				},
				"required": []string{"pattern"},
			},
		},
		{
			Name: ToolListChunks,
			Description: "List the chunks of one document in order, optionally for one page. " +
				"Use it to browse structure or to read a page back.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"doc_id": map[string]any{"type": "string", "description": "document id; omit to list across documents"},
					"page":   map[string]any{"type": "integer", "description": "restrict to one page number"},
					"offset": map[string]any{"type": "integer", "description": "skip this many matching chunks"},
					"limit":  map[string]any{"type": "integer", "description": "max results (default 6)"},
				},
			},
		},
		{
			Name: ToolMetadataSearch,
			Description: "Filter chunks by metadata only, with no text matching. " +
				"Use it for questions like \"which tables are on page 3\".",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"doc_id":      map[string]any{"type": "string", "description": "document id"},
					"source_file": map[string]any{"type": "string", "description": "source file name"},
					"block_type":  map[string]any{"type": "string", "description": "Title | Text | Table | Figure | Equation"},
					"page":        map[string]any{"type": "integer", "description": "page number"},
					"limit":       map[string]any{"type": "integer", "description": "max results (default 6)"},
				},
			},
		},
	}
}

// ToolResult is one tool execution's outcome.
type ToolResult struct {
	Tool string      `json:"tool"`
	Hits []store.Hit `json:"hits"`
	// Note describes the outcome (including empty ones), so a caller can tell
	// "nothing matched" from "the call was wrong".
	Note string `json:"note,omitempty"`
}

// Toolbox executes tool calls against the local index.
//
// It is synchronous and has no side effects beyond the optional embedder call,
// so the loop and the tests drive it the same way.
type Toolbox struct {
	Store *store.Store
	// Embedder powers the dense half of hybrid_search. When nil — or when a call
	// to it fails — hybrid_search degrades to BM25 rather than failing, because
	// a retrieval tool that returns nothing is worse than one that returns
	// keyword matches.
	Embedder embed.Embedder
	// DefaultLimit caps results when a call omits k/limit.
	DefaultLimit int
}

const defaultToolLimit = 6

func (t *Toolbox) limit(call ToolCall, key string) int {
	if value, ok := intArg(call.Arguments, key); ok && value > 0 {
		return value
	}
	if t.DefaultLimit > 0 {
		return t.DefaultLimit
	}
	return defaultToolLimit
}

// Execute runs one tool call.
//
// A malformed argument is reported as a Note rather than an error: the model can
// correct a bad regex or a missing field, but it cannot recover from the run
// failing outright.
func (t *Toolbox) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if t.Store == nil {
		return ToolResult{}, fmt.Errorf("toolbox: no store configured")
	}
	switch call.Name {
	case ToolHybridSearch:
		return t.hybridSearch(ctx, call), nil
	case ToolGrepSearch:
		return t.grepSearch(call)
	case ToolListChunks:
		return t.listChunks(call), nil
	case ToolMetadataSearch:
		return t.metadataSearch(call), nil
	default:
		return ToolResult{}, fmt.Errorf("toolbox: unknown tool %q", call.Name)
	}
}

// hybridSearch fuses keyword and dense retrieval.
//
// The query is embedded on every call rather than cached: queries are the
// cheapest thing to embed (one short string), and a cache would go stale
// silently the moment the embedder or the corpus changed.
func (t *Toolbox) hybridSearch(ctx context.Context, call ToolCall) ToolResult {
	query, _ := stringArg(call.Arguments, "query")
	query = strings.TrimSpace(query)
	if query == "" {
		return ToolResult{Tool: call.Name, Note: "query is required"}
	}
	limit := t.limit(call, "k")

	if t.Embedder == nil {
		hits := t.Store.Search(query, limit)
		return ToolResult{
			Tool: call.Name,
			Hits: hits,
			Note: fmt.Sprintf("%d hit(s) for %q (keyword only, no embedder)", len(hits), query),
		}
	}

	vector, err := embed.EmbedOne(ctx, t.Embedder, query)
	if err != nil {
		// Degrade rather than fail: a hosted embedder can be unreachable, and
		// returning keyword matches beats returning an error to the model.
		hits := t.Store.Search(query, limit)
		return ToolResult{
			Tool: call.Name,
			Hits: hits,
			Note: fmt.Sprintf("%d hit(s) for %q (keyword only; embed failed: %v)", len(hits), query, err),
		}
	}

	hits := t.Store.Hybrid(query, vector, limit)
	return ToolResult{
		Tool: call.Name,
		Hits: hits,
		Note: fmt.Sprintf("%d hit(s) for %q (hybrid, %s)", len(hits), query, t.Embedder.Name()),
	}
}

func (t *Toolbox) grepSearch(call ToolCall) (ToolResult, error) {
	pattern, _ := stringArg(call.Arguments, "pattern")
	if strings.TrimSpace(pattern) == "" {
		return ToolResult{Tool: call.Name, Note: "pattern is required"}, nil
	}
	useRegex, _ := boolArg(call.Arguments, "regex")

	found, err := t.Store.Grep(pattern, useRegex, t.limit(call, "k"))
	if err != nil {
		return ToolResult{Tool: call.Name, Note: err.Error()}, nil
	}

	hits := make([]store.Hit, 0, len(found))
	for _, hit := range found {
		// Occurrence count is the only ranking grep has; it is not comparable to
		// a BM25 score, which is why tool results carry their own tool name.
		// Line is kept so the caller sees WHERE it matched, not just that it did.
		hits = append(hits, store.Hit{Chunk: hit.Chunk, Score: float64(hit.Count), Line: hit.Line})
	}
	return ToolResult{
		Tool: call.Name,
		Hits: hits,
		Note: fmt.Sprintf("%d chunk(s) matching %q", len(hits), pattern),
	}, nil
}

func (t *Toolbox) listChunks(call ToolCall) ToolResult {
	docID, _ := stringArg(call.Arguments, "doc_id")
	page, _ := intArg(call.Arguments, "page")
	offset, _ := intArg(call.Arguments, "offset")

	hits := chunksToHits(t.Store.List(docID, page, offset, t.limit(call, "limit")))
	note := fmt.Sprintf("%d chunk(s)", len(hits))
	if docID == "" {
		if documents := t.Store.Documents(); len(documents) > 0 {
			note += "; documents: " + strings.Join(documents, ", ")
		}
	}
	return ToolResult{Tool: call.Name, Hits: hits, Note: note}
}

func (t *Toolbox) metadataSearch(call ToolCall) ToolResult {
	filter := store.MetadataFilter{}
	filter.DocID, _ = stringArg(call.Arguments, "doc_id")
	filter.SourceFile, _ = stringArg(call.Arguments, "source_file")
	filter.BlockType, _ = stringArg(call.Arguments, "block_type")
	filter.Page, _ = intArg(call.Arguments, "page")
	filter.Limit = t.limit(call, "limit")

	hits := chunksToHits(t.Store.MetadataSearch(filter))
	return ToolResult{
		Tool: call.Name,
		Hits: hits,
		Note: fmt.Sprintf("%d chunk(s) matched the filter", len(hits)),
	}
}

// chunksToHits wraps structural matches. They carry no relevance score — the
// caller asked for a shape, not a ranking — so Score stays 0.
func chunksToHits(chunks []store.Chunk) []store.Hit {
	hits := make([]store.Hit, 0, len(chunks))
	for _, chunk := range chunks {
		hits = append(hits, store.Hit{Chunk: chunk})
	}
	return hits
}

// ---- argument readers (JSON numbers arrive as float64) ----

func stringArg(args map[string]any, key string) (string, bool) {
	if args == nil {
		return "", false
	}
	value, ok := args[key]
	if !ok {
		return "", false
	}
	text, ok := value.(string)
	return text, ok
}

func intArg(args map[string]any, key string) (int, bool) {
	if args == nil {
		return 0, false
	}
	value, ok := args[key]
	if !ok {
		return 0, false
	}
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case float64:
		return int(typed), true
	case json.Number:
		if parsed, err := typed.Int64(); err == nil {
			return int(parsed), true
		}
	}
	return 0, false
}

func boolArg(args map[string]any, key string) (bool, bool) {
	if args == nil {
		return false, false
	}
	value, ok := args[key]
	if !ok {
		return false, false
	}
	typed, ok := value.(bool)
	return typed, ok
}
