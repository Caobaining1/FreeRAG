package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
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
//
// The descriptions follow RAGFlow's tool-schema template — WHEN TO CALL / DO NOT
// CALL / ARGUMENTS / OUTPUT / IF IT FAILS — ported from
// internal/rag/agentic-rag/runtime/action_session.go, whose comment is the reason
// the structure is copied rather than invented: "The descriptions are the model's
// only guide for when to pick which tool, and rephrasing them changes routing
// behaviour."
//
// Two things RAGFlow's wording does that a terse description cannot, and that the
// first version of these specs did neither of:
//
//   - it says what a MISS means, in the terms of the question being asked ("a miss
//     on an exact-term probe means the corpus lacks that term — in an enumeration
//     that is a RESULT"), so a small model stops rephrasing a probe that already
//     answered the question by finding nothing;
//   - it says which tool to use INSTEAD, so a dead end has a next step.
//
// Where RAGFlow's text describes behaviour this kernel does not have it is NOT
// copied — no `doc_scope` argument (the document scope is set by metadata_search
// and persists), no `[claim score=...]` lines (this index has no compiled claims),
// no navigate_* / graph_explore / web_search (the surface is the four tools of
// docs/plan.md §6.7) — because a description that promises behaviour the tool does
// not have is worse than a terse one: the model calls it correctly and blames the
// corpus for the result.
func ToolSpecs() []ToolSpec {
	return []ToolSpec{
		{
			Name: ToolHybridSearch,
			Description: "WHEN TO CALL: primary recall — the answer passage may share no surface words with " +
				"the question; the corpus is large and you do not know which document holds the answer; or an " +
				"exact-term probe returned nothing useful. Semantic recall fused with keyword matching.\n" +
				"DO NOT CALL: when you already hold a doc_id and want that whole document (list_chunks); when " +
				"you know the exact surface term and want literal matches (grep_search); when the question " +
				"names a document set by its metadata (metadata_search).\n" +
				"ARGUMENTS: query — ONE string of 3-12 words: the entity plus what is asked about it. Never a " +
				"sentence. Two rewordings of one thing are one query, not two. k — max passages, default 6.\n" +
				"OUTPUT: passages with doc_id, page and chunk id, added to the evidence pool; the next turn's " +
				"prompt carries a sample of up to 8 of them. The note says whether this search was hybrid or " +
				"keyword-only (a hosted embedder can be unreachable).\n" +
				"IF IT FAILS: \"0 hit(s)\" means this wording matched nothing — change the angle rather than " +
				"paraphrase it, because an identical repeat is refused as redundant and buys nothing.",
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
			Description: "WHEN TO CALL: you know or suspect an EXACT surface term — a name, a title, a code, " +
				"an identifier, a figure number, or a rare proper noun that keyword search blurs. Literal " +
				"verification of a specific string.\n" +
				"DO NOT CALL: for meaning-only questions (hybrid_search); for a whole document (list_chunks).\n" +
				"ARGUMENTS: pattern — the text to find, matched WITHOUT regard to case. regex — set true to " +
				"treat pattern as a Go regular expression, which is how ONE call probes alternatives " +
				"(`A|B|C` recalls all three) or a span between two terms (`A.*B`); an invalid expression " +
				"comes back as an error note. k — max passages, default 6.\n" +
				"OUTPUT: passages containing the pattern, ranked by how many times it occurs, each with " +
				"doc_id, page and the matching line.\n" +
				"IF IT FAILS: a miss on an exact-term probe means the corpus does not contain that term — in an " +
				"enumeration that is a RESULT, not a failure: probe the next name instead of rephrasing this " +
				"one. A miss on a guessed spelling says nothing about a differently spelled name.",
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
			Description: "WHEN TO CALL: you need the FULL text of one document — enumeration, counting or " +
				"arithmetic over several of its passages — and you already have its doc_id from a prior tool " +
				"result.\n" +
				"DO NOT CALL: when one passage would do (hybrid_search or grep_search); when you have no " +
				"doc_id yet — locate one first.\n" +
				"ARGUMENTS: doc_id — the document to read; omit it to list across the documents in scope. " +
				"page — restrict to one page. offset and limit — page through a long document. These are the " +
				"only arguments that exist; there is no chunk_ids argument.\n" +
				"OUTPUT: that document's chunks in reading order, with page numbers, added to the evidence pool.\n" +
				"IF IT FAILS: an unknown doc_id yields nothing, and that is a query-level miss rather than a " +
				"fact about the corpus — locate the document first. A doc_id outside the current scope is " +
				"refused with a note naming the scope; lift the scope with metadata_search clear=true instead " +
				"of retrying.",
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
			Description: "WHEN TO CALL: SELECT the document set by METADATA before searching — when the " +
				"question names a time (\"which documents were added today\"), names one concrete document, or " +
				"needs a named subset. This tool matches NO text: it selects. Only two fields exist — doc_id " +
				"(the file name) and indexed_at ('YYYY-MM-DD HH:MM:SS').\n" +
				"CALL AT MOST ONCE PER DIRECTION, then spend what it returned: it already returns the matched " +
				"documents' chunks, and it limits the rest of the run to those documents.\n" +
				"DO NOT CALL: when nothing names a document or a subset (use hybrid_search); when you already " +
				"hold a doc_id (use list_chunks); when you are counting or enumerating members.\n" +
				"ARGUMENTS: filters — [{key, value, op}] over the AVAILABLE METADATA fields listed in this " +
				"prompt, and no others; op — see enum; logic — 'and' or 'or', default and. String ops take " +
				"ONE keyword, so call once per keyword; 'in' takes a list; 'empty' takes no value. TIME RULE: " +
				"indexed_at stores a full timestamp, so filter ONE day with op 'start with' and the bare " +
				"'YYYY-MM-DD' — '=' never matches a stored time. Copy a value exactly as the AVAILABLE " +
				"METADATA block stores it. clear=true lifts the scope (pass it without filters). limit — max " +
				"chunks to return, default 6.\n" +
				"OUTPUT: the matched documents' chunks, added to the evidence pool, plus the document scope " +
				"those documents establish — later hybrid_search / grep_search / list_chunks calls are limited " +
				"to them without repeating the filter, and a later call REPLACES the scope.\n" +
				"IF IT FAILS: \"0 chunk(s) matched\" is a statement about the FILTER, not about the corpus — " +
				"shorten the substring, search the text instead of the metadata, or lift the scope. Do not " +
				"retry the same filter.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"clear": map[string]any{
						"type":        "boolean",
						"description": "true lifts the document scope an earlier call set; omit filters when using it",
					},
					"filters": map[string]any{
						"type":     "array",
						"minItems": 1,
						"description": "the conditions selecting documents; one entry per field, combined by " +
							"logic. RAGFlow leaves this one undescribed and the model still invents keys — the " +
							"enum on key and op is what constrains it.",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"key": map[string]any{
									"type":        "string",
									"enum":        store.MetadataFieldNames(),
									"description": "doc_id or indexed_at",
								},
								"op": map[string]any{
									"type":        "string",
									"enum":        store.MetadataOperators(),
									"description": "comparison; 'start with' for a day of indexed_at",
								},
								"value": map[string]any{
									"type": []any{"string", "array"},
									"description": "the value to match. Copy it exactly as listed under AVAILABLE METADATA — " +
										"a list only for 'in' / 'not in'; omit for 'empty' / 'not empty'.",
								},
							},
							"required": []string{"key", "op"},
						},
					},
					"logic": map[string]any{
						"type":        "string",
						"enum":        []string{"and", "or"},
						"description": "how several filters combine. Default and.",
					},
					"limit": map[string]any{"type": "integer", "description": "max chunks (default 6)"},
				},
				"required": []string{"filters"},
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
	// Scope, when set, is a document scope this call established for the rest of
	// the run. The toolbox does not apply it — it does not know the pool — it
	// reports it, and the loop hands it to the kbinfo that every later call's
	// results pass through (see kbinfo.setScope).
	Scope *ScopeUpdate `json:"scope,omitempty"`
}

// ScopeUpdate is a document scope a tool call asks the loop to adopt.
type ScopeUpdate struct {
	// DocIDs are the documents the run may draw from from now on.
	DocIDs []string
	// Clear asks for the scope to be lifted, which is the only way out of a
	// filter that turned out to be wrong.
	Clear bool
	// Note is the filter in the words the caller used, for the trace and the
	// answer prompt.
	Note string
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
	// filter is the document scope retrieval runs under, set per execution by
	// ExecuteScoped. It is pushed INTO the store's search rather than applied to
	// its results (see store.Filter), which is why it has to reach this far down
	// instead of being sifted at the kbinfo.
	filter store.Filter
}

// Execute runs one tool call with no document scope.
func (t *Toolbox) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	return t.ExecuteScoped(ctx, call, store.Filter{})
}

// ExecuteScoped runs one tool call with retrieval restricted to filter.
//
// metadata_search is deliberately NOT scoped: it is what defines the scope, so
// applying the current one to it would make a filter impossible to change. The
// copy is what keeps that exception honest — the handler reads t.filter like
// every other, and this is the call that leaves it empty.
func (t *Toolbox) ExecuteScoped(ctx context.Context, call ToolCall, filter store.Filter) (ToolResult, error) {
	scoped := *t
	scoped.filter = filter
	if call.Name == ToolMetadataSearch {
		scoped.filter = store.Filter{}
	}
	return scoped.run(ctx, call)
}

// scopeSuffix names the active scope in a tool's note, or is empty without one.
//
// The model has to see this: "0 hit(s) for a query" inside a filter is a fact
// about the filter, and a model that does not know a scope is in force will
// conclude the corpus is empty rather than that its own filter was too narrow.
func (t *Toolbox) scopeSuffix() string {
	if !t.filter.Active() {
		return ""
	}
	return fmt.Sprintf(", within the scope of %d document(s)", t.filter.Len())
}

// run dispatches one call on a toolbox whose filter is already set.
//
// A malformed argument is reported as a Note rather than an error: the model can
// correct a bad regex or a missing field, but it cannot recover from the run
// failing outright.
func (t *Toolbox) run(ctx context.Context, call ToolCall) (ToolResult, error) {
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
		hits := t.Store.SearchIn(query, limit, t.filter)
		return ToolResult{
			Tool: call.Name,
			Hits: hits,
			Note: fmt.Sprintf("%d hit(s) for %q (keyword only, no embedder%s)",
				len(hits), query, t.scopeSuffix()),
		}
	}

	vector, err := embed.EmbedOne(ctx, t.Embedder, query)
	if err != nil {
		// Degrade rather than fail: a hosted embedder can be unreachable, and
		// returning keyword matches beats returning an error to the model.
		hits := t.Store.SearchIn(query, limit, t.filter)
		return ToolResult{
			Tool: call.Name,
			Hits: hits,
			Note: fmt.Sprintf("%d hit(s) for %q (keyword only; embed failed: %v%s)",
				len(hits), query, err, t.scopeSuffix()),
		}
	}

	hits := t.Store.HybridIn(query, vector, limit, t.filter)
	return ToolResult{
		Tool: call.Name,
		Hits: hits,
		Note: fmt.Sprintf("%d hit(s) for %q (hybrid, %s%s)", len(hits), query, t.Embedder.Name(), t.scopeSuffix()),
	}
}

func (t *Toolbox) grepSearch(call ToolCall) (ToolResult, error) {
	pattern, _ := stringArg(call.Arguments, "pattern")
	if strings.TrimSpace(pattern) == "" {
		return ToolResult{Tool: call.Name, Note: "pattern is required"}, nil
	}
	useRegex, _ := boolArg(call.Arguments, "regex")

	found, err := t.Store.GrepIn(pattern, useRegex, t.limit(call, "k"), t.filter)
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
		Note: fmt.Sprintf("%d chunk(s) matching %q%s", len(hits), pattern, t.scopeSuffix()),
	}, nil
}

func (t *Toolbox) listChunks(call ToolCall) ToolResult {
	docID, _ := stringArg(call.Arguments, "doc_id")
	page, _ := intArg(call.Arguments, "page")
	offset, _ := intArg(call.Arguments, "offset")

	// Reading a document the scope excludes is refused rather than silently
	// returning nothing: the model asked a question that has an answer, and
	// "0 chunk(s)" would let it conclude the document is empty.
	if docID != "" && !t.filter.Allows(docID) {
		return ToolResult{Tool: call.Name, Note: fmt.Sprintf(
			"%s is outside the current document scope (%d document(s)); either list a document inside "+
				"it, or lift the scope with metadata_search clear=true.", docID, t.filter.Len())}
	}

	hits := chunksToHits(t.Store.List(docID, page, offset, t.limit(call, "limit")))
	note := fmt.Sprintf("%d chunk(s)", len(hits))
	if docID == "" {
		if documents := t.Store.Documents(); len(documents) > 0 {
			note += "; documents: " + strings.Join(documents, ", ")
		}
	}
	return ToolResult{Tool: call.Name, Hits: hits, Note: note}
}

// maxMetadataHintValues caps how many sample values a hint lists, so a large
// corpus cannot turn a diagnostic into a payload.
const maxMetadataHintValues = 12

// metadataSearch selects chunks by the two metadata fields the index has.
func (t *Toolbox) metadataSearch(call ToolCall) ToolResult {
	// Lifting the scope is its own call because there is otherwise no way out of
	// a filter that was wrong: without it, one bad filter narrows every
	// remaining round of the run.
	if clear, _ := boolArg(call.Arguments, "clear"); clear {
		return ToolResult{
			Tool:  call.Name,
			Scope: &ScopeUpdate{Clear: true, Note: "the document scope was lifted"},
			Note:  "document scope lifted: retrieval is unrestricted again.",
		}
	}

	query, problem := t.metadataQuery(call.Arguments)
	if problem != "" {
		return ToolResult{Tool: call.Name, Note: problem}
	}

	// The document ids are computed once and kept: they are the scope for the
	// rest of the run, not a statistic for the note (see kbinfo's doc comment
	// for why the ids are held by the loop rather than left to the model).
	conditions := renderMetadataConditions(query)
	docIDs := t.Store.MetadataDocIDs(query)
	hits := chunksToHits(t.Store.MetadataSearchChunks(query, t.limit(call, "limit")))
	if len(docIDs) > 0 {
		return ToolResult{
			Tool:  call.Name,
			Hits:  hits,
			Scope: &ScopeUpdate{DocIDs: docIDs, Note: fmt.Sprintf("%s selected %s", conditions, summariseDocIDs(docIDs))},
			Note: fmt.Sprintf(
				"%d chunk(s) from %d document(s) matched %s. The run is now limited to %s — later searches "+
					"are scoped to them automatically, so do not repeat this filter; pass clear=true to lift it.",
				len(hits), len(docIDs), conditions, summariseDocIDs(docIDs)),
		}
	}

	// Zero hits. The note says what that is a statement ABOUT.
	//
	// "0 chunk(s) matched the filter" was true and useless: it reads as "the
	// corpus has nothing", when it could equally mean the filter named a
	// document that does not exist. Those two lead to opposite answers, and
	// collapsing them is how a fabricated filter became the sentence "there are
	// no papers by this author". Naming the values that DO exist is what makes
	// the next attempt a retry rather than a conclusion.
	//
	// No scope is set on this path: a filter that matched nothing must not
	// narrow the run to nothing, and an empty document set is not a scope the
	// loop can act on — it would make every later call return nothing.
	return ToolResult{
		Tool: call.Name,
		Note: fmt.Sprintf(
			"no chunk matched %s. This is a statement about the FILTER, not the corpus — %s. "+
				"No scope was set, so retrieval is unchanged. Change or loosen the filter, or use "+
				"hybrid_search for a text query.",
			conditions, t.renderMetadataAvailability()),
	}
}

// summariseDocIDs renders a document set for a note or a prompt, capped the way
// RAGFlow caps its metadata context (metadataContextDocsMax = 20 documents and
// 200 runes per value): a note naming 400 documents is a note nobody reads.
func summariseDocIDs(docIDs []string) string {
	const shown = 8
	sorted := append([]string(nil), docIDs...)
	sort.Strings(sorted)
	parts := make([]string, 0, shown+1)
	for i, id := range sorted {
		if i == shown {
			parts = append(parts, fmt.Sprintf("…(+%d more)", len(sorted)-shown))
			break
		}
		parts = append(parts, id)
	}
	return strings.Join(parts, ", ")
}

// metadataQuery reads the filters argument, returning a problem string when the
// call cannot be run as written.
//
// A bad field or operator is REPORTED, never filtered on. Validating here is
// what lets the caller tell "the model named something that does not exist"
// apart from "nothing matched", and it is the difference RAGFlow draws with its
// key check before its executor runs (see its metadata_search: unknown keys
// return `Metadata key(s) ... do not exist in this dataset. Available: ...`).
func (t *Toolbox) metadataQuery(args map[string]any) (store.MetadataQuery, string) {
	raw, ok := args["filters"].([]any)
	if !ok || len(raw) == 0 {
		return store.MetadataQuery{}, fmt.Sprintf(
			"filters is required: a list of {key, op, value} over %s. Available: %s",
			strings.Join(store.MetadataFieldNames(), " and "), t.renderMetadataAvailability())
	}

	query := store.MetadataQuery{}
	if logic, ok := stringArg(args, "logic"); ok {
		query.Logic = logic
	}

	var missingKey, unknownKey, unknownOp bool
	var bad []string
	for _, item := range raw {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		key, _ := stringArg(entry, "key")
		op, _ := stringArg(entry, "op")

		switch {
		case key == "":
			missingKey = true
			continue
		case !store.KnownMetadataField(key):
			unknownKey = true
			bad = append(bad, key)
			continue
		case !store.KnownMetadataOperator(op):
			unknownOp = true
			bad = append(bad, fmt.Sprintf("%s %s", key, op))
			continue
		}

		query.Conditions = append(query.Conditions, store.MetadataCondition{
			Key:   key,
			Op:    op,
			Value: entry["value"],
		})
	}

	switch {
	case missingKey:
		return store.MetadataQuery{}, fmt.Sprintf(
			"every filter needs a key. Available: %s", t.renderMetadataAvailability())
	case unknownKey:
		// Naming the two fields is the whole fix for a model that reached for a
		// field this index does not have: it can only be told, not inferred.
		return store.MetadataQuery{}, fmt.Sprintf(
			"metadata field(s) %v do not exist in this index. It has exactly two: %s. "+
				"Available: %s. Use hybrid_search or grep_search for anything else.",
			bad, strings.Join(store.MetadataFieldNames(), ", "), t.renderMetadataAvailability())
	case unknownOp:
		return store.MetadataQuery{}, fmt.Sprintf(
			"unknown operator in %v. Available operators: %s. For one day of indexed_at use "+
				"'start with' and the bare 'YYYY-MM-DD'.",
			bad, strings.Join(store.MetadataOperators(), ", "))
	case len(query.Conditions) == 0:
		return store.MetadataQuery{}, fmt.Sprintf(
			"no readable filter was given. Available: %s", t.renderMetadataAvailability())
	}
	return query, ""
}

// renderMetadataConditions renders a query for a trace or a note.
func renderMetadataConditions(query store.MetadataQuery) string {
	if len(query.Conditions) == 0 {
		return "the filter"
	}
	logic := query.Logic
	if logic != "or" {
		logic = "and"
	}
	parts := make([]string, 0, len(query.Conditions))
	for _, condition := range query.Conditions {
		parts = append(parts, fmt.Sprintf("%s %s %v", condition.Key, condition.Op, condition.Value))
	}
	return strings.Join(parts, " "+logic+" ")
}

// renderMetadataAvailability names the two fields with the values that actually
// exist, which is the statistic a failed filter needs to become a retry.
func (t *Toolbox) renderMetadataAvailability() string {
	samples := t.Store.MetadataFieldSamples(maxMetadataHintValues)

	var b strings.Builder
	for _, field := range store.MetadataFieldNames() {
		if b.Len() > 0 {
			b.WriteString("; ")
		}
		b.WriteString(field)
		values := samples[field]
		if len(values) == 0 {
			b.WriteString(" has no values")
			continue
		}
		b.WriteString(" = ")
		for i, sample := range values {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%q (%d doc(s))", sample.Value, sample.Docs)
		}
	}
	fmt.Fprintf(&b, "; %d document(s) indexed", len(t.Store.Documents()))
	return b.String()
}

// renderMetadataCatalog renders the two filterable fields with the values that
// actually exist, for the tool-planning prompt.
//
// This is the half a schema cannot carry. The `key` enum tells the model which
// FIELDS exist; only the values tell it which VALUES do — and the failure this
// addresses was a filter value invented whole (`author_zhao_hui` passed as a
// doc_id), which an enum of field names does nothing about. RAGFlow reaches the
// same conclusion from the other side: its catalog carries both the declared
// field list and "the values the doc-metadata index actually carries, with
// document counts", and its seed block shows them beside each field.
//
// The closing line is RAGFlow's and is kept because it names the prohibition
// outright rather than leaving it implied. Its wording — "Fields NOT listed here
// are unusable as filters — never invent one." — is a direct response to the
// same failure mode.
func renderMetadataCatalog(s *store.Store, perField int) string {
	if s == nil {
		return ""
	}
	documents := s.Documents()
	if len(documents) == 0 {
		// An empty index advertises nothing: there is no value to copy, and a
		// field list with no values under it invites exactly the invented value
		// this block exists to prevent.
		return ""
	}

	samples := s.MetadataFieldSamples(perField)

	var b strings.Builder
	b.WriteString("AVAILABLE METADATA (the only fields metadata_search can filter on):\n")
	for _, field := range store.MetadataFieldNames() {
		b.WriteString("- ")
		b.WriteString(field)
		if field == store.FieldIndexedAt {
			fmt.Fprintf(&b, " (stored as %q, so ask for one day with op 'start with' and the bare YYYY-MM-DD)",
				store.IndexedAtFormat)
		}
		values := samples[field]
		if len(values) == 0 {
			b.WriteString("; no values recorded")
			b.WriteString("\n")
			continue
		}
		b.WriteString("; values: ")
		for i, sample := range values {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%q (%d doc(s))", sample.Value, sample.Docs)
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "%d document(s) are indexed. Values NOT listed here do not exist — never invent one.",
		len(documents))
	return b.String()
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
