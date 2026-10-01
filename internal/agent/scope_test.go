package agent

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"freerag/internal/store"
)

// A metadata_search's document ids become the run's SCOPE, and the scope is
// what makes the filter outlive the call that produced it. RAGFlow leaves the
// ids in the tool response for the model to spend on a later call (see kbinfo's
// doc comment); here the loop holds them, so the narrowing applies whether or
// not the model mentions them again — and these are the tests that say so.

func TestKBInfoScopeLimitsThePool(t *testing.T) {
	info := newKBInfo()
	info.setScope([]string{"a.pdf"}, "doc_id = a.pdf")

	if added := info.add([]store.Hit{hit("a.pdf", "c0", "inside"), hit("b.pdf", "c1", "outside")}); added != 1 {
		t.Fatalf("added = %d, want 1 (only the in-scope passage)", added)
	}
	if info.outOfScope != 1 {
		t.Fatalf("outOfScope = %d, want 1 — a dropped passage has to be counted, or the filter "+
			"reads as a corpus with holes in it", info.outOfScope)
	}
	if info.len() != 1 || info.pool()[0].Chunk.DocID != "a.pdf" {
		t.Fatalf("pool = %#v", info.pool())
	}

	// Lifting the scope is the way out of a filter that turned out to be wrong.
	info.clearScope()
	if info.scoped() {
		t.Fatal("clearScope left the pool scoped")
	}
	if added := info.add([]store.Hit{hit("b.pdf", "c1", "outside")}); added != 1 {
		t.Fatalf("after clearing the scope, added = %d, want 1", added)
	}
}

func TestKBInfoWithoutAScopeAdmitsEveryDocument(t *testing.T) {
	info := newKBInfo()
	if added := info.add([]store.Hit{hit("a.pdf", "c0", "one"), hit("b.pdf", "c1", "two")}); added != 2 {
		t.Fatalf("added = %d, want 2 — an unfiltered run is unrestricted", added)
	}
	if info.outOfScope != 0 {
		t.Fatalf("outOfScope = %d, want 0", info.outOfScope)
	}
}

func TestKBInfoScopeIDsAreSorted(t *testing.T) {
	info := newKBInfo()
	info.setScope([]string{"c.pdf", "a.pdf", "b.pdf"}, "doc_id in (...)")

	if got := info.scopeIDs(); !reflect.DeepEqual(got, []string{"a.pdf", "b.pdf", "c.pdf"}) {
		t.Fatalf("scopeIDs = %#v, want sorted — a trace line or a test must not depend on map order", got)
	}
}

// The tool reports the scope and does not apply it: the toolbox does not know the
// pool, and the ids have to reach the kbinfo that every later call's results pass
// through.
func TestMetadataSearchReportsTheScopeItEstablishes(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	result := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition(store.FieldDocID, store.OpEqual, "notes.pdf")))

	if result.Scope == nil {
		t.Fatalf("metadata_search established no scope; note = %q", result.Note)
	}
	if !reflect.DeepEqual(result.Scope.DocIDs, []string{"notes.pdf"}) {
		t.Fatalf("scope = %#v, want the matched document", result.Scope.DocIDs)
	}
	if !strings.Contains(result.Note, "limited to") {
		t.Fatalf("the note does not tell the model the search is now scoped, so it will repeat the "+
			"filter every round: %q", result.Note)
	}
}

func TestMetadataSearchWithNoMatchSetsNoScope(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	result := runMetadataSearch(t, box, metadataFilterArgs(
		metadataCondition(store.FieldDocID, store.OpEqual, "does-not-exist.pdf")))

	if result.Scope != nil {
		t.Fatalf("a filter that matched nothing set a scope to %#v — that narrows the run to the "+
			"empty set and every later call would return nothing", result.Scope.DocIDs)
	}
	if !strings.Contains(result.Note, "No scope was set") {
		t.Fatalf("the note does not say retrieval is unchanged: %q", result.Note)
	}
}

func TestMetadataSearchClearLiftsTheScope(t *testing.T) {
	box := &Toolbox{Store: toolStore(t)}

	result := runMetadataSearch(t, box, map[string]any{"clear": true})

	if result.Scope == nil || !result.Scope.Clear {
		t.Fatalf("clear=true did not ask for the scope to be lifted: %#v", result.Scope)
	}
}

// The decisive one: after a metadata filter, a text search that matches both
// documents contributes only the filtered one. Without this the filter dies with
// the call that made it, which is the failure this whole mechanism exists to
// prevent.
func TestRunToolsLimitsLaterSearchesToTheScope(t *testing.T) {
	s := newStore(t,
		store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "alpha passage about alpha"},
		store.Chunk{ChunkID: "c1", DocID: "b.pdf", PageNum: 1, Text: "alpha passage about alpha"},
	)
	loop := &Loop{Store: s, Spec: Medium()}
	info := newKBInfo()
	var trace []string
	var attempts []attempt

	loop.runTools(context.Background(), []ToolCall{{Name: ToolMetadataSearch,
		Arguments: metadataFilterArgs(metadataCondition(store.FieldDocID, store.OpEqual, "a.pdf"))}},
		info, &trace, &attempts)
	if !info.scoped() {
		t.Fatalf("the metadata filter did not establish a scope; trace:\n%s", strings.Join(trace, "\n"))
	}

	// A text query with no mention of the scope. It matches both documents.
	loop.runTools(context.Background(), []ToolCall{{Name: ToolHybridSearch,
		Arguments: map[string]any{"query": "alpha"}}}, info, &trace, &attempts)

	for _, present := range info.pool() {
		if present.Chunk.DocID != "a.pdf" {
			t.Fatalf("a passage from %s entered the pool after the run was scoped to a.pdf; pool: %#v",
				present.Chunk.DocID, info.pool())
		}
	}

	// Nothing was dropped on the way in, and that is the assertion that tells
	// the two implementations apart: b.pdf never reached the pool because the
	// search never returned it, the scope having gone INTO the search rather
	// than onto its results. A non-zero counter would mean retrieval ran
	// unscoped and the pool sifted it — which loses in-scope passages whose
	// slots were taken by out-of-scope ones (see store.Filter).
	if info.outOfScope != 0 {
		t.Fatalf("%d passage(s) had to be dropped after retrieval, so the scope is being applied "+
			"to results instead of to candidates", info.outOfScope)
	}
	// And the tool says so, because a model that cannot see the scope concludes
	// the corpus is empty rather than that its own filter was narrow.
	if !strings.Contains(strings.Join(trace, "\n"), "within the scope of") {
		t.Fatalf("the tool note does not name the scope:\n%s", strings.Join(trace, "\n"))
	}
}
