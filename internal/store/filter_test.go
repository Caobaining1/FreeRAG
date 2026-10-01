package store

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestFilterZeroValueAllowsEverything(t *testing.T) {
	var filter Filter
	if filter.Active() {
		t.Fatal("the zero Filter restricts something")
	}
	if !filter.Allows("anything.pdf") {
		t.Fatal("the zero Filter refused a document")
	}
	if filter.Docs() != nil || filter.Len() != 0 {
		t.Fatalf("zero Filter has docs: %#v", filter.Docs())
	}
}

func TestNewFilterIgnoresEmptyIDsAndKeepsOrderStable(t *testing.T) {
	// An empty document id would otherwise become a key that matches nothing,
	// and an all-empty list would mean "scope to nothing" — which is not what an
	// absent filter means.
	if filter := NewFilter([]string{"", ""}); filter.Active() {
		t.Fatal("a filter of empty ids is active")
	}
	filter := NewFilter([]string{"c.pdf", "", "a.pdf", "b.pdf"})
	if got := filter.Docs(); !reflect.DeepEqual(got, []string{"a.pdf", "b.pdf", "c.pdf"}) {
		t.Fatalf("Docs = %#v, want sorted — these reach a vector database as a request body, "+
			"so the same scope must serialise the same way twice", got)
	}
	if filter.Len() != 3 {
		t.Fatalf("Len = %d, want 3", filter.Len())
	}
}

// weakStore is one document that matches a query weakly among twenty that match
// it strongly.
//
// The weak document is a real match — the word is there once, in a long passage
// — so it is ranked last rather than excluded. That is the case a post-filter
// gets wrong: the document is a legitimate answer, and a global top-k will not
// contain it.
func weakStore(t *testing.T) *Store {
	t.Helper()
	s := New()

	chunks := make([]Chunk, 0, 21)
	for i := 0; i < 20; i++ {
		chunks = append(chunks, Chunk{
			ChunkID: fmt.Sprintf("strong-%d", i),
			DocID:   "strong.pdf",
			PageNum: 1,
			Text:    "alpha alpha alpha alpha alpha",
		})
	}
	chunks = append(chunks, Chunk{
		ChunkID: "weak-0",
		DocID:   "weak.pdf",
		PageNum: 1,
		Text:    "alpha " + strings.Repeat("filler ", 400),
	})
	s.Add(chunks)
	return s
}

// The point of the whole mechanism: the limit is taken INSIDE the scope.
//
// This test fails if SearchIn is ever rewritten to filter a finished ranking
// instead of the candidates it ranks — which is the bug the type exists to
// prevent, and the reason RAGFlow passes its document ids down as the search's
// DocScope rather than sifting what comes back.
func TestSearchInTakesTheLimitInsideTheScope(t *testing.T) {
	s := weakStore(t)

	global := s.Search("alpha", 3)
	if len(global) != 3 {
		t.Fatalf("global search returned %d hit(s), want 3", len(global))
	}
	for _, hit := range global {
		if hit.Chunk.DocID == "weak.pdf" {
			t.Skip("the weak document reached the global top 3, so this fixture no longer " +
				"tells scoping apart from sifting")
		}
	}

	scoped := s.SearchIn("alpha", 3, NewFilter([]string{"weak.pdf"}))
	if len(scoped) != 1 || scoped[0].Chunk.DocID != "weak.pdf" {
		t.Fatalf("SearchIn = %#v; the only in-scope document matches the query and must be found, "+
			"which is only possible if the limit applies within the scope", scoped)
	}
}

// The dense half has the same hole and gets the same treatment. The nearest
// vector overall belongs to another document, so a scope applied after the
// search would return nothing at all.
func TestSearchVectorInRestrictsTheScan(t *testing.T) {
	s := New()
	if _, err := s.AddEmbedded([]Chunk{
		{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "one"},
		{ChunkID: "c1", DocID: "b.pdf", PageNum: 1, Text: "two"},
	}, [][]float32{{1, 0}, {0.9, 0.1}}); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}

	unscoped := s.SearchVector([]float32{1, 0}, 1)
	if len(unscoped) != 1 || unscoped[0].Chunk.DocID != "a.pdf" {
		t.Fatalf("unscoped SearchVector = %#v, want a.pdf (the nearest)", unscoped)
	}

	scoped := s.SearchVectorIn([]float32{1, 0}, 1, NewFilter([]string{"b.pdf"}))
	if len(scoped) != 1 || scoped[0].Chunk.DocID != "b.pdf" {
		t.Fatalf("SearchVectorIn = %#v, want the in-scope document even though it is not the "+
			"nearest overall", scoped)
	}
}

// Grep and the fusion both honour it as well: a scoped run must not be able to
// reach a document the scope excludes through a different tool.
func TestGrepInAndHybridInHonourTheFilter(t *testing.T) {
	s := New()
	s.Add([]Chunk{
		{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "the identifier GB/T 1234 appears here"},
		{ChunkID: "c1", DocID: "b.pdf", PageNum: 1, Text: "the identifier GB/T 1234 appears here too"},
	})

	all, err := s.Grep("GB/T", false, 10)
	if err != nil {
		t.Fatalf("Grep: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("unscoped grep found %d hit(s), want 2", len(all))
	}

	scoped, err := s.GrepIn("GB/T", false, 10, NewFilter([]string{"b.pdf"}))
	if err != nil {
		t.Fatalf("GrepIn: %v", err)
	}
	if len(scoped) != 1 || scoped[0].Chunk.DocID != "b.pdf" {
		t.Fatalf("GrepIn = %#v, want only the in-scope document", scoped)
	}

	fused := s.HybridIn("identifier", []float32{1, 0}, 10, NewFilter([]string{"b.pdf"}))
	for _, hit := range fused {
		if hit.Chunk.DocID != "b.pdf" {
			t.Fatalf("HybridIn returned a passage from %s; both legs must be scoped",
				hit.Chunk.DocID)
		}
	}
}
