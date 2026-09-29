package store

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// twoDocStore builds a store holding a.pdf and b.pdf, each with its own terms.
func twoDocStore(t *testing.T) *Store {
	t.Helper()
	s := New()
	if _, err := s.AddEmbedded(
		[]Chunk{
			{ChunkID: "c0", DocID: "a.pdf", Text: "alpha uniqueterm"},
			{ChunkID: "c1", DocID: "a.pdf", Text: "alpha second"},
			{ChunkID: "c0", DocID: "b.pdf", Text: "beta uniqueterm"},
		},
		[][]float32{{1, 0}, {0.9, 0.1}, {0, 1}},
	); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}
	return s
}

func TestRemoveDocLeavesNoTraceInTheTermIndex(t *testing.T) {
	s := twoDocStore(t)

	if removed := s.RemoveDoc("a.pdf"); removed != 2 {
		t.Fatalf("removed = %d, want 2", removed)
	}

	if s.Len() != 1 {
		t.Fatalf("len = %d, want 1", s.Len())
	}
	// df must have been decremented, or the removed document's terms would
	// still be counted in every other document's IDF.
	if hits := s.Search("alpha", 5); len(hits) != 0 {
		t.Fatalf("search for a removed term = %#v, want nothing", hits)
	}
	if hits := s.Search("second", 5); len(hits) != 0 {
		t.Fatalf("search for a removed term = %#v, want nothing", hits)
	}
	// The survivor must be untouched, including the term both documents shared.
	if hits := s.Search("uniqueterm", 5); len(hits) != 1 || hits[0].Chunk.DocID != "b.pdf" {
		t.Fatalf("survivor search = %#v", hits)
	}
}

func TestRemoveDocRebuildsTheDedupIndex(t *testing.T) {
	fake := &fakeDense{matches: []DenseMatch{{DocID: "b.pdf", ChunkID: "c0", Score: 1}}}
	s := twoDocStore(t)
	s.SetDenseIndex(fake)

	s.RemoveDoc("a.pdf")

	// chunk indices shifted when the first document was dropped. If the dedup
	// map was not rebuilt, this either resolves to the wrong chunk or to none.
	hits := s.SearchVector([]float32{0, 1}, 5)
	if len(hits) != 1 {
		t.Fatalf("hits = %#v, want the surviving chunk", hits)
	}
	if hits[0].Chunk.DocID != "b.pdf" || hits[0].Chunk.Text != "beta uniqueterm" {
		t.Fatalf("resolved the wrong chunk: %#v", hits[0])
	}
}

func TestRemoveDocTellsTheDenseIndexToDropItsVectors(t *testing.T) {
	fake := &fakeDense{}
	s := twoDocStore(t)
	s.SetDenseIndex(fake)

	s.RemoveDoc("a.pdf")

	// Otherwise the vectors linger in the index forever. Ranking would be
	// unaffected — resolve() drops matches the store does not own — but the
	// space never comes back.
	if len(fake.deleted) != 1 || fake.deleted[0] != "a.pdf" {
		t.Fatalf("deleted = %v, want [a.pdf]", fake.deleted)
	}
}

func TestRemoveDocSurvivesADenseDeleteFailure(t *testing.T) {
	fake := &fakeDense{deleteErr: errors.New("index down")}
	s := twoDocStore(t)
	s.SetDenseIndex(fake)

	// The chunks are gone locally, which is the part that affects correctness.
	if removed := s.RemoveDoc("a.pdf"); removed != 2 {
		t.Fatalf("removed = %d, want 2", removed)
	}
	if s.Len() != 1 {
		t.Fatalf("len = %d, want 1", s.Len())
	}
}

func TestRemoveDocOnAnUnknownDocumentIsANoOp(t *testing.T) {
	s := twoDocStore(t)

	if removed := s.RemoveDoc("missing.pdf"); removed != 0 {
		t.Fatalf("removed = %d, want 0", removed)
	}
	if s.Len() != 3 {
		t.Fatalf("len = %d, want 3", s.Len())
	}
}

func TestRemoveDocDropsItsManifestEntries(t *testing.T) {
	s := twoDocStore(t)
	s.PutDocument(DocumentRecord{MD5: "aaa", DocID: "a.pdf", ChunkCount: 2, Pipeline: "p"})
	s.PutDocument(DocumentRecord{MD5: "bbb", DocID: "b.pdf", ChunkCount: 1, Pipeline: "p"})

	s.RemoveDoc("a.pdf")

	if _, ok := s.Document("aaa"); ok {
		t.Fatal("the removed document's record must not survive")
	}
	if _, ok := s.Document("bbb"); !ok {
		t.Fatal("the surviving document's record must be kept")
	}
}

func TestDocumentLookups(t *testing.T) {
	s := New()
	s.PutDocument(DocumentRecord{MD5: "aaa", DocID: "a.pdf", Pipeline: "p1"})

	if record, ok := s.Document("aaa"); !ok || record.DocID != "a.pdf" {
		t.Fatalf("Document = %#v, %v", record, ok)
	}
	if _, ok := s.Document("nope"); ok {
		t.Fatal("an unknown fingerprint must not be found")
	}
	if record, ok := s.DocumentByDocID("a.pdf"); !ok || record.MD5 != "aaa" {
		t.Fatalf("DocumentByDocID = %#v, %v", record, ok)
	}
}

func TestDocumentByDocIDPrefersTheNewestAfterAnEdit(t *testing.T) {
	s := New()
	// A document that was indexed, then edited and indexed again: two
	// fingerprints, one DocID. The newest describes what the chunks now say.
	older := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	s.PutDocument(DocumentRecord{MD5: "old", DocID: "a.pdf", Pipeline: "p", IndexedAt: older})
	s.PutDocument(DocumentRecord{MD5: "new", DocID: "a.pdf", Pipeline: "p", IndexedAt: older.Add(time.Hour)})

	record, ok := s.DocumentByDocID("a.pdf")
	if !ok || record.MD5 != "new" {
		t.Fatalf("DocumentByDocID = %#v, want the newest", record)
	}
}

func TestChunkCountInCountsOneDocumentOnly(t *testing.T) {
	s := twoDocStore(t)

	if got := s.ChunkCountIn("a.pdf"); got != 2 {
		t.Fatalf("a.pdf = %d, want 2", got)
	}
	if got := s.ChunkCountIn("b.pdf"); got != 1 {
		t.Fatalf("b.pdf = %d, want 1", got)
	}
	if got := s.ChunkCountIn("missing.pdf"); got != 0 {
		t.Fatalf("missing = %d, want 0", got)
	}
	if got := s.ChunkCountIn(""); got != 0 {
		t.Fatalf("empty doc id = %d, want 0", got)
	}
}

func TestDocumentRecordsRoundTripThroughSaveAndLoad(t *testing.T) {
	s := twoDocStore(t)
	s.PutDocument(DocumentRecord{
		MD5: "aaa", DocID: "a.pdf", SourceFile: "a.pdf", Path: "/tmp/a.pdf",
		ChunkCount: 2, PageCount: 7, Pipeline: "parse-v1",
		IndexedAt: time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC),
	})

	path := filepath.Join(t.TempDir(), "index.json")
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// The manifest has to survive the round trip in the same file as the
	// chunks: a manifest that outlived its chunks would report documents as
	// indexed that the index no longer holds.
	record, ok := loaded.Document("aaa")
	if !ok {
		t.Fatal("the record did not survive the round trip")
	}
	if record.DocID != "a.pdf" || record.ChunkCount != 2 || record.Pipeline != "parse-v1" {
		t.Fatalf("record = %#v", record)
	}
	if !record.IndexedAt.Equal(time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC)) {
		t.Fatalf("indexed_at = %v", record.IndexedAt)
	}
}

func TestDocumentRecordsAreSorted(t *testing.T) {
	s := New()
	s.PutDocument(DocumentRecord{MD5: "zzz", DocID: "z.pdf"})
	s.PutDocument(DocumentRecord{MD5: "aaa", DocID: "a.pdf"})

	records := s.DocumentRecords()
	if len(records) != 2 || records[0].MD5 != "aaa" || records[1].MD5 != "zzz" {
		t.Fatalf("DocumentRecords = %#v, want sorted by md5", records)
	}
}

func TestSaveIsByteStableForAnUnchangedIndex(t *testing.T) {
	s := twoDocStore(t)
	// Deliberately inserted in the opposite order to the one they serialise in.
	s.PutDocument(DocumentRecord{MD5: "zzz", DocID: "z.pdf"})
	s.PutDocument(DocumentRecord{MD5: "aaa", DocID: "a.pdf"})

	path := filepath.Join(t.TempDir(), "index.json")
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	first, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	reloaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := reloaded.Save(path); err != nil {
		t.Fatalf("second Save: %v", err)
	}
	second, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	// Go randomises map iteration, so without the explicit sort this differs
	// run to run and the index file turns into noise in every diff.
	if string(first) != string(second) {
		t.Fatal("saving an unchanged index twice produced different files")
	}
}

func TestReAddingARemovedDocumentWorks(t *testing.T) {
	s := twoDocStore(t)
	s.RemoveDoc("a.pdf")

	// The dedup map was rebuilt on removal, so these keys are free again.
	added, err := s.AddEmbedded(
		[]Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "alpha replaced"}},
		[][]float32{{1, 0}},
	)
	if err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}
	if added != 1 {
		t.Fatalf("added = %d, want 1", added)
	}
	if hits := s.Search("replaced", 5); len(hits) != 1 || hits[0].Chunk.DocID != "a.pdf" {
		t.Fatalf("search = %#v", hits)
	}
}

// TestChangedDocumentDoesNotKeepItsOldText is the regression this whole
// mechanism exists for: chunk ids are positional, so re-indexing an edited
// document without removing the old chunks would leave the previous text in
// place under an unchanged id.
func TestChangedDocumentDoesNotKeepItsOldText(t *testing.T) {
	s := New()
	if _, err := s.AddEmbedded(
		[]Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "the first version says oldword"}},
		[][]float32{{1, 0}},
	); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}

	// Without the removal, this add would be deduplicated away as "c0 already
	// present" and the store would keep saying "oldword".
	s.RemoveDoc("a.pdf")
	if _, err := s.AddEmbedded(
		[]Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "the second version says newword"}},
		[][]float32{{1, 0}},
	); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}

	if hits := s.Search("oldword", 5); len(hits) != 0 {
		t.Fatalf("stale text is still indexed: %#v", hits)
	}
	if hits := s.Search("newword", 5); len(hits) != 1 {
		t.Fatalf("new text is not indexed: %#v", hits)
	}
}
