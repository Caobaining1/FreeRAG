package store

import (
	"errors"
	"testing"
)

// fakeDense records what it was asked to do, so the delegation contract can be
// asserted without a running vector database.
type fakeDense struct {
	upserted  []Chunk
	vectors   [][]float32
	deleted   []string
	matches   []DenseMatch
	upsertErr error
	searchErr error
	deleteErr error
	count     int
}

func (f *fakeDense) Backend() string { return "fake" }
func (f *fakeDense) Len() int        { return f.count }

func (f *fakeDense) Delete(docID string) error {
	if f.deleteErr != nil {
		return f.deleteErr
	}
	f.deleted = append(f.deleted, docID)
	return nil
}

func (f *fakeDense) Upsert(chunks []Chunk, vectors [][]float32) error {
	if f.upsertErr != nil {
		return f.upsertErr
	}
	f.upserted = append(f.upserted, chunks...)
	f.vectors = append(f.vectors, vectors...)
	f.count += len(chunks)
	return nil
}

func (f *fakeDense) Search(vector []float32, limit int) ([]DenseMatch, error) {
	if f.searchErr != nil {
		return nil, f.searchErr
	}
	if limit > 0 && len(f.matches) > limit {
		return f.matches[:limit], nil
	}
	return f.matches, nil
}

// denseStore builds a two-chunk store with vectors and a fake index attached.
func denseStore(t *testing.T, fake *fakeDense) *Store {
	t.Helper()
	s := New()
	s.SetDenseIndex(fake)
	if _, err := s.AddEmbedded(
		[]Chunk{
			{ChunkID: "c0", DocID: "a.pdf", Text: "alpha passage"},
			{ChunkID: "c1", DocID: "a.pdf", Text: "beta passage"},
		},
		[][]float32{{1, 0}, {0, 1}},
	); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}
	return s
}

func TestSearchVectorDelegatesToAttachedIndex(t *testing.T) {
	fake := &fakeDense{matches: []DenseMatch{{DocID: "a.pdf", ChunkID: "c1", Score: 0.9}}}
	s := denseStore(t, fake)

	hits := s.SearchVector([]float32{0, 1}, 5)

	if len(hits) != 1 {
		t.Fatalf("hits = %d, want 1", len(hits))
	}
	// The match names a chunk; the store supplies the chunk itself.
	if hits[0].Chunk.ChunkID != "c1" || hits[0].Chunk.Text != "beta passage" {
		t.Fatalf("hit = %#v", hits[0])
	}
	if hits[0].Score != 0.9 {
		t.Fatalf("score = %v, want the index's own score", hits[0].Score)
	}
	if len(hits[0].Sources) != 1 || hits[0].Sources[0] != "dense" {
		t.Fatalf("sources = %v", hits[0].Sources)
	}
}

func TestSearchVectorFallsBackWhenTheIndexFails(t *testing.T) {
	fake := &fakeDense{searchErr: errors.New("connection refused")}
	s := denseStore(t, fake)

	// A vector database that is down must degrade to the exact scan, not to an
	// empty result: "no hits" and "no index" are different answers.
	hits := s.SearchVector([]float32{1, 0}, 5)

	if len(hits) != 1 || hits[0].Chunk.ChunkID != "c0" {
		t.Fatalf("fallback hits = %#v, want the in-process nearest neighbour c0", hits)
	}
}

func TestSearchVectorIgnoresIndexResultsOnWidthMismatch(t *testing.T) {
	// The index would happily answer, but the query vector came from another
	// model, so ranking it would return plausible nonsense.
	fake := &fakeDense{matches: []DenseMatch{{DocID: "a.pdf", ChunkID: "c1", Score: 0.9}}}
	s := denseStore(t, fake)

	if hits := s.SearchVector([]float32{1, 0, 0}, 5); hits != nil {
		t.Fatalf("hits = %#v, want nil for a width mismatch", hits)
	}
}

func TestResolveDropsMatchesTheStoreDoesNotOwn(t *testing.T) {
	fake := &fakeDense{matches: []DenseMatch{
		{DocID: "a.pdf", ChunkID: "c0", Score: 0.9},
		{DocID: "a.pdf", ChunkID: "missing", Score: 0.8},
		{DocID: "other.pdf", ChunkID: "c0", Score: 0.7},
	}}
	s := denseStore(t, fake)

	hits := s.SearchVector([]float32{1, 0}, 5)

	// Only the match this store actually owns survives; the others mean the
	// index and the chunk store disagree, which must not invent a hit.
	if len(hits) != 1 || hits[0].Chunk.ChunkID != "c0" {
		t.Fatalf("hits = %#v, want only the owned chunk", hits)
	}
}

func TestAddEmbeddedForwardsOnlyNewEmbeddedChunks(t *testing.T) {
	fake := &fakeDense{}
	s := New()
	s.SetDenseIndex(fake)

	if _, err := s.AddEmbedded(
		[]Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"}},
		[][]float32{{1, 0}},
	); err != nil {
		t.Fatalf("first add: %v", err)
	}
	// A duplicate is not added again, so it must not be uploaded again.
	if _, err := s.AddEmbedded(
		[]Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"}},
		[][]float32{{1, 0}},
	); err != nil {
		t.Fatalf("duplicate add: %v", err)
	}
	// A chunk without a vector has nothing to upload.
	if _, err := s.AddEmbedded(
		[]Chunk{{ChunkID: "c1", DocID: "a.pdf", Text: "beta"}},
		nil,
	); err != nil {
		t.Fatalf("keyword-only add: %v", err)
	}

	if len(fake.upserted) != 1 || fake.upserted[0].ChunkID != "c0" {
		t.Fatalf("uploaded %#v, want only c0", fake.upserted)
	}
	if len(fake.vectors) != 1 || len(fake.vectors[0]) != 2 {
		t.Fatalf("uploaded vectors = %#v", fake.vectors)
	}
}

func TestAddEmbeddedSurvivesAUploadFailure(t *testing.T) {
	fake := &fakeDense{upsertErr: errors.New("index down")}
	s := New()
	s.SetDenseIndex(fake)

	added, err := s.AddEmbedded(
		[]Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"}},
		[][]float32{{1, 0}},
	)

	// The chunk is indexed for BM25 and grep regardless, so a failed upload is
	// "dense missed one chunk", not "the add failed".
	if err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}
	if added != 1 || s.Len() != 1 {
		t.Fatalf("added=%d len=%d, want 1/1", added, s.Len())
	}
	if hits := s.Search("alpha", 5); len(hits) != 1 {
		t.Fatalf("keyword search = %d hits, want 1", len(hits))
	}
}

func TestEmbeddedChunksSkipsUnembeddedOnes(t *testing.T) {
	s := New()
	if _, err := s.AddEmbedded(
		[]Chunk{
			{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"},
			{ChunkID: "c1", DocID: "a.pdf", Text: "beta"},
		},
		[][]float32{{1, 0}, nil},
	); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}

	chunks, vectors := s.EmbeddedChunks()

	// This is what re-seeds an index, so pairing the wrong vector with a chunk
	// here would poison the index rather than merely slow it down.
	if len(chunks) != 1 || len(vectors) != 1 {
		t.Fatalf("chunks=%d vectors=%d, want 1/1", len(chunks), len(vectors))
	}
	if chunks[0].ChunkID != "c0" || len(vectors[0]) != 2 {
		t.Fatalf("chunks=%#v vectors=%#v", chunks, vectors)
	}
}

func TestDenseBackendNamesTheAttachedIndex(t *testing.T) {
	s := New()
	if got := s.DenseBackend(); got != "" {
		t.Fatalf("DenseBackend = %q, want empty with no index attached", got)
	}

	s.SetDenseIndex(&fakeDense{})
	if got := s.DenseBackend(); got != "fake" {
		t.Fatalf("DenseBackend = %q, want fake", got)
	}
}

func TestHybridFusesDenseIndexHits(t *testing.T) {
	// A chunk only the dense leg can find must still reach the fused result,
	// which is the whole point of attaching an index.
	fake := &fakeDense{matches: []DenseMatch{{DocID: "a.pdf", ChunkID: "c1", Score: 0.9}}}
	s := denseStore(t, fake)

	hits := s.Hybrid("alpha", []float32{0, 1}, 5)

	byChunk := map[string][]string{}
	for _, hit := range hits {
		byChunk[hit.Chunk.ChunkID] = hit.Sources
	}
	if _, ok := byChunk["c1"]; !ok {
		t.Fatalf("hybrid dropped the dense-only chunk: %#v", hits)
	}
	if len(byChunk["c0"]) == 0 || byChunk["c0"][0] != "bm25" {
		t.Fatalf("bm25 sources = %v", byChunk["c0"])
	}
}
