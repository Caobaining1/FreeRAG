package store

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func chunk(id, text string) Chunk {
	return Chunk{ChunkID: id, DocID: "doc", Text: text, PageNum: 1, BlockType: "Text"}
}

// fusionStore builds three chunks whose keyword and dense rankings deliberately
// disagree:
//
//	keyword : "alpha alpha" (1st), "alpha" (2nd), "unrelated" (absent)
//	dense   : A (1st), C (2nd), B (3rd)
//
// C is near the top of both, A leads only the dense list, B only the keyword
// list — which is exactly the case fusion exists to arbitrate.
func fusionStore(t *testing.T) *Store {
	t.Helper()
	s := New()

	chunks := []Chunk{
		chunk("c-a", "unrelated words"),
		chunk("c-b", "alpha"),
		chunk("c-c", "alpha alpha"),
	}
	vectors := [][]float32{
		{1, 0.05}, // most similar to the query
		{1, 1},    // least similar
		{1, 0.3},  // in between
	}
	if _, err := s.AddEmbedded(chunks, vectors); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}
	return s
}

func TestEncodeDecodeVectorRoundTrip(t *testing.T) {
	original := []float32{0, 1, -1, 0.5, 3.14159, -2.71828}
	decoded, err := decodeVector(encodeVector(original))
	if err != nil {
		t.Fatalf("decodeVector: %v", err)
	}
	if len(decoded) != len(original) {
		t.Fatalf("decoded %d values, want %d", len(decoded), len(original))
	}
	for i := range original {
		if decoded[i] != original[i] {
			t.Fatalf("value %d = %v, want %v", i, decoded[i], original[i])
		}
	}
}

func TestDecodeVectorRejectsBadInput(t *testing.T) {
	if _, err := decodeVector("not base64!!"); err == nil {
		t.Fatal("expected an error for invalid base64")
	}
	// 4 base64 characters decode to 3 bytes: not a whole number of float32s, so
	// accepting it would silently truncate the vector.
	if _, err := decodeVector("AAAA"); err == nil {
		t.Fatal("expected an error for a byte count that is not a multiple of 4")
	}
	if _, err := decodeVector("AA"); err == nil {
		t.Fatal("expected an error for invalid base64 padding")
	}

	// One float32 round-trips exactly.
	decoded, err := decodeVector(encodeVector([]float32{1.5}))
	if err != nil {
		t.Fatalf("decodeVector: %v", err)
	}
	if len(decoded) != 1 || decoded[0] != 1.5 {
		t.Fatalf("decoded = %v, want [1.5]", decoded)
	}
}

func TestCosineBasics(t *testing.T) {
	cases := []struct {
		name     string
		a, b     []float32
		expected float64
	}{
		{"identical", []float32{2, 0}, []float32{5, 0}, 1},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := cosine(tc.a, tc.b); math.Abs(got-tc.expected) > 1e-6 {
				t.Fatalf("cosine = %v, want %v", got, tc.expected)
			}
		})
	}
	if cosine([]float32{1, 0}, []float32{1, 0, 0}) != 0 {
		t.Fatal("mismatched widths must score 0, not compare prefixes")
	}
	if cosine([]float32{0, 0}, []float32{1, 0}) != 0 {
		t.Fatal("a zero vector must score 0 rather than divide by zero")
	}
}

func TestSearchVectorRanksBySimilarity(t *testing.T) {
	s := fusionStore(t)

	hits := s.SearchVector([]float32{1, 0}, 3)
	if len(hits) != 3 {
		t.Fatalf("hits = %d, want 3", len(hits))
	}
	want := []string{"c-a", "c-c", "c-b"}
	for i, id := range want {
		if hits[i].Chunk.ChunkID != id {
			t.Fatalf("rank %d = %s, want %s", i, hits[i].Chunk.ChunkID, id)
		}
	}
	if hits[0].Score <= hits[1].Score {
		t.Fatalf("scores are not descending: %v then %v", hits[0].Score, hits[1].Score)
	}
}

func TestSearchVectorSkipsUnembeddedChunksAndWrongWidth(t *testing.T) {
	s := New()
	if _, err := s.AddEmbedded([]Chunk{chunk("c-a", "alpha"), chunk("c-b", "beta")}, [][]float32{{1, 0}, nil}); err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}

	hits := s.SearchVector([]float32{1, 0}, 10)
	if len(hits) != 1 || hits[0].Chunk.ChunkID != "c-a" {
		t.Fatalf("hits = %#v, want only the embedded chunk", hits)
	}

	// A different width means a different model: ranking anyway would return
	// plausible nonsense, so it must return nothing.
	if got := s.SearchVector([]float32{1, 0, 0}, 10); got != nil {
		t.Fatalf("mismatched-width query returned %#v", got)
	}
	if got := s.SearchVector(nil, 10); got != nil {
		t.Fatalf("empty query vector returned %#v", got)
	}
}

func TestHybridPrefersAgreementAcrossRankers(t *testing.T) {
	s := fusionStore(t)

	hits := s.Hybrid("alpha", []float32{1, 0}, 3)
	if len(hits) != 3 {
		t.Fatalf("hits = %d, want 3", len(hits))
	}

	// C is top-2 in both lists, so it must beat A, which leads only the dense
	// list. This is the property that makes RRF worth using over either ranker.
	if hits[0].Chunk.ChunkID != "c-c" {
		t.Fatalf("top hit = %s, want c-c (found by both rankers)", hits[0].Chunk.ChunkID)
	}
	if hits[len(hits)-1].Chunk.ChunkID != "c-a" {
		t.Fatalf("last hit = %s, want c-a (dense only)", hits[len(hits)-1].Chunk.ChunkID)
	}

	// Sources must record the provenance of each result.
	byID := map[string]Hit{}
	for _, hit := range hits {
		byID[hit.Chunk.ChunkID] = hit
	}
	if got := byID["c-a"].Sources; len(got) != 1 || got[0] != "dense" {
		t.Fatalf("c-a sources = %v, want [dense]", got)
	}
	if got := byID["c-c"].Sources; len(got) != 2 {
		t.Fatalf("c-c sources = %v, want both rankers", got)
	}
}

func TestHybridScoreIsAlwaysAnRRFScore(t *testing.T) {
	s := fusionStore(t)

	hits := s.Hybrid("alpha", []float32{1, 0}, 3)
	// A top-ranked hit found by one ranker scores 1/(60+1); by two, 2/61.
	if got := hits[len(hits)-1].Score; math.Abs(got-1.0/(rrfK+1)) > 1e-9 {
		t.Fatalf("single-ranker score = %v, want 1/%d", got, rrfK+1)
	}
	for _, hit := range hits {
		if hit.Score > 2.0/(rrfK+1)+1e-9 {
			t.Fatalf("score %v exceeds the maximum possible fusion score", hit.Score)
		}
	}
}

func TestHybridDegradesToEachRankerAlone(t *testing.T) {
	s := fusionStore(t)

	// No query vector: the keyword half must still answer.
	keywordOnly := s.Hybrid("alpha", nil, 5)
	if len(keywordOnly) == 0 {
		t.Fatal("keyword-only fusion returned nothing")
	}
	if keywordOnly[0].Chunk.ChunkID != "c-c" {
		t.Fatalf("top keyword-only hit = %s, want c-c", keywordOnly[0].Chunk.ChunkID)
	}

	// A query the keyword index cannot match: the dense half must still answer.
	denseOnly := s.Hybrid("zzzzz", []float32{1, 0}, 5)
	if len(denseOnly) != 3 {
		t.Fatalf("dense-only fusion returned %d hits, want 3", len(denseOnly))
	}
	if denseOnly[0].Chunk.ChunkID != "c-a" {
		t.Fatalf("top dense-only hit = %s, want c-a", denseOnly[0].Chunk.ChunkID)
	}

	// Neither half can answer.
	if got := s.Hybrid("zzzzz", nil, 5); len(got) != 0 {
		t.Fatalf("fusion with no signal returned %#v", got)
	}
}

func TestHybridRespectsLimit(t *testing.T) {
	s := fusionStore(t)
	if got := s.Hybrid("alpha", []float32{1, 0}, 1); len(got) != 1 {
		t.Fatalf("limit 1 returned %d hits", len(got))
	}
	// A non-positive limit falls back to a sane default rather than returning
	// everything.
	if got := s.Hybrid("alpha", []float32{1, 0}, 0); len(got) != 3 {
		t.Fatalf("limit 0 returned %d hits, want the default", len(got))
	}
}

func TestAddEmbeddedRejectsMismatches(t *testing.T) {
	s := New()
	chunks := []Chunk{chunk("c-a", "alpha"), chunk("c-b", "beta")}

	if _, err := s.AddEmbedded(chunks, [][]float32{{1, 0}}); err == nil {
		t.Fatal("expected an error when the vector count does not match")
	}

	if _, err := s.AddEmbedded(chunks, [][]float32{{1, 0}, {1, 0, 0}}); err == nil {
		t.Fatal("expected an error for inconsistent vector widths")
	}
	if s.Len() != 0 {
		t.Fatalf("a rejected batch must not be partially indexed: %d chunk(s)", s.Len())
	}
}

func TestSetEmbedderRefusesWidthChange(t *testing.T) {
	s := fusionStore(t) // 2-dimensional vectors

	if err := s.SetEmbedder("test/2d", 2); err != nil {
		t.Fatalf("SetEmbedder with the same width: %v", err)
	}
	if got := s.EmbedderName(); got != "test/2d" {
		t.Fatalf("embedder = %q", got)
	}
	if err := s.SetEmbedder("test/3d", 3); err == nil {
		t.Fatal("expected an error: stored vectors would no longer be comparable")
	}
}

func TestStoreCountsAndDimensions(t *testing.T) {
	s := fusionStore(t)

	if got := s.Dimensions(); got != 2 {
		t.Fatalf("dimensions = %d, want 2", got)
	}
	if got := s.VectorCount(); got != 3 {
		t.Fatalf("vector count = %d, want 3", got)
	}
	if got := s.Len(); got != 3 {
		t.Fatalf("len = %d, want 3", got)
	}
}

func TestSaveLoadPreservesVectors(t *testing.T) {
	s := fusionStore(t)
	if err := s.SetEmbedder("test/2d", 2); err != nil {
		t.Fatalf("SetEmbedder: %v", err)
	}

	path := filepath.Join(t.TempDir(), "index.json")
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != 3 || loaded.VectorCount() != 3 {
		t.Fatalf("loaded %d chunks / %d vectors", loaded.Len(), loaded.VectorCount())
	}
	if loaded.Dimensions() != 2 || loaded.EmbedderName() != "test/2d" {
		t.Fatalf("embedding metadata lost: dims=%d embedder=%q", loaded.Dimensions(), loaded.EmbedderName())
	}

	// The round trip must preserve ranking, not just the count.
	before := s.SearchVector([]float32{1, 0}, 3)
	after := loaded.SearchVector([]float32{1, 0}, 3)
	for i := range before {
		if before[i].Chunk.ChunkID != after[i].Chunk.ChunkID {
			t.Fatalf("rank %d changed across Save/Load: %s -> %s", i, before[i].Chunk.ChunkID, after[i].Chunk.ChunkID)
		}
		if math.Abs(before[i].Score-after[i].Score) > 1e-6 {
			t.Fatalf("rank %d score changed: %v -> %v", i, before[i].Score, after[i].Score)
		}
	}
}

func TestLoadRejectsCorruptVector(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.json")
	raw := `{"dims":2,"chunks":[{"chunk_id":"c-a","text":"alpha","doc_id":"doc","vector":"@@@not-base64@@@"}]}`
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	if _, err := Load(path); err == nil {
		t.Fatal("expected an error rather than a silently unembedded chunk")
	}
}

func TestLoadIndexWithoutVectorsStillWorks(t *testing.T) {
	// An index written before embeddings existed must keep loading, with dense
	// retrieval simply unavailable.
	path := filepath.Join(t.TempDir(), "index.json")
	raw := `{"chunks":[{"chunk_id":"c-a","text":"alpha","doc_id":"doc"}]}`
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != 1 || loaded.VectorCount() != 0 || loaded.Dimensions() != 0 {
		t.Fatalf("len=%d vectors=%d dims=%d", loaded.Len(), loaded.VectorCount(), loaded.Dimensions())
	}
	if hits := loaded.Search("alpha", 5); len(hits) != 1 {
		t.Fatalf("keyword search on a vector-less index returned %d hits", len(hits))
	}
	if hits := loaded.SearchVector([]float32{1, 0}, 5); hits != nil {
		t.Fatalf("dense search on a vector-less index returned %#v", hits)
	}
}
