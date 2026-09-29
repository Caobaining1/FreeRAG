package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"freerag/internal/store"
)

// fakeEmbedder stands in for the hosted model so these tests stay offline.
type fakeEmbedder struct {
	vectors [][]float32
	err     error
	calls   int
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = f.vectors[i%len(f.vectors)]
	}
	return out, nil
}

func (f *fakeEmbedder) Dimensions() int { return 2 }
func (f *fakeEmbedder) Name() string    { return "fake/2d" }

// hybridStore holds two chunks the two retrievers disagree about: "alpha" is the
// only keyword match, while the unrelated chunk is the closer vector.
func hybridStore(t *testing.T) *store.Store {
	t.Helper()
	s := store.New()
	_, err := s.AddEmbedded([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Text", Text: "unrelated words"},
		{ChunkID: "c1", DocID: "a.pdf", PageNum: 2, BlockType: "Text", Text: "alpha"},
	}, [][]float32{{1, 0.05}, {1, 1}})
	if err != nil {
		t.Fatalf("AddEmbedded: %v", err)
	}
	return s
}

func TestHybridSearchUsesBothRankers(t *testing.T) {
	box := &Toolbox{
		Store:    hybridStore(t),
		Embedder: &fakeEmbedder{vectors: [][]float32{{1, 0}}},
	}

	result, err := box.Execute(context.Background(), ToolCall{
		Name:      ToolHybridSearch,
		Arguments: map[string]any{"query": "alpha", "k": float64(5)},
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(result.Hits) != 2 {
		t.Fatalf("hits = %d, want 2 (one from each ranker)", len(result.Hits))
	}
	// The keyword match wins because both rankers place it near the top, while
	// the dense-only chunk is found by one.
	if result.Hits[0].Chunk.ChunkID != "c1" {
		t.Fatalf("top hit = %s, want c1", result.Hits[0].Chunk.ChunkID)
	}
	if len(result.Hits[0].Sources) != 2 {
		t.Fatalf("top hit sources = %v, want both rankers", result.Hits[0].Sources)
	}
	if !strings.Contains(result.Note, "hybrid") {
		t.Fatalf("note = %q, want it to name the mode", result.Note)
	}
}

func TestHybridSearchDegradesWhenEmbeddingFails(t *testing.T) {
	box := &Toolbox{
		Store:    hybridStore(t),
		Embedder: &fakeEmbedder{err: errors.New("endpoint unreachable")},
	}

	result, err := box.Execute(context.Background(), ToolCall{
		Name:      ToolHybridSearch,
		Arguments: map[string]any{"query": "alpha"},
	})
	if err != nil {
		t.Fatalf("a failing embedder must not fail the tool call: %v", err)
	}
	if len(result.Hits) != 1 || result.Hits[0].Chunk.ChunkID != "c1" {
		t.Fatalf("hits = %#v, want the keyword match only", result.Hits)
	}
	if !strings.Contains(result.Note, "keyword only") {
		t.Fatalf("note = %q, want it to report the degradation", result.Note)
	}
	if !strings.Contains(result.Note, "endpoint unreachable") {
		t.Fatalf("note = %q, want it to carry the cause", result.Note)
	}
}

func TestHybridSearchWithoutEmbedderStaysKeywordOnly(t *testing.T) {
	box := &Toolbox{Store: hybridStore(t)}

	result, err := box.Execute(context.Background(), ToolCall{
		Name:      ToolHybridSearch,
		Arguments: map[string]any{"query": "alpha"},
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(result.Hits) != 1 {
		t.Fatalf("hits = %d, want 1", len(result.Hits))
	}
	if !strings.Contains(result.Note, "no embedder") {
		t.Fatalf("note = %q, want it to say why dense retrieval is off", result.Note)
	}
}

func TestOtherToolsDoNotCallTheEmbedder(t *testing.T) {
	embedder := &fakeEmbedder{vectors: [][]float32{{1, 0}}}
	box := &Toolbox{Store: hybridStore(t), Embedder: embedder}

	for _, name := range []string{ToolGrepSearch, ToolListChunks, ToolMetadataSearch} {
		arguments := map[string]any{"pattern": "alpha", "limit": float64(5)}
		if _, err := box.Execute(context.Background(), ToolCall{Name: name, Arguments: arguments}); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	if embedder.calls != 0 {
		t.Fatalf("embedder called %d time(s) by non-hybrid tools", embedder.calls)
	}
}
