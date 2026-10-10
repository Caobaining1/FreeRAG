// Package engine is the FreeRAG shim mirroring ragflow/internal/engine. It
// defines the DocEngine interface used by the migrated agentic_rag BM25/grep/kg
// service adapters and the navigate_* tools, plus a store-backed implementation
// (no Elasticsearch in FreeRAG).
package engine

import (
	"context"
	"sync"

	"freerag/internal/embed"
	"freerag/internal/engine/types"
	"freerag/internal/store"
)

// DocEngine is the document retrieval engine abstraction. The RAGFlow
// implementation is Elasticsearch-backed; the FreeRAG implementation is the
// in-process store (BM25 + dense).
type DocEngine interface {
	Search(ctx context.Context, req *types.SearchRequest) (*types.SearchResult, error)
}

// regexpSearchable is the optional capability asserted by the grep adapter; the
// store engine implements it.
type regexpSearchable interface {
	SearchByRegexp(ctx context.Context, req *types.RegexpSearchRequest) (*types.SearchResult, error)
}

var (
	mu   sync.RWMutex
	de   DocEngine
	emb  embed.Embedder
)

// SetDocEngine installs the store-backed (or test) DocEngine.
func SetDocEngine(e DocEngine) {
	mu.Lock()
	defer mu.Unlock()
	de = e
}

// SetEmbedder wires the embedding backend used by dense/hybrid search.
func SetEmbedder(e embed.Embedder) {
	mu.Lock()
	defer mu.Unlock()
	emb = e
}

// Get returns the installed DocEngine (nil until SetDocEngine is called during
// bootstrap).
func Get() DocEngine {
	mu.RLock()
	defer mu.RUnlock()
	return de
}

// NewStoreDocEngine builds a store-backed DocEngine over the given store.
func NewStoreDocEngine(s *store.Store, e embed.Embedder) DocEngine {
	return &storeDocEngine{store: s, emb: e}
}

type storeDocEngine struct {
	store *store.Store
	emb   embed.Embedder
}

func scopeFilterFromReq(req *types.SearchRequest) store.Filter {
	if v, ok := req.Filter["doc_id"].([]string); ok && len(v) > 0 {
		return store.NewFilter(v)
	}
	if len(req.KbIDs) > 0 {
		return store.NewFilter(req.KbIDs)
	}
	return store.Filter{}
}

func (e *storeDocEngine) Search(ctx context.Context, req *types.SearchRequest) (*types.SearchResult, error) {
	if e.store == nil {
		return &types.SearchResult{}, nil
	}
	var text string
	var vec []float32
	for _, me := range req.MatchExprs {
		switch v := me.(type) {
		case *types.MatchTextExpr:
			if v.MatchingText != "" {
				text = v.MatchingText
			}
		case *types.MatchDenseExpr:
			if len(v.EmbeddingData) > 0 {
				vec = f64ToF32(v.EmbeddingData)
			}
		}
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	filter := scopeFilterFromReq(req)

	var hits []store.Hit
	switch {
	case text != "" && len(vec) > 0 && e.emb != nil:
		hits = e.store.Hybrid(text, vec, limit)
	case text != "":
		hits = e.store.SearchIn(text, limit, filter)
	case len(vec) > 0:
		hits = e.store.SearchVector(vec, limit)
	}
	return &types.SearchResult{Chunks: rowsFromChunks(chunksOf(hits))}, nil
}

func (e *storeDocEngine) SearchByRegexp(ctx context.Context, req *types.RegexpSearchRequest) (*types.SearchResult, error) {
	if e.store == nil {
		return &types.SearchResult{}, nil
	}
	filter := store.Filter{}
	if v, ok := req.Filter["doc_id"].([]string); ok && len(v) > 0 {
		filter = store.NewFilter(v)
	}
	limit := 200
	if req.ReturnAll {
		limit = 10000
	}
	hits, err := e.store.GrepIn(req.Pattern, true, limit, filter)
	if err != nil {
		return nil, err
	}
	cs := make([]store.Chunk, 0, len(hits))
	for _, h := range hits {
		cs = append(cs, h.Chunk)
	}
	return &types.SearchResult{Chunks: rowsFromChunks(cs)}, nil
}

func chunksOf(hits []store.Hit) []store.Chunk {
	cs := make([]store.Chunk, 0, len(hits))
	for _, h := range hits {
		cs = append(cs, h.Chunk)
	}
	return cs
}

func rowsFromChunks(hits []store.Chunk) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(hits))
	for i, h := range hits {
		name := h.DocID
		if h.Metadata != nil {
			if n, ok := h.Metadata["doc_name"].(string); ok && n != "" {
				name = n
			}
		}
		out = append(out, map[string]interface{}{
			"id":                  h.ChunkID,
			"_id":                 h.ChunkID,
			"doc_id":              h.DocID,
			"docnm_kwd":           name,
			"kb_id":               "",
			"chunk_order_int":     i,
			"page_num_int":        h.PageNum,
			"content_with_weight": h.Text,
			"content":             h.Text,
			"knowledge_graph_kwd": "",
			"available_int":       1,
		})
	}
	return out
}

func f64ToF32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
