// Package runtime is the FreeRAG shim mirroring ragflow/internal/agent/runtime.
// It provides the RetrievalChunk/RetrievalRequest/GrepRequest/Bm25Request types
// and the RetrievalService/GrepService/Bm25Service interfaces consumed by the
// migrated agentic_rag tool layer, backed by FreeRAG's in-process store (BM25 +
// dense) instead of RAGFlow's MySQL/ES retrieval backend.
//
// Unlike RAGFlow, FreeRAG has no gorm/MySQL layer, so the service Search/Grep
// signatures drop the *gorm.DB parameter.
package runtime

import (
	"context"
	"errors"
	"sync"

	"freerag/internal/embed"
	"freerag/internal/store"
)

// RetrievalChunk is one retrieved passage. The JSON field names match the
// RAGFlow contract exactly so downstream XML formatting is unchanged.
type RetrievalChunk struct {
	ID           string  `json:"id,omitempty" yaml:"id"`
	Content      string  `json:"content,omitempty" yaml:"content"`
	DocumentID   string  `json:"document_id,omitempty" yaml:"document_id"`
	DocumentName string  `json:"document_name,omitempty" yaml:"document_name"`
	PageNum      int     `json:"page_num,omitempty" yaml:"page_num"`
	ChunkIndex   int     `json:"chunk_index,omitempty" yaml:"chunk_index"`
	DatasetID    string  `json:"dataset_id,omitempty" yaml:"dataset_id"`
	Score        float64 `json:"score,omitempty" yaml:"score"`
}

// RetrievalRequest is the locator passed to RetrievalService.Search.
type RetrievalRequest struct {
	Query                   string   `json:"query"`
	DatasetIDs              []string `json:"dataset_ids"`
	Fields                  []string `json:"fields"`
	DocIDs                  []string `json:"doc_ids"`
	TopN                    int      `json:"top_n"`
	TopK                    int      `json:"top_k"`
	SimilarityThreshold     *float64 `json:"similarity_threshold"`
	KeywordsSimilarityWeight *float64 `json:"keywords_similarity_weight"`
	DocScope                []string `json:"doc_scope"`
	TenantID                string   `json:"tenant_id"`
	VectorOnly              bool     `json:"vector_only"`
	SelectFields            []string `json:"select_fields"`
	OnlyOriginalText        bool     `json:"only_original_text"`
}

// GrepRequest scopes a grep / deep-read over one or more documents.
type GrepRequest struct {
	TenantID   string   `json:"tenant_id"`
	DatasetIDs []string `json:"dataset_ids"`
	DocScope   []string `json:"doc_scope"`
	ChunkScope []string `json:"chunk_scope"`
	Pattern    string   `json:"pattern"`
	UseRegex   bool     `json:"use_regex"`
	Sort       []string `json:"sort"`
	SelectFields []string `json:"select_fields"`
	Limit      int      `json:"limit"`
	Page       int      `json:"page"`
	PageSize   int      `json:"page_size"`
}

// Bm25Request is the lexical multi-query request for the bm25 tool.
type Bm25Request struct {
	Queries    []string `json:"queries"`
	DatasetIDs []string `json:"dataset_ids"`
	DocScope   []string `json:"doc_scope"`
	TopN       int      `json:"top_n"`
	TenantID   string   `json:"tenant_id"`
	Limit      int      `json:"limit"`
}

// Errors returned by the store-backed services.
var (
	ErrBm25ServiceMissing = errors.New("bm25 service not registered")
	ErrGrepServiceMissing = errors.New("grep service not registered")
	ErrRegexpNotSupported = errors.New("regexp search is not supported by this engine")
	ErrRegexpPushdown     = errors.New("regexp pushdown rejected by the engine")
)

// grep/bm25 constants referenced by the migrated tool layer.
const (
	searchChunksDefaultTopN = 8
	grepChunksMaxResults    = 30
	grepChunksSelectFields  = "id,doc_id,docnm_kwd,kb_id,chunk_order_int,page_num_int,content_with_weight"
	grepChunksSortFields    = "doc_id,page_num_int,chunk_order_int"
)

// RetrievalService is the locator used by the search_* agent tools.
type RetrievalService interface {
	Search(ctx context.Context, req RetrievalRequest) ([]RetrievalChunk, error)
}

// GrepService is the grep / deep-read service used by grep_chunks and
// list_chunks.
type GrepService interface {
	Grep(ctx context.Context, req GrepRequest) ([]RetrievalChunk, error)
	ResolveDocDatasetID(ctx context.Context, req GrepRequest) (string, error)
	ListDocumentChunkIndex(ctx context.Context, req GrepRequest) ([]RetrievalChunk, error)
	FetchChunksByID(ctx context.Context, req GrepRequest) ([]RetrievalChunk, error)
}

// Bm25Service backs the search_bm25_chunks tool with pure lexical ranking.
type Bm25Service interface {
	SearchBm25(ctx context.Context, req Bm25Request) ([]RetrievalChunk, error)
}

var (
	svcMu     sync.RWMutex
	retrieval RetrievalService
	grep      GrepService
	bm25      Bm25Service
)

// SetRetrievalService installs the (production or test) RetrievalService.
func SetRetrievalService(s RetrievalService) {
	svcMu.Lock()
	defer svcMu.Unlock()
	retrieval = s
}

// GetRetrievalService returns the installed RetrievalService (may be nil).
func GetRetrievalService() RetrievalService {
	svcMu.RLock()
	defer svcMu.RUnlock()
	return retrieval
}

// SetGrepService installs the (production or test) GrepService.
func SetGrepService(s GrepService) {
	svcMu.Lock()
	defer svcMu.Unlock()
	grep = s
}

// GetGrepService returns the installed GrepService (may be nil).
func GetGrepService() GrepService {
	svcMu.RLock()
	defer svcMu.RUnlock()
	return grep
}

// SetBm25Service installs the (production or test) Bm25Service.
func SetBm25Service(s Bm25Service) {
	svcMu.Lock()
	defer svcMu.Unlock()
	bm25 = s
}

// GetBm25Service returns the installed Bm25Service (may be nil).
func GetBm25Service() Bm25Service {
	svcMu.RLock()
	defer svcMu.RUnlock()
	return bm25
}

// SetStore installs a store-backed RetrievalService and GrepService as the
// package globals, wiring them to the given embedder for hybrid search.
func SetStore(s *store.Store, emb embed.Embedder) {
	SetRetrievalService(&storeRetrievalService{store: s, emb: emb})
	SetGrepService(&storeGrepService{store: s})
}

// --- store-backed implementations ---

type storeRetrievalService struct {
	store *store.Store
	emb   embed.Embedder
}

func toChunk(c store.Chunk, score float64) RetrievalChunk {
	name := c.DocID
	if c.Metadata != nil {
		if n, ok := c.Metadata["doc_name"].(string); ok && n != "" {
			name = n
		}
	}
	return RetrievalChunk{
		ID:           c.ChunkID,
		Content:      c.Text,
		DocumentID:   c.DocID,
		DocumentName: name,
		PageNum:      c.PageNum,
		ChunkIndex:   0,
		DatasetID:    "",
		Score:        score,
	}
}

func scopeFilter(req RetrievalRequest) store.Filter {
	if len(req.DocScope) > 0 {
		return store.NewFilter(req.DocScope)
	}
	if len(req.DatasetIDs) > 0 {
		return store.NewFilter(req.DatasetIDs)
	}
	return store.Filter{}
}

func (s *storeRetrievalService) Search(ctx context.Context, req RetrievalRequest) ([]RetrievalChunk, error) {
	if s.store == nil {
		return nil, nil
	}
	limit := req.TopN
	if limit <= 0 {
		limit = req.TopK
	}
	if limit <= 0 {
		limit = searchChunksDefaultTopN
	}
	filter := scopeFilter(req)

	var hits []store.Hit
	if s.emb != nil && !req.VectorOnly {
		if vecs, err := s.emb.Embed(ctx, []string{req.Query}); err == nil && len(vecs) > 0 {
			hits = s.store.Hybrid(req.Query, vecs[0], limit)
		}
	}
	if len(hits) == 0 {
		hits = s.store.SearchIn(req.Query, limit, filter)
	}
	out := make([]RetrievalChunk, 0, len(hits))
	for _, h := range hits {
		out = append(out, toChunk(h.Chunk, h.Score))
	}
	return out, nil
}

type storeGrepService struct {
	store *store.Store
}

func (s *storeGrepService) Grep(ctx context.Context, req GrepRequest) ([]RetrievalChunk, error) {
	if s.store == nil {
		return nil, nil
	}
	limit := req.Limit
	if limit <= 0 {
		limit = grepChunksMaxResults
	}
	filter := store.Filter{}
	if len(req.DocScope) > 0 {
		filter = store.NewFilter(req.DocScope)
	}
	hits, err := s.store.GrepIn(req.Pattern, req.UseRegex, limit, filter)
	if err != nil {
		return nil, err
	}
	out := make([]RetrievalChunk, 0, len(hits))
	for _, h := range hits {
		out = append(out, toChunk(h.Chunk, 0))
	}
	return out, nil
}

func (s *storeGrepService) ResolveDocDatasetID(_ context.Context, req GrepRequest) (string, error) {
	if len(req.DatasetIDs) > 0 {
		return req.DatasetIDs[0], nil
	}
	return "", nil
}

func (s *storeGrepService) ListDocumentChunkIndex(_ context.Context, req GrepRequest) ([]RetrievalChunk, error) {
	if s.store == nil || len(req.DocScope) == 0 {
		return nil, nil
	}
	chunks := s.store.List(req.DocScope[0], 0, 0, 1<<20)
	out := make([]RetrievalChunk, 0, len(chunks))
	for i, c := range chunks {
		out = append(out, RetrievalChunk{
			ID:         c.ChunkID,
			Content:    c.Text,
			DocumentID: c.DocID,
			PageNum:    c.PageNum,
			ChunkIndex: i,
		})
	}
	return out, nil
}

func (s *storeGrepService) FetchChunksByID(_ context.Context, req GrepRequest) ([]RetrievalChunk, error) {
	if s.store == nil || len(req.DocScope) == 0 {
		return nil, nil
	}
	chunks := s.store.List(req.DocScope[0], 0, 0, 1<<20)
	want := make(map[string]bool, len(req.ChunkScope))
	for _, id := range req.ChunkScope {
		want[id] = true
	}
	out := make([]RetrievalChunk, 0, len(req.ChunkScope))
	for _, c := range chunks {
		if want[c.ChunkID] {
			out = append(out, toChunk(c, 0))
		}
	}
	return out, nil
}

// StringFromMap returns m[key] as a string, or "" when absent/non-string.
func StringFromMap(m map[string]any, key string) string {
	return FirstStringFromMap(m, key)
}
