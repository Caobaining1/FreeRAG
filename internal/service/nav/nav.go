// Package nav is the FreeRAG shim mirroring ragflow/internal/service/nav. It
// defines the NavService interface consumed by the agentic navigate_* tools.
//
// FreeRAG has no compiled knowledge-graph backend (RAGFlow's nav index is built
// from knowledge_graph_kwd rows). The FreeRAG NavService is therefore a
// best-effort, document-level implementation backed by the in-process store:
// it routes queries to documents and exposes the document set as a flat
// navigation forest. The tool contracts are unchanged; only the backing data
// differs.
package nav

import (
	"context"
	"sync"

	"freerag/internal/store"
)

// NavNode mirrors RAGFlow's frontend DatasetNavNode contract (snake_case JSON).
type NavNode struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	DocCount    int    `json:"doc_count"`
	Type        string `json:"type"`
	DocID       string `json:"doc_id,omitempty"`
	HasChildren bool   `json:"has_children"`
	Parent      string `json:"parent_kwd,omitempty"`
	Matched     bool   `json:"matched,omitempty"`
}

// Nav row types.
const (
	TypeNavDoc     = "nav_doc"
	TypeNavCluster = "nav_cluster"
)

// NavHit is one KNN hit on a nav row.
type NavHit struct {
	Type   string
	DocID  string
	DocIDs []string
	Name   string
	Score  float64
}

// UpsertDocInput carries a per-document summary to place into the nav tree.
type UpsertDocInput struct {
	CompileKind string
	TenantID    string
	KbID        string
	DocID       string
	Summary     string
	Embedd      []float32
}

// NavMergeLLM summarizes or merges nav text via an LLM. nil disables LLM
// behavior and the implementation falls back to deterministic naming.
type NavMergeLLM interface {
	Merge(ctx context.Context, tenantID string, texts []string) (string, error)
	CreateSummary(ctx context.Context, tenantID string, text string) (name, summary string, err error)
}

// NavService is the single read/write entrypoint for a dataset's navigation
// tree, consumed by the agent navigate_* tools.
type NavService interface {
	UpsertDoc(ctx context.Context, in UpsertDocInput) error
	RemoveDoc(ctx context.Context, tenantID, kbID, docID string) error
	Search(ctx context.Context, tenantID, kbID, query string, embd []float32, docScope []string, topK int) ([]NavHit, error)
	ListClusters(ctx context.Context, tenantID, kbID, keywords string, page, pageSize int) ([]NavNode, int64, error)
	ListChildren(ctx context.Context, tenantID, kbID, name, keywords string, page, pageSize int) ([]NavNode, int64, error)
	SummariesByDocIDs(ctx context.Context, tenantID, kbID string, docIDs []string) map[string]string
}

var (
	svcMu   sync.RWMutex
	svcInst NavService
)

// SetNavService installs the (production or test) NavService singleton.
func SetNavService(s NavService) {
	svcMu.Lock()
	defer svcMu.Unlock()
	svcInst = s
}

// GetNavService returns the installed NavService (may be nil until SetNavService
// is called during bootstrap).
func GetNavService() NavService {
	svcMu.RLock()
	defer svcMu.RUnlock()
	return svcInst
}

// SetNavStore installs a store-backed NavService as the package singleton.
func SetNavStore(s *store.Store) {
	SetNavService(&storeNavService{store: s})
}

// storeNavService is the document-level best-effort NavService.
type storeNavService struct {
	store *store.Store
}

func (n *storeNavService) UpsertDoc(_ context.Context, _ UpsertDocInput) error { return nil }

func (n *storeNavService) RemoveDoc(_ context.Context, _, _, _ string) error { return nil }

func (n *storeNavService) Search(_ context.Context, _, kbID, query string, _ []float32, docScope []string, topK int) ([]NavHit, error) {
	if n.store == nil {
		return nil, nil
	}
	if topK <= 0 {
		topK = 12
	}
	hits := n.store.Search(query, topK)
	seen := make(map[string]bool)
	out := make([]NavHit, 0, len(hits))
	for _, h := range hits {
		if seen[h.Chunk.DocID] {
			continue
		}
		if len(docScope) > 0 && !contains(docScope, h.Chunk.DocID) {
			continue
		}
		seen[h.Chunk.DocID] = true
		out = append(out, NavHit{Type: TypeNavDoc, DocID: h.Chunk.DocID, Name: h.Chunk.DocID, Score: h.Score})
	}
	return out, nil
}

func (n *storeNavService) ListClusters(_ context.Context, _, _, keywords string, page, pageSize int) ([]NavNode, int64, error) {
	docs := n.store.Documents()
	if keywords != "" {
		filtered := docs[:0]
		for _, d := range docs {
			if containsI(d, keywords) {
				filtered = append(filtered, d)
			}
		}
		docs = filtered
	}
	total := int64(len(docs))
	start := page * pageSize
	if start > len(docs) {
		start = len(docs)
	}
	end := start + pageSize
	if pageSize <= 0 || end > len(docs) {
		end = len(docs)
	}
	nodes := make([]NavNode, 0, end-start)
	for _, d := range docs[start:end] {
		nodes = append(nodes, NavNode{Name: d, DocID: d, Type: TypeNavDoc, HasChildren: false, DocCount: 1})
	}
	return nodes, total, nil
}

func (n *storeNavService) ListChildren(_ context.Context, _, _, _, _ string, _, _ int) ([]NavNode, int64, error) {
	return nil, 0, nil
}

func (n *storeNavService) SummariesByDocIDs(_ context.Context, _, _ string, docIDs []string) map[string]string {
	out := make(map[string]string, len(docIDs))
	for _, d := range docIDs {
		out[d] = ""
	}
	return out
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func containsI(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexI(s, sub) >= 0)
}

func indexI(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if stringsEqualFold(s[i:i+len(sub)], sub) {
			return i
		}
	}
	return -1
}

func stringsEqualFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if 'A' <= ca && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if 'A' <= cb && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}
