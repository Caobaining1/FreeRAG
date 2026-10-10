// Package types mirrors the subset of ragflow/internal/engine/types consumed by
// the migrated agentic_rag BM25/grep/kg service adapters and the navigate_*
// tools. The field names match RAGFlow exactly so the adapter code is unchanged
// (only its import path differs).
//
// FreeRAG has no Elasticsearch; the FreeRAG engine.DocEngine implementation
// (freerag/internal/engine) interprets these request shapes against the
// in-process store, so the field semantics are best-effort equivalents.
package types

// SearchRequest is the unified hybrid search request.
type SearchRequest struct {
	IndexNames       []string      `json:"index_names"`
	Indices          []string      `json:"indices"`
	KbIDs            []string      `json:"kb_ids"`
	Limit            int           `json:"limit"`
	From             int           `json:"from"`
	MatchExprs       []interface{} `json:"match_exprs"`
	Filter           map[string]interface{} `json:"filter"`
	SelectFields     []string      `json:"select_fields"`
	IncludeUnavailable bool        `json:"include_unavailable"`
	OrderBy          *OrderByExpr  `json:"order_by"`
	// Highlight is accepted but ignored by the FreeRAG store engine.
	Highlight map[string]interface{} `json:"highlight"`
}

// SearchResult is the hybrid search response rows.
type SearchResult struct {
	Total  int                      `json:"total"`
	Chunks []map[string]interface{} `json:"chunks"`
}

// MatchTextExpr is a BM25 match expression.
type MatchTextExpr struct {
	MatchingText string   `json:"matching_text"`
	TopN         int      `json:"top_n"`
	Fields       []string `json:"fields"`
}

// MatchDenseExpr is a dense (vector) nearest-neighbor expression.
type MatchDenseExpr struct {
	VectorColumnName string                 `json:"vector_column_name"`
	EmbeddingData    []float64              `json:"embedding_data"`
	EmbeddingDataType string                `json:"embedding_data_type"`
	DistanceType     string                 `json:"distance_type"`
	TopN             int                    `json:"top_n"`
	ExtraOptions     map[string]interface{} `json:"extra_options"`
}

// OrderByExpr is the sort specification (decending/ascending field list).
type OrderByExpr struct {
	fields []string
	orders []string
}

// Asc appends an ascending field.
func (o *OrderByExpr) Asc(field string) *OrderByExpr {
	if o == nil {
		o = &OrderByExpr{}
	}
	o.fields = append(o.fields, field)
	o.orders = append(o.orders, "asc")
	return o
}

// Desc appends a descending field.
func (o *OrderByExpr) Desc(field string) *OrderByExpr {
	if o == nil {
		o = &OrderByExpr{}
	}
	o.fields = append(o.fields, field)
	o.orders = append(o.orders, "desc")
	return o
}

// RegexpSearchRequest is the regexp (grep) search request.
type RegexpSearchRequest struct {
	TenantID    string                 `json:"tenant_id"`
	KbIDs       []string               `json:"kb_ids"`
	Pattern     string                 `json:"pattern"`
	Sort        *OrderByExpr           `json:"sort"`
	SelectFields []string              `json:"select_fields"`
	ReturnAll   bool                   `json:"return_all"`
	Filter      map[string]interface{} `json:"filter"`
}
