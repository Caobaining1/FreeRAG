package store

import "sort"

// Filter restricts retrieval to a set of documents.
//
// The zero value allows everything, so a caller with no scope passes Filter{}
// and nothing changes.
//
// A filter is applied DURING retrieval, not to the results afterwards, and that
// distinction is the whole reason this type exists. Post-filtering a top-k can
// return nothing while in-scope passages exist, because out-of-scope passages
// took the slots: filter to one document, ask a question that document answers
// well, and a global top-10 full of other documents hides it. RAGFlow reaches
// the same conclusion — its metadata filter converges on a set of document ids
// and then passes them as the search's DocScope
// (internal/rag/agentic-rag/runtime/tool_search.go's MetadataDocIDs →
// HybridSearch with DocScope: docIDs), rather than sifting a finished ranking.
//
// Two things are deliberately NOT scoped:
//
//   - The BM25 statistics (document frequencies, average length) stay corpus
//     wide, which is what elasticsearch does under a filter as well; recomputing
//     them per scope would make one query's scores incomparable with the next.
//   - The dense index is only scoped when it can be — see ScopedDenseIndex. An
//     index that cannot filter is still asked, and its answer is filtered after
//     it, which is the weaker behaviour this type exists to avoid where possible.
type Filter struct {
	docs map[string]bool
}

// NewFilter builds a filter over docIDs. An empty or all-empty list gives the
// zero Filter, which allows everything: a scope with no documents in it would
// otherwise mean "return nothing", and that is not what an absent filter means.
func NewFilter(docIDs []string) Filter {
	docs := make(map[string]bool, len(docIDs))
	for _, id := range docIDs {
		if id != "" {
			docs[id] = true
		}
	}
	if len(docs) == 0 {
		return Filter{}
	}
	return Filter{docs: docs}
}

// Active reports whether anything is restricted.
func (f Filter) Active() bool { return len(f.docs) > 0 }

// Allows reports whether a document may appear in the results.
func (f Filter) Allows(docID string) bool {
	if len(f.docs) == 0 {
		return true
	}
	return f.docs[docID]
}

// Docs returns the documents in a stable order, and nil when inactive. Stable
// because it reaches a vector database as a request body: two runs of the same
// question must not differ in bytes.
func (f Filter) Docs() []string {
	if len(f.docs) == 0 {
		return nil
	}
	ids := make([]string, 0, len(f.docs))
	for id := range f.docs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Len reports how many documents the filter admits, or 0 when inactive.
func (f Filter) Len() int { return len(f.docs) }
