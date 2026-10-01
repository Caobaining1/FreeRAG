package store

import "log"

// DenseMatch is one nearest-neighbour result from an external dense index.
//
// It names the chunk rather than carrying it: the index owns the vectors, the
// store owns the chunks, and neither has to duplicate the other.
type DenseMatch struct {
	DocID   string
	ChunkID string
	Score   float64
}

// DenseIndex is an external dense retriever — a vector database that owns the
// vectors and answers nearest-neighbour queries through an approximate index.
//
// The interface is deliberately narrow. Only the dense leg is delegated,
// because it is the only leg that is O(N) in-process and properly indexed in a
// vector database. BM25, grep, metadata filtering and the RRF fusion stay here:
// they are exact, they need no service, and they are what keeps retrieval
// working when the vector database is unreachable.
type DenseIndex interface {
	// Backend names the implementation, for logs and GET /version.
	Backend() string
	// Upsert adds or replaces the vectors belonging to these chunks. It is
	// called with vectors[i] belonging to chunks[i], never with a nil entry.
	Upsert(chunks []Chunk, vectors [][]float32) error
	// Search returns the nearest chunks to vector, best first.
	Search(vector []float32, limit int) ([]DenseMatch, error)
	// Delete drops every vector belonging to one document, so re-indexing a
	// changed document does not leave its previous vectors behind.
	Delete(docID string) error
	// Len reports how many vectors the index holds.
	Len() int
}

// ScopedDenseIndex is a DenseIndex that can restrict a search to a set of
// documents inside the index.
//
// Optional, and separate from DenseIndex, because it is the difference between a
// scope and a post-filter: an index that cannot filter returns a global top-k
// which is then sifted, so a scoped query can come back empty while in-scope
// passages exist (see Filter). Implementations that can push the restriction
// down should, and the ones that cannot keep working — the sifter is the safety
// net, not the mechanism.
type ScopedDenseIndex interface {
	DenseIndex
	// SearchScoped returns the nearest chunks to vector among docIDs, best
	// first. docIDs is never empty.
	SearchScoped(vector []float32, limit int, docIDs []string) ([]DenseMatch, error)
}

// DroppableIndex is a DenseIndex that owns its own storage and can discard it.
//
// Separate from DenseIndex because most of them cannot: the in-process scan has
// no storage of its own, and requiring Drop of every implementation would make
// that one pretend. Deleting a knowledge base needs it, so the caller
// type-asserts and skips the step when it is absent.
type DroppableIndex interface {
	DenseIndex
	Drop() error
}

// SetDenseIndex attaches an external dense retriever.
//
// The store keeps its own copy of the vectors even when one is attached: they
// are what a reconnect needs to re-seed the index, and what the in-process scan
// falls back to when the index cannot be reached.
func (s *Store) SetDenseIndex(index DenseIndex) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dense = index
}

// DenseBackend names the attached dense index, or "" when dense search runs
// in-process.
func (s *Store) DenseBackend() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.dense == nil {
		return ""
	}
	return s.dense.Backend()
}

// SetLogger sets where degradation warnings go; nil keeps log.Default().
func (s *Store) SetLogger(logger *log.Logger) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.logger = logger
}

func (s *Store) loggerOr() *log.Logger {
	s.mu.RLock()
	logger := s.logger
	s.mu.RUnlock()
	if logger != nil {
		return logger
	}
	return log.Default()
}

// EmbeddedChunks returns the chunks that carry a vector, with their vectors.
//
// It is what re-seeds an external index after it has been wiped or replaced:
// the vectors survived in the index file, so no re-embedding is needed — which
// would otherwise mean paying the embedding provider again for the whole corpus.
func (s *Store) EmbeddedChunks() ([]Chunk, [][]float32) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	chunks := make([]Chunk, 0, len(s.chunks))
	vectors := make([][]float32, 0, len(s.chunks))
	for i, chunk := range s.chunks {
		if i >= len(s.vectors) || len(s.vectors[i]) == 0 {
			continue
		}
		chunks = append(chunks, chunk)
		vectors = append(vectors, s.vectors[i])
	}
	return chunks, vectors
}

// searchDense asks the attached index, reporting whether it answered.
//
// A failure is not propagated: the caller falls back to the in-process scan,
// which is exact and always available. Returning nothing because a vector
// database is down would turn a degraded search into a failed one, and a
// retrieval tool that returns no hits is worse than one that returns the
// keyword matches it can still compute.
func (s *Store) searchDense(vector []float32, limit int, filter Filter) ([]Hit, bool) {
	s.mu.RLock()
	index := s.dense
	dims := s.dims
	s.mu.RUnlock()

	if index == nil {
		return nil, false
	}
	// A width mismatch means the vectors came from different models. Ranking
	// them anyway would return plausible-looking nonsense, so treat it as "the
	// index did not answer" and let the local scan apply the same rule.
	if dims != 0 && len(vector) != dims {
		return nil, false
	}

	var (
		matches []DenseMatch
		err     error
	)
	if scoped, ok := index.(ScopedDenseIndex); ok && filter.Active() {
		// The index filters, so the limit it is given is a limit WITHIN the
		// scope. This is the branch that makes a scope a scope.
		matches, err = scoped.SearchScoped(vector, limit, filter.Docs())
	} else {
		// An index that cannot filter is still asked, and its answer is sifted
		// afterwards. Weaker on purpose and documented in Filter: in-scope
		// passages can lose their slots to out-of-scope ones here, and the
		// in-process fallback (which filters exactly) is what covers the case
		// where that matters.
		matches, err = index.Search(vector, limit)
	}
	if err != nil {
		s.loggerOr().Printf("warning: dense index %s search failed (%v); falling back to the in-process scan",
			index.Backend(), err)
		return nil, false
	}
	return s.resolveIn(matches, filter), true
}

// resolve maps dense matches back to the chunks this store owns.
//
// A match with no chunk here is dropped rather than fabricated: it means the
// index and the chunk store disagree, and inventing a hit would hide that.
func (s *Store) resolve(matches []DenseMatch) []Hit {
	return s.resolveIn(matches, Filter{})
}

// resolveIn is resolve with the filter applied to what comes back, which is a
// safety net rather than the mechanism: an index that can filter was already
// asked to (see searchDense), and this catches one that cannot or did not.
func (s *Store) resolveIn(matches []DenseMatch, filter Filter) []Hit {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Hit, 0, len(matches))
	for _, match := range matches {
		if !filter.Allows(match.DocID) {
			continue
		}
		index, ok := s.seen[dedupKey(Chunk{DocID: match.DocID, ChunkID: match.ChunkID})]
		if !ok || index >= len(s.chunks) {
			continue
		}
		out = append(out, Hit{
			Chunk:   s.chunks[index],
			Score:   match.Score,
			Sources: []string{"dense"},
		})
	}
	return out
}
