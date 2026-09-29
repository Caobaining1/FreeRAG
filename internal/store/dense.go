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
func (s *Store) searchDense(vector []float32, limit int) ([]Hit, bool) {
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

	matches, err := index.Search(vector, limit)
	if err != nil {
		s.loggerOr().Printf("warning: dense index %s search failed (%v); falling back to the in-process scan",
			index.Backend(), err)
		return nil, false
	}
	return s.resolve(matches), true
}

// resolve maps dense matches back to the chunks this store owns.
//
// A match with no chunk here is dropped rather than fabricated: it means the
// index and the chunk store disagree, and inventing a hit would hide that.
func (s *Store) resolve(matches []DenseMatch) []Hit {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Hit, 0, len(matches))
	for _, match := range matches {
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
