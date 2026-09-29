package store

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
)

// rrfK is the reciprocal-rank-fusion constant.
//
// 60 is the value from the original RRF paper and the common default. It sets
// how quickly rank position stops mattering: with k=60 the top hit scores
// 1/61 ≈ 0.0164 and the tenth 1/70 ≈ 0.0143, so no single ranker can run away
// with the result and agreement between rankers is rewarded.
const rrfK = 60

// defaultFusionPool is how deep each ranker is asked to go before fusing.
const defaultFusionPool = 30

// encodeVector serialises a vector as base64 of little-endian float32.
func encodeVector(vector []float32) string {
	raw := make([]byte, 4*len(vector))
	for i, value := range vector {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return base64.StdEncoding.EncodeToString(raw)
}

// decodeVector reverses encodeVector.
func decodeVector(encoded string) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("vector is not valid base64: %w", err)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("vector is %d bytes, not a multiple of 4", len(raw))
	}
	vector := make([]float32, len(raw)/4)
	for i := range vector {
		vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return vector, nil
}

// cosine returns the cosine similarity of a and b, or 0 when either is zero.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		normA += x * x
		normB += y * y
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// SearchVector ranks chunks by cosine similarity to vector.
//
// An attached dense index (dense.go) answers first when there is one: that is
// the approximate-index path, and it is what keeps this call from being a
// linear scan over every vector in the corpus. Everything below it is the
// exact in-process fallback, used when no index is attached or the attached one
// fails.
//
// Returns nil when the vector cannot be compared with this index — a width
// mismatch means the vectors came from different models, and ranking them
// anyway would return plausible-looking nonsense.
func (s *Store) SearchVector(vector []float32, limit int) []Hit {
	if len(vector) == 0 {
		return nil
	}

	if hits, ok := s.searchDense(vector, limit); ok {
		return hits
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.dims != 0 && len(vector) != s.dims {
		return nil
	}

	type scored struct {
		index int
		score float64
	}
	ranked := make([]scored, 0, len(s.chunks))
	for i := range s.chunks {
		if i >= len(s.vectors) || len(s.vectors[i]) == 0 {
			continue
		}
		score := cosine(vector, s.vectors[i])
		if score <= 0 {
			// A non-positive cosine is noise for retrieval purposes; keeping it
			// would let an unrelated chunk outrank a genuinely absent answer.
			continue
		}
		ranked = append(ranked, scored{index: i, score: score})
	}
	if len(ranked) == 0 {
		return nil
	}

	sort.SliceStable(ranked, func(a, b int) bool {
		if ranked[a].score != ranked[b].score {
			return ranked[a].score > ranked[b].score
		}
		return ranked[a].index < ranked[b].index
	})
	if limit > 0 && len(ranked) > limit {
		ranked = ranked[:limit]
	}

	hits := make([]Hit, 0, len(ranked))
	for _, item := range ranked {
		hits = append(hits, Hit{Chunk: s.chunks[item.index], Score: item.score, Sources: []string{"dense"}})
	}
	return hits
}

// Hybrid fuses keyword and dense rankings with reciprocal rank fusion.
//
// RRF combines by *rank* rather than by score, which is what lets BM25 (an
// unbounded scale that varies per corpus) and cosine (bounded in [-1, 1], and
// model-specific) be used together without tuning a weight for every corpus.
// The returned Score is therefore always an RRF score, never a raw BM25 or
// cosine value — mixing the two in one field would make results incomparable
// across queries.
//
// Either half may be empty: with no embedder, or a query the keyword index
// cannot match, this degrades to the other ranker's ranking rather than failing.
func (s *Store) Hybrid(query string, vector []float32, limit int) []Hit {
	if limit <= 0 {
		limit = 10
	}
	pool := limit * 4
	if pool < defaultFusionPool {
		pool = defaultFusionPool
	}

	keyword := s.Search(query, pool)
	dense := s.SearchVector(vector, pool)

	type fused struct {
		chunk   Chunk
		score   float64
		sources []string
	}
	merged := make(map[string]*fused, len(keyword)+len(dense))
	order := make([]string, 0, len(keyword)+len(dense))

	merge := func(hits []Hit, label string) {
		for rank, hit := range hits {
			key := dedupKey(hit.Chunk)
			entry, ok := merged[key]
			if !ok {
				entry = &fused{chunk: hit.Chunk}
				merged[key] = entry
				order = append(order, key)
			}
			entry.score += 1.0 / float64(rrfK+rank+1)
			entry.sources = append(entry.sources, label)
		}
	}
	merge(keyword, "bm25")
	merge(dense, "dense")

	out := make([]Hit, 0, len(order))
	for _, key := range order {
		entry := merged[key]
		out = append(out, Hit{Chunk: entry.chunk, Score: entry.score, Sources: entry.sources})
	}
	// Ties break on insertion order, which is keyword rank first: a fused tie
	// then still prefers the term-match, the more explainable of the two.
	sort.SliceStable(out, func(a, b int) bool { return out[a].Score > out[b].Score })

	if len(out) > limit {
		out = out[:limit]
	}
	return out
}
