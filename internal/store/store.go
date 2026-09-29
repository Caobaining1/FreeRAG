// Package store holds parsed chunks and answers retrieval queries over them.
//
// Two retrievers live here and fuse into one: BM25 over a CJK-aware tokenizer,
// and dense cosine over embeddings. docs/plan.md §3 names SQLite + FTS5 +
// sqlite-vec as the production backing store; this package keeps the same
// Add/Search/Hybrid surface so swapping the backing store is a change of
// implementation, not of interface.
package store

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// BM25 parameters (standard defaults).
const (
	bm25K1 = 1.2
	bm25B  = 0.75
	// minLatinTerm is the shortest latin token worth indexing.
	minLatinTerm = 2
)

// Chunk is one indexed passage.
type Chunk struct {
	ChunkID   string         `json:"chunk_id"`
	Text      string         `json:"text"`
	DocID     string         `json:"doc_id"`
	PageNum   int            `json:"page_num"`
	BlockType string         `json:"block_type"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// Hit is one search result.
type Hit struct {
	Chunk Chunk   `json:"chunk"`
	Score float64 `json:"score"`
	// Line carries the line a literal scan matched, when the hit came from
	// grep_search. Ranked tools leave it empty: they match a whole chunk, not
	// a line.
	Line string `json:"line,omitempty"`
	// Sources names the retrievers that returned this chunk ("bm25", "dense").
	// Only Hybrid fills it, and it is what makes a fusion result auditable.
	Sources []string `json:"sources,omitempty"`
}

// Store is an in-process chunk index with optional JSON persistence.
type Store struct {
	mu      sync.RWMutex
	chunks  []Chunk
	vectors [][]float32      // parallel to chunks; nil entries are unembedded
	terms   []map[string]int // per-chunk term frequencies
	lengths []int            // per-chunk token counts
	df      map[string]int   // document frequency per term
	// seen maps a chunk's dedup key to its position in chunks. It doubles as
	// the lookup that maps an external dense index's matches back to chunks.
	seen map[string]int

	// documents records what has been indexed, keyed by content fingerprint.
	// It is what makes a re-index skippable (doc.go).
	documents map[string]DocumentRecord

	// dense is an optional external dense retriever (see dense.go). Nil keeps
	// dense ranking in-process.
	dense DenseIndex
	// logger receives degradation warnings; nil means log.Default().
	logger *log.Logger

	// embedder names the model that produced the vectors and dims its width.
	// Both are persisted so an index is never queried with vectors from a
	// different model — those would be numerically valid and semantically
	// meaningless, which is the worst possible failure mode.
	embedder string
	dims     int
}

// New returns an empty store.
func New() *Store {
	return &Store{
		df:        map[string]int{},
		seen:      map[string]int{},
		documents: map[string]DocumentRecord{},
	}
}

// Len reports how many chunks are indexed.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunks)
}

// dedupKey identifies a chunk across documents.
func dedupKey(c Chunk) string { return c.DocID + "\x00" + c.ChunkID }

// Add indexes chunks, skipping ones already present. It returns how many were
// newly added.
func (s *Store) Add(chunks []Chunk) int {
	added, _ := s.AddEmbedded(chunks, nil)
	return added
}

// AddEmbedded indexes chunks together with their embeddings.
//
// vectors[i] belongs to chunks[i]; an entry may be nil for a chunk that could
// not be embedded, which then stays reachable through BM25 and grep. Returns
// how many chunks were newly added.
//
// A length or width mismatch is an error rather than a partial write: silently
// aligning the wrong vector with a chunk would make dense retrieval return
// confidently irrelevant passages, and nothing downstream could detect it.
//
// Newly added vectors are forwarded to an attached dense index (dense.go). A
// forwarding failure is logged, not returned: the chunks are already indexed
// for BM25 and grep, so the correct outcome is "dense retrieval missed some
// chunks", not "the whole add failed".
func (s *Store) AddEmbedded(chunks []Chunk, vectors [][]float32) (int, error) {
	if vectors != nil && len(vectors) != len(chunks) {
		return 0, fmt.Errorf("store: %d vectors for %d chunks", len(vectors), len(chunks))
	}

	s.mu.Lock()

	for i, vector := range vectors {
		if len(vector) == 0 {
			continue
		}
		if s.dims == 0 {
			s.dims = len(vector)
		} else if len(vector) != s.dims {
			s.mu.Unlock()
			return 0, fmt.Errorf("store: vector %d has %d dimensions, index uses %d", i, len(vector), s.dims)
		}
	}

	added := 0
	var freshChunks []Chunk
	var freshVectors [][]float32
	for i, chunk := range chunks {
		if strings.TrimSpace(chunk.Text) == "" {
			continue
		}
		key := dedupKey(chunk)
		if _, ok := s.seen[key]; ok {
			continue
		}
		s.seen[key] = len(s.chunks)

		tf := map[string]int{}
		for _, term := range Terms(chunk.Text) {
			tf[term]++
		}
		total := 0
		for _, count := range tf {
			total += count
		}

		var vector []float32
		if vectors != nil {
			vector = vectors[i]
		}

		s.chunks = append(s.chunks, chunk)
		s.vectors = append(s.vectors, vector)
		s.terms = append(s.terms, tf)
		s.lengths = append(s.lengths, total)
		for term := range tf {
			s.df[term]++
		}
		if len(vector) > 0 {
			freshChunks = append(freshChunks, chunk)
			freshVectors = append(freshVectors, vector)
		}
		added++
	}
	index := s.dense
	s.mu.Unlock()

	// Outside the lock: the upsert is a network call, and holding the write
	// lock across it would stall every reader for the round trip.
	if index != nil && len(freshChunks) > 0 {
		if err := index.Upsert(freshChunks, freshVectors); err != nil {
			s.loggerOr().Printf(
				"warning: dense index %s rejected %d chunk(s) (%v); they stay keyword-only",
				index.Backend(), len(freshChunks), err)
		}
	}
	return added, nil
}

// SetEmbedder declares which model produces vectors for this index.
//
// It refuses a width change once vectors exist, because the stored vectors would
// no longer be comparable to freshly embedded queries.
func (s *Store) SetEmbedder(name string, dims int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.dims != 0 && dims != 0 && s.dims != dims {
		return fmt.Errorf("store: index holds %d-dimensional vectors, embedder provides %d", s.dims, dims)
	}
	s.embedder = name
	if dims > 0 {
		s.dims = dims
	}
	return nil
}

// EmbedderName reports the model the stored vectors came from, if known.
func (s *Store) EmbedderName() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.embedder
}

// Dimensions reports the vector width, or 0 when nothing is embedded.
func (s *Store) Dimensions() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.dims
}

// VectorCount reports how many chunks carry an embedding.
func (s *Store) VectorCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	count := 0
	for _, vector := range s.vectors {
		if len(vector) > 0 {
			count++
		}
	}
	return count
}

// All returns a copy of the indexed chunks.
func (s *Store) All() []Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Chunk, len(s.chunks))
	copy(out, s.chunks)
	return out
}

// Search ranks chunks against query and returns at most limit hits.
//
// An empty query, or one whose terms are all unknown, returns no hits: a search
// that matches nothing must not look like a search that matched everything.
func (s *Store) Search(query string, limit int) []Hit {
	queryTerms := Terms(query)
	if len(queryTerms) == 0 {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	total := len(s.chunks)
	if total == 0 {
		return nil
	}
	var totalLen int
	for _, length := range s.lengths {
		totalLen += length
	}
	if totalLen == 0 {
		return nil
	}
	avgdl := float64(totalLen) / float64(total)

	counts := map[string]int{}
	for _, term := range queryTerms {
		counts[term]++
	}

	scores := make([]float64, total)
	matched := false
	for i := range s.chunks {
		score := 0.0
		for term, qtf := range counts {
			freq, ok := s.terms[i][term]
			if !ok {
				continue
			}
			matched = true
			idf := math.Log(1 + (float64(total)-float64(s.df[term])+0.5)/(float64(s.df[term])+0.5))
			tf := float64(freq)
			dl := float64(s.lengths[i])
			denom := tf + bm25K1*(1-bm25B+bm25B*dl/avgdl)
			score += idf * (tf * (bm25K1 + 1) / denom) * float64(qtf)
		}
		scores[i] = score
	}
	if !matched {
		return nil
	}

	order := make([]int, 0, total)
	for i := range s.chunks {
		if scores[i] > 0 {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(a, b int) bool {
		if scores[order[a]] != scores[order[b]] {
			return scores[order[a]] > scores[order[b]]
		}
		return order[a] < order[b] // stable: keep index order on ties
	})
	if limit > 0 && len(order) > limit {
		order = order[:limit]
	}

	hits := make([]Hit, 0, len(order))
	for _, i := range order {
		hits = append(hits, Hit{Chunk: s.chunks[i], Score: scores[i]})
	}
	return hits
}

// Terms splits text into index terms: lowercased latin words and CJK bigrams.
//
// Bigrams matter here: Chinese has no spaces, so single characters are far too
// ambiguous and whole runs are far too specific. A one-character CJK run is
// kept as-is so short headings still index.
func Terms(text string) []string {
	var out []string
	var latin strings.Builder
	runes := []rune(text)

	flushLatin := func() {
		if latin.Len() >= minLatinTerm {
			out = append(out, strings.ToLower(latin.String()))
		}
		latin.Reset()
	}

	for i := 0; i < len(runes); i++ {
		r := runes[i]
		switch {
		case unicode.Is(unicode.Han, r):
			flushLatin()
			start := i
			for i < len(runes) && unicode.Is(unicode.Han, runes[i]) {
				i++
			}
			run := runes[start:i]
			i--
			if len(run) == 1 {
				out = append(out, string(run))
				continue
			}
			for j := 0; j+1 < len(run); j++ {
				out = append(out, string(run[j:j+2]))
			}
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			latin.WriteRune(r)
		default:
			flushLatin()
		}
	}
	flushLatin()
	return out
}

// storedChunk is a chunk plus its vector, in the on-disk form.
//
// Vectors are base64 of little-endian float32 — about 5.5 KB per 1024-dimension
// chunk in JSON. That is fine for a few hundred chunks and is exactly why
// docs/plan.md §3 wants sqlite-vec for a real corpus.
type storedChunk struct {
	Chunk
	Vector string `json:"vector,omitempty"`
}

// Save writes the index to path (JSON), creating parent directories.
func (s *Store) Save(path string) error {
	s.mu.RLock()
	payload := struct {
		Embedder  string           `json:"embedder,omitempty"`
		Dims      int              `json:"dims,omitempty"`
		Chunks    []storedChunk    `json:"chunks"`
		Documents []DocumentRecord `json:"documents,omitempty"`
	}{Embedder: s.embedder, Dims: s.dims}

	payload.Chunks = make([]storedChunk, 0, len(s.chunks))
	for i, chunk := range s.chunks {
		item := storedChunk{Chunk: chunk}
		if i < len(s.vectors) && len(s.vectors[i]) > 0 {
			item.Vector = encodeVector(s.vectors[i])
		}
		payload.Chunks = append(payload.Chunks, item)
	}
	// Sorted by fingerprint so the file is byte-stable across runs and its
	// diffs stay readable.
	payload.Documents = make([]DocumentRecord, 0, len(s.documents))
	for _, record := range s.documents {
		payload.Documents = append(payload.Documents, record)
	}
	sort.Slice(payload.Documents, func(a, b int) bool {
		return payload.Documents[a].MD5 < payload.Documents[b].MD5
	})
	s.mu.RUnlock()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

// Load reads an index written by Save. A missing file yields an empty store.
func Load(path string) (*Store, error) {
	s := New()
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return nil, err
	}

	var payload struct {
		Embedder  string           `json:"embedder,omitempty"`
		Dims      int              `json:"dims,omitempty"`
		Chunks    []storedChunk    `json:"chunks"`
		Documents []DocumentRecord `json:"documents,omitempty"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}

	chunks := make([]Chunk, 0, len(payload.Chunks))
	vectors := make([][]float32, 0, len(payload.Chunks))
	for i, item := range payload.Chunks {
		chunks = append(chunks, item.Chunk)
		if item.Vector == "" {
			vectors = append(vectors, nil)
			continue
		}
		vector, decodeErr := decodeVector(item.Vector)
		if decodeErr != nil {
			return nil, fmt.Errorf("store: chunk %d: %w", i, decodeErr)
		}
		vectors = append(vectors, vector)
	}

	if _, err := s.AddEmbedded(chunks, vectors); err != nil {
		return nil, err
	}
	for _, record := range payload.Documents {
		s.documents[record.MD5] = record
	}
	s.embedder = payload.Embedder
	if payload.Dims > 0 {
		s.dims = payload.Dims
	}
	return s, nil
}
