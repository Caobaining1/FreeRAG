// Package dense holds external dense retrievers: vector databases that own the
// vectors and answer nearest-neighbour queries through an approximate index.
//
// They implement store.DenseIndex, so attaching one replaces the in-process
// linear scan without changing the retrieval interface. BM25, grep and the RRF
// fusion stay in the kernel — this package is only the dense leg.
package dense

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"freerag/internal/store"
)

// Defaults for a local Qdrant.
const (
	DefaultURL        = "http://127.0.0.1:6333"
	DefaultCollection = "freerag"
	// DefaultM and DefaultEfConstruct were picked by measurement, not taste:
	// against exact search on 20k vectors, M=32/ef_construct=100 returned
	// recall@10 = 1.000 at ~1.7ms per query. M=16 was already at 0.970, so 32
	// buys margin for roughly half the graph memory.
	DefaultM           = 32
	DefaultEfConstruct = 100
	// DefaultEfSearch is the query-time breadth. Higher only trades latency for
	// recall, and only when the corpus is big enough to be indexed at all.
	DefaultEfSearch = 128
	// uploadBatch keeps one request body in the low single-digit megabytes.
	uploadBatch = 256
	// DefaultTimeout bounds one request; a local index answers in milliseconds.
	DefaultTimeout = 30 * time.Second
)

// Qdrant is a dense index backed by a Qdrant collection.
type Qdrant struct {
	BaseURL    string
	Collection string
	// Dims is the vector width the collection was created with.
	Dims        int
	M           int
	EfConstruct int
	EfSearch    int
	Timeout     time.Duration
	HTTP        *http.Client
	// Logger receives the warnings that are not worth failing over. nil means
	// log.Default().
	Logger *log.Logger
}

// loggerOr returns the configured logger, or the process default.
func (q *Qdrant) loggerOr() *log.Logger {
	if q.Logger != nil {
		return q.Logger
	}
	return log.Default()
}

// FromEnv builds a Qdrant index from the environment, or returns (nil, nil)
// when it is not configured.
//
// A missing configuration is not a failure — keyword-only retrieval is a valid
// degraded mode — but a *broken* one is, because silently continuing would
// leave the operator believing dense retrieval is on.
func FromEnv(dims int) (*Qdrant, error) {
	return FromEnvCollection(dims, envStr("FREERAG_QDRANT_COLLECTION", DefaultCollection))
}

// FromEnvCollection is FromEnv with an explicit collection name.
//
// Each knowledge base gets its own collection (internal/kb), and that is what
// makes the isolation structural rather than a convention: a shared collection
// would need a payload filter on every query and every delete, and the first
// call site that forgot one would return another base's passages — a leak whose
// output still looks like a plausible answer, so nothing downstream would flag
// it.
//
// A nil client with a nil error means Qdrant is not configured at all. That is a
// supported state, not a failure: dense ranking then runs in-process.
func FromEnvCollection(dims int, collection string) (*Qdrant, error) {
	url := os.Getenv("FREERAG_QDRANT_URL")
	if url == "" {
		return nil, nil
	}
	if dims <= 0 {
		return nil, fmt.Errorf("dense: FREERAG_QDRANT_URL is set but the embedder provides no dimensions")
	}

	client := &Qdrant{
		BaseURL:     url,
		Collection:  collection,
		Dims:        dims,
		M:           envInt("FREERAG_QDRANT_M", DefaultM),
		EfConstruct: envInt("FREERAG_QDRANT_EF_CONSTRUCT", DefaultEfConstruct),
		EfSearch:    envInt("FREERAG_QDRANT_EF_SEARCH", DefaultEfSearch),
		Timeout:     DefaultTimeout,
	}
	if err := client.EnsureCollection(); err != nil {
		return nil, err
	}
	return client, nil
}

func envStr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envInt(name string, fallback int) int {
	if value, err := strconv.Atoi(os.Getenv(name)); err == nil && value > 0 {
		return value
	}
	return fallback
}

// Backend implements store.DenseIndex.
func (q *Qdrant) Backend() string { return "qdrant" }

func (q *Qdrant) baseURL() string {
	if q.BaseURL != "" {
		return q.BaseURL
	}
	return DefaultURL
}

func (q *Qdrant) collection() string {
	if q.Collection != "" {
		return q.Collection
	}
	return DefaultCollection
}

func (q *Qdrant) client() *http.Client {
	if q.HTTP != nil {
		return q.HTTP
	}
	return &http.Client{Timeout: q.timeout()}
}

func (q *Qdrant) timeout() time.Duration {
	if q.Timeout > 0 {
		return q.Timeout
	}
	return DefaultTimeout
}

// do issues one request. Qdrant is strict about methods: creating a collection
// and upserting points are PUT, search is POST, delete is DELETE.
func (q *Qdrant) do(method, path string, body any) (map[string]any, int, error) {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		reader = bytes.NewReader(raw)
	}

	request, err := http.NewRequest(method, q.baseURL()+path, reader)
	if err != nil {
		return nil, 0, err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := q.client().Do(request)
	if err != nil {
		return nil, 0, err
	}
	defer response.Body.Close()
	payload, _ := io.ReadAll(response.Body)

	if response.StatusCode >= 300 {
		snippet := payload
		if len(snippet) > 300 {
			snippet = snippet[:300]
		}
		return nil, response.StatusCode, fmt.Errorf("qdrant %s %s -> HTTP %d: %s",
			method, path, response.StatusCode, string(snippet))
	}

	out := map[string]any{}
	if len(payload) > 0 {
		if err := json.Unmarshal(payload, &out); err != nil {
			return nil, response.StatusCode, err
		}
	}
	return out, response.StatusCode, nil
}

// EnsureCollection creates the collection when absent and verifies its width
// when present.
//
// A width mismatch is refused rather than accepted: vectors from two different
// models are numerically comparable and semantically unrelated, so writing them
// into one collection would return confident nonsense that nothing downstream
// could detect.
func (q *Qdrant) EnsureCollection() error {
	path := "/collections/" + q.collection()

	response, status, err := q.do(http.MethodGet, path, nil)
	if err != nil && status != http.StatusNotFound {
		return fmt.Errorf("dense: qdrant is unreachable at %s: %w", q.baseURL(), err)
	}
	if status == 0 || status >= 300 {
		// Absent: create it.
		_, _, err := q.do(http.MethodPut, path, map[string]any{
			"vectors": map[string]any{"size": q.Dims, "distance": "Cosine"},
			"hnsw_config": map[string]any{
				"m":            q.M,
				"ef_construct": q.EfConstruct,
			},
		})
		if err != nil {
			return fmt.Errorf("dense: creating collection %q: %w", q.collection(), err)
		}
		return nil
	}

	size := configSize(response)
	if size != 0 && size != q.Dims {
		return fmt.Errorf(
			"dense: collection %q holds %d-dimensional vectors but the embedder provides %d",
			q.collection(), size, q.Dims)
	}

	// Ensured for a collection that already existed as much as for one just
	// created. A knowledge base indexed before this ran has no payload index,
	// and nothing else would ever add one to it: it would keep scanning, which
	// is exactly the case this is here to fix.
	q.ensurePayloadIndexes()
	return nil
}

// payloadIndexFields are the payload fields a request filters on.
//
// doc_id, because that is what a document scope filters by (SearchScoped) and
// what Delete drops by. Neither names chunk_id, so it is not indexed: an index
// nothing queries is storage and write cost for nothing.
var payloadIndexFields = []string{"doc_id"}

// ensurePayloadIndexes creates the payload indexes a filtered request needs.
//
// Without one, qdrant answers a doc_id filter by examining every point, which
// turns a scope from a narrowing into a slowdown — the opposite of what it is
// for. With one, the filter is an index lookup and the search only visits the
// scoped points.
//
// A failure here is logged and swallowed rather than returned. The filter still
// works without the index (it is just slow), so this is a performance cliff and
// not a correctness one, and the realistic causes — a qdrant older than the
// /index endpoint, or a read-only context — are not reasons to bring dense
// retrieval down. The message says what the cost is, so it is not silent.
func (q *Qdrant) ensurePayloadIndexes() {
	for _, field := range payloadIndexFields {
		if _, _, err := q.do(http.MethodPut,
			fmt.Sprintf("/collections/%s/index", q.collection()),
			map[string]any{"field_name": field, "field_schema": "keyword"}); err != nil {
			q.loggerOr().Printf(
				"warning: could not create the %q payload index on collection %q (%v); "+
					"searches filtered by %s will scan every point",
				field, q.collection(), err, field)
		}
	}
}

// configSize reads the vector width out of a collection description.
func configSize(response map[string]any) int {
	result, _ := response["result"].(map[string]any)
	config, _ := result["config"].(map[string]any)
	params, _ := config["params"].(map[string]any)
	vectors, ok := params["vectors"].(map[string]any)
	if !ok {
		return 0
	}
	// Unnamed single-vector collections expose size directly; named ones nest
	// it under the vector name.
	if size, ok := vectors["size"].(float64); ok {
		return int(size)
	}
	for _, value := range vectors {
		if named, ok := value.(map[string]any); ok {
			if size, ok := named["size"].(float64); ok {
				return int(size)
			}
		}
	}
	return 0
}

// pointID derives a stable Qdrant point id from a chunk's identity.
//
// Qdrant accepts uint64 or UUID ids, not arbitrary strings, so the dedup key is
// hashed. A collision would make one chunk unreachable through the dense leg
// only — the chunk itself still lives in the store and stays searchable by BM25
// and grep — and at 64 bits the probability is negligible for any corpus this
// runs on.
func pointID(chunk store.Chunk) uint64 {
	hash := fnv.New64a()
	_, _ = hash.Write([]byte(chunk.DocID))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(chunk.ChunkID))
	return hash.Sum64()
}

// Upsert implements store.DenseIndex.
func (q *Qdrant) Upsert(chunks []store.Chunk, vectors [][]float32) error {
	if len(chunks) != len(vectors) {
		return fmt.Errorf("dense: %d vectors for %d chunks", len(vectors), len(chunks))
	}
	if len(chunks) == 0 {
		return nil
	}

	for from := 0; from < len(chunks); from += uploadBatch {
		to := from + uploadBatch
		if to > len(chunks) {
			to = len(chunks)
		}

		points := make([]map[string]any, 0, to-from)
		for i := from; i < to; i++ {
			if len(vectors[i]) != q.Dims {
				return fmt.Errorf("dense: vector %d has %d dimensions, collection uses %d",
					i, len(vectors[i]), q.Dims)
			}
			points = append(points, map[string]any{
				"id":      pointID(chunks[i]),
				"vector":  vectors[i],
				"payload": map[string]any{"doc_id": chunks[i].DocID, "chunk_id": chunks[i].ChunkID},
			})
		}

		if _, _, err := q.do(http.MethodPut,
			fmt.Sprintf("/collections/%s/points?wait=true", q.collection()),
			map[string]any{"points": points}); err != nil {
			return err
		}
	}
	return nil
}

// Search implements store.DenseIndex.
//
// A payload-less point is skipped rather than guessed at: the store resolves
// matches by document and chunk id, and a match without them cannot be
// resolved to anything.
func (q *Qdrant) Search(vector []float32, limit int) ([]store.DenseMatch, error) {
	return q.search(vector, limit, nil)
}

// SearchScoped implements store.ScopedDenseIndex: it restricts the nearest
// neighbour search itself, rather than letting the caller sift a global top-k.
//
// The restriction goes in the request body as a payload filter over doc_id,
// which is the field Upsert writes. Without it a scoped query over a corpus
// where the scoped document is not globally near the top would come back empty
// even though that document contains what was asked for.
func (q *Qdrant) SearchScoped(vector []float32, limit int, docIDs []string) ([]store.DenseMatch, error) {
	if len(docIDs) == 0 {
		return q.search(vector, limit, nil)
	}
	return q.search(vector, limit, map[string]any{
		// `any` rather than one condition per document: one clause keeps the
		// request the same shape whether the scope holds one file or four
		// hundred.
		"must": []any{map[string]any{
			"key":   "doc_id",
			"match": map[string]any{"any": docIDs},
		}},
	})
}

func (q *Qdrant) search(vector []float32, limit int, filter map[string]any) ([]store.DenseMatch, error) {
	if limit <= 0 {
		limit = 10
	}

	body := map[string]any{
		"vector":       vector,
		"limit":        limit,
		"with_payload": true,
		"params":       map[string]any{"hnsw_ef": q.EfSearch, "exact": false},
	}
	if filter != nil {
		body["filter"] = filter
	}

	response, _, err := q.do(http.MethodPost,
		fmt.Sprintf("/collections/%s/points/search", q.collection()),
		body)
	if err != nil {
		return nil, err
	}

	raw, _ := response["result"].([]any)
	matches := make([]store.DenseMatch, 0, len(raw))
	for _, item := range raw {
		point, _ := item.(map[string]any)
		payload, _ := point["payload"].(map[string]any)
		docID, _ := payload["doc_id"].(string)
		chunkID, _ := payload["chunk_id"].(string)
		if docID == "" || chunkID == "" {
			continue
		}
		score, _ := point["score"].(float64)
		matches = append(matches, store.DenseMatch{DocID: docID, ChunkID: chunkID, Score: score})
	}
	return matches, nil
}

// Delete implements store.DenseIndex, dropping one document's points by payload
// filter.
//
// A no-op on an empty document is intentional: re-indexing a changed file calls
// this before upserting, and "nothing to delete" is the normal case on a first
// index, not an error.
func (q *Qdrant) Delete(docID string) error {
	_, _, err := q.do(http.MethodPost,
		fmt.Sprintf("/collections/%s/points/delete?wait=true", q.collection()),
		map[string]any{"filter": map[string]any{
			"must": []any{map[string]any{
				"key":   "doc_id",
				"match": map[string]any{"value": docID},
			}},
		}})
	return err
}

// DropCollection deletes a collection by name, without ensuring it exists first.
//
// Deliberately not FromEnvCollection followed by Drop: that path creates the
// collection when it is absent, so deleting a knowledge base that never had an
// index would create one in order to delete it. Worse, it would fail on a width
// mismatch — and a cleanup path that cannot run is a leak, which is the thing
// this exists to prevent.
//
// A collection that is already gone is not an error.
func DropCollection(name string) error {
	url := os.Getenv("FREERAG_QDRANT_URL")
	if url == "" {
		return nil
	}
	return (&Qdrant{BaseURL: url, Collection: name}).Drop()
}

// Drop deletes the whole collection.
//
// Used when a knowledge base is removed, so its vectors do not outlive it. A
// collection that was never created is not an error: deleting a base no document
// was ever added to is a normal outcome rather than a failure.
func (q *Qdrant) Drop() error {
	_, status, err := q.do(http.MethodDelete, "/collections/"+q.collection(), nil)
	if err != nil && status != http.StatusNotFound {
		return err
	}
	return nil
}

// Len implements store.DenseIndex.
func (q *Qdrant) Len() int {
	response, _, err := q.do(http.MethodGet, "/collections/"+q.collection(), nil)
	if err != nil {
		return 0
	}
	result, _ := response["result"].(map[string]any)
	if count, ok := result["points_count"].(float64); ok {
		return int(count)
	}
	return 0
}
