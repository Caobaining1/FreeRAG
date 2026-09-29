package dense

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"freerag/internal/store"
)

// record is one request the fake Qdrant saw.
type record struct {
	method string
	path   string
	body   map[string]any
}

// fakeQdrant serves the handful of endpoints the client uses and records every
// call, so the wire format can be asserted without a real server.
func fakeQdrant(t *testing.T, collectionBody string, status int) (*httptest.Server, *[]record) {
	t.Helper()
	calls := &[]record{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		entry := record{method: r.Method, path: r.URL.RequestURI()}
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &entry.body)
		}
		*calls = append(*calls, entry)

		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/collections/"):
			if collectionBody == "" {
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"status":{"error":"not found"}}`))
				return
			}
			w.WriteHeader(status)
			_, _ = w.Write([]byte(collectionBody))
		case strings.Contains(r.URL.Path, "/points/search"):
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"result":[]}`))
		default:
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"result":{"status":"completed"}}`))
		}
	}))
	t.Cleanup(server.Close)
	return server, calls
}

func client(server *httptest.Server, dims int) *Qdrant {
	return &Qdrant{
		BaseURL:     server.URL,
		Collection:  "freerag",
		Dims:        dims,
		M:           DefaultM,
		EfConstruct: DefaultEfConstruct,
		EfSearch:    DefaultEfSearch,
		HTTP:        server.Client(),
	}
}

func TestEnsureCollectionCreatesWhenAbsent(t *testing.T) {
	server, calls := fakeQdrant(t, "", http.StatusOK)

	if err := client(server, 1024).EnsureCollection(); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}

	var created map[string]any
	for _, call := range *calls {
		if call.method == http.MethodPut {
			created = call.body
		}
	}
	if created == nil {
		t.Fatalf("no PUT was issued; calls = %#v", *calls)
	}
	vectors, _ := created["vectors"].(map[string]any)
	if vectors["size"] != float64(1024) || vectors["distance"] != "Cosine" {
		t.Fatalf("vectors = %#v", vectors)
	}
	hnsw, _ := created["hnsw_config"].(map[string]any)
	if hnsw["m"] != float64(DefaultM) || hnsw["ef_construct"] != float64(DefaultEfConstruct) {
		t.Fatalf("hnsw_config = %#v", hnsw)
	}
}

func TestEnsureCollectionAcceptsAMatchingWidth(t *testing.T) {
	body := `{"result":{"config":{"params":{"vectors":{"size":1024,"distance":"Cosine"}}}}}`
	server, calls := fakeQdrant(t, body, http.StatusOK)

	if err := client(server, 1024).EnsureCollection(); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	for _, call := range *calls {
		if call.method == http.MethodPut {
			t.Fatalf("a matching collection must not be recreated: %#v", call)
		}
	}
}

func TestEnsureCollectionRefusesAWidthMismatch(t *testing.T) {
	// Vectors from two models are numerically comparable and semantically
	// unrelated, so sharing a collection would return confident nonsense.
	body := `{"result":{"config":{"params":{"vectors":{"size":768,"distance":"Cosine"}}}}}`
	server, _ := fakeQdrant(t, body, http.StatusOK)

	err := client(server, 1024).EnsureCollection()
	if err == nil {
		t.Fatal("expected a width mismatch to be refused")
	}
	if !strings.Contains(err.Error(), "768") || !strings.Contains(err.Error(), "1024") {
		t.Fatalf("error should name both widths, got %v", err)
	}
}

func TestUpsertSendsChunkIdentityAsPayload(t *testing.T) {
	server, calls := fakeQdrant(t, "", http.StatusOK)

	chunks := []store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"},
		{ChunkID: "c1", DocID: "a.pdf", Text: "beta"},
	}
	if err := client(server, 3).Upsert(chunks, [][]float32{{1, 0, 0}, {0, 1, 0}}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	var put map[string]any
	for _, call := range *calls {
		if call.method == http.MethodPut && strings.Contains(call.path, "/points") {
			put = call.body
		}
	}
	if put == nil {
		t.Fatalf("no points PUT was issued; calls = %#v", *calls)
	}
	points, _ := put["points"].([]any)
	if len(points) != 2 {
		t.Fatalf("sent %d points, want 2", len(points))
	}
	first, _ := points[0].(map[string]any)
	payload, _ := first["payload"].(map[string]any)
	// The store resolves matches by this payload, so it has to carry both ids.
	if payload["doc_id"] != "a.pdf" || payload["chunk_id"] != "c0" {
		t.Fatalf("payload = %#v", payload)
	}
	if vector, _ := first["vector"].([]any); len(vector) != 3 {
		t.Fatalf("vector = %#v", first["vector"])
	}
}

func TestUpsertRefusesTheWrongWidth(t *testing.T) {
	server, _ := fakeQdrant(t, "", http.StatusOK)

	err := client(server, 3).Upsert(
		[]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"}},
		[][]float32{{1, 0}},
	)
	if err == nil {
		t.Fatal("expected a width mismatch to be refused before upload")
	}
}

func TestSearchMapsPayloadsToMatches(t *testing.T) {
	body := `{"result":[
		{"id":11,"score":0.93,"payload":{"doc_id":"a.pdf","chunk_id":"c0"}},
		{"id":12,"score":0.80,"payload":{"doc_id":"a.pdf","chunk_id":"c1"}},
		{"id":13,"score":0.70,"payload":{}}
	]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)

	matches, err := client(server, 3).Search([]float32{1, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	// The payload-less point cannot be resolved to a chunk, so it is dropped
	// rather than guessed at.
	if len(matches) != 2 {
		t.Fatalf("matches = %#v, want 2", matches)
	}
	if matches[0].DocID != "a.pdf" || matches[0].ChunkID != "c0" || matches[0].Score != 0.93 {
		t.Fatalf("matches[0] = %#v", matches[0])
	}
}

func TestDeleteTargetsOneDocumentByPayload(t *testing.T) {
	server, calls := fakeQdrant(t, "", http.StatusOK)

	if err := client(server, 3).Delete("a.pdf"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	var sent map[string]any
	for _, call := range *calls {
		if call.method == http.MethodPost && strings.Contains(call.path, "/points/delete") {
			sent = call.body
		}
	}
	if sent == nil {
		t.Fatalf("no delete request was issued; calls = %#v", *calls)
	}

	filter, _ := sent["filter"].(map[string]any)
	must, _ := filter["must"].([]any)
	if len(must) != 1 {
		t.Fatalf("filter = %#v, want exactly one clause", filter)
	}
	clause, _ := must[0].(map[string]any)
	if clause["key"] != "doc_id" {
		t.Fatalf("clause key = %#v, want doc_id", clause["key"])
	}
	match, _ := clause["match"].(map[string]any)
	if match["value"] != "a.pdf" {
		t.Fatalf("match = %#v", match)
	}
}

func TestPointIDIsStableAndDistinct(t *testing.T) {
	a := pointID(store.Chunk{DocID: "a.pdf", ChunkID: "c0"})

	if a != pointID(store.Chunk{DocID: "a.pdf", ChunkID: "c0"}) {
		t.Fatal("pointID must be stable across calls, or re-indexing would duplicate points")
	}
	if a == pointID(store.Chunk{DocID: "a.pdf", ChunkID: "c1"}) {
		t.Fatal("different chunks must not share a point id")
	}
	if a == pointID(store.Chunk{DocID: "b.pdf", ChunkID: "c0"}) {
		t.Fatal("the document must take part in the id")
	}
	// The key must be unambiguous: concatenating without a separator would make
	// ("ab", "c") and ("a", "bc") collide onto one point.
	if pointID(store.Chunk{DocID: "ab", ChunkID: "c"}) == pointID(store.Chunk{DocID: "a", ChunkID: "bc"}) {
		t.Fatal("a separator is needed between document and chunk id")
	}
}

func TestFromEnvSkipsWhenUnconfigured(t *testing.T) {
	t.Setenv("FREERAG_QDRANT_URL", "")

	index, err := FromEnv(1024)
	if err != nil || index != nil {
		t.Fatalf("FromEnv = %#v, %v; want nil, nil", index, err)
	}
}

func TestFromEnvRequiresDimensions(t *testing.T) {
	t.Setenv("FREERAG_QDRANT_URL", "http://127.0.0.1:6333")

	// A collection cannot be created without a width, so this must fail loudly
	// rather than silently disabling dense retrieval.
	if _, err := FromEnv(0); err == nil {
		t.Fatal("expected an error when the width is unknown")
	}
}

func TestFromEnvReportsUnreachableServer(t *testing.T) {
	// A closed port: the kernel must learn the index is unusable at startup
	// rather than at the first query.
	t.Setenv("FREERAG_QDRANT_URL", "http://127.0.0.1:1")
	t.Setenv("FREERAG_QDRANT_COLLECTION", "freerag_test")

	if _, err := FromEnv(1024); err == nil {
		t.Fatal("expected an error for an unreachable server")
	}
}
