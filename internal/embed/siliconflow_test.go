package embed

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// fakeEmbeddings serves an OpenAI-compatible embeddings endpoint, recording the
// requests it received. A mutex guards the recording because the handler runs on
// the server's goroutine while the test reads the slice.
func fakeEmbeddings(t *testing.T, dims int, handler func(req embeddingsRequest, call int) (int, any)) (*httptest.Server, *[]embeddingsRequest, *int32) {
	t.Helper()
	var (
		calls    int32
		seen     []embeddingsRequest
		seenLock sync.Mutex
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}

		var request embeddingsRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		seenLock.Lock()
		seen = append(seen, request)
		seenLock.Unlock()

		call := int(atomic.AddInt32(&calls, 1))
		status, payload := handler(request, call)
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(payload)
	}))
	t.Cleanup(server.Close)
	return server, &seen, &calls
}

func vectorResponse(dims int, indices []int) map[string]any {
	data := make([]map[string]any, 0, len(indices))
	for _, index := range indices {
		vector := make([]float32, dims)
		for i := range vector {
			vector[i] = float32(index) + float32(i)/float32(dims)
		}
		data = append(data, map[string]any{"embedding": vector, "index": index})
	}
	return map[string]any{"object": "list", "data": data}
}

func TestSiliconFlowEmbedReturnsVectorsInInputOrder(t *testing.T) {
	server, seen, _ := fakeEmbeddings(t, 8, func(req embeddingsRequest, _ int) (int, any) {
		return http.StatusOK, vectorResponse(8, []int{0, 1, 2})
	})

	client := &SiliconFlow{BaseURL: server.URL, APIKey: "test-key", Model: "BAAI/bge-m3", Dims: 8}
	vectors, err := client.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != 3 || len(vectors[0]) != 8 {
		t.Fatalf("vectors = %dx%d, want 3x8", len(vectors), len(vectors[0]))
	}
	if (*seen)[0].EncodingFormat != "float" {
		t.Fatalf("encoding_format = %q", (*seen)[0].EncodingFormat)
	}
}

// The endpoint is not contractually ordered, so the client must sort by index.
func TestSiliconFlowEmbedReordersByIndex(t *testing.T) {
	server, _, _ := fakeEmbeddings(t, 4, func(req embeddingsRequest, _ int) (int, any) {
		return http.StatusOK, vectorResponse(4, []int{2, 0, 1})
	})

	client := &SiliconFlow{BaseURL: server.URL, APIKey: "test-key", Dims: 4}
	vectors, err := client.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	// Each fake vector starts with its own input index, so reordering is provable.
	for want, vector := range vectors {
		if vector[0] != float32(want) {
			t.Fatalf("vector %d starts with %v, want %d", want, vector[0], want)
		}
	}
}

func TestSiliconFlowEmbedBatchesRequests(t *testing.T) {
	server, seen, calls := fakeEmbeddings(t, 4, func(req embeddingsRequest, _ int) (int, any) {
		indices := make([]int, len(req.Input))
		for i := range indices {
			indices[i] = i
		}
		return http.StatusOK, vectorResponse(4, indices)
	})

	client := &SiliconFlow{BaseURL: server.URL, APIKey: "test-key", Dims: 4, Batch: 32}
	texts := make([]string, 70)
	for i := range texts {
		texts[i] = fmt.Sprintf("text %d", i)
	}
	vectors, err := client.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != 70 {
		t.Fatalf("vectors = %d, want 70", len(vectors))
	}
	if got := atomic.LoadInt32(calls); got != 3 {
		t.Fatalf("calls = %d, want 3 (32+32+6)", got)
	}
	if len((*seen)[2].Input) != 6 {
		t.Fatalf("last batch = %d inputs, want 6", len((*seen)[2].Input))
	}
	// Results must stay aligned across batch boundaries.
	for index, vector := range vectors {
		if vector[0] != float32(index%32) {
			t.Fatalf("vector %d misaligned across batches: %v", index, vector[0])
		}
	}
}

func TestSiliconFlowEmbedTruncatesLongInput(t *testing.T) {
	server, seen, _ := fakeEmbeddings(t, 4, func(req embeddingsRequest, _ int) (int, any) {
		return http.StatusOK, vectorResponse(4, []int{0})
	})

	client := &SiliconFlow{BaseURL: server.URL, APIKey: "test-key", Dims: 4, MaxChars: 10}
	if _, err := client.Embed(context.Background(), []string{strings.Repeat("字", 50)}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if got := len([]rune((*seen)[0].Input[0])); got != 10 {
		t.Fatalf("input runes = %d, want 10", got)
	}
}

func TestSiliconFlowEmbedSurfacesAPIErrors(t *testing.T) {
	server, _, _ := fakeEmbeddings(t, 4, func(req embeddingsRequest, _ int) (int, any) {
		return http.StatusUnauthorized, map[string]any{
			"error": map[string]any{"message": "invalid token", "code": "401"},
		}
	})

	client := &SiliconFlow{BaseURL: server.URL, APIKey: "test-key", Dims: 4}
	_, err := client.Embed(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "invalid token") {
		t.Fatalf("error = %v, want it to carry the API message", err)
	}
}

func TestSiliconFlowEmbedRejectsDimensionMismatch(t *testing.T) {
	// The index was built for one model; a different width means the vectors are
	// not comparable, so this must fail loudly rather than store garbage.
	server, _, _ := fakeEmbeddings(t, 512, func(req embeddingsRequest, _ int) (int, any) {
		return http.StatusOK, vectorResponse(512, []int{0})
	})

	client := &SiliconFlow{BaseURL: server.URL, APIKey: "test-key", Dims: 1024}
	_, err := client.Embed(context.Background(), []string{"a"})
	if err == nil || !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("error = %v, want a dimension mismatch", err)
	}
}

func TestSiliconFlowEmbedRequiresKeyAndHandlesEmptyInput(t *testing.T) {
	client := &SiliconFlow{BaseURL: "http://127.0.0.1:1", Dims: 4}
	if _, err := client.Embed(context.Background(), []string{"a"}); err == nil {
		t.Fatal("expected an error without an API key")
	}

	withKey := &SiliconFlow{BaseURL: "http://127.0.0.1:1", APIKey: "test-key", Dims: 4}
	vectors, err := withKey.Embed(context.Background(), nil)
	if err != nil || vectors != nil {
		t.Fatalf("empty input should be a no-op, got %v / %v", vectors, err)
	}
}

func TestSiliconFlowReportsPartialBatchFailure(t *testing.T) {
	server, _, _ := fakeEmbeddings(t, 4, func(req embeddingsRequest, call int) (int, any) {
		if call == 2 {
			return http.StatusTooManyRequests, map[string]any{"message": "rate limited"}
		}
		indices := make([]int, len(req.Input))
		for i := range indices {
			indices[i] = i
		}
		return http.StatusOK, vectorResponse(4, indices)
	})

	client := &SiliconFlow{BaseURL: server.URL, APIKey: "test-key", Dims: 4, Batch: 2}
	_, err := client.Embed(context.Background(), []string{"a", "b", "c", "d"})
	if err == nil {
		t.Fatal("expected an error: a partial result would leave chunks silently unembedded")
	}
	if !strings.Contains(err.Error(), "inputs 2-3") {
		t.Fatalf("error = %v, want it to name the failing batch", err)
	}
}

func TestSiliconFlowDefaults(t *testing.T) {
	client := &SiliconFlow{}
	if client.Dimensions() != DefaultDims {
		t.Fatalf("dims = %d", client.Dimensions())
	}
	if !strings.Contains(client.Name(), "bge-m3") {
		t.Fatalf("name = %q", client.Name())
	}
	if client.baseURL() != DefaultSiliconFlowURL {
		t.Fatalf("baseURL = %q", client.baseURL())
	}
}
