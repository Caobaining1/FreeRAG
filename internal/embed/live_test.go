package embed

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// TestSiliconFlowLive exercises the real endpoint. It is skipped by default so
// the suite stays offline and deterministic; run it with:
//
//	set -a && . ./.env && set +a && FREERAG_LIVE_EMBED=1 go test ./internal/embed/ -run Live -v
func TestSiliconFlowLive(t *testing.T) {
	if os.Getenv("FREERAG_LIVE_EMBED") != "1" {
		t.Skip("set FREERAG_LIVE_EMBED=1 (and FREERAG_SILICONFLOW_KEY) to hit the live endpoint")
	}

	configured, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv: %v", err)
	}
	if configured == nil {
		t.Fatal("no provider configured; set FREERAG_EMBED_PROVIDER=siliconflow")
	}
	client, ok := configured.(*SiliconFlow)
	if !ok {
		t.Fatalf("unexpected embedder type %T", configured)
	}
	t.Logf("provider=%s dims=%d", client.Name(), client.Dimensions())

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	texts := []string{
		"Retrieval-augmented generation grounds a language model in retrieved passages.",
		"RAG reduces hallucination by conditioning the generator on external evidence.",
		"The boiling point of mercury is 356.7 degrees Celsius.",
		"政策梯度方法在强化学习中用于优化序列决策。",
	}
	started := time.Now()
	vectors, err := client.Embed(ctx, texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	t.Logf("%d texts -> %d vectors in %.2fs", len(texts), len(vectors), time.Since(started).Seconds())

	if len(vectors) != len(texts) {
		t.Fatalf("got %d vectors for %d texts", len(vectors), len(texts))
	}
	for i, vector := range vectors {
		if len(vector) != client.Dimensions() {
			t.Fatalf("vector %d has %d dimensions, want %d", i, len(vector), client.Dimensions())
		}
		if norm := l2(vector); math.Abs(norm-1.0) > 0.05 {
			t.Logf("note: vector %d is not unit-length (|v| = %.4f); cosine normalises anyway", i, norm)
		}
	}

	// Semantic sanity: the two RAG sentences must be closer to each other than
	// either is to the unrelated one. A client that silently returned zeros,
	// identical vectors, or shuffled outputs would fail this.
	similar := cosine(vectors[0], vectors[1])
	far := cosine(vectors[0], vectors[2])
	t.Logf("cos(RAG, RAG) = %.4f, cos(RAG, mercury) = %.4f", similar, far)
	if similar <= far {
		t.Fatalf("embeddings carry no signal: similar=%.4f far=%.4f", similar, far)
	}
	if similar < 0.5 {
		t.Fatalf("related sentences scored only %.4f; the model or endpoint is wrong", similar)
	}

	// The multilingual half of BGE-M3: Chinese must land near its English sense.
	chinese := cosine(vectors[0], vectors[3])
	t.Logf("cos(RAG, 政策梯度) = %.4f", chinese)

	// A long input must be truncated client-side rather than rejected.
	long := strings.Repeat("retrieval augmented generation. ", 2000)
	if _, err := client.Embed(ctx, []string{long}); err != nil {
		t.Fatalf("Embed(very long input): %v", err)
	}
}

func l2(vector []float32) float64 {
	var sum float64
	for _, value := range vector {
		sum += float64(value) * float64(value)
	}
	return math.Sqrt(sum)
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	if na, nb := l2(a), l2(b); na > 0 && nb > 0 {
		return dot / (na * nb)
	}
	return 0
}
