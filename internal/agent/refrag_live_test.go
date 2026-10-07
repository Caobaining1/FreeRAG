package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"freerag/internal/store"
)

// Live measurement of the REFRAG-style compression against the real generator.
//
// Two levels, deliberately separated:
//
//   - The size reduction is asserted offline, on the eval corpus, because it is
//     deterministic: the same pool renders the same block every time, and a
//     claim about prompt size needs no model to verify.
//   - The latency effect is measured only when FREERAG_LIVE_OLLAMA is set,
//     because it needs a running Ollama. It is skipped otherwise so `go test
//     ./...` stays offline — a suite that silently needs a service is a suite
//     that fails for the wrong reason.
//
// The measurement holds everything but the evidence block constant: same
// question, same generator, same options, same evidence POOL. Only the
// rendering differs. That is what isolates the prefill the compression removes
// from the retrieval, planning and sampling that would otherwise swamp it —
// end to end, a question costs 90–300 s here of which prefill is a minority.
func TestRefragCompressionOnTheEvalCorpus(t *testing.T) {
	pool := evalPool(t, 20)
	if len(pool) < 20 {
		t.Skipf("eval corpus holds %d chunk(s)", len(pool))
	}

	cfg := RefragConfig{Enabled: true, ExpandTop: DefaultRefragExpandTop, GistChars: DefaultRefragGistChars}
	for _, size := range []int{6, 10, 20} {
		hits := pool[:size]
		blockFull, full := renderEvidenceWith("What did the article report?", hits,
			PromptCharBudget(DefaultContextTokens), "", RefragConfig{})
		blockCompressed, compressed := renderEvidenceWith("What did the article report?", hits,
			PromptCharBudget(DefaultContextTokens), "", cfg)

		t.Logf("pool %2d: %6d chars -> %6d chars (%.2fx), %2d expanded / %2d compressed",
			size, full.EvidenceChars, compressed.EvidenceChars, compressed.Ratio(),
			compressed.Expanded, compressed.Compressed)

		// The rank policy always expands its top ExpandTop passages, whatever
		// the pool size: that is what makes the ratio improve with the pool
		// rather than with the passage lengths.
		if compressed.Expanded < cfg.ExpandTop {
			t.Fatalf("pool %d: only %d passage(s) expanded, want at least %d",
				size, compressed.Expanded, cfg.ExpandTop)
		}
		if size > cfg.ExpandTop && compressed.Compressed == 0 {
			t.Fatalf("pool %d: nothing was compressed", size)
		}
		if compressed.EvidenceChars > full.EvidenceChars {
			t.Fatalf("pool %d: compression made the block larger (%d > %d)",
				size, compressed.EvidenceChars, full.EvidenceChars)
		}
		// The passages the budget dropped are still a real difference: a
		// smaller block means the tail of the pool survives instead of being
		// cut, which is REFRAG's context-extension benefit and the reason the
		// ratio alone understates what changed.
		t.Logf("pool %2d: %d/%d passage(s) kept in full, %d/%d kept compressed",
			size, strings.Count(blockFull, "\n["), size,
			strings.Count(blockCompressed, "\n["), size)
	}

	// With a pool deeper than the expansion budget the effect must be large
	// enough to matter: below 1.5x the prompt is essentially unchanged and the
	// feature is not worth its information loss.
	_, compressed := renderEvidenceWith("q", pool[:20],
		PromptCharBudget(DefaultContextTokens), "", cfg)
	if compressed.Ratio() < 1.5 {
		t.Fatalf("compression ratio %.2f on a 20-passage pool is too small to matter",
			compressed.Ratio())
	}
}

// TestRefragLiveTTFT measures the first-token latency of the two prompts.
//
// Run:
//
//	FREERAG_LIVE_OLLAMA=1 go test ./internal/agent -run TestRefragLiveTTFT -v
func TestRefragLiveTTFT(t *testing.T) {
	if os.Getenv("FREERAG_LIVE_OLLAMA") == "" {
		t.Skip("set FREERAG_LIVE_OLLAMA=1 to measure against a running Ollama")
	}
	hits := evalPool(t, 10)
	if len(hits) < 10 {
		t.Skipf("eval corpus holds %d chunk(s)", len(hits))
	}

	model := &OllamaModel{
		BaseURL:     os.Getenv("FREERAG_OLLAMA_URL"),
		Model:       envOr("FREERAG_MODEL", DefaultGeneratorModel),
		NumCtx:      DefaultContextTokens,
		NumPredict:  256,
		Temperature: 0,
		Think:       boolPtr(false),
		KeepAlive:   "30m",
		Timeout:     5 * time.Minute,
	}

	const question = "What did the article report?"
	cfg := RefragConfig{Enabled: true, ExpandTop: DefaultRefragExpandTop, GistChars: DefaultRefragGistChars}

	// Warm the model FIRST, and outside the measurement. The weights load in
	// ~0.5 s but the first call also builds the execution plan and the prompt
	// cache: measured back to back, whichever arm ran first would pay for both
	// and the comparison would report the warm-up as the second arm's win.
	if _, err := model.Complete(context.Background(), []Message{
		{Role: RoleUser, Content: "warm-up"},
	}, nil); err != nil {
		t.Skipf("Ollama is not serving %q: %v", model.Model, err)
	}

	type variant struct {
		name string
		cfg  RefragConfig
	}
	variants := []variant{{"full", RefragConfig{}}, {"refrag", cfg}}

	type sample struct {
		ttft   time.Duration
		total  time.Duration
		answer int
	}
	// Alternating, two passes, best-of: the reference machine's spread between
	// two identical runs is large enough (docs/performance.md) that a single
	// pair cannot resolve a change, and measuring the same arm's two passes
	// gives the noise floor to read the difference against.
	best := map[string]sample{}
	blocks := map[string]string{}
	stats := map[string]PromptStats{}
	for pass := 1; pass <= 2; pass++ {
		for _, v := range variants {
			block, blockStats := renderEvidenceWith(question, hits, PromptCharBudget(DefaultContextTokens), "", v.cfg)
			messages := []Message{
				{Role: RoleSystem, Content: answerSystemPrompt},
				{Role: RoleUser, Content: block},
			}

			var (
				first time.Duration
				start = time.Now()
			)
			reply, err := model.CompleteStream(context.Background(), messages, nil, func(string) {
				if first == 0 {
					first = time.Since(start)
				}
			})
			total := time.Since(start)
			if err != nil {
				t.Fatalf("%s: %v", v.name, err)
			}
			got := sample{ttft: first, total: total, answer: len([]rune(reply.Content))}
			t.Logf("pass %d %-7s evidence %6d chars  TTFT %6.2fs  total %6.2fs  answer %4d chars",
				pass, v.name, blockStats.EvidenceChars, first.Seconds(), total.Seconds(), got.answer)

			if previous, ok := best[v.name]; !ok || got.ttft < previous.ttft {
				best[v.name] = got
			}
			blocks[v.name], stats[v.name] = block, blockStats
		}
	}

	full, compressed := best["full"], best["refrag"]
	t.Logf("best-of-2: full TTFT %.2fs -> refrag TTFT %.2fs (%+.2fs, %.2fx); total %.2fs -> %.2fs",
		full.ttft.Seconds(), compressed.ttft.Seconds(),
		(compressed.ttft - full.ttft).Seconds(),
		full.ttft.Seconds()/compressed.ttft.Seconds(),
		full.total.Seconds(), compressed.total.Seconds())

	// The prompt must actually be smaller, or the latency numbers above are
	// measuring something else.
	if len(blocks["refrag"]) >= len(blocks["full"]) {
		t.Fatalf("the compressed prompt is not smaller: %d vs %d bytes",
			len(blocks["refrag"]), len(blocks["full"]))
	}
	if stats["full"].Compressed != 0 {
		t.Fatalf("the baseline arm compressed %d passage(s)", stats["full"].Compressed)
	}
}

// evalPool loads real chunks from the frozen eval knowledge base.
//
// Real passages rather than generated ones: the compression's ratio depends on
// how long a passage is and where its first sentence ends, so a fixture of
// uniform 500-character paragraphs would measure the fixture.
func evalPool(t *testing.T, n int) []store.Hit {
	t.Helper()
	path := filepath.Join("..", "..", "eval", "data-hybrid", "kbs", "417629bda48bdd6c", "index.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var decoded struct {
		Chunks []store.Chunk `json:"chunks"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	if n > len(decoded.Chunks) {
		n = len(decoded.Chunks)
	}
	hits := make([]store.Hit, 0, n)
	for i := 0; i < n; i++ {
		chunk := decoded.Chunks[i]
		if strings.TrimSpace(chunk.Text) == "" {
			continue
		}
		// Descending scores, because the sense step expands by rank and a pool
		// with no order would make the policy meaningless.
		hits = append(hits, store.Hit{Chunk: chunk, Score: float64(n - i)})
	}
	return hits
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func boolPtr(value bool) *bool { return &value }
