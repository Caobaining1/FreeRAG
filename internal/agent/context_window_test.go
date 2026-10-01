package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"freerag/internal/store"
)

// The context window must be requested explicitly: Ollama's default is 4096,
// smaller than the evidence block the loop assembles, and an oversized prompt is
// rejected with HTTP 400 rather than truncated. This was hit for real — a
// two-round question built a 7647-token prompt and the answer silently degraded
// to the extractive fallback.
func TestOllamaCompleteSendsNumCtx(t *testing.T) {
	var received ollamaChatRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]string{"role": "assistant", "content": "ok"},
		})
	}))
	defer server.Close()

	model := &OllamaModel{BaseURL: server.URL, Model: "freerag-qwen3", NumCtx: 8192}
	if _, err := model.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if received.Options["num_ctx"] != float64(8192) {
		t.Fatalf("num_ctx = %#v, want 8192", received.Options["num_ctx"])
	}

	// Unset means "leave the server default alone".
	received = ollamaChatRequest{}
	bare := &OllamaModel{BaseURL: server.URL, Model: "freerag-qwen3"}
	if _, err := bare.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if _, present := received.Options["num_ctx"]; present {
		t.Fatalf("num_ctx must be omitted when unset, got %#v", received.Options["num_ctx"])
	}
}

func TestPromptCharBudget(t *testing.T) {
	budget := PromptCharBudget(DefaultContextTokens)
	if budget <= 0 {
		t.Fatalf("budget = %d, want > 0", budget)
	}
	// The budget must leave room for the system prompt and the answer, so it has
	// to stay well below the window expressed in characters.
	if budget >= DefaultContextTokens*charsPerToken {
		t.Fatalf("budget %d leaves no room for the prompt or the answer", budget)
	}
	if PromptCharBudget(16384) <= budget {
		t.Fatal("a larger window must allow more characters")
	}
	if PromptCharBudget(999999) <= PromptCharBudget(16384) {
		t.Fatal("the budget must keep scaling with the window")
	}
	if small := PromptCharBudget(256); small <= 0 {
		t.Fatalf("budget for a tiny window = %d, want > 0", small)
	}
	if PromptCharBudget(0) != budget {
		t.Fatal("a zero window must fall back to the default")
	}
}

func manyHits(count int) []store.Hit {
	hits := make([]store.Hit, 0, count)
	for i := 0; i < count; i++ {
		hits = append(hits, store.Hit{Chunk: store.Chunk{
			ChunkID: fmt.Sprintf("c%d", i),
			DocID:   "paper.pdf",
			PageNum: i + 1,
			Text:    strings.Repeat("word ", 200),
		}})
	}
	return hits
}

func TestRenderEvidenceWithoutBudgetKeepsEveryPassage(t *testing.T) {
	hits := manyHits(10)
	rendered := renderEvidence("why?", hits, 0, "")

	// Passages are numbered rather than identified by chunk id, so the citation
	// markers are what must all be present.
	for index := range hits {
		if !strings.Contains(rendered, fmt.Sprintf("[%d]", index+1)) {
			t.Fatalf("unbounded render dropped passage %d", index+1)
		}
	}
	if strings.Contains(rendered, "omitted") {
		t.Fatal("an unbounded render must not claim omissions")
	}
}

func TestRenderEvidenceTruncatesToBudget(t *testing.T) {
	hits := manyHits(10)
	unbounded := renderEvidence("why?", hits, 0, "")
	bounded := renderEvidence("why?", hits, 1500, "")

	// The budget bounds the evidence block; only the omission note may push the
	// total slightly past it.
	if len(bounded) > 1500+150 {
		t.Fatalf("bounded render = %d chars for a 1500-char budget", len(bounded))
	}
	if len(bounded) >= len(unbounded)/3 {
		t.Fatalf("bounded %d vs unbounded %d: the budget is not biting", len(bounded), len(unbounded))
	}
	// The prompt still has to be usable: the question, the first passage and an
	// honest note about what was cut.
	if !strings.Contains(bounded, "Question: why?") {
		t.Fatal("the question must survive the budget")
	}
	if !strings.Contains(bounded, "[1]") {
		t.Fatal("at least one passage must survive the budget")
	}
	if !strings.Contains(bounded, "omitted to fit the context window") {
		t.Fatal("a truncated render must state that passages were omitted")
	}
	if !strings.Contains(bounded, "…") {
		t.Fatal("the last passage should be trimmed, not dropped")
	}
}

func TestRenderEvidenceStopsWhenNoRoomRemains(t *testing.T) {
	// A budget too small for a single passage yields the header and the note,
	// never a truncated stub that would mislead the model.
	rendered := renderEvidence("why?", manyHits(3), 40, "")
	if !strings.Contains(rendered, "Question: why?") {
		t.Fatalf("rendered = %q", rendered)
	}
	if strings.Contains(rendered, "[1]") {
		t.Fatalf("a 40-char budget cannot host a passage: %q", rendered)
	}
}

func TestExtractiveDraftStillWorksWithoutAModel(t *testing.T) {
	hits := manyHits(3)
	answer := extractiveAnswer(hits)
	if !strings.Contains(answer, "[1]") || !strings.Contains(answer, "[2]") {
		t.Fatalf("extractive answer = %q", answer)
	}
	if strings.Contains(answer, "\n\n") {
		t.Fatal("extractive passages must collapse onto one line each")
	}
}
