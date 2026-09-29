package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestStripThink(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"plain", "The answer is 42.", "The answer is 42."},
		{"block", "<think>reasoning</think>The answer is 42.", "The answer is 42."},
		{"block with newlines", "<think>\nstep 1\nstep 2\n</think>\n\nDone [1].", "Done [1]."},
		{"unterminated", "<think>still reasoning", ""},
		{"unterminated after text", "Partial.<think>and more", "Partial."},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := StripThink(tc.in); got != tc.want {
				t.Fatalf("StripThink(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestOllamaForwardsThinkAndKeepAlive(t *testing.T) {
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

	off := false
	model := &OllamaModel{
		BaseURL:   server.URL,
		Model:     "freerag-qwen3",
		Think:     &off,
		KeepAlive: "30m",
	}
	if _, err := model.Complete(context.Background(),
		[]Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// An explicit false is the point: reasoning is generated against the same
	// num_predict budget as the answer, so leaving it on can consume the whole
	// budget and return empty content.
	if received.Think == nil {
		t.Fatal("think must be sent explicitly when set")
	}
	if *received.Think {
		t.Fatalf("think = %v, want false", *received.Think)
	}
	if received.KeepAlive != "30m" {
		t.Fatalf("keep_alive = %q, want 30m", received.KeepAlive)
	}
}

func TestOllamaOmitsThinkAndKeepAliveWhenUnset(t *testing.T) {
	var raw string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		raw = string(body)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]string{"role": "assistant", "content": "ok"},
		})
	}))
	defer server.Close()

	// Absent must stay absent. Sending false would silently disable reasoning
	// for a caller that never expressed an opinion, which is a behaviour change
	// dressed up as a default.
	if _, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).Complete(context.Background(),
		[]Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if strings.Contains(raw, "think") {
		t.Fatalf("body = %s, want no think field", raw)
	}
	if strings.Contains(raw, "keep_alive") {
		t.Fatalf("body = %s, want no keep_alive field", raw)
	}
}

func TestOllamaCompleteForwardsMessagesAndStripsReasoning(t *testing.T) {
	var received ollamaChatRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]string{
				"role":    "assistant",
				"content": "<think>pondering</think>Cited answer [1].",
			},
		})
	}))
	defer server.Close()

	model := &OllamaModel{BaseURL: server.URL, Model: "freerag-qwen3", NumPredict: 128, Temperature: 0.2}
	reply, err := model.Complete(context.Background(), []Message{
		{Role: RoleSystem, Content: "be brief"},
		{Role: RoleUser, Content: "what?"},
	}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if reply.Content != "Cited answer [1]." {
		t.Fatalf("content = %q", reply.Content)
	}
	if received.Model != "freerag-qwen3" {
		t.Fatalf("model = %q", received.Model)
	}
	if len(received.Messages) != 2 || received.Messages[0].Role != "system" || received.Messages[1].Content != "what?" {
		t.Fatalf("messages = %#v", received.Messages)
	}
	if received.Stream {
		t.Fatal("stream must be false: the loop reads one complete reply")
	}
	if received.Options["num_predict"] != float64(128) {
		t.Fatalf("num_predict = %#v", received.Options["num_predict"])
	}
}

func TestOllamaCompleteForwardsToolsAndParsesToolCalls(t *testing.T) {
	var received ollamaChatRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Errorf("decode request: %v", err)
		}
		// Call 0 uses Ollama's object-shaped arguments, call 1 the
		// OpenAI-compatible JSON string. Both conventions are in the wild.
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"","tool_calls":[
			{"id":"c1","function":{"name":"hybrid_search","arguments":{"query":"GRPO","k":3}}},
			{"function":{"name":"grep_search","arguments":"{\"pattern\":\"DAPO\"}"}}
		]}}`))
	}))
	defer server.Close()

	model := &OllamaModel{BaseURL: server.URL, Model: "freerag-qwen3"}
	reply, err := model.Complete(context.Background(), []Message{{Role: RoleUser, Content: "q"}}, ToolSpecs())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// The specs must actually reach the server, or the model is choosing from a
	// surface it cannot see.
	if len(received.Tools) != len(ToolNames()) {
		t.Fatalf("forwarded %d tool(s), want %d", len(received.Tools), len(ToolNames()))
	}
	if received.Tools[0].Type != "function" || received.Tools[0].Function.Name == "" {
		t.Fatalf("tool wire form = %#v", received.Tools[0])
	}
	if received.Tools[0].Function.Parameters == nil {
		t.Fatal("a tool without its parameters tells the model nothing about how to call it")
	}

	if len(reply.ToolCalls) != 2 {
		t.Fatalf("tool calls = %#v", reply.ToolCalls)
	}
	if reply.ToolCalls[0].ID != "c1" || reply.ToolCalls[0].Name != "hybrid_search" {
		t.Fatalf("call 0 = %#v", reply.ToolCalls[0])
	}
	if reply.ToolCalls[0].Arguments["query"] != "GRPO" {
		t.Fatalf("call 0 arguments = %#v", reply.ToolCalls[0].Arguments)
	}
	if reply.ToolCalls[1].Name != "grep_search" || reply.ToolCalls[1].Arguments["pattern"] != "DAPO" {
		t.Fatalf("call 1 = %#v (string-encoded arguments must decode)", reply.ToolCalls[1])
	}
}

func TestOllamaCompleteIgnoresNamelessToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"hi","tool_calls":[
			{"function":{"name":"   ","arguments":{}}},
			{"function":{"name":"grep_search","arguments":{}}}
		]}}`))
	}))
	defer server.Close()

	reply, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).
		Complete(context.Background(), nil, ToolSpecs())
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// A nameless call cannot be dispatched, and keeping it would put a call in
	// the trace that was never executable.
	if len(reply.ToolCalls) != 1 || reply.ToolCalls[0].Name != "grep_search" {
		t.Fatalf("tool calls = %#v", reply.ToolCalls)
	}
}

func TestOllamaCompleteOmitsToolsWhenNoneAreOffered(t *testing.T) {
	var received ollamaChatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&received)
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"ok"}}`))
	}))
	defer server.Close()

	if _, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).
		Complete(context.Background(), nil, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// A draft turn must not advertise tools: the model would be free to answer
	// with a call instead of the text the checker is waiting for.
	if len(received.Tools) != 0 {
		t.Fatalf("tools = %#v, want none", received.Tools)
	}
}

func TestDecodeArguments(t *testing.T) {
	if got := decodeArguments(nil); got != nil {
		t.Fatalf("nil should stay nil, got %#v", got)
	}
	if got := decodeArguments(json.RawMessage(`{"a":1}`)); got["a"] != float64(1) {
		t.Fatalf("object form = %#v", got)
	}
	if got := decodeArguments(json.RawMessage(`"{\"a\":1}"`)); got["a"] != float64(1) {
		t.Fatalf("string form = %#v", got)
	}
	// Unreadable arguments become nil so the tool reports its own missing
	// argument, which names the field — more useful than a decode error.
	if got := decodeArguments(json.RawMessage(`not json`)); got != nil {
		t.Fatalf("garbage = %#v, want nil", got)
	}
}

func TestOllamaCompleteSurfacesServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"model not found"}`))
	}))
	defer server.Close()

	model := &OllamaModel{BaseURL: server.URL, Model: "nope"}
	_, err := model.Complete(context.Background(), nil, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "model not found") {
		t.Fatalf("error = %v", err)
	}
}

func TestOllamaCompleteRequiresModel(t *testing.T) {
	if _, err := (&OllamaModel{}).Complete(context.Background(), nil, nil); err == nil {
		t.Fatal("expected an error when no model is configured")
	}
}

func TestOllamaReachable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]string{{"name": "freerag-qwen3:latest"}},
		})
	}))
	defer server.Close()

	model := &OllamaModel{BaseURL: server.URL, Model: "freerag-qwen3"}
	if !model.Reachable(context.Background()) {
		t.Fatal("model should be reported reachable")
	}

	missing := &OllamaModel{BaseURL: server.URL, Model: "absent"}
	if missing.Reachable(context.Background()) {
		t.Fatal("absent model should not be reported reachable")
	}

	offline := &OllamaModel{BaseURL: "http://127.0.0.1:1", Model: "freerag-qwen3"}
	if offline.Reachable(context.Background()) {
		t.Fatal("unreachable server should not be reported reachable")
	}
}
