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

func TestThinkFilter(t *testing.T) {
	// Every case is fed as a sequence of chunks, because that is the only thing
	// that distinguishes this from StripThink: where the server happens to split
	// the text must not change what the user sees.
	cases := []struct {
		name   string
		chunks []string
		want   string
	}{
		{"plain text passes through", []string{"hello", " world"}, "hello world"},
		{"a block in one chunk", []string{"a<think>hidden</think>b"}, "ab"},
		{"opening tag split across chunks", []string{"a<thi", "nk>hidden</think>b"}, "ab"},
		{"closing tag split across chunks", []string{"a<think>hidden</thi", "nk>b"}, "ab"},
		{"an ordinary angle bracket is not a tag", []string{"a <", " b"}, "a < b"},
		{"unterminated block is dropped", []string{"a<think>never closed"}, "a"},
		{"a tag split three ways", []string{"abc<", "thi", "nk>d</think>e"}, "abce"},
		{"empty chunks are harmless", []string{"", "a", "", "b", ""}, "ab"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var filter thinkFilter
			var out strings.Builder
			for _, chunk := range tc.chunks {
				out.WriteString(filter.write(chunk))
			}
			out.WriteString(filter.flush())

			if out.String() != tc.want {
				t.Fatalf("streamed %q, want %q", out.String(), tc.want)
			}
		})
	}
}

// Temperature 0 must reach the server.
//
// It is the value every measurement run asks for and the one the wire format is
// easiest to drop: "send it only when it is positive" reads as a sensible
// default and silently turns greedy decoding into the server's 0.8. The symptom
// is not an error — it is a run that looks reproducible and is not
// (docs/performance.md §10.3).
func TestOllamaSendsZeroTemperature(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"ok"}}`)
	}))
	defer server.Close()

	if _, err := (&OllamaModel{BaseURL: server.URL, Model: "m", Temperature: 0}).Complete(
		context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	options, _ := body["options"].(map[string]any)
	if value, ok := options["temperature"]; !ok || value != float64(0) {
		t.Fatalf("options = %v, want temperature 0 on the wire", options)
	}
}

// A negative temperature is how a caller asks for the server's own default.
func TestOllamaOmitsTemperatureWhenNegative(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"ok"}}`)
	}))
	defer server.Close()

	if _, err := (&OllamaModel{BaseURL: server.URL, Model: "m", Temperature: -1}).Complete(
		context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	options, _ := body["options"].(map[string]any)
	if _, ok := options["temperature"]; ok {
		t.Fatalf("options = %v, want no temperature", options)
	}
}

// The generator's own token counts and timings are carried on the Reply.
//
// Ollama repeats the RUNNING totals on every chunk, so a collector that summed
// them would report the count multiplied by the number of chunks — a wrong
// number that looks entirely plausible, which is the dangerous kind. The
// fixture below repeats them deliberately, as the server does.
func TestCompleteStreamCapturesGeneratorUsage(t *testing.T) {
	server, _ := streamServer(t,
		map[string]any{
			"message":           map[string]string{"role": "assistant", "content": "half "},
			"done":              false,
			"prompt_eval_count": 100, "prompt_eval_duration": 40_000_000,
			"eval_count": 5, "eval_duration": 500_000_000,
		},
		map[string]any{
			"message":           map[string]string{"role": "assistant", "content": "an answer"},
			"done":              true,
			"prompt_eval_count": 100, "prompt_eval_duration": 50_000_000,
			"eval_count": 12, "eval_duration": 1_500_000_000,
		},
	)

	reply, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).CompleteStream(
		context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if reply.Usage == nil {
		t.Fatal("usage was reported by the server but is not on the reply")
	}
	if reply.Usage.PromptTokens != 100 || reply.Usage.OutputTokens != 12 {
		t.Fatalf("usage = %+v, want the final running totals (100 prompt, 12 output)", *reply.Usage)
	}
	// The last chunk's timings win, not the first's: the reply is complete when
	// the stream ends, and its cost is the whole call.
	if reply.Usage.PromptNanos != 50_000_000 || reply.Usage.OutputNanos != 1_500_000_000 {
		t.Fatalf("timings = %+v, want the final chunk's", *reply.Usage)
	}

	line := reply.Usage.TraceLine()
	for _, want := range []string{"prompt 100 tok", "output 12 tok", "2000 tok/s", "8.0 tok/s"} {
		if !strings.Contains(line, want) {
			t.Fatalf("trace line %q does not carry %q", line, want)
		}
	}
}

// A server that reports no accounting leaves Usage nil rather than reporting a
// call that cost nothing: zero prompt tokens is a claim, not an absence.
func TestCompleteStreamLeavesUsageNilWhenTheServerReportsNone(t *testing.T) {
	server, _ := streamServer(t,
		map[string]any{"message": map[string]string{"role": "assistant", "content": "ok"}, "done": true})

	reply, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).Complete(
		context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if reply.Usage != nil {
		t.Fatalf("usage = %+v, want nil", *reply.Usage)
	}
}

// streamServer replies with the given chunks as newline-delimited JSON, which is
// how Ollama streams, and returns the server plus the decoded request.
func streamServer(t *testing.T, chunks ...map[string]any) (*httptest.Server, *ollamaChatRequest) {
	t.Helper()

	received := &ollamaChatRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(received); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		encoder := json.NewEncoder(w)
		for _, chunk := range chunks {
			if err := encoder.Encode(chunk); err != nil {
				t.Errorf("encode chunk: %v", err)
				return
			}
		}
	}))
	t.Cleanup(server.Close)
	return server, received
}

func TestCompleteStreamEmitsDeltasAndMatchesComplete(t *testing.T) {
	server, received := streamServer(t,
		map[string]any{"message": map[string]string{"role": "assistant", "content": "The answer "}, "done": false},
		map[string]any{"message": map[string]string{"role": "assistant", "content": "is 42 [1]."}, "done": false},
		map[string]any{"message": map[string]string{"role": "assistant", "content": ""}, "done": true},
	)

	var deltas []string
	reply, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).CompleteStream(
		context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil,
		func(delta string) { deltas = append(deltas, delta) })
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	if !received.Stream {
		t.Fatal("stream must be true, or the server sends one reply and nothing is incremental")
	}
	// The deltas have to assemble into the Reply the caller also receives: a UI
	// that renders them would otherwise show text the result does not contain.
	if joined := strings.Join(deltas, ""); joined != reply.Content {
		t.Fatalf("deltas join to %q but Content is %q", joined, reply.Content)
	}
	if reply.Content != "The answer is 42 [1]." {
		t.Fatalf("Content = %q", reply.Content)
	}
}

func TestCompleteStreamHidesReasoningSplitAcrossChunks(t *testing.T) {
	// The case thinkFilter exists for. A per-chunk StripThink matches neither
	// half of the tag, so the user would watch the model reason — which is
	// exactly what StripThink was added to prevent.
	server, _ := streamServer(t,
		map[string]any{"message": map[string]string{"content": "Before. <thi"}, "done": false},
		map[string]any{"message": map[string]string{"content": "nk>private reasoning</thi"}, "done": false},
		map[string]any{"message": map[string]string{"content": "nk>After."}, "done": false},
		map[string]any{"message": map[string]string{"content": ""}, "done": true},
	)

	var deltas []string
	reply, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).CompleteStream(
		context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil,
		func(delta string) { deltas = append(deltas, delta) })
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	streamed := strings.Join(deltas, "")
	if strings.Contains(streamed, "private reasoning") {
		t.Fatalf("streamed %q, want the reasoning withheld", streamed)
	}
	if streamed != "Before. After." {
		t.Fatalf("streamed %q, want the text around the block", streamed)
	}
	if reply.Content != "Before. After." {
		t.Fatalf("Content = %q", reply.Content)
	}
}

func TestCompleteStreamCollectsToolCalls(t *testing.T) {
	server, _ := streamServer(t,
		map[string]any{"message": map[string]any{"role": "assistant", "content": ""}, "done": false},
		map[string]any{
			"message": map[string]any{
				"role": "assistant",
				"tool_calls": []any{
					map[string]any{"function": map[string]any{
						"name": "hybrid_search", "arguments": map[string]any{"query": "x"},
					}},
					// Nameless, so undispatable; decodeToolCalls drops it.
					map[string]any{"function": map[string]any{"name": ""}},
				},
			},
			"done": true,
		},
	)

	reply, err := (&OllamaModel{BaseURL: server.URL, Model: "m"}).CompleteStream(
		context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, ToolSpecs(), nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	if len(reply.ToolCalls) != 1 {
		t.Fatalf("tool calls = %#v, want the named one only", reply.ToolCalls)
	}
	if reply.ToolCalls[0].Name != "hybrid_search" {
		t.Fatalf("name = %q", reply.ToolCalls[0].Name)
	}
	if reply.ToolCalls[0].Arguments["query"] != "x" {
		t.Fatalf("arguments = %#v", reply.ToolCalls[0].Arguments)
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

	// A answer turn must not advertise tools: the model would be free to answer
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

func TestUnloadModelAsksOllamaToEvictNow(t *testing.T) {
	var (
		path string
		body map[string]any
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, _ = io.WriteString(w, `{"done":true}`)
	}))
	defer server.Close()

	if err := UnloadModel(context.Background(), server.URL, "qwen3-vl:4b"); err != nil {
		t.Fatalf("UnloadModel: %v", err)
	}
	if path != "/api/generate" {
		t.Fatalf("path = %q, want /api/generate", path)
	}
	if body["model"] != "qwen3-vl:4b" {
		t.Fatalf("model = %#v", body["model"])
	}
	// keep_alive: 0 is Ollama's unload, and it is a number, not a duration
	// string — "0s" would mean "keep for zero seconds", which Ollama treats as
	// the default rather than as an eviction.
	if value, ok := body["keep_alive"].(float64); !ok || value != 0 {
		t.Fatalf("keep_alive = %#v, want the number 0", body["keep_alive"])
	}
}

func TestUnloadModelWithoutAModelIsANoOp(t *testing.T) {
	// "vision is off" must not become an error, and must not call out at all —
	// the unreachable address proves nothing was dialled.
	if err := UnloadModel(context.Background(), "http://127.0.0.1:1", "   "); err != nil {
		t.Fatalf("UnloadModel with no model: %v", err)
	}
}

func TestUnloadModelReportsAFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":"boom"}`)
	}))
	defer server.Close()

	if err := UnloadModel(context.Background(), server.URL, "qwen3-vl:4b"); err == nil {
		t.Fatal("an HTTP failure must be reported, not swallowed")
	}
}
