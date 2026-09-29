package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// DefaultOllamaURL is where a local Ollama server listens.
const DefaultOllamaURL = "http://127.0.0.1:11434"

// DefaultGeneratorModel is the model name the setup script imports.
const DefaultGeneratorModel = "freerag-qwen3"

// reThinkBlock matches a complete reasoning block. Qwen3 answers in thinking
// mode by default; the loop's prompt asks for a short cited answer, so the
// reasoning must not leak into the draft the checker reviews.
var reThinkBlock = regexp.MustCompile(`(?s)<think>.*?</think>`)

// OllamaModel implements Model against a local Ollama server, which runs the
// GGUF weights through llama.cpp (docs/plan.md §3: inference stays a separate
// process).
type OllamaModel struct {
	// BaseURL defaults to DefaultOllamaURL.
	BaseURL string
	// Model is the Ollama model name, e.g. "freerag-qwen3".
	Model string
	// NumCtx is the context window requested for a call; <=0 sends nothing and
	// the server default (4096 for Ollama) applies.
	//
	// This has to be sent explicitly: Ollama's default window is smaller than
	// the evidence block the loop assembles, and the server rejects an
	// oversized prompt with HTTP 400 rather than truncating it.
	NumCtx int
	// NumPredict caps generated tokens per call; <=0 uses the server default.
	NumPredict int
	// Temperature; <=0 uses the server default.
	Temperature float64
	// Think controls Qwen3's reasoning mode; nil leaves the model's default.
	//
	// Measured on the reference machine (Apple M5, 4B Q4_K_M, fully on GPU):
	// reasoning mode generated 246 tokens in 29.7 s to answer a one-sentence
	// question, of which message.thinking was 384 characters that nothing reads.
	// With reasoning off the same question took 1.9 s for 14 tokens.
	//
	// It is also a correctness switch, not only a speed one: the reasoning is
	// generated against the same num_predict budget as the answer, so a cap
	// that is generous for an answer can be consumed entirely by reasoning —
	// observed returning zero answer content at num_predict=96.
	Think *bool
	// KeepAlive is how long Ollama keeps the weights resident after a call,
	// e.g. "30m" or "-1" for indefinitely. Empty uses Ollama's default (5m).
	//
	// Releasing the model also releases the prompt cache, so the next call
	// re-processes the shared prefix — which is the expensive half of a prompt.
	KeepAlive string
	// Timeout bounds one call; <=0 selects 3 minutes.
	Timeout time.Duration
	// HTTP is optional (tests); nil uses a default client.
	HTTP *http.Client
}

func (m *OllamaModel) baseURL() string {
	if m.BaseURL == "" {
		return DefaultOllamaURL
	}
	return strings.TrimRight(m.BaseURL, "/")
}

func (m *OllamaModel) client() *http.Client {
	if m.HTTP != nil {
		return m.HTTP
	}
	timeout := m.Timeout
	if timeout <= 0 {
		timeout = 3 * time.Minute
	}
	return &http.Client{Timeout: timeout}
}

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

// ollamaTool is the wire form of a tool the model may call (OpenAI-compatible,
// which is what Ollama's /api/chat accepts).
type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaToolFunction `json:"function"`
}

type ollamaToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ollamaToolCall is the wire form of a call the model asked for.
//
// Arguments is kept raw because the two conventions disagree: OpenAI sends a
// JSON *string*, Ollama sends an object. Decoding it as either is cheaper than
// betting on one and failing on the other.
type ollamaToolCall struct {
	ID       string `json:"id,omitempty"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments,omitempty"`
	} `json:"function"`
}

type ollamaChatRequest struct {
	Model     string          `json:"model"`
	Messages  []ollamaMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	Options   map[string]any  `json:"options,omitempty"`
	Tools     []ollamaTool    `json:"tools,omitempty"`
	Think     *bool           `json:"think,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
}

type ollamaChatResponse struct {
	Message ollamaMessage `json:"message"`
	Error   string        `json:"error"`
}

// decodeArguments accepts either an object or a JSON-encoded string of one.
//
// A tool call whose arguments cannot be read is returned as nil rather than as
// an error: the tool then fails on its own missing-argument check, which says
// which argument was missing — more useful to the trace than a decode failure.
func decodeArguments(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return nil
	}

	args := map[string]any{}
	if err := json.Unmarshal(raw, &args); err == nil {
		return args
	}

	// OpenAI-compatible shape: the arguments are a JSON string inside JSON.
	var encoded string
	if err := json.Unmarshal(raw, &encoded); err != nil {
		return nil
	}
	if err := json.Unmarshal([]byte(encoded), &args); err != nil {
		return nil
	}
	return args
}

// wireTools renders tool specs for the request body.
func wireTools(specs []ToolSpec) []ollamaTool {
	out := make([]ollamaTool, 0, len(specs))
	for _, spec := range specs {
		out = append(out, ollamaTool{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        spec.Name,
				Description: spec.Description,
				Parameters:  spec.Parameters,
			},
		})
	}
	return out
}

// Complete implements Model.
//
// Tool specs are forwarded when the caller supplies them, and any tool calls in
// the reply are returned in Reply.ToolCalls. The loop decides whether to act on
// them: it owns the budget and the fallback, so a model that asks for nothing
// useful still leaves a working pipeline behind.
func (m *OllamaModel) Complete(ctx context.Context, messages []Message, tools []ToolSpec) (*Reply, error) {
	if m.Model == "" {
		return nil, fmt.Errorf("ollama: no model configured")
	}

	wire := make([]ollamaMessage, 0, len(messages))
	for _, msg := range messages {
		wire = append(wire, ollamaMessage{Role: string(msg.Role), Content: msg.Content})
	}

	options := map[string]any{}
	if m.NumCtx > 0 {
		options["num_ctx"] = m.NumCtx
	}
	if m.NumPredict > 0 {
		options["num_predict"] = m.NumPredict
	}
	if m.Temperature > 0 {
		options["temperature"] = m.Temperature
	}

	payload, err := json.Marshal(ollamaChatRequest{
		Model:     m.Model,
		Messages:  wire,
		Stream:    false,
		Options:   options,
		Tools:     wireTools(tools),
		Think:     m.Think,
		KeepAlive: m.KeepAlive,
	})
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	endpoint := m.baseURL() + "/api/chat"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := m.client().Do(request)
	if err != nil {
		return nil, fmt.Errorf("ollama: call %s: %w", endpoint, err)
	}
	defer func() { _ = response.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("ollama: read response: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama: HTTP %d: %s", response.StatusCode, truncateRunes(string(body), 200))
	}

	var decoded ollamaChatResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}
	if decoded.Error != "" {
		return nil, fmt.Errorf("ollama: %s", decoded.Error)
	}

	reply := &Reply{Content: StripThink(decoded.Message.Content)}
	for _, call := range decoded.Message.ToolCalls {
		name := strings.TrimSpace(call.Function.Name)
		if name == "" {
			// A nameless call cannot be dispatched; dropping it here keeps the
			// trace free of a call that was never executable.
			continue
		}
		reply.ToolCalls = append(reply.ToolCalls, ToolCall{
			ID:        call.ID,
			Name:      name,
			Arguments: decodeArguments(call.Function.Arguments),
		})
	}
	return reply, nil
}

// Reachable reports whether an Ollama server answers and the model is present.
func (m *OllamaModel) Reachable(ctx context.Context) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	request, err := http.NewRequestWithContext(probeCtx, http.MethodGet, m.baseURL()+"/api/tags", nil)
	if err != nil {
		return false
	}
	response, err := m.client().Do(request)
	if err != nil {
		return false
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return false
	}

	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&tags); err != nil {
		return false
	}
	for _, model := range tags.Models {
		if model.Name == m.Model || strings.HasPrefix(model.Name, m.Model+":") {
			return true
		}
	}
	return false
}

// StripThink removes a reasoning block and trims the result.
//
// An UNTERMINATED <think> means the reply was cut off mid-reasoning: everything
// after it is reasoning, so the visible answer is what precedes it (usually
// nothing), never the reasoning itself.
func StripThink(text string) string {
	text = reThinkBlock.ReplaceAllString(text, "")
	if index := strings.Index(text, "<think>"); index >= 0 {
		text = text[:index]
	}
	return strings.TrimSpace(text)
}
