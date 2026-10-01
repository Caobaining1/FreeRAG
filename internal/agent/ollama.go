package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
// reasoning must not leak into the answer the checker reviews.
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

// CaptionImage asks a vision model to describe one image and returns its text.
//
// Separate from Complete for two reasons: the vision model is a different model
// than the loop's generator (which is text-only), and this call must not stream —
// the description is rewritten (numbers stripped, see vision.go) before it
// reaches the index, so a half-streamed caption would be unusable.
//
// maxTokens bounds generation, and it is the only lever that touches the
// dominant cost. Measured on the reference machine (Apple M5, qwen2.5vl:3b, Q4,
// 100% on Metal, image encoding ALSO on Metal — clip_ctx: MTL0 backend):
//
//	image + prompt prefill: 1230 tok in 15.1s (82 tok/s), 0.1s when the same
//	                        image is re-sent (prompt cache hit)
//	generation:             155 tok in 17.5s (8.8 tok/s)
//
// So ~8.8 tok/s is the wall, the same environment-bound rate the text model hits,
// and 155 tokens is what this model actually writes for a figure.
func (m *OllamaModel) CaptionImage(
	ctx context.Context, model, prompt, imageB64 string, maxTokens int,
) (string, error) {
	if model == "" {
		return "", fmt.Errorf("ollama: no vision model configured")
	}

	options := map[string]any{
		// Deterministic: the same figure must caption to the same text, or the
		// index is not reproducible and a rebuild silently changes answers.
		"temperature": 0,
		"seed":        7,
	}
	if maxTokens > 0 {
		options["num_predict"] = maxTokens
	}

	payload, err := json.Marshal(ollamaChatRequest{
		Model: model,
		Messages: []ollamaMessage{{
			Role:    string(RoleUser),
			Content: prompt,
			Images:  []string{imageB64},
		}},
		Stream:    false,
		Options:   options,
		KeepAlive: m.KeepAlive,
	})
	if err != nil {
		return "", fmt.Errorf("ollama: marshal caption request: %w", err)
	}

	endpoint := m.baseURL() + "/api/chat"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("ollama: build caption request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := m.client().Do(request)
	if err != nil {
		return "", fmt.Errorf("ollama: caption %s: %w", endpoint, err)
	}
	defer func() { _ = response.Body.Close() }()

	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 8<<10))
		return "", fmt.Errorf("ollama: HTTP %d: %s", response.StatusCode, truncateRunes(string(body), 200))
	}

	var decoded ollamaChatResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 8<<20)).Decode(&decoded); err != nil {
		return "", fmt.Errorf("ollama: decode caption: %w", err)
	}
	if decoded.Error != "" {
		return "", fmt.Errorf("ollama: %s", decoded.Error)
	}
	return strings.TrimSpace(decoded.Message.Content), nil
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
	// Images are base64 PNG/JPEG payloads, which is what Ollama's chat API
	// accepts inline. Only the figure captioner sends them.
	Images []string `json:"images,omitempty"`
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

// ollamaChatResponse is one object from /api/chat.
//
// The same struct covers a streamed reply and a non-streamed one: streaming
// sends a sequence of these, non-streaming sends exactly one. `done` is only
// meaningful while streaming.
type ollamaChatResponse struct {
	Message ollamaMessage `json:"message"`
	Error   string        `json:"error"`
	Done    bool          `json:"done"`
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
	return m.complete(ctx, messages, tools, false, nil)
}

// CompleteStream implements StreamingModel: Complete, with the answer reported
// as it is written.
//
// onDelta receives successive pieces of answer text, in order, and may be nil.
// The Reply is the same value Complete would have returned, so a caller can
// ignore the deltas entirely and still be correct.
//
// It exists for the UI. A detailed answer takes tens of seconds to generate, and
// the loop has nothing else to report during that window — the answer is produced
// after the tools and before the checker, so without this the window is silent.
func (m *OllamaModel) CompleteStream(
	ctx context.Context, messages []Message, tools []ToolSpec, onDelta func(string),
) (*Reply, error) {
	return m.complete(ctx, messages, tools, true, onDelta)
}

// complete performs one chat call, streamed or not.
//
// Both shapes are read with the same json.Decoder, because a streamed reply is a
// sequence of objects and a non-streamed one is a sequence of exactly one — so
// the loop does not need to know which it is. Streaming only adds the deltas;
// the assembled Reply is identical either way.
func (m *OllamaModel) complete(
	ctx context.Context, messages []Message, tools []ToolSpec, stream bool, onDelta func(string),
) (*Reply, error) {
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
		Stream:    stream,
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

	if response.StatusCode != http.StatusOK {
		// Read a little of the body for the message: an error response is small
		// and has already been fully sent, unlike the success path.
		body, _ := io.ReadAll(io.LimitReader(response.Body, 8<<10))
		return nil, fmt.Errorf("ollama: HTTP %d: %s", response.StatusCode, truncateRunes(string(body), 200))
	}

	var (
		content strings.Builder
		calls   []ToolCall
		filter  thinkFilter
	)
	// The ceiling is a safety net rather than a limit on the answer: num_predict
	// already bounds generation, and this only stops a misbehaving server from
	// filling memory.
	decoder := json.NewDecoder(io.LimitReader(response.Body, 64<<20))

	for {
		var chunk ollamaChatResponse
		if err := decoder.Decode(&chunk); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("ollama: decode response: %w", err)
		}
		if chunk.Error != "" {
			return nil, fmt.Errorf("ollama: %s", chunk.Error)
		}

		content.WriteString(chunk.Message.Content)
		if onDelta != nil {
			if text := filter.write(chunk.Message.Content); text != "" {
				onDelta(text)
			}
		}
		calls = append(calls, decodeToolCalls(chunk.Message.ToolCalls)...)

		if stream && chunk.Done {
			// Returned on the done marker rather than at EOF so a server that
			// keeps the connection open does not stall the caller.
			break
		}
	}

	// Flushed after the loop so the characters the filter was holding back are
	// not silently dropped at the end of the answer.
	if onDelta != nil {
		if text := filter.flush(); text != "" {
			onDelta(text)
		}
	}

	// StripThink still runs over the whole reply. The filter above already keeps
	// reasoning out of the deltas, but the Reply is what the checker reads, and
	// it must not depend on the filter having seen every byte.
	return &Reply{Content: StripThink(content.String()), ToolCalls: calls}, nil
}

// decodeToolCalls drops calls the loop could never dispatch.
//
// A call with no function name has nothing to resolve it against; keeping it
// would put a trace line in the result for a tool that was never executable.
func decodeToolCalls(raw []ollamaToolCall) []ToolCall {
	out := make([]ToolCall, 0, len(raw))
	for _, call := range raw {
		name := strings.TrimSpace(call.Function.Name)
		if name == "" {
			continue
		}
		out = append(out, ToolCall{
			ID:        call.ID,
			Name:      name,
			Arguments: decodeArguments(call.Function.Arguments),
		})
	}
	return out
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

// UnloadModel asks Ollama to evict a model from memory now.
//
// Called when an index job ends. The parse-time vision model (sidecar/vlm.py)
// is ~3.5 GB resident and nothing on the query path uses it, so holding it after
// the run is pure waste on a machine where memory is the binding constraint.
// `keep_alive: 0` is Ollama's documented unload.
//
// A failure is returned for the caller to log and nothing more: the model
// expires on its own keep-alive regardless, and failing an index run because a
// courtesy unload did not land would be absurd. An empty model is a no-op, which
// is what "vision is off" looks like.
func UnloadModel(ctx context.Context, baseURL, model string) error {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil
	}
	if strings.TrimSpace(baseURL) == "" {
		baseURL = DefaultOllamaURL
	}

	payload, err := json.Marshal(map[string]any{"model": model, "keep_alive": 0})
	if err != nil {
		return fmt.Errorf("ollama: marshal unload: %w", err)
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/api/generate"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("ollama: build unload: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := (&http.Client{Timeout: 60 * time.Second}).Do(request)
	if err != nil {
		return fmt.Errorf("ollama: unload %s: %w", model, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 8<<10))
		return fmt.Errorf("ollama: unload HTTP %d: %s",
			response.StatusCode, truncateRunes(string(body), 200))
	}
	return nil
}

// Reasoning markers, as they appear inline in a reply.
const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// thinkFilter removes reasoning blocks from a token stream.
//
// StripThink cannot be reused per delta, and that is the whole reason this type
// exists: Ollama splits text on its own boundaries, so "<thi" and "nk>let me
// think" arrive as separate chunks and a per-chunk regex matches neither. The
// reasoning would be typed out to the user ahead of the answer — the exact thing
// StripThink exists to prevent.
//
// So a few trailing characters are held back whenever they could still turn into
// a tag. The cost is one sub-word of latency at each chunk boundary and nothing
// else: held text is emitted as soon as it is known not to be a tag.
type thinkFilter struct {
	// held is the tail not yet safe to emit.
	held string
	// inside is true between an opening tag and its closing tag.
	inside bool
}

// write consumes one delta and returns the part of it that is safe to show.
func (f *thinkFilter) write(chunk string) string {
	f.held += chunk
	var out strings.Builder

	for {
		if f.inside {
			end := strings.Index(f.held, thinkClose)
			if end < 0 {
				// Still reasoning. Only the tail could be the closing tag.
				f.held = trailingBytes(f.held, len(thinkClose)-1)
				return out.String()
			}
			f.held = f.held[end+len(thinkClose):]
			f.inside = false
			continue
		}

		if start := strings.Index(f.held, thinkOpen); start >= 0 {
			out.WriteString(f.held[:start])
			f.held = f.held[start+len(thinkOpen):]
			f.inside = true
			continue
		}

		// No complete opening tag. Emit everything except a suffix that could
		// still become one.
		safe := len(f.held) - ambiguousTail(f.held)
		out.WriteString(f.held[:safe])
		f.held = f.held[safe:]
		return out.String()
	}
}

// flush returns whatever write is still holding back.
//
// Text inside an unterminated reasoning block is dropped rather than shown,
// which matches StripThink: a model that ran out of budget mid-thought produced
// no answer, and half a thought is worse than nothing.
func (f *thinkFilter) flush() string {
	held := ""
	if !f.inside {
		held = f.held
	}
	f.held = ""
	return held
}

// ambiguousTail reports how many trailing bytes could be the start of a tag.
//
// This is what makes the filter correct across chunk boundaries without holding
// back the whole stream: ordinary text reports 0, and only a genuine partial tag
// is delayed.
func ambiguousTail(s string) int {
	limit := len(thinkOpen) - 1
	if limit > len(s) {
		limit = len(s)
	}
	for n := limit; n > 0; n-- {
		if strings.HasPrefix(thinkOpen, s[len(s)-n:]) {
			return n
		}
	}
	return 0
}

// trailingBytes returns the last n bytes of s.
func trailingBytes(s string, n int) string {
	if n >= len(s) {
		return s
	}
	return s[len(s)-n:]
}
