// Package agent implements the medium-mode Agentic Loop described in
// docs/plan.md §6: the graph
//
//	formalize_question -> rag_agent(session -> answer -> sca) <-> query_rewrite
//
// where retrieval is local, the answer and the rewritten queries come from the
// generating model, and the sufficiency verdict comes from a decision model
// (Laya) or its deterministic fallback.
package agent

import "context"

// Role is a chat message role.
type Role string

// Message roles.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one turn handed to the model.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// ToolCall is a model request to run a tool.
type ToolCall struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

// Reply is one model turn: free text, tool calls, or both.
type Reply struct {
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// Usage is the generator's own account of the call, when it reports one.
	//
	// Expected to be absent: a model that does not report timings is not
	// broken, and nothing in the loop branches on this. It exists because
	// "cost = wall clock + generated tokens" is one half of the fitness
	// function in docs/plan.md §13.1 and half of it was unmeasurable —
	// wall clock is read off a clock, token counts were not reported at all,
	// so a change could trade latency for quality (or the reverse) invisibly.
	Usage *Usage `json:"usage,omitempty"`
}

// Usage is what one generator call cost, as the generator measured it.
//
// The split matters rather than the total: prompt tokens are what the evidence
// block decides and output tokens are what the answer length decides, and a
// single "elapsed" would leave the two indistinguishable — which is exactly how
// a prompt-side change gets credited to, or blamed on, the wrong half.
type Usage struct {
	PromptTokens int `json:"prompt_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	// PromptNanos and OutputNanos are the generator's own timings.
	PromptNanos int64 `json:"prompt_nanos,omitempty"`
	OutputNanos int64 `json:"output_nanos,omitempty"`
}

// ToolSpec describes a tool the model may call.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// Model is the generating LLM (Qwen3-4B via llama.cpp in production).
//
// It is an interface so the loop can be exercised with a scripted model: the
// graph's control flow must be verifiable without a model download.
type Model interface {
	Complete(ctx context.Context, messages []Message, tools []ToolSpec) (*Reply, error)
}

// StreamingModel is a Model that can report its reply as it is generated.
//
// Deliberately separate from Model rather than folded into it: streaming is a
// convenience for the UI, and requiring every implementation to carry it would
// make every test double do so too, for a capability the loop only uses when it
// happens to be there. The loop type-asserts (loop.go, answer).
type StreamingModel interface {
	Model
	// CompleteStream returns the same Reply Complete would, having passed each
	// piece of answer text to onDelta in order. onDelta may be nil.
	CompleteStream(ctx context.Context, messages []Message, tools []ToolSpec, onDelta func(string)) (*Reply, error)
}
