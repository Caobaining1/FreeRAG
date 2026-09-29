// Package agent implements the medium-mode Agentic Loop described in
// docs/plan.md §6: the graph
//
//	formalize_question -> rag_agent(session -> draft -> sca) <-> query_rewrite
//
// where retrieval is local, the draft and the rewritten queries come from the
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
