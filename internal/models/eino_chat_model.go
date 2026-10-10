// Package models provides the FreeRAG bridge between the agentic_rag ReAct loop
// (which is built on eino's adk.ChatModelAgent and therefore needs an eino
// model.ToolCallingChatModel) and FreeRAG's own LLM abstraction, agent.Model
// (the Ollama-backed generator).
//
// EinoChatModel adapts agent.Model — translating eino's *schema.Message
// conversation (including assistant tool calls and tool results) into
// agent.Message turns and back.
package models

import (
	"context"
	"encoding/json"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"go.uber.org/zap"

	"freerag/internal/agent"
	"freerag/internal/common"
	"freerag/internal/embed"
)

// EinoChatModel is an eino model.ToolCallingChatModel backed by agent.Model.
type EinoChatModel struct {
	// Model is the FreeRAG generating model (Ollama-backed).
	Model agent.Model
	// Embedder is currently unused by the adapter itself but kept for parity
	// with the RAGFlow EinoChatModel shape (hybrid retrieval is wired through
	// the runtime package instead).
	Embedder embed.Embedder

	tools []*schema.ToolInfo
}

// NewEinoChatModel builds an EinoChatModel over the given FreeRAG model.
func NewEinoChatModel(m agent.Model, emb embed.Embedder) *EinoChatModel {
	return &EinoChatModel{Model: m, Embedder: emb}
}

// Generate implements model.BaseChatModel.
func (e *EinoChatModel) Generate(ctx context.Context, in []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	msgs, err := toAgentMessages(in)
	if err != nil {
		return nil, err
	}
	reply, err := e.Model.Complete(ctx, msgs, toAgentTools(e.tools))
	if err != nil {
		return nil, err
	}
	return toSchemaMessage(reply), nil
}

// Stream implements model.BaseChatModel. The Ollama generator is wrapped as a
// single-message stream so the adk loop receives a complete assistant turn.
func (e *EinoChatModel) Stream(ctx context.Context, in []*schema.Message, opts ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	msg, err := e.Generate(ctx, in, opts...)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.Message{msg}), nil
}

// WithTools implements model.ToolCallingChatModel: it returns a copy with the
// given tools bound (immutable, concurrency-safe).
func (e *EinoChatModel) WithTools(tools []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	cp := *e
	cp.tools = tools
	return &cp, nil
}

// --- conversions ---

func toAgentMessages(in []*schema.Message) ([]agent.Message, error) {
	out := make([]agent.Message, 0, len(in))
	for _, m := range in {
		am := agent.Message{Role: roleFromSchema(m.Role), Content: m.Content}
		if m.Role == schema.Tool {
			am.ToolCallID = m.ToolCallID
		}
		for _, tc := range m.ToolCalls {
			args := map[string]any{}
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					common.Warn("agentic_rag: bad tool call args", zap.String("tool", tc.Function.Name), zap.Error(err))
				}
			}
			am.ToolCalls = append(am.ToolCalls, agent.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: args,
			})
		}
		out = append(out, am)
	}
	return out, nil
}

func toAgentTools(tools []*schema.ToolInfo) []agent.ToolSpec {
	if len(tools) == 0 {
		return nil
	}
	out := make([]agent.ToolSpec, 0, len(tools))
	for _, t := range tools {
		params := map[string]any{}
		if t.ParamsOneOf != nil {
			if raw, err := json.Marshal(t.ParamsOneOf); err == nil {
				_ = json.Unmarshal(raw, &params)
			}
		}
		out = append(out, agent.ToolSpec{
			Name:        t.Name,
			Description: t.Desc,
			Parameters:  params,
		})
	}
	return out
}

func toSchemaMessage(reply *agent.Reply) *schema.Message {
	msg := &schema.Message{Role: schema.Assistant, Content: reply.Content}
	for _, tc := range reply.ToolCalls {
		raw, err := json.Marshal(tc.Arguments)
		if err != nil {
			raw = []byte("{}")
		}
		msg.ToolCalls = append(msg.ToolCalls, schema.ToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: schema.FunctionCall{
				Name:      tc.Name,
				Arguments: string(raw),
			},
		})
	}
	return msg
}

func roleFromSchema(r schema.RoleType) agent.Role {
	switch r {
	case schema.System:
		return agent.RoleSystem
	case schema.User:
		return agent.RoleUser
	case schema.Assistant:
		return agent.RoleAssistant
	case schema.Tool:
		return agent.RoleTool
	default:
		return agent.RoleUser
	}
}
