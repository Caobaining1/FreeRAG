package runtime

import "context"

// EmitAgentMessage is the non-OnDelta streaming fallback used by the ReAct loop
// when no OnDelta sink is installed. FreeRAG has no session-bound SSE writer of
// the kind RAGFlow uses, so this is a no-op: the final answer is still delivered
// through the adk event stream. It returns false to signal "nothing was emitted
// via the SSE path" so callers fall back to buffering.
func EmitAgentMessage(_ context.Context, _ string, _ string) bool {
	return false
}
