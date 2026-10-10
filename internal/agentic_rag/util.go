package agentic_rag

import "strings"

// xmlEscape escapes characters that would break simple XML attribute/element
// values. The output is consumed by the LLM (forgiving parser).
//
// This is a pure string utility migrated from RAGFlow's tool_grep_chunks.go; it
// is loop-support code, not a tool, so it stays with the ReAct loop.
func xmlEscape(s string) string {
	replacer := strings.NewReplacer(
		"&", "&amp;",
		"<", "&lt;",
		">", "&gt;",
		"\"", "&quot;",
		"'", "&apos;",
	)
	return replacer.Replace(s)
}
