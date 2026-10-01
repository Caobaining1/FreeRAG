package agent

import (
	"context"
	"encoding/json"
	"strings"
)

// defaultMaxSubQuestions caps the fan-out.
//
// Each sub-question runs a full agentic loop (retrieval + answer + checker +
// answer), so the cost is linear in this number — and the answer step of every
// sub-loop is a complete generation. Four is the point past which a question
// is usually better answered by one broader pass than by four narrow ones.
const defaultMaxSubQuestions = 4

// decomposeSystemPrompt splits a complex question into independent parts.
//
// Kept to the one thing the model must not get wrong: each part has to stand
// alone, because the parts are searched in parallel and never see each other.
// The JSON shape is stated because the alternative — bulleted prose — costs a
// parser that has to guess, and guessing wrong silently merges two questions.
const decomposeSystemPrompt = `You break a multi-part question into the smallest set of INDEPENDENT sub-questions.

Rules:
- Each sub-question must be answerable on its own, without the answers to the others.
- Keep the user's own wording and language; do not translate and do not add facts.
- Do not answer anything. Output the sub-questions only.
- If the question is already a single step, return it unchanged as the only element.
- At most %d sub-questions.

Reply with a JSON array of strings and nothing else.`

// Decompose splits question into up to max independent sub-questions.
//
// Degrades rather than fails: with no model, on a model error, or on a reply
// that does not parse, it returns the original question as the single element.
// A fan-out over one element is exactly the simple path, so every failure here
// is a fallback to known-good behaviour rather than a new failure mode.
func Decompose(ctx context.Context, model Model, question string, max int) []string {
	question = strings.TrimSpace(question)
	if max <= 0 {
		max = defaultMaxSubQuestions
	}
	if question == "" {
		return nil
	}
	if model == nil {
		return []string{question}
	}

	reply, err := model.Complete(ctx, []Message{
		{Role: RoleSystem, Content: strings.Replace(decomposeSystemPrompt, "%d", itoa(max), 1)},
		{Role: RoleUser, Content: "Question: " + question},
	}, nil)
	if err != nil || reply == nil {
		return []string{question}
	}

	subs := parseSubQuestions(reply.Content, max)
	if len(subs) == 0 {
		return []string{question}
	}
	return subs
}

// parseSubQuestions reads a JSON array of strings, falling back to line
// splitting for a reply that ignored the format.
func parseSubQuestions(content string, max int) []string {
	text := strings.TrimSpace(content)
	text = strings.TrimPrefix(text, "```json")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")
	text = strings.TrimSpace(text)

	// Try the object form the prompt asked for first, then a bare array, then
	// the first [...] span inside surrounding prose.
	for _, candidate := range jsonCandidates(text) {
		var items []string
		if err := json.Unmarshal([]byte(candidate), &items); err == nil {
			if out := cleanSubQuestions(items, max); len(out) > 0 {
				return out
			}
		}
	}

	// Line-based fallback: a bulleted or numbered list, which is what a model
	// that ignores the JSON instruction usually produces.
	//
	// Two lines minimum. A single line of prose is not a decomposition, and
	// accepting it would replace the real question with the model's own
	// commentary — the caller's fallback (the original question) is strictly
	// better than that.
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		lines = append(lines, line)
	}
	if cleaned := cleanSubQuestions(lines, max); len(cleaned) >= 2 {
		return cleaned
	}
	return nil
}

// jsonCandidates yields the spans worth trying to unmarshal, most literal first.
func jsonCandidates(text string) []string {
	out := []string{text}
	if start := strings.Index(text, "["); start >= 0 {
		if end := strings.LastIndex(text, "]"); end > start {
			out = append(out, text[start:end+1])
		}
	}
	return out
}

// cleanSubQuestions trims bullets and numbering, drops blanks and echoes, and
// caps the count.
func cleanSubQuestions(items []string, max int) []string {
	var out []string
	for _, item := range items {
		item = strings.TrimSpace(item)
		item = strings.TrimLeft(item, "-*• \t")
		item = strings.TrimSpace(item)
		// Strip a leading "1." / "1)" / "1、".
		if cut := strings.IndexAny(item, ".)、"); cut == 1 && item[0] >= '0' && item[0] <= '9' {
			item = strings.TrimSpace(item[cut+1:])
		}
		if item == "" {
			continue
		}
		out = append(out, item)
		if len(out) >= max {
			break
		}
	}
	return out
}

// itoa is a tiny int formatter, kept local so decompose.go has no strconv
// import for one call.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
