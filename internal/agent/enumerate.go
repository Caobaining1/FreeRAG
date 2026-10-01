package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"freerag/internal/store"
)

// extractMembersSystemPrompt reads the members of a set out of pooled evidence.
//
// This is the one place the enumeration asks a model anything, and the wording is
// the whole safeguard: the failure mode being guarded against is a list that grows
// from the model's own knowledge, which reads exactly like a list the corpus
// supports. RAGFlow splits this judgement per window (coverage_resolve.go, one
// call per window, batched by 8) precisely so that each verdict sits over a small
// passage; the equivalent here is one call over the numbered pool, with the
// passage each name came from required and checked (see anchorMember).
const extractMembersSystemPrompt = `You list the named members of a set, from evidence you are given.

The question asks for a SET of named elements. Read the numbered evidence and name
every member it states.

Rules:
- Name only members the EVIDENCE states. Never add a name from your own knowledge,
  and never guess a name to make the list longer. An empty list is a valid answer.
- The same member named two ways is ONE member ("the 3rd Armored Division" and
  "3rd Armored" are one member).
- When the question asks what an actor DID, the actor is not a member.
- A passage that names a member without the deed does not count.
- List every member you find, not only the first few.

Reply with JSON only, in this shape: {"members": [{"name": "member name", "evidence": 2}]}
"evidence" is the number of the passage that states the member.`

// ExtractMembers names the members of the set in evidence.
//
// One call over the whole pool rather than one per passage, which is where this
// departs from RAGFlow. Its resolve pass asks a model per window because its model
// is served on a GPU and its pool can hold hundreds of windows; here generation
// runs at about 6.4 tok/s on CPU, so a per-window pass would spend minutes on one
// sub-question to answer a question the pool already fits inside.
//
// The property that is NOT traded away is the one that makes the list trustworthy:
// every member must be anchored to a passage, and anchoring is checked against the
// pool rather than trusted from the reply (see anchorMember). A member the pool
// does not contain is dropped, exactly as RAGFlow drops a ref it cannot resolve.
func ExtractMembers(
	ctx context.Context,
	model Model,
	question string,
	evidence []store.Hit,
	maxChars int,
) []Member {
	if model == nil || len(evidence) == 0 {
		return nil
	}

	reply, err := model.Complete(ctx, []Message{
		{Role: RoleSystem, Content: extractMembersSystemPrompt},
		{Role: RoleUser, Content: renderMemberPrompt(question, evidence, maxChars)},
	}, nil)
	if err != nil || reply == nil {
		return nil
	}

	named := parseMembers(reply.Content)
	if len(named) == 0 {
		return nil
	}

	out := make([]Member, 0, len(named))
	for _, member := range named {
		anchored, ok := anchorMember(member, evidence)
		if !ok {
			continue
		}
		out = append(out, anchored)
	}
	// Anchored, deduped, in the order the model named them: the model's order is
	// how it read the evidence, which is a better order to present than whichever
	// passage happens to come first in the pool.
	return AnchoredMembers(out)
}

// renderMemberPrompt numbers the evidence so the reply can point at a passage.
//
// Numbered from 1, and the numbers are the ones the ANSWER will use: the whole
// point of anchoring is that a member's passage is one the answer can cite, so the
// two audiences are given the same numbering.
func renderMemberPrompt(question string, evidence []store.Hit, maxChars int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Question: %s\n\nEvidence:\n", question)
	used := 0
	for i, hit := range evidence {
		header := fmt.Sprintf("[%d] (%s p.%d) ", i+1, hit.Chunk.DocID, hit.Chunk.PageNum)
		text := collapse(strings.TrimSpace(hit.Chunk.Text))
		if maxChars > 0 {
			room := maxChars - used - len(header) - 1
			if room < 120 {
				fmt.Fprintf(&b, "\n(%d more passage(s) omitted to fit the context window.)\n",
					len(evidence)-i)
				break
			}
			if len(text) > room {
				text = truncateRunes(text, room) + "…"
			}
		}
		fmt.Fprintf(&b, "%s%s\n", header, text)
		used += len(header) + len(text) + 1
	}
	return b.String()
}

// parseMembers reads the reply's member list.
//
// Tolerant about the SHAPE and strict about nothing else: the name is what
// matters, and a member parsed without a passage still gets a chance to be
// anchored by name (see anchorMember). A reply that ignores the JSON instruction
// and writes a bulleted list is read the same way, because the alternative —
// discarding it — turns a usable list into no list at all.
func parseMembers(reply string) []Member {
	cleaned := stripCodeFence(strings.TrimSpace(reply))
	if cleaned != "" && (cleaned[0] == '{' || cleaned[0] == '[') {
		if members := parseMembersJSON(cleaned); len(members) > 0 {
			return members
		}
	}
	return parseMembersLoose(reply)
}

// parseMembersJSON reads the documented object, and the shapes a model writes
// instead of it: a bare array of names or of objects.
func parseMembersJSON(cleaned string) []Member {
	var payload struct {
		Members []json.RawMessage `json:"members"`
	}
	if err := json.Unmarshal([]byte(cleaned), &payload); err != nil {
		var bare []json.RawMessage
		if err := json.Unmarshal([]byte(cleaned), &bare); err != nil {
			return nil
		}
		payload.Members = bare
	}

	var out []Member
	for _, entry := range payload.Members {
		var name string
		if err := json.Unmarshal(entry, &name); err == nil {
			if name = strings.TrimSpace(name); name != "" {
				out = append(out, Member{Name: name})
			}
			continue
		}
		var object map[string]any
		if err := json.Unmarshal(entry, &object); err != nil {
			continue
		}
		name = trimStringField(object, "name", "member", "entity", "value")
		if name == "" {
			continue
		}
		member := Member{
			Name:  name,
			Quote: trimStringField(object, "quote", "evidence_text", "context"),
		}
		// The passage number, in whichever key the model used for it.
		if index := intField(object, "evidence", "evidence_index", "passage", "id", "i"); index > 0 {
			member.ChunkID = fmt.Sprintf("#%d", index)
		}
		out = append(out, member)
	}
	return out
}

// parseMembersLoose takes the items of a bulleted or numbered list.
//
// Only marked lines count — the same rule as the query fallback, and for the same
// reason: prose around a list would otherwise be read as member names.
func parseMembersLoose(reply string) []Member {
	var out []Member
	for _, line := range strings.Split(reply, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !hasListMarker(line) {
			continue
		}
		line = strings.TrimLeft(line, "-*• \t")
		digits := 0
		for digits < len(line) && line[digits] >= '0' && line[digits] <= '9' {
			digits++
		}
		if digits > 0 && digits < len(line) && (line[digits] == '.' || line[digits] == ')') {
			line = line[digits+1:]
		}
		line = strings.Trim(strings.TrimSpace(line), `",`)
		// "- 华雄 — 被斩" and "- 华雄 (evidence 3)": keep the name, drop the rest.
		if cut := strings.IndexAny(line, "—(-（"); cut > 0 {
			line = strings.TrimSpace(line[:cut])
		}
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, Member{Name: line})
		}
	}
	return out
}

// intField reads the first of keys whose value is a usable integer.
func intField(object map[string]any, keys ...string) int {
	for _, key := range keys {
		switch typed := object[key].(type) {
		case float64:
			return int(typed)
		case string:
			index := 0
			for _, r := range strings.TrimSpace(typed) {
				if r < '0' || r > '9' {
					index = 0
					break
				}
				index = index*10 + int(r-'0')
			}
			if index > 0 {
				return index
			}
		}
	}
	return 0
}

// anchorMember attaches a passage to a member, and reports whether it found one.
//
// The declared passage number is checked against the pool, and when there is none
// — or it is out of range, or it points at a passage that does not state the name
// — the name is looked for in the pool directly. That second path is RAGFlow's
// resolveMemberAnchor: an anchor is RESOLVED against the evidence rather than
// trusted, so a model that named a real member but cited the wrong passage still
// produces a citable member, and a model that invented a member produces nothing.
func anchorMember(member Member, evidence []store.Hit) (Member, bool) {
	if index, ok := declaredIndex(member.ChunkID); ok {
		if hit, found := hitAt(evidence, index); found {
			if containsFold(hit.Chunk.Text, member.Name) {
				return withHit(member, hit), true
			}
		}
	}
	for _, hit := range evidence {
		if containsFold(hit.Chunk.Text, member.Name) {
			return withHit(member, hit), true
		}
	}
	return Member{}, false
}

// declaredIndex reads the "#n" a parsed member carries, if any.
func declaredIndex(marker string) (int, bool) {
	if !strings.HasPrefix(marker, "#") {
		return 0, false
	}
	index := intField(map[string]any{"n": strings.TrimPrefix(marker, "#")}, "n")
	return index, index > 0
}

// hitAt returns the 1-based evidence index, guarding the bound.
func hitAt(evidence []store.Hit, index int) (store.Hit, bool) {
	if index < 1 || index > len(evidence) {
		return store.Hit{}, false
	}
	return evidence[index-1], true
}

// withHit fills in the member's passage reference and the quote around its
// occurrence.
func withHit(member Member, hit store.Hit) Member {
	member.ChunkID = hit.Chunk.ChunkID
	member.DocID = hit.Chunk.DocID
	if strings.TrimSpace(member.Quote) == "" {
		member.Quote = quoteAround(hit.Chunk.Text, member.Name)
	}
	return member
}

// quoteAround returns the text around a name's first occurrence.
//
// The window is RAGFlow's (coverage_enumerate.go: 80 runes before the deed, 60
// after) applied to the name instead of the act word: the same width, so a quote
// shows the name in the sentence that states it rather than the name alone.
func quoteAround(text, name string) string {
	runes := []rune(text)
	needle := []rune(name)
	if len(needle) == 0 {
		return ""
	}
	start := runeIndexFold(runes, needle)
	if start < 0 {
		return ""
	}
	from := start - 80
	if from < 0 {
		from = 0
	}
	to := start + len(needle) + 60
	if to > len(runes) {
		to = len(runes)
	}
	return collapse(strings.TrimSpace(string(runes[from:to])))
}

// runeIndexFold finds needle in haystack, ignoring case, and returns its rune
// index or -1.
func runeIndexFold(haystack, needle []rune) int {
	if len(needle) == 0 || len(needle) > len(haystack) {
		return -1
	}
	lower := make([]rune, len(haystack))
	for i, r := range haystack {
		lower[i] = []rune(strings.ToLower(string(r)))[0]
	}
	target := make([]rune, len(needle))
	for i, r := range needle {
		target[i] = []rune(strings.ToLower(string(r)))[0]
	}
	for start := 0; start+len(target) <= len(lower); start++ {
		match := true
		for offset := range target {
			if lower[start+offset] != target[offset] {
				match = false
				break
			}
		}
		if match {
			return start
		}
	}
	return -1
}

// containsFold reports whether text contains needle, ignoring case.
func containsFold(text, needle string) bool {
	needle = strings.TrimSpace(needle)
	if needle == "" {
		return false
	}
	return strings.Contains(strings.ToLower(text), strings.ToLower(needle))
}
