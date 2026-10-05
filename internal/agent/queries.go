package agent

import (
	"context"
	"encoding/json"
	"strings"
)

// maxSearchQueries caps how many queries one rewrite may produce.
//
// More than one only helps when the parts would be found in different passages:
// each query is another retrieval call, and the calls in a turn share one budget
// (see ActionMaxTurns).
const maxSearchQueries = 3

// Query shape guards, modelled on RAGFlow's fanout guards
// (internal/rag/agentic-rag/graph_fanout.go's fanoutLooksLikeQuery: a character
// cap, a word cap and a list of markers that mean the model answered instead of
// searching). A guard is worth more than a stricter prompt here, because the
// failure it catches — a rewritten query that is really an answer — poisons
// retrieval silently.
const (
	maxSearchQueryChars = 200
	maxSearchQueryWords = 24
)

// answerMarkers are how a rewrite that answered the question reveals itself.
// RAGFlow's list, kept verbatim where it applies.
var answerMarkers = []string{
	"http://", "https://", "**", "sources:", "source:", "references:", "citation", "according to",
}

// queryRewriteSystemPrompt turns a question into the queries a search index can
// hit.
//
// The shape of this prompt follows what RAGFlow does before retrieval
// (internal/rag/agentic-rag/agentic_rag.go's formalizePrompt for the standalone
// question plus keywords, and rag/prompts/keyword_prompt.md for the keyword
// half), which in turn follows the observation that a question is written for a
// reader while a query is matched by an index: the index matches literal words
// (so the entity must be repeated, never pronominalised) and benefits from the
// synonyms the question left out.
//
// It is deliberately NOT asked to answer, and the rules say so twice: a rewrite
// that states a fact makes the retrieval that follows a search for its own
// guess.
const queryRewriteSystemPrompt = `You turn a user's question into the search queries that will find the passages answering it.

Rules:
- Output ONE query per part of the question that would be answered by DIFFERENT passages. If the question has one part, output one query. Never output two queries that are rewordings of each other — "X game", "X matchup" and "X contest" are one query written three times, and each one costs a retrieval call and pulls the same passage back.
- A query is a short keyword-rich phrase, roughly 3 to 12 words: the entity plus what is asked about it. Not a sentence, not an explanation.
- Name the actual thing, not the kind of thing. Generic search words — "article", "details", "summary", "information", "matchup", "overview", "event" — match nothing in particular; drop them and keep the concrete terms the document would use.
- Use alternative NAMES of the entities (aliases, abbreviations, other spellings, the short name and the full name), because a keyword index matches literally. Do NOT add alternative words for the relation or the action: "acquire" and "acquisition" are one term, not two.
  Example: "In which year did Apple acquire Beats?" -> "Apple Apple Inc. AAPL Beats acquisition year"
  Example with two parts: "What did Michigan do against Ohio State, and how did Alabama finish?" -> ["Michigan Ohio State football result", "Alabama football season finish"]
- NEVER use pronouns. Repeat the entity: "he", "it" and "this" match nothing in an index.
- NEVER answer, and never state a fact. Do not add a name, a date, a number or a value that is not in the question.
- Keep the language of the question.
- If the question is already a good search query, return it unchanged as the only query.

Then declare what the ANSWER is, as slots — one per thing the question asks for
(usually one). This is not a guess about the answer's content; it is a statement
about its SHAPE, and it decides whether the run searches for a single value or for
a whole set of them:
- "type": the shape of the answer — "entity", "person", "dataset", "list", "set",
  "count", "date", "number", "place" or "text".
- "subject": when the question is about things that someone DID, the actor with
  its aliases, joined by "|" ("Michigan|UM"). Omit when no actor is named.
- "scan": when the question is about things that someone DID, the words the
  DOCUMENTS would use for the deed — several of them, because one source words it
  one way and another words it differently ("defeated|beat|won against"). These
  become the searches that recall the statements, so a wording none of them covers
  is a statement no search finds. Omit when the answer is not a set of deeds.

Use "list", "set" or "count" as the type ONLY when the answer is a set of NAMED
elements — several teams, several people, several datasets. A question whose answer
is one name, one date or one number is not a set: declaring one as a set makes the
run search for members that do not exist, and it will report the ones it happened
to find as if the list were complete.
- A yes/no question ("Did X happen?", "Was there a change?", "Does A suggest B?") has a single yes/no answer: use type "text" and do NOT add "subject" or "scan".

Reply with JSON only, in this shape:
{"queries": ["first query", "second query"],
 "slots": [{"type": "person", "subject": "关羽|关公", "scan": ["斩", "杀"]}]}`

// QueryPlan is what one rewrite call returns: the queries round 1 should run, and
// the slots the same reply declared.
//
// One call, two answers — which is RAGFlow's shape too: its initialize_state
// prompt asks for `slots` and `first_queries` together
// (prompts/action_initialize_state.md), because both come from the same reading
// of the question and asking twice would cost a second generation to say the
// same thing twice.
type QueryPlan struct {
	// Queries are the search queries. Never empty: the question itself stands in
	// when no rewrite was possible.
	Queries []string
	// Slots are what the reply declared the answer to BE. Empty when the model
	// declared none, which is the normal case and costs the enumeration nothing
	// (see Coverage.Ok).
	Slots []Slot
	// Rewritten reports whether Queries is a rewrite, as opposed to the question
	// coming back unchanged.
	Rewritten bool
}

// PlanQueries turns question into the queries retrieval should run, and reads the
// slot declaration out of the same reply.
func PlanQueries(ctx context.Context, model Model, question string, max int) QueryPlan {
	question = strings.TrimSpace(question)
	if question == "" {
		return QueryPlan{}
	}
	if max <= 0 {
		max = maxSearchQueries
	}
	if model == nil {
		return QueryPlan{Queries: []string{question}}
	}

	reply, err := model.Complete(ctx, []Message{
		{Role: RoleSystem, Content: queryRewriteSystemPrompt},
		{Role: RoleUser, Content: "Question: " + question},
	}, nil)
	if err != nil || reply == nil {
		return QueryPlan{Queries: []string{question}}
	}

	// Read independently of the query parsing: a reply that declared its slots but
	// wrote its queries as prose still has a usable declaration, and one that
	// wrote perfect queries but no slots simply is not an enumeration.
	slots := parseDeclaredSlots(reply.Content)
	queries := parseQueries(reply.Content, max)
	if len(queries) == 0 {
		return QueryPlan{Queries: []string{question}, Slots: slots}
	}
	return QueryPlan{Queries: queries, Slots: slots, Rewritten: true}
}

// RewriteQueries turns question into the search queries retrieval should run.
//
// Returns the queries and whether they are a rewrite (false means the question
// came back unchanged, because there was no model, the call failed, or the reply
// did not parse). The second value is what lets a caller report the difference
// instead of claiming a rewrite happened.
//
// This is the round-1 query set. It runs BEFORE any retrieval, which is the
// point: the alternative — searching the question as asked and only rewriting
// once the checker objects — spends a whole round on a query the model was never
// going to match, and does it on every question.
func RewriteQueries(ctx context.Context, model Model, question string, max int) ([]string, bool) {
	plan := PlanQueries(ctx, model, question, max)
	if len(plan.Queries) == 0 {
		return nil, false
	}
	return plan.Queries, plan.Rewritten
}

// parseQueries reads the reply into queries, then applies the shape guards.
//
// Two passes on purpose. The strict one is the contract; the loose one exists
// because a model that wraps its JSON in prose or answers with a bulleted list
// is still offering usable queries, and discarding them all would send the run
// back to searching the raw question. RAGFlow parses its fanouts the same way
// (graph_fanout.go's parseFanouts falls back to splitting lines).
func parseQueries(reply string, max int) []string {
	candidates := parseQueriesStrict(reply)
	if len(candidates) == 0 {
		candidates = parseQueriesLoose(reply)
	}

	seen := map[string]bool{}
	out := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		query := collapse(strings.TrimSpace(candidate))
		if !queryLooksSearchable(query) {
			continue
		}
		key := strings.ToLower(query)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, query)
		if len(out) >= max {
			break
		}
	}
	return out
}

// parseQueriesStrict reads the documented JSON shape, tolerating the two
// variants a model actually produces: a bare array, and objects carrying the
// query under "query" or "question" (RAGFlow's query_rewriter accepts both —
// runtime/orchestrator/query_rewriter.go).
func parseQueriesStrict(reply string) []string {
	cleaned := stripCodeFence(strings.TrimSpace(reply))
	if cleaned == "" || (cleaned[0] != '{' && cleaned[0] != '[') {
		return nil
	}

	var payload struct {
		Queries []json.RawMessage `json:"queries"`
	}
	if err := json.Unmarshal([]byte(cleaned), &payload); err != nil {
		var bare []json.RawMessage
		if err := json.Unmarshal([]byte(cleaned), &bare); err != nil {
			return nil
		}
		payload.Queries = bare
	}

	out := make([]string, 0, len(payload.Queries))
	for _, entry := range payload.Queries {
		if text := queryFromEntry(entry); text != "" {
			out = append(out, text)
		}
	}
	return out
}

// queryFromEntry reads one element, which may be a string or an object.
func queryFromEntry(entry json.RawMessage) string {
	var text string
	if err := json.Unmarshal(entry, &text); err == nil {
		return text
	}
	var object map[string]any
	if err := json.Unmarshal(entry, &object); err != nil {
		return ""
	}
	for _, key := range []string{"query", "question", "sub_query", "search_query"} {
		if value, ok := object[key].(string); ok && strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

// parseDeclaredSlots reads the reply's slot declaration.
//
// Independent of the query parsing, and tolerant in the same way: the declaration
// arrives in the documented object, but a model that wrapped its JSON in prose or
// wrote the act words as one comma-separated string is still telling us the shape
// of the answer, and discarding that would silently drop the run back to
// single-value behaviour. Nothing here INVENTS a declaration: a reply with no
// slots field declares none, and that is the common case.
func parseDeclaredSlots(reply string) []Slot {
	cleaned := stripCodeFence(strings.TrimSpace(reply))
	if cleaned == "" || cleaned[0] != '{' {
		return nil
	}
	var payload struct {
		Slots []json.RawMessage `json:"slots"`
	}
	if err := json.Unmarshal([]byte(cleaned), &payload); err != nil {
		return nil
	}
	var out []Slot
	for _, entry := range payload.Slots {
		slot, ok := slotFromEntry(entry)
		if !ok {
			continue
		}
		out = append(out, slot)
		if len(out) >= maxEnumSlots {
			break
		}
	}
	return out
}

// slotFromEntry reads one declared slot, which may be an object or — in a model's
// shorthand — a bare type string.
func slotFromEntry(entry json.RawMessage) (Slot, bool) {
	var object map[string]any
	if err := json.Unmarshal(entry, &object); err != nil {
		var kind string
		if err := json.Unmarshal(entry, &kind); err != nil {
			return Slot{}, false
		}
		return Slot{Type: strings.TrimSpace(kind)}, true
	}

	slot := Slot{
		Type:    trimStringField(object, "type", "kind", "shape"),
		Subject: trimStringField(object, "subject", "actor", "who"),
	}
	// RAGFlow calls the act words `scan` (prompts/action_initialize_state.md) and
	// reads them into Variable.Terms. Both names are accepted, because a model that
	// saw either the prompt or an example may write the other, and the value may be
	// a list or one separated string.
	for _, key := range []string{"scan", "terms", "acts", "action_words"} {
		slot.Terms = append(slot.Terms, stringList(object[key])...)
	}
	slot.Terms = dedupeTerms(slot.Terms, CoverageActWordsMax)

	if slot.Type == "" && slot.Subject == "" && len(slot.Terms) == 0 {
		return Slot{}, false
	}
	return slot, true
}

// trimStringField reads the first of keys whose value is a non-empty string.
func trimStringField(object map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := object[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// stringList reads a value that may be a list of strings, one separated string, or
// nothing.
func stringList(value any) []string {
	switch typed := value.(type) {
	case string:
		return strings.FieldsFunc(typed, func(r rune) bool {
			return r == ',' || r == '，' || r == '|' || r == '｜' || r == ';' || r == '；'
		})
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok {
				out = append(out, text)
			}
		}
		return out
	}
	return nil
}

// dedupeTerms trims, drops blanks, dedupes case-insensitively and caps.
func dedupeTerms(terms []string, max int) []string {
	var out []string
	for _, term := range terms {
		out = appendUnique(out, term)
		if max > 0 && len(out) >= max {
			break
		}
	}
	return out
}

// parseQueriesLoose takes the ITEMS of a bulleted or numbered list, so a reply
// that ignored the JSON instruction still yields its queries.
//
// Only marked lines count, which is the whole of the difference from "split on
// newlines": a model that answers in prose writes a lead-in ("Here are the
// queries:") and then the list, and taking every line makes the lead-in a
// search query. The mark is the only structural signal separating the two, so
// requiring it is what keeps the fallback from inventing a query.
func parseQueriesLoose(reply string) []string {
	var out []string
	for _, line := range strings.Split(reply, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !hasListMarker(line) {
			continue
		}
		line = strings.TrimLeft(line, "-*• \t")
		// "1. query" and "1) query" are numbered items.
		digits := 0
		for digits < len(line) && line[digits] >= '0' && line[digits] <= '9' {
			digits++
		}
		if digits > 0 && digits < len(line) && (line[digits] == '.' || line[digits] == ')') {
			line = line[digits+1:]
		}
		line = strings.Trim(strings.TrimSpace(line), `",`)
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, line)
		}
	}
	return out
}

// hasListMarker reports whether a line opens a list item.
func hasListMarker(line string) bool {
	if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") || strings.HasPrefix(line, "•") {
		return true
	}
	digits := 0
	for digits < len(line) && line[digits] >= '0' && line[digits] <= '9' {
		digits++
	}
	return digits > 0 && digits < len(line) && (line[digits] == '.' || line[digits] == ')')
}

// stripCodeFence unwraps ```json … ``` around a reply.
func stripCodeFence(reply string) string {
	if !strings.HasPrefix(reply, "```") {
		return reply
	}
	reply = strings.TrimPrefix(reply, "```")
	if index := strings.Index(reply, "\n"); index >= 0 {
		// Drop the language tag on the fence line.
		first := strings.TrimSpace(reply[:index])
		if !strings.Contains(first, "{") && !strings.Contains(first, "[") {
			reply = reply[index+1:]
		}
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(reply), "```"))
}

// queryLooksSearchable rejects a candidate that would poison retrieval: an
// answer, a fragment, or something so long it is really prose.
func queryLooksSearchable(query string) bool {
	if query == "" || len([]rune(query)) > maxSearchQueryChars {
		return false
	}
	if len(strings.Fields(query)) > maxSearchQueryWords {
		return false
	}
	lower := strings.ToLower(query)
	for _, marker := range answerMarkers {
		if strings.Contains(lower, marker) {
			return false
		}
	}
	return true
}
