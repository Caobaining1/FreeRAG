package agent

import (
	"fmt"
	"strings"
)

// The enumeration strategy, ported from RAGFlow's slot table.
//
// Where it comes from: RAGFlow's agentic RAG declares, at plan time, a table of
// "fact slots" for the question (internal/rag/agentic-rag/runtime/coverage.go and
// prompts/action_initialize_state.md in the RAGFlow Go tree). A slot that asks for
// a SET of named elements — and that names both the elements' kind and the words
// the source uses for the deed — switches that run to enumeration: one recall per
// declared operand, one window per place the deed is stated, one verdict per
// window, and the members become the answer's list.
//
// WHY IT IS DECLARED AND NOT DETECTED: the gate reads the planner's own
// declaration — the slot's type, subject and terms — and never the question's
// wording. Nothing here pattern-matches "list all", "which", "how many": a
// question that LOOKS like an enumeration but whose answer is one value would
// otherwise pay the whole cost of the strategy, and the failure of a
// pattern-matched gate is silent (it spends the budget and answers as before).
// A table that does not declare the three things pays nothing — Coverage.Ok.
//
// What is deliberately NOT ported here, and why:
//
//   - the depth-3 slot tree and per-round gap PROMOTION (maxSlotDepth,
//     MergeSlotPatch). RAGFlow grows slots from the SCA's gaps across rounds; a
//     freerag sub-question has 2 rounds and no branch merging, so the table is
//     flat and fixed at declaration time.
//   - the window VERDICT pass (coverage_resolve.go: one model call per window,
//     batched by 8, up to 288 windows). Freerag asks the same question once over
//     the pooled passages instead — see ExtractMembers. The semantics kept are
//     the ones that matter: a member without a passage does not count.
//   - count-slot sync and batch/Jaccard fills (graph_slots.go). Those exist to
//     reconcile many parallel sessions writing one table; there is one writer
//     here.

// CoverageActWordsMax bounds the act words one slot may declare, as RAGFlow
// bounds it (runtime/coverage.go:26). The cap is not about prompt size: the act
// words become one retrieval call each, so a model that lists twenty of them
// spends twenty calls on one direction.
const CoverageActWordsMax = 10

// maxEnumSlots bounds how many slots a declaration may carry. RAGFlow caps the
// total at 8 across a grown tree; a flat table for one sub-question needs far
// fewer, and every slot is another direction to research.
const maxEnumSlots = 4

// enumMaxRounds is the round budget an enumeration is bounded to, from RAGFlow's
// `CoverageOf(...).Ok() && rounds > 2 → 2` (agentic_rag_graph.go:1642).
const enumMaxRounds = 2

// Slot is one declared thing the question wants.
//
// It carries no value: a slot says what to LOOK for, and the found members live
// beside it (see Result.Members). RAGFlow's Variable carries both because a
// session patches its value in place; nothing here writes a slot.
type Slot struct {
	// Type is the shape the planner declared for the answer: "entity", "person",
	// "list", "set", "count", "date", "number", ... Only the set-shaped ones
	// activate the strategy, and only when an element kind comes with them.
	Type string
	// Subject is the deed's actor as the planner declared it, alternatives
	// joined by '|' (a name and its aliases). Empty is allowed — "who was
	// killed" states the act without naming the actor.
	Subject string
	// Terms are the words the planner declared for the deed: the verbs the
	// source uses for it. A phrasing no act word covers is a passage no query
	// names, which is why the planner is asked for several.
	Terms []string
}

// Coverage is what a question asks for when its answer is a set of named
// members: an actor, the words the source uses for the deed, and the two
// declarations that say the answer is a set of names at all.
//
// It is the whole of the strategy's interface — two fields carried from the
// planner's own declaration (Actor / Acts) and two readings of the same
// declaration (Set / ItemKind). Everything the strategy needs is answered by "are
// these three things true", and everything it does is answered by "one recall per
// operand".
type Coverage struct {
	// Actor is the actor's declared forms, alternatives joined by '|'.
	Actor string
	// Acts are the verbs the planner declared for the deed.
	Acts []string
	// Set reports that the table asked for a count, a set or a list — the answer
	// is a SET rather than one value.
	Set bool
	// ItemKind is what the answer's elements are, when the table declares it:
	// "person", "entity", "dataset", "list", "set". Empty when the answer is not
	// a list of elements.
	ItemKind string
}

// CoverageItemKinds reads the planner's slot vocabulary as what the answer
// enumerates. A kind not in this table is not a list of elements (a date, a
// number, a phrase), and the gate pays nothing for it.
var CoverageItemKinds = map[string]string{
	"entity":  "entity",
	"person":  "person",
	"dataset": "dataset",
	"list":    "list",
	"set":     "set",
}

// CoverageOf reads the declaration out of the planner's own table.
//
// Nothing here reads the question or a passage: the shape and the deed are what
// the PLANNER said about the answer, so the gate cannot be fooled by how a
// passage happens to be worded. A permissive reading instead spends the strategy
// on questions that assemble nothing.
func CoverageOf(slots []Slot) Coverage {
	var c Coverage
	for _, slot := range slots {
		kind := strings.ToLower(strings.TrimSpace(slot.Type))
		switch kind {
		case "count", "set", "list":
			c.Set = true
		}
		if k, ok := CoverageItemKinds[kind]; ok && c.ItemKind == "" {
			c.ItemKind = k
		}
		if c.Actor == "" {
			c.Actor = strings.TrimSpace(slot.Subject)
		}
		for _, term := range slot.Terms {
			if term = strings.TrimSpace(term); term != "" {
				c.Acts = appendUnique(c.Acts, term)
			}
		}
	}
	return c
}

// Ok reports whether this table asks for an enumeration: a SET of ELEMENTS whose
// deed the planner wrote the words for.
//
// The conjunction is the whole gate, and it is what separates a set of elements
// from a count of EVENTS — the planner declares act words for the latter too
// ("how many times did it happen" names its verb), and no element an enumeration
// could return changes a count of events. Held only by the element-carrying half,
// the strategy would run on single-value questions, where it can only add cost.
func (c Coverage) Ok() bool {
	return c.Set && c.ItemKind != "" && len(c.Acts) > 0
}

// ActsAll is the act vocabulary the enumeration filters by: the planner's
// declaration, and nothing else.
func (c Coverage) ActsAll() []string {
	return append([]string(nil), c.Acts...)
}

// Actors splits the declared actor into its alternatives, which are what the
// corpus has to be read with: one source words one person several ways.
func (c Coverage) Actors() []string {
	var out []string
	for _, part := range strings.FieldsFunc(c.Actor, func(r rune) bool {
		return r == '|' || r == '｜' || r == '/' || r == '、' || r == ',' || r == '，'
	}) {
		if part = strings.TrimSpace(part); part != "" {
			out = appendUnique(out, part)
		}
	}
	return out
}

// Operands is the recall list: the actor's alternatives and the act words,
// deduped.
//
// ONE entry per operand is the point, and it is the part of RAGFlow's design that
// is easiest to get wrong: asking once per (actor, act) PAIR recalls the same
// operand over and over, so a common word fills its recall bound before the rarer
// ones are ever reached. The actor and every act word are recalled exactly once.
func (c Coverage) Operands() []string {
	var out []string
	for _, a := range c.Actors() {
		out = appendUnique(out, a)
	}
	for _, act := range c.Acts {
		if act = strings.TrimSpace(act); act != "" {
			out = appendUnique(out, act)
		}
	}
	return out
}

// Member is one named element the enumeration found, with the passage that
// states it.
//
// The chunk id is what makes a member count: RAGFlow's definition of a countable
// member is an item that carries a passage reference (AnchoredItems over
// slots.Value.Anchored, runtime/items.go), and the reason is the same here — a
// name the model produced from its own memory is exactly the answer this strategy
// must not accept, because the whole point is a list the corpus supports.
type Member struct {
	Name    string `json:"name"`
	ChunkID string `json:"chunk_id,omitempty"`
	DocID   string `json:"doc_id,omitempty"`
	Quote   string `json:"quote,omitempty"`
}

// AnchoredMembers keeps only the members that cite a passage, in first-seen
// order.
//
// A member with no passage is dropped rather than reported: it cannot be cited,
// so it would enter the answer as an unsupported name — and a list is read
// exactly as a set of names, with no marker on the entries that were invented.
func AnchoredMembers(members []Member) []Member {
	out := make([]Member, 0, len(members))
	seen := map[string]bool{}
	for _, member := range members {
		name := strings.TrimSpace(member.Name)
		if name == "" || member.ChunkID == "" {
			continue
		}
		key := strings.ToLower(name)
		if seen[key] {
			continue
		}
		seen[key] = true
		member.Name = name
		out = append(out, member)
	}
	return out
}

// MergeMembers folds the per-sub-question member lists into one, keeping the
// first anchored spelling of each name.
//
// Deduped by NAME, not by passage — which is the opposite of how evidence is
// merged (see kbinfo.add, where the same text in two documents is two citations).
// The difference is what the output is: a citation is a link, so dropping the
// second loses a document, while a member is an entry in a list, so two spellings
// of one name are one entry and listing it twice reads as two members.
func MergeMembers(groups ...[]Member) []Member {
	var all []Member
	for _, group := range groups {
		all = append(all, group...)
	}
	return AnchoredMembers(all)
}

// OperandCalls turns the declared operands into one retrieval call each.
//
// This is the part of the strategy that is NOT a model decision. RAGFlow's
// enumerate pass issues one recall per operand itself, in one pass
// (coverage_enumerate.go), and its loop never chooses WHICH operand matters: an
// operand exists because the declaration said that is how the actor or the deed is
// worded, and a wording nobody searches is a passage nobody reads.
//
// Measured on a real question, offering the operands to the tool chooser instead
// cost the whole run: the chooser picks ONE candidate per round, the run has two
// rounds, and the operand that would have matched ("New York Yankees") was never
// searched — the round went to "acquisition" and returned nothing.
//
// Capped by the caller's per-turn budget: past that the calls would exceed
// ActionMaxTurns and the turn would be truncated anyway.
func OperandCalls(operands []string, max int) []ToolCall {
	if max <= 0 {
		max = len(operands)
	}
	calls := make([]ToolCall, 0, len(operands))
	for _, operand := range operands {
		if len(calls) >= max {
			break
		}
		operand = strings.TrimSpace(operand)
		if operand == "" {
			continue
		}
		// Hybrid rather than the exact-match recall RAGFlow uses for this pass:
		// the four tools here are fixed (docs/plan.md §6.7) and hybrid carries a
		// keyword leg, so one operand still recalls the passages that state it
		// literally while an alias also gets the semantic half. The property that
		// matters — one call per operand, all of them — is the one preserved.
		calls = append(calls, ToolCall{
			Name:      ToolHybridSearch,
			Arguments: map[string]any{"query": operand},
		})
	}
	return calls
}

// RenderMemberRecord renders the members for the answer prompt.
//
// This is RAGFlow's RenderSlotRecord (graph_slots.go) reduced to what a freerag
// run has: the members, their passages, and the count. The count is stated
// because the answer is read as a complete list — "enumerated members: 7" is what
// lets the writer say it found seven rather than implying there are seven in the
// corpus — and the passages are stated because a list without them is a list the
// writer cannot cite.
func RenderMemberRecord(question string, members []Member) string {
	if len(members) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Enumerated members (already extracted from the evidence for this question):\n")
	for _, member := range members {
		quote := collapse(strings.TrimSpace(member.Quote))
		if quote != "" {
			fmt.Fprintf(&b, "- %s — %s\n", member.Name, truncateRunes(quote, 120))
			continue
		}
		fmt.Fprintf(&b, "- %s\n", member.Name)
	}
	fmt.Fprintf(&b, "Enumerated members across all evidence: %d\n", len(members))
	b.WriteString("List every one of them in the answer, citing the passage each came from. " +
		"Do not add a name that is not listed here, and do not drop one.\n")
	return b.String()
}

// appendUnique adds value unless it is already present, compared
// case-insensitively.
//
// Case-insensitive because the values it dedupes are names and act words read off
// a model's declaration, where "Ohio State" and "ohio state" are one operand, and
// each duplicate would be another retrieval call.
func appendUnique(values []string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return values
	}
	key := strings.ToLower(value)
	for _, existing := range values {
		if strings.ToLower(existing) == key {
			return values
		}
	}
	return append(values, value)
}
