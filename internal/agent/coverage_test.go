package agent

import (
	"context"
	"strings"
	"testing"

	"freerag/internal/store"
)

// ---- the gate ----

func TestCoverageGateNeedsAllThreeDeclarations(t *testing.T) {
	// The gate is a conjunction of three declarations the PLANNER made: a
	// set-shaped answer, the element kind, and the words for the deed. Each
	// negative below is a real shape RAGFlow's comment warns about, and each one
	// must cost the strategy nothing.
	cases := []struct {
		name  string
		slots []Slot
		want  bool
		why   string
	}{
		{
			name:  "a set of named elements with a deed",
			slots: []Slot{{Type: "list", Subject: "Michigan|UM", Terms: []string{"defeated", "beat"}}},
			want:  true,
			why:   "all three declarations are present",
		},
		{
			name:  "a count of events is not an enumeration",
			slots: []Slot{{Type: "count", Terms: []string{"happened"}}},
			want:  false,
			why:   "RAGFlow's example: the planner declares act words for a count too, and no member changes a count",
		},
		{
			name:  "a single value is not an enumeration",
			slots: []Slot{{Type: "entity", Terms: []string{"won"}}},
			want:  false,
			why:   "entity carries an item kind but is not set-shaped",
		},
		{
			name:  "a set with no deed words cannot be searched",
			slots: []Slot{{Type: "list", Subject: "Michigan"}},
			want:  false,
			why:   "the act words are the recall list; without them there is nothing to search",
		},
		{
			name:  "no declaration at all",
			slots: nil,
			want:  false,
			why:   "the common case: the model declared no slots",
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			if got := CoverageOf(testCase.slots).Ok(); got != testCase.want {
				t.Fatalf("Ok() = %v, want %v — %s", got, testCase.want, testCase.why)
			}
		})
	}
}

func TestCoverageOperandsRingOneRecallPerOperand(t *testing.T) {
	// "ONE entry per operand is the point": asking once per (actor, act) PAIR
	// recalls the same operand over and over, so a common word fills its bound
	// before the rarer ones are reached.
	cov := CoverageOf([]Slot{{
		Type:    "person",
		Subject: "关羽|关公",
		Terms:   []string{"斩", "杀"},
	}})

	got := cov.Operands()
	want := []string{"关羽", "关公", "斩", "杀"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("Operands() = %v, want %v (one entry per operand, not per pair)", got, want)
	}

	// A repeated operand is still one: an act word that is also an actor form
	// would otherwise be recalled twice.
	cov = CoverageOf([]Slot{{Type: "list", Subject: "斩", Terms: []string{"斩", "杀"}}})
	if got := cov.Operands(); strings.Join(got, ",") != "斩,杀" {
		t.Fatalf("Operands() = %v, want the duplicate dropped", got)
	}
}

func TestCoverageActorsSplitsTheDeclaredAlternatives(t *testing.T) {
	// One source words one person several ways, and the declared alternatives are
	// what the corpus is read with. Every separator RAGFlow splits on is a
	// separator a model might write.
	for _, subject := range []string{"关羽|关公", "关羽｜关公", "关羽/关公", "关羽、关公", "关羽,关公"} {
		got := CoverageOf([]Slot{{Type: "person", Subject: subject, Terms: []string{"斩"}}}).Actors()
		if len(got) != 2 || got[0] != "关羽" || got[1] != "关公" {
			t.Fatalf("Actors() from %q = %v, want [关羽 关公]", subject, got)
		}
	}
}

// ---- members ----

func TestAnchoredMembersDropsWhatCannotBeCited(t *testing.T) {
	// A name with no passage cannot be cited, so it would enter the answer as an
	// unsupported entry — and a list is read as a set of names, with nothing
	// marking the invented ones.
	members := AnchoredMembers([]Member{
		{Name: "华雄", ChunkID: "c1"},
		{Name: "荀正"},                 // no passage: dropped
		{Name: "  "},                 // blank: dropped
		{Name: "华雄", ChunkID: "c2"},  // same name again: one member
		{Name: "华 雄", ChunkID: "c3"}, // a different name: kept
	})

	if len(members) != 2 {
		t.Fatalf("kept %d member(s), want 2: %+v", len(members), members)
	}
	if members[0].ChunkID != "c1" {
		t.Fatalf("the first spelling must be the one kept, got %q", members[0].ChunkID)
	}
}

func TestMergeMembersDeduplicatesByNameAcrossSubQuestions(t *testing.T) {
	// The opposite rule from evidence: a citation is a link (two documents saying
	// one sentence are two citations), while a member is an entry in a list (two
	// spellings of one name are one entry).
	merged := MergeMembers(
		[]Member{{Name: "华雄", ChunkID: "c1"}},
		[]Member{{Name: "华雄", ChunkID: "c2"}, {Name: "颜良", ChunkID: "c3"}},
	)
	if len(merged) != 2 {
		t.Fatalf("merged %d member(s), want 2: %+v", len(merged), merged)
	}
}

func TestExtractMembersAnchorsAgainstThePool(t *testing.T) {
	// A model that cites the wrong passage but names a real member must still
	// produce a citable member, and one that names something the corpus does not
	// contain must produce nothing. Both are RAGFlow's resolveMemberAnchor rule:
	// the anchor is RESOLVED against the evidence rather than trusted.
	model := &scriptedModel{replies: []*Reply{{Content: `{"members":[
		{"name":"alpha","evidence":9},
		{"name":"alpha","evidence":1},
		{"name":"invented","evidence":1}
	]}`}}}
	evidence := []store.Hit{
		{Chunk: store.Chunk{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "nothing here"}},
		{Chunk: store.Chunk{ChunkID: "c1", DocID: "b.pdf", PageNum: 3, Text: "the alpha method was used"}},
	}

	members := ExtractMembers(context.Background(), model, "which methods were used?", evidence, 0)

	if len(members) != 1 {
		t.Fatalf("got %d member(s), want 1 (alpha, anchored): %+v", len(members), members)
	}
	if members[0].Name != "alpha" || members[0].ChunkID != "c1" {
		t.Fatalf("member = %+v, want alpha anchored to c1 (by name, not by the passage it claimed)",
			members[0])
	}
	if members[0].Quote == "" {
		t.Fatal("an anchored member must carry the quote it was found in")
	}
}

func TestParseMembersReadsWhatAModelActuallyWrites(t *testing.T) {
	cases := []struct {
		name  string
		reply string
		want  []string
	}{
		{
			"the documented object",
			`{"members":[{"name":"华雄","evidence":2},{"name":"颜良","evidence":3}]}`,
			[]string{"华雄", "颜良"},
		},
		{
			"a bare array of names",
			`["华雄","颜良"]`,
			[]string{"华雄", "颜良"},
		},
		{
			"a bulleted list in prose",
			"Here are the members:\n- 华雄 — 被斩\n- 颜良\n",
			[]string{"华雄", "颜良"},
		},
		{
			"a fenced object",
			"```json\n{\"members\":[{\"name\":\"华雄\"}]}\n```",
			[]string{"华雄"},
		},
		{
			"an empty list is an answer",
			`{"members":[]}`,
			nil,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			members := parseMembers(testCase.reply)
			got := make([]string, 0, len(members))
			for _, member := range members {
				got = append(got, member.Name)
			}
			if strings.Join(got, ",") != strings.Join(testCase.want, ",") {
				t.Fatalf("parseMembers = %v, want %v", got, testCase.want)
			}
		})
	}
}

// ---- the declaration, read out of the rewrite reply ----

func TestParseDeclaredSlotsReadsTheShapesAModelWrites(t *testing.T) {
	// RAGFlow calls the act words `scan`; a model may answer with a list or with
	// one separated string, and both are the same declaration.
	slots := parseDeclaredSlots(`{"queries":["x"],"slots":[
		{"type":"person","subject":"关羽|关公","scan":"斩, 杀"},
		{"type":"list","terms":["defeated"],"actor":"Michigan"}
	]}`)

	if len(slots) != 2 {
		t.Fatalf("got %d slot(s), want 2: %+v", len(slots), slots)
	}
	if slots[0].Type != "person" || slots[0].Subject != "关羽|关公" {
		t.Fatalf("slot 0 = %+v", slots[0])
	}
	if strings.Join(slots[0].Terms, ",") != "斩,杀" {
		t.Fatalf("one separated string must read as act words, got %v", slots[0].Terms)
	}
	if slots[1].Subject != "Michigan" {
		t.Fatalf("the actor alias must be read, got %q", slots[1].Subject)
	}
}

func TestParseDeclaredSlotsCapsWhatOneDeclarationMayCost(t *testing.T) {
	// Every act word is another retrieval call, so the cap is a cost bound: a
	// model that lists twenty of them spends twenty calls on one direction.
	terms := make([]string, 0, 20)
	for i := 0; i < 20; i++ {
		terms = append(terms, string(rune('a'+i)))
	}
	reply := `{"slots":[{"type":"list","scan":["` + strings.Join(terms, `","`) + `"]}]}`

	slots := parseDeclaredSlots(reply)
	if len(slots) != 1 {
		t.Fatalf("got %d slot(s)", len(slots))
	}
	if len(slots[0].Terms) != CoverageActWordsMax {
		t.Fatalf("act words = %d, want the cap %d", len(slots[0].Terms), CoverageActWordsMax)
	}
}

func TestAPlainReplyDeclaresNoSlots(t *testing.T) {
	// The overwhelmingly common case, and the one that must stay free: a reply
	// with no slots field is not an enumeration, and nothing here invents one.
	for _, reply := range []string{
		`{"queries":["alpha","beta"]}`,
		`["alpha","beta"]`,
		"alpha\nbeta",
	} {
		if slots := parseDeclaredSlots(reply); len(slots) != 0 {
			t.Fatalf("reply %q declared %+v, want no slots", reply, slots)
		}
	}
}

func TestOperandCallsIssuesOneRecallPerOperand(t *testing.T) {
	// One call per operand, and every operand gets one: the recall list is the
	// declaration, so nothing here ranks or picks.
	calls := OperandCalls([]string{"New York Yankees", "acquire", "  ", "acquired"}, 12)
	if len(calls) != 3 {
		t.Fatalf("got %d call(s), want 3 (the blank is not an operand): %+v", len(calls), calls)
	}
	for i, want := range []string{"New York Yankees", "acquire", "acquired"} {
		if calls[i].Name != ToolHybridSearch {
			t.Fatalf("call %d = %s, want %s", i, calls[i].Name, ToolHybridSearch)
		}
		if got := calls[i].Arguments["query"]; got != want {
			t.Fatalf("call %d queries %v, want %q", i, got, want)
		}
	}

	// The per-turn budget binds: past it the turn would be truncated anyway.
	if capped := OperandCalls([]string{"a", "b", "c", "d"}, 2); len(capped) != 2 {
		t.Fatalf("cap ignored: got %d call(s), want 2", len(capped))
	}
}
