package agent

import (
	"strings"
	"testing"

	"freerag/internal/store"
)

func TestDecisionStateDropsWholePassagesFromTheEnd(t *testing.T) {
	// The pool grows with every round, so the state has to be bounded or the
	// model is handed a prompt past its window — which the sidecar rejects
	// rather than trims. The cut keeps whole passages: half a passage is worse
	// than one fewer passage, because it reads as a complete one.
	evidence := []store.Hit{
		hit("a.pdf", "c0", strings.Repeat("alpha ", 4)),   // 24 chars
		hit("a.pdf", "c1", strings.Repeat("bravo ", 4)),   // 24 chars
		hit("a.pdf", "c2", strings.Repeat("charlie ", 3)), // 24 chars
	}

	state := renderDecisionState(evidence, 30)
	if !strings.Contains(state, "alpha") {
		t.Fatalf("the first passage was dropped: %q", state)
	}
	if strings.Contains(state, "bravo") || strings.Contains(state, "charlie") {
		t.Fatalf("a passage past the limit was included: %q", state)
	}
	if len(state) > 30 {
		t.Fatalf("state is %d chars, over the 30-char limit", len(state))
	}
}

func TestDecisionStateTruncatesASingleOversizedPassage(t *testing.T) {
	// One long document must still be judgeable: dropping it would leave the
	// checker with an empty state and turn a long passage into no verdict.
	state := renderDecisionState([]store.Hit{hit("a.pdf", "c0", strings.Repeat("x", 500))}, 100)
	if len([]rune(state)) != 100 {
		t.Fatalf("state is %d chars, want it truncated to 100", len([]rune(state)))
	}
}

func TestDecisionStateWithoutALimitKeepsEverything(t *testing.T) {
	evidence := []store.Hit{hit("a.pdf", "c0", "first"), hit("a.pdf", "c1", "second")}
	state := renderDecisionState(evidence, 0)
	if !strings.Contains(state, "first") || !strings.Contains(state, "second") {
		t.Fatalf("an unlimited state dropped a passage: %q", state)
	}
}

func TestDecisionStateSkipsEmptyPassages(t *testing.T) {
	state := renderDecisionState([]store.Hit{
		hit("a.pdf", "c0", "   "),
		hit("a.pdf", "c1", "real passage"),
	}, 1000)
	if strings.HasPrefix(state, "\n") || !strings.Contains(state, "real passage") {
		t.Fatalf("state = %q", state)
	}
	if len(state) != len("real passage") {
		t.Fatalf("state = %q, want only the non-empty passage", state)
	}
}

// The cap the checker actually applies is a setting, and it must be large
// enough not to silently drop most of a normal pool.
func TestLayaCheckerMaxStateCharsDefaults(t *testing.T) {
	if got := (LayaChecker{}).maxStateChars(); got != DefaultMaxStateChars {
		t.Fatalf("default cap = %d, want %d", got, DefaultMaxStateChars)
	}
	configured := LayaChecker{MaxStateChars: 1234}
	if got := configured.maxStateChars(); got != 1234 {
		t.Fatalf("configured cap = %d, want 1234", got)
	}
	// The cap trades context against the cost of a decision, and both ends of
	// that trade are measured (see DefaultMaxStateChars): below the lower bound
	// a multi-round pool loses whole passages, above the upper one a CPU check
	// costs minutes. Both ends are the reason the constant exists, so both are
	// asserted rather than a magic number.
	if DefaultMaxStateChars < 4000 {
		t.Fatalf("default cap %d is too small for a multi-round pool", DefaultMaxStateChars)
	}
	if DefaultMaxStateChars > 16000 {
		t.Fatalf("default cap %d is past the affordable band: on CPU a check that long costs minutes",
			DefaultMaxStateChars)
	}
}
