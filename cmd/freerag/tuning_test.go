package main

import (
	"testing"
	"time"

	"freerag/internal/agent"
)

// The compression and streaming knobs are read from the environment once, at
// base-open time. Two things have to hold, and both were learned the hard way
// elsewhere in this project:
//
//   - OFF is the default, and a stray tunable must not turn the feature on. A
//     deployment that sets FREERAG_REFRAG_GIST_CHARS and nothing else must get
//     the pre-REFRAG prompt, not a compression it never asked for.
//   - A nonsense value falls back to the default rather than to zero. Zero
//     ExpandTop means "compress every passage including the best one", which is
//     a silently worse prompt — the failure mode a config typo must not have.
func TestRefragConfigRequiresAnExplicitSwitch(t *testing.T) {
	t.Setenv("FREERAG_REFRAG", "")
	t.Setenv("FREERAG_REFRAG_GIST_CHARS", "80")
	if cfg := refragConfig(); cfg.Enabled {
		t.Fatalf("a tunable alone enabled compression: %+v", cfg)
	}
}

func TestRefragConfigRefusesUnknownSwitchValues(t *testing.T) {
	for _, value := range []string{"maybe", "2", "0", "off"} {
		t.Setenv("FREERAG_REFRAG", value)
		if cfg := refragConfig(); cfg.Enabled {
			t.Fatalf("%q enabled compression", value)
		}
	}
}

func TestRefragConfigReadsItsTunables(t *testing.T) {
	t.Setenv("FREERAG_REFRAG", "1")
	t.Setenv("FREERAG_REFRAG_EXPAND_TOP", "7")
	t.Setenv("FREERAG_REFRAG_GIST_CHARS", "512")

	cfg := refragConfig()
	if !cfg.Enabled || cfg.ExpandTop != 7 || cfg.GistChars != 512 {
		t.Fatalf("cfg = %+v, want enabled with 7 / 512", cfg)
	}
}

func TestRefragConfigFallsBackOnNonsenseValues(t *testing.T) {
	t.Setenv("FREERAG_REFRAG", "on")
	t.Setenv("FREERAG_REFRAG_EXPAND_TOP", "-3")
	t.Setenv("FREERAG_REFRAG_GIST_CHARS", "0")

	cfg := refragConfig()
	if cfg.ExpandTop != agent.DefaultRefragExpandTop {
		t.Fatalf("ExpandTop = %d, want the default after a negative value", cfg.ExpandTop)
	}
	if cfg.GistChars != agent.DefaultRefragGistChars {
		t.Fatalf("GistChars = %d, want the default after a zero value", cfg.GistChars)
	}
}

// Zero ExpandTop is a legitimate setting — "represent every passage, expand
// none" — so it must survive, unlike a negative one.
func TestRefragConfigAllowsZeroExpandTop(t *testing.T) {
	t.Setenv("FREERAG_REFRAG", "1")
	t.Setenv("FREERAG_REFRAG_EXPAND_TOP", "0")
	if cfg := refragConfig(); cfg.ExpandTop != 0 {
		t.Fatalf("ExpandTop = %d, want 0 to be honoured", cfg.ExpandTop)
	}
}

func TestStreamCoalescingIsOffByDefault(t *testing.T) {
	t.Setenv("FREERAG_STREAM_COALESCE_MS", "")
	if window := streamCoalesceWindow(); window != 0 {
		t.Fatalf("window = %s, want 0 by default", window)
	}

	t.Setenv("FREERAG_STREAM_COALESCE_MS", "120")
	if window := streamCoalesceWindow(); window != 120*time.Millisecond {
		t.Fatalf("window = %s, want 120ms", window)
	}

	t.Setenv("FREERAG_STREAM_COALESCE_MS", "-5")
	if window := streamCoalesceWindow(); window != 0 {
		t.Fatalf("window = %s, want 0 for a negative value", window)
	}
}
