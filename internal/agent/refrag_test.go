package agent

import (
	"strconv"
	"strings"
	"testing"

	"freerag/internal/store"
)

func refragHits(n int, body string) []store.Hit {
	hits := make([]store.Hit, 0, n)
	for i := 0; i < n; i++ {
		hits = append(hits, store.Hit{Chunk: store.Chunk{
			ChunkID: "c" + string(rune('0'+i)),
			DocID:   "paper.pdf",
			PageNum: i + 1,
			Text:    body,
		}})
	}
	return hits
}

// A long body whose first sentence is complete, so the gist has a boundary to
// cut at. Well past the default gist of 240 characters.
var refragLongBody = "This section reports the measured throughput of the retriever. " +
	strings.Repeat("It then describes the harness and the hardware in some detail. ", 11)

// The disabled configuration must reproduce the pre-REFRAG prompt byte for byte.
//
// This is the baseline the whole change is measured against. If turning the
// feature off still altered the prompt, every comparison would be taken against
// a moving baseline and the feature's own effect would be unreadable.
func TestRefragDisabledRendersTheUncompressedPrompt(t *testing.T) {
	hits := refragHits(8, refragLongBody)

	plain, stats := renderEvidenceWith("why?", hits, 0, "", RefragConfig{})
	if plain != renderEvidence("why?", hits, 0, "") {
		t.Fatal("a disabled RefragConfig changed the rendered prompt")
	}
	if stats.Compressed != 0 || stats.EvidenceChars != stats.EvidenceCharsRaw {
		t.Fatalf("compression reported while disabled: %+v", stats)
	}
	if line := stats.TraceLine(); line != "" {
		t.Fatalf("a run that compressed nothing still traced: %q", line)
	}
}

// The sense step: the top-ranked passages stay whole, the rest are compressed.
func TestRefragExpandsTopRankedAndCompressesTheRest(t *testing.T) {
	hits := refragHits(8, refragLongBody)
	cfg := RefragConfig{Enabled: true, ExpandTop: 3, GistChars: 200}

	block, stats := renderEvidenceWith("why?", hits, 0, "", cfg)

	if stats.Expanded != 3 {
		t.Fatalf("expanded = %d, want the 3 highest-ranked passages", stats.Expanded)
	}
	if stats.Compressed != 5 {
		t.Fatalf("compressed = %d, want the remaining 5", stats.Compressed)
	}
	if stats.EvidenceChars >= stats.EvidenceCharsRaw {
		t.Fatalf("compression did not shrink the block: %d -> %d",
			stats.EvidenceCharsRaw, stats.EvidenceChars)
	}

	// Every passage keeps its number and its header. A citation the model
	// writes has to resolve, and the rank stop is the only thing the model
	// sees that says where a passage came from.
	for i := 1; i <= len(hits); i++ {
		if !strings.Contains(block, "["+strconv.Itoa(i)+"] (paper.pdf p."+strconv.Itoa(i)+")") &&
			!strings.Contains(block, "["+strconv.Itoa(i)+"] (paper.pdf p."+strconv.Itoa(i)+" | "+refragMarker+")") {
			t.Fatalf("passage %d lost its numbered header:\n%s", i, block)
		}
	}
}

// A compressed passage must say so in its header.
//
// The decoder reading this is a 4B model that was never trained on compressed
// chunks, so an abbreviated passage that is not labelled reads as a complete
// one and invites an answer drawn from half a paragraph.
func TestRefragLabelsCompressedPassagesInTheirHeader(t *testing.T) {
	hits := refragHits(4, refragLongBody)
	block, stats := renderEvidenceWith("why?", hits, 0, "",
		RefragConfig{Enabled: true, ExpandTop: 1, GistChars: 200})

	if !strings.Contains(block, "| "+refragMarker+")") {
		t.Fatalf("no passage was labelled as compressed:\n%s", block)
	}
	for i := 2; i <= 4; i++ {
		marker := "[" + strconv.Itoa(i) + "] (paper.pdf p." + strconv.Itoa(i) + " | " + refragMarker + ")"
		if !strings.Contains(block, marker) {
			t.Fatalf("passage %d is not labelled compressed:\n%s", i, block)
		}
	}
	if stats.Compressed != 3 {
		t.Fatalf("compressed = %d, want 3", stats.Compressed)
	}
}

// Compressing a passage that is not longer than the gist loses information and
// saves nothing, so it is expanded on size rather than on rank.
func TestRefragLeavesShortPassagesWhole(t *testing.T) {
	short := "Abstract. We present a retriever."
	hits := refragHits(10, short)
	_, stats := renderEvidenceWith("why?", hits, 0, "",
		RefragConfig{Enabled: true, ExpandTop: 1, GistChars: 240})

	if stats.Compressed != 0 {
		t.Fatalf("compressed %d passage(s) that would not have shrunk: %+v", stats.Compressed, stats)
	}
}

// The gist cuts at a sentence end whenever one falls in the tail of its window.
//
// The leading fragment of a passage is what names its subject, so the cut has
// to leave a whole clause behind: a mid-word cut hands the model a subject with
// no predicate. The search is restricted to the last 40% of the window so the
// gist is not thrown away for an early period — see the fallback test below.
func TestCompressPassageCutsAtASentenceBoundary(t *testing.T) {
	text := strings.Repeat("A long clause that keeps the gist window full. ", 20)
	gist := compressPassage(text, 200)

	if !strings.HasSuffix(gist, "[…]") {
		t.Fatalf("the cut is not marked: %q", gist)
	}
	body := strings.TrimSuffix(gist, " […]")
	if !strings.HasSuffix(body, ".") {
		t.Fatalf("the gist did not end at a sentence boundary: %q", body)
	}
	if runes := len([]rune(body)); runes > 200 || runes < 120 {
		t.Fatalf("the gist used %d runes of a 200-rune window", runes)
	}
}

// When no sentence ends in the tail of the window, the cut falls back to
// whitespace rather than mid-word.
func TestCompressPassageFallsBackToAWordBoundary(t *testing.T) {
	// A single unbroken sentence, so there is no boundary to prefer.
	text := strings.Repeat("midsentence ", 40)
	gist := compressPassage(text, 60)

	body := strings.TrimSuffix(gist, " […]")
	if strings.HasSuffix(body, "mid") || strings.HasSuffix(body, "mi") {
		t.Fatalf("the gist was cut mid-word: %q", body)
	}
	if !strings.HasSuffix(body, "midsentence") {
		t.Fatalf("the gist did not fall back to a word boundary: %q", body)
	}
}

// A gist shorter than the guard is refused: a config typo must not silently
// reduce every passage to a stub.
func TestRefragRefusesAGistTooShortToCarryAClaim(t *testing.T) {
	cfg := RefragConfig{Enabled: true, ExpandTop: 1, GistChars: 3}.normalized()
	if cfg.GistChars != DefaultRefragGistChars {
		t.Fatalf("GistChars = %d, want the default %d", cfg.GistChars, DefaultRefragGistChars)
	}
}

// A negative ExpandTop would mean "compress everything including the best
// passage", which is a silently worse prompt rather than an error.
func TestRefragClampsNegativeExpandTop(t *testing.T) {
	cfg := RefragConfig{Enabled: true, ExpandTop: -5, GistChars: 200}.normalized()
	if cfg.ExpandTop != 0 {
		t.Fatalf("ExpandTop = %d, want 0", cfg.ExpandTop)
	}
}

// The stats have to describe the passages that were actually rendered. Counting
// the builder's whole length instead would include the question and the
// "Evidence:" scaffolding, and the disabled configuration would then report a
// ratio below 1.0 — the number exists to make exactly that comparison.
func TestRefragStatsDescribeTheRenderedPassages(t *testing.T) {
	hits := refragHits(6, refragLongBody)
	_, stats := renderEvidenceWith("why?", hits, 0, "",
		RefragConfig{Enabled: true, ExpandTop: 2, GistChars: 200})

	if stats.Passages != len(hits) {
		t.Fatalf("Passages = %d, want %d", stats.Passages, len(hits))
	}
	if stats.Expanded+stats.Compressed != len(hits) {
		t.Fatalf("expanded+compressed = %d, want %d",
			stats.Expanded+stats.Compressed, len(hits))
	}
	if ratio := stats.Ratio(); ratio <= 1 {
		t.Fatalf("ratio = %.2f, want > 1 when compressing", ratio)
	}

	// The used figure is the sum of the passage bodies actually written, so it
	// must be smaller than the block by exactly the scaffolding's size.
	block, _ := renderEvidenceWith("why?", hits, 0, "", RefragConfig{Enabled: true, ExpandTop: 2, GistChars: 200})
	if stats.EvidenceChars >= len(block) {
		t.Fatalf("EvidenceChars = %d should exclude the scaffolding the block (%d) carries",
			stats.EvidenceChars, len(block))
	}
}

// The complex path applies the same compression to the merged pool.
func TestRenderSynthesisPromptCompressesTheMergedPool(t *testing.T) {
	hits := refragHits(7, refragLongBody)
	prompt, stats := renderSynthesisPrompt("why?", []string{"a", "b"}, hits, "", 0,
		RefragConfig{Enabled: true, ExpandTop: 2, GistChars: 200})

	if stats.Compressed != 5 {
		t.Fatalf("compressed = %d, want 5", stats.Compressed)
	}
	if !strings.Contains(prompt, refragMarker) {
		t.Fatalf("the merged pool was not compressed:\n%s", prompt)
	}
	if stats.EvidenceChars >= stats.EvidenceCharsRaw {
		t.Fatalf("the merged pool did not shrink: %d -> %d",
			stats.EvidenceCharsRaw, stats.EvidenceChars)
	}
}

// The stats ride on the result and into the trace, because a ratio that only
// exists inside the renderer is not evidence for anything.
func TestPromptStatsReachTheTrace(t *testing.T) {
	result := &Result{}
	setPromptStatsOn(result, &PromptStats{
		Passages: 5, Expanded: 2, Compressed: 3,
		EvidenceCharsRaw: 4000, EvidenceChars: 1000,
	})

	if result.PromptStats == nil || result.PromptStats.Compressed != 3 {
		t.Fatalf("stats did not reach the result: %+v", result.PromptStats)
	}
	if len(result.Trace) != 1 || !strings.Contains(result.Trace[0], "4.00x") {
		t.Fatalf("trace = %v, want one line carrying the ratio", result.Trace)
	}
}

// A run that compressed nothing still reports the block's cost, but adds no
// trace line.
//
// The figure is kept because it is the comparison the feature exists to make:
// the arm that compressed nothing has to state how large its evidence block
// was, or there is nothing to compare against. The trace line is dropped
// because a "[Refrag] 1.00x" on every run trains the reader to skip the one
// line that matters.
func TestPromptStatsReportedWithoutATraceLineWhenNothingShrank(t *testing.T) {
	result := &Result{}
	setPromptStatsOn(result, &PromptStats{Passages: 4, Expanded: 4,
		EvidenceCharsRaw: 900, EvidenceChars: 900})

	if result.PromptStats == nil {
		t.Fatal("an uncompressed prompt reported no cost at all")
	}
	if result.PromptStats.Ratio() != 1 {
		t.Fatalf("ratio = %.2f, want 1 for an uncompressed block",
			result.PromptStats.Ratio())
	}
	if len(result.Trace) != 0 {
		t.Fatalf("trace = %v, want empty for an uncompressed block", result.Trace)
	}
}

// A run that rendered no prompt reports nothing: a zero-valued block would
// claim knowledge about a prompt that never existed.
func TestPromptStatsOmittedWhenNoPromptWasRendered(t *testing.T) {
	result := &Result{}
	setPromptStatsOn(result, &PromptStats{})
	setPromptStatsOn(result, nil)

	if result.PromptStats != nil {
		t.Fatalf("stats recorded with no prompt rendered: %+v", result.PromptStats)
	}
	if len(result.Trace) != 0 {
		t.Fatalf("trace = %v, want empty", result.Trace)
	}
}
