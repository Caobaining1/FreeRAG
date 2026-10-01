package agent

import (
	"testing"

	"freerag/internal/store"
)

func hit(docID, chunkID, text string) store.Hit {
	return store.Hit{Chunk: store.Chunk{DocID: docID, ChunkID: chunkID, Text: text}}
}

func TestKBInfoAccumulatesAcrossRounds(t *testing.T) {
	info := newKBInfo()
	if !info.empty() || info.len() != 0 {
		t.Fatal("a new kbinfo is not empty")
	}

	if added := info.add([]store.Hit{hit("a.pdf", "c0", "first")}); added != 1 {
		t.Fatalf("added = %d, want 1", added)
	}
	if added := info.add([]store.Hit{hit("a.pdf", "c1", "second")}); added != 1 {
		t.Fatalf("added = %d, want 1", added)
	}

	if info.len() != 2 {
		t.Fatalf("len = %d, want 2", info.len())
	}
	if info.empty() {
		t.Fatal("a pool with two passages reports empty")
	}
	// Arrival order is what a citation's [n] refers to, so it must be preserved.
	if info.pool()[0].Chunk.Text != "first" || info.pool()[1].Chunk.Text != "second" {
		t.Fatalf("the pool is out of order: %v", info.pool())
	}
}

func TestKBInfoDeduplicatesTheSamePassage(t *testing.T) {
	// Two calls can reach one chunk by different routes — a keyword match and a
	// dense one — and the checker counts passages, not queries: a duplicate would
	// inflate the evidence and read as corroboration.
	info := newKBInfo()
	info.add([]store.Hit{hit("a.pdf", "c0", "passage")})

	if added := info.add([]store.Hit{hit("a.pdf", "c0", "passage")}); added != 0 {
		t.Fatalf("a duplicate counted as %d new passage(s)", added)
	}
	if info.len() != 1 {
		t.Fatalf("len = %d, want 1", info.len())
	}

	// Same chunk id under a different document is a different passage.
	if added := info.add([]store.Hit{hit("b.pdf", "c0", "passage")}); added != 1 {
		t.Fatalf("a chunk id is being reused as a key across documents (added=%d)", added)
	}
}

func TestKBInfoDeduplicatesTheSameTextUnderDifferentChunkIDs(t *testing.T) {
	// The same words reach the pool twice from one file: a caption exists both
	// as its own Caption block and again inside the merged FigureWithCaption
	// that absorbed it, and overlapping chunks repeat the sentences between
	// them. Identity alone does not catch this, and the repeat spends the
	// checker's budget while reading as corroboration.
	info := newKBInfo()
	info.add([]store.Hit{hit("a.pdf", "c0", "Figure 1: the overall framework")})

	if added := info.add([]store.Hit{hit("a.pdf", "c7", "Figure 1: the overall framework")}); added != 0 {
		t.Fatalf("the same text under a new chunk id counted as %d new passage(s)", added)
	}
	if info.len() != 1 {
		t.Fatalf("len = %d, want 1", info.len())
	}
}

func TestKBInfoContentMatchIgnoresWhitespaceAndCase(t *testing.T) {
	// Two copies of one caption differ in wrapping and capitalisation; that is
	// not two passages.
	info := newKBInfo()
	info.add([]store.Hit{hit("a.pdf", "c0", "The Overall   Framework\nof MEDAL")})

	if added := info.add([]store.Hit{hit("a.pdf", "c1", "the overall framework of medal")}); added != 0 {
		t.Fatalf("a whitespace/case variant counted as %d new passage(s)", added)
	}
}

func TestKBInfoKeepsTheSameTextFromAnotherDocument(t *testing.T) {
	// Deliberate: the same sentence in two documents is two citations, and
	// dropping the second would lose the link to it.
	info := newKBInfo()
	info.add([]store.Hit{hit("a.pdf", "c0", "identical sentence")})

	if added := info.add([]store.Hit{hit("b.pdf", "c0", "identical sentence")}); added != 1 {
		t.Fatalf("a passage from a second document was dropped (added=%d)", added)
	}
	if info.len() != 2 {
		t.Fatalf("len = %d, want 2", info.len())
	}
}

func TestKBInfoDoesNotCollapseHitsWithoutAChunkID(t *testing.T) {
	// An empty chunk id must not become a shared key: every such hit from one
	// document would collapse into the first, which is a silent loss of
	// evidence rather than a deduplication.
	info := newKBInfo()
	info.add([]store.Hit{hit("a.pdf", "", "first passage")})

	if added := info.add([]store.Hit{hit("a.pdf", "", "second passage")}); added != 1 {
		t.Fatalf("a distinct passage with no chunk id was dropped (added=%d)", added)
	}
	if info.len() != 2 {
		t.Fatalf("len = %d, want 2", info.len())
	}

	// And the same text with no chunk id is still a duplicate.
	if added := info.add([]store.Hit{hit("a.pdf", "", "first passage")}); added != 0 {
		t.Fatalf("an identical passage with no chunk id counted as %d new", added)
	}
}

func TestKBInfoToleratesNil(t *testing.T) {
	// The checker may be handed one on a path with nothing retrieved, and a nil
	// dereference there would be a crash rather than an empty-pool verdict.
	var info *kbinfo
	if !info.empty() || info.len() != 0 || info.pool() != nil {
		t.Fatal("a nil kbinfo does not read as empty")
	}
}
