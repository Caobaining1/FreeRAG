package tree

import (
	"path/filepath"
	"testing"

	"freerag/internal/store"
)

// corpus is two documents indexed the way the kernel indexes them: chunks into
// the store, and a tree built from exactly those chunks.
func corpus(t *testing.T) (*store.Store, *Index) {
	t.Helper()

	s := store.New()
	handbookChunks := handbook()
	for i := range handbookChunks {
		handbookChunks[i].ChunkID = filepath.Base(handbookChunks[i].DocID) + "-" + string(rune('a'+i))
	}
	policyChunks := policyDoc()
	for i := range policyChunks {
		policyChunks[i].ChunkID = filepath.Base(policyChunks[i].DocID) + "-" + string(rune('a'+i))
	}
	s.Add(handbookChunks)
	s.Add(policyChunks)

	ix, err := Open(filepath.Join(t.TempDir(), "tree.json"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	ix.BuildAndPut("handbook.pdf", "handbook.pdf", handbookChunks)
	ix.BuildAndPut("policy.pdf", "policy.pdf", policyChunks)
	return s, ix
}

// TestSearchReachesTheRightSection is the whole claim: a query about annual leave
// descends to the leave section and the returned passage names the sections above
// it. A tree search that finds the right sentence without the path is just BM25
// wearing a hat.
func TestSearchReachesTheRightSection(t *testing.T) {
	s, ix := corpus(t)

	result := Search(ix, s, "年假有多少天", DefaultOptions())
	if len(result.Hits) == 0 {
		t.Fatal("no hits")
	}

	var best *Hit
	for i := range result.Hits {
		if result.Hits[i].DocID == "handbook.pdf" && len(result.Hits[i].Path) > 0 {
			if best == nil || result.Hits[i].Score > best.Score {
				best = &result.Hits[i]
			}
		}
	}
	if best == nil {
		t.Fatalf("no routed hit in handbook.pdf; hits: %+v", brief(result.Hits))
	}
	if joinPath(best.Path) != "handbook.pdf > 员工手册 2025 > 第三章 考勤与假期 > 3.1 年假" {
		t.Fatalf("path = %q", joinPath(best.Path))
	}
	if best.AID == "" || best.NodeID == "" {
		t.Fatalf("routed hit carries no tree coordinate: %+v", best)
	}
	// The passage itself has to be the one under that section.
	if best.Chunk.PageNum != 9 {
		t.Fatalf("hit landed on page %d, want 9 (%s)", best.Chunk.PageNum, best.Chunk.ChunkID)
	}
}

// TestSearchNeverFallsBelowBM25 is the hard acceptance criterion from the plan.
//
// The tree may add what BM25 missed, and may reorder what it found, but the
// merged result may not contain fewer relevant passages than the keyword
// baseline. Every BM25 hit is therefore expected to survive into the result,
// which is only true because the fallback is fused rather than consulted "when
// the tree is unsure".
func TestSearchNeverFallsBelowBM25(t *testing.T) {
	s, ix := corpus(t)

	baseline := s.SearchIn("年假", 0, store.Filter{})
	if len(baseline) == 0 {
		t.Fatal("fixture: BM25 found nothing; the test would assert nothing")
	}
	result := Search(ix, s, "年假", DefaultOptions())

	present := map[string]bool{}
	for _, hit := range result.Hits {
		present[hit.DocID+"\x00"+hit.ChunkID] = true
	}
	for _, hit := range baseline {
		if !present[hit.Chunk.DocID+"\x00"+hit.Chunk.ChunkID] {
			t.Fatalf("BM25 hit %s/%s missing from the fused result; recall regressed below the baseline",
				hit.Chunk.DocID, hit.Chunk.ChunkID)
		}
	}
}

// TestSearchWithoutFallbackStillRoutes measures the tree on its own, with nothing
// to fall back on. This is the number that says whether the channel earns its
// keep: if it finds the section unaided, the fallback is insurance; if not, the
// fallback is the channel and the tree is decoration.
func TestSearchWithoutFallbackStillRoutes(t *testing.T) {
	s, ix := corpus(t)

	opts := DefaultOptions()
	opts.Fallback = false
	result := Search(ix, s, "年假", opts)

	if result.Trace.Routed == 0 {
		t.Fatalf("nothing was routed: %+v", result.Trace.Selected)
	}
	found := false
	for _, hit := range result.Hits {
		if hit.DocID == "handbook.pdf" && joinPath(hit.Path) != "" {
			found = true
		}
	}
	if !found {
		t.Fatalf("the tree alone did not reach handbook.pdf: %+v", brief(result.Hits))
	}
	if len(result.Hits) > opts.MaxDocs*opts.MaxAnchorsPerDoc {
		t.Fatalf("%d hits: more than MaxDocs could contribute", len(result.Hits))
	}
}

// TestTraceRecordsThePrunedSiblings is what makes the channel auditable: a wrong
// answer can be traced to the exact sibling that was cut, in the same run, rather
// than reconstructed afterwards.
func TestTraceRecordsThePrunedSiblings(t *testing.T) {
	s, ix := corpus(t)

	result := Search(ix, s, "年终奖什么时候发放", DefaultOptions())
	var pruned, kept int
	for _, route := range result.Trace.Routes {
		for _, step := range route.Steps {
			pruned += len(step.Pruned)
			kept += len(step.Kept)
		}
	}
	if kept == 0 {
		t.Fatal("no node survived any level")
	}
	if pruned == 0 {
		t.Fatal("nothing was pruned: either the beam is unbounded or the trace is not recording it")
	}
	if result.Trace.Documents != 2 || result.Trace.Routed == 0 {
		t.Fatalf("trace reports %d documents, %d routed", result.Trace.Documents, result.Trace.Routed)
	}
}

// TestIndexRoundTrip covers what a restart does: the structure comes back from
// disk, and the scorer is rebuilt from the store rather than from a second copy
// that could go stale.
func TestIndexRoundTrip(t *testing.T) {
	s, ix := corpus(t)
	path := filepath.Join(t.TempDir(), "nested", "tree.json")
	ix2, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Simulated fresh process: only the file carried over.
	for _, id := range ix.Docs() {
		ix2.Put(ix.Tree(id), nil)
	}
	if err := ix2.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reopened, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if reopened.Len() != 2 {
		t.Fatalf("reopened index holds %d documents, want 2", reopened.Len())
	}
	if reopened.Attached("handbook.pdf") {
		t.Fatal("the scorer must never survive a restart — it is derived from the store's text")
	}
	for _, id := range reopened.Docs() {
		reopened.Put(reopened.Tree(id), s.List(id, 0, 0, 0))
	}
	result := Search(reopened, s, "住宿报销上限", DefaultOptions())
	ok := false
	for _, hit := range result.Hits {
		if hit.DocID == "policy.pdf" && len(hit.Path) > 0 {
			ok = true
		}
	}
	if !ok {
		t.Fatalf("after a reload the tree did not reach policy.pdf: %+v", brief(result.Hits))
	}
}

// TestUnknownDocumentIsNotSilentlySkipped guards document selection: a base whose
// documents were indexed before this channel existed has to say so, because the
// fallback quietly covers for it and its answers would otherwise look complete.
func TestUnknownDocumentIsNotSilentlySkipped(t *testing.T) {
	s, ix := corpus(t)
	s.Add([]store.Chunk{{
		ChunkID: "loose-1", Text: "一份没有建树的文档，讲的是年假制度。",
		DocID: "orphan.pdf", PageNum: 1, BlockType: "Text",
	}})

	result := Search(ix, s, "年假", DefaultOptions())
	if len(result.Trace.Untreed) != 1 || result.Trace.Untreed[0] != "orphan.pdf" {
		t.Fatalf("Untreed = %v, want [orphan.pdf]", result.Trace.Untreed)
	}
}

func brief(hits []Hit) []string {
	out := make([]string, 0, len(hits))
	for _, hit := range hits {
		out = append(out, hit.DocID+"/"+hit.ChunkID+" src="+hit.Source+" path="+joinPath(hit.Path))
	}
	return out
}

func joinPath(path []string) string {
	if len(path) == 0 {
		return ""
	}
	out := path[0]
	for _, part := range path[1:] {
		out += " > " + part
	}
	return out
}
