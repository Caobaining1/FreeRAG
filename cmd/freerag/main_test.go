package main

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"freerag/internal/agent"
	"freerag/internal/ipc"
	"freerag/internal/kb"
	"freerag/internal/store"
)

// newTestKernel builds a kernel with one knowledge base and no parse sidecar,
// which is the state a machine without the Python environment is in.
//
// The registry is real, in a temporary directory, rather than a stub: the base
// it creates has the same shape as a user's, so these tests exercise the
// resolution and loading path instead of a simplified stand-in for it.
func newTestKernel(t *testing.T) *kernel {
	t.Helper()

	dir := t.TempDir()
	registry, err := kb.Open(filepath.Join(dir, "kbs.json"), filepath.Join(dir, "kbs"), "")
	if err != nil {
		t.Fatalf("kb.Open: %v", err)
	}

	k := &kernel{kbs: registry, open: map[string]*kbRuntime{}}
	// Opened up front so a test can reach the store without an RPC first.
	// loadBase is exactly what kbFor would have called.
	live, err := k.loadBase(registry.List()[0])
	if err != nil {
		t.Fatalf("loadBase: %v", err)
	}
	k.open[live.base.ID] = live
	return k
}

// storeOf is the index of the kernel's default knowledge base.
func storeOf(t *testing.T, k *kernel) *store.Store {
	t.Helper()
	live, err := k.kbFor("")
	if err != nil {
		t.Fatalf("kbFor(\"\"): %v", err)
	}
	return live.store
}

// call runs one method through a server built by kernel.register.
func call(t *testing.T, k *kernel, line string) ipc.Response {
	t.Helper()
	resp, _ := callWithEvents(t, k, line)
	return resp
}

// callWithEvents runs one method and also returns the progress notifications it
// emitted before answering.
//
// Responses and notifications share one stream, so a helper that unmarshals the
// whole output as a response breaks the moment a handler reports progress. Both
// are returned here so tests can assert either.
func callWithEvents(t *testing.T, k *kernel, line string) (ipc.Response, []map[string]any) {
	t.Helper()
	srv := ipc.NewServer()
	k.register(srv)

	var out bytes.Buffer
	if err := srv.Serve(context.Background(), strings.NewReader(line+"\n"), &out); err != nil {
		t.Fatalf("Serve: %v", err)
	}

	var resp ipc.Response
	found := false
	var events []map[string]any
	for _, raw := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		var probe struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params map[string]any  `json:"params"`
		}
		if err := json.Unmarshal([]byte(raw), &probe); err != nil {
			t.Fatalf("decode message %q: %v", raw, err)
		}
		if probe.ID == nil {
			// No id means the server sent it: a notification, not a response.
			probe.Params["method"] = probe.Method
			events = append(events, probe.Params)
			continue
		}
		if err := json.Unmarshal([]byte(raw), &resp); err != nil {
			t.Fatalf("decode response %q: %v", raw, err)
		}
		found = true
	}
	if !found {
		t.Fatalf("no response in %q", out.String())
	}
	return resp, events
}

// stages lists the progress stages in the order they arrived.
func stages(events []map[string]any) []string {
	out := make([]string, 0, len(events))
	for _, event := range events {
		stage, _ := event["stage"].(string)
		out = append(out, stage)
	}
	return out
}

func result(t *testing.T, resp ipc.Response) map[string]any {
	t.Helper()
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	obj, ok := resp.Result.(map[string]any)
	if !ok {
		t.Fatalf("result is %T", resp.Result)
	}
	return obj
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
}

// md5Of recomputes the fingerprint the kernel stores, independently of it.
func md5Of(t *testing.T, path string) string {
	t.Helper()
	handle, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer handle.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, handle); err != nil {
		t.Fatalf("hash: %v", err)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// indexedKernel builds a kernel that already holds one indexed document.
func indexedKernel(t *testing.T, path, content string) (*kernel, string) {
	t.Helper()
	writeFile(t, path, content)
	sum := md5Of(t, path)

	k := newTestKernel(t)
	docID := filepath.Base(path)
	storeOf(t, k).Add([]store.Chunk{{ChunkID: "c0", DocID: docID, Text: "alpha passage"}})
	storeOf(t, k).PutDocument(store.DocumentRecord{
		MD5:        sum,
		DocID:      docID,
		ChunkCount: 1,
		PageCount:  3,
		Pipeline:   fingerprintFor(parseParams{}),
	})
	return k, sum
}

func indexRequest(path string, force bool) string {
	params := map[string]any{"path": path}
	if force {
		params["force"] = true
	}
	raw, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": "index", "params": params,
	})
	return string(raw)
}

func TestIndexSkipsAnUnchangedDocument(t *testing.T) {
	path := filepath.Join(t.TempDir(), "paper.pdf")
	k, sum := indexedKernel(t, path, "the same bytes as before")

	// This kernel has no parse sidecar at all. The skip has to happen before
	// the sidecar is required, or a machine without Python could never report a
	// document as already indexed.
	payload := result(t, call(t, k, indexRequest(path, false)))

	if payload["skipped"] != true {
		t.Fatalf("skipped = %#v, want true", payload["skipped"])
	}
	if payload["md5"] != sum {
		t.Fatalf("md5 = %#v, want %s", payload["md5"], sum)
	}
	if payload["note"] != "already indexed" {
		t.Fatalf("note = %#v", payload["note"])
	}
	if payload["added"] != float64(0) {
		t.Fatalf("added = %#v, want 0", payload["added"])
	}
}

func TestIndexForceBypassesTheSkip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "paper.pdf")
	k, _ := indexedKernel(t, path, "the same bytes as before")

	// Forcing means a parse is required, and this kernel has no sidecar — so
	// reaching that error proves the skip was bypassed rather than hit.
	resp := call(t, k, indexRequest(path, true))
	if resp.Error == nil {
		t.Fatal("force must proceed to the parse, which needs the sidecar")
	}
	if !strings.Contains(resp.Error.Message, "sidecar") {
		t.Fatalf("error = %v, want the sidecar failure", resp.Error)
	}
}

func TestIndexDoesNotSkipWhenTheContentChanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "paper.pdf")
	k, _ := indexedKernel(t, path, "the first version")

	// Different content at the same path. A path-and-mod-time check would call
	// this unchanged.
	writeFile(t, path, "a completely different version")

	resp := call(t, k, indexRequest(path, false))
	// Without a sidecar it cannot finish, but it must have got past the skip.
	if resp.Error == nil {
		t.Fatal("changed content must not be skipped")
	}
	if !strings.Contains(resp.Error.Message, "sidecar") {
		t.Fatalf("error = %v, want the sidecar failure", resp.Error)
	}
}

func TestStaleReason(t *testing.T) {
	s := store.New()
	s.Add([]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"}})

	same := store.DocumentRecord{MD5: "x", DocID: "a.pdf", ChunkCount: 1, Pipeline: "p1"}
	if got := staleReason(s, same, "p1"); got != "" {
		t.Fatalf("staleReason = %q, want empty for a matching record", got)
	}
	if got := staleReason(s, same, "p2"); got == "" {
		t.Fatal("a pipeline change must invalidate the record")
	}

	// A record whose chunks are gone means the index was rebuilt without this
	// document — the case a manifest stored separately would fail to notice.
	missing := store.DocumentRecord{MD5: "x", DocID: "a.pdf", ChunkCount: 9, Pipeline: "p1"}
	if got := staleReason(s, missing, "p1"); got == "" {
		t.Fatal("a chunk-count mismatch must invalidate the record")
	}
}

func TestFileMD5FollowsContentNotModTime(t *testing.T) {
	path := filepath.Join(t.TempDir(), "paper.pdf")
	writeFile(t, path, "the first version")
	before := md5Of(t, path)

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}

	// Overwrite the content and restore the timestamp. This is what `cp -p`,
	// rsync and a restored backup do, and it is the case a path-plus-mod-time
	// check reads as "unchanged" — serving chunks from the previous version.
	writeFile(t, path, "a completely different version")
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatalf("Chtimes: %v", err)
	}

	if after := md5Of(t, path); before == after {
		t.Fatal("the fingerprint must follow the content, not the modification time")
	}
}

func TestFingerprintForCoversTheChunkingOptions(t *testing.T) {
	base := fingerprintFor(parseParams{})

	if base != fingerprintFor(parseParams{}) {
		t.Fatal("the fingerprint must be stable for identical options")
	}
	for name, params := range map[string]parseParams{
		"profile":   {Profile: "zh"},
		"max_chars": {MaxChars: 512},
		"max_pages": {MaxPages: 3},
	} {
		if fingerprintFor(params) == base {
			t.Fatalf("%s must change the fingerprint: the chunks it produces differ", name)
		}
	}
}

func TestIndexEmitsProgressForANewDocument(t *testing.T) {
	path := filepath.Join(t.TempDir(), "paper.pdf")
	writeFile(t, path, "never indexed before")

	resp, events := callWithEvents(t, newTestKernel(t), indexRequest(path, false))
	if resp.Error == nil {
		t.Fatal("expected the sidecar failure for an unindexed document")
	}

	// The hash happens before the sidecar is needed, so the UI must see it even
	// when the call is about to fail — that is the feedback that distinguishes
	// "working" from "hung".
	if got := stages(events); len(got) == 0 || got[0] != "hash" {
		t.Fatalf("stages = %v, want hash first", got)
	}
}

func TestIndexReportsASkipAsProgress(t *testing.T) {
	path := filepath.Join(t.TempDir(), "paper.pdf")
	k, _ := indexedKernel(t, path, "the same bytes as before")

	_, events := callWithEvents(t, k, indexRequest(path, false))

	got := stages(events)
	if len(got) != 2 || got[0] != "hash" || got[1] != "skipped" {
		t.Fatalf("stages = %v, want [hash skipped]", got)
	}
}

func TestDocumentsListsTheManifest(t *testing.T) {
	k := newTestKernel(t)
	storeOf(t, k).Add([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"},
		{ChunkID: "c1", DocID: "a.pdf", Text: "beta"},
	})
	storeOf(t, k).PutDocument(store.DocumentRecord{
		MD5: "aaa", DocID: "a.pdf", SourceFile: "a.pdf", Path: "/tmp/a.pdf",
		ChunkCount: 2, PageCount: 7, Pipeline: "parse-v1", IndexedAt: time.Now(),
	})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"documents"}`))

	documents, _ := payload["documents"].([]any)
	if len(documents) != 1 {
		t.Fatalf("documents = %#v", payload["documents"])
	}
	entry, _ := documents[0].(map[string]any)
	if entry["md5"] != "aaa" || entry["doc_id"] != "a.pdf" || entry["source_file"] != "a.pdf" {
		t.Fatalf("entry = %#v", entry)
	}
	if entry["chunk_count"] != float64(2) || entry["page_count"] != float64(7) {
		t.Fatalf("counts = %#v", entry)
	}
	// Reported live so the UI can spot a document whose chunks went missing
	// instead of trusting the manifest.
	if entry["chunks_present"] != float64(2) {
		t.Fatalf("chunks_present = %#v", entry["chunks_present"])
	}
	if payload["indexed"] != float64(2) {
		t.Fatalf("indexed = %#v", payload["indexed"])
	}
}

func TestForgetRemovesDocumentChunksAndManifest(t *testing.T) {
	k := newTestKernel(t)
	storeOf(t, k).Add([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"},
		{ChunkID: "c0", DocID: "b.pdf", Text: "beta"},
	})
	storeOf(t, k).PutDocument(store.DocumentRecord{MD5: "aaa", DocID: "a.pdf", ChunkCount: 1})
	storeOf(t, k).PutDocument(store.DocumentRecord{MD5: "bbb", DocID: "b.pdf", ChunkCount: 1})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"forget","params":{"md5":"aaa"}}`))

	if payload["doc_id"] != "a.pdf" || payload["removed"] != float64(1) {
		t.Fatalf("reply = %#v", payload)
	}
	if storeOf(t, k).Len() != 1 {
		t.Fatalf("len = %d, want 1", storeOf(t, k).Len())
	}
	if _, ok := storeOf(t, k).Document("aaa"); ok {
		t.Fatal("the forgotten document's manifest entry must be gone")
	}
	// The other document must be untouched.
	if hits := storeOf(t, k).Search("beta", 5); len(hits) != 1 {
		t.Fatalf("the surviving document lost its chunks: %#v", hits)
	}
}

func TestForgetRejectsAnUnknownOrMissingSelector(t *testing.T) {
	k := newTestKernel(t)

	if resp := call(t, k, `{"jsonrpc":"2.0","id":1,"method":"forget","params":{}}`); resp.Error == nil {
		t.Fatal("expected an error when neither md5 nor doc_id is given")
	}
	if resp := call(t, k, `{"jsonrpc":"2.0","id":2,"method":"forget","params":{"md5":"nope"}}`); resp.Error == nil {
		t.Fatal("expected an error for an unknown md5")
	}
}

func TestStatusReportsEverySubsystem(t *testing.T) {
	payload := result(t, call(t, newTestKernel(t), `{"jsonrpc":"2.0","id":1,"method":"status"}`))

	for _, key := range []string{"index", "sidecar", "embedding", "dense_index", "generator", "checker"} {
		if _, ok := payload[key]; !ok {
			t.Fatalf("status is missing %q: %#v", key, payload)
		}
	}

	// This kernel has no sidecar, embedder or generator: each must say so
	// rather than reporting itself healthy.
	sidecar, _ := payload["sidecar"].(map[string]any)
	if sidecar["configured"] != false {
		t.Fatalf("sidecar = %#v", sidecar)
	}
	embedding, _ := payload["embedding"].(map[string]any)
	if embedding["enabled"] != false {
		t.Fatalf("embedding = %#v", embedding)
	}
	generator, _ := payload["generator"].(map[string]any)
	if generator["configured"] != false {
		t.Fatalf("generator = %#v", generator)
	}
}

func TestStatusMarksWhatItCouldNotCheck(t *testing.T) {
	payload := result(t, call(t, newTestKernel(t), `{"jsonrpc":"2.0","id":1,"method":"status"}`))

	// Probing the hosted embedder costs a request and money on every poll, so
	// it is not probed — and saying so is the point of the flag.
	embedding, _ := payload["embedding"].(map[string]any)
	if embedding["checked"] != false {
		t.Fatalf("embedding.checked = %#v, want false", embedding["checked"])
	}
	// Ollama is a local HTTP GET, so that one really is checked.
	generator, _ := payload["generator"].(map[string]any)
	if generator["checked"] != false {
		t.Fatalf("generator.checked = %#v, want false with no generator configured", generator["checked"])
	}
}

func TestReasoningIsOffUnlessAskedFor(t *testing.T) {
	// The default is a measured one: with reasoning on, a one-sentence answer
	// cost 246 tokens and 29.7 s against 14 tokens and 1.9 s with it off — and
	// reasoning is generated against the answer's own num_predict budget, so it
	// can consume the whole of it and leave the answer empty.
	t.Setenv("FREERAG_THINK", "")
	if thinkingEnabled() {
		t.Fatal("reasoning must be off by default")
	}

	for _, value := range []string{"1", "true", "TRUE", "yes", "on", " 1 "} {
		t.Setenv("FREERAG_THINK", value)
		if !thinkingEnabled() {
			t.Fatalf("FREERAG_THINK=%q must enable reasoning", value)
		}
	}
	for _, value := range []string{"0", "false", "off", "maybe"} {
		t.Setenv("FREERAG_THINK", value)
		if thinkingEnabled() {
			t.Fatalf("FREERAG_THINK=%q must not enable reasoning", value)
		}
	}
}

func TestKeepAliveOutlastsOllamasDefault(t *testing.T) {
	// Ollama's own default is 5 minutes, which is shorter than the gap between
	// two questions for a person reading the first answer — so the weights, and
	// with them the prompt cache, would be gone exactly when they are next
	// needed.
	t.Setenv("FREERAG_KEEP_ALIVE", "")
	if got := generatorKeepAlive(); got != "30m" {
		t.Fatalf("keep-alive = %q, want 30m", got)
	}

	t.Setenv("FREERAG_KEEP_ALIVE", "-1")
	if got := generatorKeepAlive(); got != "-1" {
		t.Fatalf("keep-alive = %q, want the override to win", got)
	}
}

func TestChunkBoxReadsAPageSpaceBox(t *testing.T) {
	cases := []struct {
		name     string
		metadata map[string]any
		want     []float64
	}{
		{
			// What the index actually holds: after the JSON round trip the
			// numbers are float64 inside a []any.
			name:     "after the index round trip",
			metadata: map[string]any{"bbox": []any{float64(1), float64(2), float64(30), float64(40)}},
			want:     []float64{1, 2, 30, 40},
		},
		{
			name:     "in-memory floats",
			metadata: map[string]any{"bbox": []float64{1, 2, 30, 40}},
			want:     []float64{1, 2, 30, 40},
		},
		{name: "missing", metadata: map[string]any{}, want: nil},
		{name: "nil metadata", metadata: nil, want: nil},
		{name: "wrong length", metadata: map[string]any{"bbox": []any{float64(1), float64(2)}}, want: nil},
		{name: "not numbers", metadata: map[string]any{"bbox": []any{"a", "b", "c", "d"}}, want: nil},
		// A zero-area or inverted box is not a region. Drawing one would produce
		// a dot that reads as a rendering bug rather than as a missing box.
		{
			name:     "zero area",
			metadata: map[string]any{"bbox": []any{float64(5), float64(5), float64(5), float64(5)}},
			want:     nil,
		},
		{
			name:     "inverted",
			metadata: map[string]any{"bbox": []any{float64(9), float64(9), float64(1), float64(1)}},
			want:     nil,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got := chunkBox(store.Chunk{Metadata: testCase.metadata})
			if len(got) != len(testCase.want) {
				t.Fatalf("box = %v, want %v", got, testCase.want)
			}
			for i := range got {
				if got[i] != testCase.want[i] {
					t.Fatalf("box = %v, want %v", got, testCase.want)
				}
			}
		})
	}
}

func TestChunksRPCReportsBoxesAndPerPageCounts(t *testing.T) {
	k := newTestKernel(t)
	storeOf(t, k).Add([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Title", Text: "hello",
			Metadata: map[string]any{"bbox": []any{float64(10), float64(20), float64(100), float64(40)}}},
		{ChunkID: "c1", DocID: "a.pdf", PageNum: 3, BlockType: "Text", Text: "world",
			Metadata: map[string]any{"bbox": []any{float64(10), float64(60), float64(100), float64(90)}}},
		// Another document, and a chunk with no box at all.
		{ChunkID: "c2", DocID: "b.pdf", PageNum: 1, BlockType: "Text", Text: "other"},
	})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"chunks","params":{"doc_id":"a.pdf"}}`))

	if payload["total"] != float64(2) {
		t.Fatalf("total = %#v, want only a.pdf's chunks", payload["total"])
	}

	// The per-page counts describe the whole document, which is what lets the
	// UI's arrows know which pages exist before one is chosen.
	pages, ok := payload["pages"].([]any)
	if !ok || len(pages) != 2 {
		t.Fatalf("pages = %#v, want two entries", payload["pages"])
	}
	first := pages[0].(map[string]any)
	if first["page"] != float64(1) || first["chunks"] != float64(1) {
		t.Fatalf("pages[0] = %#v, want page 1 with one chunk", first)
	}
	if second := pages[1].(map[string]any); second["page"] != float64(3) {
		t.Fatalf("pages[1] = %#v, want page 3 — the order must be ascending", second)
	}

	chunks, ok := payload["chunks"].([]any)
	if !ok || len(chunks) != 2 {
		t.Fatalf("chunks = %#v", payload["chunks"])
	}
	boxed := chunks[0].(map[string]any)
	if box, ok := boxed["bbox"].([]any); !ok || len(box) != 4 {
		t.Fatalf("bbox = %#v, want four numbers", boxed["bbox"])
	}
	if boxed["block_type"] != "Title" || boxed["page_num"] != float64(1) {
		t.Fatalf("chunk = %#v", boxed)
	}
}

func TestChunksRPCFiltersByPage(t *testing.T) {
	k := newTestKernel(t)
	storeOf(t, k).Add([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, Text: "one"},
		{ChunkID: "c1", DocID: "a.pdf", PageNum: 2, Text: "two"},
		{ChunkID: "c2", DocID: "a.pdf", PageNum: 2, Text: "three"},
	})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"chunks","params":{"doc_id":"a.pdf","page":2}}`))

	chunks := payload["chunks"].([]any)
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want the two on page 2", len(chunks))
	}
	// The page filter narrows the list, never the counts: the arrows still need
	// to know about every page.
	if payload["total"] != float64(3) {
		t.Fatalf("total = %#v, want the whole document", payload["total"])
	}
}

func TestChunksRPCNeedsADocument(t *testing.T) {
	_, err := newTestKernel(t).handleChunks(context.Background(), []byte(`{}`))
	if err == nil {
		t.Fatal("a request without doc_id must be refused")
	}
}

func TestPageRPCNeedsAPage(t *testing.T) {
	k := newTestKernel(t)
	if _, err := k.handlePage(context.Background(), []byte(`{"doc_id":"a.pdf"}`)); err == nil {
		t.Fatal("a request without a page must be refused")
	}
	if _, err := k.handlePage(context.Background(), []byte(`{"doc_id":"a.pdf","page":0}`)); err == nil {
		t.Fatal("page 0 must be refused — pages are 1-based")
	}
}

func TestPageRPCReportsAnUnknownDocument(t *testing.T) {
	k := newTestKernel(t)
	_, err := k.handlePage(context.Background(), []byte(`{"doc_id":"none.pdf","page":1}`))
	if err == nil {
		t.Fatal("an unknown document must be refused rather than rendered")
	}
}

func TestPingAndVersion(t *testing.T) {
	k := newTestKernel(t)

	if resp := call(t, k, `{"jsonrpc":"2.0","id":1,"method":"ping"}`); resp.Error != nil {
		t.Fatalf("ping failed: %v", resp.Error)
	}

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"version"}`))
	if payload["name"] != "freerag" || payload["version"] != version {
		t.Fatalf("unexpected version payload: %#v", payload)
	}
	// No index figures here. They are per knowledge base and moved to `status`,
	// which can name one; reporting the default base from `version` would mean
	// opening an index to answer what build this is.
	if _, present := payload["indexed"]; present {
		t.Fatalf("version must not report index figures: %#v", payload)
	}
	if payload["knowledge_bases"] != float64(1) {
		t.Fatalf("knowledge_bases = %#v, want 1", payload["knowledge_bases"])
	}
	// Without a sidecar the field is present and null, so the shell can tell
	// "not configured" from "configured but idle".
	if sidecar, present := payload["parse_sidecar"]; !present || sidecar != nil {
		t.Fatalf("parse_sidecar = %#v (present=%v)", sidecar, present)
	}
}

func TestStatusReportsTheDefaultKnowledgeBase(t *testing.T) {
	k := newTestKernel(t)

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"status"}`))

	// The id was omitted, so the first base answers — and the reply says which
	// one, because a caller that forgot the id has no other way to tell.
	base, ok := payload["kb"].(map[string]any)
	if !ok || base["id"] == "" || base["name"] == "" {
		t.Fatalf("status did not name a knowledge base: %#v", payload["kb"])
	}
	index, _ := payload["index"].(map[string]any)
	if index["chunks"] != float64(0) {
		t.Fatalf("index = %#v, want an empty base", index)
	}
}

// ---- knowledge bases ----

func TestKnowledgeBaseLifecycle(t *testing.T) {
	k := newTestKernel(t)

	created := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"kb.create","params":{"name":"论文库"}}`))
	id, _ := created["id"].(string)
	if id == "" || created["name"] != "论文库" {
		t.Fatalf("kb.create = %#v", created)
	}

	listed := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"kb.list"}`))
	if listed["count"] != float64(2) {
		t.Fatalf("count = %#v, want 2", listed["count"])
	}

	renamed := result(t, call(t, k, `{"jsonrpc":"2.0","id":3,"method":"kb.rename","params":{"id":"`+id+`","name":"论文集"}}`))
	if renamed["name"] != "论文集" || renamed["id"] != id {
		t.Fatalf("kb.rename = %#v", renamed)
	}

	// A rename must not move anything: the id is what the index directory and
	// the Qdrant collection are named after.
	if renamed["id"] != created["id"] {
		t.Fatal("renaming must not change the id")
	}

	if resp := call(t, k, `{"jsonrpc":"2.0","id":4,"method":"kb.delete","params":{"id":"`+id+`"}}`); resp.Error != nil {
		t.Fatalf("kb.delete: %v", resp.Error)
	}
	after := result(t, call(t, k, `{"jsonrpc":"2.0","id":5,"method":"kb.list"}`))
	if after["count"] != float64(1) {
		t.Fatalf("count = %#v, want 1 after deleting the second base", after["count"])
	}
}

func TestKnowledgeBasesAreIsolated(t *testing.T) {
	k := newTestKernel(t)

	created := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"kb.create","params":{"name":"另一个库"}}`))
	other, _ := created["id"].(string)

	// One document in the default base, none in the other. An omitted id must
	// not reach across — that is the whole point of the boundary.
	storeOf(t, k).Add([]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "alpha passage"}})

	def := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"documents"}`))
	if def["indexed"] != float64(1) {
		t.Fatalf("default indexed = %#v, want 1", def["indexed"])
	}

	second := result(t, call(t, k, `{"jsonrpc":"2.0","id":3,"method":"documents","params":{"kb":"`+other+`"}}`))
	if second["indexed"] != float64(0) {
		t.Fatalf("the second base sees the first base's documents: %#v", second["indexed"])
	}

	if _, err := k.kbFor("nope"); err == nil {
		t.Fatal("an unknown id must be an error rather than a fallback")
	}
}

func TestLastKnowledgeBaseCannotBeDeleted(t *testing.T) {
	// Every call that omits an id resolves to the first base, and the UI has
	// nothing to bind a session to without one — so the last base is
	// load-bearing rather than just another row.
	k := newTestKernel(t)
	only, _ := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"kb.list"}`))["bases"].([]any)[0].(map[string]any)

	resp := call(t, k, `{"jsonrpc":"2.0","id":2,"method":"kb.delete","params":{"id":"`+only["id"].(string)+`"}}`)
	if resp.Error == nil || resp.Error.Code != ipc.CodeInvalidParams {
		t.Fatalf("want invalid-params, got %#v", resp.Error)
	}
}

func TestDuplicateKnowledgeBaseNamesAreRejected(t *testing.T) {
	// Two bases with one name is not a naming quirk, it is a list the user
	// cannot act on, because every action is chosen by name.
	k := newTestKernel(t)
	if resp := call(t, k, `{"jsonrpc":"2.0","id":1,"method":"kb.create","params":{"name":"默认知识库"}}`); resp.Error == nil {
		t.Fatal("a duplicate name must be rejected")
	}
	if resp := call(t, k, `{"jsonrpc":"2.0","id":2,"method":"kb.create","params":{"name":"  "}}`); resp.Error == nil {
		t.Fatal("a blank name must be rejected")
	}
}

func TestDeletingAKnowledgeBaseDropsItsVectors(t *testing.T) {
	// The base is never opened here, and that is the case that was broken: the
	// cleanup used to run only through a loaded runtime, so a base deleted
	// without having been opened first kept its collection — and deleting
	// something you never opened is the ordinary case, not the exotic one.
	var (
		mu      sync.Mutex
		deleted []string
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			mu.Lock()
			deleted = append(deleted, r.URL.Path)
			mu.Unlock()
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"result":true}`))
	}))
	defer server.Close()
	t.Setenv("FREERAG_QDRANT_URL", server.URL)

	k := newTestKernel(t)
	created := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"kb.create","params":{"name":"待删"}}`))
	id, _ := created["id"].(string)

	if resp := call(t, k, `{"jsonrpc":"2.0","id":2,"method":"kb.delete","params":{"id":"`+id+`"}}`); resp.Error != nil {
		t.Fatalf("kb.delete: %v", resp.Error)
	}

	mu.Lock()
	defer mu.Unlock()
	want := "/collections/freerag_" + id
	for _, path := range deleted {
		if path == want {
			return
		}
	}
	t.Fatalf("no DELETE for %s was issued; saw %#v", want, deleted)
}

func TestParseRejectsMissingPath(t *testing.T) {
	resp := call(t, newTestKernel(t), `{"jsonrpc":"2.0","id":1,"method":"parse","params":{}}`)
	if resp.Error == nil || resp.Error.Code != ipc.CodeInvalidParams {
		t.Fatalf("want invalid-params error, got %#v", resp.Error)
	}
}

func TestParseWithoutSidecarReportsUnavailable(t *testing.T) {
	resp := call(t, newTestKernel(t), `{"jsonrpc":"2.0","id":1,"method":"parse","params":{"path":"/tmp/x.pdf"}}`)
	if resp.Error == nil {
		t.Fatal("expected an error when no sidecar is configured")
	}
	if resp.Error.Code != ipc.CodeInternalError {
		t.Fatalf("code = %d, want %d", resp.Error.Code, ipc.CodeInternalError)
	}
	if !strings.Contains(resp.Error.Message, "sidecar unavailable") {
		t.Fatalf("message = %q", resp.Error.Message)
	}
}

func TestIndexWithoutSidecarReportsUnavailable(t *testing.T) {
	// A real file, and one that is not already indexed: the fingerprint is
	// computed first, so a missing path is now reported as a missing path
	// rather than as a missing sidecar.
	path := filepath.Join(t.TempDir(), "paper.pdf")
	writeFile(t, path, "never indexed before")

	resp := call(t, newTestKernel(t), indexRequest(path, false))
	if resp.Error == nil || resp.Error.Code != ipc.CodeInternalError {
		t.Fatalf("want internal error, got %#v", resp.Error)
	}
	if !strings.Contains(resp.Error.Message, "sidecar unavailable") {
		t.Fatalf("message = %q", resp.Error.Message)
	}
}

func TestSearchValidatesAndReturnsHits(t *testing.T) {
	k := newTestKernel(t)

	if resp := call(t, k, `{"jsonrpc":"2.0","id":1,"method":"search","params":{}}`); resp.Error == nil {
		t.Fatal("expected an error for a missing query")
	}

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"search","params":{"query":"anything"}}`))
	if payload["count"] != float64(0) {
		t.Fatalf("empty index returned %#v hits", payload["count"])
	}

	storeOf(t, k).Add([]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "sufficiency checking over evidence"}})
	payload = result(t, call(t, k, `{"jsonrpc":"2.0","id":3,"method":"search","params":{"query":"sufficiency","limit":5}}`))
	if payload["count"] != float64(1) {
		t.Fatalf("count = %#v, want 1", payload["count"])
	}
}

func TestToolsListsTheToolSurface(t *testing.T) {
	payload := result(t, call(t, newTestKernel(t), `{"jsonrpc":"2.0","id":1,"method":"tools"}`))

	if payload["mode"] != "medium" {
		t.Fatalf("mode = %#v", payload["mode"])
	}
	// The four of §6.7 plus dirtree_search, which is a channel rather than a
	// structure: whether it applies is a property of the knowledge base, asked
	// at call time, so it is advertised like any other tool.
	want := []string{"hybrid_search", "grep_search", "dirtree_search", "list_chunks", "metadata_search"}
	names, ok := payload["names"].([]any)
	if !ok || len(names) != len(want) {
		t.Fatalf("names = %#v, want %v", payload["names"], want)
	}
	for index, name := range names {
		if name != want[index] {
			t.Fatalf("names[%d] = %v, want %s", index, name, want[index])
		}
	}
	specs, ok := payload["tools"].([]any)
	if !ok || len(specs) != len(want) {
		t.Fatalf("tools = %#v", payload["tools"])
	}
}

func TestToolExecutesOneCall(t *testing.T) {
	k := newTestKernel(t)
	storeOf(t, k).Add([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Text", Text: "the standard is GB/T 1234"},
		{ChunkID: "c1", DocID: "a.pdf", PageNum: 2, BlockType: "Table", Text: "| --- |"},
	})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"tool","params":{"name":"hybrid_search","arguments":{"query":"standard"}}}`))
	if payload["tool"] != "hybrid_search" {
		t.Fatalf("tool = %#v", payload["tool"])
	}
	if hits, _ := payload["hits"].([]any); len(hits) == 0 {
		t.Fatalf("hits = %#v", payload["hits"])
	}

	grep := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"tool","params":{"name":"grep_search","arguments":{"pattern":"GB/T 1234"}}}`))
	if hits, _ := grep["hits"].([]any); len(hits) != 1 {
		t.Fatalf("grep hits = %#v", grep["hits"])
	}

	if resp := call(t, k, `{"jsonrpc":"2.0","id":3,"method":"tool","params":{"name":"nope"}}`); resp.Error == nil {
		t.Fatal("expected an error for a tool outside the surface")
	}
	if resp := call(t, k, `{"jsonrpc":"2.0","id":4,"method":"tool","params":{}}`); resp.Error == nil {
		t.Fatal("expected an error for a missing tool name")
	}
}

func TestAskRunsTheLoop(t *testing.T) {
	k := newTestKernel(t)

	if resp := call(t, k, `{"jsonrpc":"2.0","id":1,"method":"ask","params":{}}`); resp.Error == nil {
		t.Fatal("expected an error for a missing question")
	}

	storeOf(t, k).Add([]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "sufficiency checking decides the verdict"}})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"ask","params":{"question":"sufficiency checking"}}`))
	if payload["mode"] != "medium" {
		t.Fatalf("mode = %#v, want medium", payload["mode"])
	}
	if payload["verdict"] != string(agent.VerdictSufficient) {
		t.Fatalf("verdict = %#v", payload["verdict"])
	}
	if payload["rounds"] != float64(1) {
		t.Fatalf("rounds = %#v, want 1", payload["rounds"])
	}
}

// The ask reply IS agent.Result, marshalled through its json tags, so the field
// names are a contract with the renderer — and one the tests here did not cover.
// That gap shipped a real bug: the answer field was renamed from `draft` to
// `answer` in Go, and desktop/renderer/app.js kept reading `result.draft`, so
// every reply replaced the streamed text with its "no draft" placeholder. A
// rename in Go must fail HERE rather than in front of a user.
func TestAskReplyCarriesTheAnswerFieldUnderTheNameTheRendererReads(t *testing.T) {
	k := newTestKernel(t)
	storeOf(t, k).Add([]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "sufficiency checking decides the verdict"}})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"ask","params":{"question":"sufficiency checking"}}`))

	answer, present := payload["answer"]
	if !present {
		// Name the old name too, because that is the mistake that was made.
		t.Fatalf("the reply has no \"answer\" field (keys: %v); the renderer reads result.answer, "+
			"and reading undefined is what showed a placeholder instead of the answer", keysOf(payload))
	}
	text, ok := answer.(string)
	if !ok {
		t.Fatalf("answer = %#v, want a string", answer)
	}
	if strings.TrimSpace(text) == "" {
		t.Fatal("the answer is empty, so the renderer would show its placeholder")
	}
	if _, stale := payload["draft"]; stale {
		t.Fatal("the reply still carries a \"draft\" field, but nothing writes one any more")
	}
}

func keysOf(payload map[string]any) []string {
	keys := make([]string, 0, len(payload))
	for key := range payload {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
