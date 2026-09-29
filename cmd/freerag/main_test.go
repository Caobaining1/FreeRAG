package main

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"freerag/internal/agent"
	"freerag/internal/ipc"
	"freerag/internal/store"
)

// newTestKernel builds a kernel with a local index and no parse sidecar, which
// is the state a machine without the Python environment is in.
func newTestKernel(t *testing.T) *kernel {
	t.Helper()
	s := store.New()
	return &kernel{
		store:    s,
		dataPath: filepath.Join(t.TempDir(), "index.json"),
		toolbox:  &agent.Toolbox{Store: s, DefaultLimit: agent.Medium().SnippetsPerQuery},
		loop:     &agent.Loop{Store: s, Spec: agent.Medium()},
	}
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
	k.store.Add([]store.Chunk{{ChunkID: "c0", DocID: docID, Text: "alpha passage"}})
	k.store.PutDocument(store.DocumentRecord{
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
	k.store.Add([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"},
		{ChunkID: "c1", DocID: "a.pdf", Text: "beta"},
	})
	k.store.PutDocument(store.DocumentRecord{
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
	k.store.Add([]store.Chunk{
		{ChunkID: "c0", DocID: "a.pdf", Text: "alpha"},
		{ChunkID: "c0", DocID: "b.pdf", Text: "beta"},
	})
	k.store.PutDocument(store.DocumentRecord{MD5: "aaa", DocID: "a.pdf", ChunkCount: 1})
	k.store.PutDocument(store.DocumentRecord{MD5: "bbb", DocID: "b.pdf", ChunkCount: 1})

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":1,"method":"forget","params":{"md5":"aaa"}}`))

	if payload["doc_id"] != "a.pdf" || payload["removed"] != float64(1) {
		t.Fatalf("reply = %#v", payload)
	}
	if k.store.Len() != 1 {
		t.Fatalf("len = %d, want 1", k.store.Len())
	}
	if _, ok := k.store.Document("aaa"); ok {
		t.Fatal("the forgotten document's manifest entry must be gone")
	}
	// The other document must be untouched.
	if hits := k.store.Search("beta", 5); len(hits) != 1 {
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

func TestPingAndVersion(t *testing.T) {
	k := newTestKernel(t)

	if resp := call(t, k, `{"jsonrpc":"2.0","id":1,"method":"ping"}`); resp.Error != nil {
		t.Fatalf("ping failed: %v", resp.Error)
	}

	payload := result(t, call(t, k, `{"jsonrpc":"2.0","id":2,"method":"version"}`))
	if payload["name"] != "freerag" || payload["version"] != version {
		t.Fatalf("unexpected version payload: %#v", payload)
	}
	if payload["indexed"] != float64(0) {
		t.Fatalf("indexed = %#v, want 0", payload["indexed"])
	}
	// Without a sidecar the field is present and null, so the shell can tell
	// "not configured" from "configured but idle".
	if sidecar, present := payload["parse_sidecar"]; !present || sidecar != nil {
		t.Fatalf("parse_sidecar = %#v (present=%v)", sidecar, present)
	}
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

	k.store.Add([]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "sufficiency checking over evidence"}})
	payload = result(t, call(t, k, `{"jsonrpc":"2.0","id":3,"method":"search","params":{"query":"sufficiency","limit":5}}`))
	if payload["count"] != float64(1) {
		t.Fatalf("count = %#v, want 1", payload["count"])
	}
}

func TestToolsListsTheFourToolSurface(t *testing.T) {
	payload := result(t, call(t, newTestKernel(t), `{"jsonrpc":"2.0","id":1,"method":"tools"}`))

	if payload["mode"] != "medium" {
		t.Fatalf("mode = %#v", payload["mode"])
	}
	names, ok := payload["names"].([]any)
	if !ok || len(names) != 4 {
		t.Fatalf("names = %#v, want the four tools", payload["names"])
	}
	want := []string{"hybrid_search", "grep_search", "list_chunks", "metadata_search"}
	for index, name := range names {
		if name != want[index] {
			t.Fatalf("names[%d] = %v, want %s", index, name, want[index])
		}
	}
	specs, ok := payload["tools"].([]any)
	if !ok || len(specs) != 4 {
		t.Fatalf("tools = %#v", payload["tools"])
	}
}

func TestToolExecutesOneCall(t *testing.T) {
	k := newTestKernel(t)
	k.store.Add([]store.Chunk{
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

	k.store.Add([]store.Chunk{{ChunkID: "c0", DocID: "a.pdf", Text: "sufficiency checking decides the verdict"}})

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
