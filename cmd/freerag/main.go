// Command freerag is the Go kernel of the desktop Agentic RAG system.
//
// It speaks JSON-RPC 2.0 over stdio (one message per line, NDJSON) with the
// Electron shell, forwards document parsing to the Python parse sidecar and
// runs the medium-mode agentic loop over a local chunk index.
//
// stdout carries the protocol stream ONLY; every log line goes to stderr.
package main

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"freerag/internal/agent"
	"freerag/internal/dense"
	"freerag/internal/embed"
	"freerag/internal/ipc"
	"freerag/internal/parser"
	"freerag/internal/sidecar"
	"freerag/internal/store"
)

// version is the kernel version reported by the "version" method.
const version = "0.3.0"

func main() {
	// stderr only: stdout is the JSON-RPC stream.
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ltime | log.Lmicroseconds)

	if err := run(); err != nil {
		log.Fatalf("freerag: %v", err)
	}
}

// kernel holds the process-wide state the RPC methods operate on.
type kernel struct {
	parse    *parser.Service // nil when the sidecar cannot be located
	store    *store.Store
	dataPath string
	loop     *agent.Loop
	// toolbox executes the retrieval tool calls (docs/plan.md §6.7).
	toolbox *agent.Toolbox
	// generator is the local LLM client; nil when Ollama is unavailable and the
	// loop must fall back to extractive drafts.
	generator *agent.OllamaModel
	// embedder powers dense retrieval; nil leaves hybrid_search keyword-only.
	embedder embed.Embedder
	// denseIndex is the attached ANN index (docs/plan.md §4); nil means dense
	// ranking runs in-process. Kept here as well as on the store so `status`
	// can report which backend is live without reaching into the store.
	denseIndex store.DenseIndex
	// notify emits a JSON-RPC notification to the UI. Set by register; nil when
	// the server has no transport (unit tests), where dropping progress is fine.
	notify func(method string, params any) error
}

// progress reports one stage of a long-running call.
//
// `index` takes tens of seconds and `ask` can take minutes, so the UI needs
// something that changes while it waits. A notification that cannot be
// delivered is logged, never returned: the work is unaffected either way, and
// failing a parse because a progress line did not go out would be absurd.
func (k *kernel) progress(stage string, fields map[string]any) {
	if k.notify == nil {
		return
	}
	params := map[string]any{"stage": stage}
	for key, value := range fields {
		params[key] = value
	}
	if err := k.notify("progress", params); err != nil {
		log.Printf("warning: could not emit %s progress: %v", stage, err)
	}
}

func run() error {
	app, err := newKernel()
	if err != nil {
		return err
	}
	if app.parse != nil {
		defer func() {
			if err := app.parse.Close(); err != nil {
				log.Printf("parse sidecar shutdown: %v", err)
			}
		}()
	}

	srv := ipc.NewServer()
	app.register(srv)

	// Started before Serve so it overlaps with the shell's own startup, and
	// deliberately not awaited: it exists to keep the first question fast, so
	// blocking readiness on it would trade one wait for another.
	app.warmGenerator()

	log.Printf("freerag kernel %s ready (JSON-RPC 2.0 over stdio); index=%s (%d chunk(s))",
		version, app.dataPath, app.store.Len())
	return srv.Serve(context.Background(), os.Stdin, os.Stdout)
}

// newKernel assembles the kernel, degrading rather than failing when optional
// pieces are missing: a desktop install without the sidecar still answers.
func newKernel() (*kernel, error) {
	app := &kernel{}

	if cfg, err := parser.Discover(); err != nil {
		log.Printf("warning: parse sidecar unavailable: %v", err)
	} else {
		app.parse = parser.NewService(cfg)
		log.Printf("parse sidecar: %s %s", cfg.Python, cfg.Script)
	}

	app.dataPath = dataFilePath()
	loaded, err := store.Load(app.dataPath)
	if err != nil {
		log.Printf("warning: could not load the index at %s (%v); starting empty", app.dataPath, err)
		loaded = store.New()
	}
	app.store = loaded
	app.embedder = newEmbedder(app.store)
	app.newDenseIndex()
	app.toolbox = &agent.Toolbox{
		Store:        app.store,
		Embedder:     app.embedder,
		DefaultLimit: agent.Medium().SnippetsPerQuery,
	}
	app.loop = &agent.Loop{
		Store:   app.store,
		Tools:   app.toolbox,
		Checker: newChecker(app.parse),
		Spec:    agent.Medium(),
		Logger:  log.Default(),
		// Keep the prompt budget tied to the window we ask the generator for,
		// so the two cannot drift apart.
		MaxPromptChars: agent.PromptCharBudget(contextTokens()),
	}

	app.generator = newGenerator()
	if app.generator != nil {
		app.loop.Model = app.generator
	}

	return app, nil
}

// checkerName reports which sufficiency checker is active, so a caller can tell
// a real model review from the heuristic without reading the logs.
func checkerName(checker agent.Checker) string {
	switch checker.(type) {
	case agent.LayaChecker:
		return "laya"
	case agent.CoverageChecker:
		return "coverage-heuristic"
	case nil:
		return "none"
	default:
		return "custom"
	}
}

// newChecker returns the sufficiency checker: Laya when the sidecar can run it,
// else the deterministic coverage heuristic.
//
// The fallback is logged rather than silent. Collapsing "Laya is unavailable"
// into "the evidence is insufficient" would mislabel every run on a machine
// without the model, and the heuristic still reports UNKNOWN when it has nothing
// to judge.
func newChecker(svc *parser.Service) agent.Checker {
	if svc == nil {
		log.Printf("sufficiency checker: coverage heuristic (no parse sidecar)")
		return agent.CoverageChecker{}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if !svc.HasLaya(ctx) {
		log.Printf("sufficiency checker: coverage heuristic (Laya unavailable; run scripts/download-models.sh laya)")
		return agent.CoverageChecker{}
	}

	log.Printf("sufficiency checker: Laya (typed decisions)")
	return agent.LayaChecker{
		Decide: func(ctx context.Context, instructions string, criteria map[string]string, state string) (string, float64, error) {
			decision, err := svc.Decide(ctx, parser.DecisionRequest{
				Instructions: instructions,
				Criteria:     criteria,
				State:        state,
			}, 0)
			if err != nil {
				return "", 0, err
			}
			return decision.Choice, decision.Probability, nil
		},
	}
}

// newEmbedder wires dense retrieval.
//
// Every failure here is non-fatal and degrades to keyword-only search: an
// unreachable embedder must not stop the kernel from answering, but it must be
// loud, because a silent fallback would leave the operator believing hybrid
// retrieval is on when it is not.
func newEmbedder(s *store.Store) embed.Embedder {
	configured, err := embed.FromEnv()
	if err != nil {
		log.Printf("warning: dense retrieval disabled: %v", err)
		return nil
	}
	if configured == nil {
		log.Printf("dense retrieval disabled (no FREERAG_EMBED_PROVIDER); hybrid_search is keyword-only")
		return nil
	}

	// An index built by a different model cannot be queried with this one's
	// vectors, so refuse the mismatch instead of returning ranked nonsense.
	if err := s.SetEmbedder(configured.Name(), configured.Dimensions()); err != nil {
		log.Printf("warning: dense retrieval disabled: %v", err)
		return nil
	}

	log.Printf("embedder: %s (%d dims); %d/%d chunk(s) embedded",
		configured.Name(), configured.Dimensions(), s.VectorCount(), s.Len())
	return configured
}

// newDenseIndex attaches an external dense index when one is configured.
//
// Every failure here is non-fatal and leaves dense ranking in-process: the
// exact scan is slower on a large corpus but never wrong, so a vector database
// that is unreachable or misconfigured must not stop the kernel from answering.
// It is logged loudly for the same reason it is tolerated — a silent fallback
// would leave the operator believing the ANN index is in use when it is not.
func (k *kernel) newDenseIndex() {
	s := k.store
	s.SetLogger(log.Default())

	dimensions := s.Dimensions()
	if dimensions == 0 {
		// Without a known width a collection cannot be created, and there are
		// no vectors to put in one.
		return
	}

	index, err := dense.FromEnv(dimensions)
	if err != nil {
		log.Printf("warning: dense index disabled: %v", err)
		return
	}
	if index == nil {
		log.Printf("dense index: none configured; %d vector(s) are ranked by the in-process scan",
			s.VectorCount())
		return
	}

	s.SetDenseIndex(index)
	k.denseIndex = index
	if err := syncDenseIndex(s, index); err != nil {
		log.Printf("warning: dense index is out of sync: %v", err)
	}
	log.Printf("dense index: %s collection %q (%d dims, %d vector(s) in the index, %d in the store)",
		index.Backend(), index.Collection, index.Dims, index.Len(), s.VectorCount())
}

// syncDenseIndex re-seeds the collection when it holds fewer vectors than the
// store.
//
// The vectors survive in the index file, so re-seeding avoids re-embedding —
// which against a hosted provider means paying for the whole corpus again. The
// reverse case (the index holding more than the store) is left alone: it means
// a stale collection, and dropping points here could delete vectors this
// process has not loaded yet.
func syncDenseIndex(s *store.Store, index store.DenseIndex) error {
	chunks, vectors := s.EmbeddedChunks()
	if len(chunks) == 0 || index.Len() >= len(chunks) {
		return nil
	}
	return index.Upsert(chunks, vectors)
}

// newGenerator wires the local generating LLM.
//
// A missing or unreachable Ollama is NOT an error: the loop then runs its
// extractive fallback, which is exactly what a machine without the models
// installed gets. Reporting that as a failure would make a valid install look
// broken.
// thinkingEnabled reports whether Qwen3's reasoning mode should be requested.
//
// Off by default, and that default is a measured decision rather than a
// preference. On the reference machine (Apple M5, 4B Q4_K_M, fully resident on
// the GPU) answering a one-sentence question took 246 generated tokens and
// 29.7 s with reasoning on — of which `message.thinking` was 384 characters
// that nothing in this pipeline reads — against 14 tokens and 1.9 s with it
// off.
//
// It is also a correctness switch. Reasoning is generated against the same
// num_predict budget as the answer, so a budget that is generous for a short
// cited answer can be consumed entirely by reasoning: at num_predict=96 the
// model returned empty content and 160 characters of thinking. Reasoning is
// worth having when the answer needs it and the budget accounts for it; here
// the loop asks for two to four cited sentences, which does not.
func thinkingEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FREERAG_THINK"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// generatorKeepAlive is how long Ollama holds the weights after a call.
//
// Long by default because unloading also discards the prompt cache: the tool
// definitions and system prompt are the bulk of the first request of every
// round, and re-processing that prefix costs more than reloading the weights
// (load measured 0.5 s; prompt processing 465 tok/s over a prefix of a few
// thousand tokens).
func generatorKeepAlive() string {
	if value := strings.TrimSpace(os.Getenv("FREERAG_KEEP_ALIVE")); value != "" {
		return value
	}
	return "30m"
}

func newGenerator() *agent.OllamaModel {
	modelName := os.Getenv("FREERAG_MODEL")
	if modelName == "" {
		modelName = agent.DefaultGeneratorModel
	}
	think := thinkingEnabled()
	gen := &agent.OllamaModel{
		BaseURL:     os.Getenv("FREERAG_OLLAMA_URL"),
		Model:       modelName,
		NumCtx:      contextTokens(),
		NumPredict:  512,
		Temperature: 0.2,
		Think:       &think,
		KeepAlive:   generatorKeepAlive(),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if !gen.Reachable(ctx) {
		log.Printf("warning: generator %q unreachable; the loop will use extractive drafts (run scripts/setup-ollama.sh)", modelName)
		return nil
	}
	log.Printf("generator: %s", modelName)
	return gen
}

// warmGenerator loads the model and primes its prompt cache in the background.
//
// Worth being precise about what this buys, because it is not what it looks
// like: loading the weights is 0.5 s on the reference machine, so preloading
// does not fix a slow first answer — generation does, and that is what
// thinkingEnabled addresses. It is still done because it costs nothing and it
// removes both latencies from the first question, which is the one a new user
// judges the app by.
//
// Errors are logged and dropped. Ollama being absent already degrades the loop
// to extractive drafts, and a warm-up failure is the same situation one step
// less convenient — never a reason to fail startup.
func (k *kernel) warmGenerator() {
	if k.loop == nil || k.loop.Model == nil {
		return
	}
	go func() {
		started := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := k.loop.Warm(ctx); err != nil {
			log.Printf("warning: model warm-up failed (%v); the first question will pay the load", err)
			return
		}
		log.Printf("model warm-up done in %s", time.Since(started).Round(time.Millisecond))
	}()
}

// contextTokens is the context window requested from the generator.
//
// Ollama's own default is 4096, which is smaller than the evidence block the
// loop assembles, so it must be raised explicitly (docs/plan.md §5 budgets
// 4K–8K; the top of that range is the useful default).
func contextTokens() int {
	if raw := os.Getenv("FREERAG_NUM_CTX"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return agent.DefaultContextTokens
}

// dataFilePath resolves the index file: env override, else <root>/data/index.json.
func dataFilePath() string {
	if path := os.Getenv("FREERAG_DATA"); path != "" {
		return path
	}
	if cfg, err := parser.Discover(); err == nil {
		return filepath.Join(filepath.Dir(filepath.Dir(cfg.Script)), "data", "index.json")
	}
	return filepath.Join("data", "index.json")
}

// register wires the kernel's exposed RPC surface.
func (k *kernel) register(srv *ipc.Server) {
	// The transport is the server's during Serve, so the kernel borrows it
	// rather than owning a writer — there is exactly one stdout, and two
	// writers on it would interleave lines and corrupt the NDJSON stream.
	k.notify = srv.Notify
	if k.loop != nil {
		// The loop's steps are the only thing that changes during a round, so
		// forwarding them is what turns a frozen spinner into real progress.
		k.loop.OnStep = func(line string) {
			k.progress("agent", map[string]any{"line": line})
		}
	}

	srv.Register("ping", func(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
		return map[string]any{"pong": true}, nil
	})

	srv.Register("version", func(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
		reply := map[string]any{
			"name":      "freerag",
			"version":   version,
			"indexed":   k.store.Len(),
			"data_path": k.dataPath,
		}
		if k.parse != nil {
			cfg := k.parse.Config()
			reply["parse_sidecar"] = map[string]any{"python": cfg.Python, "script": cfg.Script}
		} else {
			reply["parse_sidecar"] = nil
		}
		reply["embedding"] = map[string]any{
			"enabled":  k.embedder != nil,
			"provider": k.store.EmbedderName(),
			"dims":     k.store.Dimensions(),
			"vectors":  k.store.VectorCount(),
			"indexed":  k.store.Len(),
		}
		// Which backend answers dense ranking changes both latency and memory
		// at scale, so the caller must be able to tell an approximate index
		// from the linear scan instead of assuming one.
		backend := k.store.DenseBackend()
		if backend == "" {
			backend = "in-process"
		}
		reply["dense_index"] = map[string]any{
			"backend": backend,
			"vectors": k.store.VectorCount(),
		}
		reply["checker"] = checkerName(k.loop.Checker)
		if k.generator != nil {
			reply["generator"] = map[string]any{"model": k.generator.Model, "base_url": k.generator.BaseURL}
		} else {
			reply["generator"] = nil
		}
		return reply, nil
	})

	srv.Register("parse", k.handleParse)
	srv.Register("index", k.handleIndex)
	srv.Register("search", k.handleSearch)
	srv.Register("ask", k.handleAsk)
	srv.Register("tools", k.handleTools)
	srv.Register("tool", k.handleTool)
	srv.Register("documents", k.handleDocuments)
	srv.Register("forget", k.handleForget)
	srv.Register("status", k.handleStatus)
}

// handleTools advertises the retrieval tool surface (docs/plan.md §6.7), so the
// UI and the acceptance run can assert it rather than assume it.
func (k *kernel) handleTools(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
	specs := agent.ToolSpecs()
	names := make([]string, 0, len(specs))
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	return map[string]any{"mode": k.loop.Spec.Label, "tools": specs, "names": names}, nil
}

// handleTool executes one retrieval tool call directly, without a model in the
// loop. It is the deterministic path the desktop console and tests use.
func (k *kernel) handleTool(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.Name == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.name is required"}
	}

	result, err := k.toolbox.Execute(ctx, agent.ToolCall{Name: params.Name, Arguments: params.Arguments})
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: err.Error()}
	}
	return result, nil
}

// ---- document management and health, for the desktop UI ----

// handleDocuments lists what the index holds.
//
// Without it the UI can show a chunk count but not a document list: only the
// manifest knows which file a document came from and when it was indexed.
func (k *kernel) handleDocuments(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
	records := k.store.DocumentRecords()
	documents := make([]map[string]any, 0, len(records))
	for _, record := range records {
		documents = append(documents, map[string]any{
			"md5":         record.MD5,
			"doc_id":      record.DocID,
			"source_file": record.SourceFile,
			"path":        record.Path,
			"chunk_count": record.ChunkCount,
			"page_count":  record.PageCount,
			"pipeline":    record.Pipeline,
			"indexed_at":  record.IndexedAt,
			// Live, not from the manifest: a document whose chunks went missing
			// should look wrong rather than look indexed.
			"chunks_present": k.store.ChunkCountIn(record.DocID),
		})
	}
	return map[string]any{
		"documents": documents,
		"indexed":   k.store.Len(),
		"embedded":  k.store.VectorCount(),
		"data_path": k.dataPath,
	}, nil
}

// handleForget drops a document's chunks, its manifest entry and its vectors.
func (k *kernel) handleForget(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		MD5   string `json:"md5"`
		DocID string `json:"doc_id"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	// Either key is accepted so the UI can forget a row straight from
	// `documents` without knowing which one is canonical.
	docID := params.DocID
	if docID == "" && params.MD5 != "" {
		record, ok := k.store.Document(params.MD5)
		if !ok {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams,
				Message: fmt.Sprintf("no document with md5 %s", params.MD5)}
		}
		docID = record.DocID
	}
	if docID == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams,
			Message: "params.md5 or params.doc_id is required"}
	}

	removed := k.store.RemoveDoc(docID)
	reply := map[string]any{
		"doc_id":        docID,
		"removed":       removed,
		"indexed_total": k.store.Len(),
		"embedded":      k.store.VectorCount(),
	}
	if removed > 0 {
		if saveErr := k.store.Save(k.dataPath); saveErr != nil {
			log.Printf("warning: could not persist the index: %v", saveErr)
			reply["save_error"] = saveErr.Error()
		}
	}
	return reply, nil
}

// handleStatus answers "is this app working", which is a different question from
// the one `version` answers ("what build is this").
//
// A UI needs it before letting someone press Index. Each subsystem reports its
// own state, and one that cannot be checked cheaply says `checked: false` rather
// than claiming to be fine — a health check that cannot fail is worse than none.
func (k *kernel) handleStatus(ctx context.Context, _ json.RawMessage) (any, *ipc.Error) {
	sidecar := map[string]any{"configured": k.parse != nil, "checked": false}
	if k.parse != nil {
		cfg := k.parse.Config()
		sidecar["python"] = cfg.Python
		sidecar["script"] = cfg.Script
	}

	embedding := map[string]any{
		"enabled": k.embedder != nil,
		"name":    k.store.EmbedderName(),
		"dims":    k.store.Dimensions(),
		"vectors": k.store.VectorCount(),
		// Not probed on purpose: verifying a hosted embedder costs a request
		// and a little money, on every poll.
		"checked": false,
	}

	denseIndex := map[string]any{"backend": k.store.DenseBackend(), "checked": false}
	if k.denseIndex != nil {
		denseIndex["backend"] = k.denseIndex.Backend()
		denseIndex["vectors"] = k.denseIndex.Len()
		denseIndex["checked"] = true
	}

	generator := map[string]any{"configured": k.generator != nil, "checked": false}
	if k.generator != nil {
		generator["model"] = k.generator.Model
		// Cheap: one HTTP GET against a local server. This is the failure the
		// user hits most, so it is worth actually checking.
		generator["reachable"] = k.generator.Reachable(ctx)
		generator["checked"] = true
	}

	return map[string]any{
		"index": map[string]any{
			"chunks":    k.store.Len(),
			"vectors":   k.store.VectorCount(),
			"documents": len(k.store.DocumentRecords()),
		},
		"sidecar":     sidecar,
		"embedding":   embedding,
		"dense_index": denseIndex,
		"generator":   generator,
		"checker":     map[string]any{"name": checkerName(k.loop.Checker)},
	}, nil
}

// parseParams is the argument object of `parse`, `index` and (partly) `search`.
type parseParams struct {
	Path     string `json:"path"`
	Profile  string `json:"profile"`
	MaxChars int    `json:"max_chars"`
	MaxPages int    `json:"max_pages"`
	// Force re-indexes even when the content fingerprint says it is unchanged.
	Force bool `json:"force"`
}

// pipelineFingerprint names the parse + chunk rules that produce chunks.
//
// Bump it whenever a change would make the same file yield different chunks:
// the chunker, the chunk profile defaults, the layout model, or the embedder.
// The skip decision compares it, so without a bump an old index would keep
// serving chunks built under the previous rules while reporting a cache hit —
// which is worse than not skipping at all, because nothing looks wrong.
const pipelineFingerprint = "parse-v1"

// fingerprintFor covers everything known *before* parsing that changes the
// output, so a fingerprint can be compared without paying for the parse.
func fingerprintFor(params parseParams) string {
	return fmt.Sprintf("%s|profile=%s|max_chars=%d|max_pages=%d",
		pipelineFingerprint, params.Profile, params.MaxChars, params.MaxPages)
}

// fileMD5 returns the content fingerprint of a file.
//
// MD5 is used as a change detector, not as a security primitive: the threat
// here is a stale index, and a collision would only mean a missed re-index.
//
// Streamed rather than read whole: a corpus holds PDFs, and the hash must not
// depend on the file fitting in memory.
func fileMD5(path string) (string, error) {
	handle, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("cannot read %s: %w", path, err)
	}
	defer handle.Close()

	hash := md5.New() // #nosec G401 -- change detection, not authentication
	if _, err := io.Copy(hash, handle); err != nil {
		return "", fmt.Errorf("cannot hash %s: %w", path, err)
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func (k *kernel) handleParse(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	params, err := decodeParseParams(raw)
	if err != nil {
		return nil, err
	}
	if k.parse == nil {
		return nil, errNoSidecar()
	}

	result, callErr := k.parse.Parse(ctx, params.Path, parser.Options{
		Profile:  params.Profile,
		MaxChars: params.MaxChars,
		MaxPages: params.MaxPages,
	})
	if callErr != nil {
		return nil, parseError(callErr)
	}
	return result, nil
}

// staleReason explains why a recorded document cannot be skipped, or returns ""
// when it can.
//
// Two things must hold. The pipeline fingerprint must match, or the chunks were
// built under different rules. The chunks must still be there, or the index was
// rebuilt without this document — which is exactly the case a manifest stored
// separately from the chunks would fail to notice.
func staleReason(s *store.Store, record store.DocumentRecord, fingerprint string) string {
	if record.Pipeline != fingerprint {
		return "parse rules changed"
	}
	if present := s.ChunkCountIn(record.DocID); present != record.ChunkCount {
		return fmt.Sprintf("chunks missing (%d of %d present)", present, record.ChunkCount)
	}
	return ""
}

// shortHash is the fingerprint prefix used in logs, long enough to be unique
// among a desktop corpus and short enough to read.
func shortHash(sum string) string {
	if len(sum) <= 12 {
		return sum
	}
	return sum[:12]
}

// indexReply is the `index` response.
//
// `skipped` says whether the parse and the embedding were avoided, so a caller
// can tell "nothing to do" from "nothing was found" instead of inferring it
// from a zero chunk count.
func (k *kernel) indexReply(sourceFile, sum string, pageCount, chunkCount, added, removed int, note string, skipped bool) map[string]any {
	return map[string]any{
		"source_file":   sourceFile,
		"md5":           sum,
		"page_count":    pageCount,
		"chunk_count":   chunkCount,
		"added":         added,
		"removed":       removed,
		"indexed_total": k.store.Len(),
		"embedded":      k.store.VectorCount(),
		"data_path":     k.dataPath,
		"skipped":       skipped,
		"note":          note,
	}
}

// handleIndex parses a document into the index, skipping work that is already
// done.
//
// The decision is made *before* parsing, which is what makes it worth having:
// layout detection measured 0.2–0.7 s/page with an accelerator and up to
// 7.8 s/page without, while hashing the file costs one sequential read. The
// embedding call is skipped with it, and that one is billed per token.
func (k *kernel) handleIndex(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	params, err := decodeParseParams(raw)
	if err != nil {
		return nil, err
	}

	sum, hashErr := fileMD5(params.Path)
	if hashErr != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: hashErr.Error()}
	}
	fingerprint := fingerprintFor(params)
	docID := filepath.Base(params.Path)
	k.progress("hash", map[string]any{"file": docID, "md5": sum})

	stale := ""
	if !params.Force {
		if record, ok := k.store.Document(sum); ok {
			stale = staleReason(k.store, record, fingerprint)
			if stale == "" {
				log.Printf("index: %s is unchanged (%s); skipped the parse and the embedding",
					docID, shortHash(sum))
				k.progress("skipped", map[string]any{
					"file": docID, "md5": sum, "chunk_count": record.ChunkCount,
				})
				return k.indexReply(filepath.Base(params.Path), sum, record.PageCount,
					record.ChunkCount, 0, 0, "already indexed", true), nil
			}
		}
	}

	// The sidecar is required only from here: a document that is already
	// indexed never needs a parse, so a machine without the sidecar can still
	// answer for it instead of failing outright.
	if k.parse == nil {
		return nil, errNoSidecar()
	}

	k.progress("parse", map[string]any{"file": docID})
	result, callErr := k.parse.Parse(ctx, params.Path, parser.Options{
		Profile:  params.Profile,
		MaxChars: params.MaxChars,
		MaxPages: params.MaxPages,
	})
	if callErr != nil {
		return nil, parseError(callErr)
	}
	k.progress("parsed", map[string]any{
		"file": docID, "pages": result.PageCount, "blocks": result.BlockCount,
		"chunks": result.ChunkCount, "layout": result.LayoutProvider,
	})

	// The old version is dropped *after* the parse succeeded and *before* the
	// new chunks are added. Parsing first means a failed parse leaves the
	// existing document untouched; removing before adding means the old chunks
	// cannot deduplicate the new ones away. Chunk ids come from position, not
	// content, so an edited file yields the same ids at the same offsets — and
	// without the removal the previous text would stay, under an unchanged id,
	// silently.
	removed := 0
	if previous, ok := k.store.DocumentByDocID(docID); ok {
		if previous.MD5 != sum || previous.Pipeline != fingerprint {
			removed = k.store.RemoveDoc(docID)
			log.Printf("index: %s changed (%s -> %s); dropped %d stale chunk(s)",
				docID, shortHash(previous.MD5), shortHash(sum), removed)
		}
	}

	added, indexErr := k.indexChunks(ctx, toStoreChunks(result))
	if indexErr != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: indexErr.Error()}
	}

	k.progress("stored", map[string]any{"file": docID, "added": added, "removed": removed})
	present := k.store.ChunkCountIn(docID)
	changed := added > 0 || removed > 0
	note := fmt.Sprintf("%d chunk(s) indexed", added)
	if removed > 0 {
		note = fmt.Sprintf("%d stale chunk(s) replaced, %d added", removed, added)
	}
	if stale != "" {
		note = stale + "; " + note
	}

	// Only a document that actually landed in the index is recorded. Recording
	// one with no chunks would make the next call skip a document that is not
	// there — a permanent, silent hole.
	if present > 0 {
		k.store.PutDocument(store.DocumentRecord{
			MD5:        sum,
			DocID:      docID,
			SourceFile: result.SourceFile,
			Path:       params.Path,
			ChunkCount: present,
			PageCount:  result.PageCount,
			Pipeline:   fingerprint,
			IndexedAt:  time.Now(),
		})
		changed = true
	} else {
		note = "the parse produced no chunks; the document was not recorded"
	}

	reply := k.indexReply(result.SourceFile, sum, result.PageCount, result.ChunkCount,
		added, removed, note, false)

	// A failed persist is reported in the reply, not only the log: otherwise the
	// caller sees a successful index that disappears with the process.
	if changed {
		if saveErr := k.store.Save(k.dataPath); saveErr != nil {
			log.Printf("warning: could not persist the index: %v", saveErr)
			reply["save_error"] = saveErr.Error()
		}
	}
	return reply, nil
}

// indexChunks embeds a parsed document and stores it.
//
// Embedding failure is not fatal: the chunks are stored keyword-only, and the
// reply records how many carry a vector, so the caller can tell "indexed with
// dense retrieval" from "indexed without" instead of guessing.
func (k *kernel) indexChunks(ctx context.Context, chunks []store.Chunk) (int, error) {
	if k.embedder == nil || len(chunks) == 0 {
		return k.store.Add(chunks), nil
	}

	texts := make([]string, len(chunks))
	for i, chunk := range chunks {
		texts[i] = chunk.Text
	}

	vectors, err := k.embedder.Embed(ctx, texts)
	if err != nil {
		log.Printf("warning: embedding failed (%v); indexing keyword-only", err)
		return k.store.Add(chunks), nil
	}
	return k.store.AddEmbedded(chunks, vectors)
}

func (k *kernel) handleSearch(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.Query == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.query is required"}
	}
	if params.Limit <= 0 {
		params.Limit = 10
	}

	hits := k.store.Search(params.Query, params.Limit)
	return map[string]any{"query": params.Query, "hits": hits, "count": len(hits)}, nil
}

func (k *kernel) handleAsk(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Question string `json:"question"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.Question == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.question is required"}
	}

	// Emitted before the first model call, which is the longest silent stretch
	// of the whole call: the loop reports a step only once it completes, so
	// without this the UI shows nothing at all for the first ~70 s — long
	// enough that a working call and a hung one look the same. It also feeds
	// the shell's idle timeout, which is reset by any message from the kernel.
	k.progress("thinking", map[string]any{"question": params.Question})

	result, err := k.loop.Run(ctx, params.Question)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}
	return result, nil
}

// decodeParseParams validates the shared argument object.
func decodeParseParams(raw json.RawMessage) (parseParams, *ipc.Error) {
	var params parseParams
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return params, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.Path == "" {
		return params, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.path is required"}
	}
	return params, nil
}

// errNoSidecar is the error reported when parsing is requested without a
// usable Python environment.
func errNoSidecar() *ipc.Error {
	return &ipc.Error{
		Code:    ipc.CodeInternalError,
		Message: "parse sidecar unavailable",
		Data:    map[string]any{"hint": "create .venv314 and install pymupdf (see README)"},
	}
}

// toStoreChunks converts parsed chunks into indexable passages.
func toStoreChunks(result *parser.Result) []store.Chunk {
	docID := result.SourceFile
	if docID == "" {
		docID = result.Path
	}
	out := make([]store.Chunk, 0, len(result.Chunks))
	for _, chunk := range result.Chunks {
		page, _ := chunk.Metadata["page_num"].(float64)
		blockType, _ := chunk.Metadata["block_type"].(string)
		out = append(out, store.Chunk{
			ChunkID:   chunk.ChunkID,
			Text:      chunk.Text,
			DocID:     docID,
			PageNum:   int(page),
			BlockType: blockType,
			Metadata:  chunk.Metadata,
		})
	}
	return out
}

// parseError maps a sidecar failure onto a JSON-RPC error, preserving the
// sidecar's own code so callers can distinguish "not found" (-32001) from
// "internal" (-32603).
func parseError(err error) *ipc.Error {
	var rpcErr *sidecar.Error
	if errors.As(err, &rpcErr) {
		return &ipc.Error{Code: rpcErr.Code, Message: rpcErr.Message, Data: rpcErr.Data}
	}
	return &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
}
