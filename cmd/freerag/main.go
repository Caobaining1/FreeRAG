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
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"freerag/internal/agent"
	"freerag/internal/embed"
	"freerag/internal/ipc"
	"freerag/internal/kb"
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
	parse   *parser.Service // nil when the sidecar cannot be located
	checker agent.Checker
	// decider is the raw Laya entry point, shared by the sufficiency checker,
	// the simple/complex router and the per-round tool chooser. Nil when Laya
	// is unavailable, in which case all three fall back to deterministic paths.
	decider agent.DecideFunc
	// generator is the local LLM client; nil when Ollama is unavailable and the
	// loop must fall back to extractive drafts.
	generator *agent.OllamaModel
	// embedder powers dense retrieval; nil leaves hybrid_search keyword-only.
	// Shared across knowledge bases: it is a stateless client, and every base
	// uses the same model (mixing widths is what the dimension checks refuse).
	embedder embed.Embedder

	// kbs is the persisted list of knowledge bases. Only the list: an index is
	// opened on first use and lives in `open`.
	kbs *kb.Registry
	// mu guards open, and is held across a load rather than only around the map
	// write. See kbFor for why.
	mu sync.Mutex
	// open caches loaded bases by id, for the process lifetime.
	//
	// They are not evicted. Unloading would mean re-reading an index and
	// re-checking its Qdrant collection on the next question, and what it would
	// free is the index the user is working with — the one thing least likely to
	// be idle.
	open map[string]*kbRuntime

	// notify emits a JSON-RPC notification to the UI. Set by register; nil when
	// the server has no transport (unit tests), where dropping progress is fine.
	notify func(method string, params any) error

	// indexJobs tracks the background batch-index jobs, so `index_cancel` can
	// find the one it names. Guarded by indexJobsMu: a job outlives the request
	// that started it, and finishes on its own goroutine.
	indexJobsMu sync.Mutex
	indexJobs   map[string]context.CancelFunc
	nextJobID   int64

	// captioner describes figures with a vision model; nil when the stage is
	// disabled or no model is reachable, in which case figures keep the text
	// they were extracted with. renderer crops a figure out of its page; both
	// are interfaces so the stage can be tested without Ollama or PyMuPDF.
	captioner captioner
	renderer  figureRenderer
	// visionOnce and visionSlot bound the caption stage process-wide (see
	// visionGate): the batch indexes several documents at once, so the bound
	// cannot live per document.
	visionOnce sync.Once
	visionSlot chan struct{}

	// settings is what the UI can change while the app runs (settings.go). The
	// app persists them; the kernel only holds and applies them.
	settingsMu sync.Mutex
	settings   kernelSettings
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

	log.Printf("freerag kernel %s ready (JSON-RPC 2.0 over stdio); %d knowledge base(s) under %s",
		version, len(app.kbs.List()), dataDir())
	return srv.Serve(context.Background(), os.Stdin, os.Stdout)
}

// newKernel assembles the kernel, degrading rather than failing when optional
// pieces are missing: a desktop install without the sidecar still answers.
func newKernel() (*kernel, error) {
	app := &kernel{open: map[string]*kbRuntime{}}

	if cfg, err := parser.Discover(); err != nil {
		log.Printf("warning: parse sidecar unavailable: %v", err)
	} else {
		app.parse = parser.NewService(cfg)
		app.renderer = app.parse
		log.Printf("parse sidecar: %s %s", cfg.Python, cfg.Script)
	}
	app.decider = newDecider(app.parse)
	app.checker = newChecker(app.parse, app.decider)
	app.embedder = newEmbedder()

	registry, err := kb.Open(kbRegistryPath(), kbRootDir(), legacyIndexPath())
	if err != nil {
		return nil, err
	}
	app.kbs = registry
	bases := registry.List()
	log.Printf("%d knowledge base(s); %q is the default and nothing is opened until it is used",
		len(bases), bases[0].Name)

	// No index is read here, and that is the point. The kernel used to load one
	// at startup; with several bases that would mean paying to parse every base
	// before answering a question about one. kbFor loads whichever is actually
	// asked for.
	app.generator = newGenerator()
	// The figure captioner shares the generator's client: same host, same
	// plumbing, a different model name per call. Left nil when the stage is off
	// or there is no client, and then figures keep the text they were extracted
	// with — which is what shipped before this stage existed.
	app.settings = defaultSettings()
	if settings, enabled := app.captionSettings(); enabled && app.generator != nil {
		// A copy with its own timeout, not the generator itself. One caption
		// can wait for a cold model load AND a full description while three
		// others run alongside it; measured on the reference machine, that
		// exceeded the generator's 3-minute budget and lost the figure
		// ("context deadline exceeded" with 4 figures in flight). A lost figure
		// is only a missing description — the extracted text stays — but it is
		// avoidable, and the vision model is the slowest thing here.
		vision := *app.generator
		vision.Timeout = 10 * time.Minute
		app.captioner = &vision
		log.Printf("figure captions: %s (concurrency %d, max %d tokens, longest side %dpx)",
			settings.Model, settings.Workers, settings.MaxTokens, settings.MaxSide)
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

// newDecider returns the raw Laya decision entry point, or nil when the sidecar
// cannot run it.
//
// One closure, shared by three callers: the sufficiency checker, the
// simple/complex router and the per-round tool chooser. They are the same
// sidecar method with three different option sets, so building it once is what
// keeps them from drifting — and lets a single `HasLaya` probe gate all three.
func newDecider(svc *parser.Service) agent.DecideFunc {
	if svc == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if !svc.HasLaya(ctx) {
		log.Printf("Laya unavailable (run scripts/download-models.sh laya); " +
			"routing, tool choice and sufficiency fall back to heuristics")
		return nil
	}
	log.Printf("Laya: typed decisions wired for routing, tool choice and sufficiency")
	return func(ctx context.Context, kind agent.DecisionKind, instructions string,
		criteria map[string]string, state string) (string, float64, error) {
		decision, err := svc.Decide(ctx, parser.DecisionRequest{
			Instructions: instructions,
			Criteria:     criteria,
			State:        state,
			// The CALLER states which question this is. It used to be fixed here
			// at "noul", for the sufficiency check, and that one word disabled
			// two of the three callers: the tool chooser's options became
			// {false, true} instead of the candidate calls, so it failed on every
			// round and the loop silently fell back to the generating model's own
			// plan, and the router's answer never matched "complex"/"simple" so
			// every route came from the heuristic. See agent.DecisionKind.
			//
			// noul and choice are different SHAPES, not two names for one thing:
			// noul is the bare boolean pair the sufficiency check was trained on
			// (DEPLOY_CONTRACT.md §10.2), and choice renders "<label>: <text>"
			// lines whose positions are scored.
			QType: string(kind),
		}, 0)
		if err != nil {
			return "", 0, err
		}
		return decision.Choice, decision.Probability, nil
	}
}

// newChecker returns the sufficiency checker: Laya when the decider exists,
// else the deterministic coverage heuristic.
//
// The fallback is logged rather than silent. Collapsing "Laya is unavailable"
// into "the evidence is insufficient" would mislabel every run on a machine
// without the model, and the heuristic still reports UNKNOWN when it has nothing
// to judge.
func newChecker(svc *parser.Service, decide agent.DecideFunc) agent.Checker {
	if svc == nil {
		log.Printf("sufficiency checker: coverage heuristic (no parse sidecar)")
		return agent.CoverageChecker{}
	}
	if decide == nil {
		log.Printf("sufficiency checker: coverage heuristic (Laya unavailable)")
		return agent.CoverageChecker{}
	}
	return agent.LayaChecker{Decide: decide}
}

// newEmbedder wires dense retrieval.
//
// Every failure here is non-fatal and degrades to keyword-only search: an
// unreachable embedder must not stop the kernel from answering, but it must be
// loud, because a silent fallback would leave the operator believing hybrid
// retrieval is on when it is not.
func newEmbedder() embed.Embedder {
	configured, err := embed.FromEnv()
	if err != nil {
		log.Printf("warning: dense retrieval disabled: %v", err)
		return nil
	}
	if configured == nil {
		log.Printf("dense retrieval disabled (no FREERAG_EMBED_PROVIDER); hybrid_search is keyword-only")
		return nil
	}
	log.Printf("embedder: %s (%d dims)", configured.Name(), configured.Dimensions())
	return configured
}

// bindEmbedder checks one base's index against the configured embedder.
//
// The check is per index rather than per process, because that is where the
// mismatch lives: an index built by one model cannot be queried with another's
// vectors — they are numerically comparable and semantically unrelated, so the
// result would be ranked nonsense that nothing downstream could detect.
//
// It also records the model on a base that has never been embedded, which is
// what makes the vector width known before the first document is added.
func bindEmbedder(s *store.Store, configured embed.Embedder) bool {
	if configured == nil {
		return false
	}
	if err := s.SetEmbedder(configured.Name(), configured.Dimensions()); err != nil {
		log.Printf("warning: dense retrieval disabled for this base: %v", err)
		return false
	}
	return true
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

// vlmModel is the parse-time vision model, or "" when vision is off.
//
// Off by default, and that is the safe default rather than a conservative one:
// the model is ~3.5 GB resident, this class of machine has 16 GB and is already
// swapping, and the answering model must never be resident alongside it. An
// operator who wants figure captions asks for them by name.
//
// Set on the parse request, never read on the query path — the model is for
// turning a figure into text once, at index time (see sidecar/vlm.py).
func vlmModel() string {
	return strings.TrimSpace(os.Getenv("FREERAG_VLM_MODEL"))
}

// releaseVlm gives the parse-time vision model back to the OS.
//
// Called when an index job ends, which is the only moment the model is not
// needed: it exists to turn a figure into text during a parse, and nothing on
// the query path touches it. Bounded by a short timeout — the whole point is to
// free memory promptly, so waiting on a wedged server would defeat it — and
// every failure is logged rather than returned, because the index work is
// already done by the time this runs.
func (k *kernel) releaseVlm() {
	model := vlmModel()
	if model == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := agent.UnloadModel(ctx, os.Getenv("FREERAG_OLLAMA_URL"), model); err != nil {
		log.Printf("warning: could not unload the vision model %q (%v); it expires on its own keep-alive",
			model, err)
		return
	}
	log.Printf("vision model %q unloaded after indexing", model)
}

func newGenerator() *agent.OllamaModel {
	modelName := os.Getenv("FREERAG_MODEL")
	if modelName == "" {
		modelName = agent.DefaultGeneratorModel
	}
	think := thinkingEnabled()
	gen := &agent.OllamaModel{
		BaseURL: os.Getenv("FREERAG_OLLAMA_URL"),
		Model:   modelName,
		NumCtx:  contextTokens(),
		// Raised from 512 alongside the draft prompt's change from "two to four
		// sentences" to a thorough answer. Observed answers reach ~300 tokens, so
		// 512 was already comfortable, but it is a hard stop rather than a
		// guideline — hitting it returns a half-written answer with no error —
		// and the reserve in PromptCharBudget has to accommodate whatever is set
		// here.
		NumPredict:  1024,
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
	if k.generator == nil {
		return
	}
	go func() {
		started := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		// A throwaway loop, because the real ones are per knowledge base and none
		// has been opened yet. Warming up must not be the thing that decides
		// which base to open.
		warm := &agent.Loop{Model: k.generator, Spec: agent.Medium()}
		if err := warm.Warm(ctx); err != nil {
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

// dataDir is the directory the knowledge-base registry and the per-base index
// directories live under.
//
// FREERAG_DATA still names the legacy index *file*, because that is what the
// acceptance scripts and the CLI override. Only its directory is used, so an
// override moves the whole layout — a registry left behind would be shared by
// two configurations that were meant to be independent.
func dataDir() string {
	if path := os.Getenv("FREERAG_DATA"); path != "" {
		return filepath.Dir(path)
	}
	if cfg, err := parser.Discover(); err == nil {
		return filepath.Join(filepath.Dir(filepath.Dir(cfg.Script)), "data")
	}
	return "data"
}

// legacyIndexPath is the single index file the kernel used before knowledge
// bases existed.
//
// Still honoured as the adoption source. An install that predates this would
// otherwise come up looking empty after an upgrade, and "my documents are gone"
// is a far worse outcome than an extra copy left on disk.
func legacyIndexPath() string {
	if path := os.Getenv("FREERAG_DATA"); path != "" {
		return path
	}
	return filepath.Join(dataDir(), "index.json")
}

// kbRegistryPath is the registry file.
func kbRegistryPath() string {
	if path := os.Getenv("FREERAG_KB_REGISTRY"); path != "" {
		return path
	}
	return filepath.Join(dataDir(), "kbs.json")
}

// kbRootDir holds one subdirectory per knowledge base.
func kbRootDir() string {
	if path := os.Getenv("FREERAG_KB_ROOT"); path != "" {
		return path
	}
	return filepath.Join(dataDir(), "kbs")
}

// register wires the kernel's exposed RPC surface.
func (k *kernel) register(srv *ipc.Server) {
	// The transport is the server's during Serve, so the kernel borrows it
	// rather than owning a writer — there is exactly one stdout, and two
	// writers on it would interleave lines and corrupt the NDJSON stream.
	k.notify = srv.Notify
	// Progress callbacks are attached per knowledge base, when a base is opened
	// (kb.go, attachProgress). There is no longer one process-wide loop to attach
	// them to, and a base loaded without them would answer silently.

	srv.Register("ping", func(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
		return map[string]any{"pong": true}, nil
	})

	srv.Register("version", func(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
		// Build and configuration only. The index figures that used to be here
		// moved to `status`, because they are per knowledge base now and this
		// call has no base to report on — reporting the default one would mean
		// opening it, and opening an index to answer "what build is this" is
		// exactly the cost lazy loading exists to avoid.
		reply := map[string]any{
			"name":            "freerag",
			"version":         version,
			"data_dir":        dataDir(),
			"kb_registry":     kbRegistryPath(),
			"knowledge_bases": len(k.kbs.List()),
		}
		if k.parse != nil {
			cfg := k.parse.Config()
			reply["parse_sidecar"] = map[string]any{"python": cfg.Python, "script": cfg.Script}
		} else {
			reply["parse_sidecar"] = nil
		}

		// Read from the embedder rather than from a store: the model is a
		// process-wide setting, and the per-index check (that the vectors were
		// built by this model) happens in bindEmbedder when a base is opened.
		embedding := map[string]any{"enabled": k.embedder != nil}
		if k.embedder != nil {
			embedding["provider"] = k.embedder.Name()
			embedding["dims"] = k.embedder.Dimensions()
		}
		reply["embedding"] = embedding

		reply["checker"] = checkerName(k.checker)
		if k.generator != nil {
			reply["generator"] = map[string]any{"model": k.generator.Model, "base_url": k.generator.BaseURL}
		} else {
			reply["generator"] = nil
		}
		return reply, nil
	})

	srv.Register("parse", k.handleParse)
	srv.Register("index", k.handleIndex)
	// The batch form is asynchronous on purpose: Serve handles one request at a
	// time, so a synchronous batch could never be cancelled.
	srv.Register("index_batch", k.handleIndexBatch)
	srv.Register("index_cancel", k.handleIndexCancel)
	// Settings the UI can change while the app runs. The app owns the file; the
	// kernel is told what is in it (here, and again on every change).
	srv.Register("settings_get", func(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
		return handleSettingsGet(k)
	})
	srv.Register("settings_set", func(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
		return handleSettingsSet(k, raw)
	})
	srv.Register("search", k.handleSearch)
	srv.Register("ask", k.handleAsk)
	srv.Register("tools", k.handleTools)
	srv.Register("tool", k.handleTool)
	srv.Register("documents", k.handleDocuments)
	// The chunk inspector: one document's chunks with their page-space boxes,
	// and the original page as an image. Together they are what the UI needs to
	// show a chunk beside the region it was cut from.
	srv.Register("chunks", k.handleChunks)
	srv.Register("page", k.handlePage)
	srv.Register("forget", k.handleForget)
	srv.Register("status", k.handleStatus)

	// Knowledge bases. Listing never opens one; creating one only writes the
	// registry, so an empty base costs a name rather than an index.
	srv.Register("kb.list", k.handleKBList)
	srv.Register("kb.create", k.handleKBCreate)
	srv.Register("kb.rename", k.handleKBRename)
	srv.Register("kb.delete", k.handleKBDelete)
}

// handleTools advertises the retrieval tool surface (docs/plan.md §6.7), so the
// UI and the acceptance run can assert it rather than assume it.
func (k *kernel) handleTools(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
	specs := agent.ToolSpecs()
	names := make([]string, 0, len(specs))
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	return map[string]any{"mode": agent.Medium().Label, "tools": specs, "names": names}, nil
}

// handleTool executes one retrieval tool call directly, without a model in the
// loop. It is the deterministic path the desktop console and tests use.
func (k *kernel) handleTool(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
		KB        string         `json:"kb"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.Name == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.name is required"}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	result, err := live.toolbox.Execute(ctx, agent.ToolCall{Name: params.Name, Arguments: params.Arguments})
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
func (k *kernel) handleDocuments(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		KB string `json:"kb"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	records := live.store.DocumentRecords()
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
			"chunks_present": live.store.ChunkCountIn(record.DocID),
		})
	}

	// The registry's cached counts for this base are refreshed here. It is a
	// read call, but the write is a no-op unless the figures actually moved, and
	// this is the one place they are known to be live — so the base list cannot
	// keep advertising documents that something removed.
	k.recordCounts(live)

	return map[string]any{
		"kb":        baseSummary(live.base),
		"documents": documents,
		"indexed":   live.store.Len(),
		"embedded":  live.store.VectorCount(),
		"data_path": live.dataPath,
	}, nil
}

// handleChunks lists one document's chunks, each with the page-space box it was
// cut from.
//
// The desktop UI puts a chunk beside the page it came from, so the box has to
// travel with the text. It is already in the index — the parser writes bbox into
// every chunk's metadata (sidecar/chunking.py) — and this handler is what stops
// it being unreachable from the shell. Boxes are in PDF POINTS with a top-left
// origin, page-local, which is the same space `page.rect` uses: the renderer
// divides by the size the `page` call returns.
func (k *kernel) handleChunks(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		KB     string `json:"kb"`
		DocID  string `json:"doc_id"`
		Page   int    `json:"page"`
		Offset int    `json:"offset"`
		Limit  int    `json:"limit"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.DocID == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.doc_id is required"}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	// The per-page counts span the WHOLE document, not the page slice that was
	// asked for: the UI's page arrows have to know which pages exist before one
	// is chosen, and a count per page is what lets it offer only those.
	onPage := map[int]int{}
	total := 0
	for _, chunk := range live.store.List(params.DocID, 0, 0, 0) {
		total++
		onPage[chunk.PageNum]++
	}
	pageNumbers := make([]int, 0, len(onPage))
	for page := range onPage {
		pageNumbers = append(pageNumbers, page)
	}
	sort.Ints(pageNumbers)

	pages := make([]map[string]any, 0, len(pageNumbers))
	for _, page := range pageNumbers {
		pages = append(pages, map[string]any{"page": page, "chunks": onPage[page]})
	}

	selected := live.store.List(params.DocID, params.Page, params.Offset, params.Limit)
	chunks := make([]map[string]any, 0, len(selected))
	for _, chunk := range selected {
		chunks = append(chunks, map[string]any{
			"chunk_id":   chunk.ChunkID,
			"page_num":   chunk.PageNum,
			"block_type": chunk.BlockType,
			"chars":      len([]rune(chunk.Text)),
			"bbox":       chunkBox(chunk),
			"text":       chunk.Text,
			// Consolidation lineage, when there is any. A merged chunk came from
			// several blocks, and a chunk attached to another is why a document's
			// chunk count and its block count differ — worth showing rather than
			// leaving as an unexplained gap.
			"merged_from": chunk.Metadata["merged_from"],
			"attached_to": chunk.Metadata["attached_to"],
		})
	}

	doc := map[string]any{"doc_id": params.DocID}
	if record, ok := live.store.DocumentByDocID(params.DocID); ok {
		doc = map[string]any{
			"doc_id":      record.DocID,
			"source_file": record.SourceFile,
			"path":        record.Path,
			"page_count":  record.PageCount,
			"indexed_at":  record.IndexedAt,
		}
	}

	return map[string]any{
		"doc":    doc,
		"pages":  pages,
		"total":  total,
		"offset": params.Offset,
		"limit":  params.Limit,
		"chunks": chunks,
	}, nil
}

// handlePage renders one page of a document's original file.
//
// The file is read from where it was indexed; the manifest keeps that path. It
// may have moved or been deleted since, and that is reported as an error the UI
// can show — drawing highlights over a blank page would read as a data problem
// rather than as a missing file.
func (k *kernel) handlePage(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		KB    string `json:"kb"`
		DocID string `json:"doc_id"`
		Page  int    `json:"page"`
		DPI   int    `json:"dpi"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.DocID == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.doc_id is required"}
	}
	if params.Page < 1 {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.page is required and 1-based"}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	if k.parse == nil {
		return nil, &ipc.Error{
			Code:    ipc.CodeInternalError,
			Message: "the parse sidecar is unavailable, so pages cannot be rendered",
		}
	}

	record, ok := live.store.DocumentByDocID(params.DocID)
	if !ok {
		return nil, &ipc.Error{
			Code:    ipc.CodeInvalidParams,
			Message: fmt.Sprintf("unknown document %q", params.DocID),
		}
	}
	if record.Path == "" {
		return nil, &ipc.Error{
			Code:    ipc.CodeInvalidParams,
			Message: fmt.Sprintf("%s was indexed without a recorded path, so its pages cannot be read", record.DocID),
		}
	}

	page, err := k.parse.Render(ctx, parser.RenderRequest{
		Path: record.Path,
		Page: params.Page,
		DPI:  params.DPI,
	}, 0)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}

	return map[string]any{
		"doc_id":      record.DocID,
		"source_file": record.SourceFile,
		"page":        page.Page,
		"pages":       page.Pages,
		"dpi":         page.DPI,
		"width_pt":    page.WidthPT,
		"height_pt":   page.HeightPT,
		"width_px":    page.WidthPX,
		"height_px":   page.HeightPX,
		"image":       page.Image,
	}, nil
}

// chunkBox reads a chunk's page-space box out of its metadata.
//
// After the index's JSON round trip the numbers are float64 inside a []any, so
// this is what turns them back into the four floats a highlight is positioned
// with. A chunk with no usable box returns nil, and the UI then draws no
// rectangle rather than one at the origin.
func chunkBox(chunk store.Chunk) []float64 {
	raw, ok := chunk.Metadata["bbox"]
	if !ok {
		return nil
	}

	var items []any
	switch typed := raw.(type) {
	case []any:
		items = typed
	case []float64:
		items = make([]any, 0, len(typed))
		for _, value := range typed {
			items = append(items, value)
		}
	default:
		return nil
	}
	if len(items) != 4 {
		return nil
	}

	box := make([]float64, 4)
	for i, item := range items {
		value, ok := item.(float64)
		if !ok {
			return nil
		}
		box[i] = value
	}
	// A zero-area or inverted box is not a region; it would draw as a dot or as
	// nothing at all, and either reads as a rendering bug.
	if box[2] <= box[0] || box[3] <= box[1] {
		return nil
	}
	return box
}

// handleForget drops a document's chunks, its manifest entry and its vectors.
func (k *kernel) handleForget(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		MD5   string `json:"md5"`
		DocID string `json:"doc_id"`
		KB    string `json:"kb"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	// Either key is accepted so the UI can forget a row straight from
	// `documents` without knowing which one is canonical.
	docID := params.DocID
	if docID == "" && params.MD5 != "" {
		record, ok := live.store.Document(params.MD5)
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

	removed := live.store.RemoveDoc(docID)
	reply := map[string]any{
		"kb":            baseSummary(live.base),
		"doc_id":        docID,
		"removed":       removed,
		"indexed_total": live.store.Len(),
		"embedded":      live.store.VectorCount(),
	}
	if removed > 0 {
		if saveErr := live.store.Save(live.dataPath); saveErr != nil {
			log.Printf("warning: could not persist the index: %v", saveErr)
			reply["save_error"] = saveErr.Error()
		}
		k.recordCounts(live)
	}
	return reply, nil
}

// handleStatus answers "is this app working", which is a different question from
// the one `version` answers ("what build is this").
//
// A UI needs it before letting someone press Index. Each subsystem reports its
// own state, and one that cannot be checked cheaply says `checked: false` rather
// than claiming to be fine — a health check that cannot fail is worse than none.
func (k *kernel) handleStatus(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	// The index half of this report is per knowledge base, so the caller names
	// one. An omitted id resolves to the first base, which is what the health
	// poll did before knowledge bases existed.
	var params struct {
		KB string `json:"kb"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	sidecar := map[string]any{"configured": k.parse != nil, "checked": false}
	if k.parse != nil {
		cfg := k.parse.Config()
		sidecar["python"] = cfg.Python
		sidecar["script"] = cfg.Script
	}

	embedding := map[string]any{
		"enabled": k.embedder != nil,
		"name":    live.store.EmbedderName(),
		"dims":    live.store.Dimensions(),
		"vectors": live.store.VectorCount(),
		// Not probed on purpose: verifying a hosted embedder costs a request
		// and a little money, on every poll.
		"checked": false,
	}

	denseIndex := map[string]any{"backend": live.store.DenseBackend(), "checked": false}
	if live.denseIndex != nil {
		denseIndex["backend"] = live.denseIndex.Backend()
		denseIndex["vectors"] = live.denseIndex.Len()
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
		"kb": baseSummary(live.base),
		"index": map[string]any{
			"chunks":    live.store.Len(),
			"vectors":   live.store.VectorCount(),
			"documents": len(live.store.DocumentRecords()),
		},
		"sidecar":     sidecar,
		"embedding":   embedding,
		"dense_index": denseIndex,
		"generator":   generator,
		"checker":     map[string]any{"name": checkerName(k.checker)},
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
	// KB names the knowledge base to act on; empty means the first one.
	KB string `json:"kb"`
	// VlmModel names the parse-time vision model (figures → text). Process
	// configuration, not a request field, so it is filled from the environment
	// and never parsed from the wire — one setting for every parse this kernel
	// performs.
	VlmModel string `json:"-"`
}

// pipelineFingerprint names the parse + chunk rules that produce chunks.
//
// Bump it whenever a change would make the same file yield different chunks:
// the chunker, the chunk profile defaults, the layout model, or the embedder.
// The skip decision compares it, so without a bump an old index would keep
// serving chunks built under the previous rules while reporting a cache hit —
// which is worse than not skipping at all, because nothing looks wrong.
const pipelineFingerprint = "parse-v9"

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
		VlmModel: params.VlmModel,
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
	// The index totals are not here: they belong to a knowledge base, and this
	// function does not have one. withKB adds them along with the label, so a
	// reply cannot report one base's counts beside another's source file.
	return map[string]any{
		"source_file": sourceFile,
		"md5":         sum,
		"page_count":  pageCount,
		"chunk_count": chunkCount,
		"added":       added,
		"removed":     removed,
		"skipped":     skipped,
		"note":        note,
	}
}

// handleIndex parses a document into the index, skipping work that is already
// done.
//
// The decision is made *before* parsing, which is what makes it worth having:
// layout detection measured 0.2–0.7 s/page with an accelerator and up to
// 7.8 s/page without, while hashing the file costs one sequential read. The
// embedding call is skipped with it, and that one is billed per token.
// handleIndex indexes one document and waits for the result.
//
// A single-file add stays a single round trip, and the per-document path is the
// one the tests drive without having to run a batch job.
func (k *kernel) handleIndex(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	params, err := decodeParseParams(raw)
	if err != nil {
		return nil, err
	}
	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	// Exclusive against `ask` on this base (see kbRuntime.indexing).
	live.indexing.Lock()
	defer live.indexing.Unlock()

	reply, indexErr := k.indexOne(ctx, live, params, nil)
	// Released even when this document failed: the model was loaded during the
	// attempt, and a partially-parsed file is no reason to keep 3.5 GB resident.
	k.releaseVlm()
	if indexErr != nil {
		return nil, indexErr
	}
	return withKB(reply, live), nil
}

// indexOne indexes one document into an already-resolved base.
//
// `gate` is the batch job's parse gate, or nil for a lone document: it bounds
// how many parses run at once, and is held only for the parse call itself so the
// embedding that follows can overlap (see indexjob.go).
func (k *kernel) indexOne(ctx context.Context, live *kbRuntime, params parseParams, gate parseGate) (map[string]any, *ipc.Error) {
	sum, hashErr := fileMD5(params.Path)
	if hashErr != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: hashErr.Error()}
	}
	fingerprint := fingerprintFor(params)
	docID := filepath.Base(params.Path)
	k.progress("hash", map[string]any{"file": docID, "md5": sum, "kb": live.base.ID})

	stale := ""
	if !params.Force {
		if record, ok := live.store.Document(sum); ok {
			stale = staleReason(live.store, record, fingerprint)
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
	result, callErr := k.parseDocument(ctx, params, gate)
	if callErr != nil {
		return nil, parseError(callErr)
	}
	k.progress("parsed", map[string]any{
		"file": docID, "pages": result.PageCount, "blocks": result.BlockCount,
		"chunks": result.ChunkCount, "layout": result.LayoutProvider,
	})

	// Figure descriptions belong here: before the chunks are embedded, because
	// the description is part of what gets embedded, and after the parse,
	// because the figures are what the parse found.
	if settings, enabled := k.captionSettings(); enabled {
		if described := k.captionFigures(ctx, params.Path, result, settings); described > 0 {
			k.progress("figures", map[string]any{"file": docID, "described": described})
		}
	}

	// The old version is dropped *after* the parse succeeded and *before* the
	// new chunks are added. Parsing first means a failed parse leaves the
	// existing document untouched; removing before adding means the old chunks
	// cannot deduplicate the new ones away. Chunk ids come from position, not
	// content, so an edited file yields the same ids at the same offsets — and
	// without the removal the previous text would stay, under an unchanged id,
	// silently.
	removed := 0
	if previous, ok := live.store.DocumentByDocID(docID); ok {
		if previous.MD5 != sum || previous.Pipeline != fingerprint {
			removed = live.store.RemoveDoc(docID)
			log.Printf("index: %s changed (%s -> %s); dropped %d stale chunk(s)",
				docID, shortHash(previous.MD5), shortHash(sum), removed)
		}
	}

	added, indexErr := k.indexChunks(ctx, live, toStoreChunks(result))
	if indexErr != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: indexErr.Error()}
	}

	k.progress("stored", map[string]any{"file": docID, "added": added, "removed": removed})
	present := live.store.ChunkCountIn(docID)
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
		live.store.PutDocument(store.DocumentRecord{
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
		if saveErr := live.store.Save(live.dataPath); saveErr != nil {
			log.Printf("warning: could not persist the index: %v", saveErr)
			reply["save_error"] = saveErr.Error()
		}
		k.recordCounts(live)
	}
	return reply, nil
}

// withKB labels a reply with the base it acted on, and its live index figures.
//
// Every index-touching reply carries it, because the caller may have omitted the
// id and the result is otherwise indistinguishable from one for a base it did
// name — which is exactly the mistake worth being able to see.
//
// The totals are read here rather than at each call site so they always come
// from the same base the label does.
func withKB(reply map[string]any, live *kbRuntime) map[string]any {
	reply["kb"] = baseSummary(live.base)
	reply["indexed_total"] = live.store.Len()
	reply["embedded"] = live.store.VectorCount()
	reply["data_path"] = live.dataPath
	return reply
}

// indexChunks embeds a parsed document and stores it.
//
// Embedding failure is not fatal: the chunks are stored keyword-only, and the
// reply records how many carry a vector, so the caller can tell "indexed with
// dense retrieval" from "indexed without" instead of guessing.
func (k *kernel) indexChunks(ctx context.Context, live *kbRuntime, chunks []store.Chunk) (int, error) {
	if k.embedder == nil || len(chunks) == 0 {
		return live.store.Add(chunks), nil
	}

	texts := make([]string, len(chunks))
	for i, chunk := range chunks {
		texts[i] = chunk.Text
	}

	vectors, err := k.embedder.Embed(ctx, texts)
	if err != nil {
		log.Printf("warning: embedding failed (%v); indexing keyword-only", err)
		return live.store.Add(chunks), nil
	}
	return live.store.AddEmbedded(chunks, vectors)
}

func (k *kernel) handleSearch(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
		KB    string `json:"kb"`
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

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	hits := live.store.Search(params.Query, params.Limit)
	return withKB(map[string]any{
		"query": params.Query, "hits": hits, "count": len(hits),
	}, live), nil
}

func (k *kernel) handleAsk(ctx context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Question string `json:"question"`
		KB       string `json:"kb"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.Question == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.question is required"}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	// Held for the whole answer: an index into this base must not run while a
	// question is being answered from it. RLock, so several questions may share
	// the base — only indexing is exclusive.
	live.indexing.RLock()
	defer live.indexing.RUnlock()

	// Emitted before the first model call, which is the longest silent stretch
	// of the whole call: the loop reports a step only once it completes, so
	// without this the UI shows nothing at all for the first ~70 s — long
	// enough that a working call and a hung one look the same. It also feeds
	// the shell's idle timeout, which is reset by any message from the kernel.
	k.progress("thinking", map[string]any{"question": params.Question, "kb": live.base.ID})

	// The flow routes the question and, on the complex path, fans out over
	// sub-questions; the loop is the unit it runs. A base whose flow failed to
	// build still answers through the single agentic pass.
	var (
		result *agent.Result
		err    error
	)
	if live.flow != nil {
		result, err = live.flow.Run(ctx, params.Question)
	} else {
		result, err = live.loop.Run(ctx, params.Question)
	}
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}
	return askReply{Result: result, KB: baseSummary(live.base)}, nil
}

// askReply is the `ask` response: the loop's result, plus the base it came from.
//
// The loop's fields are embedded rather than nested so the reply keeps the shape
// callers already read; `kb` is added alongside them, because a caller that
// omitted the id has no other way to tell which base answered.
type askReply struct {
	*agent.Result
	KB map[string]any `json:"kb"`
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
	params.VlmModel = vlmModel()
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
