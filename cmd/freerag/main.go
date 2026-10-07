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
	// layaReady is whether the typed-decision model answered at startup.
	//
	// Kept separately from `decider` because the two are not the same question:
	// decider is the agent-loop closure, and it is nil both when Laya is missing
	// and when a caller wants a different one. The directory-tree router asks
	// "may I route with the model", and a wrong yes there would send every query
	// through a decider that does not exist — silently keyword-routed, which is
	// indistinguishable from the model having chosen those folders.
	layaReady bool
	// deepdoc is what the sidecar's warm-up reported: which provider the layout
	// and table-structure models actually run on. Nil until the warm-up
	// finishes, and never a guess in the meantime — a machine can offer an
	// accelerator and still land on CPU, and that is silent everywhere except
	// in this report.
	deepdocMu sync.Mutex
	deepdoc   *parser.WarmupReport
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
	// chart is the Laya-Chart backend (vision_chart.go). Set instead of
	// captioner, never alongside it: the two describe the same figure and one
	// of them would be describing it twice.
	chart *chartClient
	// chartService is that backend's process, when this kernel started it.
	// Nil when the service was already running at FREERAG_CHART_ENDPOINT, in
	// which case it is not ours to stop.
	chartService *chartService
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
	if app.chartService != nil {
		// Stopped on the way out so its 7.4 GB of weights do not stay resident
		// in a machine the user has moved on from.
		defer func() {
			if err := app.chartService.Close(); err != nil {
				log.Printf("chart service shutdown: %v", err)
			}
		}()
	}

	srv := ipc.NewServer()
	app.register(srv)

	// Started before Serve so it overlaps with the shell's own startup, and
	// deliberately not awaited: it exists to keep the first question fast, so
	// blocking readiness on it would trade one wait for another.
	//
	// All three warm-ups overlap on purpose: one builds the generator's weights
	// on the GPU, the other two build ONNX sessions on the CPU, and none is on
	// the path of another. Serialising them would spend the same wall time
	// twice for no reason.
	//
	// The two ONNX ones do share the sidecar process, which is fine because
	// they are different models under different registry locks — and because
	// Laya's session stays on CPU (measured 15x slower on CoreML), so it is not
	// competing for the ANE compiler that the layout model is using.
	app.warmGenerator()
	app.warmDecider()
	app.warmDeepdoc()

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
	app.layaReady = app.decider != nil
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
	if settings, enabled := app.captionSettings(); enabled {
		switch settings.Backend {
		case chartBackend:
			// The chart backend needs no generator: it reaches its own service,
			// so a machine with no Ollama can still index charts.
			//
			// Started here unless the operator pointed at one that is already
			// running. Owning the process is the difference between "charts are
			// indexed" and "the backend is configured but nothing describes
			// anything", which look identical from the outside.
			endpoint := settings.Endpoint
			if _, external := os.LookupEnv("FREERAG_CHART_ENDPOINT"); !external {
				// Checked before the process is started rather than after it
				// has swapped: the weights are resident, so the machine this
				// runs on has to have room for them, and a model that does not
				// fit does not error — it takes the desktop down with it.
				if err := checkChartMemory(); err != nil {
					log.Printf("warning: figure captions disabled: %v", err)
					break
				}
				service, err := startChartService()
				if err != nil {
					// Not fatal: figures keep their extracted text, which is
					// what shipped before this backend existed.
					log.Printf("warning: figure captions disabled: %v", err)
					break
				}
				app.chartService = service
				endpoint = service.endpoint
			}
			app.chart = newChartClient(endpoint, defaultChartTimeout)
			log.Printf("figure captions: %s at %s (concurrency %d, longest side %dpx)%s",
				settings.Model, endpoint, settings.Workers, settings.MaxSide,
				ownedSuffix(app.chartService))
		default:
			if app.generator == nil {
				break
			}
			// A copy with its own timeout, not the generator itself. One caption
			// can wait for a cold model load AND a full description while three
			// others run alongside it; measured on the reference machine, that
			// exceeded the generator's 3-minute budget and lost the figure
			// ("context deadline exceeded" with 4 figures in flight). A lost
			// figure is only a missing description — the extracted text stays —
			// but it is avoidable, and the vision model is the slowest thing
			// here.
			vision := *app.generator
			vision.Timeout = 10 * time.Minute
			app.captioner = &vision
			log.Printf("figure captions: %s (concurrency %d, max %d tokens, longest side %dpx)",
				settings.Model, settings.Workers, settings.MaxTokens, settings.MaxSide)
		}
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
// generationTemperature is the sampler temperature for answers.
//
// 0.2 by default, not 0: answers read better with a little variation, and this is a
// product, not a benchmark.
//
// But a MEASUREMENT cannot afford it, and this number is the reason the
// self-improvement loop's first two decisions were meaningless. Three dev-loop runs
// of the same effective code scored quality_macro 0.3672, 0.3735 and 0.4465 — a
// range of 0.079 — and the loop accepted a change worth +0.079 that turned out to be
// inert (docs/rsi-ledger.md). The spread is this sampler, and it is wider than every
// change the loop has proposed. Evaluation runs set FREERAG_GENERATION_TEMPERATURE=0
// (scripts/ragas_eval.py does it for every kernel it starts); 0 is also the value
// OllamaModel already uses for the tool-plan path.
func generationTemperature() float64 {
	if raw := strings.TrimSpace(os.Getenv("FREERAG_GENERATION_TEMPERATURE")); raw != "" {
		if value, err := strconv.ParseFloat(raw, 64); err == nil && value >= 0 {
			return value
		}
	}
	return 0.2
}

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
	// The parse-time captioner (sidecar/vlm.py) only speaks Ollama. With the
	// chart backend selected, FREERAG_VLM_MODEL names a service Ollama has never
	// heard of, so passing it along would both fail a pull for
	// "laya-chart-1.7b" on every figure and describe every figure twice — once
	// per path, which is exactly what the split between them exists to prevent.
	if env, enabled := visionConfig(); enabled && env.Backend == chartBackend {
		return ""
	}
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
		Temperature: generationTemperature(),
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

// warmDecider builds Laya's inference session in the background.
//
// The session is built on the first decision, not on `HasLaya` — a 1.7 GB fp32
// checkpoint is not something a parse-only workload should pay for — so without
// this the cost lands inside the first question's routing step. Measured on the
// reference machine (MacBook Air M5): ~20 s of the first answer's wait, spent
// in a step the UI has already announced as "正在决定检索动作", i.e. a wait
// that looks exactly like a hang.
//
// Same contract as warmGenerator: not awaited, errors logged and dropped. A
// decision that cannot be made degrades to the heuristics the kernel already
// falls back to, and a warm-up failure is that one step earlier.
func (k *kernel) warmDecider() {
	if k.decider == nil {
		return
	}
	go func() {
		started := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		// Two options and a `choice` kind, which is the shape routing and tool
		// choice use. The point is to exercise the whole path — tokenize,
		// build, score — not to get a meaningful answer, so the text is
		// throwaway.
		_, _, err := k.decider(ctx, agent.DecisionChoice,
			"Warm-up: which option?", map[string]string{"a": "a", "b": "b"}, "warm-up")
		if err != nil {
			log.Printf("warning: Laya warm-up failed (%v); the first question will pay for building the session", err)
			return
		}
		log.Printf("Laya warm-up done in %s", time.Since(started).Round(time.Millisecond))
	}()
}

// warmDeepdoc builds the layout and table-structure sessions in the sidecar.
//
// The cost being moved is real and one-off: the accelerated layout session
// compiles 657 of the model's 681 nodes and measured 7.5 s here, and it used to
// be charged to whichever document the user parsed first — inside a step the UI
// had already announced, which reads as a hang rather than as a one-off.
//
// The second thing this buys is the provider each session actually got, logged
// rather than assumed. That is the whole point of measuring instead of
// trusting the platform: the same build resolves "auto" to CoreML on this
// machine and to CPU on one whose onnxruntime lacks the provider, and the
// difference is silent everywhere except here.
//
// Same contract as the other two: not awaited, errors logged and dropped.
func (k *kernel) warmDeepdoc() {
	if k.parse == nil {
		return
	}
	go func() {
		started := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()

		report, err := k.parse.Warmup(ctx)
		if err != nil {
			log.Printf("warning: deepdoc warm-up failed (%v); the first document will pay for building the session", err)
			return
		}
		k.deepdocMu.Lock()
		k.deepdoc = report
		k.deepdocMu.Unlock()
		if !report.Layout.Available && !report.TSR.Available {
			log.Printf("deepdoc warm-up: neither the layout nor the table model is installed")
			return
		}
		for name, model := range map[string]parser.WarmupModel{
			"layout": report.Layout,
			"tsr":    report.TSR,
		} {
			if !model.Available {
				continue
			}
			if model.Error != "" {
				log.Printf("warning: deepdoc %s session failed (%s); it will fall back to CPU", name, model.Error)
				continue
			}
			log.Printf("deepdoc %s session built in %.1fs on %s", name, model.Seconds, model.Provider)
		}
		log.Printf("deepdoc warm-up done in %s; the sessions are reused for the life of the sidecar",
			time.Since(started).Round(time.Millisecond))
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

// refragConfig reads the selective evidence compression settings (refrag.go).
//
// OFF unless FREERAG_REFRAG says otherwise, and the switch is spelled out
// rather than implied by the presence of the other two: a deployment that sets
// only a tunable must not silently change its prompt. Anything unrecognised is
// off, for the same reason.
func refragConfig() agent.RefragConfig {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FREERAG_REFRAG"))) {
	case "1", "on", "true", "yes":
	default:
		return agent.RefragConfig{}
	}
	cfg := agent.RefragConfig{
		Enabled:   true,
		ExpandTop: agent.DefaultRefragExpandTop,
		GistChars: agent.DefaultRefragGistChars,
	}
	if raw := strings.TrimSpace(os.Getenv("FREERAG_REFRAG_EXPAND_TOP")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed >= 0 {
			cfg.ExpandTop = parsed
		}
	}
	if raw := strings.TrimSpace(os.Getenv("FREERAG_REFRAG_GIST_CHARS")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			cfg.GistChars = parsed
		}
	}
	return cfg
}

// streamCoalesceWindow is how long an answer fragment may wait to be merged
// with the next one. Zero — the default — emits every fragment immediately.
//
// Off by default because the measurement says so (docs/performance.md): the
// generator emits ~6.4 tokens/s, so a frame costs microseconds and there is
// nothing to amortise, while the window is pure added latency. The switch
// exists for machines where the generator is much faster.
func streamCoalesceWindow() time.Duration {
	if raw := strings.TrimSpace(os.Getenv("FREERAG_STREAM_COALESCE_MS")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return time.Duration(parsed) * time.Millisecond
		}
	}
	return 0
}

// streamCoalesceMaxRunes bounds one coalesced frame; <=0 lets the window alone
// decide when to flush.
func streamCoalesceMaxRunes() int {
	if raw := strings.TrimSpace(os.Getenv("FREERAG_STREAM_COALESCE_RUNES")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 0
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

	// What this machine is actually running on, as opposed to what it offers.
	//
	// The gap between those two is why this exists: the shell can list a GPU,
	// but an onnxruntime build without the matching provider resolves "auto" to
	// CPU and says nothing about it. The session's own report is the only
	// honest answer, so that is what is handed back.
	srv.Register("hardware", func(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
		k.deepdocMu.Lock()
		report := k.deepdoc
		k.deepdocMu.Unlock()

		reply := map[string]any{}
		if report == nil {
			// Not warmed yet, or no sidecar: reported as such rather than
			// filled in with what the platform would suggest.
			reply["deepdoc"] = map[string]any{"warmed": false}
		} else {
			reply["deepdoc"] = map[string]any{
				"warmed":    true,
				"providers": report.Providers,
				"layout":    report.Layout,
				"tsr":       report.TSR,
			}
		}
		// A startup decision, read the way the shell makes it: FREERAG_VLM is
		// how the chart backend is asked for, and a service this kernel started
		// is the same answer arriving by another route.
		reply["chart"] = map[string]any{
			"enabled": k.chartService != nil || os.Getenv("FREERAG_VLM") == "chart",
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

	// The tree channel: documents reduced to their own structure, searched
	// without vectors (internal/tree). Separate methods rather than a flag on
	// `search`, because the two return different things — `search` returns
	// passages, `tree.search` returns passages WITH the section path that found
	// them and the trace proving it.
	srv.Register("tree.status", k.handleTreeStatus)
	srv.Register("tree.index", k.handleTreeIndex)
	srv.Register("tree.search", k.handleTreeSearch)

	// The document-level channel: the corpus as a directory tree whose leaves
	// are documents, routed one typed decision at a time (internal/dirtree).
	// Named `fs.*` because that is what it is — a file system over the corpus —
	// and because a caller that confuses it with `tree.*` would be asking for
	// passages from a channel that never returns one.
	srv.Register("fs.status", k.handleDirStatus)
	srv.Register("fs.index", k.handleDirIndex)
	srv.Register("fs.search", k.handleDirSearch)

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
		k.forgetTree(live, docID)
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
		// The tree belongs to this document, so it is rebuilt here rather than
		// on demand: a search that found no tree would silently fall back to flat
		// BM25, and nothing in its answer would say so. A failure is reported in
		// the reply for the same reason a failed index save is — otherwise the
		// caller sees a successful index without this channel in it.
		//
		// A batch hands the documents in one at a time and saves once at the end
		// (indexjob), because tree.json holds every document and writing it per
		// file would make the last write of a hundred cost a hundred other
		// documents' worth of bytes.
		k.syncTree(live, docID, result.SourceFile, reply, gate == nil)
	}
	return reply, nil
}

// syncTree rebuilds one document's tree after its chunks changed.
//
// A failure never fails the index: the chunks are stored and answerable, and the
// tree is an additional channel rather than a precondition for retrieval. It IS
// reported in the reply, because a silent degradation here would look like the
// tree channel having nothing to say when it actually has no tree.
func (k *kernel) syncTree(live *kbRuntime, docID, sourceFile string, reply map[string]any, persist bool) {
	count, err := k.rebuildTree(live, docID, sourceFile, persist)
	if err != nil {
		log.Printf("warning: could not rebuild the tree for %s: %v", docID, err)
		reply["tree_error"] = err.Error()
		return
	}
	reply["tree_chunks"] = count
	if count > 0 {
		if doc := live.trees.Tree(docID); doc != nil {
			reply["tree_levels"] = doc.LevelsFrom
			reply["tree_nodes"] = len(doc.Nodes)
		}
	}
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
	//
	// The stream sink is installed around the run and removed after it, so two
	// concurrent questions cannot share one coalescer's buffer. When coalescing
	// is off (the default) this is a no-op and the default sink is used.
	endStream := k.beginAnswerStream(live)
	var (
		result *agent.Result
		err    error
	)
	if live.flow != nil {
		result, err = live.flow.Run(ctx, params.Question)
	} else {
		result, err = live.loop.Run(ctx, params.Question)
	}
	endStream(result)
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
