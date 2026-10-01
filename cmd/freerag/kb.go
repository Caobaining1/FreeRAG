package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"

	"freerag/internal/agent"
	"freerag/internal/dense"
	"freerag/internal/embed"
	"freerag/internal/ipc"
	"freerag/internal/kb"
	"freerag/internal/store"
)

// kbRuntime is one knowledge base's live state.
//
// The store, the ANN index, the toolbox and the loop are all per base, because
// all four are bound to one index. A shared loop would mean swapping its Store
// between calls, and the first caller that forgot to would search the wrong base
// while still looking like it worked.
type kbRuntime struct {
	base       kb.Base
	dataPath   string
	store      *store.Store
	denseIndex store.DenseIndex
	toolbox    *agent.Toolbox
	loop       *agent.Loop
	// indexing excludes indexing against asking, per base.
	//
	// RWMutex, not Mutex: several questions may run at once (they only read),
	// but none may run while this base is being indexed. The store has its own
	// lock, so this is not about a crash — it is about not answering from a
	// half-updated index, and about the two flows holding a resident model each
	// on a machine where that memory is the scarce resource.
	indexing sync.RWMutex
	// flow is the eino-compiled top-level orchestration (route → simple loop,
	// or decompose → fan-out → synthesize). Nil when it could not be built, in
	// which case `ask` falls back to the single agentic loop.
	flow *agent.Flow
}

// kbFor resolves a knowledge base by id and loads it on first use.
//
// An empty id means the first base. That is what keeps the acceptance scripts
// and anything else written before knowledge bases existed working unchanged —
// and, importantly, it never means "search everything": a caller that forgets
// the id gets one base, so the failure mode is a narrower answer rather than
// another base's documents appearing in it.
func (k *kernel) kbFor(id string) (*kbRuntime, *ipc.Error) {
	base, err := k.kbs.Resolve(id)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: err.Error()}
	}

	// Held across the load. Loading reads an index and may create a collection,
	// and two concurrent loads of the same base would each build a store — with
	// the loser's vectors already pushed to Qdrant by the time it is discarded.
	k.mu.Lock()
	defer k.mu.Unlock()

	if live, ok := k.open[base.ID]; ok {
		return live, nil
	}
	live, err := k.loadBase(base)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}
	k.open[base.ID] = live
	return live, nil
}

// loadBase builds the runtime for one base.
//
// Never fails on a missing or unreadable index: a base that cannot be read is
// reported and started empty, exactly as the single-index kernel did. Failing
// here would make one damaged base take down every other one.
func (k *kernel) loadBase(base kb.Base) (*kbRuntime, error) {
	dataPath := k.kbs.DataPath(base.ID)

	loaded, err := store.Load(dataPath)
	if err != nil {
		log.Printf("warning: could not load the index for %q at %s (%v); starting empty",
			base.Name, dataPath, err)
		loaded = store.New()
	}
	loaded.SetLogger(log.Default())

	live := &kbRuntime{base: base, dataPath: dataPath, store: loaded}

	// Bound per base, and the result decides whether dense retrieval is on for
	// it: an index built by one model cannot be queried with another's vectors,
	// so the check belongs to the index rather than to the process.
	var bound embed.Embedder
	if bindEmbedder(loaded, k.embedder) {
		bound = k.embedder
	}

	if bound != nil {
		// The embedder knows the width before any vector exists, so a base with
		// no documents yet still gets its collection created up front. Waiting
		// for the first vector instead would create the collection inside the
		// very request that is writing to it.
		dims := loaded.Dimensions()
		if dims == 0 {
			dims = bound.Dimensions()
		}
		if dims > 0 {
			index, err := dense.FromEnvCollection(dims, k.kbs.Collection(base.ID))
			switch {
			case err != nil:
				log.Printf("warning: no dense index for %q: %v", base.Name, err)
			case index == nil:
				// Qdrant is not configured. Dense ranking runs in-process, which
				// is slower but never wrong.
			default:
				loaded.SetDenseIndex(index)
				live.denseIndex = index
				if err := syncDenseIndex(loaded, index); err != nil {
					log.Printf("warning: the dense index for %q is out of sync: %v", base.Name, err)
				}
			}
		}
	}

	live.toolbox = &agent.Toolbox{
		Store:        loaded,
		Embedder:     bound,
		DefaultLimit: agent.Medium().SnippetsPerQuery,
	}
	live.loop = &agent.Loop{
		Store:   loaded,
		Tools:   live.toolbox,
		Checker: k.checker,
		// The language the user chose, seeded here and re-applied by
		// settings_set to every open base (settings.go).
		Spec: func() agent.Spec {
			spec := agent.Medium()
			spec.AnswerLanguage = k.currentSettings().AnswerLanguage
			return spec
		}(),
		Logger: log.Default(),
		// Kept tied to the window the generator is asked for, so the two cannot
		// drift apart.
		MaxPromptChars: agent.PromptCharBudget(contextTokens()),
	}
	if k.generator != nil {
		live.loop.Model = k.generator
	}
	// Laya chooses the next tool call each round when it is available; the loop
	// otherwise falls back to the model plan and then the deterministic plan.
	if k.decider != nil {
		live.loop.Chooser = agent.LayaToolChooser{Decide: k.decider}
	}
	k.attachProgress(live.loop)

	// The eino flow wraps this loop: it routes the question, and on the complex
	// path splits it and runs several copies of the loop concurrently.
	flow, err := agent.NewFlow(agent.FlowDeps{
		Loop:           live.loop,
		Router:         agent.Router{Decide: k.decider},
		Model:          live.loop.Model,
		MaxPromptChars: agent.PromptCharBudget(contextTokens()),
		OnStep: func(line string) {
			k.progress("agent", map[string]any{"line": line})
		},
		OnAnswerDelta: func(delta string) {
			k.progress("answer", map[string]any{"delta": delta})
		},
	})
	if err != nil {
		log.Printf("warning: could not build the ask flow for %q (%v); ask will use a single agentic pass",
			base.Name, err)
	} else {
		live.flow = flow
	}

	log.Printf("knowledge base %q (%s): %d document(s), %d chunk(s), dense=%s",
		base.Name, base.ID, len(loaded.DocumentRecords()), loaded.Len(), denseBackendName(live.denseIndex))

	// Recorded on open, because this is the first moment the real figures are
	// known. Without it a base that was adopted or edited elsewhere would keep
	// advertising the counts it was created with — and `kb.list` deliberately
	// does not open anything to find out, so it would show 0 documents for a
	// base full of them.
	k.recordCounts(live)

	return live, nil
}

// attachProgress wires a loop's progress to this process's notification channel.
//
// Called for every base's loop rather than once: each base gets its own loop, and
// a loop whose callbacks were never set answers without reporting anything —
// which is invisible from the outside, so it has to be impossible to forget.
func (k *kernel) attachProgress(loop *agent.Loop) {
	// The loop's steps are the only thing that changes during a round, so
	// forwarding them is what turns a frozen spinner into real progress.
	loop.OnStep = func(line string) {
		k.progress("agent", map[string]any{"line": line})
	}
	// The answer gets its own stage rather than a trace line: it IS the answer,
	// so the UI renders these where the answer will appear instead of appending
	// them to the log.
	//
	// Only the final answer arrives here. The loop's per-round drafts — the ones
	// the checker judges — are not forwarded (agent.Loop.OnAnswerDelta), because
	// a answer that may be rejected and replaced has no business in the place the
	// answer goes: forwarding them is what made the answer appear to change its
	// mind mid-run.
	loop.OnAnswerDelta = func(delta string) {
		k.progress("answer", map[string]any{"delta": delta})
	}
}

func denseBackendName(index store.DenseIndex) string {
	if index == nil {
		return "in-process"
	}
	return index.Backend()
}

// recordCounts refreshes a base's cached summary from what its store now holds.
//
// A failure is logged, not returned: the counts are a convenience for the base
// list, and losing them must not fail the indexing run that produced them.
func (k *kernel) recordCounts(live *kbRuntime) {
	docs := len(live.store.DocumentRecords())
	if err := k.kbs.SetCounts(live.base.ID, docs, live.store.Len()); err != nil {
		log.Printf("warning: could not record counts for %q: %v", live.base.Name, err)
		return
	}
	live.base.DocCount = docs
	live.base.ChunkCount = live.store.Len()
}

// ---- knowledge base management, for the desktop UI ----

// baseSummary renders one base for a list.
//
// Deliberately built from the registry alone, so listing bases does not open
// any of them: with several bases, parsing every index to draw a menu would make
// startup cost grow with the number of bases rather than with the one in use.
// `documents` reports the live figures for the base it is asked about.
func baseSummary(base kb.Base) map[string]any {
	return map[string]any{
		"id":          base.ID,
		"name":        base.Name,
		"created_at":  base.CreatedAt,
		"doc_count":   base.DocCount,
		"chunk_count": base.ChunkCount,
	}
}

func (k *kernel) handleKBList(_ context.Context, _ json.RawMessage) (any, *ipc.Error) {
	bases := k.kbs.List()
	out := make([]map[string]any, 0, len(bases))
	for _, base := range bases {
		out = append(out, baseSummary(base))
	}
	return map[string]any{"bases": out, "count": len(out)}, nil
}

func (k *kernel) handleKBCreate(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Name string `json:"name"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	base, err := k.kbs.Create(params.Name)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: err.Error()}
	}
	log.Printf("kb: created %q (%s)", base.Name, base.ID)
	return baseSummary(base), nil
}

func (k *kernel) handleKBRename(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	base, err := k.kbs.Rename(params.ID, params.Name)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: err.Error()}
	}
	// The id is unchanged, so a loaded runtime keeps working; only its display
	// name differs, and that is read from the registry each time.
	k.mu.Lock()
	if live, ok := k.open[base.ID]; ok {
		live.base = base
	}
	k.mu.Unlock()

	log.Printf("kb: renamed %s to %q", base.ID, base.Name)
	return baseSummary(base), nil
}

// handleKBDelete removes a base, its vectors and its index file.
//
// Deliberately not a "soft delete": a knowledge base is a boundary, and leaving
// its vectors in Qdrant would mean the next base to reuse the id inherited them.
func (k *kernel) handleKBDelete(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		ID string `json:"id"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	if params.ID == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.id is required"}
	}

	// Closed before the registry forgets it. A loaded runtime holds an open
	// Qdrant client and a store full of vectors; dropping the collection out
	// from under it would leave this process answering from an index that no
	// longer exists.
	k.mu.Lock()
	live := k.open[params.ID]
	delete(k.open, params.ID)
	k.mu.Unlock()

	removed, err := k.kbs.Remove(params.ID)
	if err != nil {
		// Put it back: the registry refused, so the runtime is still valid and
		// discarding it would only cost a reload.
		if live != nil {
			k.mu.Lock()
			k.open[params.ID] = live
			k.mu.Unlock()
		}
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: err.Error()}
	}

	reply := map[string]any{"id": removed.ID, "name": removed.Name}

	// The vectors are dropped through a client built for this base's collection
	// rather than through `live`. A base is usually NOT open when it is deleted
	// — deleting something you never opened this session is the normal case —
	// and gating the cleanup on `live != nil` leaked the collection every time.
	if err := dense.DropCollection(k.kbs.Collection(removed.ID)); err != nil {
		log.Printf("warning: could not drop the vectors for %q: %v", removed.Name, err)
		reply["vectors_error"] = err.Error()
	}
	if err := os.RemoveAll(k.kbs.Dir(removed.ID)); err != nil {
		log.Printf("warning: could not delete the index directory for %q: %v", removed.Name, err)
		reply["files_error"] = err.Error()
	}

	log.Printf("kb: deleted %q (%s)", removed.Name, removed.ID)
	return reply, nil
}
