package main

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"freerag/internal/ipc"
	"freerag/internal/parser"
)

// Batch indexing: many documents, one call, bounded concurrency, cancellable.
//
// Why this exists at all: `Serve` handles requests one at a time
// (internal/ipc/jsonrpc.go), so a synchronous `index` loop cannot be cancelled —
// nothing can be read from the stream until the call returns. So the batch form
// starts a job, returns at once, and reports through `progress` notifications,
// which leaves the kernel free to answer `index_cancel` while documents are
// still being worked on.
//
// The concurrency here is deliberately bounded and asymmetric, because the two
// stages of an index scale in OPPOSITE directions. See indexBudget.

// indexBudget is the per-job concurrency allowance.
//
// Two numbers rather than one, because the stages disagree:
//
//   - Parse (layout detection + text extraction) is memory-bandwidth bound. On
//     a MacBook Air M5, 1 / 2 / 4 documents parsed concurrently took 6.0 / 10.5 /
//     11.8 s per document: MORE concurrency is SLOWER, and a second thread
//     inside one model costs 2x (1/2/3/4 threads = 1.9/3.7/5.4/7.4 s per page).
//     So Parse=1 is not a cautious default, it is the fast one.
//   - Embed + upsert is network I/O — SiliconFlow over HTTPS, then Qdrant. A
//     socket wait saturates nothing, so several can be in flight at once.
//
// This is RAGFlow's arrangement (an explicit inference budget in front of the
// model, pools around everything else), with numbers measured on this machine
// rather than inherited. It is a struct so a future machine can raise it in one
// place instead of rediscovering it in the code.
type indexBudget struct {
	// Parse is how many documents may be in layout detection at once.
	Parse int
	// Write is how many documents may be in embed+upsert at once, and therefore
	// how many documents a batch works on at a time.
	Write int
}

func defaultIndexBudget() indexBudget {
	return indexBudget{Parse: 1, Write: 4}
}

// parseGate bounds concurrent parses. Capacity comes from indexBudget.Parse.
type parseGate chan struct{}

// hold takes a slot and returns the function that gives it back.
func (g parseGate) hold() func() {
	g <- struct{}{}
	var once sync.Once
	return func() { once.Do(func() { <-g }) }
}

// indexOutcome is one document's result inside a batch.
//
// It carries the error rather than aborting the batch: one unreadable file in a
// hundred must not discard the other ninety-nine, and the UI wants to list which
// one failed, not just that something did.
type indexOutcome struct {
	File    string `json:"file"`
	Chunks  int    `json:"chunks"`
	Added   int    `json:"added"`
	Removed int    `json:"removed"`
	Skipped bool   `json:"skipped"`
	Note    string `json:"note,omitempty"`
	Error   string `json:"error,omitempty"`
}

// runBounded applies work to every path, with Write workers in flight and at
// most Parse of them inside a parse.
//
// The gate is handed to the work function instead of being held here, so it can
// be released the moment the parse returns: holding it for the whole document
// would serialise the embed and upsert too, which is the exact overlap this
// exists to get.
//
// Outcomes keep the order of paths — a summary is read by a human, and results
// that come back shuffled are a bug report waiting to happen.
func runBounded(
	ctx context.Context,
	paths []string,
	budget indexBudget,
	work func(ctx context.Context, index int, gate parseGate) indexOutcome,
) []indexOutcome {
	outcomes := make([]indexOutcome, len(paths))

	workers := budget.Write
	if workers < 1 {
		workers = 1
	}
	slots := budget.Parse
	if slots < 1 {
		slots = 1
	}
	gate := make(parseGate, slots)

	next := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				outcomes[i] = work(ctx, i, gate)
			}
		}()
	}

	// Feeding stops on cancellation; the entries never handed out are filled in
	// below rather than by a worker, so nothing is written concurrently.
	fed := 0
	for ; fed < len(paths); fed++ {
		if ctx.Err() != nil {
			break
		}
		next <- fed
	}
	close(next)
	wg.Wait()

	for i := fed; i < len(paths); i++ {
		outcomes[i] = indexOutcome{File: filepath.Base(paths[i]), Error: "cancelled"}
	}
	return outcomes
}

// indexBatch indexes every path, reporting progress as it goes.
//
// Progress is what keeps the caller's request alive: the UI's RPC timeout means
// "no progress" rather than "slow" (desktop/main.js), so a batch that went quiet
// would be abandoned by the client even while it kept working.
func (k *kernel) indexBatch(ctx context.Context, live *kbRuntime, paths []string, params parseParams) []indexOutcome {
	budget := defaultIndexBudget()
	var done atomic.Int64
	total := len(paths)

	outcomes := runBounded(ctx, paths, budget, func(ctx context.Context, index int, gate parseGate) indexOutcome {
		path := paths[index]
		file := filepath.Base(path)

		one := params
		one.Path = path

		reply, err := k.indexOne(ctx, live, one, gate)
		outcome := indexOutcome{File: file}
		if err != nil {
			outcome.Error = err.Message
		} else {
			outcome.Chunks = intAt(reply, "chunk_count")
			outcome.Added = intAt(reply, "added")
			outcome.Removed = intAt(reply, "removed")
			outcome.Skipped, _ = reply["skipped"].(bool)
			outcome.Note, _ = reply["note"].(string)
		}

		// Counted after the work, so "3 of 60" means three documents are done
		// rather than three have started.
		n := done.Add(1)
		fields := map[string]any{"file": file, "done": n, "total": total}
		if outcome.Error != "" {
			fields["error"] = outcome.Error
		}
		k.progress("index-batch", fields)
		return outcome
	})

	return outcomes
}

// intAt reads a numeric reply field, which JSON round-tripping makes awkward:
// the index reply is assembled as map[string]any and some entries are ints and
// others float64 by the time a caller sees them.
func intAt(reply map[string]any, key string) int {
	switch value := reply[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	}
	return 0
}

// handleIndexBatch starts a batch and returns immediately with its id.
//
// Returning at once is the whole point: the request stream stays readable, so
// `index_cancel` can be answered while the job runs. The results arrive as a
// notification when it finishes.
func (k *kernel) handleIndexBatch(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Paths    []string `json:"paths"`
		Profile  string   `json:"profile"`
		MaxChars int      `json:"max_chars"`
		MaxPages int      `json:"max_pages"`
		Force    bool     `json:"force"`
		KB       string   `json:"kb"`
	}
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
	}
	if len(params.Paths) == 0 {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.paths must not be empty"}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}

	base := parseParams{
		Profile:  params.Profile,
		MaxChars: params.MaxChars,
		MaxPages: params.MaxPages,
		Force:    params.Force,
		KB:       params.KB,
	}

	jobCtx, cancel := context.WithCancel(context.Background())

	k.indexJobsMu.Lock()
	if k.indexJobs == nil {
		k.indexJobs = map[string]context.CancelFunc{}
	}
	k.nextJobID++
	id := fmt.Sprintf("index-%d", k.nextJobID)
	k.indexJobs[id] = cancel
	k.indexJobsMu.Unlock()

	go func() {
		defer cancel()

		// Held for the whole job, not per document: "creating a knowledge base"
		// is one action, and a question answered in the middle of it would read
		// a pool that is still growing. It is the slowest thing the kernel does
		// (minutes for a batch), so an ask that arrives during it waits — which
		// is the point: the two flows each want a resident model, and this
		// machine cannot afford both.
		live.indexing.Lock()
		defer live.indexing.Unlock()

		outcomes := k.indexBatch(jobCtx, live, params.Paths, base)
		// The job is the unit that owns the vision model: loaded for the first
		// document that has a figure, released once the whole batch is done.
		k.releaseVlm()

		// Removed BEFORE the completion event, so the event means "finished and
		// tidied up". The other order leaves a window in which a caller that has
		// seen the event still finds the job in the map, and a cancel arriving in
		// that window reports success for work that is already over.
		k.indexJobsMu.Lock()
		delete(k.indexJobs, id)
		k.indexJobsMu.Unlock()

		k.progress("index-batch-done", map[string]any{
			"job": id, "total": len(params.Paths), "outcomes": outcomes,
		})
	}()

	return map[string]any{"job": id, "total": len(params.Paths), "started_at": time.Now()}, nil
}

// handleIndexCancel stops a running batch. Documents already indexed stay
// indexed; what is cancelled is the work not yet done.
func (k *kernel) handleIndexCancel(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Job string `json:"job"`
	}
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
	}
	if params.Job == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.job is required"}
	}

	k.indexJobsMu.Lock()
	cancel, ok := k.indexJobs[params.Job]
	k.indexJobsMu.Unlock()
	if !ok {
		return map[string]any{"job": params.Job, "cancelled": false, "reason": "not running"}, nil
	}
	cancel()
	return map[string]any{"job": params.Job, "cancelled": true}, nil
}

// parserOptions is the parse request a batch document runs through.
func parserOptions(params parseParams) parser.Options {
	return parser.Options{
		Profile:  params.Profile,
		MaxChars: params.MaxChars,
		MaxPages: params.MaxPages,
		VlmModel: params.VlmModel,
	}
}

// parseDocument runs the sidecar parse, holding the batch's parse gate when
// there is one. The gate is released as soon as this returns, so the embed and
// upsert that follow overlap with the next document's parse rather than queueing
// behind it — which is the entire reason the gate is not held for the whole
// document.
func (k *kernel) parseDocument(ctx context.Context, params parseParams, gate parseGate) (*parser.Result, error) {
	if gate != nil {
		release := gate.hold()
		defer release()
	}
	return k.parse.Parse(ctx, params.Path, parserOptions(params))
}
