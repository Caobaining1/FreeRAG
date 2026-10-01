package main

import (
	"context"
	"encoding/json"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func bumpMax(max *atomic.Int32, value int32) {
	for {
		old := max.Load()
		if value <= old || max.CompareAndSwap(old, value) {
			return
		}
	}
}

func pathsFor(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = string(rune('a'+i%26)) + ".pdf"
	}
	return out
}

func TestRunBoundedKeepsInputOrder(t *testing.T) {
	// A summary is read by a person; results arriving shuffled would be a
	// permanent source of "which file failed?" confusion.
	paths := pathsFor(12)
	seen := make([]int, len(paths))
	outcomes := runBounded(context.Background(), paths, indexBudget{Parse: 1, Write: 4},
		func(_ context.Context, index int, _ parseGate) indexOutcome {
			seen[index]++
			return indexOutcome{File: paths[index], Added: index}
		})

	for i, outcome := range outcomes {
		if outcome.File != paths[i] {
			t.Fatalf("outcome %d is for %q, want %q", i, outcome.File, paths[i])
		}
		if outcome.Added != i {
			t.Fatalf("outcome %d was produced by worker %d", i, outcome.Added)
		}
		if seen[i] != 1 {
			t.Fatalf("path %d was worked on %d times", i, seen[i])
		}
	}
}

// The property the gate exists for: parses never overlap, while the write phase
// that follows them does. If the gate were held for the whole document this test
// would see maxInWrite == 1 and the batch would be as serial as the loop it
// replaced.
func TestRunBoundedSerialisesParsesButOverlapsWrites(t *testing.T) {
	var inGate, maxInGate, inWrite, maxInWrite atomic.Int32

	outcomes := runBounded(context.Background(), pathsFor(8), indexBudget{Parse: 1, Write: 4},
		func(_ context.Context, _ int, gate parseGate) indexOutcome {
			release := gate.hold()
			bumpMax(&maxInGate, inGate.Add(1))
			time.Sleep(5 * time.Millisecond)
			inGate.Add(-1)
			release()

			bumpMax(&maxInWrite, inWrite.Add(1))
			time.Sleep(5 * time.Millisecond)
			inWrite.Add(-1)
			return indexOutcome{}
		})

	if len(outcomes) != 8 {
		t.Fatalf("got %d outcomes, want 8", len(outcomes))
	}
	if got := maxInGate.Load(); got != 1 {
		t.Fatalf("parses overlapped: %d in the gate at once, want 1", got)
	}
	if got := maxInWrite.Load(); got < 2 {
		t.Fatalf("writes did not overlap: at most %d at once, want >= 2 "+
			"(the gate is being held too long)", got)
	}
}

func TestRunBoundedHonoursTheWriteBudget(t *testing.T) {
	var inFlight, maxInFlight atomic.Int32

	runBounded(context.Background(), pathsFor(16), indexBudget{Parse: 2, Write: 3},
		func(_ context.Context, _ int, _ parseGate) indexOutcome {
			bumpMax(&maxInFlight, inFlight.Add(1))
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			return indexOutcome{}
		})

	if got := maxInFlight.Load(); got > 3 {
		t.Fatalf("%d documents in flight at once, want at most 3", got)
	}
	if got := maxInFlight.Load(); got < 2 {
		t.Fatalf("only %d in flight at once; the pool is not running in parallel", got)
	}
}

func TestRunBoundedKeepsFailuresWithTheirDocument(t *testing.T) {
	paths := pathsFor(6)
	outcomes := runBounded(context.Background(), paths, indexBudget{Parse: 1, Write: 2},
		func(_ context.Context, index int, _ parseGate) indexOutcome {
			if index == 2 {
				return indexOutcome{File: paths[index], Error: "unsupported document format"}
			}
			return indexOutcome{File: paths[index], Added: 1}
		})

	// One bad file must not discard the others.
	if outcomes[2].Error == "" {
		t.Fatal("the failing document lost its error")
	}
	for i, outcome := range outcomes {
		if i == 2 {
			continue
		}
		if outcome.Error != "" {
			t.Fatalf("document %d failed unexpectedly: %s", i, outcome.Error)
		}
		if outcome.Added != 1 {
			t.Fatalf("document %d was not indexed", i)
		}
	}
}

func TestRunBoundedStopsEarlyWhenCancelled(t *testing.T) {
	paths := pathsFor(30)
	ctx, cancel := context.WithCancel(context.Background())
	var started atomic.Int32

	outcomes := runBounded(ctx, paths, indexBudget{Parse: 1, Write: 1},
		func(_ context.Context, index int, _ parseGate) indexOutcome {
			if started.Add(1) == 2 {
				cancel() // as a user pressing cancel would, mid-batch
			}
			time.Sleep(2 * time.Millisecond)
			return indexOutcome{File: paths[index], Added: 1}
		})

	// Every path still gets an outcome: the caller lists them, and a missing
	// entry would read as "nothing happened" rather than "stopped".
	if len(outcomes) != len(paths) {
		t.Fatalf("got %d outcomes for %d paths", len(outcomes), len(paths))
	}
	cancelled := 0
	for _, outcome := range outcomes {
		if outcome.Error == "cancelled" {
			cancelled++
		}
	}
	if cancelled == 0 {
		t.Fatal("nothing was marked cancelled after the context was cancelled")
	}
	if started.Load() >= int32(len(paths)) {
		t.Fatalf("all %d documents were started despite cancellation", started.Load())
	}
	for i, outcome := range outcomes {
		if outcome.Error == "" && outcome.Added != 1 {
			t.Fatalf("document %d has neither a result nor a reason", i)
		}
		if outcome.File == "" {
			t.Fatalf("document %d has no file name", i)
		}
	}
}

// The wiring end to end: `index_batch` accepts a job and returns at once, and
// the per-document results arrive as a notification. That split is what makes
// the job cancellable — Serve reads one request at a time, so a batch that
// blocked the stream could never be stopped.
func TestIndexBatchReportsThroughNotifications(t *testing.T) {
	k := newTestKernel(t)

	var mu sync.Mutex
	var events []map[string]any
	k.notify = func(method string, params any) error {
		if method != "progress" {
			return nil
		}
		if fields, ok := params.(map[string]any); ok {
			mu.Lock()
			events = append(events, fields)
			mu.Unlock()
		}
		return nil
	}

	// Files that do not exist: every document must come back with its own error
	// rather than the batch failing as a whole.
	paths := []string{"/nonexistent/one.pdf", "/nonexistent/two.pdf", "/nonexistent/three.pdf"}
	raw, err := json.Marshal(map[string]any{"paths": paths})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	reply, rpcErr := k.handleIndexBatch(context.Background(), raw)
	if rpcErr != nil {
		t.Fatalf("index_batch: %s", rpcErr.Message)
	}
	started, _ := reply.(map[string]any)
	job, _ := started["job"].(string)
	if job == "" {
		t.Fatalf("index_batch returned no job: %v", started)
	}
	if total, _ := started["total"].(int); total != len(paths) {
		t.Fatalf("total = %v, want %d", started["total"], len(paths))
	}

	done := waitForStage(&mu, &events, "index-batch-done", 5*time.Second)
	if done == nil {
		t.Fatalf("no index-batch-done event; saw %v", stagesOf(&mu, &events))
	}

	outcomes, ok := done["outcomes"].([]indexOutcome)
	if !ok {
		t.Fatalf("the done event carries %T, want []indexOutcome", done["outcomes"])
	}
	if len(outcomes) != len(paths) {
		t.Fatalf("%d outcomes for %d paths", len(outcomes), len(paths))
	}
	for i, outcome := range outcomes {
		if outcome.File != filepath.Base(paths[i]) {
			t.Fatalf("outcome %d is for %q", i, outcome.File)
		}
		if outcome.Error == "" {
			t.Fatalf("outcome %d reported no error for a missing file", i)
		}
	}

	// One "n of N" event per document, so the UI can show progress.
	perFile := 0
	for _, event := range eventsOf(&mu, &events, "index-batch") {
		if event["total"] == len(paths) {
			perFile++
		}
	}
	if perFile != len(paths) {
		t.Fatalf("%d per-document progress events for %d documents", perFile, len(paths))
	}

	// The job must not leak: a finished job left in the map would answer a later
	// cancel with "cancelled: true" for work that is long over.
	k.indexJobsMu.Lock()
	remaining := len(k.indexJobs)
	k.indexJobsMu.Unlock()
	if remaining != 0 {
		t.Fatalf("%d job(s) left in the map after finishing", remaining)
	}
}

func TestIndexCancelOfAnUnknownJobDoesNotClaimSuccess(t *testing.T) {
	k := newTestKernel(t)
	raw, err := json.Marshal(map[string]any{"job": "index-999"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	reply, rpcErr := k.handleIndexCancel(context.Background(), raw)
	if rpcErr != nil {
		t.Fatalf("index_cancel: %s", rpcErr.Message)
	}
	fields, _ := reply.(map[string]any)
	if cancelled, _ := fields["cancelled"].(bool); cancelled {
		t.Fatal("cancelling a job that is not running reported success")
	}
}

func TestIndexBatchRejectsAnEmptyList(t *testing.T) {
	k := newTestKernel(t)
	raw, err := json.Marshal(map[string]any{"paths": []string{}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, rpcErr := k.handleIndexBatch(context.Background(), raw); rpcErr == nil {
		t.Fatal("an empty batch was accepted")
	}
}

func eventsOf(mu *sync.Mutex, events *[]map[string]any, stage string) []map[string]any {
	mu.Lock()
	defer mu.Unlock()
	var out []map[string]any
	for _, event := range *events {
		if event["stage"] == stage {
			out = append(out, event)
		}
	}
	return out
}

func stagesOf(mu *sync.Mutex, events *[]map[string]any) []any {
	mu.Lock()
	defer mu.Unlock()
	out := make([]any, 0, len(*events))
	for _, event := range *events {
		out = append(out, event["stage"])
	}
	return out
}

func waitForStage(mu *sync.Mutex, events *[]map[string]any, stage string, timeout time.Duration) map[string]any {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if found := eventsOf(mu, events, stage); len(found) > 0 {
			return found[len(found)-1]
		}
		time.Sleep(5 * time.Millisecond)
	}
	return nil
}

func TestDefaultBudgetSerialisesParsing(t *testing.T) {
	// The numbers are a measurement (see indexBudget), not a preference: parse
	// gets SLOWER with concurrency on this hardware.
	budget := defaultIndexBudget()
	if budget.Parse != 1 {
		t.Fatalf("Parse = %d, want 1 (measured: concurrency makes parsing slower)", budget.Parse)
	}
	if budget.Write < 2 {
		t.Fatalf("Write = %d, want > 1 so embed/upsert can overlap", budget.Write)
	}
}
