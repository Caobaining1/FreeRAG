package sidecar

import (
	"context"
	"encoding/json"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// chunkedWriter writes in pieces and yields between them, which is what a pipe
// does with a payload larger than PIPE_BUF — 512 bytes on macOS.
//
// The point is to make the interleaving reproducible: two goroutines writing
// through this at once WILL land in each other's gaps, deterministically rather
// than occasionally, so a test built on it either enforces the lock or fails.
type chunkedWriter struct {
	mu     sync.Mutex
	buffer []byte
}

func (w *chunkedWriter) Write(p []byte) (int, error) {
	const piece = 16
	for i := 0; i < len(p); i += piece {
		end := i + piece
		if end > len(p) {
			end = len(p)
		}
		w.mu.Lock()
		w.buffer = append(w.buffer, p[i:end]...)
		w.mu.Unlock()
		runtime.Gosched()
	}
	return len(p), nil
}

func (w *chunkedWriter) Close() error { return nil }

func (w *chunkedWriter) bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]byte(nil), w.buffer...)
}

// Concurrent requests must not interleave inside the pipe.
//
// A request carrying a state can be tens of kilobytes, and a write that large
// is split by the kernel. Two goroutines whose halves interleave put a
// malformed line in front of the sidecar's parser, which answers with a JSON
// parse error — a failure whose reported cause ("parse error") has nothing to
// do with its actual one (two requests collided). The sub-loops of the complex
// path issue their decisions concurrently, so this is the normal case now.
func TestConcurrentCallsWriteWholeLines(t *testing.T) {
	recorder := &chunkedWriter{}
	client := &Client{name: "test", in: recorder, pending: map[int64]chan response{}}

	const callers = 8
	// A timeout that expires long before any response: the writes have happened
	// by then, and this test is about the bytes rather than the answers.
	payload := map[string]any{"state": strings.Repeat("x", 2000)}

	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var out any
			_ = client.Call(context.Background(), "decide", payload, &out, 20*time.Millisecond)
		}()
	}
	wg.Wait()

	body := strings.TrimRight(string(recorder.bytes()), "\n")
	if body == "" {
		t.Fatal("nothing was written")
	}
	lines := strings.Split(body, "\n")
	if len(lines) != callers {
		t.Fatalf("%d request(s) produced %d line(s): a write was split across a line break, "+
			"which is what interleaving looks like", callers, len(lines))
	}

	ids := map[float64]bool{}
	for i, line := range lines {
		var decoded struct {
			ID     *float64 `json:"id"`
			Method string   `json:"method"`
		}
		if err := json.Unmarshal([]byte(line), &decoded); err != nil {
			t.Fatalf("line %d is not a whole request (%v): %.80q", i, err, line)
		}
		if decoded.ID == nil || decoded.Method != "decide" {
			t.Fatalf("line %d lost its id or method: %.80q", i, line)
		}
		if ids[*decoded.ID] {
			t.Fatalf("id %v was used twice", *decoded.ID)
		}
		ids[*decoded.ID] = true
	}
}
