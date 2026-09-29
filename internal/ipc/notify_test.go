package ipc

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func TestNotifyWritesAMessageWithoutAnID(t *testing.T) {
	srv := NewServer()
	srv.Register("work", func(_ context.Context, _ json.RawMessage) (any, *Error) {
		if err := srv.Notify("progress", map[string]any{"stage": "half"}); err != nil {
			t.Errorf("Notify: %v", err)
		}
		return map[string]any{"done": true}, nil
	})

	var out bytes.Buffer
	if err := srv.Serve(context.Background(), strings.NewReader(
		`{"jsonrpc":"2.0","id":1,"method":"work"}`+"\n"), &out); err != nil {
		t.Fatalf("Serve: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("wrote %d lines, want 2 (notification then response): %q", len(lines), out.String())
	}

	var note struct {
		ID     json.RawMessage `json:"id"`
		Method string          `json:"method"`
		Params map[string]any  `json:"params"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &note); err != nil {
		t.Fatalf("decode notification: %v", err)
	}
	// The absent id is the contract: a peer must read this as an event, not as
	// a response it cannot match to a request.
	if note.ID != nil {
		t.Fatalf("notification carries an id: %s", lines[0])
	}
	if note.Method != "progress" || note.Params["stage"] != "half" {
		t.Fatalf("notification = %s", lines[0])
	}

	var resp Response
	if err := json.Unmarshal([]byte(lines[1]), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Error != nil {
		t.Fatalf("response error: %v", resp.Error)
	}
}

func TestNotifyOutsideServeDoesNotPanic(t *testing.T) {
	// A handler reused by a unit test has no transport; that must be an error
	// to ignore, not a crash.
	if err := NewServer().Notify("progress", nil); err != ErrNoTransport {
		t.Fatalf("err = %v, want ErrNoTransport", err)
	}
}

func TestConcurrentNotifyDoesNotCorruptTheStream(t *testing.T) {
	srv := NewServer()
	const writers = 8
	srv.Register("fan", func(_ context.Context, _ json.RawMessage) (any, *Error) {
		var wg sync.WaitGroup
		for i := 0; i < writers; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				_ = srv.Notify("progress", map[string]any{"i": i})
			}(i)
		}
		wg.Wait()
		return map[string]any{"done": true}, nil
	})

	var out bytes.Buffer
	if err := srv.Serve(context.Background(), strings.NewReader(
		`{"jsonrpc":"2.0","id":1,"method":"fan"}`+"\n"), &out); err != nil {
		t.Fatalf("Serve: %v", err)
	}

	// Every line must be complete JSON. Without a mutex around the encoder two
	// writers interleave halves and no line parses.
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != writers+1 {
		t.Fatalf("wrote %d lines, want %d", len(lines), writers+1)
	}
	for _, line := range lines {
		var probe map[string]any
		if err := json.Unmarshal([]byte(line), &probe); err != nil {
			t.Fatalf("corrupt line %q: %v", line, err)
		}
	}
}
