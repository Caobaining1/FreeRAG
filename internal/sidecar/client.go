// Package sidecar runs a JSON-RPC 2.0 / NDJSON child process and calls its
// methods — the Go half of the kernel <-> sidecar link (docs/plan.md §3).
//
// The same client serves the parse sidecar and, later, the inference sidecar:
// both are independent processes spoken to over stdio, which keeps a crash in
// either from taking the kernel down.
package sidecar

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"
)

// Error is a JSON-RPC error returned by the peer.
type Error struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// Error implements the error interface.
func (e *Error) Error() string {
	return fmt.Sprintf("sidecar error %d: %s", e.Code, e.Message)
}

// defaultTimeout bounds one call when the caller passes none.
const defaultTimeout = 5 * time.Minute

// maxLineBytes caps one response line; parsed chunk payloads can be large.
const maxLineBytes = 32 * 1024 * 1024

type request struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type response struct {
	ID     *int64          `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  *Error          `json:"error"`
}

// Client is a running sidecar process spoken to over stdio.
type Client struct {
	name string
	cmd  *exec.Cmd
	in   io.WriteCloser

	mu       sync.Mutex
	nextID   int64
	pending  map[int64]chan response
	closed   bool
	closeErr error

	done chan struct{}
}

// Options configures Start.
type Options struct {
	// OnStderr receives each stderr line; nil discards it.
	OnStderr func(line string)
	// Dir is the child's working directory; empty inherits the parent's.
	Dir string
	// Env is the child's environment; nil inherits the parent's.
	Env []string
}

// Start launches the sidecar and begins reading its responses.
func Start(name string, args []string, opts Options) (*Client, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = opts.Dir
	if opts.Env != nil {
		cmd.Env = opts.Env
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("sidecar %s: stdin pipe: %w", name, err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("sidecar %s: stdout pipe: %w", name, err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("sidecar %s: stderr pipe: %w", name, err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("sidecar %s: start: %w", name, err)
	}

	c := &Client{
		name:    name,
		cmd:     cmd,
		in:      stdin,
		pending: make(map[int64]chan response),
		done:    make(chan struct{}),
	}

	go c.readLoop(stdout)
	if opts.OnStderr != nil {
		go c.readStderr(stderr, opts.OnStderr)
	} else {
		go drain(stderr)
	}
	go c.waitLoop()

	return c, nil
}

func drain(r io.Reader) {
	_, _ = io.Copy(io.Discard, r)
}

func (c *Client) readStderr(r io.Reader, onLine func(string)) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 8*1024), 1*1024*1024)
	for sc.Scan() {
		onLine(sc.Text())
	}
}

// readLoop decodes NDJSON responses and hands each to its waiter.
func (c *Client) readLoop(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), maxLineBytes)

	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var resp response
		if err := json.Unmarshal(line, &resp); err != nil {
			continue // a malformed line must not abort the stream
		}
		if resp.ID == nil {
			continue // notification from the peer
		}
		c.mu.Lock()
		ch, ok := c.pending[*resp.ID]
		if ok {
			delete(c.pending, *resp.ID)
		}
		c.mu.Unlock()
		if ok {
			ch <- resp
		}
	}
}

// waitLoop marks the client closed once the process exits.
func (c *Client) waitLoop() {
	err := c.cmd.Wait()
	c.mu.Lock()
	if !c.closed {
		c.closed = true
		c.closeErr = err
	}
	pending := c.pending
	c.pending = make(map[int64]chan response)
	c.mu.Unlock()

	close(c.done)
	for _, ch := range pending {
		close(ch)
	}
}

// Alive reports whether the sidecar process is still running.
func (c *Client) Alive() bool {
	select {
	case <-c.done:
		return false
	default:
		return true
	}
}

// Call invokes method and decodes its result into out (out may be nil).
func (c *Client) Call(ctx context.Context, method string, params any, out any, timeout time.Duration) error {
	if timeout <= 0 {
		timeout = defaultTimeout
	}

	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return fmt.Errorf("sidecar %s: not running", c.name)
	}
	c.nextID++
	id := c.nextID
	ch := make(chan response, 1)
	c.pending[id] = ch
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	payload, err := json.Marshal(request{JSONRPC: "2.0", ID: id, Method: method, Params: params})
	if err != nil {
		return fmt.Errorf("sidecar %s: marshal %s: %w", c.name, method, err)
	}
	payload = append(payload, '\n')

	if _, err := c.in.Write(payload); err != nil {
		return fmt.Errorf("sidecar %s: write %s: %w", c.name, method, err)
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case resp, ok := <-ch:
		if !ok {
			return fmt.Errorf("sidecar %s: exited before answering %s", c.name, method)
		}
		if resp.Error != nil {
			return resp.Error
		}
		if out == nil {
			return nil
		}
		if err := json.Unmarshal(resp.Result, out); err != nil {
			return fmt.Errorf("sidecar %s: decode %s result: %w", c.name, method, err)
		}
		return nil
	case <-timer.C:
		return fmt.Errorf("sidecar %s: %s timed out after %s", c.name, method, timeout)
	case <-ctx.Done():
		return ctx.Err()
	case <-c.done:
		return fmt.Errorf("sidecar %s: exited before answering %s", c.name, method)
	}
}

// Close terminates the sidecar, waiting briefly for a clean exit.
func (c *Client) Close() error {
	c.mu.Lock()
	if c.closed {
		err := c.closeErr
		c.mu.Unlock()
		return err
	}
	c.mu.Unlock()

	_ = c.in.Close()
	select {
	case <-c.done:
	case <-time.After(2 * time.Second):
		_ = c.cmd.Process.Kill()
		<-c.done
	}
	return c.cmd.Wait()
}
