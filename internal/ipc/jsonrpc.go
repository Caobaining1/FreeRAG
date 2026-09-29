// Package ipc implements the JSON-RPC 2.0 over stdio transport shared by the
// Electron shell, the Go kernel and the local sidecars (parse / inference).
//
// Framing is NDJSON: one complete JSON-RPC message per line. stdout carries the
// protocol stream only — diagnostics must go to stderr.
package ipc

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
)

// Standard JSON-RPC 2.0 error codes.
const (
	CodeParseError     = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInvalidParams  = -32602
	CodeInternalError  = -32603
)

// Request is a JSON-RPC 2.0 request. A request without an ID is a notification
// and receives no response.
type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// Response is a JSON-RPC 2.0 response. Exactly one of Result or Error is set.
type Response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *Error          `json:"error,omitempty"`
}

// Error is a JSON-RPC 2.0 error object.
type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// Error implements the error interface so handlers can return *Error directly.
func (e *Error) Error() string {
	return fmt.Sprintf("jsonrpc error %d: %s", e.Code, e.Message)
}

// Notification is a server-initiated JSON-RPC 2.0 message.
//
// It carries no ID on purpose: a notification is never answered, which is what
// lets a long-running handler report progress without inventing a request to
// attach it to. Clients must treat an incoming message without an ID as an
// event, not as a response they forgot about.
type Notification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// Handler processes one method call. A nil *Error means success.
type Handler func(ctx context.Context, params json.RawMessage) (any, *Error)

// Server dispatches NDJSON JSON-RPC requests read from a stream. It is safe to
// register methods before Serve; registration during Serve is not expected.
type Server struct {
	mu       sync.RWMutex
	handlers map[string]Handler

	// out is the transport, set for the duration of Serve. Guarded because a
	// handler may notify from a goroutine it started.
	outMu sync.Mutex
	out   *json.Encoder
}

// ErrNoTransport is returned by Notify when the server is not serving.
var ErrNoTransport = fmt.Errorf("ipc: no transport (Notify outside Serve)")

// NewServer returns a server with no methods registered.
func NewServer() *Server {
	return &Server{handlers: make(map[string]Handler)}
}

// Register binds a method name to its handler. Re-registering replaces it.
func (s *Server) Register(method string, h Handler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.handlers[method] = h
}

func (s *Server) lookup(method string) (Handler, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	h, ok := s.handlers[method]
	return h, ok
}

// write emits one value, serialised so concurrent writers cannot interleave
// halves of two lines — which would corrupt the NDJSON stream irrecoverably.
func (s *Server) write(value any) error {
	s.outMu.Lock()
	defer s.outMu.Unlock()
	if s.out == nil {
		return ErrNoTransport
	}
	return s.out.Encode(value)
}

// Notify emits a progress or event message to the peer.
//
// A handler calls this while it runs to report progress on work the caller is
// already waiting on. It returns an error rather than panicking so a handler
// can report to a transport-less server (a unit test, say) without special
// casing; callers that do not care may ignore it.
func (s *Server) Notify(method string, params any) error {
	return s.write(Notification{JSONRPC: "2.0", Method: method, Params: params})
}

// Serve reads requests from r and writes responses and notifications to w until
// r is exhausted or ctx is cancelled. A request without an ID produces no
// response.
func (s *Server) Serve(ctx context.Context, r io.Reader, w io.Writer) error {
	sc := bufio.NewScanner(r)
	// Tool payloads (parsed chunks, search results) can be large; raise the line
	// cap well above bufio.Scanner's 64 KiB default.
	sc.Buffer(make([]byte, 0, 64*1024), 32*1024*1024)

	s.outMu.Lock()
	s.out = json.NewEncoder(w)
	s.outMu.Unlock()
	defer func() {
		s.outMu.Lock()
		s.out = nil
		s.outMu.Unlock()
	}()

	for sc.Scan() {
		if err := ctx.Err(); err != nil {
			return err
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		resp := s.handleLine(ctx, []byte(line))
		if resp == nil {
			continue // notification
		}
		if err := s.write(resp); err != nil {
			return fmt.Errorf("write response: %w", err)
		}
	}
	return sc.Err()
}

// handleLine decodes one line and dispatches it, returning the response to
// write (nil for a notification or a malformed notification).
func (s *Server) handleLine(ctx context.Context, line []byte) *Response {
	var req Request
	if err := json.Unmarshal(line, &req); err != nil {
		return errorResponse(nil, CodeParseError, "parse error: "+err.Error())
	}
	if req.JSONRPC != "" && req.JSONRPC != "2.0" {
		return errorResponse(req.ID, CodeInvalidRequest, `unsupported jsonrpc version: `+req.JSONRPC)
	}
	if strings.TrimSpace(req.Method) == "" {
		return errorResponse(req.ID, CodeInvalidRequest, "method is required")
	}

	handler, ok := s.lookup(req.Method)
	if !ok {
		return errorResponse(req.ID, CodeMethodNotFound, "method not found: "+req.Method)
	}

	result, rpcErr := handler(ctx, req.Params)
	if req.ID == nil {
		return nil // notification: never respond, even on error
	}
	if rpcErr != nil {
		return &Response{JSONRPC: "2.0", ID: req.ID, Error: rpcErr}
	}
	return &Response{JSONRPC: "2.0", ID: req.ID, Result: result}
}

// errorResponse builds an error response, omitting the ID for a malformed
// request that carried none.
func errorResponse(id json.RawMessage, code int, msg string) *Response {
	return &Response{JSONRPC: "2.0", ID: id, Error: &Error{Code: code, Message: msg}}
}
