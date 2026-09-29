package ipc

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// newTestServer returns a server with an "echo" method that replies with its
// params, plus a "boom" method that fails with a fixed code.
func newTestServer() *Server {
	srv := NewServer()
	srv.Register("echo", func(_ context.Context, params json.RawMessage) (any, *Error) {
		var value any
		if len(params) > 0 {
			_ = json.Unmarshal(params, &value)
		}
		return map[string]any{"echo": value}, nil
	})
	srv.Register("boom", func(_ context.Context, _ json.RawMessage) (any, *Error) {
		return nil, &Error{Code: -32001, Message: "document not found"}
	})
	return srv
}

// serve feeds input to the server and returns the decoded response lines.
func serve(t *testing.T, srv *Server, input string) []Response {
	t.Helper()
	var out bytes.Buffer
	if err := srv.Serve(context.Background(), strings.NewReader(input), &out); err != nil {
		t.Fatalf("Serve returned error: %v", err)
	}

	var responses []Response
	for _, line := range strings.Split(out.String(), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var resp Response
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			t.Fatalf("response line is not JSON (%q): %v", line, err)
		}
		responses = append(responses, resp)
	}
	return responses
}

func TestSuccessfulCall(t *testing.T) {
	responses := serve(t, newTestServer(), `{"jsonrpc":"2.0","id":1,"method":"echo","params":{"a":1}}`+"\n")
	if len(responses) != 1 {
		t.Fatalf("got %d responses, want 1", len(responses))
	}
	resp := responses[0]
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	if string(resp.ID) != "1" {
		t.Fatalf("id = %s, want 1", resp.ID)
	}
	result, ok := resp.Result.(map[string]any)
	if !ok {
		t.Fatalf("result is %T, want object", resp.Result)
	}
	inner, ok := result["echo"].(map[string]any)
	if !ok || inner["a"] != float64(1) {
		t.Fatalf("echo payload = %#v", result["echo"])
	}
}

func TestUnknownMethod(t *testing.T) {
	responses := serve(t, newTestServer(), `{"jsonrpc":"2.0","id":2,"method":"nope"}`+"\n")
	if len(responses) != 1 || responses[0].Error == nil {
		t.Fatalf("want one error response, got %#v", responses)
	}
	if responses[0].Error.Code != CodeMethodNotFound {
		t.Fatalf("code = %d, want %d", responses[0].Error.Code, CodeMethodNotFound)
	}
}

func TestMalformedJSON(t *testing.T) {
	responses := serve(t, newTestServer(), "not json\n")
	if len(responses) != 1 || responses[0].Error == nil {
		t.Fatalf("want one parse-error response, got %#v", responses)
	}
	if responses[0].Error.Code != CodeParseError {
		t.Fatalf("code = %d, want %d", responses[0].Error.Code, CodeParseError)
	}
	if len(responses[0].ID) != 0 {
		t.Fatalf("id should be absent for an unparsable request, got %s", responses[0].ID)
	}
}

func TestNotificationProducesNoResponse(t *testing.T) {
	responses := serve(t, newTestServer(), `{"jsonrpc":"2.0","method":"echo","params":{}}`+"\n")
	if len(responses) != 0 {
		t.Fatalf("notification should produce no response, got %#v", responses)
	}
}

func TestHandlerErrorIsReturned(t *testing.T) {
	responses := serve(t, newTestServer(), `{"jsonrpc":"2.0","id":3,"method":"boom"}`+"\n")
	if len(responses) != 1 || responses[0].Error == nil {
		t.Fatalf("want one error response, got %#v", responses)
	}
	if responses[0].Error.Code != -32001 {
		t.Fatalf("handler code was not preserved: %d", responses[0].Error.Code)
	}
	if !strings.Contains(responses[0].Error.Message, "not found") {
		t.Fatalf("message = %q", responses[0].Error.Message)
	}
}

func TestUnsupportedVersion(t *testing.T) {
	responses := serve(t, newTestServer(), `{"jsonrpc":"1.0","id":4,"method":"echo"}`+"\n")
	if len(responses) != 1 || responses[0].Error == nil || responses[0].Error.Code != CodeInvalidRequest {
		t.Fatalf("want invalid-request error, got %#v", responses)
	}
}

func TestMissingMethod(t *testing.T) {
	responses := serve(t, newTestServer(), `{"jsonrpc":"2.0","id":5}`+"\n")
	if len(responses) != 1 || responses[0].Error == nil || responses[0].Error.Code != CodeInvalidRequest {
		t.Fatalf("want invalid-request error, got %#v", responses)
	}
}

func TestBlankLinesIgnored(t *testing.T) {
	responses := serve(t, newTestServer(), "\n\n"+`{"jsonrpc":"2.0","id":6,"method":"echo"}`+"\n\n")
	if len(responses) != 1 || responses[0].Error != nil {
		t.Fatalf("blank lines must be ignored, got %#v", responses)
	}
}

func TestBatchOnOneStream(t *testing.T) {
	input := `{"jsonrpc":"2.0","id":1,"method":"echo"}` + "\n" +
		`{"jsonrpc":"2.0","id":2,"method":"nope"}` + "\n" +
		`{"jsonrpc":"2.0","id":3,"method":"echo","params":{"x":"y"}}` + "\n"
	responses := serve(t, newTestServer(), input)
	if len(responses) != 3 {
		t.Fatalf("got %d responses, want 3", len(responses))
	}
	if responses[0].Error != nil || responses[1].Error == nil || responses[2].Error != nil {
		t.Fatalf("unexpected error pattern: %#v", responses)
	}
	if string(responses[2].ID) != "3" {
		t.Fatalf("responses must keep their own ids, got %s", responses[2].ID)
	}
}
