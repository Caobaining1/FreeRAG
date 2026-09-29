package parser

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"freerag/internal/sidecar"
)

// newTestService discovers the sidecar, skipping the test when the Python
// environment is not set up — the parser cannot run without it.
func newTestService(t *testing.T) *Service {
	t.Helper()
	cfg, err := Discover()
	if err != nil {
		t.Skipf("parse sidecar not discoverable: %v", err)
	}
	if _, err := os.Stat(cfg.Python); err != nil {
		t.Skipf("python interpreter %s not found: %v", cfg.Python, err)
	}
	svc := NewService(cfg)
	t.Cleanup(func() { _ = svc.Close() })
	return svc
}

func TestDiscoverFindsSidecarScript(t *testing.T) {
	cfg, err := Discover()
	if err != nil {
		t.Skipf("discovery failed: %v", err)
	}
	if filepath.Base(cfg.Script) != "parse_server.py" {
		t.Fatalf("script = %s", cfg.Script)
	}
	if _, err := os.Stat(cfg.Script); err != nil {
		t.Fatalf("script does not exist: %v", err)
	}
	if cfg.Python == "" {
		t.Fatal("python interpreter was not resolved")
	}
}

func TestInfoReportsCapabilities(t *testing.T) {
	svc := newTestService(t)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	info, err := svc.Info(ctx)
	if err != nil {
		t.Fatalf("Info: %v", err)
	}
	if info.Name != "freerag-parse-sidecar" {
		t.Fatalf("name = %q", info.Name)
	}
	if info.Version == "" {
		t.Fatal("version is empty")
	}
	if _, ok := info.Capabilities["pymupdf"]; !ok {
		t.Fatalf("capabilities missing pymupdf: %#v", info.Capabilities)
	}
}

func TestParseMissingFileReportsNotFound(t *testing.T) {
	svc := newTestService(t)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	_, err := svc.Parse(ctx, filepath.Join(t.TempDir(), "missing.pdf"), Options{})
	if err == nil {
		t.Fatal("expected an error for a missing file")
	}

	var rpcErr *sidecar.Error
	if !errors.As(err, &rpcErr) {
		t.Fatalf("error is %T (%v), want *sidecar.Error", err, err)
	}
	if rpcErr.Code != -32001 {
		t.Fatalf("code = %d, want -32001 (not found)", rpcErr.Code)
	}
}

func TestParseRequiresPath(t *testing.T) {
	svc := NewService(Config{Python: "python3", Script: "unused.py"})
	// An empty path must be rejected before the sidecar is ever started.
	if _, err := svc.Parse(context.Background(), "", Options{}); err == nil {
		t.Fatal("expected an error for an empty path")
	}
}
