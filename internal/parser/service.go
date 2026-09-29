// Package parser orchestrates the Python parse sidecar (docs/plan.md §5).
//
// The Go kernel never parses PDFs itself: it locates the sidecar, keeps the
// process warm, and forwards calls. Parsing stays in Python where PyMuPDF,
// PP-DocLayout and the TSR model already live (docs/plan.md §5.4).
package parser

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"freerag/internal/sidecar"
)

// Parse timeout for one document. Large PDFs are slow; this is a backstop.
const defaultParseTimeout = 10 * time.Minute

// Decide timeout. A Laya decision is ~100 ms once the session is warm; the
// budget covers the one-off session build on a cold sidecar.
const defaultDecideTimeout = 2 * time.Minute

// Chunk is one parsed chunk as the sidecar returns it.
type Chunk struct {
	ChunkID  string         `json:"chunk_id"`
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata"`
}

// Result is one parse call's outcome.
type Result struct {
	Path        string `json:"path"`
	SourceFile  string `json:"source_file"`
	PageCount   int    `json:"page_count"`
	PagesParsed int    `json:"pages_parsed"`
	BlockCount  int    `json:"block_count"`
	ChunkCount  int    `json:"chunk_count"`
	// LayoutProvider names the detector that actually ran ("pp-doclayout" or
	// "pymupdf"). Without it a caller cannot tell a real layout pass from the
	// fallback, and a silent degradation looks like a quality regression.
	LayoutProvider string  `json:"layout_provider,omitempty"`
	Profile        string  `json:"profile"`
	Chunks         []Chunk `json:"chunks"`
}

// Options tune a parse call; the zero value parses everything with defaults.
type Options struct {
	// Profile selects the chunk split threshold: "zh" | "en" | "mixed".
	Profile string
	// MaxChars overrides the profile's split threshold.
	MaxChars int
	// MaxPages limits how many pages are read (0 = all).
	MaxPages int
	// Timeout bounds the call; 0 uses defaultParseTimeout.
	Timeout time.Duration
}

// Config locates the sidecar process.
type Config struct {
	Python string // interpreter path
	Script string // parse_server.py path
}

// Service owns the sidecar process and forwards parse calls to it.
type Service struct {
	cfg Config

	mu     sync.Mutex
	client *sidecar.Client
}

// NewService returns a service bound to cfg.
func NewService(cfg Config) *Service { return &Service{cfg: cfg} }

// Config reports the resolved configuration.
func (s *Service) Config() Config { return s.cfg }

// Discover resolves the interpreter and script.

// Resolution order keeps dev and packaged layouts working:
//
//  1. FREERAG_PYTHON / FREERAG_PARSE_SIDECAR environment overrides;
//  2. a repository root found by walking up from the executable, then from the
//     working directory;
//  3. `<root>/.venv314/bin/python`, then `<root>/.venv/bin/python`, then the
//     `python3` on PATH.
func Discover() (Config, error) {
	cfg := Config{
		Python: os.Getenv("FREERAG_PYTHON"),
		Script: os.Getenv("FREERAG_PARSE_SIDECAR"),
	}
	if cfg.Script != "" {
		if cfg.Python == "" {
			cfg.Python = "python3"
		}
		return cfg, nil
	}

	root, err := repoRoot()
	if err != nil {
		return cfg, err
	}
	cfg.Script = filepath.Join(root, "sidecar", "parse_server.py")
	if _, err := os.Stat(cfg.Script); err != nil {
		return cfg, fmt.Errorf("parse sidecar not found at %s", cfg.Script)
	}

	if cfg.Python == "" {
		for _, candidate := range []string{
			filepath.Join(root, ".venv314", "bin", "python"),
			filepath.Join(root, ".venv", "bin", "python"),
		} {
			if _, err := os.Stat(candidate); err == nil {
				cfg.Python = candidate
				break
			}
		}
	}
	if cfg.Python == "" {
		cfg.Python = "python3"
	}
	return cfg, nil
}

// repoRoot walks up from the executable, then from the working directory,
// looking for sidecar/parse_server.py.
func repoRoot() (string, error) {
	starts := make([]string, 0, 2)
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}

	for _, start := range starts {
		dir := start
		for range 5 {
			if _, err := os.Stat(filepath.Join(dir, "sidecar", "parse_server.py")); err == nil {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return "", fmt.Errorf("could not locate the freerag repository root (set FREERAG_PARSE_SIDECAR)")
}

// ensureClient returns a live sidecar, restarting it if it has exited.
func (s *Service) ensureClient() (*sidecar.Client, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.client != nil && s.client.Alive() {
		return s.client, nil
	}
	if s.client != nil {
		_ = s.client.Close()
		s.client = nil
	}

	client, err := sidecar.Start(s.cfg.Python, []string{s.cfg.Script}, sidecar.Options{})
	if err != nil {
		return nil, err
	}
	s.client = client
	return client, nil
}

// Info mirrors the sidecar's `version` reply.
type Info struct {
	Name         string         `json:"name"`
	Version      string         `json:"version"`
	Capabilities map[string]any `json:"capabilities"`
}

// Info asks the sidecar what it is and what it can do here.
func (s *Service) Info(ctx context.Context) (*Info, error) {
	client, err := s.ensureClient()
	if err != nil {
		return nil, err
	}
	var info Info
	if err := client.Call(ctx, "version", nil, &info, 30*time.Second); err != nil {
		return nil, err
	}
	return &info, nil
}

// Parse runs the pipeline on one document.
func (s *Service) Parse(ctx context.Context, path string, opts Options) (*Result, error) {
	if path == "" {
		return nil, fmt.Errorf("parse: path is required")
	}
	client, err := s.ensureClient()
	if err != nil {
		return nil, err
	}

	params := map[string]any{"path": path}
	if opts.Profile != "" {
		params["profile"] = opts.Profile
	}
	if opts.MaxChars > 0 {
		params["max_chars"] = opts.MaxChars
	}
	if opts.MaxPages > 0 {
		params["max_pages"] = opts.MaxPages
	}

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = defaultParseTimeout
	}

	var result Result
	if err := client.Call(ctx, "parse", params, &result, timeout); err != nil {
		return nil, err
	}
	return &result, nil
}

// DecisionRequest is one Laya typed-decision request (docs/plan.md §6.6).
type DecisionRequest struct {
	// Instructions is the question put to the model, e.g. "Does the draft
	// answer the question?".
	Instructions string `json:"instructions"`
	// Criteria maps each option to the description that helps the model tell
	// them apart. The returned Choice is rendered from one of these keys.
	Criteria map[string]string `json:"criteria"`
	// State is the material the decision is about.
	State string `json:"state,omitempty"`
	// QType is "choice" (default), "score" or "noul".
	QType string `json:"qtype,omitempty"`
}

// Decision is Laya's answer.
type Decision struct {
	QType string `json:"qtype"`
	// Index and Choice identify the winning option; Choice is the rendered
	// option text, so callers should match on it rather than on Index — the
	// option order depends on how the criteria map was serialised.
	Index       int       `json:"index"`
	Choice      string    `json:"choice"`
	Options     []string  `json:"options"`
	Confidence  float64   `json:"confidence"`
	Probability float64   `json:"probability"`
	Scores      []float64 `json:"scores"`
}

// Decide runs one typed decision through the Laya sidecar.
func (s *Service) Decide(ctx context.Context, req DecisionRequest, timeout time.Duration) (*Decision, error) {
	if req.Instructions == "" {
		return nil, fmt.Errorf("decide: instructions are required")
	}
	if len(req.Criteria) == 0 {
		return nil, fmt.Errorf("decide: criteria are required")
	}

	client, err := s.ensureClient()
	if err != nil {
		return nil, err
	}
	if timeout <= 0 {
		timeout = defaultDecideTimeout
	}

	var decision Decision
	if err := client.Call(ctx, "decide", req, &decision, timeout); err != nil {
		return nil, err
	}
	return &decision, nil
}

// HasLaya reports whether the sidecar can run Laya decisions.
func (s *Service) HasLaya(ctx context.Context) bool {
	info, err := s.Info(ctx)
	if err != nil {
		return false
	}
	available, _ := info.Capabilities["laya"].(bool)
	return available
}

// Close shuts the sidecar down.
func (s *Service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.client == nil {
		return nil
	}
	err := s.client.Close()
	s.client = nil
	return err
}
