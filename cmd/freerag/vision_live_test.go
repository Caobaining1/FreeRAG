package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"freerag/internal/agent"
	"freerag/internal/parser"
)

// TestLiveFigureCaptioning runs the whole stage against the real sidecar and a
// real Ollama: parse a document, crop each figure, describe it, and print what
// landed in the chunk.
//
// Skipped by default — it needs the parse sidecar, a pulled vision model, and a
// minute or two. The unit tests above cover the concurrency and failure
// behaviour; this one exists to prove the pieces actually fit.
//
//	FREERAG_VISION_LIVE=1 \
//	FREERAG_VISION_LIVE_PDF=/Users/cbn/workspace/papers_zhaohui/2403.03558.pdf \
//	go test ./cmd/freerag/ -run LiveFigureCaptioning -v
func TestLiveFigureCaptioning(t *testing.T) {
	if os.Getenv("FREERAG_VISION_LIVE") == "" {
		t.Skip("set FREERAG_VISION_LIVE=1 to run against the sidecar and Ollama")
	}
	source := os.Getenv("FREERAG_VISION_LIVE_PDF")
	if source == "" {
		t.Fatal("FREERAG_VISION_LIVE_PDF is required")
	}

	settings, enabled := visionConfig()
	if !enabled {
		t.Fatal("the caption stage is disabled by FREERAG_VLM")
	}
	t.Logf("model=%s workers=%d max_tokens=%d max_side=%d",
		settings.Model, settings.Workers, settings.MaxTokens, settings.MaxSide)

	cfg, err := parser.Discover()
	if err != nil {
		t.Fatalf("parse sidecar: %v", err)
	}
	service := parser.NewService(cfg)
	defer func() { _ = service.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	started := time.Now()
	result, err := service.Parse(ctx, source, parser.Options{})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	t.Logf("parsed in %.1fs: %d chunks", time.Since(started).Seconds(), len(result.Chunks))

	k := &kernel{renderer: service, captioner: &agent.OllamaModel{}}

	figures := 0
	for _, chunk := range result.Chunks {
		if isFigureChunk(chunk) {
			figures++
		}
	}
	if figures == 0 {
		t.Skip("this document has no figure blocks")
	}

	started = time.Now()
	described := k.captionFigures(ctx, source, result, settings)
	elapsed := time.Since(started)
	t.Logf("%d of %d figures described in %.1fs (%.1fs each)",
		described, figures, elapsed.Seconds(), elapsed.Seconds()/float64(figures))

	if described == 0 {
		t.Fatal("no figure was described; is the vision model pulled?")
	}

	for _, chunk := range result.Chunks {
		if !isFigureChunk(chunk) || !strings.Contains(chunk.Text, figureMarker) {
			continue
		}
		body := chunk.Text
		if index := strings.Index(body, figureMarker); index >= 0 {
			body = body[index:]
		}
		t.Logf("%s p%d\n%s", chunk.ChunkID, int(chunk.Metadata["page_num"].(float64)), body)

		// The description that gets appended must be fully stripped already.
		// (Identifiers like "F1" or "gpt-4-0613" survive on purpose, so the
		// check is that stripping again changes nothing, not that no digit is
		// present.)
		if stripped := stripNumbers(body); stripped != body {
			t.Errorf("%s: values survived into the indexed text:\n%s", chunk.ChunkID, stripped)
		}
	}
}
