package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"freerag/internal/parser"
)

// A 1x1 PNG. Only its bytes matter: the stage never decodes the image, it hashes
// it for the cache key and forwards it to the model.
const tinyPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="

type fakeRenderer struct {
	calls  atomic.Int32
	byPage map[int]string
	err    error
}

func (f *fakeRenderer) Render(_ context.Context, req parser.RenderRequest, _ time.Duration) (*parser.Page, error) {
	f.calls.Add(1)
	if f.err != nil {
		return nil, f.err
	}
	image := tinyPNG
	if f.byPage != nil {
		if found, ok := f.byPage[req.Page]; ok {
			image = found
		}
	}
	return &parser.Page{Page: req.Page, Image: image, WidthPT: 100, HeightPT: 50}, nil
}

type fakeCaptioner struct {
	mu       sync.Mutex
	seen     []string
	inFlight atomic.Int32
	maxSeen  atomic.Int32
	fail     map[string]bool
	answer   string
	delay    time.Duration
}

func (f *fakeCaptioner) CaptionImage(
	_ context.Context, _model, _prompt, imageB64 string, _ int,
) (string, error) {
	bumpMax(&f.maxSeen, f.inFlight.Add(1))
	defer f.inFlight.Add(-1)
	if f.delay > 0 {
		time.Sleep(f.delay)
	}

	f.mu.Lock()
	f.seen = append(f.seen, imageB64)
	f.mu.Unlock()

	if f.fail[imageB64] {
		return "", fmt.Errorf("vision model refused")
	}
	if f.answer != "" {
		return f.answer, nil
	}
	return "折线图，显示指令形式优于直接输入。", nil
}

func figureChunk(id string, page int, text string) parser.Chunk {
	return parser.Chunk{
		ChunkID: id,
		Text:    text,
		Metadata: map[string]any{
			"block_type": "FigureWithCaption",
			"page_num":   float64(page),
			"bbox":       []any{10.0, 20.0, 210.0, 120.0},
		},
	}
}

func newVisionKernel(t *testing.T, captioner captioner, renderer figureRenderer) *kernel {
	t.Helper()
	t.Setenv("FREERAG_VISION_CACHE", t.TempDir())
	return &kernel{captioner: captioner, renderer: renderer}
}

func settingsFor(workers int) visionSettings {
	return visionSettings{Model: "qwen2.5vl:3b", MaxTokens: 155, MaxSide: 768, Workers: workers}
}

func TestStripNumbersKeepsIdentifiersAndDropsValues(t *testing.T) {
	// The model was asked not to quote numbers and quoted them anyway
	// ("人类的F1得分最高，为93.16%"), so the rule is enforced here rather than
	// trusted to the prompt. Identifiers stay: dropping "gpt-4-0613" would undo
	// the reason the extracted text is kept alongside the description.
	dropped := "柱状图，人类得分最高，为 93.16%，davinci 最低，为 46.37%。"
	cleaned := stripNumbers(dropped)
	for _, banned := range []string{"93.16", "46.37"} {
		if strings.Contains(cleaned, banned) {
			t.Fatalf("stripNumbers left %q in %q", banned, cleaned)
		}
	}
	for _, kept := range []string{"柱状图", "davinci", "最低"} {
		if !strings.Contains(cleaned, kept) {
			t.Fatalf("stripNumbers dropped %q from %q", kept, cleaned)
		}
	}

	for _, identifier := range []string{
		"折线图显示 gpt-4-0613 与 LLaMA-65b 的对比。",
		"表格对比 gpt-3.5-turbo-0613 与 claude-2。",
		"F1 分数随模型版本提升。",
	} {
		if got := stripNumbers(identifier); got != identifier {
			t.Fatalf("stripNumbers mangled an identifier sentence:\n got %q\nwant %q", got, identifier)
		}
	}

	// The vertical κ and H values are exactly the ones a vision model misreads,
	// so they go even though they sit in a formula.
	if got := stripNumbers("侧栏给出 κ=0、H=0.52，说明一致性低。"); strings.ContainsAny(got, "0123456789") {
		t.Fatalf("stripNumbers left a formula value: %q", got)
	}

	// A comma-grouped value is two runs; both are values.
	if got := stripNumbers("提升 1,234 个百分点。"); strings.ContainsAny(got, "0123456789") {
		t.Fatalf("stripNumbers left a comma-grouped value: %q", got)
	}
}

func TestCaptionFiguresOnlyTouchesFiguresAndKeepsExtractedText(t *testing.T) {
	renderer := &fakeRenderer{}
	captioner := &fakeCaptioner{}
	k := newVisionKernel(t, captioner, renderer)

	result := &parser.Result{Chunks: []parser.Chunk{
		{ChunkID: "c0", Text: "正文段落", Metadata: map[string]any{"block_type": "Text"}},
		figureChunk("c1", 3, "Figure 1: 总体框架"),
		{ChunkID: "c2", Text: "表格", Metadata: map[string]any{"block_type": "TableWithCaption"}},
	}}

	if got := k.captionFigures(context.Background(), "/tmp/paper.pdf", result, settingsFor(4)); got != 1 {
		t.Fatalf("described %d figures, want 1", got)
	}

	if result.Chunks[0].Text != "正文段落" {
		t.Fatal("a text chunk was modified")
	}
	if result.Chunks[2].Text != "表格" {
		t.Fatal("a table chunk was modified")
	}
	if renderer.calls.Load() != 1 {
		t.Fatalf("rendered %d crops, want 1 (only figures)", renderer.calls.Load())
	}

	figure := result.Chunks[1]
	// Appended, not substituted: exact strings stay searchable.
	if !strings.Contains(figure.Text, "Figure 1: 总体框架") {
		t.Fatalf("the extracted text was lost: %q", figure.Text)
	}
	if !strings.Contains(figure.Text, figureMarker) {
		t.Fatalf("no marker in %q", figure.Text)
	}
	if figure.Metadata["figure_summary_model"] != "qwen2.5vl:3b" {
		t.Fatal("the description is not attributed to a model")
	}
}

func TestCaptionFiguresBoundsConcurrency(t *testing.T) {
	// Process-wide bound: a batch indexes several documents at once, so a
	// per-document limit would multiply out on the vision model.
	k := newVisionKernel(t, &fakeCaptioner{delay: 30 * time.Millisecond}, &fakeRenderer{})

	chunks := make([]parser.Chunk, 0, 12)
	for i := 0; i < 12; i++ {
		chunks = append(chunks, figureChunk(fmt.Sprintf("c%d", i), i+1, "Figure: caption"))
	}

	if got := k.captionFigures(context.Background(), "/tmp/p.pdf",
		&parser.Result{Chunks: chunks}, settingsFor(3)); got != 12 {
		t.Fatalf("described %d, want 12", got)
	}

	captioner := k.captioner.(*fakeCaptioner)
	if captioner.maxSeen.Load() > 3 {
		t.Fatalf("%d descriptions ran at once, want at most 3", captioner.maxSeen.Load())
	}
	if captioner.maxSeen.Load() < 2 {
		t.Fatalf("only %d ran at once; the stage is not running in parallel", captioner.maxSeen.Load())
	}
}

func TestCaptionFiguresSurvivesAFailingModel(t *testing.T) {
	// One bad figure must not cost the document: it keeps its extracted text
	// and the rest are still described.
	failing := base64.StdEncoding.EncodeToString([]byte("second-image"))
	renderer := &fakeRenderer{byPage: map[int]string{2: failing}}
	captioner := &fakeCaptioner{fail: map[string]bool{failing: true}}
	k := newVisionKernel(t, captioner, renderer)

	result := &parser.Result{Chunks: []parser.Chunk{
		figureChunk("c0", 1, "Figure 1: 第一张"),
		figureChunk("c1", 2, "Figure 2: 第二张"),
		figureChunk("c2", 3, "Figure 3: 第三张"),
	}}

	if got := k.captionFigures(context.Background(), "/tmp/p.pdf", result, settingsFor(2)); got != 2 {
		t.Fatalf("described %d, want 2", got)
	}
	if strings.Contains(result.Chunks[1].Text, figureMarker) {
		t.Fatal("a failed figure got a marker anyway")
	}
	if result.Chunks[1].Text != "Figure 2: 第二张" {
		t.Fatalf("the failed figure lost its text: %q", result.Chunks[1].Text)
	}
}

func TestCaptionFiguresSkipsWhenTheModelAnswersNothingUseful(t *testing.T) {
	// Every token was a number, so stripping leaves nothing. An empty marker
	// would be worse than no description at all.
	k := newVisionKernel(t, &fakeCaptioner{answer: "93.16 46.37 0.52"}, &fakeRenderer{})
	result := &parser.Result{Chunks: []parser.Chunk{figureChunk("c0", 1, "Figure 1: x")}}

	if got := k.captionFigures(context.Background(), "/tmp/p.pdf", result, settingsFor(2)); got != 0 {
		t.Fatalf("described %d, want 0", got)
	}
	if strings.Contains(result.Chunks[0].Text, figureMarker) {
		t.Fatalf("an empty description was appended: %q", result.Chunks[0].Text)
	}
}

func TestCaptionFiguresIsANoOpWithoutAModelOrRenderer(t *testing.T) {
	result := &parser.Result{Chunks: []parser.Chunk{figureChunk("c0", 1, "Figure 1: x")}}

	for name, k := range map[string]*kernel{
		"no captioner": {renderer: &fakeRenderer{}},
		"no renderer":  {captioner: &fakeCaptioner{}},
	} {
		if got := k.captionFigures(context.Background(), "/tmp/p.pdf", result, settingsFor(2)); got != 0 {
			t.Fatalf("%s: described %d figures", name, got)
		}
		if result.Chunks[0].Text != "Figure 1: x" {
			t.Fatalf("%s: the chunk was modified", name)
		}
	}
}

func TestCaptionFiguresAbortsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	failing := &fakeCaptioner{}
	k := newVisionKernel(t, failing, &fakeRenderer{})
	result := &parser.Result{Chunks: []parser.Chunk{figureChunk("c0", 1, "Figure 1: x")}}

	if got := k.captionFigures(ctx, "/tmp/p.pdf", result, settingsFor(2)); got != 0 {
		t.Fatalf("described %d figures after cancellation", got)
	}
	if len(failing.seen) != 0 {
		t.Fatal("the model was called after cancellation")
	}
}

func TestVisionSettingsDefaultToTheMeasuredConfiguration(t *testing.T) {
	for _, key := range []string{
		"FREERAG_VLM", "FREERAG_VLM_MODEL", "FREERAG_VLM_MAX_TOKENS",
		"FREERAG_VLM_MAX_SIDE", "FREERAG_VLM_CONCURRENCY",
	} {
		t.Setenv(key, "")
	}
	// Off by default: sidecar/vlm.py does the captioning, at parse time, and two
	// enabled paths would describe every figure twice.
	if _, enabled := visionConfig(); enabled {
		t.Fatal("the Go-side stage is on by default; the sidecar one is the default")
	}
	t.Setenv("FREERAG_VLM", "go")

	settings, enabled := visionConfig()
	if !enabled {
		t.Fatal("FREERAG_VLM=go did not enable the stage")
	}
	if settings.Model != "qwen2.5vl:3b" {
		t.Fatalf("model = %q", settings.Model)
	}
	// 155 is a ceiling, not a target: it is what this model actually writes for
	// a figure, at 8.8 tok/s.
	if settings.MaxTokens != 155 {
		t.Fatalf("MaxTokens = %d, want 155", settings.MaxTokens)
	}
	// Knee of the measured throughput curve: 4 gave 2.75x, 6 added 4%.
	if settings.Workers != 4 {
		t.Fatalf("Workers = %d, want 4", settings.Workers)
	}

	t.Setenv("FREERAG_VLM", "off")
	if _, enabled := visionConfig(); enabled {
		t.Fatal("FREERAG_VLM=off did not disable the stage")
	}
}

func TestFigureGeometryRejectsIncompleteMetadata(t *testing.T) {
	cases := map[string]parser.Chunk{
		"no page": {ChunkID: "c", Metadata: map[string]any{"bbox": []any{0.0, 0.0, 1.0, 1.0}}},
		"no bbox": {ChunkID: "c", Metadata: map[string]any{"page_num": float64(1)}},
		"empty":   {ChunkID: "c", Metadata: map[string]any{"page_num": float64(1), "bbox": []any{5.0, 5.0, 5.0, 5.0}}},
	}
	for name, chunk := range cases {
		if _, _, err := figureGeometry(chunk); err == nil {
			t.Fatalf("%s: expected an error", name)
		}
	}
}
