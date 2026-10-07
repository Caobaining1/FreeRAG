package main

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"freerag/internal/parser"
)

// Figure captioning: describe the image in a Figure block, then append that
// description to the block's text before it is embedded.
//
// Why this is worth doing was measured rather than assumed. Inside a figure's
// box, the extracted text of a vector PDF is a jumble of axis labels, tick
// values and rotated annotations — 2,754 characters for one MEDAL figure. The
// words are already indexed; the figure's *point* is not: which curve sits above
// which, which answer was marked wrong, what the flow is. So the description is
// APPENDED, not substituted: the exact strings stay (grep_search and identifier
// queries depend on them) and the description adds the meaning on top.
//
// Only Figure blocks, never every bitmap. On the same corpus the pages held
// 4-38 raster images each but only 0-12 layout Figure blocks; the rest are
// logos, rules and stray bitmaps, and paying a vision model for those would
// dwarf the useful work.
//
// The model was chosen by measurement too (see docs/plan.md): qwen2.5vl:3b at
// 3.2 GB, ~36s per figure sequentially, 2.75x throughput at 4 concurrent.
// qwen3-vl:4b was rejected for producing 1,319 characters of unreadable
// reasoning tokens per image and ignoring think=false — 122s per figure.

// figureMarker separates the description from the extracted text so a reader
// (and a grep) can tell which is derived.
const figureMarker = "【图内容】"

// visionPromptVersion is part of the cache key: change the prompt or the model
// and every cached description is invalidated, which is exactly the intent.
const visionPromptVersion = "v1"

// figurePrompt asks for the figure's meaning, not its words.
//
// The no-numbers rule is enforced in code as well (stripNumbers) because the
// model does not reliably obey it: asked not to quote values, it answered with
// "人类的F1得分最高，为93.16%" — correct that time, but a misread value that
// lands in the index is retrieved as evidence forever and never seen by anyone,
// whereas the same mistake in a chat answer is immediately visible.
var figurePrompt = `你是文档图表理解助手。这张图来自一篇学术论文，图中文字通常为英文。

请用中文写 2-4 句描述，说明：
1. 这是什么类型的图（折线图/柱状图/示意图/流程图/截图等）；
2. 它展示了什么对比、趋势或结构；
3. 能得出的主要结论。

要求：
- 不要引用任何具体数值、分数、百分比或坐标轴刻度；
- 不要复述图注（caption）的文字；
- 只描述你能明确看出的内容，看不清就直说看不清；
- 直接给出描述，不要客套话。`

// visionSettings is the caption stage's configuration.
type visionSettings struct {
	// Backend picks the describer: "" or "ollama" for a general vision model
	// over Ollama, "chart" for Laya-Chart (see vision_chart.go). They differ in
	// what they do with numbers, which is why it is a setting and not a default.
	Backend string
	// Endpoint is the chart service's base URL. Unused by the Ollama backend,
	// which shares the generator's host (main.go).
	Endpoint string
	// Model names the describer. An Ollama tag for one backend, provenance
	// only for the other.
	Model string
	// MaxTokens caps the description. Generation, not image encoding, is the
	// cost: measured 155 tokens at 8.8 tok/s, so this bounds the whole stage.
	// It is a ceiling, not a target — a shorter description is not padded.
	//
	// Ignored by the chart backend, whose budget is fixed server-side at 768
	// because below that the transcribed tables come back half-written.
	MaxTokens int
	// MaxSide caps the crop's longest side in pixels.
	MaxSide int
	// Workers is how many descriptions may be in flight at once. Measured
	// 2.75x throughput at 4 against 1, with the knee at 4 (6 added 4%).
	Workers int
}

// visionConfig reads the caption stage's settings from the environment.
//
// FREERAG_VLM=off disables the stage entirely, which is the switch to reach for
// when the vision model is not pulled: without it every figure logs a failure.
func visionConfig() (visionSettings, bool) {
	// Neither value is "on": both name WHICH Go-side describer runs.
	//
	// There are two captioning paths in the tree: sidecar/vlm.py, which runs
	// inside the parse and is what FREERAG_VLM_MODEL drives, and this one, which
	// runs in Go after the parse. The sidecar one is the default because it
	// describes a figure BEFORE the figure/caption merge, so the description can
	// take part in that merge and in noise filtering, and because it needs no
	// image round trip over the JSON-RPC pipe. Both enabled at once would caption
	// every figure twice and append the text twice — so this one is opt-in.
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FREERAG_VLM"))) {
	case "go":
		return ollamaVisionConfig()
	case chartBackend:
		return chartVisionConfig()
	}
	return visionSettings{}, false
}

func ollamaVisionConfig() (visionSettings, bool) {
	settings := visionSettings{
		Model:     envString("FREERAG_VLM_MODEL", "qwen2.5vl:3b"),
		MaxTokens: envInt("FREERAG_VLM_MAX_TOKENS", 155),
		MaxSide:   envInt("FREERAG_VLM_MAX_SIDE", 768),
		Workers:   envInt("FREERAG_VLM_CONCURRENCY", 4),
	}
	if settings.Model == "" || settings.Workers < 1 {
		return visionSettings{}, false
	}
	return settings, true
}

// chartVisionConfig is the Laya-Chart backend.
//
// Workers defaults to 1, against 4 for Ollama, because the two scale
// differently: a chart transcription is a 1.8B model decoding up to 768 tokens
// inside one Python process, so a second figure in flight does not run in
// parallel — it waits, and both finish later. Throughput here comes from a
// shorter decode, not from more of them at once.
func chartVisionConfig() (visionSettings, bool) {
	settings := visionSettings{
		Backend: chartBackend,
		// Deliberately NOT FREERAG_VLM_MODEL. That variable holds an Ollama tag,
		// and the two backends are mutually exclusive, so a machine carrying both
		// would otherwise report the Ollama model for work Laya-Chart did — the
		// one place a reader checks to find out what produced a description.
		// Overridable by its own variable for anyone who runs a differently
		// named build.
		Model:     envString("FREERAG_CHART_MODEL", defaultChartModel),
		Endpoint:  envString("FREERAG_CHART_ENDPOINT", defaultChartEndpoint),
		MaxTokens: envInt("FREERAG_VLM_MAX_TOKENS", 768),
		MaxSide:   envInt("FREERAG_VLM_MAX_SIDE", 768),
		Workers:   envInt("FREERAG_CHART_CONCURRENCY", 1),
	}
	if settings.Workers < 1 {
		return visionSettings{}, false
	}
	return settings, true
}

func envString(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return fallback
	}
	return parsed
}

// captioner is the vision model as this stage needs it. An interface so the
// stage can be tested — and so an unreachable Ollama is a nil, not a hard
// dependency of indexing.
type captioner interface {
	CaptionImage(ctx context.Context, model, prompt, imageB64 string, maxTokens int) (string, error)
}

// figureRenderer crops a figure out of its page. Also an interface, for the same
// reason: the stage's concurrency and failure behaviour are what need testing,
// and neither is about PyMuPDF.
type figureRenderer interface {
	Render(ctx context.Context, req parser.RenderRequest, timeout time.Duration) (*parser.Page, error)
}

// valueRun matches a maximal run of identifier-ish characters that contains at
// least one digit: "93.16%", "gpt-4-0613", "F1", "0.52".
//
// Splitting on runs rather than on whitespace is what makes this work on Chinese
// text, where a value is glued to the next word: "93.16%，davinci" has no space
// in it at all, so a whitespace tokenizer sees one token and matches nothing.
// CJK characters are outside the class, so a run never spans into them.
var valueRun = regexp.MustCompile(`[0-9A-Za-z_%.,\-]*[0-9][0-9A-Za-z_%.,\-]*`)

// stripNumbers removes the values from a description, and keeps identifiers.
//
// Enforced here rather than trusted to the prompt: asked not to quote numbers,
// the model quoted them anyway. A misread value in a chat answer is visible and
// someone will correct it; the same value in the index is retrieved as evidence
// forever and read by no one. The real values stay in the block's extracted
// text, which is exact — so nothing is lost except the risk.
//
// "F1" survives on purpose: it names a metric. "gpt-4-0613" survives because a
// run containing a letter is an identifier, and dropping it would undo the
// reason the extracted text is kept alongside the description.
func stripNumbers(text string) string {
	return valueRun.ReplaceAllStringFunc(text, func(run string) string {
		if strings.ContainsFunc(run, unicode.IsLetter) {
			return run
		}
		return ""
	})
}

// isFigureChunk reports whether a chunk is a figure, caption included.
func isFigureChunk(chunk parser.Chunk) bool {
	blockType, _ := chunk.Metadata["block_type"].(string)
	return strings.HasPrefix(blockType, "Figure")
}

// figureGeometry reads a chunk's page and box out of its metadata.
//
// The box arrives through JSON, so its numbers are float64 whatever they were
// on the Python side — and it is absent for any chunk whose provider did not
// measure one.
func figureGeometry(chunk parser.Chunk) (int, []float64, error) {
	page := 0
	switch value := chunk.Metadata["page_num"].(type) {
	case float64:
		page = int(value)
	case int:
		page = value
	}
	if page < 1 {
		return 0, nil, fmt.Errorf("figure %s has no page number", chunk.ChunkID)
	}

	raw, ok := chunk.Metadata["bbox"].([]any)
	if !ok || len(raw) != 4 {
		return 0, nil, fmt.Errorf("figure %s has no bbox", chunk.ChunkID)
	}
	box := make([]float64, 0, 4)
	for _, value := range raw {
		number, ok := value.(float64)
		if !ok {
			return 0, nil, fmt.Errorf("figure %s has a malformed bbox", chunk.ChunkID)
		}
		box = append(box, number)
	}
	if box[2] <= box[0] || box[3] <= box[1] {
		return 0, nil, fmt.Errorf("figure %s has an empty bbox", chunk.ChunkID)
	}
	return page, box, nil
}

// visionCacheDir is where descriptions are remembered between runs.
//
// On disk rather than in memory because the parse cache sits in front of this
// stage: re-indexing an unchanged document hits the parse cache and never sees
// the figure again, so an in-memory cache would re-ask the model for every
// figure on every rebuild.
func visionCacheDir() string {
	if dir := strings.TrimSpace(os.Getenv("FREERAG_VISION_CACHE")); dir != "" {
		return dir
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(base, "freerag", "vision")
}

// visionCacheKey binds a description to the image AND to what produced it.
//
// The bytes carry the resolution, so lowering MaxSide misses the cache on
// purpose — a smaller crop can yield a different description, and serving the
// old one would make a resolution change silently ineffective.
func visionCacheKey(image []byte, settings visionSettings) string {
	sum := sha256.New()
	sum.Write(image)
	// The backend is in the key because the two of them answer the same image
	// differently on purpose: one is told to leave the values out and the other
	// exists to keep them. Serving one's answer to the other's key would make
	// the choice silently ineffective.
	fmt.Fprintf(sum, "|%s|%s|%s|%d", settings.Backend, settings.Model, visionPromptVersion, settings.MaxTokens)
	return hex.EncodeToString(sum.Sum(nil))[:32]
}

func readVisionCache(key string) (string, bool) {
	dir := visionCacheDir()
	if dir == "" {
		return "", false
	}
	body, err := os.ReadFile(filepath.Join(dir, key+".txt"))
	if err != nil {
		return "", false
	}
	text := strings.TrimSpace(string(body))
	return text, text != ""
}

func writeVisionCache(key, text string) {
	dir := visionCacheDir()
	if dir == "" {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	if err := os.WriteFile(filepath.Join(dir, key+".txt"), []byte(text+"\n"), 0o644); err != nil {
		log.Printf("warning: could not cache a figure description: %v", err)
	}
}

// A chart transcription is cached as JSON, not as bare text, because its text is
// only half of what it produced: the quality flags (NHR, whether the table came
// back whole, whether the chart was too dense to read) are what let a reader
// decide how much to trust it, and they cannot be recomputed from the text.
type chartCacheEntry struct {
	Text string         `json:"text"`
	Meta map[string]any `json:"metadata"`
}

func readChartCache(key string) (string, map[string]any, bool) {
	dir := visionCacheDir()
	if dir == "" {
		return "", nil, false
	}
	body, err := os.ReadFile(filepath.Join(dir, key+".chart.json"))
	if err != nil {
		return "", nil, false
	}
	var entry chartCacheEntry
	if err := json.Unmarshal(body, &entry); err != nil {
		return "", nil, false
	}
	if strings.TrimSpace(entry.Text) == "" {
		return "", nil, false
	}
	return entry.Text, entry.Meta, true
}

func writeChartCache(key, text string, meta map[string]any) {
	dir := visionCacheDir()
	if dir == "" {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	body, err := json.Marshal(chartCacheEntry{Text: text, Meta: meta})
	if err != nil {
		return
	}
	if err := os.WriteFile(filepath.Join(dir, key+".chart.json"), body, 0o644); err != nil {
		log.Printf("warning: could not cache a chart transcription: %v", err)
	}
}

// captionFigures appends a vision description to every Figure block in place,
// and returns how many were described.
//
// It never returns an error: a figure that cannot be described keeps the text
// it already had, which is exactly the behaviour that shipped before this stage
// existed. Failing an entire document because a vision model was slow would
// trade a missing description for a missing document.
func (k *kernel) captionFigures(
	ctx context.Context, sourcePath string, result *parser.Result, settings visionSettings,
) int {
	if k.renderer == nil || (k.captioner == nil && k.chart == nil) {
		return 0
	}

	targets := make([]*parser.Chunk, 0, 8)
	for index := range result.Chunks {
		if isFigureChunk(result.Chunks[index]) {
			targets = append(targets, &result.Chunks[index])
		}
	}
	if len(targets) == 0 {
		return 0
	}

	gate := k.visionGate(settings.Workers)
	var (
		wg        sync.WaitGroup
		mu        sync.Mutex
		captioned int
	)

	for _, target := range targets {
		if ctx.Err() != nil {
			break
		}
		// The slot is taken before the goroutine starts, so the bound also
		// bounds how many of these exist at once.
		gate <- struct{}{}
		wg.Add(1)
		go func(chunk *parser.Chunk) {
			defer wg.Done()
			defer func() { <-gate }()

			text, extra, err := k.captionOne(ctx, sourcePath, *chunk, settings)
			if err != nil {
				log.Printf("warning: could not describe figure %s: %v", chunk.ChunkID, err)
				return
			}

			mu.Lock()
			// Recorded whether or not there is text: a figure the chart model
			// refused (its table came back half-written) must still be marked,
			// or a reader cannot tell "tried and failed" from "never tried".
			for key, value := range extra {
				if chunk.Metadata == nil {
					chunk.Metadata = map[string]any{}
				}
				chunk.Metadata[key] = value
			}
			if text == "" {
				mu.Unlock()
				return
			}

			chunk.Text = chunk.Text + "\n" + figureMarker + text
			if chunk.Metadata == nil {
				chunk.Metadata = map[string]any{}
			}
			// Marked as derived so nothing downstream mistakes a description
			// for extracted text.
			chunk.Metadata["figure_summary"] = text
			chunk.Metadata["figure_summary_model"] = settings.Model
			captioned++
			mu.Unlock()

			k.progress("figure", map[string]any{"file": filepath.Base(sourcePath), "chunk": chunk.ChunkID})
		}(target)
	}

	wg.Wait()
	return captioned
}

// captionOne renders one figure's box and describes it, cache first.
//
// The second result is metadata the describer wants recorded on the chunk —
// the chart backend's quality flags. It is separate from the text because the
// two do not always arrive together: a transcription can be refused (the table
// came back half-written) and still be worth recording, since "this figure was
// tried and did not come out whole" is not the same as "this figure was never
// looked at".
func (k *kernel) captionOne(
	ctx context.Context, sourcePath string, chunk parser.Chunk, settings visionSettings,
) (string, map[string]any, error) {
	page, box, err := figureGeometry(chunk)
	if err != nil {
		return "", nil, err
	}

	page0, err := k.renderer.Render(ctx, parser.RenderRequest{
		Path:    sourcePath,
		Page:    page,
		BBox:    box,
		MaxSide: settings.MaxSide,
		// Full resolution for the crop; MaxSide is what brings it down, and
		// doing it once, here, avoids rendering big and shrinking after.
		DPI: 150,
	}, 0)
	if err != nil {
		return "", nil, err
	}

	image, err := base64.StdEncoding.DecodeString(page0.Image)
	if err != nil {
		return "", nil, fmt.Errorf("render returned undecodable image data: %w", err)
	}

	if settings.Backend == chartBackend {
		return k.captionOneChart(ctx, image, settings)
	}

	key := visionCacheKey(image, settings)
	if cached, ok := readVisionCache(key); ok {
		return cached, nil, nil
	}

	text, err := k.captioner.CaptionImage(ctx, settings.Model, figurePrompt, page0.Image, settings.MaxTokens)
	if err != nil {
		return "", nil, err
	}
	text = strings.TrimSpace(stripNumbers(text))
	if text == "" {
		return "", nil, nil
	}
	writeVisionCache(key, text)
	return text, nil, nil
}

// captionOneChart transcribes one figure with Laya-Chart.
//
// The numbers in the table are kept and the numbers in the analysis are not —
// see chartText for why the split goes that way round.
func (k *kernel) captionOneChart(
	ctx context.Context, image []byte, settings visionSettings,
) (string, map[string]any, error) {
	if k.chart == nil {
		return "", nil, errors.New("the chart backend is selected but no chart service is configured")
	}

	key := visionCacheKey(image, settings)
	if cached, meta, ok := readChartCache(key); ok {
		return cached, meta, nil
	}

	result, err := k.chart.Transcribe(ctx, image)
	if err != nil {
		return "", nil, err
	}
	text, meta := chartText(*result)
	// Nothing is cached when the transcription was refused: a refusal is not a
	// result, and remembering it would make a transient service fault permanent
	// for the whole life of the cache.
	if text == "" {
		return "", meta, nil
	}
	writeChartCache(key, text, meta)
	return text, meta, nil
}

// visionGate bounds how many descriptions run at once, process-wide.
//
// Process-wide rather than per document on purpose: a batch indexes up to
// `Write` documents at a time, so a per-document bound of 4 would put 16
// requests on the vision model and lose the throughput the bound is there to
// keep.
func (k *kernel) visionGate(workers int) chan struct{} {
	k.visionOnce.Do(func() {
		k.visionSlot = make(chan struct{}, workers)
	})
	return k.visionSlot
}
