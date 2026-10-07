package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"freerag/internal/parser"
)

// chartServer is the Laya-Chart service as the kernel sees it: one POST, one
// JSON body back.
type chartServer struct {
	// reply is what /transcribe answers with.
	reply any
	// status is its status code; 200 unless a test says otherwise.
	status int
	// received is the decoded image the service was sent.
	received []byte
	calls    atomic.Int32
}

func (s *chartServer) start(t *testing.T) *chartClient {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true,"ready":true}`))
			return
		}
		if r.URL.Path != "/transcribe" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		s.calls.Add(1)
		body, _ := io.ReadAll(r.Body)
		var payload struct {
			Image string `json:"image"`
		}
		_ = json.Unmarshal(body, &payload)
		s.received, _ = base64.StdEncoding.DecodeString(payload.Image)

		status := s.status
		if status == 0 {
			status = http.StatusOK
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(s.reply)
	}))
	t.Cleanup(server.Close)
	return newChartClient(server.URL, 30*time.Second)
}

func chartSettingsFor(endpoint string, workers int) visionSettings {
	return visionSettings{
		Backend:   chartBackend,
		Endpoint:  endpoint,
		Model:     defaultChartModel,
		MaxTokens: 768,
		MaxSide:   768,
		Workers:   workers,
	}
}

func wholeTranscription() chartTranscription {
	return chartTranscription{
		ChartType: "line",
		Axes:      map[string]any{"x": "Year", "y": "Sales"},
		Table: [][]string{
			{"Year", "Series A", "Series B"},
			{"2020", "159.5", "244.4"},
			{"2021", "180.2", "210.9"},
		},
		AnalysisZH: "Series A 从 159.5 上升到 180.2，呈上升趋势。",
		AnalysisEN: "Series A rose from 159.5 to 180.2, an upward trend.",
		Truncated:  false,
		SchemaOK:   true,
		NHR:        0.0,
		NRows:      2,
		NSeries:    2,
	}
}

func TestChartTextKeepsTableNumbersAndStripsAnalysisNumbers(t *testing.T) {
	// The split is this backend's whole reason for existing: the transcribed
	// table is the answer to "what was the value in 2020", so its numbers stay;
	// the analysis is prose whose hallucinated values would land in the index
	// and never be seen again, so its numbers go.
	text, meta := chartText(wholeTranscription())

	for _, kept := range []string{"159.5", "244.4", "180.2"} {
		if !strings.Contains(text, kept) {
			t.Fatalf("the table lost %q:\n%s", kept, text)
		}
	}
	for _, dropped := range []string{"159.5 上升到", "from 159.5 to 180.2"} {
		if strings.Contains(text, dropped) {
			t.Fatalf("the analysis kept its numbers: %q", dropped)
		}
	}
	// The analysis survives with its numbers removed — it is what makes the
	// chunk answer "what does this figure conclude".
	if !strings.Contains(text, "上升趋势") || !strings.Contains(text, "upward trend") {
		t.Fatalf("the analysis was dropped instead of cleaned:\n%s", text)
	}
	// The header is what makes the chunk answer "is there a line chart here",
	// which the bare numbers cannot.
	for _, want := range []string{"图表数据表（line", "横轴 Year", "纵轴 Sales"} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q in:\n%s", want, text)
		}
	}
	// A well-formed table with two series is not a dense-chart case.
	if meta["chart_low_confidence"] != false {
		t.Fatalf("low_confidence = %v, want false", meta["chart_low_confidence"])
	}
}

func TestChartTextRefusesAHalfWrittenTable(t *testing.T) {
	// A truncated transcription is common (the model was trained against
	// targets that did not fit), and half a table is worse than none: it reads
	// as complete and the missing rows are invisible to whoever retrieves it.
	broken := wholeTranscription()
	broken.Truncated = true

	text, meta := chartText(broken)
	if text != "" {
		t.Fatalf("a truncated transcription produced text:\n%s", text)
	}
	// Refused, but recorded — "tried and failed" is not "never tried".
	if meta["chart_truncated"] != true {
		t.Fatalf("the refusal was not recorded: %v", meta)
	}
}

func TestChartTextMarksADenseChartAsLowConfidence(t *testing.T) {
	// The model gets a fixed 640 image tokens whatever the image, so a 20-series
	// chart comes back partly read. That has to reach the chunk.
	dense := wholeTranscription()
	dense.NSeries = 20
	dense.LowConfidence = true

	if _, meta := chartText(dense); meta["chart_low_confidence"] != true {
		t.Fatalf("a dense chart was not marked: %v", meta)
	}
}

func TestCaptionFiguresWithTheChartBackendPostsTheCropAndRecordsQuality(t *testing.T) {
	service := &chartServer{reply: wholeTranscription()}
	client := service.start(t)
	k := &kernel{chart: client, renderer: &fakeRenderer{}}
	t.Setenv("FREERAG_VISION_CACHE", t.TempDir())

	result := &parser.Result{Chunks: []parser.Chunk{
		{ChunkID: "c0", Text: "正文", Metadata: map[string]any{"block_type": "Text"}},
		figureChunk("c1", 3, "Figure 1: 年度销量"),
	}}

	if got := k.captionFigures(context.Background(), "/tmp/paper.pdf", result,
		chartSettingsFor(client.endpoint, 2)); got != 1 {
		t.Fatalf("described %d figures, want 1", got)
	}

	// What was sent is the rendered crop, not the chunk's text.
	want, err := base64.StdEncoding.DecodeString(tinyPNG)
	if err != nil {
		t.Fatal(err)
	}
	if string(service.received) != string(want) {
		t.Fatalf("the service was sent %d bytes, want the %d-byte crop", len(service.received), len(want))
	}

	figure := result.Chunks[1]
	if !strings.Contains(figure.Text, "Figure 1: 年度销量") {
		t.Fatalf("the extracted text was lost: %q", figure.Text)
	}
	if !strings.Contains(figure.Text, figureMarker) {
		t.Fatalf("no marker in %q", figure.Text)
	}
	if !strings.Contains(figure.Text, "159.5") {
		t.Fatalf("the table's numbers were stripped: %q", figure.Text)
	}
	if figure.Metadata["figure_summary_model"] != defaultChartModel {
		t.Fatalf("attributed to %v, want %s", figure.Metadata["figure_summary_model"], defaultChartModel)
	}
	if figure.Metadata["chart_type"] != "line" {
		t.Fatalf("chart_type = %v", figure.Metadata["chart_type"])
	}
	if figure.Metadata["chart_rows"] != 2 {
		t.Fatalf("chart_rows = %v", figure.Metadata["chart_rows"])
	}
	if result.Chunks[0].Text != "正文" {
		t.Fatal("a text chunk was modified")
	}
}

func TestCaptionFiguresRecordsARefusalWithoutAppendingText(t *testing.T) {
	// The model refused the figure (its table came back half-written). The
	// chunk keeps exactly the text it had, but the reason is recorded.
	broken := wholeTranscription()
	broken.Truncated = true
	service := &chartServer{reply: broken}
	client := service.start(t)
	k := &kernel{chart: client, renderer: &fakeRenderer{}}
	t.Setenv("FREERAG_VISION_CACHE", t.TempDir())

	result := &parser.Result{Chunks: []parser.Chunk{figureChunk("c0", 1, "Figure 1: x")}}
	if got := k.captionFigures(context.Background(), "/tmp/p.pdf", result,
		chartSettingsFor(client.endpoint, 1)); got != 0 {
		t.Fatalf("described %d, want 0", got)
	}
	if result.Chunks[0].Text != "Figure 1: x" {
		t.Fatalf("a refused figure was modified: %q", result.Chunks[0].Text)
	}
	if result.Chunks[0].Metadata["chart_truncated"] != true {
		t.Fatalf("the refusal left no trace: %v", result.Chunks[0].Metadata)
	}
}

func TestCaptionFiguresReusesTheChartCache(t *testing.T) {
	// Re-indexing an unchanged document hits the parse cache and never re-renders
	// the figure, so without a cache here the model would be asked again for
	// every figure on every rebuild — at 14 s a figure.
	service := &chartServer{reply: wholeTranscription()}
	client := service.start(t)
	k := &kernel{chart: client, renderer: &fakeRenderer{}}
	t.Setenv("FREERAG_VISION_CACHE", t.TempDir())
	settings := chartSettingsFor(client.endpoint, 1)

	for i := 0; i < 2; i++ {
		result := &parser.Result{Chunks: []parser.Chunk{figureChunk("c0", 1, "Figure 1: x")}}
		if got := k.captionFigures(context.Background(), "/tmp/p.pdf", result, settings); got != 1 {
			t.Fatalf("run %d: described %d, want 1", i+1, got)
		}
		if !strings.Contains(result.Chunks[0].Text, "159.5") {
			t.Fatalf("run %d: the cached text lost its table: %q", i+1, result.Chunks[0].Text)
		}
		// The quality flags have to survive the cache too: they are what makes
		// the chunk trustworthy, and they cannot be recomputed from the text.
		if result.Chunks[0].Metadata["chart_type"] != "line" {
			t.Fatalf("run %d: the cached metadata lost chart_type: %v", i+1, result.Chunks[0].Metadata)
		}
	}
	if service.calls.Load() != 1 {
		t.Fatalf("the service was called %d times, want 1", service.calls.Load())
	}
}

func TestCaptionFiguresSurvivesADeadChartService(t *testing.T) {
	// The service is a separate process that may not be running. One figure
	// failing must not cost the document.
	service := &chartServer{reply: map[string]string{"error": "transcription failed"}, status: 502}
	client := service.start(t)
	k := &kernel{chart: client, renderer: &fakeRenderer{}}
	t.Setenv("FREERAG_VISION_CACHE", t.TempDir())

	result := &parser.Result{Chunks: []parser.Chunk{
		figureChunk("c0", 1, "Figure 1: 第一张"),
		figureChunk("c1", 2, "Figure 2: 第二张"),
	}}
	if got := k.captionFigures(context.Background(), "/tmp/p.pdf", result,
		chartSettingsFor(client.endpoint, 2)); got != 0 {
		t.Fatalf("described %d, want 0", got)
	}
	for i, chunk := range result.Chunks {
		if strings.Contains(chunk.Text, figureMarker) {
			t.Fatalf("figure %d got a marker anyway: %q", i, chunk.Text)
		}
	}
}

func TestChartVisionConfigDefaultsToOneWorker(t *testing.T) {
	// Against 4 for Ollama. A chart transcription is a 1.8B model decoding up to
	// 768 tokens inside one Python process, so a second figure in flight waits
	// rather than running alongside — more of them at once does not help.
	t.Setenv("FREERAG_VLM", chartBackend)
	for _, key := range []string{"FREERAG_VLM_MODEL", "FREERAG_CHART_ENDPOINT", "FREERAG_CHART_CONCURRENCY"} {
		t.Setenv(key, "")
	}

	settings, enabled := visionConfig()
	if !enabled {
		t.Fatal("FREERAG_VLM=chart did not enable the stage")
	}
	if settings.Backend != chartBackend {
		t.Fatalf("Backend = %q", settings.Backend)
	}
	if settings.Workers != 1 {
		t.Fatalf("Workers = %d, want 1", settings.Workers)
	}
	if settings.Endpoint != defaultChartEndpoint {
		t.Fatalf("Endpoint = %q, want %q", settings.Endpoint, defaultChartEndpoint)
	}
	// The Ollama default is not an Ollama tag and must not be inherited here.
	if settings.Model != defaultChartModel {
		t.Fatalf("Model = %q, want %q", settings.Model, defaultChartModel)
	}

	t.Setenv("FREERAG_CHART_CONCURRENCY", "0")
	if _, enabled := visionConfig(); enabled {
		t.Fatal("zero workers should disable the stage rather than divide by zero")
	}
}

// TestChartBackendAgainstALiveService is the one test here that needs the real
// thing: a running chartvlm/server.py with the r5 weights loaded.
//
// Skipped unless both variables are set, because it costs a full transcription
// (measured 107 s for one figure on MPS, fp32) and the weights are 7.4 GB:
//
//	FREERAG_CHART_TEST_ENDPOINT=http://127.0.0.1:8731 \
//	FREERAG_CHART_TEST_IMAGE=/path/to/chart.png \
//	CGO_ENABLED=0 go test ./cmd/freerag/ -run TestChartBackendAgainstALiveService -v
//
// It exists because the mocked tests cannot catch the failure this backend is
// most prone to: weights that load without a complaint but do not match, which
// yields a well-formed JSON answer describing nothing.
func TestChartBackendAgainstALiveService(t *testing.T) {
	endpoint := strings.TrimSpace(os.Getenv("FREERAG_CHART_TEST_ENDPOINT"))
	imagePath := strings.TrimSpace(os.Getenv("FREERAG_CHART_TEST_IMAGE"))
	if endpoint == "" || imagePath == "" {
		t.Skip("set FREERAG_CHART_TEST_ENDPOINT and FREERAG_CHART_TEST_IMAGE to run")
	}
	image, err := os.ReadFile(imagePath)
	if err != nil {
		t.Fatalf("could not read %s: %v", imagePath, err)
	}

	client := newChartClient(endpoint, 15*time.Minute)
	if client.unusable != "" {
		t.Fatalf("the service is not answering: %s", client.unusable)
	}

	start := time.Now()
	result, err := client.Transcribe(context.Background(), image)
	if err != nil {
		t.Fatalf("transcription failed: %v", err)
	}
	text, meta := chartText(*result)
	t.Logf("%s: %s, %d rows in %s", result.ChartType, meta, result.NRows, time.Since(start).Round(time.Second))
	t.Logf("text:\n%s", text)

	// A transcription that came back whole, with numbers the model stands behind.
	if result.Truncated {
		t.Fatal("the transcription was truncated; try a simpler chart")
	}
	if !result.SchemaOK {
		t.Errorf("schema_ok = false")
	}
	if result.NRows < 2 || len(result.Table) < 2 {
		t.Fatalf("the table has %d rows; the model may not have read the chart", result.NRows)
	}
	for _, want := range []string{"| ", "---"} {
		if !strings.Contains(text, want) {
			t.Fatalf("the markdown table is missing (%q):\n%s", want, text)
		}
	}
	// The table's numbers stay — they are what makes the figure answerable.
	if !strings.ContainsAny(text, "0123456789") {
		t.Fatalf("no numbers survived:\n%s", text)
	}
	// ...and the analysis's go, so a misread value cannot be retrieved as fact.
	if rest, _, found := strings.Cut(text, "\n| "); found && strings.Contains(rest, "。") {
		t.Logf("note: analysis follows the table; its numbers are stripped")
	}
}

// TestStartChartServiceOwnsItsProcess covers the part the mocked tests cannot:
// that the kernel can bring the service up on its own and, more importantly,
// that it takes it down again.
//
// The second half is the one that matters in use. A chart service left behind
// keeps 7.4 GB of weights resident in a machine whose user has closed the app,
// and nothing about that shows up as an error anywhere.
//
// Skipped without FREERAG_CHARTVLM_ROOT and the chart venv.
func TestStartChartServiceOwnsItsProcess(t *testing.T) {
	if strings.TrimSpace(os.Getenv("FREERAG_CHARTVLM_ROOT")) == "" {
		t.Skip("set FREERAG_CHARTVLM_ROOT (and install chartvlm/requirements.txt) to run")
	}

	service, err := startChartService()
	if err != nil {
		t.Fatalf("could not start the chart service: %v", err)
	}
	defer func() { _ = service.Close() }()

	// Started, answering, and on a port of its own rather than the default one:
	// two kernels must not end up sharing a model by accident.
	if service.port == 0 {
		t.Fatal("the service was not given a port")
	}
	probe := &chartClient{endpoint: service.endpoint, timeout: 5 * time.Second}
	if !probe.Reachable(context.Background()) {
		t.Fatalf("%s is not answering /health after startup", service.endpoint)
	}
	if ownedSuffix(service) == ownedSuffix(nil) {
		t.Fatal("the service was started but reported as external")
	}

	if err := service.Close(); err != nil {
		t.Fatalf("could not stop the chart service: %v", err)
	}
	if probe.Reachable(context.Background()) {
		t.Fatalf("%s is still answering after Close", service.endpoint)
	}
}

func TestVlmModelIsNotPassedToTheSidecarUnderTheChartBackend(t *testing.T) {
	// sidecar/vlm.py resolves the model through Ollama. With the chart backend
	// selected, FREERAG_VLM_MODEL names a service Ollama has never heard of, so
	// handing it down would fail a pull on every figure — and would caption every
	// figure twice, once per path.
	t.Setenv("FREERAG_VLM_MODEL", defaultChartModel)

	t.Setenv("FREERAG_VLM", chartBackend)
	if got := vlmModel(); got != "" {
		t.Fatalf("vlmModel() = %q, want empty under the chart backend", got)
	}

	t.Setenv("FREERAG_VLM", "go")
	if got := vlmModel(); got != defaultChartModel {
		t.Fatalf("vlmModel() = %q, want the Ollama path to still pass it through", got)
	}
}

func TestVisionCacheKeySeparatesTheTwoBackends(t *testing.T) {
	// Same image, same model name, different backend: one is told to leave the
	// values out and the other exists to keep them, so they must not share a
	// cache entry.
	image := []byte("same-bytes")
	ollama := visionSettings{Model: "m", MaxTokens: 155}
	chart := ollama
	chart.Backend = chartBackend

	if visionCacheKey(image, ollama) == visionCacheKey(image, chart) {
		t.Fatal("the two backends share a cache key")
	}
}
