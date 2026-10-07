package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// The chart backend: describe a figure with Laya-Chart instead of with a
// general vision model.
//
// Why a second backend exists is measured, not stylistic. The default describer
// is told not to quote numbers — a misread value lands in the index and is then
// retrieved as evidence forever, read by nobody, whereas the same mistake in a
// chat answer is visible immediately. That is correct for a photograph of a
// flow diagram and wrong for a bar chart: with the values stripped out, "what
// was revenue in 2020" has nothing to hit. Laya-Chart is a 1.825B model trained
// for exactly one thing — chart image to markdown table plus analysis — and it
// is measured at NHR 0.0095, i.e. 9.5 per 1000 numbers it writes are not
// derivable from the table it just wrote. So this backend keeps the table's
// numbers and strips the analysis's, which is the opposite split from the
// default and is the whole reason it is worth having.
//
// It is reached over HTTP rather than through the parse sidecar because the
// weights are 7.4 GB of torch that the sidecar (pymupdf + onnxruntime) has no
// use for, and because the model has to stay resident across a whole index run.

const (
	// chartBackend selects this backend in FREERAG_VLM.
	chartBackend = "chart"
	// defaultChartEndpoint is where chartvlm/server.py listens.
	defaultChartEndpoint = "http://127.0.0.1:8731"
	// defaultChartModel is the name recorded in a chunk's metadata. It is not an
	// Ollama tag — nothing resolves it — it is provenance for the reader.
	defaultChartModel = "laya-chart-1.7b"
	// defaultChartTimeout bounds ONE transcription. A cold load of 7 GB followed
	// by a 768-token decode on CPU is minutes; the bound exists so a wedged
	// service loses the figure rather than the document.
	defaultChartTimeout = 15 * time.Minute
	// chartMaxImageBytes refuses an oversized crop before it is sent. A rendered
	// figure is tens of KB; a megabyte means the crop went wrong, and shipping it
	// would only make the service slower for no gain.
	chartMaxImageBytes = 32 << 20
)

// chartTranscription is one figure as Laya-Chart sees it.
//
// The quality flags are the important half. `truncated` means the table stops
// mid-row and `low_confidence` means the model ran out of image tokens on a
// dense chart; both are measured failure modes of this model, not hypotheticals,
// and both are why the result cannot be treated as a plain caption and appended
// unconditionally.
type chartTranscription struct {
	ChartType     string         `json:"chart_type"`
	Axes          map[string]any `json:"axes"`
	Table         [][]string     `json:"table"`
	AnalysisZH    string         `json:"analysis_zh"`
	AnalysisEN    string         `json:"analysis_en"`
	Truncated     bool           `json:"truncated"`
	SchemaOK      bool           `json:"schema_ok"`
	NHR           float64        `json:"nhr"`
	NRows         int            `json:"n_rows"`
	NSeries       int            `json:"n_series"`
	LowConfidence bool           `json:"low_confidence"`
}

// chartClient talks to chartvlm/server.py.
type chartClient struct {
	endpoint string
	timeout  time.Duration

	once     sync.Once
	client   *http.Client
	unusable string // why the backend is off; empty means usable
}

// newChartClient builds a client, probing the service once.
//
// The probe is what keeps a missing service quiet. Without it every figure in
// every document would log a connection error, which is noise a user cannot act
// on; with it the stage is simply off and figures keep their extracted text,
// which is exactly what shipped before this backend existed.
func newChartClient(endpoint string, timeout time.Duration) *chartClient {
	if strings.TrimSpace(endpoint) == "" {
		endpoint = defaultChartEndpoint
	}
	if timeout <= 0 {
		timeout = defaultChartTimeout
	}
	c := &chartClient{endpoint: strings.TrimRight(endpoint, "/"), timeout: timeout}
	if !c.Reachable(context.Background()) {
		c.unusable = fmt.Sprintf("chart service at %s is not answering", c.endpoint)
	}
	return c
}

func (c *chartClient) httpClient() *http.Client {
	c.once.Do(func() {
		c.client = &http.Client{Timeout: c.timeout}
	})
	return c.client
}

// Reachable reports whether the service is up. It does not load the weights:
// /health answers before they are resident, which is what makes the probe cheap
// enough to run at startup.
func (c *chartClient) Reachable(ctx context.Context) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	request, err := http.NewRequestWithContext(probeCtx, http.MethodGet, c.endpoint+"/health", nil)
	if err != nil {
		return false
	}
	response, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
	if err != nil {
		return false
	}
	defer func() { _ = response.Body.Close() }()
	_, _ = io.Copy(io.Discard, response.Body)
	return response.StatusCode == http.StatusOK
}

// Transcribe sends one rendered figure and returns its transcription.
func (c *chartClient) Transcribe(ctx context.Context, image []byte) (*chartTranscription, error) {
	if c.unusable != "" {
		return nil, fmt.Errorf("%s", c.unusable)
	}
	if len(image) > chartMaxImageBytes {
		return nil, fmt.Errorf("figure crop is %d bytes, above the %d byte limit", len(image), chartMaxImageBytes)
	}

	payload, err := json.Marshal(map[string]string{
		"image": base64.StdEncoding.EncodeToString(image),
	})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/transcribe", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := c.httpClient().Do(request)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		// The service reports its own failure in `error`; it has already been
		// logged server-side, so carry it and let the caller skip the figure.
		var failure struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(body, &failure)
		if failure.Error != "" {
			return nil, fmt.Errorf("chart service: %s", failure.Error)
		}
		return nil, fmt.Errorf("chart service returned %d", response.StatusCode)
	}

	var result chartTranscription
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("chart service returned unreadable JSON: %w", err)
	}
	return &result, nil
}

// chartText turns a transcription into the text appended to a figure block,
// plus the metadata that lets a reader tell how much to trust it.
//
// The number policy is the point of this function, and it is per-part:
//
//   - the table keeps its numbers. They are the reason this backend exists: they
//     are what turns "what was revenue in 2020" into a retrievable fact;
//   - the analysis loses them (stripNumbers), for the same reason the default
//     describer does. The model's NHR is 0.0095 and its hallucinations cluster
//     in the prose, so the analysis is allowed to say "the trend is upward" and
//     is not allowed to put a value into the index.
//
// A truncated transcription yields no text at all. Half a table is worse than
// no table: it reads as complete, and a missing row is invisible to whoever
// retrieves it later. The model is known to stop early (it was trained against
// targets that did not fit its context), so this case is common enough to matter.
func chartText(result chartTranscription) (string, map[string]any) {
	meta := map[string]any{
		"chart_type":           result.ChartType,
		"chart_nhr":            result.NHR,
		"chart_rows":           result.NRows,
		"chart_series":         result.NSeries,
		"chart_low_confidence": result.LowConfidence || !result.SchemaOK,
	}
	if result.Truncated {
		meta["chart_truncated"] = true
		return "", meta
	}

	var parts []string
	if table := chartTableMarkdown(result); table != "" {
		parts = append(parts, table)
	}
	// Both analyses are kept. The model's `lang` attribute is unusable — 52.8%
	// of the slots labelled zh hold English — so there is nothing to choose by,
	// and a figure that is described twice cannot be under-described.
	for _, analysis := range []string{result.AnalysisZH, result.AnalysisEN} {
		if cleaned := strings.TrimSpace(stripNumbers(analysis)); cleaned != "" {
			parts = append(parts, cleaned)
		}
	}
	return strings.Join(parts, "\n"), meta
}

// chartTableMarkdown renders the transcribed rows as a markdown table, headed by
// what the axes are.
//
// The header line is not decoration: "图表数据表（line，横轴 Year，纵轴 Sales）"
// is what makes the chunk answer "is there a line chart in this document" —
// the question the table's bare numbers cannot.
func chartTableMarkdown(result chartTranscription) string {
	if len(result.Table) == 0 {
		return ""
	}
	var builder strings.Builder
	label := fmt.Sprintf("图表数据表（%s", orUnknown(result.ChartType))
	if x, ok := result.Axes["x"].(string); ok && strings.TrimSpace(x) != "" {
		label += "，横轴 " + strings.TrimSpace(x)
	}
	if y, ok := result.Axes["y"].(string); ok && strings.TrimSpace(y) != "" {
		label += "，纵轴 " + strings.TrimSpace(y)
	}
	builder.WriteString(label + "）\n")

	width := 0
	for _, row := range result.Table {
		if len(row) > width {
			width = len(row)
		}
	}
	for index, row := range result.Table {
		cells := make([]string, width)
		for i := range cells {
			if i < len(row) {
				cells[i] = strings.TrimSpace(row[i])
			}
		}
		builder.WriteString("| " + strings.Join(cells, " | ") + " |\n")
		if index == 0 {
			rule := make([]string, width)
			for i := range rule {
				rule[i] = "---"
			}
			builder.WriteString("| " + strings.Join(rule, " | ") + " |\n")
		}
	}
	return strings.TrimRight(builder.String(), "\n")
}

func orUnknown(value string) string {
	if strings.TrimSpace(value) == "" {
		return "未知类型"
	}
	return value
}
