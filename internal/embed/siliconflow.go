package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// SiliconFlow defaults. BGE-M3 is 1024-dimensional and accepts 8192 tokens.
const (
	DefaultSiliconFlowURL   = "https://api.siliconflow.cn/v1"
	DefaultSiliconFlowModel = "BAAI/bge-m3"

	// DefaultDims is BGE-M3's vector width, matching docs/plan.md §4.
	DefaultDims = 1024

	// defaultBatch keeps one request's payload modest. The endpoint accepts
	// more, but a smaller batch makes a retry cheap and bounds the request size
	// when chunks are long.
	defaultBatch = 32

	// defaultMaxChars truncates each input before embedding.
	//
	// BGE-M3 reads 8192 tokens and silently ignores the rest, so sending more
	// only wastes bandwidth. Counting characters rather than tokens is
	// deliberate: 8000 characters is roughly 8000 tokens for Chinese and about
	// a quarter of that for English, so it is safe for the mixed profile without
	// pulling in a tokenizer.
	defaultMaxChars = 8000

	defaultTimeout = 2 * time.Minute
)

// SiliconFlow calls SiliconFlow's OpenAI-compatible embeddings endpoint.
type SiliconFlow struct {
	// BaseURL defaults to DefaultSiliconFlowURL.
	BaseURL string
	// APIKey is the bearer token; required.
	APIKey string
	// Model defaults to DefaultSiliconFlowModel.
	Model string
	// Dims is the expected vector width; <=0 means DefaultDims.
	Dims int
	// Batch is the inputs per request; <=0 means defaultBatch.
	Batch int
	// MaxChars truncates each input; <=0 means defaultMaxChars.
	MaxChars int
	// Timeout bounds one request; <=0 selects defaultTimeout.
	Timeout time.Duration
	// HTTP is optional (tests); nil uses a default client.
	HTTP *http.Client
}

func (s *SiliconFlow) baseURL() string {
	if s.BaseURL == "" {
		return DefaultSiliconFlowURL
	}
	return strings.TrimRight(s.BaseURL, "/")
}

func (s *SiliconFlow) model() string {
	if s.Model == "" {
		return DefaultSiliconFlowModel
	}
	return s.Model
}

func (s *SiliconFlow) batch() int {
	if s.Batch > 0 {
		return s.Batch
	}
	return defaultBatch
}

func (s *SiliconFlow) maxChars() int {
	if s.MaxChars > 0 {
		return s.MaxChars
	}
	return defaultMaxChars
}

func (s *SiliconFlow) client() *http.Client {
	if s.HTTP != nil {
		return s.HTTP
	}
	timeout := s.Timeout
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	return &http.Client{Timeout: timeout}
}

// Dimensions implements Embedder.
func (s *SiliconFlow) Dimensions() int {
	if s.Dims > 0 {
		return s.Dims
	}
	return DefaultDims
}

// Name implements Embedder.
func (s *SiliconFlow) Name() string {
	return "siliconflow/" + s.model()
}

type embeddingsRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
}

type embeddingsResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error"`
	// Some OpenAI-compatible servers report failures as a bare string.
	Message string `json:"message"`
}

// Embed implements Embedder.
//
// Inputs are batched, and every batch must succeed: a partial result would put
// an index into a state where some chunks are invisible to dense retrieval with
// nothing recording which. The error names the failing batch so the caller can
// retry the whole call.
func (s *SiliconFlow) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if s.APIKey == "" {
		return nil, fmt.Errorf("embed: no API key configured")
	}

	size := s.batch()
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += size {
		end := start + size
		if end > len(texts) {
			end = len(texts)
		}
		vectors, err := s.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, fmt.Errorf("embed: inputs %d-%d: %w", start, end-1, err)
		}
		out = append(out, vectors...)
	}

	if len(out) != len(texts) {
		return nil, fmt.Errorf("embed: got %d vectors for %d inputs", len(out), len(texts))
	}
	return out, nil
}

func (s *SiliconFlow) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	inputs := make([]string, len(texts))
	for i, text := range texts {
		inputs[i] = truncate(text, s.maxChars())
	}

	payload, err := json.Marshal(embeddingsRequest{
		Model:          s.model(),
		Input:          inputs,
		EncodingFormat: "float",
	})
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL()+"/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+s.APIKey)

	response, err := s.client().Do(request)
	if err != nil {
		return nil, fmt.Errorf("call endpoint: %w", err)
	}
	defer response.Body.Close()

	body, err := io.ReadAll(io.LimitReader(response.Body, 64<<20))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var decoded embeddingsResponse
	if jsonErr := json.Unmarshal(body, &decoded); jsonErr != nil {
		return nil, fmt.Errorf("HTTP %d: unreadable response: %s", response.StatusCode, snippet(body))
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", response.StatusCode, apiError(decoded))
	}
	if decoded.Error != nil {
		return nil, fmt.Errorf("HTTP %d: %s", response.StatusCode, apiError(decoded))
	}
	if len(decoded.Data) != len(texts) {
		return nil, fmt.Errorf("HTTP %d: %d vectors for %d inputs", response.StatusCode, len(decoded.Data), len(texts))
	}

	// The endpoint is not required to preserve input order, so reorder by the
	// index it reports rather than trusting the array position.
	sort.Slice(decoded.Data, func(a, b int) bool { return decoded.Data[a].Index < decoded.Data[b].Index })

	want := s.Dimensions()
	vectors := make([][]float32, len(decoded.Data))
	for i, item := range decoded.Data {
		if len(item.Embedding) != want {
			return nil, fmt.Errorf("dimension mismatch: %s returned %d values, expected %d",
				s.model(), len(item.Embedding), want)
		}
		vectors[i] = item.Embedding
	}
	return vectors, nil
}

func apiError(decoded embeddingsResponse) string {
	if decoded.Error != nil && decoded.Error.Message != "" {
		return decoded.Error.Message
	}
	if decoded.Message != "" {
		return decoded.Message
	}
	return "no message"
}

func snippet(body []byte) string {
	const limit = 300
	text := strings.TrimSpace(string(body))
	if len(text) > limit {
		return text[:limit] + "…"
	}
	return text
}

// truncate cuts text to maxChars runes, never splitting a multi-byte character.
func truncate(text string, maxChars int) string {
	if maxChars <= 0 {
		return text
	}
	runes := []rune(text)
	if len(runes) <= maxChars {
		return text
	}
	return string(runes[:maxChars])
}
