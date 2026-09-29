// Package embed turns text into vectors for dense retrieval.
//
// The shipping target is a local BGE-M3 (docs/plan.md §4). A hosted
// OpenAI-compatible endpoint is wired first because it needs no 2.3 GB model
// download, and the Embedder interface is what keeps the local implementation a
// drop-in replacement rather than a rewrite of the retrieval layer.
package embed

import (
	"context"
	"fmt"
	"os"
	"strconv"
)

// Embedder turns text into vectors.
type Embedder interface {
	// Embed returns one vector per input, in the order the inputs were given.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimensions is the vector width; the store validates against it so an
	// index built by one model is never queried with another's vectors.
	Dimensions() int
	// Name identifies provider and model, for GET /version and the index file.
	Name() string
}

// EmbedOne is the single-text convenience used on the query path.
func EmbedOne(ctx context.Context, e Embedder, text string) ([]float32, error) {
	vectors, err := e.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("embed: expected 1 vector, got %d", len(vectors))
	}
	return vectors[0], nil
}

// FromEnv builds the configured embedder.
//
// A missing configuration returns (nil, nil): running keyword-only is a valid
// degraded mode, not a failure, so the caller should not treat it as one. A
// *broken* configuration returns an error, because silently falling back would
// leave the operator believing dense retrieval is on.
func FromEnv() (Embedder, error) {
	switch provider := os.Getenv("FREERAG_EMBED_PROVIDER"); provider {
	case "", "none":
		return nil, nil
	case "siliconflow":
		key := os.Getenv("FREERAG_SILICONFLOW_KEY")
		if key == "" {
			return nil, fmt.Errorf("embed: FREERAG_EMBED_PROVIDER=siliconflow needs FREERAG_SILICONFLOW_KEY")
		}
		return &SiliconFlow{
			BaseURL: os.Getenv("FREERAG_SILICONFLOW_URL"),
			APIKey:  key,
			Model:   os.Getenv("FREERAG_EMBED_MODEL"),
			Dims:    envInt("FREERAG_EMBED_DIMS", 0),
		}, nil
	default:
		return nil, fmt.Errorf("embed: unknown provider %q", provider)
	}
}

func envInt(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}
