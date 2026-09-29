package store

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// GrepHit is one grep_search result: the chunk plus where inside it matched.
type GrepHit struct {
	Chunk Chunk  `json:"chunk"`
	Line  string `json:"line"`
	Count int    `json:"count"`
}

// Grep scans chunk text for pattern.
//
// useRegex selects a regular expression; otherwise the match is a
// case-insensitive substring. Grep covers what BM25 and embeddings are both bad
// at — identifiers, codes and rare proper nouns, where the exact characters
// matter more than the surrounding words.
func (s *Store) Grep(pattern string, useRegex bool, limit int) ([]GrepHit, error) {
	pattern = strings.TrimSpace(pattern)
	if pattern == "" {
		return nil, nil
	}

	var re *regexp.Regexp
	if useRegex {
		compiled, err := regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid regex %q: %w", pattern, err)
		}
		re = compiled
	}
	needle := strings.ToLower(pattern)

	s.mu.RLock()
	defer s.mu.RUnlock()

	hits := make([]GrepHit, 0, 8)
	for _, chunk := range s.chunks {
		var (
			line  string
			count int
		)
		for _, candidate := range strings.Split(chunk.Text, "\n") {
			matched := 0
			if re != nil {
				matched = len(re.FindAllString(candidate, -1))
			} else {
				matched = strings.Count(strings.ToLower(candidate), needle)
			}
			if matched == 0 {
				continue
			}
			count += matched
			if line == "" {
				line = strings.TrimSpace(candidate)
			}
		}
		if count == 0 {
			continue
		}
		hits = append(hits, GrepHit{Chunk: chunk, Line: line, Count: count})
	}

	// More occurrences means the term is more central to that chunk; ties keep
	// stored order so the result is deterministic.
	sort.SliceStable(hits, func(a, b int) bool { return hits[a].Count > hits[b].Count })

	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

// List returns chunks in stored order, optionally narrowed to one document and
// one page, then paginated.
//
// This is the tool that browses a document's structure, and the one that reads a
// page back after the model's context was compressed to drafts (docs/plan.md
// §6.6 / §6.7).
func (s *Store) List(docID string, page, offset, limit int) []Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Chunk, 0, 16)
	skipped := 0
	for _, chunk := range s.chunks {
		if docID != "" && chunk.DocID != docID {
			continue
		}
		if page > 0 && chunk.PageNum != page {
			continue
		}
		if skipped < offset {
			skipped++
			continue
		}
		out = append(out, chunk)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

// MetadataFilter selects chunks by their metadata fields alone — no text
// matching, so it answers "which chunks are tables on page 3 of a.pdf".
type MetadataFilter struct {
	DocID      string `json:"doc_id,omitempty"`
	SourceFile string `json:"source_file,omitempty"`
	BlockType  string `json:"block_type,omitempty"`
	Page       int    `json:"page,omitempty"`
	Limit      int    `json:"limit,omitempty"`
}

// MetadataSearch returns chunks matching every non-empty field.
func (s *Store) MetadataSearch(filter MetadataFilter) []Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Chunk, 0, 16)
	for _, chunk := range s.chunks {
		if filter.DocID != "" && chunk.DocID != filter.DocID {
			continue
		}
		if filter.BlockType != "" && !strings.EqualFold(chunk.BlockType, filter.BlockType) {
			continue
		}
		if filter.Page > 0 && chunk.PageNum != filter.Page {
			continue
		}
		if filter.SourceFile != "" && !matchesSourceFile(chunk, filter.SourceFile) {
			continue
		}
		out = append(out, chunk)
		if filter.Limit > 0 && len(out) >= filter.Limit {
			break
		}
	}
	return out
}

// matchesSourceFile checks the chunk's doc id and its source_file metadata
// field, so callers can pass whichever they have.
func matchesSourceFile(chunk Chunk, want string) bool {
	if strings.EqualFold(chunk.DocID, want) {
		return true
	}
	if chunk.Metadata == nil {
		return false
	}
	value, ok := chunk.Metadata["source_file"].(string)
	return ok && strings.EqualFold(value, want)
}

// Documents returns the distinct document ids in stored order, so a caller can
// discover what there is to list before listing it.
func (s *Store) Documents() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	seen := map[string]bool{}
	out := make([]string, 0, 4)
	for _, chunk := range s.chunks {
		if chunk.DocID == "" || seen[chunk.DocID] {
			continue
		}
		seen[chunk.DocID] = true
		out = append(out, chunk.DocID)
	}
	return out
}
