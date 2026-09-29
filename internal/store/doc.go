package store

import (
	"sort"
	"time"
)

// DocumentRecord is what the index remembers about one indexed document.
//
// It exists so a re-index can be skipped, and it is persisted inside the index
// file rather than beside it: a manifest in a separate file could outlive the
// chunks it describes, and then a reset index would silently report documents
// as "already indexed" that it no longer holds.
type DocumentRecord struct {
	// MD5 is the content fingerprint of the source file. This, not the path, is
	// the document's identity: a renamed or moved file is the same document,
	// and a file restored with its original mtime is not unchanged.
	MD5 string `json:"md5"`
	// DocID is the identity used by chunks, list_chunks and metadata_search.
	DocID string `json:"doc_id"`
	// SourceFile and Path describe where it came from, for reporting only.
	SourceFile string `json:"source_file,omitempty"`
	Path       string `json:"path,omitempty"`
	// ChunkCount is how many chunks the document contributed. It is checked
	// against the store on every skip decision, which is what catches an index
	// that was rebuilt without this document.
	ChunkCount int `json:"chunk_count"`
	PageCount  int `json:"page_count,omitempty"`
	// Pipeline fingerprints the parse+chunk rules that produced the chunks.
	// Without it, changing the chunker would keep serving chunks built under
	// the old rules while looking like a cache hit — worse than no skip at all.
	Pipeline string `json:"pipeline"`
	// IndexedAt is when the document was last (re)indexed.
	IndexedAt time.Time `json:"indexed_at"`
}

// ChunkCountIn reports how many chunks the store holds for one document.
func (s *Store) ChunkCountIn(docID string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	count := 0
	for _, chunk := range s.chunks {
		if chunk.DocID == docID {
			count++
		}
	}
	return count
}

// Document returns the record for a content fingerprint.
func (s *Store) Document(md5 string) (DocumentRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.documents[md5]
	return record, ok
}

// DocumentByDocID returns the record for a document id.
//
// More than one fingerprint can map to one DocID only if a document was edited,
// in which case the newest wins — the older one no longer describes what the
// chunks say.
func (s *Store) DocumentByDocID(docID string) (DocumentRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var newest DocumentRecord
	found := false
	for _, record := range s.documents {
		if record.DocID != docID {
			continue
		}
		if !found || record.IndexedAt.After(newest.IndexedAt) {
			newest, found = record, true
		}
	}
	return newest, found
}

// PutDocument records (or replaces) a document's manifest entry.
func (s *Store) PutDocument(record DocumentRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.documents == nil {
		s.documents = map[string]DocumentRecord{}
	}
	s.documents[record.MD5] = record
}

// DocumentRecords returns every record, ordered by document id then fingerprint
// so the output is stable across runs.
//
// Named distinctly from Documents, which lists document ids for list_chunks.
func (s *Store) DocumentRecords() []DocumentRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]DocumentRecord, 0, len(s.documents))
	for _, record := range s.documents {
		out = append(out, record)
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].DocID != out[b].DocID {
			return out[a].DocID < out[b].DocID
		}
		return out[a].MD5 < out[b].MD5
	})
	return out
}

// RemoveDoc drops every chunk of one document, along with its manifest entries.
// It returns how many chunks were removed.
//
// This is what makes re-indexing a *changed* document correct. Chunk ids are
// derived from position, not content, so an edited file yields the same ids at
// the same offsets: adding the new chunks without removing the old ones would
// leave the previous text in place, silently, under an unchanged id.
//
// Removal cannot be selective, so it rebuilds the per-chunk slices. That is
// O(N) over the corpus, which is the price of keeping the term index and the
// dedup map consistent — an incremental deletion that left either of them
// stale would corrupt ranking in ways nothing downstream could detect.
func (s *Store) RemoveDoc(docID string) int {
	s.mu.Lock()

	chunks := make([]Chunk, 0, len(s.chunks))
	vectors := make([][]float32, 0, len(s.chunks))
	terms := make([]map[string]int, 0, len(s.chunks))
	lengths := make([]int, 0, len(s.chunks))
	seen := make(map[string]int, len(s.seen))

	removed := 0
	for i, chunk := range s.chunks {
		if chunk.DocID == docID {
			removed++
			// Decrement rather than rebuild the whole document frequency
			// table: terms are counted once per chunk, so subtracting here
			// keeps df exactly in step.
			for term := range s.terms[i] {
				s.df[term]--
				if s.df[term] <= 0 {
					delete(s.df, term)
				}
			}
			continue
		}

		seen[dedupKey(chunk)] = len(chunks)
		chunks = append(chunks, chunk)
		if i < len(s.vectors) {
			vectors = append(vectors, s.vectors[i])
		} else {
			vectors = append(vectors, nil)
		}
		terms = append(terms, s.terms[i])
		lengths = append(lengths, s.lengths[i])
	}

	s.chunks, s.vectors, s.terms, s.lengths, s.seen = chunks, vectors, terms, lengths, seen
	for md5, record := range s.documents {
		if record.DocID == docID {
			delete(s.documents, md5)
		}
	}

	index := s.dense
	s.mu.Unlock()

	// Vectors that are no longer referenced still sit in an external dense
	// index. They are harmless for ranking — resolve() drops matches the store
	// does not own — but they waste space, so tell the index to drop them.
	// Done outside the lock: it is a network call.
	if index != nil && removed > 0 {
		if err := index.Delete(docID); err != nil {
			s.loggerOr().Printf("warning: dense index %s could not drop %q (%v); its vectors are now orphaned",
				index.Backend(), docID, err)
		}
	}
	return removed
}
