// Package kb holds the knowledge-base registry: which bases exist, what they
// are called, and where each one's index lives.
//
// A knowledge base is a named, isolated index — its own chunks, its own vectors
// and its own Qdrant collection. This package only tracks the list; the index
// itself belongs to internal/store and the registry never opens one.
//
// The registry is stored beside the indices rather than inside them, because a
// base has to exist before it has an index: creating one and adding the first
// document are separate steps, and a registry living in the index file would
// have nowhere to record an empty base.
package kb

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Base is one knowledge base.
type Base struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`

	// DocCount and ChunkCount are a cached summary, refreshed whenever the base
	// is written. Listing bases therefore loads no index at all — with several
	// bases, parsing every index file to draw a menu would make opening the app
	// cost grow with the number of bases rather than with the one in use.
	//
	// The cost of caching is that they can drift if an index is edited outside
	// the kernel. `documents` reports the live figures for the base it is asked
	// about, so a discrepancy is visible instead of silently authoritative.
	DocCount   int `json:"doc_count"`
	ChunkCount int `json:"chunk_count"`
}

// Registry is the persisted list of knowledge bases.
type Registry struct {
	mu   sync.Mutex
	path string // the registry file
	root string // directory holding <id>/index.json

	bases []Base
}

// Open loads the registry, creating it on first run.
//
// legacyPath, when it names an existing file, is adopted as the first base's
// index. That file is the single-index layout the kernel used before knowledge
// bases existed, and adopting it is what keeps an upgrade from looking like data
// loss. The original is copied, not moved: if anything about the adoption turns
// out to be wrong, the old file is still there.
func Open(path, root, legacyPath string) (*Registry, error) {
	r := &Registry{path: path, root: root}

	raw, err := os.ReadFile(path)
	switch {
	case err == nil:
		var payload struct {
			Bases []Base `json:"bases"`
		}
		if err := json.Unmarshal(raw, &payload); err != nil {
			// A corrupt registry is worth stopping for. Continuing would create
			// a second registry over the top of the first and orphan every index
			// the surviving bases point at.
			return nil, fmt.Errorf("kb: %s is not readable (%w); move it aside to start over", path, err)
		}
		r.bases = payload.Bases
		return r, nil
	case !os.IsNotExist(err):
		return nil, fmt.Errorf("kb: read %s: %w", path, err)
	}

	// First run. Always has a base, so every caller can assume one exists and
	// nothing has to handle "no knowledge base yet" as a special case.
	base := Base{ID: mustID(), Name: "默认知识库", CreatedAt: time.Now()}
	r.bases = []Base{base}

	if migrated, err := r.adopt(legacyPath, base.ID); err != nil {
		return nil, err
	} else if migrated {
		fmt.Fprintf(os.Stderr, "kb: adopted the existing index at %s as %q\n", legacyPath, base.Name)
	}

	if err := r.save(); err != nil {
		return nil, err
	}
	return r, nil
}

// adopt copies a pre-knowledge-base index into the new base's directory.
func (r *Registry) adopt(legacyPath, id string) (bool, error) {
	if legacyPath == "" {
		return false, nil
	}
	source, err := os.Open(legacyPath)
	if err != nil {
		// Nothing to adopt is the normal case on a machine that never ran the
		// previous layout.
		return false, nil
	}
	defer func() { _ = source.Close() }()

	target := r.DataPath(id)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return false, fmt.Errorf("kb: %w", err)
	}
	destination, err := os.Create(target)
	if err != nil {
		return false, fmt.Errorf("kb: %w", err)
	}
	defer func() { _ = destination.Close() }()

	if _, err := io.Copy(destination, source); err != nil {
		return false, fmt.Errorf("kb: adopting %s: %w", legacyPath, err)
	}
	return true, nil
}

// Dir is the directory holding one base's files.
func (r *Registry) Dir(id string) string { return filepath.Join(r.root, id) }

// DataPath is one base's index file.
func (r *Registry) DataPath(id string) string {
	return filepath.Join(r.Dir(id), "index.json")
}

// Collection is the Qdrant collection name for one base.
//
// One collection per base is what keeps bases apart. A shared collection would
// need a payload filter on every query and every delete, and the first call site
// that forgot one would return another base's passages — a leak that no test
// would catch because the results would still look plausible.
func (r *Registry) Collection(id string) string { return "freerag_" + id }

// List returns the bases in creation order.
func (r *Registry) List() []Base {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Base(nil), r.bases...)
}

// Get looks one up by id.
func (r *Registry) Get(id string) (Base, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, base := range r.bases {
		if base.ID == id {
			return base, true
		}
	}
	return Base{}, false
}

// Resolve turns a possibly-empty id into a base.
//
// An empty id means the first base. That keeps every existing caller working —
// the acceptance scripts and the CLI never learned about knowledge bases — while
// a caller that does pass an id gets exactly that base, never a silent default.
func (r *Registry) Resolve(id string) (Base, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if id == "" {
		if len(r.bases) == 0 {
			return Base{}, fmt.Errorf("kb: no knowledge base exists")
		}
		return r.bases[0], nil
	}
	for _, base := range r.bases {
		if base.ID == id {
			return base, nil
		}
	}
	return Base{}, fmt.Errorf("kb: no knowledge base %q", id)
}

// Create adds a base and returns it.
func (r *Registry) Create(name string) (Base, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Base{}, fmt.Errorf("kb: a name is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.findByName(name); ok {
		return Base{}, fmt.Errorf("kb: %q already exists", existing.Name)
	}

	base := Base{ID: mustID(), Name: name, CreatedAt: time.Now()}
	r.bases = append(r.bases, base)
	return base, r.save()
}

// Rename changes a base's name, leaving its id, index and collection alone.
func (r *Registry) Rename(id, name string) (Base, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Base{}, fmt.Errorf("kb: a name is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if existing, ok := r.findByName(name); ok && existing.ID != id {
		return Base{}, fmt.Errorf("kb: %q already exists", existing.Name)
	}
	for i := range r.bases {
		if r.bases[i].ID != id {
			continue
		}
		r.bases[i].Name = name
		return r.bases[i], r.save()
	}
	return Base{}, fmt.Errorf("kb: no knowledge base %q", id)
}

// Remove drops a base from the registry and returns it.
//
// The index file and the Qdrant collection are left to the caller: this package
// does not know about either, and deleting a directory it does not own would be
// the wrong thing to do from a bookkeeping type.
func (r *Registry) Remove(id string) (Base, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.bases) == 1 {
		// The last base is load-bearing: every call that omits an id resolves to
		// the first one, and the UI has nothing to bind a session to without it.
		return Base{}, fmt.Errorf("kb: the last knowledge base cannot be deleted")
	}
	for i := range r.bases {
		if r.bases[i].ID != id {
			continue
		}
		removed := r.bases[i]
		r.bases = append(r.bases[:i], r.bases[i+1:]...)
		return removed, r.save()
	}
	return Base{}, fmt.Errorf("kb: no knowledge base %q", id)
}

// SetCounts records the summary shown in the base list.
//
// A failure to persist is returned rather than swallowed, but callers treat it
// as a warning: the counts are a convenience, and losing them must not fail the
// indexing run that produced them.
func (r *Registry) SetCounts(id string, docs, chunks int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i := range r.bases {
		if r.bases[i].ID != id {
			continue
		}
		if r.bases[i].DocCount == docs && r.bases[i].ChunkCount == chunks {
			return nil
		}
		r.bases[i].DocCount = docs
		r.bases[i].ChunkCount = chunks
		return r.save()
	}
	return fmt.Errorf("kb: no knowledge base %q", id)
}

// findByName is a case-insensitive lookup; the caller holds the lock.
func (r *Registry) findByName(name string) (Base, bool) {
	for _, base := range r.bases {
		if strings.EqualFold(base.Name, name) {
			return base, true
		}
	}
	return Base{}, false
}

// save writes the registry atomically; the caller holds the lock.
func (r *Registry) save() error {
	payload, err := json.MarshalIndent(struct {
		Bases []Base `json:"bases"`
	}{Bases: r.bases}, "", "  ")
	if err != nil {
		return fmt.Errorf("kb: marshal: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(r.path), 0o755); err != nil {
		return fmt.Errorf("kb: %w", err)
	}

	// Written to a temporary file and renamed. This file is the only record of
	// which bases exist; a crash midway through writing it in place would lose
	// every one of them, and the indices they point at would still be on disk
	// with nothing left that refers to them.
	temporary := r.path + ".tmp"
	if err := os.WriteFile(temporary, payload, 0o644); err != nil {
		return fmt.Errorf("kb: %w", err)
	}
	if err := os.Rename(temporary, r.path); err != nil {
		return fmt.Errorf("kb: %w", err)
	}
	return nil
}

// mustID returns an identifier safe for a directory name and a Qdrant
// collection name.
//
// Random rather than derived from the name: names are editable, and an id that
// followed the name would make renaming a base move its index directory and
// rename its collection — two chances to lose data for a cosmetic change.
func mustID() string {
	raw := make([]byte, 8)
	if _, err := rand.Read(raw); err != nil {
		// crypto/rand does not fail in practice, and there is no sensible way to
		// carry on without an identity for the base.
		panic("kb: no randomness available: " + err.Error())
	}
	return hex.EncodeToString(raw)
}
