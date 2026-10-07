package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"freerag/internal/ipc"
	"freerag/internal/tree"
)

// The tree channel's kernel surface.
//
// Two methods, and one hook. `tree.index` builds every document's structure out
// of the chunks already in the store; `tree.search` answers through that
// structure and through the keyword safety net; and indexing a document rebuilds
// its tree on the way past, which is what keeps the two from drifting apart.
// Nothing here needs the sidecar: the tree is built from chunks, not from the
// file, so a base indexed months ago can be structured without re-parsing it.

// treePath is where a knowledge base keeps its trees: beside index.json.
func treePath(live *kbRuntime) string {
	return filepath.Join(filepath.Dir(live.dataPath), "tree.json")
}

// treesFor opens a base's tree index once per process.
//
// The runtime half is rebuilt from the store every time, because that is where
// the text lives: tree.json holds structure and nothing else, deliberately — a
// file that also held every passage would need its own invalidation rules, and
// the first rule missed would score yesterday's text.
func (k *kernel) treesFor(live *kbRuntime) (*tree.Index, error) {
	live.treesMu.Lock()
	defer live.treesMu.Unlock()

	if live.trees != nil {
		return live.trees, nil
	}
	ix, err := tree.Open(treePath(live))
	if err != nil {
		return nil, err
	}
	for _, docID := range ix.Docs() {
		if ix.Attached(docID) {
			continue
		}
		doc := ix.Tree(docID)
		chunks := live.store.List(docID, 0, 0, 0)
		if len(chunks) == 0 {
			// The chunks are gone but the tree file still knows the document.
			// Keeping the entry would let a search descend into a document with
			// no text behind it, which is worse than admitting none exists.
			ix.Remove(docID)
			continue
		}
		// The tree survives from disk; its text does not, so it is re-attached
		// rather than rebuilt — rebuilding here would hide a document whose tree
		// file is stale behind a freshly computed one.
		_ = doc
		ix.Put(doc, chunks)
	}
	live.trees = ix
	return ix, nil
}

// rebuildTree rebuilds one document's structure from the chunks it owns.
//
// persist=false leaves the result in memory, because tree.json holds every
// document and rewriting it once per file turns a batch into a quadratic write.
// The caller that owns the batch saves once at the end; anything else saves as it
// goes, so a tree is never lost to a crash that killed the rest of the batch.
func (k *kernel) rebuildTree(live *kbRuntime, docID, sourceFile string, persist bool) (int, error) {
	ix, err := k.treesFor(live)
	if err != nil {
		return 0, err
	}
	if sourceFile == "" {
		if record, ok := live.store.DocumentByDocID(docID); ok {
			sourceFile = record.SourceFile
		}
	}
	chunks := live.store.List(docID, 0, 0, 0)
	if len(chunks) == 0 {
		ix.Remove(docID)
		if !persist {
			return 0, nil
		}
		return 0, ix.Save()
	}
	ix.BuildAndPut(docID, sourceFile, chunks)
	if !persist {
		return len(chunks), nil
	}
	return len(chunks), ix.Save()
}

// handleTreeStatus reports which documents have a tree, and where each one's
// hierarchy came from.
//
// Distinct from `tree.index` because it changes nothing: an existing corpus gets
// its trees at index time, but a corpus indexed before this channel existed has
// none, and the UI has to be able to say so. Without this the only symptom would
// be search quietly falling back to flat BM25 — indistinguishable from the tree
// having found nothing.
func (k *kernel) handleTreeStatus(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		KB string `json:"kb"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	live.indexing.RLock()
	defer live.indexing.RUnlock()

	ix, err := k.treesFor(live)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}

	treeDocs := map[string]*tree.DocTree{}
	for _, docID := range ix.Docs() {
		treeDocs[docID] = ix.Tree(docID)
	}

	docs := []map[string]any{}
	levels := map[string]int{}
	for _, docID := range live.store.Documents() {
		doc, ok := treeDocs[docID]
		if !ok || doc == nil {
			// Named rather than omitted: a document without a tree is the single
			// most useful thing this report can surface.
			docs = append(docs, map[string]any{
				"doc_id": docID, "has_tree": false, "nodes": 0, "anchors": 0,
				"levels_from": "", "attached": false,
			})
			continue
		}
		levels[doc.LevelsFrom]++
		docs = append(docs, map[string]any{
			"doc_id":      docID,
			"has_tree":    true,
			"nodes":       len(doc.Nodes),
			"anchors":     len(doc.Anchors),
			"levels_from": doc.LevelsFrom,
			"attached":    ix.Attached(docID),
			"policy":      doc.Policy,
		})
	}
	return withKB(map[string]any{
		"documents": len(docs),
		"docs":      docs,
		"levels":    levels,
		"policy":    tree.Policy,
		"data_path": treePath(live),
	}, live), nil
}

// handleTreeIndex (re)builds the tree index for one base.
//
// It reads no document files: everything it needs is in the index already, so it
// is the command that brings a corpus up to date after this channel was added,
// and the one to run after editing the level heuristics.
func (k *kernel) handleTreeIndex(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		KB    string `json:"kb"`
		DocID string `json:"doc_id"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	// Exclusive against asking, like every other index write.
	live.indexing.Lock()
	defer live.indexing.Unlock()

	if _, err := k.treesFor(live); err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}

	targets := live.store.Documents()
	if params.DocID != "" {
		targets = []string{params.DocID}
	}

	built := 0
	chunks := 0
	levels := map[string]int{}
	failed := []string{}
	for _, docID := range targets {
		count, err := k.rebuildTree(live, docID, "", false)
		if err != nil {
			failed = append(failed, docID)
			log.Printf("warning: tree rebuild failed for %s: %v", docID, err)
			continue
		}
		if count == 0 {
			continue
		}
		built++
		chunks += count
		if doc := live.trees.Tree(docID); doc != nil {
			levels[doc.LevelsFrom]++
		}
	}

	reply := map[string]any{
		"built":     built,
		"documents": len(targets),
		"chunks":    chunks,
		"levels":    levels,
		"policy":    tree.Policy,
		"data_path": treePath(live),
	}
	if len(failed) > 0 {
		reply["failed"] = failed
	}
	// One write for the whole run: every document rebuilt above is already in
	// memory, and this is the earliest point at which the result is complete.
	if saveErr := live.trees.Save(); saveErr != nil {
		log.Printf("warning: could not persist the tree index: %v", saveErr)
		reply["save_error"] = saveErr.Error()
	}
	return withKB(reply, live), nil
}

// handleTreeSearch answers through the tree channel.
//
// The reply keeps both channels' ranks per hit, so a result can be questioned:
// `source` says whether the tree, the keyword index, or both found it, and `path`
// names the sections above it when the tree did. A hit with an empty path is a
// keyword match and says so.
func (k *kernel) handleTreeSearch(_ context.Context, raw json.RawMessage) (any, *ipc.Error) {
	var params struct {
		Query string `json:"query"`
		KB    string `json:"kb"`
		Limit int    `json:"limit"`
		// MaxDocs and Beam override the defaults for evaluation sweeps.
		MaxDocs int `json:"max_docs"`
		Beam    int `json:"beam"`
		// NoFallback drops the BM25 safety net. It exists so an evaluation can
		// measure what the tree alone contributes — it is not a production
		// setting, and the default (false) is the supported one.
		NoFallback bool `json:"no_fallback"`
	}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: fmt.Sprintf("invalid params: %v", err)}
		}
	}
	params.Query = strings.TrimSpace(params.Query)
	if params.Query == "" {
		return nil, &ipc.Error{Code: ipc.CodeInvalidParams, Message: "params.query is required"}
	}

	live, kbErr := k.kbFor(params.KB)
	if kbErr != nil {
		return nil, kbErr
	}
	live.indexing.RLock()
	defer live.indexing.RUnlock()

	ix, err := k.treesFor(live)
	if err != nil {
		return nil, &ipc.Error{Code: ipc.CodeInternalError, Message: err.Error()}
	}
	if ix.Len() == 0 {
		return nil, &ipc.Error{
			Code:    ipc.CodeInvalidParams,
			Message: "this knowledge base has no tree; run tree.index first",
		}
	}

	opts := tree.DefaultOptions()
	opts.Fallback = !params.NoFallback
	if params.Limit > 0 {
		opts.Limit = params.Limit
	}
	if params.MaxDocs > 0 {
		opts.MaxDocs = params.MaxDocs
	}
	if params.Beam > 0 {
		opts.BeamWidth = params.Beam
	}

	result := tree.Search(ix, live.store, params.Query, opts)
	return withKB(map[string]any{
		"query": result.Query,
		"hits":  result.Hits,
		"count": len(result.Hits),
		"trace": result.Trace,
	}, live), nil
}

// persistTrees writes the tree index once, for a caller that rebuilt several
// documents without saving between them.
//
// Nothing to save when no tree was ever opened: a batch that indexed nothing, or
// that failed before reaching its first parse, has no file to write — and
// creating an empty tree.json would advertise a base with structures nobody built.
func (k *kernel) persistTrees(live *kbRuntime) {
	live.treesMu.Lock()
	ix := live.trees
	live.treesMu.Unlock()
	if ix == nil {
		return
	}
	if err := ix.Save(); err != nil {
		log.Printf("warning: could not persist the tree index: %v", err)
	}
}

// forgetTree drops a document's tree when its chunks are forgotten.
//
// Called rather than left to the next `tree.index`: a tree outliving the document
// it describes would let a search descend to coordinates whose text no longer
// exists, and nothing downstream could tell that apart from an empty section.
func (k *kernel) forgetTree(live *kbRuntime, docID string) {
	if _, err := k.treesFor(live); err != nil {
		log.Printf("warning: could not open the tree index (%v); %s's tree was left in place", err, docID)
		return
	}
	live.trees.Remove(docID)
	if err := live.trees.Save(); err != nil {
		log.Printf("warning: could not persist the tree index: %v", err)
	}
}
