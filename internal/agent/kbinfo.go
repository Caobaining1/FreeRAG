package agent

import (
	"sort"
	"strings"

	"freerag/internal/store"
)

// kbinfo is the store of everything retrieval has returned for one question.
//
// Named after the pool RAGFlow keeps for the same purpose, and introduced for
// the same reason: the sufficiency decision is "is this pool enough to answer
// the question", so the pool is what the decision has to see. It used to be
// asked about a DRAFT instead — one full generation per round, written only to
// be judged, measured at 60 s on the reference machine, and thrown away. The
// answer is gone; the pool accumulates and the checker reads all of it.
//
// It owns the pool and the two indexes that keep it a set, which were
// previously a slice and a set passed around the loop as separate parameters:
//
//   - the hits in arrival order, because a citation's [n] refers to that order;
//   - `seen`, the identities already in the pool, so a repeat tool call adds
//     nothing;
//   - `content`, the normalized texts already in the pool, so the same passage
//     arriving under two chunk ids does not spend the checker's budget twice
//     (see contentKey for why that is per document).
//
// It also owns the run's document SCOPE, which is where a metadata_search's
// document ids live. RAGFlow returns doc ids as a handle the model is expected
// to spend on its next call (internal/rag/agentic-rag/runtime/tool_executor.go's
// metadataSearch returns `doc_ids` and guards a second call with
// `MetadataSearchUsed bool`, and its prompt tells the model to pass them as
// doc_scope) — which relies on the model remembering. Here the ids are kept and
// applied by the loop instead, so a filter narrows the corpus for every call
// that follows whether or not the model mentions it again. A scope that exists
// only in a prompt is a scope a small model can drop.
type kbinfo struct {
	hits    []store.Hit
	seen    map[string]bool
	content map[string]bool

	// scope is the set of documents retrieval may draw from; empty means no
	// filter is active. scopeNote is the filter that produced it, kept for the
	// trace and for the answer prompt, because a narrow search that says
	// "nothing found" owes the reader the fact that it was narrow.
	scope     map[string]bool
	scopeNote string
	// outOfScope counts passages dropped by the scope, so a call can report
	// "3 passage(s) were outside the scope" instead of silently returning less.
	outOfScope int
}

func newKBInfo() *kbinfo {
	return &kbinfo{seen: map[string]bool{}, content: map[string]bool{}}
}

// setScope narrows retrieval to docIDs and records the filter behind it.
//
// Replacing an existing scope rather than intersecting it: a new filter is a
// new question about which documents matter, and intersecting two filters the
// model issued in different rounds would leave it unable to widen the search
// again except by naming every document.
func (k *kbinfo) setScope(docIDs []string, note string) {
	if k == nil {
		return
	}
	k.scope = make(map[string]bool, len(docIDs))
	for _, id := range docIDs {
		if id != "" {
			k.scope[id] = true
		}
	}
	k.scopeNote = note
}

// clearScope lifts the document scope, restoring unrestricted retrieval.
func (k *kbinfo) clearScope() {
	if k == nil {
		return
	}
	k.scope = nil
	k.scopeNote = ""
}

// scoped reports whether a document scope is active.
func (k *kbinfo) scoped() bool { return k != nil && len(k.scope) > 0 }

// inScope reports whether a document may contribute to the pool. With no scope
// every document may, which is the behaviour of a run that never filtered.
func (k *kbinfo) inScope(docID string) bool {
	if !k.scoped() {
		return true
	}
	return k.scope[docID]
}

// filter is the scope in the form retrieval takes, so the restriction happens
// inside the search rather than to its results (see store.Filter).
//
// The pool still checks inScope when hits arrive. That is not redundancy: this
// is what makes the scope effective, and inScope is what catches a tool that
// never consulted it — list_chunks names a document directly, and the metadata
// search itself runs unscoped.
func (k *kbinfo) filter() store.Filter {
	if !k.scoped() {
		return store.Filter{}
	}
	return store.NewFilter(k.scopeIDs())
}

// scopeIDs returns the scope's documents in a stable order, for a trace line or
// a test.
func (k *kbinfo) scopeIDs() []string {
	if !k.scoped() {
		return nil
	}
	ids := make([]string, 0, len(k.scope))
	for id := range k.scope {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// hitKey identifies a passage for deduplication. Two calls can reach the same
// chunk by different routes — a keyword match and a dense one — and the pool is
// a set of passages, not of queries.
//
// A hit with no chunk id falls back to its text, because an empty id would make
// every such hit from one document share a single key: the pool would then keep
// one passage where retrieval found several, which is a silent loss rather than
// a deduplication.
func hitKey(hit store.Hit) string {
	if hit.Chunk.ChunkID == "" {
		return hit.Chunk.DocID + "\x00" + "text\x00" + normalizeText(hit.Chunk.Text)
	}
	return hit.Chunk.DocID + "\x00" + hit.Chunk.ChunkID
}

// contentKey identifies a passage by what it SAYS, within one document.
//
// The same text legitimately arrives twice from one file: a caption exists both
// as its own Caption block and again inside the merged FigureWithCaption block
// that absorbed it, and overlapping chunks repeat the sentences between them.
// Those duplicates spend the checker's budget and read as corroboration that is
// not there.
//
// Scoped to one document on purpose. The same sentence in two documents is two
// citations, not a duplicate, and dropping the second would lose the link.
func contentKey(hit store.Hit) string {
	return hit.Chunk.DocID + "\x00" + normalizeText(hit.Chunk.Text)
}

// normalizeText folds the differences that do not change what a passage says,
// so two copies of a caption separated by different whitespace still match.
func normalizeText(text string) string {
	return strings.ToLower(strings.Join(strings.Fields(text), " "))
}

// add merges freshly retrieved hits and returns how many were new.
//
// This is where the document scope is enforced, and it is the only place it
// needs to be: every tool's results reach the pool through here, so one check
// covers hybrid_search, grep_search, list_chunks and metadata_search alike.
func (k *kbinfo) add(hits []store.Hit) int {
	fresh := 0
	for _, hit := range hits {
		// Counted rather than silently dropped: a call that returns fewer
		// passages than it found is a fact the caller reports.
		if !k.inScope(hit.Chunk.DocID) {
			k.outOfScope++
			continue
		}
		key := hitKey(hit)
		if k.seen[key] {
			continue
		}
		// The content check only earns its cost for a passage that is new by
		// identity, which is the common case for a repeat retrieval.
		if strings.TrimSpace(hit.Chunk.Text) != "" {
			content := contentKey(hit)
			if k.content[content] {
				continue
			}
			k.content[content] = true
		}
		k.seen[key] = true
		k.hits = append(k.hits, hit)
		fresh++
	}
	return fresh
}

func (k *kbinfo) len() int {
	if k == nil {
		return 0
	}
	return len(k.hits)
}

func (k *kbinfo) empty() bool { return k.len() == 0 }

// pool is the passage list, in the order it was gathered. Returned as-is rather
// than copied: the loop is the only writer, and it stops writing before anyone
// reads.
func (k *kbinfo) pool() []store.Hit {
	if k == nil {
		return nil
	}
	return k.hits
}
