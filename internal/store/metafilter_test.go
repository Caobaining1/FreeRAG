package store

import (
	"testing"
	"time"
)

// metaStore holds two documents, indexed on different days, so both the doc_id
// and the indexed_at half of the filter have something to discriminate.
func metaStore() *Store {
	s := New()
	s.Add([]Chunk{
		{ChunkID: "c0", DocID: "alpha.pdf", PageNum: 1, BlockType: "Text", Text: "alpha"},
		{ChunkID: "c1", DocID: "alpha.pdf", PageNum: 2, BlockType: "Table", Text: "alpha table"},
		{ChunkID: "c2", DocID: "beta.pdf", PageNum: 1, BlockType: "Text", Text: "beta"},
	})
	s.PutDocument(DocumentRecord{MD5: "m1", DocID: "alpha.pdf",
		IndexedAt: time.Date(2026, 9, 29, 14, 26, 1, 0, time.Local)})
	s.PutDocument(DocumentRecord{MD5: "m2", DocID: "beta.pdf",
		IndexedAt: time.Date(2026, 9, 20, 9, 0, 0, 0, time.Local)})
	return s
}

func one(key, op string, value any) MetadataQuery {
	return MetadataQuery{Conditions: []MetadataCondition{{Key: key, Op: op, Value: value}}}
}

func docIDs(t *testing.T, s *Store, q MetadataQuery) []string {
	t.Helper()
	return s.MetadataDocIDs(q)
}

func TestMetadataFilterOperators(t *testing.T) {
	cases := []struct {
		name string
		key  string
		op   string
		val  any
		want []string
	}{
		{name: "doc_id =", key: FieldDocID, op: OpEqual, val: "alpha.pdf", want: []string{"alpha.pdf"}},
		{name: "doc_id ≠", key: FieldDocID, op: OpNotEqual, val: "alpha.pdf", want: []string{"beta.pdf"}},
		{name: "doc_id contains", key: FieldDocID, op: OpContains, val: "PHA", want: []string{"alpha.pdf"}},
		{name: "doc_id not contains", key: FieldDocID, op: OpNotContains, val: "pha", want: []string{"beta.pdf"}},
		{name: "doc_id start with", key: FieldDocID, op: OpStartWith, val: "ALP", want: []string{"alpha.pdf"}},
		{name: "doc_id end with", key: FieldDocID, op: OpEndWith, val: ".PDF", want: []string{"alpha.pdf", "beta.pdf"}},
		{name: "doc_id in", key: FieldDocID, op: OpIn, val: []any{"beta.pdf"}, want: []string{"beta.pdf"}},
		{name: "doc_id not in", key: FieldDocID, op: OpNotIn, val: []any{"beta.pdf"}, want: []string{"alpha.pdf"}},
		{name: "doc_id empty", key: FieldDocID, op: OpEmpty, val: nil, want: nil},
		{name: "doc_id not empty", key: FieldDocID, op: OpNotEmpty, val: nil, want: []string{"alpha.pdf", "beta.pdf"}},

		// The time rule: a day is asked for with "start with", and the bare date
		// never satisfies "=", because the stored value carries a time too.
		{name: "indexed_at start with a day", key: FieldIndexedAt, op: OpStartWith,
			val: "2026-09-29", want: []string{"alpha.pdf"}},
		{name: "indexed_at = a bare day", key: FieldIndexedAt, op: OpEqual,
			val: "2026-09-29", want: nil},
		{name: "indexed_at = the full timestamp", key: FieldIndexedAt, op: OpEqual,
			val: "2026-09-29 14:26:01", want: []string{"alpha.pdf"}},
		{name: "indexed_at ≥", key: FieldIndexedAt, op: OpGreaterEqual,
			val: "2026-09-25", want: []string{"alpha.pdf"}},
		{name: "indexed_at <", key: FieldIndexedAt, op: OpLess,
			val: "2026-09-25", want: []string{"beta.pdf"}},
		{name: "indexed_at empty", key: FieldIndexedAt, op: OpEmpty, val: nil, want: nil},

		// A list operator with no list must match nothing rather than widen the
		// selection to everything.
		{name: "doc_id in without a list", key: FieldDocID, op: OpIn, val: "alpha.pdf", want: nil},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := docIDs(t, metaStore(), one(c.key, c.op, c.val))
			if len(got) != len(c.want) {
				t.Fatalf("doc ids = %v, want %v", got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Fatalf("doc ids = %v, want %v", got, c.want)
				}
			}
		})
	}
}

func TestMetadataFilterLogic(t *testing.T) {
	s := metaStore()

	// AND is the default, and it intersects.
	both := MetadataQuery{
		Logic: "and",
		Conditions: []MetadataCondition{
			{Key: FieldDocID, Op: OpStartWith, Value: "a"},
			{Key: FieldIndexedAt, Op: OpStartWith, Value: "2026-09-29"},
		},
	}
	if got := s.MetadataDocIDs(both); len(got) != 1 || got[0] != "alpha.pdf" {
		t.Fatalf("and doc ids = %v, want only alpha.pdf", got)
	}

	// A condition no document can satisfy empties an AND...
	impossible := MetadataQuery{Conditions: []MetadataCondition{
		{Key: FieldDocID, Op: OpEqual, Value: "alpha.pdf"},
		{Key: FieldIndexedAt, Op: OpStartWith, Value: "1999-01-01"},
	}}
	if got := s.MetadataDocIDs(impossible); len(got) != 0 {
		t.Fatalf("and doc ids = %v, want none", got)
	}

	// ...but with OR it only contributes nothing.
	or := MetadataQuery{
		Logic: "or",
		Conditions: []MetadataCondition{
			{Key: FieldIndexedAt, Op: OpStartWith, Value: "1999-01-01"},
			{Key: FieldDocID, Op: OpEqual, Value: "beta.pdf"},
		},
	}
	if got := s.MetadataDocIDs(or); len(got) != 1 || got[0] != "beta.pdf" {
		t.Fatalf("or doc ids = %v, want beta.pdf", got)
	}
}

// TestMetadataValuesComeFromTheChunksNotTheManifest pins the half of the design
// that is easy to get wrong: the document list is the CHUNKS' list, with the
// manifest consulted only for the value it alone carries.
//
// A document whose manifest row is missing still has chunks, and list_chunks and
// hybrid_search both reach those chunks. Filtering them out here would make
// metadata_search the one tool that cannot see a document the others can.
func TestMetadataValuesComeFromTheChunksNotTheManifest(t *testing.T) {
	s := New()
	s.Add([]Chunk{{ChunkID: "c0", DocID: "orphan.pdf", PageNum: 1, BlockType: "Text", Text: "orphan"}})

	if got := docIDs(t, s, one(FieldDocID, OpEqual, "orphan.pdf")); len(got) != 1 {
		t.Fatalf("doc ids = %v, want the document that has no manifest row", got)
	}
	// Its time is unknown, and unknown is not "indexed in 2026". This is the
	// failure mode the whole change exists to prevent, in miniature: "no match"
	// must never be produced by a value that was never there.
	if got := docIDs(t, s, one(FieldIndexedAt, OpStartWith, "2026")); len(got) != 0 {
		t.Fatalf("doc ids = %v, want none — the time is unknown, not 2026", got)
	}
	// An unknown time still answers "empty", so the field is not simply absent.
	if got := docIDs(t, s, one(FieldIndexedAt, OpEmpty, nil)); len(got) != 1 {
		t.Fatalf("doc ids = %v, want the document with no recorded time", got)
	}
}

func TestMetadataFieldSamplesCountDocuments(t *testing.T) {
	samples := metaStore().MetadataFieldSamples(10)

	byDoc := samples[FieldDocID]
	if len(byDoc) != 2 {
		t.Fatalf("doc_id samples = %#v, want one per document", byDoc)
	}
	for _, sample := range byDoc {
		if sample.Docs != 1 {
			t.Fatalf("sample %#v, want 1 document each", sample)
		}
	}

	byTime := samples[FieldIndexedAt]
	if len(byTime) != 2 {
		t.Fatalf("indexed_at samples = %#v, want a value per document", byTime)
	}

	// The cap is what keeps a hint a hint.
	if capped := metaStore().MetadataFieldSamples(1); len(capped[FieldDocID]) != 1 {
		t.Fatalf("capped samples = %#v, want one", capped[FieldDocID])
	}
}

func TestMetadataFieldAndOperatorSetsAreClosed(t *testing.T) {
	for _, field := range MetadataFieldNames() {
		if !KnownMetadataField(field) {
			t.Fatalf("%q is listed but not known", field)
		}
	}
	// The list is closed on purpose: an invented field must be refusable, and it
	// can only be refused if there is a set to refuse it against.
	if KnownMetadataField("author") || KnownMetadataField("block_type") || KnownMetadataField("") {
		t.Fatal("the field set must contain exactly doc_id and indexed_at")
	}

	for _, op := range MetadataOperators() {
		if !KnownMetadataOperator(op) {
			t.Fatalf("%q is listed but not known", op)
		}
	}
	if KnownMetadataOperator("matches") || KnownMetadataOperator("") {
		t.Fatal("an unknown operator must not be accepted")
	}
	// The operator the time rule depends on must be present.
	if !KnownMetadataOperator(OpStartWith) {
		t.Fatal("start with is what makes a day filterable")
	}
}
