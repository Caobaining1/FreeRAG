package store

import (
	"sort"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// The two filterable metadata fields
// ---------------------------------------------------------------------------
//
// freerag offers exactly two, and the list is closed on purpose. The alternative
// — advertising whatever fields the corpus happens to carry — needs a
// declaration layer that freerag does not have, and a field that is merely
// *visible* in the index is not the same as a field the dataset means to filter
// on. Two fields that are always present for every document are honest; a wider
// list would be guesswork rendered as a schema.
//
// RAGFlow reaches the same place from the other direction: its MetadataCatalog
// offers only what the dataset declares or the index carries, and its
// metadata_search rejects any key outside that set rather than returning an
// empty result for a key that never existed.

// Metadata field names.
const (
	// FieldDocID is the document's identity: the source file name.
	FieldDocID = "doc_id"
	// FieldIndexedAt is when the document was last (re)indexed.
	FieldIndexedAt = "indexed_at"
)

// IndexedAtFormat is how indexed_at is stored, rendered and filtered.
//
// A full timestamp, which is exactly what makes `start with` the operator for
// "one day": HasPrefix("2026-09-29 14:26:01", "2026-09-29") is true, while `=`
// would require the caller to reproduce the seconds and so never matches a real
// document. This is RAGFlow's rule for its own time fields, reproduced rather
// than reinvented — see its metadata_search description: "A 'time' field stores
// 'YYYY-MM-DD HH:MM:SS', so filter ONE day with op 'start with' and the bare
// 'YYYY-MM-DD' — '=' never matches a stored time."
const IndexedAtFormat = "2006-01-02 15:04:05"

// MetadataFieldNames returns the filterable fields, in a stable order.
func MetadataFieldNames() []string {
	return []string{FieldDocID, FieldIndexedAt}
}

// KnownMetadataField reports whether key is one of the two offered fields.
//
// Callers check this before filtering. A key outside the set must become a hint
// naming the real fields, never an empty result: "you asked for a field that
// does not exist" and "the corpus has nothing" are different answers, and
// collapsing them is how a fabricated filter reads as a finding about the data.
func KnownMetadataField(key string) bool {
	return key == FieldDocID || key == FieldIndexedAt
}

// ---------------------------------------------------------------------------
// Operators (ported from RAGFlow's metadata filter)
// ---------------------------------------------------------------------------

// The operator set, and the semantics that go with each:
//
//	=  ≠  >  <  ≥  ≤   date-aware comparison of stored value against filter value
//	contains / not contains     case-insensitive substring
//	start with / end with       case-insensitive prefix / suffix
//	in / not in                 set membership over a list of values
//	empty / not empty           the field has no value at all
//
// `start with` is not a convenience here: for indexed_at it is the only way to
// ask for a day.
const (
	OpEqual        = "="
	OpNotEqual     = "≠"
	OpGreater      = ">"
	OpLess         = "<"
	OpGreaterEqual = "≥"
	OpLessEqual    = "≤"
	OpIn           = "in"
	OpNotIn        = "not in"
	OpContains     = "contains"
	OpNotContains  = "not contains"
	OpStartWith    = "start with"
	OpEndWith      = "end with"
	OpEmpty        = "empty"
	OpNotEmpty     = "not empty"
)

// MetadataOperators returns the accepted operators, in a stable order.
func MetadataOperators() []string {
	return []string{
		OpEqual, OpNotEqual, OpGreater, OpLess, OpGreaterEqual, OpLessEqual,
		OpIn, OpNotIn,
		OpContains, OpNotContains,
		OpStartWith, OpEndWith,
		OpEmpty, OpNotEmpty,
	}
}

// KnownMetadataOperator reports whether op is accepted.
//
// Validated for the same reason the field is: an unrecognised operator that
// silently matches nothing is indistinguishable from a filter that honestly
// matched nothing.
func KnownMetadataOperator(op string) bool {
	for _, known := range MetadataOperators() {
		if op == known {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// The query
// ---------------------------------------------------------------------------

// MetadataCondition is one {key, op, value} condition.
type MetadataCondition struct {
	Key   string `json:"key"`
	Op    string `json:"op"`
	Value any    `json:"value,omitempty"`
}

// MetadataQuery is a set of conditions combined by Logic.
type MetadataQuery struct {
	Conditions []MetadataCondition
	// Logic is "and" (default) or "or".
	Logic string
}

// MetadataSample is one value of a field plus how many documents carry it.
//
// Ported from RAGFlow's MetadataSample: showing the model the values that exist
// is what turns "fill in a filter" from a guess into a copy.
type MetadataSample struct {
	Value string `json:"value"`
	Docs  int    `json:"docs"`
}

// ---------------------------------------------------------------------------
// Reading the values
// ---------------------------------------------------------------------------

// MetadataValues returns doc_id → field → rendered value for every document the
// index holds.
//
// Document-level, because that is where these two fields live: the manifest
// records when a document was indexed, while chunks carry only text and a page
// number. A chunk-based filter therefore resolves to a *set of documents* and
// then selects their chunks — RAGFlow's split too, where metadata_search selects
// documents and a separate call reads them.
func (s *Store) MetadataValues() map[string]map[string]string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.metadataValuesLocked()
}

func (s *Store) metadataValuesLocked() map[string]map[string]string {
	// Newest record per doc id, mirroring DocumentByDocID: an edited document
	// leaves two fingerprints behind, and only the newer one describes the
	// chunks that are actually stored.
	newest := make(map[string]DocumentRecord, len(s.documents))
	for _, record := range s.documents {
		previous, ok := newest[record.DocID]
		if !ok || record.IndexedAt.After(previous.IndexedAt) {
			newest[record.DocID] = record
		}
	}

	// The document list comes from the CHUNKS, not from the manifest.
	//
	// Every document that can be retrieved must be filterable, and the two sets
	// are not the same: a document can have chunks and no manifest row (an
	// index written before the manifest existed, or a manifest that was lost),
	// and filtering such a document out would make metadata_search unable to
	// reach something list_chunks and hybrid_search both return. The manifest is
	// consulted only for the value it alone carries.
	out := make(map[string]map[string]string, len(s.chunks))
	for _, chunk := range s.chunks {
		docID := chunk.DocID
		if docID == "" {
			continue
		}
		if _, ok := out[docID]; ok {
			continue
		}

		fields := map[string]string{
			FieldDocID: docID,
			// Always present, so `empty` / `not empty` mean something: an
			// absent key would make both operators match nothing, which reads
			// as "no document has this field" rather than "its time is
			// unknown".
			FieldIndexedAt: "",
		}
		if record, ok := newest[docID]; ok && !record.IndexedAt.IsZero() {
			// Local, because the timestamp is written by this process in this
			// timezone: formatting it in UTC would make the value the user reads
			// in the document list and the value a filter must match disagree.
			fields[FieldIndexedAt] = record.IndexedAt.Local().Format(IndexedAtFormat)
		}
		out[docID] = fields
	}
	return out
}

// MetadataFieldSamples returns field → its values, each with the number of
// documents carrying it, most common first and capped at perField.
//
// This is the "statistics of what is available" half: it is what goes into the
// prompt so a filter value is copied from the corpus rather than invented.
func (s *Store) MetadataFieldSamples(perField int) map[string][]MetadataSample {
	values := s.MetadataValues()

	counts := make(map[string]map[string]int, len(MetadataFieldNames()))
	for _, fields := range values {
		for field, value := range fields {
			if value == "" {
				continue
			}
			if counts[field] == nil {
				counts[field] = map[string]int{}
			}
			counts[field][value]++
		}
	}

	out := make(map[string][]MetadataSample, len(counts))
	for field, byValue := range counts {
		samples := make([]MetadataSample, 0, len(byValue))
		for value, docs := range byValue {
			samples = append(samples, MetadataSample{Value: value, Docs: docs})
		}
		// Most documents first, then value, so the rendered list is stable
		// rather than dependent on map iteration.
		sort.Slice(samples, func(i, j int) bool {
			if samples[i].Docs != samples[j].Docs {
				return samples[i].Docs > samples[j].Docs
			}
			return samples[i].Value < samples[j].Value
		})
		if perField > 0 && len(samples) > perField {
			samples = samples[:perField]
		}
		out[field] = samples
	}
	return out
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

// MetadataDocIDs returns the ids of the documents matching the query, sorted.
//
// An empty result means no document matched — which is a statement about the
// filter, not about the corpus. Callers report it as such; see the
// metadata_search executor for how the two are kept apart.
func (s *Store) MetadataDocIDs(q MetadataQuery) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return metadataDocIDs(s.metadataValuesLocked(), q)
}

// MetadataSearchChunks returns the chunks of the documents matching the query.
func (s *Store) MetadataSearchChunks(q MetadataQuery, limit int) []Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()

	matched := metadataDocIDs(s.metadataValuesLocked(), q)
	if len(matched) == 0 {
		return nil
	}
	want := make(map[string]bool, len(matched))
	for _, docID := range matched {
		want[docID] = true
	}

	out := make([]Chunk, 0, 16)
	for _, chunk := range s.chunks {
		if !want[chunk.DocID] {
			continue
		}
		out = append(out, chunk)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func metadataDocIDs(values map[string]map[string]string, q MetadataQuery) []string {
	if len(q.Conditions) == 0 {
		return nil
	}
	logic := q.Logic
	if logic != "or" {
		logic = "and"
	}

	var result map[string]bool
	for _, condition := range q.Conditions {
		matched := make(map[string]bool, len(values))
		for docID, fields := range values {
			value, ok := fields[condition.Key]
			if !ok {
				// A field the document does not carry cannot satisfy a
				// condition on it. With AND that empties the result; with OR
				// the condition simply contributes nothing.
				continue
			}
			if metadataMatch(value, condition.Op, condition.Value) {
				matched[docID] = true
			}
		}

		if result == nil {
			result = matched
		} else if logic == "and" {
			for docID := range result {
				if !matched[docID] {
					delete(result, docID)
				}
			}
		} else {
			for docID := range matched {
				result[docID] = true
			}
		}

		if len(result) == 0 && logic == "and" {
			return nil
		}
	}

	out := make([]string, 0, len(result))
	for docID := range result {
		out = append(out, docID)
	}
	sort.Strings(out)
	return out
}

// metadataMatch decides one condition against one stored value.
func metadataMatch(value, op string, want any) bool {
	switch op {
	case OpEmpty:
		return value == ""
	case OpNotEmpty:
		return value != ""
	case OpIn, OpNotIn:
		return metadataMatchSet(value, op, want)
	}

	text := metadataText(want)
	switch op {
	case OpContains, OpNotContains:
		hit := strings.Contains(strings.ToLower(value), strings.ToLower(text))
		if op == OpNotContains {
			return !hit
		}
		return hit
	case OpStartWith:
		return strings.HasPrefix(strings.ToLower(value), strings.ToLower(text))
	case OpEndWith:
		return strings.HasSuffix(strings.ToLower(value), strings.ToLower(text))
	}
	return metadataCompare(value, text, op)
}

// metadataMatchSet handles the two set operators over a list of values.
//
// The case-sensitivity asymmetry is RAGFlow's and is reproduced rather than
// corrected: "in" is case-sensitive and "not in" is case-insensitive. Silently
// making them agree would be a behaviour change dressed as a bug fix, and the
// difference is invisible for these two fields, whose values are file names and
// timestamps.
func metadataMatchSet(value, op string, want any) bool {
	items, ok := want.([]any)
	if !ok {
		if list, isStrings := want.([]string); isStrings {
			items = make([]any, 0, len(list))
			for _, item := range list {
				items = append(items, item)
			}
		} else {
			// A list operator with no list matches nothing. For "not in" this
			// is deliberately not "match everything": a filter that cannot be
			// read must not silently widen the selection.
			return false
		}
	}

	if op == OpNotIn {
		for _, item := range items {
			if strings.EqualFold(metadataText(item), value) {
				return false
			}
		}
		return true
	}
	for _, item := range items {
		if metadataText(item) == value {
			return true
		}
	}
	return false
}

// metadataCompare is the comparison half: =, ≠, >, <, ≥, ≤.
//
// Date-shaped filter values are compared only against date-shaped stored
// values, which is what stops "indexed_at > 3" from being read as a time
// comparison and matching something arbitrary. Same order as RAGFlow's: dates,
// then numbers, then case-insensitive strings.
func metadataCompare(stored, want, op string) bool {
	if isDateValue(want) {
		if !isDateValue(stored) {
			return op == OpNotEqual
		}
		return compareOrdered(stored, want, op)
	}
	if a, errA := strconv.ParseFloat(stored, 64); errA == nil {
		if b, errB := strconv.ParseFloat(want, 64); errB == nil {
			return compareOrderedFloat(a, b, op)
		}
	}
	return compareOrdered(strings.ToLower(stored), strings.ToLower(want), op)
}

func compareOrdered(a, b, op string) bool {
	switch op {
	case OpEqual:
		return a == b
	case OpNotEqual:
		return a != b
	case OpGreater:
		return a > b
	case OpLess:
		return a < b
	case OpGreaterEqual:
		return a >= b
	case OpLessEqual:
		return a <= b
	}
	return false
}

func compareOrderedFloat(a, b float64, op string) bool {
	switch op {
	case OpEqual:
		return a == b
	case OpNotEqual:
		return a != b
	case OpGreater:
		return a > b
	case OpLess:
		return a < b
	case OpGreaterEqual:
		return a >= b
	case OpLessEqual:
		return a <= b
	}
	return false
}

// metadataDateLayouts are the shapes isDateValue recognises.
var metadataDateLayouts = []string{
	IndexedAtFormat,
	"2006-01-02T15:04:05",
	"2006-01-02",
}

// isDateValue reports whether text reads as a date or a timestamp.
func isDateValue(text string) bool {
	for _, layout := range metadataDateLayouts {
		if _, err := time.Parse(layout, text); err == nil {
			return true
		}
	}
	return false
}

// metadataText renders a filter value as the string a stored value is compared
// against.
func metadataText(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	case float64:
		// JSON numbers arrive as float64; rendering 2.0 as "2" keeps a numeric
		// condition readable, and 15:04:05 never comes through this path.
		return strconv.FormatFloat(typed, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(typed)
	default:
		return strings.TrimSpace(strings.Trim(strings.TrimSpace(
			toStringValue(value)), `"`))
	}
}

// toStringValue is a last-resort rendering for values a model sent in a shape
// nobody expected (a nested list, an object). It must never panic and never
// return something that silently matches everything.
func toStringValue(value any) string {
	if text, ok := value.(interface{ String() string }); ok {
		return text.String()
	}
	return ""
}
