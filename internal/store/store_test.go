package store

import (
	"path/filepath"
	"testing"
)

func sampleChunks() []Chunk {
	return []Chunk{
		{ChunkID: "c0", DocID: "a.pdf", PageNum: 1, BlockType: "Title", Text: "Freerag Acceptance Report"},
		{ChunkID: "c1", DocID: "a.pdf", PageNum: 1, BlockType: "Text",
			Text: "The retrieval pipeline parses documents into layout blocks."},
		{ChunkID: "c2", DocID: "a.pdf", PageNum: 2, BlockType: "Text",
			Text: "Sufficiency checking decides whether the draft answers the question."},
		{ChunkID: "c3", DocID: "b.pdf", PageNum: 1, BlockType: "Text",
			Text: "文档解析管线将版面块切分为检索单元。"},
	}
}

func TestTermsLatinAndCJK(t *testing.T) {
	latin := Terms("Hello World, go1")
	for _, want := range []string{"hello", "world", "go1"} {
		if !contains(latin, want) {
			t.Fatalf("latin terms %v missing %q", latin, want)
		}
	}

	cjk := Terms("文档解析")
	want := []string{"文档", "档解", "解析"}
	for _, term := range want {
		if !contains(cjk, term) {
			t.Fatalf("CJK terms %v missing %q", cjk, term)
		}
	}

	if got := Terms("中"); len(got) != 1 || got[0] != "中" {
		t.Fatalf("single CJK rune terms = %v", got)
	}
}

func contains(list []string, want string) bool {
	for _, item := range list {
		if item == want {
			return true
		}
	}
	return false
}

func TestAddDedupes(t *testing.T) {
	s := New()
	if added := s.Add(sampleChunks()); added != 4 {
		t.Fatalf("added = %d, want 4", added)
	}
	if added := s.Add(sampleChunks()); added != 0 {
		t.Fatalf("second add = %d, want 0", added)
	}
	if s.Len() != 4 {
		t.Fatalf("len = %d, want 4", s.Len())
	}
}

func TestAddSkipsBlankText(t *testing.T) {
	s := New()
	if added := s.Add([]Chunk{{ChunkID: "x", Text: "   "}}); added != 0 {
		t.Fatalf("blank chunk was indexed: %d", added)
	}
}

func TestSearchRanksRelevantChunkFirst(t *testing.T) {
	s := New()
	s.Add(sampleChunks())

	hits := s.Search("sufficiency draft", 3)
	if len(hits) == 0 {
		t.Fatal("expected hits")
	}
	if hits[0].Chunk.ChunkID != "c2" {
		t.Fatalf("top hit = %s, want c2", hits[0].Chunk.ChunkID)
	}
	if hits[0].Score <= 0 {
		t.Fatalf("score = %f", hits[0].Score)
	}
}

func TestSearchCJK(t *testing.T) {
	s := New()
	s.Add(sampleChunks())

	hits := s.Search("文档解析", 3)
	if len(hits) == 0 {
		t.Fatal("expected CJK hits")
	}
	if hits[0].Chunk.ChunkID != "c3" {
		t.Fatalf("top CJK hit = %s, want c3", hits[0].Chunk.ChunkID)
	}
}

func TestSearchRespectsLimit(t *testing.T) {
	s := New()
	s.Add(sampleChunks())
	if hits := s.Search("the", 2); len(hits) > 2 {
		t.Fatalf("limit ignored: %d hits", len(hits))
	}
}

func TestSearchNoMatchReturnsNothing(t *testing.T) {
	s := New()
	s.Add(sampleChunks())
	if hits := s.Search("zzzzz-nonexistent", 5); len(hits) != 0 {
		t.Fatalf("unexpected hits: %#v", hits)
	}
}

func TestSearchEmptyQueryReturnsNothing(t *testing.T) {
	s := New()
	s.Add(sampleChunks())
	if hits := s.Search("   ", 5); len(hits) != 0 {
		t.Fatalf("empty query returned %d hits", len(hits))
	}
}

func TestSearchOnEmptyStore(t *testing.T) {
	if hits := New().Search("anything", 5); len(hits) != 0 {
		t.Fatalf("empty store returned %d hits", len(hits))
	}
}

func TestSaveAndLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "index.json")

	s := New()
	s.Add(sampleChunks())
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != 4 {
		t.Fatalf("loaded len = %d, want 4", loaded.Len())
	}
	hits := loaded.Search("sufficiency", 1)
	if len(hits) != 1 || hits[0].Chunk.ChunkID != "c2" {
		t.Fatalf("search after reload failed: %#v", hits)
	}
}

func TestLoadMissingFileIsEmpty(t *testing.T) {
	loaded, err := Load(filepath.Join(t.TempDir(), "absent.json"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != 0 {
		t.Fatalf("len = %d, want 0", loaded.Len())
	}
}
