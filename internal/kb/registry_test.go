package kb

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// openIn builds a registry inside a temporary directory.
func openIn(t *testing.T, legacy string) (*Registry, string) {
	t.Helper()
	dir := t.TempDir()
	r, err := Open(filepath.Join(dir, "kbs.json"), filepath.Join(dir, "kbs"), legacy)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return r, dir
}

func TestFirstRunCreatesADefaultBase(t *testing.T) {
	// Always having one base is what lets every caller assume one exists, and
	// keeps "no knowledge base yet" from being a state the UI has to render.
	r, _ := openIn(t, "")

	bases := r.List()
	if len(bases) != 1 {
		t.Fatalf("bases = %#v, want exactly one", bases)
	}
	if bases[0].Name != "默认知识库" || bases[0].ID == "" {
		t.Fatalf("default base = %#v", bases[0])
	}
}

func TestAdoptsTheLegacyIndexOnFirstRun(t *testing.T) {
	dir := t.TempDir()
	legacy := filepath.Join(dir, "index.json")
	if err := os.WriteFile(legacy, []byte(`{"chunks":[{"chunk_id":"c0"}]}`), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	r, err := Open(filepath.Join(dir, "kbs.json"), filepath.Join(dir, "kbs"), legacy)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	base := r.List()[0]
	adopted, err := os.ReadFile(r.DataPath(base.ID))
	if err != nil {
		t.Fatalf("the legacy index was not adopted: %v", err)
	}
	if !strings.Contains(string(adopted), "c0") {
		t.Fatalf("adopted file = %q", adopted)
	}

	// Left in place rather than moved: if the adoption turns out to be wrong,
	// the original is still there. An extra file is cheaper than a lost corpus.
	if _, err := os.Stat(legacy); err != nil {
		t.Fatalf("the legacy file must be left alone: %v", err)
	}
}

func TestNothingIsAdoptedWhenThereIsNoLegacyIndex(t *testing.T) {
	// The normal case on a fresh machine, and it must not be an error.
	r, _ := openIn(t, filepath.Join(t.TempDir(), "missing.json"))
	if _, err := os.Stat(r.DataPath(r.List()[0].ID)); !os.IsNotExist(err) {
		t.Fatalf("an index was created for a machine that had none: %v", err)
	}
}

func TestRegistrySurvivesReopening(t *testing.T) {
	dir := t.TempDir()
	path, root := filepath.Join(dir, "kbs.json"), filepath.Join(dir, "kbs")

	first, err := Open(path, root, "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	created, err := first.Create("论文库")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := first.SetCounts(created.ID, 7, 42); err != nil {
		t.Fatalf("SetCounts: %v", err)
	}

	second, err := Open(path, root, "")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	bases := second.List()
	if len(bases) != 2 {
		t.Fatalf("bases = %#v, want the default plus one", bases)
	}
	if bases[1].ID != created.ID || bases[1].Name != "论文库" {
		t.Fatalf("reopened = %#v", bases[1])
	}
	if bases[1].DocCount != 7 || bases[1].ChunkCount != 42 {
		t.Fatalf("counts = %d/%d, want 7/42", bases[1].DocCount, bases[1].ChunkCount)
	}

	// The counts are cached, and the reason they are cached is that listing
	// bases must not open an index. Opening one would make the cost of showing
	// the menu grow with the number of bases.
	if _, err := os.Stat(second.DataPath(created.ID)); !os.IsNotExist(err) {
		t.Fatalf("listing bases must not create or read an index: %v", err)
	}
}

func TestResolveFallsBackToTheFirstBaseOnlyWhenEmpty(t *testing.T) {
	r, _ := openIn(t, "")
	second, err := r.Create("第二个")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	byDefault, err := r.Resolve("")
	if err != nil {
		t.Fatalf("Resolve(\"\"): %v", err)
	}
	if byDefault.ID == second.ID {
		t.Fatal("an empty id must resolve to the first base, not the newest")
	}

	byID, err := r.Resolve(second.ID)
	if err != nil || byID.ID != second.ID {
		t.Fatalf("Resolve(%q) = %#v, %v", second.ID, byID, err)
	}

	// An unknown id is an error rather than a fallback. Silently answering from
	// the default base would look like a successful query against a base the
	// caller does not have.
	if _, err := r.Resolve("nope"); err == nil {
		t.Fatal("an unknown id must not resolve")
	}
}

func TestRenameDoesNotMoveAnything(t *testing.T) {
	r, _ := openIn(t, "")
	original := r.List()[0]

	renamed, err := r.Rename(original.ID, "改过的名字")
	if err != nil {
		t.Fatalf("Rename: %v", err)
	}

	// The id names the index directory and the Qdrant collection. If a rename
	// changed either, a cosmetic edit would move the user's data.
	if renamed.ID != original.ID {
		t.Fatalf("id changed on rename: %s -> %s", original.ID, renamed.ID)
	}
	if renamed.Name != "改过的名字" {
		t.Fatalf("name = %q", renamed.Name)
	}
	if r.DataPath(renamed.ID) != r.DataPath(original.ID) {
		t.Fatal("the index path must not depend on the name")
	}
	if r.Collection(renamed.ID) != r.Collection(original.ID) {
		t.Fatal("the collection name must not depend on the name")
	}
}

func TestNamesAreUniqueAndNonEmpty(t *testing.T) {
	r, _ := openIn(t, "")

	if _, err := r.Create("   "); err == nil {
		t.Fatal("a blank name must be rejected")
	}
	if _, err := r.Create("默认知识库"); err == nil {
		t.Fatal("a duplicate name must be rejected")
	}
	// Case-insensitively: the list is read by a person, and two entries that
	// differ only in case are one entry they cannot tell apart.
	if _, err := r.Create("默认知识库 "); err == nil {
		t.Fatal("a name differing only by whitespace must be rejected")
	}

	created, err := r.Create("论文库")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := r.Rename(r.List()[0].ID, "论文库"); err == nil {
		t.Fatal("renaming onto an existing name must be rejected")
	}
	// Renaming a base to its own name is not a collision.
	if _, err := r.Rename(created.ID, "论文库"); err != nil {
		t.Fatalf("renaming to the same name: %v", err)
	}
}

func TestTheLastBaseCannotBeRemoved(t *testing.T) {
	r, _ := openIn(t, "")
	only := r.List()[0]

	if _, err := r.Remove(only.ID); err == nil {
		t.Fatal("the last base must not be removable")
	}

	// With a second base, either one can go.
	second, err := r.Create("第二个")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := r.Remove(second.ID); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if len(r.List()) != 1 {
		t.Fatalf("bases = %#v", r.List())
	}
}

func TestIDsAreUniqueAndPathSafe(t *testing.T) {
	r, _ := openIn(t, "")

	seen := map[string]bool{}
	for i := 0; i < 32; i++ {
		base, err := r.Create("库" + string(rune('A'+i)))
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		if seen[base.ID] {
			t.Fatalf("duplicate id %q", base.ID)
		}
		seen[base.ID] = true

		// The id becomes a directory name and part of a Qdrant collection name,
		// so it has to survive both without escaping either.
		if strings.ContainsAny(base.ID, `/\ .`) {
			t.Fatalf("id %q is not usable as a directory or collection name", base.ID)
		}
	}
}

func TestSetCountsIsANoOpWhenNothingChanged(t *testing.T) {
	// `documents` calls this on every poll to keep the base list honest. If it
	// rewrote the registry each time, a read-only screen would cause disk churn.
	dir := t.TempDir()
	path := filepath.Join(dir, "kbs.json")
	r, err := Open(path, filepath.Join(dir, "kbs"), "")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	base := r.List()[0]
	if err := r.SetCounts(base.ID, 3, 9); err != nil {
		t.Fatalf("SetCounts: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	before := info.ModTime()

	if err := r.SetCounts(base.ID, 3, 9); err != nil {
		t.Fatalf("SetCounts (unchanged): %v", err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if !after.ModTime().Equal(before) {
		t.Fatal("an unchanged SetCounts rewrote the registry")
	}
}
