package ipc

import (
	"strings"
	"sync"
	"testing"
	"time"
)

// collector is an Emit sink that records frames and their arrival times.
type collector struct {
	mu     sync.Mutex
	frames []string
	times  []time.Time
}

func (c *collector) emit(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, text)
	c.times = append(c.times, time.Now())
}

func (c *collector) snapshot() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.frames...)
}

// The first fragment is emitted immediately and is never held.
//
// Time to first token is the number a stream is judged by. Coalescing exists to
// save frames; buying them with first-content latency would trade the one thing
// it must not.
func TestCoalescerFirstFragmentIsImmediate(t *testing.T) {
	sink := &collector{}
	c := NewCoalescer(200*time.Millisecond, 0, sink.emit)

	started := time.Now()
	c.Write("Hello")
	if frames := sink.snapshot(); len(frames) != 1 || frames[0] != "Hello" {
		t.Fatalf("frames = %v, want the first fragment emitted at once", frames)
	}
	if elapsed := time.Since(started); elapsed > 50*time.Millisecond {
		t.Fatalf("the first fragment waited %s", elapsed)
	}
	c.Flush()
}

// A burst inside one window becomes one frame.
func TestCoalescerMergesABurstIntoOneFrame(t *testing.T) {
	sink := &collector{}
	c := NewCoalescer(80*time.Millisecond, 0, sink.emit)

	c.Write("a") // emitted at once
	c.Write("b")
	c.Write("c")
	c.Write("d")
	c.Write("e")
	c.Flush()

	frames := sink.snapshot()
	if len(frames) != 2 {
		t.Fatalf("frames = %v, want the first fragment then one merged frame", frames)
	}
	if frames[1] != "bcde" {
		t.Fatalf("merged frame = %q, want %q", frames[1], "bcde")
	}
	if written, emitted := c.Metrics(); written != 5 || emitted != 2 {
		t.Fatalf("metrics = (%d, %d), want (5, 2)", written, emitted)
	}
}

// The window alone flushes: a stream that goes quiet mid-answer must not leave
// text sitting in the buffer.
func TestCoalescerFlushesOnItsWindow(t *testing.T) {
	sink := &collector{}
	c := NewCoalescer(40*time.Millisecond, 0, sink.emit)

	c.Write("first")
	c.Write("second")

	deadline := time.Now().Add(2 * time.Second)
	for len(sink.snapshot()) < 2 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	frames := sink.snapshot()
	if len(frames) != 2 || frames[1] != "second" {
		t.Fatalf("frames = %v, want the remainder after the window", frames)
	}
}

// MaxRunes flushes early, so a fast producer does not build an unbounded frame.
func TestCoalescerFlushesAtMaxRunes(t *testing.T) {
	sink := &collector{}
	c := NewCoalescer(time.Hour, 5, sink.emit)

	c.Write("a")     // immediate
	c.Write("bc")    // buffered
	c.Write("defgh") // crosses the cap
	c.Flush()

	frames := sink.snapshot()
	if len(frames) != 2 || frames[1] != "bcdefgh" {
		t.Fatalf("frames = %v, want a flush at the rune cap", frames)
	}
}

// Nothing is lost. A truncated answer is far worse than a redundant frame, and
// this is the invariant that says the batching is lossless.
func TestCoalescerLosesNoText(t *testing.T) {
	sink := &collector{}
	c := NewCoalescer(20*time.Millisecond, 7, sink.emit)

	var want strings.Builder
	for i := 0; i < 500; i++ {
		fragment := "tok" + itoaTest(i%10) + " "
		want.WriteString(fragment)
		c.Write(fragment)
	}
	c.Flush()

	if got := strings.Join(sink.snapshot(), ""); got != want.String() {
		t.Fatalf("the stream lost or reordered text:\ngot  %d chars\nwant %d chars",
			len(got), want.Len())
	}
	if written, _ := c.Metrics(); written != 500 {
		t.Fatalf("written = %d, want 500", written)
	}
}

// After Flush the coalescer stops accepting: a late fragment must not be held
// for a window that will never fire.
func TestCoalescerAfterFlushIsClosed(t *testing.T) {
	sink := &collector{}
	c := NewCoalescer(20*time.Millisecond, 0, sink.emit)

	c.Write("done")
	c.Flush()
	c.Write("late")

	if got := strings.Join(sink.snapshot(), ""); got != "done" {
		t.Fatalf("frames joined to %q, want %q", got, "done")
	}
}

// Coalescing off is a nil coalescer, and every method on it is safe.
//
// The off switch is the default, so a caller must not have to branch at each
// call site to avoid paying for a feature it disabled.
func TestNilCoalescerIsInert(t *testing.T) {
	c := NewCoalescer(0, 0, func(string) { t.Fatal("a disabled coalescer emitted") })
	if c != nil {
		t.Fatalf("window 0 returned %v, want nil", c)
	}
	c.Write("x")
	c.Flush()
	if written, frames := c.Metrics(); written != 0 || frames != 0 {
		t.Fatalf("metrics = (%d, %d), want zeroes", written, frames)
	}
}

// itoaTest renders a single digit; the test only needs distinct fragments.
func itoaTest(n int) string { return string(rune('0' + n)) }
