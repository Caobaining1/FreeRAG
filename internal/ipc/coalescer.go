package ipc

import (
	"strings"
	"sync"
	"time"
)

// Coalescer merges a burst of streamed text fragments into fewer notifications.
//
// The stream at issue is the answer: the generator emits it token by token and
// every token becomes one JSON-RPC notification, so a 650-character answer is
// ~360 separate frames, each one a JSON encode, a write(2), a readline, a
// json.parse and an IPC hop into the renderer. Coalescing turns those into
// fewer, larger frames — the standard batching move on the server side of an
// SSE stream.
//
// It is OPT-IN and defaults to OFF (window <= 0), because on the reference
// machine it cannot pay for itself and the measurement says so (see
// docs/performance.md): generation runs at ~6.4 tok/s, so fragments arrive
// about every 156 ms and a frame costs microseconds. There is nothing to
// amortise and the window is pure added latency. It is kept because the trade
// reverses on a machine whose generator is an order of magnitude faster — there
// the frame overhead stops being free and the same batching is a win — and
// because "we tried it" is worth more with the code than without it.
//
// Timing is deliberate rather than incidental:
//
//   - The FIRST fragment is emitted immediately and is never held. Time to
//     first token is the number a stream is judged by, and it is the one thing
//     coalescing must not change.
//   - Later fragments flush when the window elapses or the buffer reaches
//     MaxRunes, whichever comes first. So the added latency is bounded by the
//     window, and a producer faster than the window gets real merging.
//   - Flush is explicit at the end of the stream: a fragment still inside the
//     window when the generator stops would otherwise never be delivered, and a
//     truncated answer is a far worse failure than an extra frame.
type Coalescer struct {
	// Window is how long a fragment may wait to be merged with the next one.
	// <= 0 emits every fragment immediately (coalescing off).
	Window time.Duration
	// MaxRunes flushes early once the buffer is this large; <= 0 means the
	// window alone decides.
	MaxRunes int
	// Emit receives the merged text. Required.
	Emit func(string)

	mu      sync.Mutex
	buf     strings.Builder
	timer   *time.Timer
	first   bool
	closed  bool
	written int
	frames  int
}

// NewCoalescer returns a coalescer, or nil when window <= 0.
//
// nil is the off switch and is safe to call methods on, so a caller does not
// have to branch at every call site to avoid paying for a feature it disabled.
func NewCoalescer(window time.Duration, maxRunes int, emit func(string)) *Coalescer {
	if window <= 0 || emit == nil {
		return nil
	}
	return &Coalescer{Window: window, MaxRunes: maxRunes, Emit: emit, first: true}
}

// Write adds one fragment to the stream.
func (c *Coalescer) Write(fragment string) {
	if c == nil || fragment == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	c.written++
	c.buf.WriteString(fragment)

	// The first fragment of a stream goes out now: holding it for a window
	// would delay the answer's first character by exactly that window, which is
	// what a stream must never do.
	if c.first {
		c.first = false
		c.flushLocked()
		return
	}
	if c.MaxRunes > 0 && c.buf.Len() >= c.MaxRunes {
		c.flushLocked()
		return
	}
	if c.timer == nil {
		c.timer = time.AfterFunc(c.Window, c.flushTimer)
	}
}

// flushTimer is the deadline callback: emit whatever has accumulated.
func (c *Coalescer) flushTimer() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.timer = nil
	c.flushLocked()
}

// Flush emits anything still buffered and stops the timer.
//
// Must be called when the stream ends. Without it the last fragment can sit in
// the buffer forever: generation is the only thing that drives the timer, so
// the fragment produced just before `done` may never see another write.
func (c *Coalescer) Flush() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	if c.timer != nil {
		c.timer.Stop()
		c.timer = nil
	}
	c.flushLocked()
}

// flushLocked emits the buffer. Caller holds the lock.
func (c *Coalescer) flushLocked() {
	if c.buf.Len() == 0 {
		return
	}
	text := c.buf.String()
	c.buf.Reset()
	c.frames++
	c.Emit(text)
}

// Metrics reports fragments in and frames out.
//
// Reported rather than logged: the only claim coalescing makes is a frame
// count, so the count has to be readable by whatever measures it.
func (c *Coalescer) Metrics() (written, frames int) {
	if c == nil {
		return 0, 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.written, c.frames
}
