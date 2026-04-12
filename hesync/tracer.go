// Package hesync implements storage-assisted evaluation key (EVK) management
// for homomorphic encryption inference, based on the HESync paper.
// It reduces peak memory usage by storing EVKs on disk and prefetching them
// on-demand using a plan generated from a dry-run trace.
package hesync

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// KeyType distinguishes between Galois keys and relinearization keys in the trace.
type KeyType int

const (
	KeyTypeGalois KeyType = iota
	KeyTypeRelin
)

// TraceEntry records a single EVK access during an HE computation.
type TraceEntry struct {
	Step     int     // monotonic step counter
	Type     KeyType // type of key accessed
	GaloisEl uint64  // Galois element (only meaningful for KeyTypeGalois)
}

// Trace is an ordered sequence of EVK accesses recorded during a computation.
type Trace struct {
	Entries []TraceEntry
}

// TracingEvaluationKeySet wraps a real EvaluationKeySet and records every
// EVK access. It implements the rlwe.EvaluationKeySet interface so it can
// be used transparently with any evaluator.
//
// Usage:
//  1. Create a TracingEvaluationKeySet wrapping your MemEvaluationKeySet
//  2. Run inference using an evaluator constructed with this key set
//  3. Call GetTrace() to retrieve the ordered sequence of EVK accesses
//
// Since neural network inference has data-independent control flow,
// the trace from one run is valid for all future runs with the same circuit.
type TracingEvaluationKeySet struct {
	inner   rlwe.EvaluationKeySet
	entries []TraceEntry
	step    int
	mu      sync.Mutex
}

// NewTracingEvaluationKeySet creates a new tracing wrapper around an existing EvaluationKeySet.
func NewTracingEvaluationKeySet(inner rlwe.EvaluationKeySet) *TracingEvaluationKeySet {
	return &TracingEvaluationKeySet{
		inner:   inner,
		entries: make([]TraceEntry, 0, 256),
	}
}

// GetGaloisKey retrieves the Galois key while recording the access.
func (t *TracingEvaluationKeySet) GetGaloisKey(galEl uint64) (*rlwe.GaloisKey, error) {
	t.mu.Lock()
	entry := TraceEntry{
		Step:     t.step,
		Type:     KeyTypeGalois,
		GaloisEl: galEl,
	}
	t.entries = append(t.entries, entry)
	t.step++
	t.mu.Unlock()

	return t.inner.GetGaloisKey(galEl)
}

// GetGaloisKeysList returns the list of available Galois elements (delegates to inner).
func (t *TracingEvaluationKeySet) GetGaloisKeysList() []uint64 {
	return t.inner.GetGaloisKeysList()
}

// GetRelinearizationKey retrieves the relinearization key while recording the access.
func (t *TracingEvaluationKeySet) GetRelinearizationKey() (*rlwe.RelinearizationKey, error) {
	t.mu.Lock()
	entry := TraceEntry{
		Step: t.step,
		Type: KeyTypeRelin,
	}
	t.entries = append(t.entries, entry)
	t.step++
	t.mu.Unlock()

	return t.inner.GetRelinearizationKey()
}

// GetTrace returns the recorded trace of EVK accesses.
func (t *TracingEvaluationKeySet) GetTrace() Trace {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Return a copy to avoid race conditions
	entries := make([]TraceEntry, len(t.entries))
	copy(entries, t.entries)
	return Trace{Entries: entries}
}

// Reset clears the recorded trace, allowing the key set to be reused.
func (t *TracingEvaluationKeySet) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.entries = t.entries[:0]
	t.step = 0
}

// KeyID returns a unique string identifier for a trace entry's key.
func (e TraceEntry) KeyID() string {
	if e.Type == KeyTypeRelin {
		return "rlk"
	}
	return fmt.Sprintf("gk_%d", e.GaloisEl)
}
