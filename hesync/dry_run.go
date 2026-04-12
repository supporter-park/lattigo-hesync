package hesync

import (
	"fmt"
	"time"
)

// DryRunEntry records a single operation in the dry run trace.
type DryRunEntry struct {
	Step       int
	OpType     string   // "conv", "relu", "bootstrap", "rotate", etc.
	Level      int      // ciphertext level at this operation
	KeyType    KeyType  // type of EVK used (if any)
	GaloisEl   uint64   // Galois element (for rotation-type ops)
	NeedsEVK   bool     // whether this operation accesses an EVK
}

// DryRunTrace is a lightweight trace of the CNN inference that records
// only operation metadata (type, level, EVK requirements) without
// performing any actual HE computation.
//
// This is much faster than the full TracingEvaluationKeySet approach,
// which re-runs the entire inference. The dry run only needs to know:
//  1. The sequence of operations
//  2. The ciphertext level at each step
//  3. Which EVKs are needed and when
type DryRunTrace struct {
	Entries []DryRunEntry
}

// DryRunCNNTracer generates a lightweight trace of CNN inference by
// simulating level changes without performing actual HE computation.
//
// For each layer:
//   - Conv: consumes 1 level (mul + rescale)
//   - ReLU: consumes ~11 levels (3-polynomial evaluation)
//   - Bootstrap: restores level to maxLevel
//
// This takes microseconds instead of minutes.
func DryRunCNNTracer(
	numLayers int,
	maxLevel int,
	layerHasActivation []bool,
	layerHasBootstrap []bool,
	reluLevels int,
	galoisElements []uint64,
) *DryRunTrace {
	trace := &DryRunTrace{
		Entries: make([]DryRunEntry, 0, numLayers*3),
	}

	currentLevel := maxLevel
	step := 0

	for l := 0; l < numLayers; l++ {
		// Conv operation: ct-pt multiply + rescale → consumes 1 level
		trace.Entries = append(trace.Entries, DryRunEntry{
			Step:     step,
			OpType:   "conv",
			Level:    currentLevel,
			NeedsEVK: false, // ct-pt multiply doesn't need Galois keys
		})
		step++
		currentLevel-- // rescale consumes 1 level

		// ReLU activation
		if l < len(layerHasActivation) && layerHasActivation[l] {
			trace.Entries = append(trace.Entries, DryRunEntry{
				Step:     step,
				OpType:   "relu",
				Level:    currentLevel,
				NeedsEVK: true, // polynomial evaluation may need relinearization
				KeyType:  KeyTypeRelin,
			})
			step++
			currentLevel -= reluLevels
			if currentLevel < 0 {
				currentLevel = 0
			}
		}

		// Bootstrapping: restores level
		if l < len(layerHasBootstrap) && layerHasBootstrap[l] {
			// Record Galois key accesses for bootstrap rotations
			for _, galEl := range galoisElements {
				trace.Entries = append(trace.Entries, DryRunEntry{
					Step:     step,
					OpType:   "bootstrap",
					Level:    currentLevel,
					KeyType:  KeyTypeGalois,
					GaloisEl: galEl,
					NeedsEVK: true,
				})
				step++
			}

			// Bootstrap restores level
			currentLevel = maxLevel
		}
	}

	return trace
}

// ToTrace converts a DryRunTrace to a standard Trace for use with the Planner.
func (d *DryRunTrace) ToTrace() Trace {
	entries := make([]TraceEntry, 0, len(d.Entries))
	for _, e := range d.Entries {
		if e.NeedsEVK {
			entries = append(entries, TraceEntry{
				Step:     e.Step,
				Type:     e.KeyType,
				GaloisEl: e.GaloisEl,
			})
		}
	}
	return Trace{Entries: entries}
}

// DryRunProfile generates a profile from dry run data using synthetic
// timing estimates based on level and operation type.
func DryRunProfile(maxLevel int, evkLoadLatency time.Duration) Profile {
	profile := Profile{
		OpLatency:      make(map[int]time.Duration),
		EVKLoadLatency: evkLoadLatency,
	}

	// Estimate per-level operation latency
	// Higher levels have larger polynomials and take longer
	for level := 0; level <= maxLevel; level++ {
		// Rough estimate: latency scales linearly with level
		profile.OpLatency[level] = time.Duration(level+1) * time.Millisecond
	}

	return profile
}

// String returns a summary of the dry run trace.
func (d *DryRunTrace) String() string {
	convOps, reluOps, btsOps, evkAccesses := 0, 0, 0, 0
	for _, e := range d.Entries {
		switch e.OpType {
		case "conv":
			convOps++
		case "relu":
			reluOps++
		case "bootstrap":
			btsOps++
		}
		if e.NeedsEVK {
			evkAccesses++
		}
	}
	return fmt.Sprintf("DryRunTrace: %d ops (%d conv, %d relu, %d bootstrap), %d EVK accesses",
		len(d.Entries), convOps, reluOps, btsOps, evkAccesses)
}
