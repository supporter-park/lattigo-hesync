package hesync

import (
	"fmt"
	"time"
)

// Profile stores the timing data collected by the Profiler.
// It contains per-level operation execution times and EVK load latency.
type Profile struct {
	// OpLatency maps (ciphertext level) -> estimated operation latency.
	// Level-aware profiling (the paper's "Level 2") uses the actual ciphertext
	// level for each operation, which achieves the design goal of reducing
	// peak memory by 58-89% with ~1% execution overhead.
	OpLatency map[int]time.Duration

	// EVKLoadLatency is the measured time to load a single EVK from disk.
	EVKLoadLatency time.Duration
}

// Profiler measures the execution time of HE operations at each ciphertext
// level and the latency of loading EVKs from storage.
//
// Level-aware profiling is critical: the paper shows that profiling at a
// single fixed level either over-estimates latency (causing premature loads
// and increased memory) or under-estimates it (causing stalls). The
// level-aware approach profiles at each level actually used by the circuit.
type Profiler struct {
	profile Profile
}

// NewProfiler creates a new Profiler.
func NewProfiler() *Profiler {
	return &Profiler{
		profile: Profile{
			OpLatency: make(map[int]time.Duration),
		},
	}
}

// MeasureOpLatency records the execution time of an HE operation at a given
// ciphertext level. Call this with a representative operation (e.g., rotation
// or multiplication) at each level that appears in the trace.
//
// The measurer function should perform the operation and return its duration.
// This is called externally because the profiler doesn't know the specific
// HE operations — it only stores the timings.
func (p *Profiler) MeasureOpLatency(level int, measurer func() time.Duration) {
	duration := measurer()
	p.profile.OpLatency[level] = duration
}

// SetOpLatency directly sets the operation latency for a given level.
// Useful when latencies are known from prior measurements or calibration.
func (p *Profiler) SetOpLatency(level int, d time.Duration) {
	p.profile.OpLatency[level] = d
}

// MeasureEVKLoadLatency measures the time to load a single EVK from disk.
// It loads the key at the given path and records the elapsed time.
// Multiple measurements are averaged for stability.
func (p *Profiler) MeasureEVKLoadLatency(keyPath string, samples int) error {
	if samples <= 0 {
		samples = 1
	}

	var total time.Duration
	for i := 0; i < samples; i++ {
		start := time.Now()
		_, err := LoadGaloisKey(keyPath)
		elapsed := time.Since(start)

		if err != nil {
			return fmt.Errorf("MeasureEVKLoadLatency: %w", err)
		}

		total += elapsed
	}

	p.profile.EVKLoadLatency = total / time.Duration(samples)
	return nil
}

// SetEVKLoadLatency directly sets the EVK load latency.
func (p *Profiler) SetEVKLoadLatency(d time.Duration) {
	p.profile.EVKLoadLatency = d
}

// GetProfile returns the collected profile data.
func (p *Profiler) GetProfile() Profile {
	return p.profile
}

// GetOpLatency returns the operation latency for a given level.
// If no measurement exists for the exact level, it returns the closest
// available measurement (conservative: uses the maximum measured latency).
func (prof Profile) GetOpLatency(level int) time.Duration {
	if d, ok := prof.OpLatency[level]; ok {
		return d
	}

	// Fallback: return maximum measured latency (conservative estimate)
	var maxD time.Duration
	for _, d := range prof.OpLatency {
		if d > maxD {
			maxD = d
		}
	}
	return maxD
}
