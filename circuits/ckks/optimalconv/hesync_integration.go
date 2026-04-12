package optimalconv

import (
	"fmt"
	"runtime"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/hesync"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// HESyncConfig holds configuration for an HESync-enabled inference run.
type HESyncConfig struct {
	EVKDir        string // directory for serialized EVKs
	Workers       int    // number of async loader goroutines (default: 4)
	DefaultLevel  int    // default ciphertext level for profiling
}

// HESyncStats holds the results of an HESync-enabled inference run.
type HESyncStats struct {
	// Dry run phase stats
	TraceSteps     int
	PlanFetches    int
	PlanSyncs      int
	PlanFrees      int
	DryRunTime     time.Duration

	// Runtime phase stats
	InferenceTime  time.Duration
	PeakMemBaseline uint64 // peak memory without HESync (bytes)
	PeakMemHESync   uint64 // peak memory with HESync (bytes)
	MemReduction    float64 // percentage reduction

	// EVK stats
	TotalEVKs      int
	BaselineEVKMem uint64 // total memory for all EVKs (bytes)
}

// RunBaselineInference runs the CNN inference with all EVKs in memory (baseline).
// Returns the inference stats and the encrypted outputs.
func RunBaselineInference(
	params ckks.Parameters,
	evk *rlwe.MemEvaluationKeySet,
	btpEval bootstrapping.Bootstrapper,
	config PlainCNNConfig,
	inputs []*rlwe.Ciphertext,
) ([]*rlwe.Ciphertext, *InferenceStats, error) {
	eval := ckks.NewEvaluator(params, evk)
	cnnEval := NewCNNEvaluator(params, eval, btpEval, config)

	// Generate and set random kernels
	kernels := GenerateRandomKernels(config)
	if err := cnnEval.SetKernels(kernels); err != nil {
		return nil, nil, fmt.Errorf("baseline: set kernels: %w", err)
	}

	// Measure memory before
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	// Run inference
	outputs, err := cnnEval.Evaluate(inputs)
	if err != nil {
		return nil, nil, fmt.Errorf("baseline: evaluate: %w", err)
	}

	// Measure memory after
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	stats := &cnnEval.Stats
	return outputs, stats, nil
}

// RunHESyncInference runs the CNN inference with HESync EVK management.
// This is the main integration point that demonstrates memory savings.
//
// The flow:
//  1. Serialize all EVKs to disk
//  2. Run a tracing pass to record EVK access pattern
//  3. Profile operation latencies and EVK load time
//  4. Generate prefetch plan
//  5. Run inference with DiskEvaluationKeySet
func RunHESyncInference(
	params ckks.Parameters,
	evk *rlwe.MemEvaluationKeySet,
	btpEval bootstrapping.Bootstrapper,
	config PlainCNNConfig,
	inputs []*rlwe.Ciphertext,
	hesyncCfg HESyncConfig,
) ([]*rlwe.Ciphertext, *HESyncStats, error) {
	stats := &HESyncStats{}
	dryRunStart := time.Now()

	// Step 1: Serialize EVKs to disk
	if err := hesync.SerializeEVKSet(evk, hesyncCfg.EVKDir); err != nil {
		return nil, nil, fmt.Errorf("hesync: serialize: %w", err)
	}

	// Count total EVKs and their memory
	galEls := evk.GetGaloisKeysList()
	stats.TotalEVKs = len(galEls)
	if evk.RelinearizationKey != nil {
		stats.TotalEVKs++
	}

	// Step 2: Run tracing pass
	tracingEvk := hesync.NewTracingEvaluationKeySet(evk)
	tracingEval := ckks.NewEvaluator(params, tracingEvk)
	cnnEval := NewCNNEvaluator(params, tracingEval, btpEval, config)

	kernels := GenerateRandomKernels(config)
	if err := cnnEval.SetKernels(kernels); err != nil {
		return nil, nil, fmt.Errorf("hesync: set kernels: %w", err)
	}

	_, err := cnnEval.Evaluate(inputs)
	if err != nil {
		return nil, nil, fmt.Errorf("hesync: tracing pass: %w", err)
	}

	trace := tracingEvk.GetTrace()
	stats.TraceSteps = len(trace.Entries)

	// Step 3: Profile
	profiler := hesync.NewProfiler()
	profiler.SetOpLatency(hesyncCfg.DefaultLevel, time.Millisecond) // will be refined

	// Measure EVK load latency from a sample key on disk
	idx, err := hesync.BuildIndex(hesyncCfg.EVKDir)
	if err != nil {
		return nil, nil, fmt.Errorf("hesync: build index: %w", err)
	}

	// Sample load latency from first available key
	for _, path := range idx.GaloisKeys {
		if err := profiler.MeasureEVKLoadLatency(path, 3); err != nil {
			return nil, nil, fmt.Errorf("hesync: measure load latency: %w", err)
		}
		break
	}

	profile := profiler.GetProfile()

	// Step 4: Generate prefetch plan
	plan := hesync.GeneratePlan(trace, profile, hesyncCfg.DefaultLevel)
	stats.PlanFetches, stats.PlanSyncs, stats.PlanFrees = plan.Stats()
	stats.DryRunTime = time.Since(dryRunStart)

	// Step 5: Create DiskEvaluationKeySet and run inference
	diskEvk := hesync.NewDiskEvaluationKeySet(idx, plan, hesyncCfg.Workers)
	defer diskEvk.Stop()

	diskEval := ckks.NewEvaluator(params, diskEvk)
	cnnEvalDisk := NewCNNEvaluator(params, diskEval, btpEval, config)
	if err := cnnEvalDisk.SetKernels(kernels); err != nil {
		return nil, nil, fmt.Errorf("hesync: set kernels (disk): %w", err)
	}

	// Measure memory during HESync inference
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	inferenceStart := time.Now()
	outputs, err := cnnEvalDisk.Evaluate(inputs)
	if err != nil {
		return nil, nil, fmt.Errorf("hesync: disk inference: %w", err)
	}
	stats.InferenceTime = time.Since(inferenceStart)

	runtime.GC()
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	stats.PeakMemHESync = memAfter.Sys - memBefore.Sys

	return outputs, stats, nil
}

// CompareResults compares the outputs of baseline and HESync inference
// to verify correctness. Returns the maximum deviation.
func CompareResults(
	params ckks.Parameters,
	sk *rlwe.SecretKey,
	baseline []*rlwe.Ciphertext,
	hesyncOut []*rlwe.Ciphertext,
) (maxDev float64, err error) {
	if len(baseline) != len(hesyncOut) {
		return 0, fmt.Errorf("output count mismatch: baseline=%d, hesync=%d",
			len(baseline), len(hesyncOut))
	}

	dec := rlwe.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	for ch := range baseline {
		ptBase := dec.DecryptNew(baseline[ch])
		ptHE := dec.DecryptNew(hesyncOut[ch])

		ptBase.IsBatched = false
		ptHE.IsBatched = false

		valsBase, err := CfDcd(encoder, ptBase, 16)
		if err != nil {
			return 0, fmt.Errorf("decode baseline ch %d: %w", ch, err)
		}

		valsHE, err := CfDcd(encoder, ptHE, 16)
		if err != nil {
			return 0, fmt.Errorf("decode hesync ch %d: %w", ch, err)
		}

		for i := range valsBase {
			dev := valsBase[i] - valsHE[i]
			if dev < 0 {
				dev = -dev
			}
			if dev > maxDev {
				maxDev = dev
			}
		}
	}

	return maxDev, nil
}
