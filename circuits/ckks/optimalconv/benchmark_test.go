package optimalconv

import (
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/hesync"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// benchParams returns parameters suitable for benchmarking.
// Uses LogN=12 for reasonable speed; set LogN=16 for paper-equivalent benchmarks.
func benchParams(b *testing.B) (ckks.Parameters, *rlwe.SecretKey) {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            12,
		LogQ:            []int{55, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61},
		LogDefaultScale: 45,
	})
	require.NoError(b, err)

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	return params, sk
}

func BenchmarkConvSingle(b *testing.B) {
	params, sk := benchParams(b)

	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)

	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	image := make([]float64, params.N())
	for i := range image {
		image[i] = float64(i) / float64(params.N())
	}
	kernel := make([]float64, 9)
	for i := range kernel {
		kernel[i] = 0.1
	}

	ptImage, _ := CfEcd(encoder, image, params, params.MaxLevel(), params.DefaultScale())
	ctImage, _ := enc.EncryptNew(ptImage)
	ptKernel, _ := CfEcd(encoder, kernel, params, ctImage.Level(), params.DefaultScale())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ConvSingle(eval, ctImage, ptKernel)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPackLWEs(b *testing.B) {
	params, sk := benchParams(b)

	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)

	n := 8
	galEls := GaloisElementsForPackLWEs(params, n)
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)
	eval := ckks.NewEvaluator(params, evk)

	cts := make([]*rlwe.Ciphertext, n)
	for i := 0; i < n; i++ {
		pt, _ := CfEcd(encoder, []float64{float64(i + 1)}, params,
			params.MaxLevel(), params.DefaultScale())
		cts[i], _ = enc.EncryptNew(pt)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := PackLWEs(eval, cts, 0, 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCNNLayer(b *testing.B) {
	params, sk := benchParams(b)

	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	config := SmallTestConfig()
	cnnEval := NewCNNEvaluator(params, eval, nil, config)

	kernels := GenerateRandomKernels(config)
	require.NoError(b, cnnEval.SetKernels(kernels))

	input := make([]float64, 64)
	for i := range input {
		input[i] = float64(i) / 64.0
	}
	ptInput, _ := CfEcd(encoder, input, params, params.MaxLevel(), params.DefaultScale())
	ctInput, _ := enc.EncryptNew(ptInput)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := cnnEval.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkHESyncOverhead measures the overhead of HESync EVK management
// compared to in-memory EVKs for a simple rotation circuit.
func BenchmarkHESyncOverhead(b *testing.B) {
	params, sk := benchParams(b)

	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)

	// Generate rotation keys
	rotations := []int{1, 2, 3, 4, 5}
	galEls := params.GaloisElements(rotations)
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Create test ciphertext
	encoder := ckks.NewEncoder(params)
	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = float64(i)
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	require.NoError(b, encoder.Encode(values, pt))
	ct, err := enc.EncryptNew(pt)
	require.NoError(b, err)

	// Serialize EVKs
	tmpDir := b.TempDir()
	require.NoError(b, hesync.SerializeEVKSet(evk, tmpDir))

	// Trace
	tracingEvk := hesync.NewTracingEvaluationKeySet(evk)
	evalTrace := ckks.NewEvaluator(params, tracingEvk)
	for _, r := range rotations {
		_, err := evalTrace.RotateNew(ct, r)
		require.NoError(b, err)
	}
	trace := tracingEvk.GetTrace()

	// Plan
	profiler := hesync.NewProfiler()
	profiler.SetOpLatency(0, 100*1000) // 100µs
	idx, err := hesync.BuildIndex(tmpDir)
	require.NoError(b, err)
	for _, path := range idx.GaloisKeys {
		require.NoError(b, profiler.MeasureEVKLoadLatency(path, 1))
		break
	}
	plan := hesync.GeneratePlan(trace, profiler.GetProfile(), 0)

	b.Run("Baseline_InMemory", func(b *testing.B) {
		evalMem := ckks.NewEvaluator(params, evk)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, r := range rotations {
				_, err := evalMem.RotateNew(ct, r)
				if err != nil {
					b.Fatal(err)
				}
			}
		}
	})

	b.Run("HESync_DiskBacked", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			diskEvk := hesync.NewDiskEvaluationKeySet(idx, plan, 2)
			evalDisk := ckks.NewEvaluator(params, diskEvk)
			for _, r := range rotations {
				_, err := evalDisk.RotateNew(ct, r)
				if err != nil {
					b.Fatal(err)
				}
			}
			diskEvk.Stop()
		}
	})
}

// TestHESyncIntegration is an integration test that runs the full HESync
// pipeline: serialize, trace, plan, and disk-backed inference.
func TestHESyncIntegration(t *testing.T) {
	params, sk := testCKKSParams(t)

	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)
	encoder := ckks.NewEncoder(params)

	// Generate keys
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	config := SmallTestConfig()

	// Create CNN evaluator with in-memory keys (baseline)
	cnnBaseline := NewCNNEvaluator(params, eval, nil, config)
	kernels := GenerateRandomKernels(config)
	require.NoError(t, cnnBaseline.SetKernels(kernels))

	// Create encrypted input
	input := make([]float64, 64)
	for i := range input {
		input[i] = float64(i) / 64.0
	}
	ptInput, err := CfEcd(encoder, input, params, params.MaxLevel(), params.DefaultScale())
	require.NoError(t, err)
	ctInput, err := enc.EncryptNew(ptInput)
	require.NoError(t, err)

	// Baseline inference
	baselineOutputs, err := cnnBaseline.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
	require.NoError(t, err)

	// HESync inference
	tmpDir := t.TempDir()
	require.NoError(t, hesync.SerializeEVKSet(evk, tmpDir))

	// Trace
	tracingEvk := hesync.NewTracingEvaluationKeySet(evk)
	tracingEval := ckks.NewEvaluator(params, tracingEvk)
	cnnTracing := NewCNNEvaluator(params, tracingEval, nil, config)
	require.NoError(t, cnnTracing.SetKernels(kernels))
	_, err = cnnTracing.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
	require.NoError(t, err)
	trace := tracingEvk.GetTrace()

	// Plan
	profile := hesync.Profile{
		OpLatency:      map[int]time.Duration{0: time.Millisecond},
		EVKLoadLatency: time.Millisecond,
	}
	plan := hesync.GeneratePlan(trace, profile, 0)

	idx, err := hesync.BuildIndex(tmpDir)
	require.NoError(t, err)

	diskEvk := hesync.NewDiskEvaluationKeySet(idx, plan, 2)
	defer diskEvk.Stop()

	diskEval := ckks.NewEvaluator(params, diskEvk)
	cnnDisk := NewCNNEvaluator(params, diskEval, nil, config)
	require.NoError(t, cnnDisk.SetKernels(kernels))

	hesyncOutputs, err := cnnDisk.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
	require.NoError(t, err)

	// Compare results
	require.Equal(t, len(baselineOutputs), len(hesyncOutputs))

	maxDev, err := CompareResults(params, sk, baselineOutputs, hesyncOutputs)
	require.NoError(t, err)
	t.Logf("Max deviation between baseline and HESync: %.6e", maxDev)

	// Results should be identical (same key material from disk)
	require.Less(t, maxDev, 1e-3, "deviation too large between baseline and HESync")
}

// TestMemoryStats demonstrates memory tracking capability.
func TestMemoryStats(t *testing.T) {
	params, sk := testCKKSParams(t)

	kgen := rlwe.NewKeyGenerator(params)

	// Generate several rotation keys
	rotations := []int{1, 2, 4, 8, 16}
	galEls := params.GaloisElements(rotations)
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Measure baseline memory
	runtime.GC()
	var memBaseline runtime.MemStats
	runtime.ReadMemStats(&memBaseline)

	t.Logf("Baseline memory (all %d EVKs in memory):", len(galEls)+1)
	t.Logf("  HeapInuse: %.2f MB", float64(memBaseline.HeapInuse)/(1024*1024))
	t.Logf("  Sys: %.2f MB", float64(memBaseline.Sys)/(1024*1024))

	// Serialize and create disk-backed key set
	tmpDir := t.TempDir()
	require.NoError(t, hesync.SerializeEVKSet(evk, tmpDir))

	idx, err := hesync.BuildIndex(tmpDir)
	require.NoError(t, err)

	// Report file sizes
	var totalSize int64
	entries, _ := os.ReadDir(tmpDir)
	for _, entry := range entries {
		info, _ := entry.Info()
		totalSize += info.Size()
		t.Logf("  %s: %.2f KB", entry.Name(), float64(info.Size())/1024)
	}
	t.Logf("  Total on disk: %.2f MB", float64(totalSize)/(1024*1024))

	// Disk-backed key set only loads keys on demand
	plan := &hesync.Plan{TotalSteps: 0, Instructions: nil}
	diskEvk := hesync.NewDiskEvaluationKeySet(idx, plan, 2)
	defer diskEvk.Stop()

	loaded, relinLoaded := diskEvk.MemStats()
	t.Logf("DiskEvaluationKeySet: %d GaloisKeys loaded, relin=%v", loaded, relinLoaded)
	require.Equal(t, 0, loaded)
	require.False(t, relinLoaded)

	_ = evk // keep reference to prevent GC
}
