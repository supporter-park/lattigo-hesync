package hesync

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// testParams returns small CKKS parameters for fast testing.
func testParams(t *testing.T) (ckks.Parameters, *rlwe.SecretKey) {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{55, 45, 45, 45},
		LogP:            []int{60},
		LogDefaultScale: 45,
	})
	require.NoError(t, err)

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	return params, sk
}

func TestTracingEvaluationKeySet(t *testing.T) {
	params, sk := testParams(t)
	kgen := rlwe.NewKeyGenerator(params)

	// Generate some Galois keys and a relin key
	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := params.GaloisElements([]int{1, 2, 4})
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Wrap with tracing
	tracing := NewTracingEvaluationKeySet(evk)

	// Simulate some EVK accesses
	_, err := tracing.GetGaloisKey(galEls[0])
	require.NoError(t, err)
	_, err = tracing.GetRelinearizationKey()
	require.NoError(t, err)
	_, err = tracing.GetGaloisKey(galEls[1])
	require.NoError(t, err)
	_, err = tracing.GetGaloisKey(galEls[0])
	require.NoError(t, err)

	// Verify the trace
	trace := tracing.GetTrace()
	require.Len(t, trace.Entries, 4)

	require.Equal(t, KeyTypeGalois, trace.Entries[0].Type)
	require.Equal(t, galEls[0], trace.Entries[0].GaloisEl)
	require.Equal(t, 0, trace.Entries[0].Step)

	require.Equal(t, KeyTypeRelin, trace.Entries[1].Type)
	require.Equal(t, 1, trace.Entries[1].Step)

	require.Equal(t, KeyTypeGalois, trace.Entries[2].Type)
	require.Equal(t, galEls[1], trace.Entries[2].GaloisEl)
	require.Equal(t, 2, trace.Entries[2].Step)

	require.Equal(t, KeyTypeGalois, trace.Entries[3].Type)
	require.Equal(t, galEls[0], trace.Entries[3].GaloisEl)
	require.Equal(t, 3, trace.Entries[3].Step)

	// Verify GetGaloisKeysList still works
	list := tracing.GetGaloisKeysList()
	require.Len(t, list, 3)

	// Test reset
	tracing.Reset()
	trace = tracing.GetTrace()
	require.Len(t, trace.Entries, 0)
}

func TestSerializeAndBuildIndex(t *testing.T) {
	params, sk := testParams(t)
	kgen := rlwe.NewKeyGenerator(params)

	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := params.GaloisElements([]int{1, 2, 4})
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Serialize to temp directory
	tmpDir := t.TempDir()
	err := SerializeEVKSet(evk, tmpDir)
	require.NoError(t, err)

	// Verify files exist
	require.FileExists(t, filepath.Join(tmpDir, "rlk.bin"))
	for _, galEl := range galEls {
		require.FileExists(t, filepath.Join(tmpDir, galoisKeyFilename(galEl)))
	}

	// Build index
	idx, err := BuildIndex(tmpDir)
	require.NoError(t, err)
	require.NotEmpty(t, idx.RelinKeyPath)
	require.Len(t, idx.GaloisKeys, 3)

	// Verify all galois elements are in the index
	for _, galEl := range galEls {
		_, ok := idx.GaloisKeys[galEl]
		require.True(t, ok, "galois element %d not found in index", galEl)
	}

	// Load individual keys and verify they're valid
	for _, galEl := range galEls {
		gk, err := LoadGaloisKey(idx.GaloisKeys[galEl])
		require.NoError(t, err)
		require.Equal(t, galEl, gk.GaloisElement)
	}

	loadedRlk, err := LoadRelinearizationKey(idx.RelinKeyPath)
	require.NoError(t, err)
	require.NotNil(t, loadedRlk)
}

func TestPlanGeneration(t *testing.T) {
	// Create a simple trace
	trace := Trace{
		Entries: []TraceEntry{
			{Step: 0, Type: KeyTypeGalois, GaloisEl: 100},
			{Step: 1, Type: KeyTypeRelin},
			{Step: 2, Type: KeyTypeGalois, GaloisEl: 200},
			{Step: 3, Type: KeyTypeGalois, GaloisEl: 100}, // reuse of key 100
		},
	}

	// Create a simple profile
	profile := Profile{
		OpLatency:      map[int]time.Duration{0: 10 * time.Millisecond},
		EVKLoadLatency: 5 * time.Millisecond,
	}

	plan := GeneratePlan(trace, profile, 0)
	require.NotNil(t, plan)
	require.Equal(t, 4, plan.TotalSteps)

	// Verify plan has fetch, sync, and free instructions
	fetches, syncs, frees := plan.Stats()
	require.Greater(t, fetches, 0, "plan should have fetch instructions")
	require.Greater(t, syncs, 0, "plan should have sync instructions")
	require.Greater(t, frees, 0, "plan should have free instructions")

	// Verify string representation doesn't panic
	s := plan.String()
	require.NotEmpty(t, s)
}

func TestDiskEvaluationKeySetIntegration(t *testing.T) {
	params, sk := testParams(t)
	kgen := rlwe.NewKeyGenerator(params)

	// Generate keys
	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := params.GaloisElements([]int{1, 2})
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Step 1: Serialize keys to disk
	tmpDir := t.TempDir()
	err := SerializeEVKSet(evk, tmpDir)
	require.NoError(t, err)

	// Step 2: Create a trace by wrapping the in-memory key set
	tracing := NewTracingEvaluationKeySet(evk)

	// Simulate an inference that uses gk[0], rlk, gk[1], gk[0]
	_, err = tracing.GetGaloisKey(galEls[0])
	require.NoError(t, err)
	_, err = tracing.GetRelinearizationKey()
	require.NoError(t, err)
	_, err = tracing.GetGaloisKey(galEls[1])
	require.NoError(t, err)
	_, err = tracing.GetGaloisKey(galEls[0])
	require.NoError(t, err)

	trace := tracing.GetTrace()

	// Step 3: Profile (use synthetic timings for test)
	profile := Profile{
		OpLatency:      map[int]time.Duration{0: time.Millisecond},
		EVKLoadLatency: time.Millisecond,
	}

	// Step 4: Generate plan
	plan := GeneratePlan(trace, profile, 0)

	// Step 5: Create DiskEvaluationKeySet
	idx, err := BuildIndex(tmpDir)
	require.NoError(t, err)

	diskEvk := NewDiskEvaluationKeySet(idx, plan, 2)
	defer diskEvk.Stop()

	// Step 6: Verify it works - replay the same access pattern
	gk, err := diskEvk.GetGaloisKey(galEls[0])
	require.NoError(t, err)
	require.Equal(t, galEls[0], gk.GaloisElement)

	loadedRlk, err := diskEvk.GetRelinearizationKey()
	require.NoError(t, err)
	require.NotNil(t, loadedRlk)

	gk, err = diskEvk.GetGaloisKey(galEls[1])
	require.NoError(t, err)
	require.Equal(t, galEls[1], gk.GaloisElement)

	gk, err = diskEvk.GetGaloisKey(galEls[0])
	require.NoError(t, err)
	require.Equal(t, galEls[0], gk.GaloisElement)

	// Step 7: Verify GetGaloisKeysList works (no disk I/O)
	list := diskEvk.GetGaloisKeysList()
	require.Len(t, list, 2)
}

func TestDiskEvaluationKeySetWithEvaluator(t *testing.T) {
	params, sk := testParams(t)
	kgen := rlwe.NewKeyGenerator(params)

	// Generate keys for rotation by 1
	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := params.GaloisElements([]int{1})
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Serialize to disk
	tmpDir := t.TempDir()
	require.NoError(t, SerializeEVKSet(evk, tmpDir))

	// Encrypt a test vector
	encoder := ckks.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)
	dec := rlwe.NewDecryptor(params, sk)

	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = float64(i)
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel())
	require.NoError(t, encoder.Encode(values, pt))
	ct, err := enc.EncryptNew(pt)
	require.NoError(t, err)

	// Run with in-memory keys (baseline)
	evalMem := ckks.NewEvaluator(params, evk)
	rotatedMem, err := evalMem.RotateNew(ct, 1)
	require.NoError(t, err)

	// Run with disk-backed keys
	// First trace the operation
	tracing := NewTracingEvaluationKeySet(evk)
	evalTrace := ckks.NewEvaluator(params, tracing)
	_, err = evalTrace.RotateNew(ct, 1)
	require.NoError(t, err)
	trace := tracing.GetTrace()

	// Generate plan
	profile := Profile{
		OpLatency:      map[int]time.Duration{0: time.Millisecond},
		EVKLoadLatency: time.Millisecond,
	}
	plan := GeneratePlan(trace, profile, 0)

	idx, err := BuildIndex(tmpDir)
	require.NoError(t, err)
	diskEvk := NewDiskEvaluationKeySet(idx, plan, 2)
	defer diskEvk.Stop()

	evalDisk := ckks.NewEvaluator(params, diskEvk)
	rotatedDisk, err := evalDisk.RotateNew(ct, 1)
	require.NoError(t, err)

	// Decrypt both results and compare
	ptMem := dec.DecryptNew(rotatedMem)
	ptDisk := dec.DecryptNew(rotatedDisk)

	valuesMem := make([]float64, params.MaxSlots())
	valuesDisk := make([]float64, params.MaxSlots())
	require.NoError(t, encoder.Decode(ptMem, valuesMem))
	require.NoError(t, encoder.Decode(ptDisk, valuesDisk))

	// Results should be identical (same key material)
	for i := range valuesMem {
		require.InDelta(t, valuesMem[i], valuesDisk[i], 1e-6,
			"mismatch at slot %d: mem=%.6f disk=%.6f", i, valuesMem[i], valuesDisk[i])
	}
}

func TestProfiler(t *testing.T) {
	profiler := NewProfiler()

	// Set synthetic latencies
	profiler.SetOpLatency(0, 10*time.Millisecond)
	profiler.SetOpLatency(1, 8*time.Millisecond)
	profiler.SetEVKLoadLatency(5 * time.Millisecond)

	profile := profiler.GetProfile()
	require.Equal(t, 10*time.Millisecond, profile.GetOpLatency(0))
	require.Equal(t, 8*time.Millisecond, profile.GetOpLatency(1))

	// Fallback: unknown level returns max
	require.Equal(t, 10*time.Millisecond, profile.GetOpLatency(99))

	require.Equal(t, 5*time.Millisecond, profile.EVKLoadLatency)
}

// galoisKeyFilename returns the expected filename for a serialized GaloisKey.
func galoisKeyFilename(galEl uint64) string {
	return fmt.Sprintf("%s%d%s", galoisKeyPrefix, galEl, galoisKeySuffix)
}
