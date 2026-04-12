// ckks_optimal_conv demonstrates the HESync storage-assisted EVK management
// for CNN inference using the optimal_conv convolution method.
//
// This example:
//  1. Sets up CKKS parameters and generates keys
//  2. Encrypts a test input image
//  3. Runs CNN inference with all EVKs in memory (baseline)
//  4. Runs CNN inference with HESync (disk-backed EVKs)
//  5. Compares results and reports memory/timing statistics
//
// Usage:
//
//	go run main.go [flags]
//	  -hesync       enable HESync mode (default: true)
//	  -evk-dir      directory for serialized EVKs (default: temp dir)
//	  -logN         log2 of ring degree (default: 12, use 16 for paper benchmarks)
//	  -workers      number of async loader goroutines (default: 4)
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/optimalconv"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/hesync"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	enableHESync := flag.Bool("hesync", true, "enable HESync mode")
	evkDir := flag.String("evk-dir", "", "directory for serialized EVKs (default: temp dir)")
	logN := flag.Int("logN", 12, "log2 of ring degree (12 for fast test, 16 for paper benchmarks)")
	workers := flag.Int("workers", 4, "number of async loader goroutines")
	flag.Parse()

	fmt.Println("=== HESync + optimal_conv Benchmark ===")
	fmt.Printf("LogN: %d, HESync: %v, Workers: %d\n\n", *logN, *enableHESync, *workers)

	// Step 1: Setup parameters
	fmt.Println("[1/6] Setting up CKKS parameters...")
	paramLit := ckks.ParametersLiteral{
		LogN:            *logN,
		LogQ:            []int{55, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61},
		LogDefaultScale: 45,
	}

	params, err := ckks.NewParametersFromLiteral(paramLit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating parameters: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("  N=%d, MaxLevel=%d, MaxSlots=%d\n", params.N(), params.MaxLevel(), params.MaxSlots())

	// Step 2: Generate keys
	fmt.Println("[2/6] Generating keys...")
	start := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	config := optimalconv.SmallTestConfig()
	galEls := config.TotalGaloisElements(params)
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)
	fmt.Printf("  Generated %d GaloisKeys + 1 RelinKey in %v\n", len(galEls), time.Since(start))

	// Report EVK memory
	runtime.GC()
	var memAfterKeys runtime.MemStats
	runtime.ReadMemStats(&memAfterKeys)
	fmt.Printf("  EVK memory: %.2f MB (HeapInuse)\n", float64(memAfterKeys.HeapInuse)/(1024*1024))

	// Step 3: Encrypt test input
	fmt.Println("[3/6] Encrypting test input...")
	encoder := ckks.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)

	input := make([]float64, 64)
	for i := range input {
		input[i] = float64(i) / 64.0
	}

	ptInput, err := optimalconv.CfEcd(encoder, input, params, params.MaxLevel(), params.DefaultScale())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding: %v\n", err)
		os.Exit(1)
	}

	ctInput, err := enc.EncryptNew(ptInput)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error encrypting: %v\n", err)
		os.Exit(1)
	}

	// Step 4: Baseline inference
	fmt.Println("[4/6] Running baseline inference (all EVKs in memory)...")
	evalMem := ckks.NewEvaluator(params, evk)
	cnnBaseline := optimalconv.NewCNNEvaluator(params, evalMem, nil, config)
	kernels := optimalconv.GenerateRandomKernels(config)
	if err := cnnBaseline.SetKernels(kernels); err != nil {
		fmt.Fprintf(os.Stderr, "Error setting kernels: %v\n", err)
		os.Exit(1)
	}

	runtime.GC()
	var memBeforeBaseline runtime.MemStats
	runtime.ReadMemStats(&memBeforeBaseline)

	baselineStart := time.Now()
	baselineOutputs, err := cnnBaseline.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
	baselineTime := time.Since(baselineStart)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Baseline error: %v\n", err)
		os.Exit(1)
	}

	runtime.GC()
	var memAfterBaseline runtime.MemStats
	runtime.ReadMemStats(&memAfterBaseline)

	fmt.Printf("  Baseline time: %v\n", baselineTime)
	fmt.Printf("  Baseline peak memory: %.2f MB\n", float64(memAfterBaseline.HeapInuse)/(1024*1024))

	if !*enableHESync {
		fmt.Println("\nHESync disabled. Exiting.")
		return
	}

	// Step 5: HESync inference
	fmt.Println("[5/6] Running HESync inference (disk-backed EVKs)...")

	dir := *evkDir
	if dir == "" {
		dir, err = os.MkdirTemp("", "hesync-evk-*")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating temp dir: %v\n", err)
			os.Exit(1)
		}
		defer os.RemoveAll(dir)
	}

	// Serialize EVKs to disk
	serStart := time.Now()
	if err := hesync.SerializeEVKSet(evk, dir); err != nil {
		fmt.Fprintf(os.Stderr, "Serialize error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Serialized EVKs to %s in %v\n", dir, time.Since(serStart))

	// Trace
	tracingEvk := hesync.NewTracingEvaluationKeySet(evk)
	tracingEval := ckks.NewEvaluator(params, tracingEvk)
	cnnTracing := optimalconv.NewCNNEvaluator(params, tracingEval, nil, config)
	if err := cnnTracing.SetKernels(kernels); err != nil {
		fmt.Fprintf(os.Stderr, "Tracing error: %v\n", err)
		os.Exit(1)
	}
	_, err = cnnTracing.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Tracing inference error: %v\n", err)
		os.Exit(1)
	}
	trace := tracingEvk.GetTrace()
	fmt.Printf("  Trace: %d EVK accesses recorded\n", len(trace.Entries))

	// Profile
	profiler := hesync.NewProfiler()
	profiler.SetOpLatency(0, time.Millisecond)
	idx, err := hesync.BuildIndex(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Build index error: %v\n", err)
		os.Exit(1)
	}
	for _, path := range idx.GaloisKeys {
		if err := profiler.MeasureEVKLoadLatency(path, 3); err != nil {
			fmt.Fprintf(os.Stderr, "Profile error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("  EVK load latency: %v\n", profiler.GetProfile().EVKLoadLatency)
		break
	}

	// Plan
	plan := hesync.GeneratePlan(trace, profiler.GetProfile(), 0)
	fetches, syncs, frees := plan.Stats()
	fmt.Printf("  Plan: %d fetches, %d syncs, %d frees\n", fetches, syncs, frees)

	// Run with disk-backed EVKs
	diskEvk := hesync.NewDiskEvaluationKeySet(idx, plan, *workers)
	defer diskEvk.Stop()

	diskEval := ckks.NewEvaluator(params, diskEvk)
	cnnDisk := optimalconv.NewCNNEvaluator(params, diskEval, nil, config)
	if err := cnnDisk.SetKernels(kernels); err != nil {
		fmt.Fprintf(os.Stderr, "Disk CNN error: %v\n", err)
		os.Exit(1)
	}

	runtime.GC()
	var memBeforeHESync runtime.MemStats
	runtime.ReadMemStats(&memBeforeHESync)

	hesyncStart := time.Now()
	hesyncOutputs, err := cnnDisk.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
	hesyncTime := time.Since(hesyncStart)
	if err != nil {
		fmt.Fprintf(os.Stderr, "HESync inference error: %v\n", err)
		os.Exit(1)
	}

	runtime.GC()
	var memAfterHESync runtime.MemStats
	runtime.ReadMemStats(&memAfterHESync)

	fmt.Printf("  HESync time: %v\n", hesyncTime)
	fmt.Printf("  HESync peak memory: %.2f MB\n", float64(memAfterHESync.HeapInuse)/(1024*1024))

	// Step 6: Compare results
	fmt.Println("[6/6] Comparing results...")
	maxDev, err := optimalconv.CompareResults(params, sk, baselineOutputs, hesyncOutputs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Compare error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n=== Results ===\n")
	fmt.Printf("Max deviation: %.6e\n", maxDev)
	fmt.Printf("Baseline time: %v\n", baselineTime)
	fmt.Printf("HESync time:   %v (%.1f%% overhead)\n", hesyncTime,
		float64(hesyncTime-baselineTime)/float64(baselineTime)*100)
	fmt.Printf("Baseline peak mem: %.2f MB\n", float64(memAfterBaseline.HeapInuse)/(1024*1024))
	fmt.Printf("HESync peak mem:   %.2f MB\n", float64(memAfterHESync.HeapInuse)/(1024*1024))

	if maxDev < 1e-3 {
		fmt.Println("\nSUCCESS: Results match within tolerance.")
	} else {
		fmt.Println("\nWARNING: Results differ significantly.")
	}
}
