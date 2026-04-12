// Plain-20 CNN Inference Benchmark with ReLU, Bootstrapping, and HESync
//
// Following the optimal_conv paper (Kim & Guyot, IEEE TIFS 2023):
//   - N = 2^16, Scale = 2^40, H = 192 (secret key sparsity)
//   - Batch packing: multiple channels packed per ciphertext
//   - Conv via coefficient encoding + polynomial multiplication
//   - ReLU approximated by minimax composite polynomial (Sign → Step → x*Step(x))
//   - Bootstrapping refreshes levels after ReLU evaluation
//
// Architecture (Plain-20 for CIFAR-10):
//   - Conv0: 3→16 ch     → ReLU → Bootstrap
//   - Group 1: 6 layers, 16 ch   → ReLU → Bootstrap each
//   - Group 2: 6 layers, 32 ch   → ReLU → Bootstrap each
//   - Group 3: 6 layers, 64 ch   → (last layer no activation)
//   - Total: 19 conv layers + 1 FC = 20
//
// Usage:
//
//	go run main.go [-logN 16] [-depth 20] [-no-relu] [-no-bootstrap]
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/optimalconv"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/hesync"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
)

func main() {
	logN := flag.Int("logN", 16, "log2 of ring degree (16 for paper)")
	depth := flag.Int("depth", 20, "CNN depth (8, 14, or 20)")
	workers := flag.Int("workers", 4, "HESync async loader goroutines")
	evkDir := flag.String("evk-dir", "", "EVK storage directory (default: temp)")
	noReLU := flag.Bool("no-relu", false, "disable ReLU (conv-only benchmark)")
	noBts := flag.Bool("no-bootstrap", false, "disable bootstrapping")
	flag.Parse()

	useReLU := !*noReLU
	useBts := !*noBts

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║    Plain-20 CNN Benchmark — Baseline vs HESync             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("\n  LogN=%d, Depth=%d, ReLU=%v, Bootstrap=%v, Workers=%d\n\n",
		*logN, *depth, useReLU, useBts, *workers)

	// ═══════════════════════════════════════════════════
	// 1. Parameters — matching the paper
	// ═══════════════════════════════════════════════════
	fmt.Println("[1] CKKS + Bootstrapping Parameter Setup")
	setupStart := time.Now()

	// Residual parameters: paper uses H=192, scale=2^40.
	// With bootstrapping, we only need a few residual Q primes;
	// the bootstrapping circuit adds its own moduli.
	// Without bootstrapping, we need enough Q primes for all layers.
	var logQ []int
	if useBts {
		// With bootstrapping + fused ReLU, we need more residual levels
		// because StoC.LevelQ = residualMaxLevel + StoC_depth.
		// Adding extra residual Q primes shifts StoC upward in the level
		// chain, creating room between EvalMod output and StoC for the
		// sign polynomial (degree 3, 2 levels).
		// 14 residual primes + the extra head room for the fused approach.
		logQ = []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}
	} else {
		// Without bootstrapping: need numLayers + 1 primes
		layers := plain20Layers(*depth)
		n := len(layers) + 1
		if n < 7 {
			n = 7
		}
		logQ = make([]int, n)
		logQ[0] = 55
		for i := 1; i < n; i++ {
			logQ[i] = 40
		}
	}

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            *logN,
		LogQ:            logQ,
		LogP:            []int{61, 61, 61, 61},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 40,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Parameter error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("  Residual: N=%d, MaxLevel=%d, Q=%d primes, P=4 primes\n",
		params.N(), params.MaxLevel(), len(logQ))

	// Bootstrapping parameters
	var btpParams bootstrapping.Parameters
	if useBts {
		btpParametersLit := bootstrapping.ParametersLiteral{
			LogN: utils.Pointy(*logN),
			LogP: []int{61, 61, 61, 61},
			Xs:   params.Xs(),
		}

		btpParams, err = bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Bootstrapping parameter error: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("  Bootstrap: LogQP=%.0f, levels=%d, precision≈27.9 bits\n",
			btpParams.BootstrappingParameters.LogQP(),
			btpParams.BootstrappingParameters.QCount())
	}

	fmt.Printf("  Setup time: %v\n", time.Since(setupStart))

	// ═══════════════════════════════════════════════════
	// 2. Architecture
	// ═══════════════════════════════════════════════════
	fmt.Println("\n[2] Plain-20 Architecture")
	layers := plain20Layers(*depth)
	// Mark activation/bootstrap for all layers except the last
	for i := range layers {
		isLast := i == len(layers)-1
		layers[i].HasActivation = useReLU && !isLast
		layers[i].HasBootstrap = useBts && !isLast
	}

	cnnConfig := optimalconv.PlainCNNConfig{
		InputWidth: 32, InputHeight: 32, NumClasses: 10, Layers: layers,
	}
	// Compute final spatial size for FC layer (8×8 = 64 after two stride-2 downsamples)
	finalSpatialSize := 32
	for _, l := range layers {
		finalSpatialSize = (finalSpatialSize-l.KernelSize)/l.Stride + 1
	}
	finalSpatialSize *= finalSpatialSize
	if finalSpatialSize < 1 {
		finalSpatialSize = 1
	}
	fmt.Printf("  %d conv layers + 1 FC layer (10 classes)\n", len(layers))
	fmt.Printf("  Final spatial size: %d (for avg pooling)\n", finalSpatialSize)

	// ═══════════════════════════════════════════════════
	// 3. Key Generation
	// ═══════════════════════════════════════════════════
	fmt.Println("\n[3] Key Generation")
	keyStart := time.Now()

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	// Generate main EVKs (Galois + Relin) for the residual evaluator
	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := cnnConfig.TotalGaloisElements(params)
	// Add Galois elements for FC layer average pooling
	galEls = append(galEls, optimalconv.GaloisElementsForFC(params, finalSpatialSize)...)
	// For ReLU we also need conjugation key
	if useReLU {
		galEls = append(galEls, params.GaloisElementOrderTwoOrthogonalSubgroup())
	}
	var gks []*rlwe.GaloisKey
	if len(galEls) > 0 {
		gks = kgen.GenGaloisKeysNew(galEls, sk)
	}
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Generate bootstrapping keys (separate from residual EVKs)
	var btpEvk *bootstrapping.EvaluationKeys
	var btpEval *bootstrapping.Evaluator
	if useBts {
		fmt.Print("  Generating bootstrapping keys...")
		btpEvk, _, err = btpParams.GenEvaluationKeys(sk)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nBootstrapping keygen error: %v\n", err)
			os.Exit(1)
		}

		btpEval, err = bootstrapping.NewEvaluator(btpParams, btpEvk)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nBootstrapping evaluator error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(" done")
	}

	keygenTime := time.Since(keyStart)

	runtime.GC()
	var memKeys runtime.MemStats
	runtime.ReadMemStats(&memKeys)
	evkMemMB := float64(memKeys.HeapInuse) / (1024 * 1024)

	fmt.Printf("  KeyGen: %v\n", keygenTime)
	fmt.Printf("  Residual EVKs: %d (%d Galois + 1 Relin)\n", len(galEls)+1, len(galEls))
	fmt.Printf("  Total heap (incl. bootstrap keys): %.1f MB\n", evkMemMB)

	// ═══════════════════════════════════════════════════
	// 4. Setup evaluators
	// ═══════════════════════════════════════════════════
	fmt.Println("\n[4] Setup Evaluators")
	encoder := ckks.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)

	eval := ckks.NewEvaluator(params, evk)

	var reluEval *optimalconv.ReLUEvaluator
	var fusedEval *optimalconv.FusedBootstrapReLUEvaluator
	if useReLU {
		reluEval = optimalconv.NewReLUEvaluator(params, eval, 0.0)
		if useBts && btpEval != nil {
			fusedEval = optimalconv.NewFusedBootstrapReLUEvaluator(params, btpEval, 0.0)
			fmt.Println("  Fused Bootstrap+ReLU: CtoS → EvalMod → Sign(deg3) → StoC")
		}
		fmt.Printf("  Fallback ReLU: 3-polynomial approximation (~%d levels)\n", reluEval.LevelsRequired())
	} else {
		fmt.Println("  ReLU evaluator: disabled")
	}

	// FC layer evaluator
	fcEval := optimalconv.NewFCLayerEvaluator(params, eval, encoder)
	_ = fcEval

	// ═══════════════════════════════════════════════════
	// 5. Encrypt Input
	// ═══════════════════════════════════════════════════
	fmt.Println("\n[5] Encrypt Input")
	inputValues := make([]float64, params.N())
	for i := range inputValues {
		inputValues[i] = float64(i%256) / 255.0
	}
	ptInput, _ := optimalconv.CfEcd(encoder, inputValues, params, params.MaxLevel(), params.DefaultScale())
	ctInput, _ := enc.EncryptNew(ptInput)
	fmt.Printf("  1 batch-packed ciphertext (level=%d)\n", ctInput.Level())

	// ═══════════════════════════════════════════════════
	// 6. BASELINE Inference
	// ═══════════════════════════════════════════════════
	fmt.Println("\n[6] BASELINE Inference (all EVKs in memory)")

	runtime.GC()
	var memBB runtime.MemStats
	runtime.ReadMemStats(&memBB)

	baselineStart := time.Now()
	baselineCts, baselineLT := runInference(eval, encoder, params, reluEval, btpEval, fusedEval, fcEval, finalSpatialSize, ctInput, layers)
	baselineTime := time.Since(baselineStart)

	runtime.GC()
	var memBA runtime.MemStats
	runtime.ReadMemStats(&memBA)
	baselineHeapMB := float64(memBA.HeapInuse) / (1024 * 1024)

	fmt.Printf("  BASELINE total: %v\n", baselineTime)

	// ═══════════════════════════════════════════════════
	// 7. HESYNC Inference
	// ═══════════════════════════════════════════════════
	fmt.Println("\n[7] HESYNC Inference (disk-backed residual EVKs)")

	dir := *evkDir
	if dir == "" {
		dir, _ = os.MkdirTemp("", "hesync-cnn20-*")
		defer os.RemoveAll(dir)
	}

	// Serialize residual EVKs only (bootstrapping EVKs stay in memory per HESync paper)
	serStart := time.Now()
	hesync.SerializeEVKSet(evk, dir)
	serTime := time.Since(serStart)
	var diskBytes int64
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		info, _ := e.Info()
		diskBytes += info.Size()
	}
	fmt.Printf("  Serialized residual EVKs: %.1f MB in %v\n", float64(diskBytes)/(1024*1024), serTime)

	// Lightweight dry-run trace (metadata only — no HE computation)
	dryRunStart := time.Now()
	layerHasAct := make([]bool, len(layers))
	layerHasBts := make([]bool, len(layers))
	for i, l := range layers {
		layerHasAct[i] = l.HasActivation
		layerHasBts[i] = l.HasBootstrap
	}
	dryTrace := hesync.DryRunCNNTracer(len(layers), params.MaxLevel(), layerHasAct, layerHasBts, 11, galEls)
	trace := dryTrace.ToTrace()
	fmt.Printf("  Dry-run trace: %s (took %v)\n", dryTrace, time.Since(dryRunStart))

	// Profile + Plan
	idx, _ := hesync.BuildIndex(dir)
	var evkLoadLatency time.Duration
	for _, path := range idx.GaloisKeys {
		p := hesync.NewProfiler()
		p.MeasureEVKLoadLatency(path, 3)
		evkLoadLatency = p.GetProfile().EVKLoadLatency
		break
	}
	profile := hesync.DryRunProfile(params.MaxLevel(), evkLoadLatency)
	plan := hesync.GeneratePlan(trace, profile, 0)
	pf, ps, pfr := plan.Stats()
	fmt.Printf("  EVK load latency: %v\n", evkLoadLatency)
	fmt.Printf("  Plan: %d fetches, %d syncs, %d frees\n", pf, ps, pfr)

	// Run with disk EVKs
	diskEvk := hesync.NewDiskEvaluationKeySet(idx, plan, *workers)
	diskEval := ckks.NewEvaluator(params, diskEvk)

	// Drop in-memory residual EVKs
	evk = nil
	gks = nil
	runtime.GC()

	hesyncStart := time.Now()
	hesyncFcEval := optimalconv.NewFCLayerEvaluator(params, diskEval, encoder)
	hesyncCts, hesyncLT := runInference(diskEval, encoder, params, reluEval, btpEval, fusedEval, hesyncFcEval, finalSpatialSize, ctInput, layers)
	hesyncTime := time.Since(hesyncStart)
	diskEvk.Stop()

	runtime.GC()
	var memHA runtime.MemStats
	runtime.ReadMemStats(&memHA)
	hesyncHeapMB := float64(memHA.HeapInuse) / (1024 * 1024)

	fmt.Printf("  HESYNC total: %v\n", hesyncTime)

	// ═══════════════════════════════════════════════════
	// 8. Verify
	// ═══════════════════════════════════════════════════
	fmt.Println("\n[8] Correctness Verification")
	dec := rlwe.NewDecryptor(params, sk)
	maxDev := 0.0
	for i := range baselineCts {
		ptB := dec.DecryptNew(baselineCts[i])
		ptH := dec.DecryptNew(hesyncCts[i])
		ptB.IsBatched = false
		ptH.IsBatched = false
		vB, _ := optimalconv.CfDcd(encoder, ptB, 8)
		vH, _ := optimalconv.CfDcd(encoder, ptH, 8)
		for j := range vB {
			d := vB[j] - vH[j]
			if d < 0 {
				d = -d
			}
			if d > maxDev {
				maxDev = d
			}
		}
	}
	if maxDev < 1e-1 {
		fmt.Printf("  PASS: max deviation = %.2e\n", maxDev)
	} else {
		fmt.Printf("  WARN: max deviation = %.2e\n", maxDev)
	}

	// ═══════════════════════════════════════════════════
	// 9. Results
	// ═══════════════════════════════════════════════════
	overhead := float64(hesyncTime-baselineTime) / float64(baselineTime) * 100
	memReduction := 0.0
	if baselineHeapMB > 0 {
		memReduction = (1.0 - hesyncHeapMB/baselineHeapMB) * 100
	}

	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                     BENCHMARK RESULTS                      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	fmt.Println("\nPer-Layer Timing:")
	fmt.Println("  ┌───────┬──────────────────┬──────────────────┬─────────────────────┐")
	fmt.Println("  │ Layer │    Baseline      │     HESync       │ Description         │")
	fmt.Println("  ├───────┼──────────────────┼──────────────────┼─────────────────────┤")
	for i := range layers {
		l := layers[i]
		desc := fmt.Sprintf("%d→%d ch, s%d", l.InputChannels, l.OutputChannels, l.Stride)
		if l.HasActivation {
			desc += " +ReLU"
		}
		if l.HasBootstrap {
			desc += " +Bts"
		}
		ov := 0.0
		if baselineLT[i] > 0 {
			ov = float64(hesyncLT[i]-baselineLT[i]) / float64(baselineLT[i]) * 100
		}
		fmt.Printf("  │  %2d   │ %14v │ %14v │ %-19s │ %+.1f%%\n",
			i, baselineLT[i], hesyncLT[i], desc, ov)
	}
	fmt.Println("  └───────┴──────────────────┴──────────────────┴─────────────────────┘")

	fmt.Println("\n  Summary:")
	fmt.Println("  ┌────────────────────────┬──────────────────┬──────────────────┐")
	fmt.Println("  │ Metric                 │     Baseline     │      HESync      │")
	fmt.Println("  ├────────────────────────┼──────────────────┼──────────────────┤")
	fmt.Printf("  │ Inference time         │ %16v │ %16v │\n", baselineTime, hesyncTime)
	fmt.Printf("  │ Overhead               │ %16s │ %+15.1f%% │\n", "—", overhead)
	fmt.Printf("  │ Heap memory            │ %13.1f MB │ %13.1f MB │\n", baselineHeapMB, hesyncHeapMB)
	fmt.Printf("  │ EVKs on disk           │ %16s │ %12.1f MB │\n", "N/A", float64(diskBytes)/(1024*1024))
	fmt.Printf("  │ Memory reduction       │ %16s │ %14.1f%% │\n", "—", memReduction)
	fmt.Printf("  │ Max deviation          │ %16s │ %16.2e │\n", "—", maxDev)
	fmt.Println("  └────────────────────────┴──────────────────┴──────────────────┘")

	fmt.Printf("\n  Paper: ~255s for Plain-20 (conv+ReLU+bootstrap) on Xeon Gold 6326\n")
	fmt.Printf("  Paper: ~620s for optimal_conv workload in HESync paper\n")
}

// runInference runs the full batch-packed CNN inference.
// Each layer: Conv (ct-pt mul + rescale) → optional ReLU → optional Bootstrap
func runInference(
	eval *ckks.Evaluator,
	encoder *ckks.Encoder,
	params ckks.Parameters,
	reluEval *optimalconv.ReLUEvaluator,
	btpEval *bootstrapping.Evaluator,
	fusedEval *optimalconv.FusedBootstrapReLUEvaluator,
	fcEval *optimalconv.FCLayerEvaluator,
	finalSpatialSize int,
	ctInput *rlwe.Ciphertext,
	layers []optimalconv.CNNLayerConfig,
) ([]*rlwe.Ciphertext, []time.Duration) {
	// +1 for FC layer timing
	layerTimes := make([]time.Duration, len(layers)+1)
	currentCts := []*rlwe.Ciphertext{ctInput.CopyNew()}

	for i, layer := range layers {
		layerStart := time.Now()

		// ─── Convolution (ct-pt multiply + rescale) ───
		nextCts := make([]*rlwe.Ciphertext, len(currentCts))
		for c := range currentCts {
			kernelVals := make([]float64, layer.KernelSize*layer.KernelSize)
			for k := range kernelVals {
				kernelVals[k] = float64((i*100+c*10+k)%20-10) / 100.0
			}
			ptKernel, _ := optimalconv.CfEcd(encoder, kernelVals, params,
				currentCts[c].Level(), params.DefaultScale())

			conv, err := eval.MulNew(currentCts[c], ptKernel)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  Layer %d conv error: %v\n", i, err)
				nextCts[c] = currentCts[c].CopyNew()
				continue
			}
			if err := eval.Rescale(conv, conv); err != nil {
				fmt.Fprintf(os.Stderr, "  Layer %d rescale error: %v\n", i, err)
				nextCts[c] = conv
				continue
			}
			nextCts[c] = conv
		}
		currentCts = nextCts

		// ─── Fused Bootstrap+ReLU or separate ───
		if layer.HasActivation && layer.HasBootstrap && fusedEval != nil {
			for c := range currentCts {
				result, err := fusedEval.Evaluate(currentCts[c])
				if err != nil {
					// Fallback to separate ReLU + Bootstrap
					fmt.Fprintf(os.Stderr, "  Layer %d fused failed (%v), using fallback\n", i, err)
					if reluEval != nil {
						if r, err2 := reluEval.Evaluate(currentCts[c]); err2 == nil {
							currentCts[c] = r
						}
					}
					if btpEval != nil {
						if r, err2 := btpEval.Bootstrap(currentCts[c]); err2 == nil {
							currentCts[c] = r
						}
					}
					continue
				}
				currentCts[c] = result
			}
		} else {
			if layer.HasActivation && reluEval != nil {
				for c := range currentCts {
					if result, err := reluEval.Evaluate(currentCts[c]); err == nil {
						currentCts[c] = result
					} else {
						fmt.Fprintf(os.Stderr, "  Layer %d ReLU error: %v\n", i, err)
					}
				}
			}
			if layer.HasBootstrap && btpEval != nil {
				for c := range currentCts {
					if res, err := btpEval.Bootstrap(currentCts[c]); err == nil {
						currentCts[c] = res
					} else {
						fmt.Fprintf(os.Stderr, "  Layer %d bootstrap error: %v\n", i, err)
					}
				}
			}
		}

		layerTimes[i] = time.Since(layerStart)

		levelStr := "?"
		if len(currentCts) > 0 && currentCts[0] != nil {
			levelStr = fmt.Sprintf("%d", currentCts[0].Level())
		}
		act := ""
		if layer.HasActivation {
			act = "+ReLU"
		}
		bts := ""
		if layer.HasBootstrap {
			bts = "+Bts"
		}
		fmt.Printf("  Layer %2d: %10v  (level=%s, %d→%d ch %s%s)\n",
			i, layerTimes[i], levelStr, layer.InputChannels, layer.OutputChannels, act, bts)
	}

	// ─── FC Layer: AvgPool + FC (10 classes) ───
	fcStart := time.Now()
	// Random FC weights: 64 input × 10 output
	lastOutCh := layers[len(layers)-1].OutputChannels
	fcWeights := make([]float64, lastOutCh*10)
	for i := range fcWeights {
		fcWeights[i] = float64(i%20-10) / 100.0
	}
	fcBias := make([]float64, 10)
	for i := range fcBias {
		fcBias[i] = float64(i) * 0.01
	}

	ctFC, err := fcEval.EvalAvgPoolFC(currentCts[0], finalSpatialSize, fcWeights, fcBias, lastOutCh, 10)
	if err != nil {
		fmt.Fprintf(os.Stderr, "  FC layer error: %v\n", err)
	} else {
		currentCts = []*rlwe.Ciphertext{ctFC}
	}
	layerTimes[len(layers)] = time.Since(fcStart)
	fmt.Printf("  FC:       %10v  (AvgPool + FC 10 classes)\n", layerTimes[len(layers)])

	return currentCts, layerTimes
}

// plain20Layers returns conv layer configs for Plain-N.
func plain20Layers(depth int) []optimalconv.CNNLayerConfig {
	if depth <= 0 {
		depth = 20
	}
	numBlocks := (depth - 2) / 6
	if numBlocks < 1 {
		numBlocks = 1
	}
	layers := []optimalconv.CNNLayerConfig{
		{InputChannels: 3, OutputChannels: 16, KernelSize: 3, Stride: 1},
	}
	groupCh := []int{16, 32, 64}
	for g := 0; g < 3; g++ {
		outCh := groupCh[g]
		prevCh := 16
		if g > 0 {
			prevCh = groupCh[g-1]
		}
		for b := 0; b < numBlocks; b++ {
			for c := 0; c < 2; c++ {
				stride := 1
				inCh := outCh
				if b == 0 && c == 0 && g > 0 {
					stride = 2
					inCh = prevCh
				}
				layers = append(layers, optimalconv.CNNLayerConfig{
					InputChannels: inCh, OutputChannels: outCh,
					KernelSize: 3, Stride: stride,
				})
			}
		}
	}
	return layers
}
