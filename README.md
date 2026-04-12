# HESync: Storage-Assisted Encrypted CNN Inference on Lattigo v6

This repository extends the [Lattigo](https://github.com/tuneinsight/lattigo) v6 homomorphic encryption library with techniques from two papers:

1. **HESync** — "Storage-Assisted Encrypted Neural Network Inference for Reduced Memory Requirements" (Chung, Park, Moon — IEEE CAL 2025)
2. **optimal_conv** — "Optimized Privacy-Preserving CNN Inference with Fully Homomorphic Encryption" (Kim, Guyot — IEEE TIFS 2023)

The implementation includes a full **Plain-20 CNN inference benchmark** with fused Bootstrap+ReLU evaluation, HESync disk-backed EVK management, and end-to-end comparison of baseline vs HESync approaches.

---

## What Was Implemented

### 1. HESync Infrastructure (`hesync/`)

Storage-assisted evaluation key (EVK) management that reduces peak memory by **95%+** by storing EVKs on disk and prefetching them on-demand.

| File | Purpose |
|------|---------|
| `tracer.go` | `TracingEvaluationKeySet` — wraps `rlwe.EvaluationKeySet` to record EVK access patterns |
| `profiler.go` | Level-aware operation timing and EVK load latency measurement |
| `planner.go` | Backward-walk algorithm generating fetch/sync/free prefetch plans |
| `prefetcher.go` | Goroutine pool for async EVK loading from disk |
| `disk_evk_set.go` | `DiskEvaluationKeySet` implementing `rlwe.EvaluationKeySet` backed by disk with prefetching |
| `serialize.go` | Per-key serialization/deserialization to individual binary files |
| `dry_run.go` | Lightweight trace generator (microseconds) without actual HE computation |
| `hesync_test.go` | Unit and integration tests |

**Key design:** `DiskEvaluationKeySet` implements the existing `rlwe.EvaluationKeySet` interface, making it a drop-in replacement for `MemEvaluationKeySet`. No changes to existing evaluators are needed.

### 2. optimal_conv Primitives (`circuits/ckks/optimalconv/`)

Homomorphic CNN inference building blocks from the optimal_conv paper.

| File | Purpose |
|------|---------|
| `cfencode.go` | Coefficient encoding (`CfEcd`/`CfDcd`) — embeds values as polynomial coefficients for negacyclic convolution |
| `convolution.go` | Single convolution via polynomial multiplication in R = Z[X]/(X^N+1) |
| `batch_convolution.go` | Batch convolution (Algorithm 1) — packs multiple channels per ciphertext |
| `packlwes.go` | PackLWEs (Algorithm 2) — tree-based LWE-to-RLWE packing |
| `relu.go` | 3-polynomial ReLU approximation (degree 7+7+13) from the paper's `evalReLU` |
| `fused_bootstrap_relu.go` | **Fused Bootstrap+ReLU**: CtoS → EvalMod → scale_StoC → 3-poly ReLU → StoC |
| `fc_layer.go` | Fully connected layer with global average pooling |
| `cnn.go` | `CNNEvaluator` orchestrating full CNN inference |
| `params.go` | Plain-20 architecture and parameter definitions |
| `hesync_integration.go` | HESync integration for CNN inference comparison |

### 3. Lattigo Bootstrapping Modifications (`circuits/ckks/bootstrapping/`)

Modifications to the Lattigo v6 bootstrapping module to support fused Bootstrap+ReLU, replicating the approach from [dwkim606/test_lattigo](https://github.com/dwkim606/test_lattigo).

#### `parameters_literal.go` — Added fields:
```go
ReLUDepth    *int  // Number of extra levels between EvalMod and StoC for fused ReLU
ReLULogScale *int  // Log2 of scale for the ReLU-dedicated Q primes
```

#### `parameters.go` — Modified level chain construction:

The standard bootstrap level chain is `StoC → EvalMod → CtoS` with zero gap between EvalMod and StoC. The modification inserts `ReLUDepth` extra Q primes between EvalMod and StoC:

```
Before: ... StoC levels | EvalMod levels | CtoS levels ...
After:  ... StoC levels | ReLU levels | EvalMod levels | CtoS levels ...
```

This replicates the fork's `ReLUEvalModuli` concept — dedicated moduli for polynomial ReLU evaluation inside the bootstrap circuit.

**Modified line** (originally `Mod1.LevelQ = StoC.LevelQ + Mod1.Depth()`):
```go
Mod1ParametersLiteral.LevelQ = S2CParams.LevelQ + reluDepth + Mod1ParametersLiteral.Depth()
```

#### `evaluator.go` — Updated consistency check:

The level consistency check now accounts for the ReLU gap:
```go
Mod1.LevelQ - Mod1.Depth() - btpParams.ReLUDepth == StoC.LevelQ
```

#### `parameters.go` — Added `ReLUDepth` to Parameters struct:
```go
type Parameters struct {
    // ... existing fields ...
    ReLUDepth int  // Extra levels between EvalMod and StoC for fused ReLU
}
```

### 4. Fused Bootstrap+ReLU (`circuits/ckks/optimalconv/fused_bootstrap_relu.go`)

Replicates the fork's `BootstrappConv_CtoS` + `evalReLU` + `BootstrappConv_StoC` approach:

1. **ScaleDown → ModUp → CtoS → EvalMod** (standard bootstrap first half)
2. **scale_StoC correction**: Apply `qDiff * Scale / postscale` as scalar multiply + rescale (matching the fork's `MultByConst(scale_StoC)`)
3. **3-polynomial ReLU** in the dedicated ReLU levels (degree 7+7+13, 11 levels)
4. **StoC** (standard bootstrap second half, ct lands exactly at StoC.LevelQ)

### 5. Plain-20 CNN Benchmark (`examples/singleparty/ckks_cnn20_benchmark/`)

End-to-end benchmark matching the optimal_conv paper's Plain-20 architecture:

- **Architecture**: Conv0 + 3 groups × 3 BasicBlocks × 2 conv = 19 conv layers + FC (10 classes)
- **Parameters**: N=2^16, Scale=2^30, H=192, matching fork's SET VII
- **Bootstrap**: K=25, SinDeg=63, SinRescal=2, EvalModLogScale=55
- **ReLU**: Full 3-polynomial fused inside bootstrap with ReLUDepth=12
- **FC layer**: Average pooling + fully connected (64→10)
- **Dual mode**: Runs baseline (all EVKs in memory) then HESync (disk-backed) and compares

---

## Build & Run

```bash
# Build all packages
go build ./...

# Run tests
go test -timeout=0 ./hesync/...
go test -timeout=0 ./circuits/ckks/optimalconv/...

# Run Plain-20 benchmark (convolution only, fast)
go run ./examples/singleparty/ckks_cnn20_benchmark/main.go \
    -logN 16 -depth 20 -no-relu -no-bootstrap

# Run Plain-20 benchmark (full: conv + fused ReLU + bootstrap + HESync)
go run ./examples/singleparty/ckks_cnn20_benchmark/main.go \
    -logN 16 -depth 20

# Run at smaller scale for quick testing
go run ./examples/singleparty/ckks_cnn20_benchmark/main.go \
    -logN 14 -depth 20 -no-relu -no-bootstrap
```

---

## Benchmark Results

### Plain-20 at N=2^16 (Conv + Fused ReLU + Bootstrap + FC)

| Metric | Baseline | HESync | Change |
|--------|----------|--------|--------|
| Inference time | ~540s | ~525s | -3% |
| Heap memory | ~23 GB | ~1 GB | **-95.5%** |
| EVKs on disk | N/A | ~4.5 GB | — |
| Correctness | — | 0.00 deviation | exact match |
| Dry-run trace | — | ~70 µs | instant |

### Comparison with Papers

| | This Implementation | optimal_conv Paper |
|--|--------------------|--------------------|
| Baseline time | ~540s | 255s |
| Memory reduction | **95.5%** | — |
| HESync overhead | -3% (faster) | — |
| Per-layer (fused) | ~30s | ~13s |

The ~2× gap vs the paper's 255s is due to Lattigo v6 vs the fork's Lattigo v2 implementation differences (v6 uses more general-purpose polynomial evaluation, different NTT implementation, and abstractions that add overhead).

---

## Architecture

```
hesync/                          HESync EVK management infrastructure
├── tracer.go                    EVK access pattern recording
├── profiler.go                  Operation latency measurement
├── planner.go                   Backward-walk prefetch planning
├── prefetcher.go                Async goroutine pool EVK loader
├── disk_evk_set.go              Disk-backed EvaluationKeySet
├── serialize.go                 Per-key binary serialization
├── dry_run.go                   Lightweight CNN trace generator
└── hesync_test.go               Tests

circuits/ckks/optimalconv/       CNN inference primitives
├── cfencode.go                  Coefficient encoding
├── convolution.go               Polynomial multiplication convolution
├── batch_convolution.go         Multi-channel batch convolution
├── packlwes.go                  LWE packing (Algorithm 2)
├── relu.go                      3-polynomial ReLU approximation
├── fused_bootstrap_relu.go      Fused CtoS→EvalMod→ReLU→StoC
├── fc_layer.go                  FC layer with average pooling
├── cnn.go                       CNN evaluator
├── params.go                    Plain-20 architecture config
├── hesync_integration.go        HESync + CNN integration
├── cnn_test.go                  Tests
└── benchmark_test.go            Benchmarks

circuits/ckks/bootstrapping/     Modified Lattigo bootstrap (3 files)
├── parameters_literal.go        + ReLUDepth, ReLULogScale fields
├── parameters.go                + ReLU level insertion in modulus chain
└── evaluator.go                 + ReLUDepth-aware consistency check

examples/singleparty/
├── ckks_optimal_conv/main.go    Simple HESync demo
└── ckks_cnn20_benchmark/main.go Plain-20 full benchmark
```

---

## References

- **HESync paper**: Chung, Park, Moon. "Storage-Assisted Encrypted Neural Network Inference for Reduced Memory Requirements." IEEE CAL, 2025.
- **optimal_conv paper**: Kim, Guyot. "Optimized Privacy-Preserving CNN Inference with Fully Homomorphic Encryption." IEEE TIFS, 2023.
- **optimal_conv source**: https://github.com/dwkim606/optimal_conv
- **Forked Lattigo v2**: https://github.com/dwkim606/test_lattigo
- **Lattigo v6**: https://github.com/tuneinsight/lattigo
