// Specialized polynomial evaluation helpers that reduce overhead from the
// general-purpose Paterson-Stockmeyer framework for the fixed ReLU polynomials.
//
// Key optimizations over calling polyEval.Evaluate() directly:
//   - PowerBasis created WITHOUT CopyNew (~31 MB saved per eval at LogN=16)
//   - In-place EvalMod (avoids 2x CopyNew: 1 in EvalMod + 1 in PowerBasis)
//   - No redundant ctIn copy in evalSign
//   - Per-phase timing instrumentation for profiling
package optimalconv

import (
	"fmt"
	"math/big"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/mod1"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	commonpoly "github.com/tuneinsight/lattigo/v6/circuits/common/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// newPowerBasisNoCopy creates a PowerBasis that references the input ciphertext
// directly instead of copying it. This is safe when:
//   - The caller does not modify ct during evaluation
//   - ct has degree 1 (standard ciphertext)
//
// Saves ~31 MB of allocation at LogN=16 per call.
func newPowerBasisNoCopy(ct *rlwe.Ciphertext, basis bignum.Basis) commonpoly.PowerBasis {
	return commonpoly.PowerBasis{
		Value: map[int]*rlwe.Ciphertext{1: ct},
		Basis: basis,
	}
}

// EvalSignTiming captures per-phase timing for profiling the evalSign hot path.
type EvalSignTiming struct {
	Poly1    time.Duration
	Poly2    time.Duration
	Poly3    time.Duration
	AddConst time.Duration
	FinalMul time.Duration
	Total    time.Duration
}

func (t EvalSignTiming) String() string {
	return fmt.Sprintf("poly1=%v poly2=%v poly3=%v add=%v mul=%v total=%v",
		t.Poly1, t.Poly2, t.Poly3, t.AddConst, t.FinalMul, t.Total)
}

// FusedBootstrapTiming captures per-phase timing for the full fused evaluation.
type FusedBootstrapTiming struct {
	ScaleDown    time.Duration
	ModUp        time.Duration
	CoeffsToSlot time.Duration
	EvalMod      time.Duration
	ScaleCorr    time.Duration
	SignEval     EvalSignTiming
	SlotsToCoeff time.Duration
	Total        time.Duration
}

func (t FusedBootstrapTiming) String() string {
	return fmt.Sprintf(
		"ScaleDown=%v ModUp=%v CtoS=%v EvalMod=%v ScaleCorr=%v Sign={%v} StoC=%v Total=%v",
		t.ScaleDown, t.ModUp, t.CoeffsToSlot, t.EvalMod, t.ScaleCorr, t.SignEval, t.SlotsToCoeff, t.Total)
}

// evalSignFast applies the 3-polynomial sign approximation with optimized
// PowerBasis handling. Same algorithm as evalSign but:
//   - No CopyNew of input (saves ~31 MB)
//   - PowerBasis without copy (saves ~31 MB per poly eval)
//
// Returns result and timing information.
func (f *FusedBootstrapReLUEvaluator) evalSignFast(ct *rlwe.Ciphertext, targetScale rlwe.Scale) (*rlwe.Ciphertext, EvalSignTiming, error) {
	var timing EvalSignTiming
	totalStart := time.Now()

	// Save reference to original input for final multiplication.
	// The polynomial evaluator does NOT modify the input ciphertext
	// (it creates its own PowerBasis copy internally, or we pass a no-copy version),
	// so we can safely reference ct directly instead of CopyNew.
	ctIn := ct

	// s1(x) — degree 7, 3 levels
	t0 := time.Now()
	pb1 := newPowerBasisNoCopy(ct, bignum.Monomial)
	ct1, err := f.polyEval.EvaluateFromPowerBasis(pb1, polynomial.NewPolynomial(f.signPoly1), targetScale)
	if err != nil {
		return nil, timing, fmt.Errorf("poly1: %w", err)
	}
	ct1.Scale = ct.Scale
	timing.Poly1 = time.Since(t0)

	// s2(s1(x)) — degree 7, 3 levels
	t0 = time.Now()
	pb2 := newPowerBasisNoCopy(ct1, bignum.Monomial)
	ct2, err := f.polyEval.EvaluateFromPowerBasis(pb2, polynomial.NewPolynomial(f.signPoly2), targetScale)
	if err != nil {
		return nil, timing, fmt.Errorf("poly2: %w", err)
	}
	ct2.Scale = ct1.Scale
	timing.Poly2 = time.Since(t0)

	// s3(s2(s1(x))) — degree 13, 4 levels
	t0 = time.Now()
	pb3 := newPowerBasisNoCopy(ct2, bignum.Monomial)
	ct3, err := f.polyEval.EvaluateFromPowerBasis(pb3, polynomial.NewPolynomial(f.signPoly3), targetScale)
	if err != nil {
		return nil, timing, fmt.Errorf("poly3: %w", err)
	}
	ct3.Scale = ct2.Scale
	timing.Poly3 = time.Since(t0)

	// + aconst
	t0 = time.Now()
	if err := f.btpEval.Add(ct3, f.aconst, ct3); err != nil {
		return nil, timing, fmt.Errorf("add aconst: %w", err)
	}
	timing.AddConst = time.Since(t0)

	// Drop input level to match
	t0 = time.Now()
	for ctIn.Level() > ct3.Level() {
		f.btpEval.DropLevel(ctIn, 1)
	}

	// x * (sign(x) + aconst)
	result, err := f.btpEval.MulRelinNew(ct3, ctIn)
	if err != nil {
		return nil, timing, fmt.Errorf("final mul: %w", err)
	}
	if err := f.btpEval.Rescale(result, result); err != nil {
		return nil, timing, fmt.Errorf("final rescale: %w", err)
	}
	timing.FinalMul = time.Since(t0)
	timing.Total = time.Since(totalStart)

	return result, timing, nil
}

// evalModFast is an optimized version of mod1.Evaluator.EvaluateNew that:
//   - Works in-place on the input (no CopyNew of the input ciphertext)
//   - Uses no-copy PowerBasis for the Chebyshev polynomial evaluation
//
// This saves ~62 MB of allocation per call at LogN=16 (31 MB for the input copy
// + 31 MB for the PowerBasis copy).
//
// Safe because after EvalMod returns, the caller replaces the input reference.
func evalModFast(
	eval *ckks.Evaluator,
	polyEval *polynomial.Evaluator,
	evm mod1.Parameters,
	ct *rlwe.Ciphertext,
) (*rlwe.Ciphertext, error) {

	if ct.Level() < evm.LevelQ {
		return nil, fmt.Errorf("evalModFast: ct.Level() %d < evm.LevelQ %d", ct.Level(), evm.LevelQ)
	}

	if ct.Level() > evm.LevelQ {
		eval.DropLevel(ct, ct.Level()-evm.LevelQ)
	}

	// Work in-place instead of CopyNew: set scale directly on ct.
	// The caller must not use ct after this function returns.
	res := ct
	res.Scale = evm.ScalingFactor()

	// Compute target scales for double angle formula
	Qi := eval.GetParameters().Q()
	targetScale := res.Scale
	for i := 0; i < evm.DoubleAngle; i++ {
		targetScale = targetScale.Mul(rlwe.NewScale(Qi[ct.Level()-evm.Mod1Poly.Depth()-evm.DoubleAngle+i+1]))
		targetScale.Value.Sqrt(&targetScale.Value)
	}

	// Change of variable for Chebyshev evaluation
	if evm.Mod1Type == mod1.CosDiscrete || evm.Mod1Type == mod1.CosContinuous {
		offset := new(big.Float).Sub(&evm.Mod1Poly.B, &evm.Mod1Poly.A)
		offset.Mul(offset, new(big.Float).SetFloat64(evm.IntervalShrinkFactor()))
		offset.Quo(new(big.Float).SetFloat64(-0.5), offset)
		if err := eval.Add(res, offset, res); err != nil {
			return nil, fmt.Errorf("evalModFast: offset: %w", err)
		}
	}

	// Double angle setup
	sqrt2pi := complex(evm.Sqrt2Pi, 0)
	mod1Poly := evm.Mod1Poly // scaling=1 case, no coefficient modification needed

	// Chebyshev evaluation using no-copy PowerBasis
	pb := newPowerBasisNoCopy(res, bignum.Chebyshev)
	var err error
	if res, err = polyEval.EvaluateFromPowerBasis(pb, mod1Poly, rlwe.NewScale(targetScale)); err != nil {
		return nil, fmt.Errorf("evalModFast: Chebyshev: %w", err)
	}

	// Double angle formula
	for i := 0; i < evm.DoubleAngle; i++ {
		sqrt2pi *= sqrt2pi

		if err = eval.MulRelin(res, res, res); err != nil {
			return nil, fmt.Errorf("evalModFast: double angle mul: %w", err)
		}
		if err = eval.Add(res, res, res); err != nil {
			return nil, fmt.Errorf("evalModFast: double angle add: %w", err)
		}
		if err = eval.Add(res, -sqrt2pi, res); err != nil {
			return nil, fmt.Errorf("evalModFast: double angle sub: %w", err)
		}
		if err = eval.Rescale(res, res); err != nil {
			return nil, fmt.Errorf("evalModFast: double angle rescale: %w", err)
		}
	}

	// ArcSine (if needed — not used with CosDiscrete + DoubleAngle)
	if evm.Mod1InvPoly != nil {
		pb2 := newPowerBasisNoCopy(res, evm.Mod1InvPoly.Basis)
		if res, err = polyEval.EvaluateFromPowerBasis(pb2, *evm.Mod1InvPoly, res.Scale); err != nil {
			return nil, fmt.Errorf("evalModFast: ArcSine: %w", err)
		}
	}

	res.Scale = ct.Scale
	return res, nil
}

// EvaluateFast is the optimized version of Evaluate with reduced allocations
// and per-phase timing instrumentation.
func (f *FusedBootstrapReLUEvaluator) EvaluateFast(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, FusedBootstrapTiming, error) {
	var timing FusedBootstrapTiming
	totalStart := time.Now()

	// Phase 1: Standard bootstrap CtoS + EvalMod
	t0 := time.Now()
	ctScaled, _, err := f.btpEval.ScaleDown(ct.CopyNew())
	if err != nil {
		return nil, timing, fmt.Errorf("fused: ScaleDown: %w", err)
	}
	timing.ScaleDown = time.Since(t0)

	t0 = time.Now()
	ctModUp, err := f.btpEval.ModUp(ctScaled)
	if err != nil {
		return nil, timing, fmt.Errorf("fused: ModUp: %w", err)
	}
	timing.ModUp = time.Since(t0)

	t0 = time.Now()
	ctReal, ctImag, err := f.btpEval.CoeffsToSlots(ctModUp)
	if err != nil {
		return nil, timing, fmt.Errorf("fused: CoeffsToSlots: %w", err)
	}
	timing.CoeffsToSlot = time.Since(t0)

	// Optimized EvalMod: in-place + no-copy PowerBasis
	t0 = time.Now()
	mod1Eval := f.btpEval.Mod1Evaluator
	ctReal, err = evalModFast(f.btpEval.Evaluator, mod1Eval.PolynomialEvaluator, mod1Eval.Parameters, ctReal)
	if err != nil {
		return nil, timing, fmt.Errorf("fused: EvalMod real: %w", err)
	}
	if ctImag != nil {
		ctImag, err = evalModFast(f.btpEval.Evaluator, mod1Eval.PolynomialEvaluator, mod1Eval.Parameters, ctImag)
		if err != nil {
			return nil, timing, fmt.Errorf("fused: EvalMod imag: %w", err)
		}
	}
	timing.EvalMod = time.Since(t0)

	// Phase 2: Apply scale_StoC correction
	t0 = time.Now()
	if err = f.applyScaleCorrection(ctReal); err != nil {
		return nil, timing, fmt.Errorf("fused: scaleStoC real: %w", err)
	}
	if ctImag != nil {
		if err = f.applyScaleCorrection(ctImag); err != nil {
			return nil, timing, fmt.Errorf("fused: scaleStoC imag: %w", err)
		}
	}
	timing.ScaleCorr = time.Since(t0)

	// Phase 3: Full 3-polynomial ReLU in dedicated levels (optimized path)
	targetScale := f.btpEval.BootstrappingParameters.DefaultScale()

	var signTiming EvalSignTiming
	ctReal, signTiming, err = f.evalSignFast(ctReal, targetScale)
	if err != nil {
		return nil, timing, fmt.Errorf("fused: sign real: %w", err)
	}
	timing.SignEval = signTiming

	if ctImag != nil {
		ctImag, _, err = f.evalSignFast(ctImag, targetScale)
		if err != nil {
			return nil, timing, fmt.Errorf("fused: sign imag: %w", err)
		}
	}

	// Phase 4: StoC
	t0 = time.Now()
	ctOut, err := f.btpEval.SlotsToCoeffs(ctReal, ctImag)
	if err != nil {
		return nil, timing, fmt.Errorf("fused: SlotsToCoeffs: %w", err)
	}
	ctOut.Scale = f.params.DefaultScale()
	timing.SlotsToCoeff = time.Since(t0)
	timing.Total = time.Since(totalStart)

	return ctOut, timing, nil
}

// applyScaleCorrection applies the SlotsToCoeffs diff scale as a scalar
// multiply + rescale (fork's BootstrappConv_CtoS approach).
func (f *FusedBootstrapReLUEvaluator) applyScaleCorrection(ct *rlwe.Ciphertext) error {
	if err := f.btpEval.Mul(ct, f.scaleStoC, ct); err != nil {
		return err
	}
	if err := f.btpEval.Rescale(ct, ct); err != nil {
		return err
	}
	ct.Scale = f.btpEval.BootstrappingParameters.DefaultScale()
	return nil
}
