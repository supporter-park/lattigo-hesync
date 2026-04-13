package optimalconv

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// FusedBootstrapReLUEvaluator replicates the fork's BootstrappConv_CtoS +
// evalReLU + BootstrappConv_StoC approach from github.com/dwkim606/test_lattigo.
//
// Key techniques from the fork:
//  1. Split bootstrap: CtoS → EvalMod → [scale_StoC correction] → ReLU → StoC
//  2. scale_StoC correction: after EvalMod, apply the StoC diff scale as a scalar
//     multiply + rescale (instead of distributing it across StoC DFT levels)
//  3. 3-polynomial ReLU (degree 7+7+13) in dedicated ReLUEvalModuli levels
//  4. StoC with unit scaling (scale correction already applied)
//
// Requires ReLUDepth > 0 in the bootstrapping ParametersLiteral.
type FusedBootstrapReLUEvaluator struct {
	btpEval  *bootstrapping.Evaluator
	polyEval *polynomial.Evaluator
	params   ckks.Parameters

	// Paper's 3-polynomial sign approximation
	signPoly1 bignum.Polynomial // degree 7
	signPoly2 bignum.Polynomial // degree 7
	signPoly3 bignum.Polynomial // degree 13
	aconst    float64

	reluDepth int

	// Pre-computed scale correction factor (avoids recomputing per evaluation)
	scaleStoC float64
}

// NewFusedBootstrapReLUEvaluator creates a new fused evaluator.
func NewFusedBootstrapReLUEvaluator(
	params ckks.Parameters,
	btpEval *bootstrapping.Evaluator,
	btpParams bootstrapping.Parameters,
	alpha float64,
) (*FusedBootstrapReLUEvaluator, error) {
	btpCkksParams := btpEval.BootstrappingParameters
	reluDepth := btpParams.ReLUDepth

	if reluDepth < 12 {
		return nil, fmt.Errorf("ReLUDepth=%d too small (need >= 12: 11 for 3-poly ReLU + 1 for scale correction)", reluDepth)
	}

	aconst := (alpha + 1) / 2.0
	bconst := (1 - alpha) / 2.0

	coeffs1 := []complex128{0, 10.8541842577442, 0, -62.2833925211098, 0, 114.369227820443, 0, -62.8023496973074}
	coeffs2 := []complex128{0, 4.13976170985111, 0, -5.84997640211679, 0, 2.94376255659280, 0, -0.454530437460152}
	coeffs3Raw := []complex128{0, 3.29956739043733, 0, -7.84227260291355, 0, 12.8907764115564, 0, -12.4917112584486, 0, 6.94167991428074, 0, -2.04298067399942, 0, 0.246407138926031}
	coeffs3 := make([]complex128, len(coeffs3Raw))
	for i := range coeffs3Raw {
		coeffs3[i] = coeffs3Raw[i] * complex(bconst, 0)
	}

	// Pre-compute scale_StoC correction factor
	Q0 := float64(btpCkksParams.Q()[0])
	qDiff := Q0 / math.Exp2(math.Round(math.Log2(Q0)))
	postscale := math.Exp2(math.Round(math.Log2(Q0/btpEval.Mod1Parameters.MessageRatio()))) / btpEval.Mod1Parameters.MessageRatio()
	scaleStoC := qDiff * btpCkksParams.DefaultScale().Float64() / postscale

	return &FusedBootstrapReLUEvaluator{
		btpEval:   btpEval,
		polyEval:  polynomial.NewEvaluator(btpCkksParams, btpEval.Evaluator),
		params:    params,
		signPoly1: bignum.NewPolynomial(bignum.Monomial, coeffs1, [2]float64{-1, 1}),
		signPoly2: bignum.NewPolynomial(bignum.Monomial, coeffs2, [2]float64{-1, 1}),
		signPoly3: bignum.NewPolynomial(bignum.Monomial, coeffs3, [2]float64{-1, 1}),
		aconst:    aconst,
		reluDepth: reluDepth,
		scaleStoC: scaleStoC,
	}, nil
}

// Evaluate performs the full fused CtoS → EvalMod → scale_StoC → ReLU → StoC.
func (f *FusedBootstrapReLUEvaluator) Evaluate(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {

	// Phase 1: Standard bootstrap CtoS + EvalMod
	ctScaled, _, err := f.btpEval.ScaleDown(ct.CopyNew())
	if err != nil {
		return nil, fmt.Errorf("fused: ScaleDown: %w", err)
	}
	ctModUp, err := f.btpEval.ModUp(ctScaled)
	if err != nil {
		return nil, fmt.Errorf("fused: ModUp: %w", err)
	}
	ctReal, ctImag, err := f.btpEval.CoeffsToSlots(ctModUp)
	if err != nil {
		return nil, fmt.Errorf("fused: CoeffsToSlots: %w", err)
	}
	ctReal, err = f.btpEval.EvalMod(ctReal)
	if err != nil {
		return nil, fmt.Errorf("fused: EvalMod real: %w", err)
	}
	if ctImag != nil {
		ctImag, err = f.btpEval.EvalMod(ctImag)
		if err != nil {
			return nil, fmt.Errorf("fused: EvalMod imag: %w", err)
		}
	}

	// Phase 2: Apply scale_StoC correction (fork's BootstrappConv_CtoS approach)
	// This applies the SlotsToCoeffs diff scale as a single scalar multiply + rescale,
	// so the ReLU operates on properly scaled values and StoC can use unit scaling.
	Q0 := float64(f.btpEval.BootstrappingParameters.Q()[0])
	qDiff := Q0 / math.Exp2(math.Round(math.Log2(Q0)))
	postscale := math.Exp2(math.Round(math.Log2(Q0/f.btpEval.Mod1Parameters.MessageRatio()))) / f.btpEval.Mod1Parameters.MessageRatio()
	scaleStoC := qDiff * f.btpEval.BootstrappingParameters.DefaultScale().Float64() / postscale

	if err = f.btpEval.Mul(ctReal, scaleStoC, ctReal); err != nil {
		return nil, fmt.Errorf("fused: scaleStoC real: %w", err)
	}
	if err = f.btpEval.Rescale(ctReal, ctReal); err != nil {
		return nil, fmt.Errorf("fused: rescale real: %w", err)
	}
	ctReal.Scale = f.btpEval.BootstrappingParameters.DefaultScale()

	if ctImag != nil {
		if err = f.btpEval.Mul(ctImag, scaleStoC, ctImag); err != nil {
			return nil, fmt.Errorf("fused: scaleStoC imag: %w", err)
		}
		if err = f.btpEval.Rescale(ctImag, ctImag); err != nil {
			return nil, fmt.Errorf("fused: rescale imag: %w", err)
		}
		ctImag.Scale = f.btpEval.BootstrappingParameters.DefaultScale()
	}

	// Phase 3: Full 3-polynomial ReLU in the dedicated ReLU levels
	targetScale := f.btpEval.BootstrappingParameters.DefaultScale()

	ctReal, err = f.evalSign(ctReal, targetScale)
	if err != nil {
		return nil, fmt.Errorf("fused: sign real: %w", err)
	}
	if ctImag != nil {
		ctImag, err = f.evalSign(ctImag, targetScale)
		if err != nil {
			return nil, fmt.Errorf("fused: sign imag: %w", err)
		}
	}

	// Phase 4: StoC (ReLU levels consumed, ct should now be at StoC.LevelQ)
	ctOut, err := f.btpEval.SlotsToCoeffs(ctReal, ctImag)
	if err != nil {
		return nil, fmt.Errorf("fused: SlotsToCoeffs: %w", err)
	}
	ctOut.Scale = f.params.DefaultScale()
	return ctOut, nil
}

// evalSign applies the 3-polynomial sign approximation + final x * (sign + aconst).
func (f *FusedBootstrapReLUEvaluator) evalSign(ct *rlwe.Ciphertext, targetScale rlwe.Scale) (*rlwe.Ciphertext, error) {
	ctIn := ct.CopyNew()

	// s1(x) — degree 7, 3 levels
	ct1, err := f.polyEval.Evaluate(ct, polynomial.NewPolynomial(f.signPoly1), targetScale)
	if err != nil {
		return nil, fmt.Errorf("poly1: %w", err)
	}
	ct1.Scale = ctIn.Scale

	// s2(s1(x)) — degree 7, 3 levels
	ct2, err := f.polyEval.Evaluate(ct1, polynomial.NewPolynomial(f.signPoly2), targetScale)
	if err != nil {
		return nil, fmt.Errorf("poly2: %w", err)
	}
	ct2.Scale = ctIn.Scale

	// s3(s2(s1(x))) — degree 13, 4 levels
	ct3, err := f.polyEval.Evaluate(ct2, polynomial.NewPolynomial(f.signPoly3), targetScale)
	if err != nil {
		return nil, fmt.Errorf("poly3: %w", err)
	}
	ct3.Scale = ctIn.Scale

	// + aconst
	if err := f.btpEval.Add(ct3, f.aconst, ct3); err != nil {
		return nil, fmt.Errorf("add aconst: %w", err)
	}

	// Drop input level to match
	for ctIn.Level() > ct3.Level() {
		f.btpEval.DropLevel(ctIn, 1)
	}

	// x * (sign(x) + aconst)
	result, err := f.btpEval.MulRelinNew(ct3, ctIn)
	if err != nil {
		return nil, fmt.Errorf("final mul: %w", err)
	}
	if err := f.btpEval.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("final rescale: %w", err)
	}
	return result, nil
}
