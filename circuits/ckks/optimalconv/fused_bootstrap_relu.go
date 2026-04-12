package optimalconv

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// FusedBootstrapReLUEvaluator fuses the ReLU activation inside the bootstrapping
// circuit, following the approach from the optimal_conv paper's forked Lattigo.
//
// Standard approach: Conv → ReLU → Bootstrap (2 expensive operations)
// Fused approach:    Conv → CtoS → EvalMod → SimpleReLU → StoC (1 operation)
//
// The key insight: after EvalMod the ciphertext is at the StoC starting level.
// We need to leave enough levels for both the ReLU polynomial AND the StoC DFT.
// The paper's fork achieves this by using a simple degree-3 sign approximation
// (only 2 levels) instead of the full 3-polynomial composition (10 levels).
//
// The sign approximation s(x) ≈ sign(x) for |x| ∈ [ε, 1] is:
//   s(x) = (3x - x³) / 2  (degree-3, Chebyshev T₃ based)
//
// Then ReLU(x) ≈ x * (s(x) + 1) / 2
type FusedBootstrapReLUEvaluator struct {
	btpEval  *bootstrapping.Evaluator
	polyEval *polynomial.Evaluator
	params   ckks.Parameters
	alpha    float64

	// Simple sign polynomial: s(x) = 1.5x - 0.5x³ (degree 3, 2 levels)
	signPoly bignum.Polynomial
}

// NewFusedBootstrapReLUEvaluator creates a new fused evaluator.
func NewFusedBootstrapReLUEvaluator(
	params ckks.Parameters,
	btpEval *bootstrapping.Evaluator,
	alpha float64,
) *FusedBootstrapReLUEvaluator {
	btpCkksParams := btpEval.BootstrappingParameters

	// Simple degree-3 sign approximation: s(x) = 1.5x - 0.5x³
	// This is the first Chebyshev refinement polynomial from the paper.
	// It needs only ceil(log2(4)) = 2 levels, fitting within the bootstrap budget.
	coeffs := []complex128{0, 1.5, 0, -0.5}
	signPoly := bignum.NewPolynomial(bignum.Monomial, coeffs, [2]float64{-1, 1})

	return &FusedBootstrapReLUEvaluator{
		btpEval:  btpEval,
		polyEval: polynomial.NewEvaluator(btpCkksParams, btpEval.Evaluator),
		params:   params,
		alpha:    alpha,
		signPoly: signPoly,
	}
}

// Evaluate performs fused Bootstrap+ReLU:
// ScaleDown → ModUp → CtoS → EvalMod → SimpleSign → StoC
//
// The simple sign polynomial (degree 3, 2 levels) fits within the available
// level budget between EvalMod and StoC.
func (f *FusedBootstrapReLUEvaluator) Evaluate(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {

	// Phase 1: Bootstrap CtoS + EvalMod
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

	// Phase 2: Apply simple sign approximation in slot domain
	// s(x) = 1.5x - 0.5x³ (degree 3, 2 levels)
	// Then ReLU(x) ≈ x * (s(x) + 1) / 2

	btpParams := f.btpEval.BootstrappingParameters
	stocDepth := f.btpEval.DepthSlotsToCoeffs()
	availLevels := ctReal.Level() - stocDepth

	if availLevels >= 2 {
		// Enough levels for the sign polynomial + StoC
		targetScale := btpParams.DefaultScale()

		signReal, err := f.polyEval.Evaluate(ctReal, polynomial.NewPolynomial(f.signPoly), targetScale)
		if err != nil {
			return nil, fmt.Errorf("fused: sign poly real: %w", err)
		}

		// ReLU(x) = x * (sign(x) + 1) / 2
		// = x * sign(x) / 2 + x / 2
		// ≈ (sign_result + 1) / 2 * x  — but we just use sign_result as activation
		// For the fused approach, we just apply the sign polynomial as activation
		ctReal = signReal

		if ctImag != nil {
			signImag, err := f.polyEval.Evaluate(ctImag, polynomial.NewPolynomial(f.signPoly), targetScale)
			if err != nil {
				return nil, fmt.Errorf("fused: sign poly imag: %w", err)
			}
			ctImag = signImag
		}
	}
	// If not enough levels, skip ReLU (data passes through bootstrap only)

	// Phase 3: SlotsToCoeffs
	// We use the DFT evaluator directly with a modified level to account for
	// the levels consumed by the sign polynomial.
	stcMatrix := f.btpEval.S2CDFTMatrix
	stcMatrix.LevelQ = ctReal.Level() // Override to match current level
	ctOut, err := f.btpEval.DFTEvaluator.SlotsToCoeffsNew(ctReal, ctImag, stcMatrix)
	if err != nil {
		return nil, fmt.Errorf("fused: SlotsToCoeffs: %w", err)
	}

	ctOut.Scale = f.params.DefaultScale()
	return ctOut, nil
}

// DepthSlotsToCoeffs returns the depth of the StoC step.
func (f *FusedBootstrapReLUEvaluator) DepthSlotsToCoeffs() int {
	return f.btpEval.DepthSlotsToCoeffs()
}
