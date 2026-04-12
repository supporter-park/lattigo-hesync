package optimalconv

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// ReLUEvaluator evaluates the ReLU activation function using the polynomial
// approximation from the optimal_conv paper (Kim & Guyot, IEEE TIFS 2023).
//
// The approximation uses three successive polynomial compositions:
//   s1(x) = sign approximation (degree 7)
//   s2(x) = sign refinement (degree 7)
//   s3(x) = final sign with leaky parameter (degree 13)
//   ReLU(x) ≈ x * (bconst * s3(s2(s1(x))) + aconst)
//
// This is significantly faster than the full minimax composite polynomial
// (which uses 8 compositions with 80-bit precision), because the 3-polynomial
// approximation only needs ~12 levels vs ~50+ for the full minimax.
//
// Polynomial level consumption: each degree-7 poly needs 3 levels, degree-13
// needs 4 levels, so total ~10 levels for the sign approximation, plus 1 for
// the final multiplication with x.
type ReLUEvaluator struct {
	eval    *ckks.Evaluator
	polyEval *polynomial.Evaluator
	params  ckks.Parameters

	// Polynomials from the paper
	signPoly1 bignum.Polynomial // degree 7: first sign approximation
	signPoly2 bignum.Polynomial // degree 7: sign refinement
	signPoly3 bignum.Polynomial // degree 13: final sign with leaky param

	// Leaky ReLU parameters
	alpha  float64 // negative slope (0 for standard ReLU)
	aconst float64 // (alpha + 1) / 2
	bconst float64 // (1 - alpha) / 2
}

// NewReLUEvaluator creates a new ReLU evaluator using the paper's polynomial
// approximation. alpha=0 for standard ReLU, alpha>0 for leaky ReLU.
func NewReLUEvaluator(params ckks.Parameters, eval *ckks.Evaluator, alpha float64) *ReLUEvaluator {
	aconst := (alpha + 1) / 2.0
	bconst := (1 - alpha) / 2.0

	// Polynomial 1: first sign approximation (degree 7, monomial basis)
	// From the paper's evalReLU function
	coeffs1 := []complex128{
		0.0, 10.8541842577442, 0.0, -62.2833925211098,
		0.0, 114.369227820443, 0.0, -62.8023496973074,
	}

	// Polynomial 2: sign refinement (degree 7, monomial basis)
	coeffs2 := []complex128{
		0.0, 4.13976170985111, 0.0, -5.84997640211679,
		0.0, 2.94376255659280, 0.0, -0.454530437460152,
	}

	// Polynomial 3: final sign (degree 13, monomial basis), scaled by bconst
	coeffs3Raw := []complex128{
		0.0, 3.29956739043733, 0.0, -7.84227260291355,
		0.0, 12.8907764115564, 0.0, -12.4917112584486,
		0.0, 6.94167991428074, 0.0, -2.04298067399942,
		0.0, 0.246407138926031,
	}
	coeffs3 := make([]complex128, len(coeffs3Raw))
	for i := range coeffs3Raw {
		coeffs3[i] = coeffs3Raw[i] * complex(bconst, 0)
	}

	// Create bignum.Polynomial objects with monomial basis in [-1, 1]
	poly1 := bignum.NewPolynomial(bignum.Monomial, coeffs1, [2]float64{-1, 1})
	poly2 := bignum.NewPolynomial(bignum.Monomial, coeffs2, [2]float64{-1, 1})
	poly3 := bignum.NewPolynomial(bignum.Monomial, coeffs3, [2]float64{-1, 1})

	return &ReLUEvaluator{
		eval:      eval,
		polyEval:  polynomial.NewEvaluator(params, eval),
		params:    params,
		signPoly1: poly1,
		signPoly2: poly2,
		signPoly3: poly3,
		alpha:     alpha,
		aconst:    aconst,
		bconst:    bconst,
	}
}

// Evaluate computes ReLU(x) ≈ x * (bconst * s3(s2(s1(x))) + aconst)
// where s1, s2, s3 are the three sign approximation polynomials.
//
// Level consumption: approximately 11 levels total
// (3 for poly1 + 3 for poly2 + 4 for poly3 + 1 for final multiplication).
func (r *ReLUEvaluator) Evaluate(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	targetScale := r.params.DefaultScale()

	// Save input for final multiplication (ReLU = x * sign(x))
	ctIn := ct.CopyNew()

	// s1(x) - first sign approximation
	ct1, err := r.polyEval.Evaluate(ct, polynomial.NewPolynomial(r.signPoly1), targetScale)
	if err != nil {
		return nil, fmt.Errorf("ReLU poly1: %w", err)
	}

	// s2(s1(x)) - sign refinement
	ct2, err := r.polyEval.Evaluate(ct1, polynomial.NewPolynomial(r.signPoly2), targetScale)
	if err != nil {
		return nil, fmt.Errorf("ReLU poly2: %w", err)
	}

	// s3(s2(s1(x))) - final sign with leaky parameter
	ct3, err := r.polyEval.Evaluate(ct2, polynomial.NewPolynomial(r.signPoly3), targetScale)
	if err != nil {
		return nil, fmt.Errorf("ReLU poly3: %w", err)
	}

	// Add aconst: sign_result + aconst
	if err := r.eval.Add(ct3, r.aconst, ct3); err != nil {
		return nil, fmt.Errorf("ReLU add aconst: %w", err)
	}

	// Drop input level to match sign output level
	for ctIn.Level() > ct3.Level() {
		r.eval.DropLevel(ctIn, 1)
	}

	// Final: x * (sign_result + aconst)
	result, err := r.eval.MulRelinNew(ct3, ctIn)
	if err != nil {
		return nil, fmt.Errorf("ReLU final mul: %w", err)
	}

	if err := r.eval.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("ReLU final rescale: %w", err)
	}

	return result, nil
}

// LevelsRequired returns the number of levels consumed by one ReLU evaluation.
// Degree 7 polynomial: depth = ceil(log2(7+1)) = 3 levels
// Degree 13 polynomial: depth = ceil(log2(13+1)) = 4 levels
// Total: 3 + 3 + 4 + 1 (final mul) = 11 levels
func (r *ReLUEvaluator) LevelsRequired() int {
	return 11
}
