// Package optimalconv implements optimized privacy-preserving CNN inference
// with fully homomorphic encryption, based on the paper by Kim and Guyot.
// It provides coefficient encoding, batch convolution, and PackLWEs primitives.
package optimalconv

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// CfEcd encodes a float64 slice into a plaintext using coefficient encoding.
// Unlike slot encoding, this embeds values directly as polynomial coefficients
// scaled by the plaintext scale: pt(X) = Δ·(v[0] + v[1]·X + ... + v[n-1]·X^{n-1}).
//
// This encoding does NOT preserve point-wise multiplication. Instead,
// ciphertext multiplication results in negacyclic polynomial convolution
// in the plaintext domain, which is exactly what is needed for computing
// Conv(I, K) via polynomial multiplication.
//
// The plaintext is created at the given level and scale.
func CfEcd(encoder *ckks.Encoder, values []float64, params ckks.Parameters, level int, scale rlwe.Scale) (*rlwe.Plaintext, error) {

	if len(values) > params.N() {
		return nil, fmt.Errorf("CfEcd: len(values)=%d exceeds ring degree N=%d", len(values), params.N())
	}

	pt := ckks.NewPlaintext(params, level)
	pt.Scale = scale
	pt.IsBatched = false

	if err := encoder.Encode(values, pt); err != nil {
		return nil, fmt.Errorf("CfEcd: %w", err)
	}

	return pt, nil
}

// CfDcd decodes a coefficient-encoded plaintext back to a float64 slice.
// It reverses the CfEcd operation, recovering the original values from
// the polynomial coefficients.
//
// The parameter n specifies how many coefficients to decode (up to N).
func CfDcd(encoder *ckks.Encoder, pt *rlwe.Plaintext, n int) ([]float64, error) {

	if pt.IsBatched {
		return nil, fmt.Errorf("CfDcd: plaintext is batched (slot-encoded), expected coefficient encoding")
	}

	values := make([]float64, n)
	if err := encoder.Decode(pt, values); err != nil {
		return nil, fmt.Errorf("CfDcd: %w", err)
	}

	return values, nil
}
