package optimalconv

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ConvSingle computes a single convolution Conv(I, K) homomorphically via
// polynomial multiplication. Both the image and kernel must be coefficient-encoded
// (IsBatched=false). The image is an encrypted ciphertext and the kernel is a plaintext.
//
// The result is a ciphertext whose coefficient-decoded values yield Conv(I, K)
// as defined in the optimal_conv paper: the (i*w+j)-th coefficient of
// I(X)·K(X) computed in R = Z[X]/(X^N + 1) is Conv(I, K)_{i,j}.
func ConvSingle(eval *ckks.Evaluator, ctImage *rlwe.Ciphertext, ptKernel *rlwe.Plaintext) (*rlwe.Ciphertext, error) {

	if ctImage.IsBatched {
		return nil, fmt.Errorf("ConvSingle: ctImage must be coefficient-encoded (IsBatched=false)")
	}

	if ptKernel.IsBatched {
		return nil, fmt.Errorf("ConvSingle: ptKernel must be coefficient-encoded (IsBatched=false)")
	}

	// Ciphertext-plaintext multiplication computes polynomial product in R,
	// which yields negacyclic convolution — exactly Conv(I, K).
	opOut, err := eval.MulNew(ctImage, ptKernel)
	if err != nil {
		return nil, fmt.Errorf("ConvSingle: %w", err)
	}

	// Rescale to manage scale growth from multiplication
	if err = eval.Rescale(opOut, opOut); err != nil {
		return nil, fmt.Errorf("ConvSingle: rescale: %w", err)
	}

	return opOut, nil
}

// EncodeKernel encodes a convolution kernel as a coefficient-encoded plaintext.
// The kernel values are laid out so that polynomial multiplication with a
// coefficient-encoded image yields the convolution result.
//
// For a kernel K of size kh x kw, the values are packed as:
// K(X) = sum_{i,j} K_{i,j} · X^{e·k·(i'-k)·w + j}
// where the indexing follows the paper's convention.
func EncodeKernel(encoder *ckks.Encoder, kernel [][]float64, kh, kw int, params ckks.Parameters, level int, scale rlwe.Scale) (*rlwe.Plaintext, error) {
	N := params.N()

	coeffs := make([]float64, N)

	for i := 0; i < kh; i++ {
		for j := 0; j < kw; j++ {
			idx := i*kw + j
			if idx < N {
				coeffs[idx] = kernel[i][j]
			}
		}
	}

	return CfEcd(encoder, coeffs, params, level, scale)
}

// EncodeImage encodes an image (or feature map) as a coefficient-encoded plaintext.
// The image values of size ih x iw are packed as polynomial coefficients:
// I(X) = sum_{i,j} I_{i,j} · X^{i·w + j}
func EncodeImage(encoder *ckks.Encoder, image []float64, params ckks.Parameters, level int, scale rlwe.Scale) (*rlwe.Plaintext, error) {
	return CfEcd(encoder, image, params, level, scale)
}
