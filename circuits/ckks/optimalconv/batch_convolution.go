package optimalconv

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// BatchConvConfig holds the configuration for batch convolution (Algorithm 1).
// It describes how multiple input-kernel pairs are packed into single polynomials
// so that one ciphertext multiplication computes all convolutions simultaneously.
type BatchConvConfig struct {
	InputWidth   int // w: width of input feature map
	InputHeight  int // h: height of input feature map
	KernelWidth  int // kw: width of convolution kernel
	KernelHeight int // kh: height of convolution kernel
	InputBatch   int // B: number of input channels packed per ciphertext
	OutputBatch  int // B': number of output batches
	Stride       int // convolution stride (1 or 2)
}

// OutputWidth returns the output feature map width after convolution.
func (c BatchConvConfig) OutputWidth() int {
	return (c.InputWidth - c.KernelWidth) / c.Stride + 1
}

// OutputHeight returns the output feature map height after convolution.
func (c BatchConvConfig) OutputHeight() int {
	return (c.InputHeight - c.KernelHeight) / c.Stride + 1
}

// SparsePacking returns true if the coefficient layout results in sparsely-packed
// polynomials (i.e., nonzero only at positions divisible by stride s).
func (c BatchConvConfig) SparsePacking() bool {
	return c.Stride > 1
}

// BatchConvEvaluator evaluates batch convolutions following Algorithm 1.
//
// The core idea: multiple image patches and kernels are packed into single
// polynomials I(X) and K(X) using coefficient encoding. A single polynomial
// multiplication I(X)·K(X) computes all convolutions simultaneously.
// The results are then extracted from the product's coefficients.
type BatchConvEvaluator struct {
	eval    *ckks.Evaluator
	encoder *ckks.Encoder
	params  ckks.Parameters
}

// NewBatchConvEvaluator creates a new BatchConvEvaluator.
func NewBatchConvEvaluator(eval *ckks.Evaluator, encoder *ckks.Encoder, params ckks.Parameters) *BatchConvEvaluator {
	return &BatchConvEvaluator{
		eval:    eval,
		encoder: encoder,
		params:  params,
	}
}

// EncodeBatchInput packs multiple input feature maps into a single
// coefficient-encoded plaintext following the paper's packing strategy.
//
// inputs is a slice of B flattened feature maps, each of size inputHeight*inputWidth.
// The packing places each input at a different offset in the polynomial to avoid
// overlap after multiplication with the kernel polynomial.
func (bce *BatchConvEvaluator) EncodeBatchInput(inputs [][]float64, cfg BatchConvConfig, level int, scale rlwe.Scale) (*rlwe.Plaintext, error) {
	N := bce.params.N()
	B := cfg.InputBatch

	if len(inputs) > B {
		return nil, fmt.Errorf("EncodeBatchInput: got %d inputs but batch size is %d", len(inputs), B)
	}

	iw := cfg.InputWidth
	ih := cfg.InputHeight
	imageSize := iw * ih

	// Each input channel is placed at offset b*imageSize in the polynomial
	coeffs := make([]float64, N)
	for b := 0; b < len(inputs); b++ {
		if len(inputs[b]) != imageSize {
			return nil, fmt.Errorf("EncodeBatchInput: input[%d] has size %d, expected %d", b, len(inputs[b]), imageSize)
		}
		offset := b * imageSize
		for i := 0; i < imageSize && offset+i < N; i++ {
			coeffs[offset+i] = inputs[b][i]
		}
	}

	return CfEcd(bce.encoder, coeffs, bce.params, level, scale)
}

// EncodeBatchKernel packs convolution kernels for batch convolution.
//
// kernels[b] is the flattened kernel for the b-th input channel,
// each of size kernelHeight*kernelWidth. The kernel polynomial K(X) is
// constructed so that I(X)·K(X) yields all B convolution results
// in non-overlapping coefficient ranges.
func (bce *BatchConvEvaluator) EncodeBatchKernel(kernels [][]float64, cfg BatchConvConfig, level int, scale rlwe.Scale) (*rlwe.Plaintext, error) {
	N := bce.params.N()
	B := cfg.InputBatch

	if len(kernels) > B {
		return nil, fmt.Errorf("EncodeBatchKernel: got %d kernels but batch size is %d", len(kernels), B)
	}

	kw := cfg.KernelWidth
	kh := cfg.KernelHeight
	kernelSize := kw * kh

	coeffs := make([]float64, N)

	// Encode first kernel at low-order coefficients
	for i := 0; i < len(kernels[0]) && i < kernelSize; i++ {
		coeffs[i] = kernels[0][i]
	}

	// Remaining kernels are placed so their products with the corresponding
	// input channels land in the correct coefficient range.
	// For batch b > 0, kernel is placed with an offset that accounts for
	// the image size spacing used in input packing.
	iw := cfg.InputWidth
	ih := cfg.InputHeight
	imageSize := iw * ih

	for b := 1; b < len(kernels); b++ {
		offset := b * imageSize
		for i := 0; i < len(kernels[b]) && i < kernelSize; i++ {
			idx := offset + i
			if idx < N {
				coeffs[idx] = kernels[b][i]
			}
		}
	}

	return CfEcd(bce.encoder, coeffs, bce.params, level, scale)
}

// EvalBatchConv evaluates a batch convolution by performing a single
// ciphertext-plaintext multiplication.
//
// ctInput is the encrypted batch input (from EncodeBatchInput).
// ptKernel is the encoded batch kernel (from EncodeBatchKernel).
// The output is a ciphertext whose coefficients encode all B convolution results.
func (bce *BatchConvEvaluator) EvalBatchConv(ctInput *rlwe.Ciphertext, ptKernel *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
	return ConvSingle(bce.eval, ctInput, ptKernel)
}

// ExtractConvResult extracts the b-th convolution result from a batch
// convolution output. The result is decoded from the appropriate coefficient
// range of the output ciphertext.
//
// This is a plaintext-domain operation used for verification and testing.
func (bce *BatchConvEvaluator) ExtractConvResult(decoded []float64, b int, cfg BatchConvConfig) []float64 {
	ow := cfg.OutputWidth()
	oh := cfg.OutputHeight()
	resultSize := ow * oh

	iw := cfg.InputWidth
	ih := cfg.InputHeight
	imageSize := iw * ih

	offset := b * imageSize
	result := make([]float64, resultSize)
	for i := 0; i < resultSize && offset+i < len(decoded); i++ {
		result[i] = decoded[offset+i]
	}

	_ = oh // used in computation of resultSize
	return result
}
