package optimalconv

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/comparison"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/minimax"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// CNNEvaluator evaluates a CNN on CKKS-encrypted data using the optimal_conv
// convolution method. It orchestrates convolution layers, activation functions
// (approximated via minimax composite polynomials), and bootstrapping for
// level refresh.
type CNNEvaluator struct {
	params  ckks.Parameters
	eval    *ckks.Evaluator
	encoder *ckks.Encoder
	bce     *BatchConvEvaluator

	// Bootstrapper for level refresh (nil if not using bootstrapping)
	btpEval bootstrapping.Bootstrapper

	// Comparison evaluator for ReLU approximation via Step function
	cmpEval *comparison.Evaluator

	// CNN architecture config
	config PlainCNNConfig

	// Pre-encoded kernels per layer: [layerIdx][outputCh][inputCh] -> plaintext
	kernels [][][]*rlwe.Plaintext

	// Pre-encoded biases per layer (nil if no bias)
	biases []*rlwe.Plaintext

	// Stats tracking
	Stats InferenceStats
}

// InferenceStats tracks timing and memory statistics during inference.
type InferenceStats struct {
	LayerTimes     []time.Duration // time per layer
	BootstrapTimes []time.Duration // time per bootstrap
	TotalTime      time.Duration
	PeakEVKCount   int // peak number of EVKs in memory
}

// NewCNNEvaluator creates a new CNN evaluator.
//
// Parameters:
//   - params: CKKS parameters
//   - eval: CKKS evaluator with required Galois and relinearization keys
//   - btpEval: bootstrapper (nil to disable bootstrapping)
//   - config: CNN architecture configuration
func NewCNNEvaluator(
	params ckks.Parameters,
	eval *ckks.Evaluator,
	btpEval bootstrapping.Bootstrapper,
	config PlainCNNConfig,
) *CNNEvaluator {
	encoder := ckks.NewEncoder(params)

	cnnEval := &CNNEvaluator{
		params:  params,
		eval:    eval,
		encoder: encoder,
		bce:     NewBatchConvEvaluator(eval, encoder, params),
		btpEval: btpEval,
		config:  config,
		Stats: InferenceStats{
			LayerTimes:     make([]time.Duration, len(config.Layers)),
			BootstrapTimes: make([]time.Duration, 0),
		},
	}

	// Set up minimax evaluator for ReLU if bootstrapper is available
	if btpEval != nil {
		minimaxEval := minimax.NewEvaluator(params, eval, btpEval)
		cnnEval.cmpEval = comparison.NewEvaluator(params, minimaxEval)
	}

	return cnnEval
}

// SetKernels sets the pre-trained convolution kernels for all layers.
// kernels[layer][outputCh][inputCh] is a flattened kernel of size kernelH*kernelW.
func (cnn *CNNEvaluator) SetKernels(kernels [][][][]float64) error {
	cnn.kernels = make([][][]*rlwe.Plaintext, len(cnn.config.Layers))

	for l, layerCfg := range cnn.config.Layers {
		if l >= len(kernels) {
			break
		}

		cnn.kernels[l] = make([][]*rlwe.Plaintext, layerCfg.OutputChannels)
		for oc := 0; oc < layerCfg.OutputChannels && oc < len(kernels[l]); oc++ {
			cnn.kernels[l][oc] = make([]*rlwe.Plaintext, layerCfg.InputChannels)
			for ic := 0; ic < layerCfg.InputChannels && ic < len(kernels[l][oc]); ic++ {
				pt, err := CfEcd(cnn.encoder, kernels[l][oc][ic], cnn.params,
					cnn.params.MaxLevel(), cnn.params.DefaultScale())
				if err != nil {
					return fmt.Errorf("SetKernels: layer %d, oc %d, ic %d: %w", l, oc, ic, err)
				}
				cnn.kernels[l][oc][ic] = pt
			}
		}
	}

	return nil
}

// EvalConvLayer evaluates a single convolutional layer.
//
// For each output channel, it computes the sum of convolutions over all
// input channels: out[oc] = sum_{ic} Conv(in[ic], kernel[oc][ic])
//
// This uses coefficient encoding and polynomial multiplication for each
// individual convolution, then sums the results.
func (cnn *CNNEvaluator) EvalConvLayer(inputs []*rlwe.Ciphertext, layerIdx int) ([]*rlwe.Ciphertext, error) {
	layerCfg := cnn.config.Layers[layerIdx]

	if len(inputs) != layerCfg.InputChannels {
		return nil, fmt.Errorf("EvalConvLayer: expected %d input channels, got %d",
			layerCfg.InputChannels, len(inputs))
	}

	outputs := make([]*rlwe.Ciphertext, layerCfg.OutputChannels)

	for oc := 0; oc < layerCfg.OutputChannels; oc++ {
		for ic := 0; ic < layerCfg.InputChannels; ic++ {
			// Get pre-encoded kernel
			var kernel *rlwe.Plaintext
			if cnn.kernels != nil && layerIdx < len(cnn.kernels) &&
				oc < len(cnn.kernels[layerIdx]) && ic < len(cnn.kernels[layerIdx][oc]) {
				kernel = cnn.kernels[layerIdx][oc][ic]
			}

			if kernel == nil {
				// Use a dummy zero kernel for testing
				dummyKernel := make([]float64, layerCfg.KernelSize*layerCfg.KernelSize)
				var err error
				kernel, err = CfEcd(cnn.encoder, dummyKernel, cnn.params,
					inputs[ic].Level(), cnn.params.DefaultScale())
				if err != nil {
					return nil, fmt.Errorf("EvalConvLayer: encode dummy kernel: %w", err)
				}
			}

			conv, err := ConvSingle(cnn.eval, inputs[ic], kernel)
			if err != nil {
				return nil, fmt.Errorf("EvalConvLayer: layer %d, oc %d, ic %d: %w",
					layerIdx, oc, ic, err)
			}

			if outputs[oc] == nil {
				outputs[oc] = conv
			} else {
				if err := cnn.eval.Add(outputs[oc], conv, outputs[oc]); err != nil {
					return nil, fmt.Errorf("EvalConvLayer: accumulate oc %d: %w", oc, err)
				}
			}
		}
	}

	return outputs, nil
}

// EvalReLU evaluates the ReLU activation function on an encrypted value.
// ReLU(x) = max(0, x) ≈ x * Step(x)
// where Step is approximated via minimax composite polynomial.
func (cnn *CNNEvaluator) EvalReLU(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if cnn.cmpEval == nil {
		return nil, fmt.Errorf("EvalReLU: comparison evaluator not initialized (requires bootstrapper)")
	}

	// Step(x) ≈ 1 if x > 0, 0 if x < 0
	step, err := cnn.cmpEval.Step(ct)
	if err != nil {
		return nil, fmt.Errorf("EvalReLU: step: %w", err)
	}

	// ReLU(x) = x * Step(x)
	result, err := cnn.eval.MulRelinNew(ct, step)
	if err != nil {
		return nil, fmt.Errorf("EvalReLU: mul: %w", err)
	}

	if err := cnn.eval.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("EvalReLU: rescale: %w", err)
	}

	return result, nil
}

// EvalBootstrap bootstraps a ciphertext to refresh its level.
func (cnn *CNNEvaluator) EvalBootstrap(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if cnn.btpEval == nil {
		return nil, fmt.Errorf("EvalBootstrap: bootstrapper not initialized")
	}

	start := time.Now()
	result, err := cnn.btpEval.Bootstrap(ct)
	elapsed := time.Since(start)

	cnn.Stats.BootstrapTimes = append(cnn.Stats.BootstrapTimes, elapsed)

	if err != nil {
		return nil, fmt.Errorf("EvalBootstrap: %w", err)
	}

	return result, nil
}

// EvalAvgPool evaluates global average pooling on encrypted feature maps.
// It computes the mean of all spatial positions for each channel.
func (cnn *CNNEvaluator) EvalAvgPool(cts []*rlwe.Ciphertext, spatialSize int) ([]*rlwe.Ciphertext, error) {
	results := make([]*rlwe.Ciphertext, len(cts))

	scale := 1.0 / float64(spatialSize)

	for i, ct := range cts {
		// Sum all spatial positions via rotations and additions
		acc := ct.CopyNew()
		for shift := 1; shift < spatialSize; shift++ {
			rotated, err := cnn.eval.RotateNew(ct, shift)
			if err != nil {
				return nil, fmt.Errorf("EvalAvgPool: ch %d, shift %d: %w", i, shift, err)
			}
			if err := cnn.eval.Add(acc, rotated, acc); err != nil {
				return nil, fmt.Errorf("EvalAvgPool: add ch %d: %w", i, err)
			}
		}

		// Multiply by 1/spatialSize
		scaled, err := cnn.eval.MulNew(acc, scale)
		if err != nil {
			return nil, fmt.Errorf("EvalAvgPool: scale ch %d: %w", i, err)
		}

		if err := cnn.eval.Rescale(scaled, scaled); err != nil {
			return nil, fmt.Errorf("EvalAvgPool: rescale ch %d: %w", i, err)
		}

		results[i] = scaled
	}

	return results, nil
}

// Evaluate runs the full CNN inference on encrypted input feature maps.
// Returns the encrypted output (class scores) and any error.
func (cnn *CNNEvaluator) Evaluate(inputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	start := time.Now()
	current := inputs

	for l, layerCfg := range cnn.config.Layers {
		layerStart := time.Now()

		// Convolution
		var err error
		current, err = cnn.EvalConvLayer(current, l)
		if err != nil {
			return nil, fmt.Errorf("Evaluate: conv layer %d: %w", l, err)
		}

		// Activation (ReLU approximation)
		if layerCfg.HasActivation && cnn.cmpEval != nil {
			for i := range current {
				current[i], err = cnn.EvalReLU(current[i])
				if err != nil {
					return nil, fmt.Errorf("Evaluate: relu layer %d, ch %d: %w", l, i, err)
				}
			}
		}

		// Bootstrapping for level refresh
		if layerCfg.HasBootstrap && cnn.btpEval != nil {
			for i := range current {
				current[i], err = cnn.EvalBootstrap(current[i])
				if err != nil {
					return nil, fmt.Errorf("Evaluate: bootstrap layer %d, ch %d: %w", l, i, err)
				}
			}
		}

		cnn.Stats.LayerTimes[l] = time.Since(layerStart)
	}

	// Global average pooling
	lastLayer := cnn.config.Layers[len(cnn.config.Layers)-1]
	spatialSize := computeSpatialSize(cnn.config, len(cnn.config.Layers))
	_ = lastLayer

	if spatialSize > 1 {
		var err error
		current, err = cnn.EvalAvgPool(current, spatialSize)
		if err != nil {
			return nil, fmt.Errorf("Evaluate: avgpool: %w", err)
		}
	}

	cnn.Stats.TotalTime = time.Since(start)
	return current, nil
}

// computeSpatialSize calculates the spatial dimension (width*height) of the
// feature map at a given layer, accounting for all strides applied before it.
func computeSpatialSize(config PlainCNNConfig, numLayers int) int {
	w := config.InputWidth
	h := config.InputHeight

	for l := 0; l < numLayers && l < len(config.Layers); l++ {
		layer := config.Layers[l]
		w = (w-layer.KernelSize)/layer.Stride + 1
		h = (h-layer.KernelSize)/layer.Stride + 1
	}

	return w * h
}

// GenerateRandomKernels generates random kernel weights for testing.
// Returns kernels[layer][outputCh][inputCh] as flattened float64 slices.
func GenerateRandomKernels(config PlainCNNConfig) [][][][]float64 {
	kernels := make([][][][]float64, len(config.Layers))

	for l, layer := range config.Layers {
		kernels[l] = make([][][]float64, layer.OutputChannels)
		for oc := 0; oc < layer.OutputChannels; oc++ {
			kernels[l][oc] = make([][]float64, layer.InputChannels)
			for ic := 0; ic < layer.InputChannels; ic++ {
				k := layer.KernelSize * layer.KernelSize
				kernel := make([]float64, k)
				// Xavier initialization: scale by 1/sqrt(fan_in)
				scale := 1.0 / math.Sqrt(float64(layer.InputChannels*k))
				for i := range kernel {
					// Simple deterministic values for reproducibility
					kernel[i] = scale * float64((l*1000+oc*100+ic*10+i)%20-10) / 10.0
				}
				kernels[l][oc][ic] = kernel
			}
		}
	}

	return kernels
}
