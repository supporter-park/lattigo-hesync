package optimalconv

import (
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
)

// CNNParametersLiteral defines the CKKS parameters for CNN inference.
// These follow the optimal_conv paper and HESync benchmark settings:
// N=2^16, 128-bit security, LogDefaultScale=45.
var CNNParametersLiteral = ckks.ParametersLiteral{
	LogN:            16,
	LogQ:            []int{55, 45, 45, 45, 45, 45, 45, 45, 45, 45},
	LogP:            []int{61, 61, 61, 61},
	Xs:              ring.Ternary{H: 192},
	LogDefaultScale: 45,
}

// TestCNNParametersLiteral defines small parameters for fast testing.
// NOT secure — only for correctness verification.
var TestCNNParametersLiteral = ckks.ParametersLiteral{
	LogN:            12,
	LogQ:            []int{55, 45, 45, 45, 45, 45, 45},
	LogP:            []int{61},
	LogDefaultScale: 45,
}

// CNNBootstrappingParametersLiteral defines bootstrapping parameters
// compatible with the CNN residual parameters.
var CNNBootstrappingParametersLiteral = bootstrapping.ParametersLiteral{}

// CNNLayerConfig describes a single convolutional layer.
type CNNLayerConfig struct {
	InputChannels  int // number of input channels
	OutputChannels int // number of output channels
	KernelSize     int // kernel width and height (square kernels)
	Stride         int // convolution stride
	HasActivation  bool // whether this layer is followed by a ReLU activation
	HasBootstrap   bool // whether to bootstrap after activation
}

// PlainCNNConfig describes the full CNN architecture.
// Based on the Plain-20 architecture from the optimal_conv paper:
//   - 1 initial convolution (Conv0)
//   - 3 convolutional blocks (Conv1, Conv2, Conv3)
//   - Each block has multiple layers
//   - ReLU activations approximated by minimax composite polynomial
//   - Bootstrapping after each activation for level refresh
type PlainCNNConfig struct {
	InputWidth  int              // input image width (32 for CIFAR-10)
	InputHeight int              // input image height (32 for CIFAR-10)
	NumClasses  int              // number of output classes (10 for CIFAR-10)
	Layers      []CNNLayerConfig // ordered layer configurations
}

// DefaultPlain20Config returns the Plain-20 CNN configuration for CIFAR-10.
// Architecture: Conv0 -> [Conv1 x 3] -> [Conv2 x 3] -> [Conv3 x 3] -> AvgPool -> FC
// Following the optimal_conv paper's Table VI.
func DefaultPlain20Config() PlainCNNConfig {
	return PlainCNNConfig{
		InputWidth:  32,
		InputHeight: 32,
		NumClasses:  10,
		Layers: []CNNLayerConfig{
			// Conv0: initial convolution
			{InputChannels: 3, OutputChannels: 16, KernelSize: 3, Stride: 1, HasActivation: true, HasBootstrap: true},

			// Conv1 block (3 layers, no downsampling)
			{InputChannels: 16, OutputChannels: 16, KernelSize: 3, Stride: 1, HasActivation: true, HasBootstrap: true},
			{InputChannels: 16, OutputChannels: 16, KernelSize: 3, Stride: 1, HasActivation: true, HasBootstrap: true},
			{InputChannels: 16, OutputChannels: 16, KernelSize: 3, Stride: 1, HasActivation: true, HasBootstrap: true},

			// Conv2 block (3 layers, first has stride 2 for downsampling)
			{InputChannels: 16, OutputChannels: 32, KernelSize: 3, Stride: 2, HasActivation: true, HasBootstrap: true},
			{InputChannels: 32, OutputChannels: 32, KernelSize: 3, Stride: 1, HasActivation: true, HasBootstrap: true},
			{InputChannels: 32, OutputChannels: 32, KernelSize: 3, Stride: 1, HasActivation: true, HasBootstrap: true},

			// Conv3 block (3 layers, first has stride 2 for downsampling)
			{InputChannels: 32, OutputChannels: 64, KernelSize: 3, Stride: 2, HasActivation: true, HasBootstrap: true},
			{InputChannels: 64, OutputChannels: 64, KernelSize: 3, Stride: 1, HasActivation: true, HasBootstrap: true},
			{InputChannels: 64, OutputChannels: 64, KernelSize: 3, Stride: 1, HasActivation: false, HasBootstrap: false},
		},
	}
}

// SmallTestConfig returns a minimal CNN configuration for fast testing.
func SmallTestConfig() PlainCNNConfig {
	return PlainCNNConfig{
		InputWidth:  8,
		InputHeight: 8,
		NumClasses:  10,
		Layers: []CNNLayerConfig{
			{InputChannels: 1, OutputChannels: 4, KernelSize: 3, Stride: 1, HasActivation: false, HasBootstrap: false},
			{InputChannels: 4, OutputChannels: 4, KernelSize: 3, Stride: 1, HasActivation: false, HasBootstrap: false},
		},
	}
}

// TotalGaloisElements returns the Galois elements needed for all convolution
// rotations in this CNN configuration.
func (cfg PlainCNNConfig) TotalGaloisElements(params ckks.Parameters) []uint64 {
	rotSet := make(map[int]bool)

	for _, layer := range cfg.Layers {
		// Rotations needed for this layer's convolution
		ow := (cfg.InputWidth-layer.KernelSize)/layer.Stride + 1
		for r := 0; r < ow; r++ {
			rotSet[r] = true
			rotSet[-r] = true
		}
	}

	rotations := make([]int, 0, len(rotSet))
	for r := range rotSet {
		rotations = append(rotations, r)
	}

	return params.GaloisElements(rotations)
}
