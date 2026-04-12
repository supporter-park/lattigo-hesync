package optimalconv

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func testCKKSParams(t *testing.T) (ckks.Parameters, *rlwe.SecretKey) {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{55, 45, 45, 45, 45},
		LogP:            []int{60},
		LogDefaultScale: 45,
	})
	require.NoError(t, err)

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	return params, sk
}

func TestCfEcdCfDcd(t *testing.T) {
	params, sk := testCKKSParams(t)

	encoder := ckks.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)
	dec := rlwe.NewDecryptor(params, sk)

	// Test coefficient encoding round-trip on plaintext
	values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	pt, err := CfEcd(encoder, values, params, params.MaxLevel(), params.DefaultScale())
	require.NoError(t, err)
	require.False(t, pt.IsBatched)

	decoded, err := CfDcd(encoder, pt, len(values))
	require.NoError(t, err)
	require.Len(t, decoded, len(values))

	for i, v := range values {
		require.InDelta(t, v, decoded[i], 0.01, "mismatch at index %d", i)
	}

	// Test encrypt -> decrypt round-trip
	ct, err := enc.EncryptNew(pt)
	require.NoError(t, err)

	ptDec := dec.DecryptNew(ct)
	ptDec.IsBatched = false
	decodedDec, err := CfDcd(encoder, ptDec, len(values))
	require.NoError(t, err)

	for i, v := range values {
		require.InDelta(t, v, decodedDec[i], 0.1, "enc/dec mismatch at index %d", i)
	}
}

func TestConvSingle(t *testing.T) {
	params, sk := testCKKSParams(t)

	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)
	dec := rlwe.NewDecryptor(params, sk)

	// No Galois keys needed for ct-pt multiplication
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	// Simple 1D convolution test:
	// Image: [1, 2, 3, 4] (polynomial I(X) = 1 + 2X + 3X^2 + 4X^3)
	// Kernel: [1, 1] (polynomial K(X) = 1 + X)
	// Expected product in Z[X]/(X^N+1):
	//   I(X)·K(X) = 1 + 3X + 5X^2 + 7X^3 + 4X^4
	image := []float64{1, 2, 3, 4}
	kernel := []float64{1, 1}

	ptImage, err := CfEcd(encoder, image, params, params.MaxLevel(), params.DefaultScale())
	require.NoError(t, err)

	ctImage, err := enc.EncryptNew(ptImage)
	require.NoError(t, err)

	ptKernel, err := CfEcd(encoder, kernel, params, ctImage.Level(), params.DefaultScale())
	require.NoError(t, err)

	// ConvSingle does multiplication + rescale
	ctResult, err := ConvSingle(eval, ctImage, ptKernel)
	require.NoError(t, err)

	ptResult := dec.DecryptNew(ctResult)
	ptResult.IsBatched = false
	decoded, err := CfDcd(encoder, ptResult, 5)
	require.NoError(t, err)

	// Check the first few coefficients of the product
	expected := []float64{1, 3, 5, 7, 4}
	for i, v := range expected {
		require.InDelta(t, v, decoded[i], 0.5,
			"coefficient %d: expected %.1f, got %.4f", i, v, decoded[i])
	}
}

func TestBatchConvEvaluator(t *testing.T) {
	params, sk := testCKKSParams(t)

	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)
	dec := rlwe.NewDecryptor(params, sk)

	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	bce := NewBatchConvEvaluator(eval, encoder, params)

	cfg := BatchConvConfig{
		InputWidth:   4,
		InputHeight:  4,
		KernelWidth:  3,
		KernelHeight: 3,
		InputBatch:   1,
		Stride:       1,
	}

	// Single channel input
	input := make([]float64, 16)
	for i := range input {
		input[i] = float64(i + 1)
	}
	inputs := [][]float64{input}

	ptInput, err := bce.EncodeBatchInput(inputs, cfg, params.MaxLevel(), params.DefaultScale())
	require.NoError(t, err)

	ctInput, err := enc.EncryptNew(ptInput)
	require.NoError(t, err)

	// Simple kernel
	kernel := make([]float64, 9)
	for i := range kernel {
		kernel[i] = 1.0
	}
	kernels := [][]float64{kernel}

	ptKernel, err := bce.EncodeBatchKernel(kernels, cfg, ctInput.Level(), params.DefaultScale())
	require.NoError(t, err)

	ctResult, err := bce.EvalBatchConv(ctInput, ptKernel)
	require.NoError(t, err)

	ptResult := dec.DecryptNew(ctResult)
	ptResult.IsBatched = false
	decoded, err := CfDcd(encoder, ptResult, 16)
	require.NoError(t, err)

	// First coefficient should be non-zero (convolution output)
	require.NotEqual(t, 0.0, decoded[0], "convolution output should not be zero")
}

func TestCNNEvaluatorSmall(t *testing.T) {
	params, sk := testCKKSParams(t)

	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)

	// Generate keys for a minimal CNN
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	config := SmallTestConfig()

	// Create CNN evaluator without bootstrapper (no activations)
	cnnEval := NewCNNEvaluator(params, eval, nil, config)

	// Generate random kernels
	kernels := GenerateRandomKernels(config)
	require.NoError(t, cnnEval.SetKernels(kernels))

	// Create encrypted input (single channel, 8x8)
	encoder := ckks.NewEncoder(params)
	input := make([]float64, 64)
	for i := range input {
		input[i] = float64(i) / 64.0
	}

	ptInput, err := CfEcd(encoder, input, params, params.MaxLevel(), params.DefaultScale())
	require.NoError(t, err)

	ctInput, err := enc.EncryptNew(ptInput)
	require.NoError(t, err)

	// Run first layer only
	outputs, err := cnnEval.EvalConvLayer([]*rlwe.Ciphertext{ctInput}, 0)
	require.NoError(t, err)
	require.Len(t, outputs, config.Layers[0].OutputChannels)

	// Verify outputs are non-nil ciphertexts
	for i, ct := range outputs {
		require.NotNil(t, ct, "output channel %d is nil", i)
	}
}

func TestPackLWEsSmall(t *testing.T) {
	params, sk := testCKKSParams(t)

	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	enc := rlwe.NewEncryptor(params, sk)

	// Generate Galois keys for PackLWEs
	galEls := GaloisElementsForPackLWEs(params, 4)
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)
	eval := ckks.NewEvaluator(params, evk)

	// Create 4 ciphertexts each with a value at coefficient 0
	values := []float64{1.0, 2.0, 3.0, 4.0}
	cts := make([]*rlwe.Ciphertext, len(values))

	for i, v := range values {
		pt, err := CfEcd(encoder, []float64{v}, params, params.MaxLevel(), params.DefaultScale())
		require.NoError(t, err)
		cts[i], err = enc.EncryptNew(pt)
		require.NoError(t, err)
	}

	// Pack
	packed, err := PackLWEs(eval, cts, 0, 0)
	require.NoError(t, err)
	require.NotNil(t, packed)
}

func TestComputeSpatialSize(t *testing.T) {
	config := DefaultPlain20Config()

	// After all layers, spatial size should be reduced
	finalSize := computeSpatialSize(config, len(config.Layers))
	require.Greater(t, finalSize, 0, "spatial size should be positive")

	// Initial spatial size
	initialSize := computeSpatialSize(config, 0)
	require.Equal(t, 32*32, initialSize)
}
