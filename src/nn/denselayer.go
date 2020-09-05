package nn

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

// Dense layer represents a single dense layer
type DenseLayer struct {
	learningRate float64
	weights *mat.Dense
	biases *mat.Dense
	input *mat.Dense
}

// Returns a Layer interface for this dense layer
func NewDenseLayer(inputUnits int, outputUnits int, learningRate float64) Layer {
	var initweights = make([]float64, inputUnits*outputUnits) // initialize the weights
	for i := range initweights {
		initweights[i] = rand.NormFloat64() // random normal variable
	}
	// Create the weight matrix
	weights := mat.NewDense(inputUnits, outputUnits, initweights)
	var initbiases = make([]float64, outputUnits) // init the biases
	for i := range initbiases {
		initbiases[i] = 1 // init all to 1
	}
	// Create bias matrix
	biases := mat.NewDense(1, outputUnits, initbiases)
	// Return a dense layer
	return &DenseLayer{learningRate, weights, biases, nil}
}

// Forward propogation for this dense layer
func (d *DenseLayer) Forward(input mat.Dense) mat.Dense {
	d.input = &input // set the input
	var output mat.Dense
	output.Mul(d.input, d.weights)  // multiply input (x) by weights (w)
	output.Add(&output, d.biases) // Add biases to the output
	// return the output
	return output
}

// Backward propogation for the dense layer
func (d *DenseLayer) Backward(gradOutput mat.Dense, y float64) mat.Dense {
	var gradInput mat.Dense
	// This is what we really care about
	gradInput.Mul(&gradOutput, d.weights.T()) // mutliply output by weights transpose

	// Now update the weights
	var gradWeights mat.Dense
	gradWeights.Mul(d.input.T(), &gradOutput) // multiply input transpose by output
	r, c := gradOutput.Dims()
	// gradient for the biases, sum of each column in grad output
	var gradBiasesSlice = make([]float64, c)
	for i := 0; i < r; i++ {
		rawRow := gradOutput.RawRowView(i)
		for j:= 0; j < c; j++ {
			gradBiasesSlice[j] += rawRow[j]
		}
	}
	gradBiases := mat.NewDense(1, c, gradBiasesSlice) // matrix for grad bias

	// now want to multiply the negative learning rate by the gradients
	learning := func(_, _ int, v float64) float64 {return -d.learningRate*v}
	// Apply function
	gradBiases.Apply(learning, gradBiases)
	gradWeights.Apply(learning, &gradWeights)
	// Finally update the weights and biases based on gradients
	d.weights.Add(d.weights, &gradWeights)
	d.biases.Add(d.biases, gradBiases)
	// return the gradInput
	return gradInput
}

// Get the type of this layer
func (d *DenseLayer) GetType() string {
	return "Dense"
}

// Set the weights for this layer
func (d *DenseLayer) SetWeights(weights mat.Dense, biases mat.Dense) {
	d.weights = &weights
	d.biases = &biases
}

// Get the weights for this layer
func (d *DenseLayer) GetWeights() (mat.Dense, mat.Dense) {
	return *d.weights, *d.biases
}