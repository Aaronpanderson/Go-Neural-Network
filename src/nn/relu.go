package nn

import (
	"gonum.org/v1/gonum/mat"
	"math"
	)

// Relu represents a ReLU activation function layer
type Relu struct {
	input mat.Dense
}

// Returns a Relu layer
func NewRelu() Layer {
	// Fake init data
	data := make([]float64, 1)
	data[0]=0
	input := mat.NewDense(1,1,data)
	return &Relu{*input}
}

// Forward propogation for Relu
// Defined as v if v > 0, or 0 if v < 0
func (r *Relu) Forward(input mat.Dense) mat.Dense {
	r.input = input // set the input
	// Relu function
	relu := func(_, _ int, v float64) float64 {return math.Max(0, v)}
	var output mat.Dense
	output.Apply(relu, &input) // apply the function
	return input
}

// Backward propogation for Relu
// Defined as 1 for v > 0 or 0 for v < 0
func (r *Relu) Backward(gradOutput mat.Dense, y float64)  mat.Dense {
	input := r.input // set the input
	// Relu back propogation function
	reluBack := func(_, _ int, v float64) float64 {
		if v > 0 {
			return 1
		} else {
			return 0
		}
	}
	var output mat.Dense
	// apply the function to the input
	output.Apply(reluBack, &input)
	// Elementwise multiplication of the gradoutput
	gradOutput.MulElem(&gradOutput, &output)
	// Return the gradoutput
	return gradOutput
}

// Get the type of this layer
func (r *Relu) GetType() string {
	return "Relu"
}

// Get weights does nothing, required for layer interface
func (r *Relu) GetWeights() (mat.Dense, mat.Dense) {
	zero := mat.NewDense(1, 1, nil)
	return *zero, *zero
}

// Set weights does nothing, required for layer interface
func (r *Relu) SetWeights(weights mat.Dense, biases mat.Dense) {}