package nn

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Softmax is a struct representing a softmax activation layer
// At the end of the neural network
type SoftMax struct {
	input mat.Dense
	y float64
}

// Returns a new softmax layer
func NewSoftmax() Layer {
	// fake data for the input
	data := make([]float64, 1)
	data[0]=0
	input := mat.NewDense(1,1,data)
	// Return the softmax layer
	return &SoftMax{*input, -1}
}

// Forward propogation for softmax
// defined as exp(x)/sum(exp(x))
func (s *SoftMax) Forward(input mat.Dense) mat.Dense {
	s.input = input // set input
	var output mat.Dense
	// exponent function
	exp := func(_, _ int, v float64) float64 {return math.Exp(v)}
	output.Apply(exp, &input) // apply the exponent function
	sum := mat.Sum(&output) // get the sum of the exponentiated matrix
	// divide function
	divide := func(_, _ int, v float64) float64 {return v/sum}
	output.Apply(divide, &output) // apply the divide function
	// Return the output
	return output
}

// Backward propogation for softmax
func (s *SoftMax) Backward(gradOutput mat.Dense, y float64)  mat.Dense {
	s.y = y // set the y value
	row := gradOutput.RawRowView(0)
	// subtract the index of the classification by 1
	for i:= range row {
		if i == int(s.y) {
			row[i] -= 1
		}
	}
	gradOutput.SetRow(0, row) // re-set the row
	// Now divide through by the length of the row
	divide := func(_, _ int, v float64) float64 {return v/float64(len(s.input.RawRowView(0)))}
	gradOutput.Apply(divide, &gradOutput) // apply the function
	// Return the gradient
	return gradOutput
}

// Return the type of the softmax
func (s *SoftMax) GetType() string {
	return "Softmax"
}

// Get the weights, does nothing for this layer but needed for Layer interface
func (s *SoftMax) GetWeights() (mat.Dense, mat.Dense) {
	zero := mat.NewDense(1, 1, nil)
	return *zero, *zero
}

// Set weights does nothing, needed for Layer interface
func (s *SoftMax) SetWeights(weights mat.Dense, biases mat.Dense) {}
