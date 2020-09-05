package nn

import (
	"gonum.org/v1/gonum/mat"
	"math")

// Sigmoid represents a sigmoid activation layer in a neural network
type Sigmoid struct {
	input mat.Dense
}

// Returns a new sigmoid activation layer
func NewSigmoid() Layer {
	// Fake data for initializing
	data := make([]float64, 1)
	data[0]=0
	input := mat.NewDense(1,1,data)
	return &Sigmoid{*input}
}

// Forward propogation for sigmoid function
// Defined as 1/(1+exp(x))
func (s *Sigmoid) Forward(input mat.Dense) mat.Dense {
	s.input = input // Set the input
	var output mat.Dense
	// Sigmoid function
	sigmoid := func(_, _ int, v float64) float64 {return 1/(1+math.Exp(v))}
	// Apply the function to each element
	output.Apply(sigmoid, &input)
	return output
}

// Backward propogation for sigmoid function
func (s *Sigmoid) Backward(gradOutput mat.Dense, y float64) mat.Dense {
	forward := s.Forward(s.input) // calculate forward propagation
	var output mat.Dense
	output.MulElem(&gradOutput, &forward) // multiply the grad output by the forward output
	return output
}

// Get the type of this layer
func (s *Sigmoid) GetType() string {
	return "Sigmoid"
}

// Get weights does nothing, required for Layer interface
func (s *Sigmoid) GetWeights() (mat.Dense, mat.Dense) {
	zero := mat.NewDense(1, 1, nil)
	return *zero, *zero
}

// Setweights does nothing, required for layer interface
func (s *Sigmoid) SetWeights(weights mat.Dense, biases mat.Dense) {}