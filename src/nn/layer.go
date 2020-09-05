package nn

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

// Layer is an interface representing a layer in a neural network
type Layer interface {
	Forward(input mat.Dense) mat.Dense
	Backward(gradOutput mat.Dense, y float64) mat.Dense
	GetType() string
	GetWeights() (mat.Dense, mat.Dense)
	SetWeights(weights mat.Dense, biases mat.Dense)
}

// Matrix printing function for debugging
// From: https://medium.com/wireless-registry-engineering/gonum-tutorial-linear-algebra-in-go-21ef136fc2d7
func MatPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}