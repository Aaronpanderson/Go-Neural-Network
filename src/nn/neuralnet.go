package nn

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// NeuralNet is a struct that represents an artificial neural network
type NeuralNet struct {
	layers []Layer
}

// Returns a new neural network.
// nodesSlice: slice of ints representing number of nodes in each layer
// activation: either "sigmoid" or "relu" for the two different activation functions
// learning rate: value of how much change to the model at each epoch
func NewNeuralNet(nodesSlice []int, activation string, learningRate float64) *NeuralNet {
	// Init a slice of layers
	var layers []Layer
	// Each value in the nodeSlice is a dense layer (last one assumed to be softmax)
	for i:=0; i < len(nodesSlice)-1; i++ {
		// Create a dense layer
		layers = append(layers, NewDenseLayer(nodesSlice[i], nodesSlice[i+1], learningRate))
		// Create an activation function except for the last one which is softmax
		if i != len(nodesSlice)-2 {
			switch activation {
			case "sigmoid":
				// Sigmoid layer
				layers = append(layers, NewSigmoid())
			case "relu":
				// Relu layer
				layers = append(layers, NewRelu())
			}
		}
	}
	// add a softmax layer
	layers = append(layers, NewSoftmax())
	// Return the nueral network
	return &NeuralNet{layers}
}

// Propogates the input X forward through the network
func (nn *NeuralNet) Forward(X mat.Dense) mat.Dense {
	input := X
	// Loop over the layers
	for i:=0; i < len(nn.layers); i++ {
		// Set the input to the next layer as the forward step for this layer
		input = nn.layers[i].Forward(input)
	}
	// return final matrix after all forward steps
	return input
}

// Predict classifications based on the X data
func (nn *NeuralNet) Predict(X mat.Dense) []int {
	// Get the dimensions of the X data
	r, c := X.Dims()
	var row []float64 // final forward row data for this example
	output := make([]int, r) // Output y classifcations
	for i:= 0; i < r; i++ {
		example := make([]float64, c) // This example data
		example = mat.Row(example, i, &X) // Get example data
		Xi := mat.NewDense(1,c,example) // Matrix of this example
		logits := nn.Forward(*Xi) // Forward propogation for this example
		row = mat.Row(row,0,&logits) // Logits for this example
		// The index of the max logit is the predicted classification
		max := 0.0
		for j:=0; j < len(row); j++ {
			if row[j] > max {
				max = row[j]
				output[i] = j
			}
		}
	}
	// Return classifications
	return output
}

// Train the neural network based on X (example) and y (classification).
// Returns the loss value for this epoch
func (nn *NeuralNet) Train(X mat.Dense, y float64) float64 {
	// Forward propogation on the X data to get logits
	logits := nn.Forward(X)
	// Compute loss gradient
	lossGrad := logits
	// Log probabilites is the -log(logits)
	var logProbs mat.Dense
	log := func(_, _ int, v float64) float64 {return -math.Log(v)}
	logProbs.Apply(log, &logits)
	// Calculate the cross entropy loss = -log of index of classification
	loss := 0.0
	row := logProbs.RawRowView(0)
	for i := range row {
		if i==int(y) {
			loss = row[i] // get the loss
		}
	}

	// Back propogation
	// Start at last layer and go to first
	for i := len(nn.layers)-1; i >= 0; i-- {
		if i==len(nn.layers)-1 {
			// Softmax backward propogation needs to know y
			lossGrad = nn.layers[i].Backward(lossGrad, y)
		} else {
			// Otherwise regular backward propogation
			lossGrad = nn.layers[i].Backward(lossGrad, 0)
		}
	}
	return loss // return the loss
}

// Fit fits the neural network based on the X and y data
// trains for a given number of epochs
func (nn *NeuralNet) Fit(X mat.Dense, y mat.Dense, epochs int, done chan bool) {
	// get the dimensions of the X data
	r, c := X.Dims()
	yrows := y.RawRowView(0) // y data (single row)
	// Loop over epochs
	for i:= 0; i < epochs; i++ {
		loss := 0.0 // Loss for this epoch
		// Loop over each example
		for j:=0; j < r; j++ {
			// get the data for this example
			example := make([]float64, c)
			example = mat.Row(example, j, &X)
			// Make the data into a matrix
			Xi := mat.NewDense(1,c,example)
			yi := yrows[j] // classification for this example
			loss += nn.Train(*Xi, yi) // Train this example
		}
		//// Print the loss
		//if i % 10000 == 0 {
		//	fmt.Printf("Epoch: %d  ", i)
		//	fmt.Printf("Loss: %v", loss)
		//	fmt.Println()
		//}
	}
	done<-true // Signal that this is done training
}

// Get the weights for this nn
func (nn *NeuralNet) GetWeights() ([]mat.Dense, []mat.Dense) {
	weights := make([]mat.Dense, 0) // weights
	biases := make([]mat.Dense, 0) // biases
	for i := 0; i < len(nn.layers); i++ {
		if nn.layers[i].GetType() == "Dense" {
			// Weights and biases for this dense layer
			wi, bi := nn.layers[i].GetWeights()
			// Add the weights to the slice
			weights = append(weights, wi)
			biases = append(biases, bi)
		}
	}
	// return the weights
	return weights, biases
}

// Set the weights for this nn
func (nn *NeuralNet) SetWeights(weights []mat.Dense, biases []mat.Dense) {
	widx := 0 // index for this weight
	for i:=0; i < len(nn.layers); i++ {
		// Set the weights for this dense layer
		if nn.layers[i].GetType() == "Dense" {
			nn.layers[i].SetWeights(weights[widx], biases[widx])
			widx++
		}
	}
}

// Get the layers for this nn
func (nn *NeuralNet) GetLayers() []Layer {
	return nn.layers
}