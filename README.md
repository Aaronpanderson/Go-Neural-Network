# Neural Network in Go

Neural Network implementation in Go

## Installation

Extract the files to desired output location.

The datasets directory contains all the json command files used to test the program (all .txt files) as well as the fitting and prediction data sets needed (the .csv files). All source code can be found in the src directory. report.pdf contains a description of the problem and implementation, as well as description, results, and discussion of the tests performed on classify.go.

Inside the src directory, the nn package implements an artificial neural network in go, specifically in neuralnet.go. The classify package contains classify.go which is the main executable needed for testing and reproducing results. The generating package contains the files used to generate the data used for testing.

## Dependencies

The third party linear algebra library gonum/mat is required to run the matrix manipulation and linear algebra sections of the code. The library can be found at: https://godoc.org/gonum.org/v1/gonum/mat and can be imported with:

```
import "gonum.org/v1/gonum/mat"
```

## Usage

Classify.go can be run by first building it as an executable via the command line:

```
go build classify.go
```

Then it can be run by supplying four required command line arguments as well as an optional -p flag and number of threads. The usage can be found below:

```
./classify activation nodes epochs rate -p threads
```

The description of each agrument is as follows:
activation:
    sigmoid - Uses sigmoid activation function in training the neural network.
    relu - Uses ReLU activation function in training the neural network.
nodes: Number of nodes in the dense hidden layer of the neural network
epochs: Number of epochs to train the neural network
rate: Learning rate used in training the neural network
-p: Optional parallel flag to run the program in parallel
threads: the number of threads to spawn when run in parallel. GOMAXPROCS will be set to this value.

For example running the program sequentially with the sigmoid activation function, 10 nodes in the hidden layer, for 1000 epochs and a learning rate of 0.01:

```
./classify sigmoid 10 1000 0.01
```
