package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io"
	"os"
	"runtime"
	"src/nn"
	"strconv"
)

// Task is a struct to represent each task decoded from stdin
type Task struct {
	Command  string `json:"command"`
	DataPath string `json:"datapath"`
	PredictPath string `json:"predictpath"`
}

// printUsage prints the usage for the program.
func printUsage() {
	activation := "activation:\n\tsigmoid - Uses sigmoid activation function in training the neural network.\n" +
		"\trelu - Uses ReLU activation function in training the neural network.\n"
	nodes := "nodes: Number of nodes in the dense hidden layer of the neural network\n"
	epochs := "epochs: Number of epochs to train the neural network\n"
	learning := "rate: Learning rate used in training the neural network\n"
	p := "-p: Optional parallel flag to run the program in parallel\n"
	threads := "threads: the number of threads to spawn when run in parallel. GOMAXPROCS will be set to this value.\n"

	fmt.Printf("Usage: classify activation nodes epochs rate -p threads\n" + activation + nodes + epochs + learning + p + threads)
}

// Sequential runs the neural network in sequential fashion (default)
func sequential(decoder *json.Decoder, myNN *nn.NeuralNet) {
	// Get the parameters from command line argument
	activation := os.Args[1] // activation function
	hidden, _ := strconv.Atoi(os.Args[2]) // Number of nodes in hidden layer
	epochs, _ := strconv.Atoi(os.Args[3]) // Number of epochs to train for
	learning, _ := strconv.ParseFloat(os.Args[4], 64) // Learning rate

	// Loop while there are tasks in stdin
	for {
		// Init a task
		var task Task
		// Decode the task and log any errors
		err := decoder.Decode(&task)
		if err == io.EOF {break} // If EOF then we're done
		if err != nil {panic(err)}

		// Create slices to hold the data for X (input) and y (output)
		Xdata := make([]float64, 0)
		Ydata := make([]float64, 0)
		// Read the csv data from the task path
		csvfile, err := os.Open(task.DataPath)
		if err != nil {panic(err)}
		defer csvfile.Close()
		// Parse the file
		r := csv.NewReader(csvfile)
		// Iterate through the records
		rows := 0
		cols := 0
		for {
			// Read each record from csv
			record, err := r.Read()
			if err == io.EOF {break} // If EOF then we are done
			if err != nil {panic(err)}
			// Number of columns in the data
			cols = len(record)
			// convert data to floats and add to data slices
			for i:= 0; i < cols; i++ {
				x , _ := strconv.ParseFloat(record[i], 64)
				if i==cols-1 && task.Command == "fit" {
					// If fitting, assume last column is y data
					Ydata = append(Ydata, x)
				}else {
					// Otherwise all X data
					Xdata = append(Xdata, x)
				}
			}
			rows++ // Count rows
		}
		csvfile.Close()
		// Convert data slices to matrices
		if task.Command == "fit" {
			// Init matrices
			X := mat.NewDense(rows, cols-1, Xdata)
			y := mat.NewDense(1,rows, Ydata)
			// find number of unique classifications in y data
			numUnique := 0
			keys := make(map[int]bool)
			for _, entry := range Ydata {
				if _, value := keys[int(entry)]; !value {
					keys[int(entry)] = true
					numUnique++
				}
			}
			// Create the nodeSlice, represent the number of layers and number of nodes
			// First is size of X input, second is hidden nodes, third is y classifications
			nodeSlice := []int{cols-1, hidden, numUnique}
			// Create the neural network
			myNN = nn.NewNeuralNet(nodeSlice, activation, learning)
			buff := make(chan bool, 1) // This is not important for sequential so just buffered channel
			// Fit the neural network based on the given data
			myNN.Fit(*X, *y, epochs, buff)
		}else if task.Command == "predict" {
			// Convert X data to matrix
			X := mat.NewDense(rows, cols, Xdata)
			// Predict based on the neural network
			predictions := myNN.Predict(*X)

			// Write to output file
			outfile, err := os.Create(task.PredictPath)
			if err != nil {panic(err)}
			writer := csv.NewWriter(outfile)
			// Convert the values in the predictions to strings and write
			writes := [][]string{}
			for _, value := range predictions {
				s := []string{strconv.Itoa(int(value))}
				writes = append(writes, s)
			}
			writer.WriteAll(writes)
			writer.Flush()
			outfile.Close()
		}
	}
}

// Reader reads in tasks from json and either fits the data or assigns workers to predict the data
func reader(decoder *json.Decoder, myNN *nn.NeuralNet, done chan bool) {
	// Get the parameters from the command line
	activation := os.Args[1] // activation function
	hidden, _ := strconv.Atoi(os.Args[2]) // Number of nodes in hidden layer
	epochs, _ := strconv.Atoi(os.Args[3]) // Number of epochs to train for
	learning, _ := strconv.ParseFloat(os.Args[4], 64) // Learning rate
	numthreads, _ := strconv.Atoi(os.Args[6]) // Number of threads to spawn
	workerDone := make(chan bool) // Channel to know when workers are completed
	workers := 0 // Number of workers
	runtime.GOMAXPROCS(numthreads) // Set GOMAXPROCS
	for {
		// Init the task
		var task Task
		// Decode the task and log any errors
		err := decoder.Decode(&task)
		if err == io.EOF {break} // If EOF then we are done
		if err != nil {
			panic(err)
		}

		// Initialize slices to hold the X (input) data and y (classification) data
		Xdata := make([]float64, 0)
		Ydata := make([]float64, 0)
		// Open the csv file specified by the task
		csvfile, err := os.Open(task.DataPath)
		if err != nil {panic(err)}
		// Parse the file
		r := csv.NewReader(csvfile)
		// Iterate through the records
		rows := 0
		cols := 0
		for {
			// Read each record from csv
			record, err := r.Read()
			if err == io.EOF {break}
			if err != nil {panic(err)}
			cols = len(record)
			// convert data to floats and add to data slices
			for i := 0; i < cols; i++ {
				x, _ := strconv.ParseFloat(record[i], 64)
				if i == cols-1 && task.Command == "fit" {
					// If fitting then assume last column is y data
					Ydata = append(Ydata, x)
				} else {
					// Otherwise all data is X data
					Xdata = append(Xdata, x)
				}
			}
			rows++
		}
		csvfile.Close()
		// If fitting, we have to do that first (before predicting)
		// So take care of this now and wait for it to be done before spawning predicting threads
		if task.Command == "fit" {
			// Find unique y values, this is the number of nodes in the output layer
			numUnique := 0
			keys := make(map[int]bool)
			for _, entry := range Ydata {
				if _, value := keys[int(entry)]; !value {
					keys[int(entry)] = true
					numUnique++
				}
			}
			// Set up the node slice which represents number of layers and nodes in each layer
			nodeSlice := []int{cols - 1, hidden, numUnique}
			// Fit the data in parallel. Idea is to split up the data into even chunks based on
			// num threads. Then fit a different nn to each chunk of data. Finally average all
			// the weights to create the new final nn for all data
			future := make(chan []mat.Dense) // channel to send final averaged weights
			// Future implementation
			go func(future chan []mat.Dense) {
				avgWeights, avgBiases := parallelFit(Xdata, Ydata, cols, numthreads, nodeSlice, activation, learning, epochs)
				future<-avgWeights
				future<-avgBiases
			}(future)
			// get the futures
			var avgWeights []mat.Dense
			var avgBiases []mat.Dense
			avgWeights = <-future
			avgBiases = <-future
			// Set up the final nn
			myNN = nn.NewNeuralNet(nodeSlice, activation, learning)
			// Set the weights to be the average of the weights computed above
			myNN.SetWeights(avgWeights, avgBiases)
		} else {
			// Otherwise we are predicting and we don't have to wait so spawn a goroutine to
			// predict for this data
			go parallelPredict(task, Xdata, cols, rows, myNN, workerDone)
			workers++ // Increment workers
		}
	}
	// Wait for all workers to be done
	for i:=0; i < workers; i++ {
		<-workerDone
	}
	// Notify main we are done
	done<-true
}

// Fits one master Neural Network to the X and y data by creating [numthreads] small neural networks
// trained on portion of the data and then averaging the results. Returns average weights to be used
// on the master nn.
func parallelFit(Xdata []float64, Ydata []float64, cols int, numthreads int, nodeSlice []int, activation string, learning float64, epochs int ) ([]mat.Dense, []mat.Dense) {
	// Split up the data into roughly equal chunks to train different nn's
	fittingDone := make(chan bool) // Channel to know when all workers are done
	chunksize := len(Xdata) / numthreads // Chunk of data for each thread
	chunksize -= chunksize % (cols-1) // Make sure we have even examples for each cunk
	xidx := 0 // Index for where we are in xdata
	yidx := 0 // Index for where we are in ydata
	NNs := make([]*nn.NeuralNet, 0) // Slice to hold all the intermediate neural networks

	// Loop over number of threads to fit small nn's
	for t := 0; t < numthreads; t++ {
		xi := make([]float64, 0) // X data for this chunk
		yi := make([]float64, 0) // y data for this chunk
		// x gets a chunksize of data
		xi = Xdata[xidx : xidx+chunksize]
		// y gets an amount of data equal to number of examples in xi
		// this is equal to len(xi)/(cols-1)
		yi = Ydata[yidx : yidx+len(xi)/(cols-1)]

		// Create a neural network for this data
		thisNN := nn.NewNeuralNet(nodeSlice, activation, learning)
		// Create the matrices from the slice data
		X := mat.NewDense(len(xi)/(cols-1), cols-1, xi)
		y := mat.NewDense(1, len(yi), yi)
		// Spawn a goroutine to fit this nn to the data
		go thisNN.Fit(*X, *y, epochs, fittingDone)
		// add this nn to the slice
		NNs = append(NNs, thisNN)
		// update the two indices
		xidx += chunksize
		yidx += len(xi) / (cols - 1)
	}
	// Wait for all fits to be done
	for i := 0; i < numthreads; i++ {
		<-fittingDone
	}
	// Now need to find the average of all the weights and biases
	avgWeights := make([]mat.Dense, 0)
	avgBiases := make([]mat.Dense, 0)
	// Loop over all the intermediate NN's
	for i := range NNs {
		thisNN := NNs[i]
		// Get the weights and the biases
		weights, biases := thisNN.GetWeights()
		if i == 0 {
			for j := range weights {
				// Add them to the slices for the first one
				avgWeights = append(avgWeights, weights[j])
				avgBiases = append(avgBiases, biases[j])
			}
		} else {
			for j := range avgWeights {
				// Keep a running total of the weights and biases.
				// Will divide through to average after this
				avgWeights[j].Add(&avgWeights[j], &weights[j])
				avgBiases[j].Add(&avgBiases[j], &biases[j])
			}
		}
	}
	// Now want to divide through by the total NN's to find the average
	divide := func(_, _ int, v float64) float64 { return v / float64(len(NNs)) }
	for i := range avgWeights {
		// Apply the function to divide through
		avgWeights[i].Apply(divide, &avgWeights[i])
		avgBiases[i].Apply(divide, &avgBiases[i])
	}
	// return the average weights and biases
	return avgWeights, avgBiases

}

// Predict y (classification) based on X (input data) in parallel
func parallelPredict(task Task, Xdata []float64, cols int, rows int, myNN *nn.NeuralNet, workerDone chan bool) {
	// Create a matrix for the X data
	X := mat.NewDense(rows, cols, Xdata)
	// Predict using the neural network that has been trained
	predictions := myNN.Predict(*X)
	// Write the predictions to the out path specified by the task
	outfile, err := os.Create(task.PredictPath)
	if err != nil {panic(err)}
	writer := csv.NewWriter(outfile)
	// Convert values to strings
	writes := [][]string{}
	for _, value := range predictions {
		s := []string{strconv.Itoa(int(value))}
		writes = append(writes, s)
	}
	// Write
	writer.WriteAll(writes)
	writer.Flush()
	outfile.Close()
	// Notify the reader that this worker is done
	workerDone<-true
}


func main() {
	// Set up the decoder and encoder based on stdin and stdout
	decoder := json.NewDecoder(os.Stdin)
	// Initialize neural network
	var myNN *nn.NeuralNet

	if len(os.Args) != 5 && len(os.Args) != 7 {
		// Incorrect usage
		printUsage()
		return
	}else if len(os.Args) == 5 {
		// Sequential
		sequential(decoder,myNN)
	}else if len(os.Args) == 7 {
		//// Parallel
		done := make(chan bool)
		go reader(decoder, myNN, done)
		//// Wait for reader to be done
		<-done
	}
}