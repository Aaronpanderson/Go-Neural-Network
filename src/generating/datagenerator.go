package main

import (
	"encoding/csv"
	"math/rand"
	"os"
	"strconv"
)

func generateFitData(numrows int, numcols int, outpath string) {
	// Write to output file
	outfile, err := os.Create(outpath)
	if err != nil {panic(err)}
	writer := csv.NewWriter(outfile)
	// Convert the values in the predictions to strings and write
	writes := [][]string{}
	for i := 0; i < numrows; i++ {
		s := make([]string, 0)
		for j := 0; j < numcols; j++ {
			var bound int
			if j == numcols-1 {
				bound = 10
			} else {
				bound = 100
			}
			num := rand.Intn(bound)
			sNum := strconv.Itoa(num)
			s = append(s, sNum)
		}
		writes = append(writes, s)
	}
	writer.WriteAll(writes)
	writer.Flush()
}

func generatePredictData(numrows int, numcols int, outpath string) {
	// Write to output file
	outfile, err := os.Create(outpath)
	if err != nil {panic(err)}
	writer := csv.NewWriter(outfile)
	// Convert the values in the predictions to strings and write
	writes := [][]string{}
	for i := 0; i < numrows; i++ {
		s := make([]string, 0)
		for j := 0; j < numcols-1; j++ {
			num := rand.Intn(100)
			sNum := strconv.Itoa(num)
			s = append(s, sNum)
		}
		writes = append(writes, s)
	}
	writer.WriteAll(writes)
	writer.Flush()
}

func main() {
	categories := []string{"small", "medium", "large"}
	numrows := []int{100, 1000, 10000}
	for i := 0; i < len(categories); i++ {
		s := "./proj3/src/classify/fitdata" + categories[i] + ".csv"
		generateFitData(numrows[i], 10, s)
	}
	for i := 0; i < len(categories); i++ {
		s := "./proj3/src/classify/predictdata" + categories[i] + ".csv"
		generatePredictData(numrows[i], 10, s)
	}

}