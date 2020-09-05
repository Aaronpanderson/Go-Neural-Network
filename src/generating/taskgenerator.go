package main

import (
	"fmt"
	"os"
	"strconv"
)

const usage = "Usage: taskgenerator <num_of_predicts> <predict_path> <fit_path>\n" +
	"\t <num_of_predicts> = the number of predictions you want to generate\n" +
	"\t <predict_path> = file path of predict data\n" +
	"\t <fit_path> = file path of fit data\n" +
	"Sample Run:\n" +
	"\t./taskgenerator 1000 predictdatasmall.csv fitdatasmall.csv > small1000.txt\n" +
	"\tGenerates 1 fit on the fit data and 1000 predicts saves it to a file\n"

func genPredicts(num int, datapath string) []string {
	predictpath := datapath[0:len(datapath)-4] + "predictions.csv"
	var tasks []string
	for i := 0; i < num; i++ {
		tasks = append(tasks, fmt.Sprintf("{\"command\":\"%v\",\"datapath\":\"%v\",\"predictpath\":\"%v\"}", "predict", datapath, predictpath))
	}
	return tasks
}


func main() {
	if len(os.Args) != 4 {
		fmt.Println(usage)
		return
	}
	numpredicts, _ := strconv.Atoi(os.Args[1])
	predictdatapath := os.Args[2]
	fitdatapath := os.Args[3]
	tasks := genPredicts(numpredicts, predictdatapath)
	fit := fmt.Sprintf("{\"command\":\"fit\", \"datapath\":\"%v\"}", fitdatapath)
	fmt.Println(fit)
	for _, str := range tasks {
		fmt.Println(str)
	}
}