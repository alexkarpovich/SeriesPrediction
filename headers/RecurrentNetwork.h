#ifndef RECURRENTNETWORK_H_
#define RECURRENTNETWORK_H_

#include "Neuron.h"
#include "FunctionService.h"
#include <iostream>

using namespace std;

class RecurrentNetwork {
private:
	int inCount;
	int hidCount;
	int L;
	int size;
	double minError;
	double learnRate;

	double ** trainingSample;

	double * inputs;
	double * hidden;
	double * context;
	double * output;
	double * target;
	double * actual;

	double ** ihWeights;
	double ** chWeights;
	double ** hoWeights;

	double * oError;
	double * hError;

	void prepareTrainingSample(double * sequence, int sequenceSize);
	void prepareLayers();
	void prepareWeights();
	void initWeights();
	void feedforward();
	void backpropagation();
	double error();
	double activate(double S);
	double derivative(double y);
public:
	RecurrentNetwork(double * sequence, int sequenceSize, int inCount, int hidCount, double minError);
	void training();
	double * process(int predictCount);
};

#endif /* RECURRENTNETWORK_H_ */
