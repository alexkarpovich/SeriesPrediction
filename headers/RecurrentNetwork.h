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
	int conCount;
	int L;
	int size;
	double minError;

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

	void prepareTrainingSample(double * sequence);
	void prepareLayers();
	void prepareWeights();
	void initWeights();
	void feedForward();
	void backPropagation();
	double activate(double S);
public:
	RecurrentNetwork(double * sequence, int inCount, int hidCount, int L, double minError);
	void training();
	double * process();
};

#endif /* RECURRENTNETWORK_H_ */
