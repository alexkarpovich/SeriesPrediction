#ifndef RECURRENTNETWORK_H_
#define RECURRENTNETWORK_H_

#include "Neuron.h"
#include "FunctionService.h"
#include <iostream>

using namespace std;

class RecurrentNetwork {
private:
	int p;
	int m;
	int size;
	double minError;
	Neuron ** inputLayer;
	Neuron ** hiddenLayer;
	Neuron ** outputLayer;
	Neuron ** context;
	double ** inputWeights;
	double ** contextWeights;
	double * outputWeights;

	void prepareInputLayer(double * sequence);
	void prepareLayers();
	void prepareWeights();
	void initWeights();
public:
	RecurrentNetwork(double * sequence, int p, int m, double minError);
	void training();
	double * process();
};

#endif /* RECURRENTNETWORK_H_ */
