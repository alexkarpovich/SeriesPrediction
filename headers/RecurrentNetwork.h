#ifndef RECURRENTNETWORK_H_
#define RECURRENTNETWORK_H_

#include "Neuron.h"

class RecurrentNetwork {
private:
	int p;
	int m;
	double minError;
	Neuron * inputLayer;
	Neuron * hiddenLayer;
	Neuron * outputLayer;
	Neuron * context;
	double ** inputWeights;
	double ** contextWeights;
	double ** outputWeights;

	void prepareInputLayer(int * sequence);
	void prepareLayers();
public:
	RecurrentNetwork(int * sequence, int p, int m, double minError);
};

#endif /* RECURRENTNETWORK_H_ */
