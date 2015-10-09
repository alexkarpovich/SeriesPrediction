#ifndef RECURRENTNETWORK_H_
#define RECURRENTNETWORK_H_

#include "Neuron.h"

class RecurrentNetwork {
private:
	Neuron * inputLayer;
	Neuron * hiddenLayer;
	Neuron * outputLayer;
	Neuron * context;
	double ** inputWeights;
	double ** contextWeights;
	double ** outputWeights;
public:
	RecurrentNetwork(double ** inputImages, int p);
};

#endif /* RECURRENTNETWORK_H_ */
