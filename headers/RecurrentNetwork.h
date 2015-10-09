#ifndef RECURRENTNETWORK_H_
#define RECURRENTNETWORK_H_

#include "Neuron.h"

class RecurrentNetwork {
private:
	Neuron * inputLayer;
	Neuron * hiddenLayer;
	Neuron * outputLayer;
public:
	RecurrentNetwork();
};

#endif /* RECURRENTNETWORK_H_ */
