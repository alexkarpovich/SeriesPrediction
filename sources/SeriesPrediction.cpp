#include <iostream>
#include "../headers/RecurrentNetwork.h"
#include "../headers/FunctionService.h"

using namespace std;

int main() {

	int imageSize = 3;
	int hiddenSize = 4;
	int sequenceSize = 9;
	int predictCount = 4;
	double minError = 0.001;

//	double * sequence = FunctionService::getFibonacciSequence(sequenceSize);
//
//	cout << "Analysing sequence: " << endl;
//
//	for (int i = 0; i < sequenceSize; i++) {
//		cout << sequence[i] << " ";
//	}
//	cout << endl;

	double * sequence = new double[sequenceSize]{
		0, 2, 4, 2, 0, -2, -4, -2, 0
	};

	cout << "Recurrent Network" << endl;

	RecurrentNetwork * network = new RecurrentNetwork(sequence, sequenceSize, imageSize, hiddenSize, minError);
	network->training();

	double * predictedSequence = network->process(predictCount);

	cout << "Predicted sequence: " << endl;
	for (int i = 0; i < predictCount; i++) {
		cout << predictedSequence[i] << " ";
	}
	cout << endl;

	return 0;
}
