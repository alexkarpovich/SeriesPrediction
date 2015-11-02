#include <iostream>
#include "../headers/RecurrentNetwork.h"
#include "../headers/FunctionService.h"

using namespace std;

int main() {

	int imageSize = 3;
	int hiddenSize = 5;
	int sequenceSize = 15;
	int predictCount = 4;
	double minError = 0.00001;

	double * sequence = FunctionService::getFibonacciSequence(sequenceSize);

	cout << "Analysing sequence: " << endl;

	for (int i = 0; i < sequenceSize; i++) {
		cout << sequence[i] << " ";
	}
	cout << endl;

//	double * sequence = new double[sequenceSize]{
//		0, 2, 4, 2, 0, -2, -4, -2, 0
//	};

//	double * sequence = new double[sequenceSize]{
//		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
//	};

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
