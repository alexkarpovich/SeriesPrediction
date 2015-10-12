#include <iostream>
#include "../headers/RecurrentNetwork.h"
#include "../headers/FunctionService.h"

using namespace std;

int main() {

	int imageSize = 3;
	int hiddenSize = 4;
	int sequenceSize = 15;
	double minError = 0.01;

	double * sequence = FunctionService::getFibonacciSequence(sequenceSize);

	cout << "Analysing sequence: " << endl;

	for (int i = 0; i < sequenceSize; i++) {
		cout << sequence[i] << " ";
	}
	cout << endl;

	cout << "Recurrent Network" << endl;

	RecurrentNetwork * network = new RecurrentNetwork(sequence, sequenceSize, imageSize, hiddenSize, minError);
	network->training();

	return 0;
}
