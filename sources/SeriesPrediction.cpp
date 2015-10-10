#include <iostream>
#include "../headers/RecurrentNetwork.h"
#include "../headers/FunctionService.h"

using namespace std;

int main() {

	int p = 3;
	int m = 10;
	double minError = 0.01;

	int seqSize = m + p - 1;

	double * sequence = FunctionService::getFibonacciSequence(seqSize);

	cout << "Analysing sequence: " << endl;

	for (int i=0; i<seqSize; i++) {
		cout << sequence[i] << " ";
	}
	cout << endl;

	RecurrentNetwork * network = new RecurrentNetwork(sequence, p, m, minError);
	return 0;
}
