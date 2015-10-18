#include <stdio.h>
#include <string.h>
#include <cmath>
#include "../headers/RecurrentNetwork.h"


RecurrentNetwork::RecurrentNetwork(double * sequence, int sequenceSize, int inCount, int hidCount, double minError) {
	this->inCount = inCount;
	this->hidCount = hidCount;
	this->minError = minError;
	this->learnRate = 0.2;

	prepareTrainingSample(sequence, sequenceSize);
	prepareLayers();
	prepareWeights();
	initWeights();
}

void RecurrentNetwork::prepareTrainingSample(double * sequence, int sequenceSize) {
	cout << "- Prepare training sample";

	L = sequenceSize - inCount + 1;

	trainingSample = new double*[L];
	for (int i = 0; i < L; i++) {
		trainingSample[i] = new double[inCount];

		memcpy(trainingSample[i], sequence + i, inCount * sizeof(double));
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::prepareLayers() {
	hidden = new double[hidCount];
	context = new double[hidCount + 1];
}

void RecurrentNetwork::prepareWeights() {
	cout << "- Prepare weights";

	wih = new double*[inCount];
	for (int i = 0; i < inCount; i++) {
		wih[i] = new double[hidCount];
	}

	wch = new double*[hidCount];
	for (int i = 0; i < hidCount; i++) {
		wch[i] = new double[hidCount];
		context[i] = 0;
	}

	context[hidCount] = 0;
	who = new double[hidCount];
	woh = new double[hidCount];
}

void RecurrentNetwork::initWeights() {
	for (int i = 0; i < inCount; i++) {
		for (int j = 0; j < hidCount; j++) {
			wih[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	for (int i = 0; i < hidCount; i++) {
		for (int j = 0; j < hidCount; j++) {
			wch[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	for (int i = 0; i < hidCount; i++) {
		who[i] = woh[i] = FunctionService::getRandom(-1, 1);
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::feedForward() {
	double S = 0;

	for (int j = 0; j < hidCount; j++) {
		S = 0;

		for (int i = 0; i < inCount; i++) {
			S += wih[i][j] * inputs[i];
		}

		for (int i = 0; i < hidCount; i++) {
			S += wch[i][j] * context[i];
		}

		S += woh[j] * context[hidCount];

		hidden[j] = activate(S);
	}

	S = 0.0;

	for (int i = 0; i < hidCount; i++) {
		S += who[i] * hidden[i];
		context[i] = hidden[i];
	}

	actual = activate(S);

	context[hidCount] = actual;
}

double RecurrentNetwork::error() {
	return pow(target - actual, 2) / 2;
}

double RecurrentNetwork::adaptiveStep() {
	double numerator = 0;
	double SI = 1;
	double SO = 0;

	for (int i = 0; i < hidCount; i++) {
		numerator += pow(who[i], 2) * derivative(hidden[i]);
		SO += pow(who[i] * derivative(hidden[i]), 2);
	}

	for (int i = 0; i < inCount; i++) {
		SI += pow(inputs[i], 2);
	}

	return numerator / (SI * SO);
}

void RecurrentNetwork::backPropagation() {
	double a = 0.1; //adaptiveStep();
	double diff = a * (target - actual);

	for (int i = 0; i < hidCount; i++) {
		for (int j = 0; j < inCount; j++) {
			wih[j][i] -= diff * who[i] * derivative(hidden[i]) * inputs[j];
		}

		for (int j = 0; j < hidCount; j++) {
			wch[j][i] -= diff * who[i] * derivative(hidden[i]) * context[j];
		}

		who[i] -= diff * hidden[i];
		woh[i] -= diff * who[i] * derivative(hidden[i]) * context[hidCount];
	}
}

double RecurrentNetwork::activate(double S) {
	return log(S + sqrt(S * S + 1));
}

double RecurrentNetwork::derivative(double y) {
	return 1 / sqrt(y * y + 1);
}

void RecurrentNetwork::training() {
	double e;
	int iteration = 0;

	cout << "- Training:" << endl;

	do {
		e = 0;
		++iteration;

		for (int i = 0; i < L - 1; i++) {
			inputs = trainingSample[i];
			target = trainingSample[i + 1][inCount - 1];

			feedForward();

			e += error();

			backPropagation();
		}

		cout << "[ " << iteration << " ] E = " << e << ", actual: " << actual << ", target: " << target << endl;

	} while (e > 0.05);
}

double * RecurrentNetwork::process(int predictCount) {
	int iteration = 0;
	double * predictedSequence = new double[predictCount];

	do {
		++iteration;


	} while (iteration < predictCount);

	return predictedSequence;
}
