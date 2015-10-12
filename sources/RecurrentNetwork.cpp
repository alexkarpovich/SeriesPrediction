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
	hidden = new double[hidCount + 1];
	context = new double[hidCount + inCount + 1];
	actual = new double[inCount + 1];
	target = new double[inCount + 1];

	oError = new double[inCount];
	hError = new double[hidCount];
}

void RecurrentNetwork::prepareWeights() {
	cout << "- Prepare weights";

	int iwSize = inCount + 1;
	int cwSize = hidCount + inCount + 1;
	int owSize = hidCount + 1;

	ihWeights = new double*[iwSize];
	for (int i = 0; i < iwSize; i++) {
		ihWeights[i] = new double[hidCount];
	}

	chWeights = new double*[cwSize];
	for (int i = 0; i < cwSize; i++) {
		chWeights[i] = new double[hidCount];
	}

	hoWeights = new double*[owSize];
	for (int i = 0; i < owSize; i++) {
		hoWeights[i] = new double[inCount];
	}
}

void RecurrentNetwork::initWeights() {
	int iwSize = inCount + 1;
	int cwSize = hidCount + inCount + 1;
	int owSize = hidCount + 1;

	for (int i = 0; i < iwSize; i++) {
		for (int j = 0; j < hidCount; j++) {
			ihWeights[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	for (int i = 0; i < cwSize; i++) {
		for (int j = 0; j < hidCount; j++) {
			chWeights[i][j] = 1;
		}
	}

	for (int i = 0; i < owSize; i++) {
		for (int j = 0; j < inCount; j++) {
			hoWeights[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::feedforward() {
	double S = 0;
	int conCount = hidCount + inCount + 1;

	for (int j = 0; j < hidCount; j++) {
		for (int i = 0; i < inCount; i++) {
			S += ihWeights[i][j] * inputs[i];
		}

		for (int i = 0; i < conCount; i++) {
			S += chWeights[i][j] * context[i];
		}

		// Add bias
		S += ihWeights[inCount][j];
		S += chWeights[conCount - 1][j];

		hidden[j] = activate(S);
	}

	for (int i = 0; i < inCount; i++) {
		S = 0.0;

		for (int j = 0; j < hidCount; j++) {
			S += hoWeights[j][i] * hidden[j];
		}

		S += hoWeights[hidCount][i];

		actual[i] = activate(S);
	}

	for (int i = 0; i < hidCount; i++) {
		context[i] = hidden[i];
	}

	for (int i = 0; i < inCount; i++) {
		context[i + i] = actual[i];
	}
}

double RecurrentNetwork::error() {
	double err = 0;

	for (int i = 0; i < inCount; i++) {
		err += pow(target[i] - actual[i], 2);
	}

	return err;
}

void RecurrentNetwork::backpropagation() {
	for (int i = 0; i < inCount; i++) {
		oError[i] = (target[i] - actual[i]) * derivative(actual[i]);
	}

	for (int i = 0; i < hidCount; i++) {
		hError[i] = 0;

		for (int j = 0; j < inCount; j++) {
			hError[i] += oError[j] * hoWeights[i][j];
		}

		hError[i] *= derivative(hidden[i]);
	}

	for (int j = 0; j < inCount; j++) {
		for (int i = 0; i < hidCount; i++) {
			hoWeights[i][j] += learnRate * oError[j] * hidden[i];
		}

		hoWeights[hidCount][j] += learnRate * oError[j];
	}

	for (int j = 0; j < hidCount; j++) {
		for (int i = 0; i < inCount; i++) {
			ihWeights[i][j] += learnRate * hError[j] * inputs[i];
		}

		ihWeights[inCount][j] += learnRate * hError[j];
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
	double err;
	int iteration = 0;

	cout << "- Training:" << endl;

	do {
		e = 0;
		++iteration;

		for (int i = 0; i < L - 1; i++) {
			inputs = trainingSample[i];
			target = trainingSample[i + 1];

			feedforward();

			e += error();
			e /= 2;

			backpropagation();
		}

		cout << "[ " << iteration << " ] E = " << e << ", actual: [ ";
		for (int i = 0; i < inCount; i++) {
			cout << actual[i] << " ";
		}
		cout << "]" << endl;

	} while (iteration < 1000000);
}

double * RecurrentNetwork::process(int predictCount) {
	int iteration = 0;
	double * predictedSequence = new double[predictCount];

	do {
		++iteration;


	} while (iteration < predictCount);

	return predictedSequence;
}
