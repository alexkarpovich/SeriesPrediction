#include <stdio.h>
#include <string.h>
#include <cmath>
#include "../headers/RecurrentNetwork.h"


RecurrentNetwork::RecurrentNetwork(double * sequence, int inCount, int hidCount, int L, double minError) {
	this->inCount = inCount;
	this->hidCount = hidCount;
	this->conCount = hidCount;
	this->L = L;
	this->size = this->inCount + this->L;
	this->minError = minError;

	this->prepareTrainingSample(sequence);
	this->prepareLayers();
	this->prepareWeights();
	this->initWeights();
}

void RecurrentNetwork::prepareTrainingSample(double * sequence) {
	this->trainingSample = new double*[this->L];
	for (int i = 0; i < this->L; i++) {
		this->trainingSample[i] = new double[this->inCount];

		memcpy(this->trainingSample[i], sequence + i, this->inCount * sizeof(double));
	}

}

void RecurrentNetwork::prepareLayers() {
	this->hidden = new double[this->hidCount + 1];
	this->context = new double[this->hidCount + 1];
	this->actual = new double[this->inCount + 1];
	this->target = new double[this->inCount + 1];
}

void RecurrentNetwork::prepareWeights() {
	int iwSize = this->inCount + 1;
	int cwSize = this->conCount + 1;

	this->ihWeights = new double*[iwSize];
	for (int i = 0; i < iwSize; i++) {
		this->ihWeights[i] = new double[this->hidCount];
	}

	this->chWeights = new double*[cwSize];
	for (int i = 0; i < cwSize; i++) {
		this->chWeights[i] = new double[this->hidCount];
	}

	this->hoWeights = new double*[iwSize];
	for (int i = 0; i < iwSize; i++) {
		this->chWeights[i] = new double[this->hidCount];
	}
}

void RecurrentNetwork::initWeights() {
	int iwSize = this->inCount + 1;
	int cwSize = this->conCount + 1;

	for (int i = 0; i < iwSize; i++) {
		for (int j = 0; j < this->hidCount; j++) {
			this->ihWeights[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	for (int i = 0; i < cwSize; i++) {
		for (int j = 0; j < this->hidCount; j++) {
			this->chWeights[i][j] = 0;
		}
	}

	for (int i = 0; i < iwSize; i++) {
		for (int j = 0; j < this->hidCount; j++) {
			this->hoWeights[i][j] = FunctionService::getRandom(-1, 1);
		}
	}
}

void RecurrentNetwork::feedForward() {
	double S = 0;

	for (int j = 0; j < this->hidCount; j++) {
		for (int i = 0; i < this->inCount; i++) {
			S += this->ihWeights[i][j] * this->inputs[i];
		}

		for (int i = 0; i < this->conCount; i++) {
			S += this->chWeights[i][j] * this->context[i];
		}

		// Add bias
		S += this->ihWeights[this->inCount][j];
		S += this->chWeights[this->conCount][j];

		this->hidden[j] = this->activate(S);
	}

	for (int i = 0; i < this->inCount; i++) {
		S = 0.0;

		for (int j = 0; j < this->hidCount; j++) {
			S += this->hoWeights[j][i] * this->hidden[j];
		}

		S += this->hoWeights[this->hidCount][i];

		this->actual[i] = this->activate(S);
	}

	for (int i = 0; i < this->hidCount; i++) {
		this->context[i] = this->hidden[i];
	}
}

void RecurrentNetwork::backPropagation() {

}

double RecurrentNetwork::activate(double S) {
	return log(S + sqrt(S * S + 1));
}

void RecurrentNetwork::training() {
	double e;
	int iteration = 0;

	cout << "Training started" << endl;

	do {
		e = 0;
		++iteration;

		for (int i = 0; i < this->L; i++) {
			this->inputs = this->trainingSample[i];

			this->feedForward();

			this->backPropagation();
		}

		cout << "[ " << iteration << " ] E = " << e << endl;

	} while (iteration < 1000);
}

double * RecurrentNetwork::process() {

}
