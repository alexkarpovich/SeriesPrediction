#include <stdio.h>
#include <string.h>
#include <cmath>
#include "../headers/RecurrentNetwork.h"


RecurrentNetwork::RecurrentNetwork(double * sequence, int inCount, int hidCount, int L, double minError) {
	cout << "Recurrent Network:" << endl;

	this->inCount = inCount;
	this->hidCount = hidCount;
	this->conCount = hidCount;
	this->L = L;
	this->size = this->inCount + this->L;
	this->minError = minError;
	this->learnRate = 0.2;

	this->prepareTrainingSample(sequence);
	this->prepareLayers();
	this->prepareWeights();
	this->initWeights();
}

void RecurrentNetwork::prepareTrainingSample(double * sequence) {
	cout << "- Prepare training sample";

	this->trainingSample = new double*[this->L];
	for (int i = 0; i < this->L; i++) {
		this->trainingSample[i] = new double[this->inCount];

		memcpy(this->trainingSample[i], sequence + i, this->inCount * sizeof(double));
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::prepareLayers() {
	this->hidden = new double[this->hidCount + 1];
	this->context = new double[this->hidCount + 1];
	this->actual = new double[this->inCount + 1];
	this->target = new double[this->inCount + 1];

	this->oError = new double[this->inCount];
	this->hError = new double[this->hidCount];
}

void RecurrentNetwork::prepareWeights() {
	cout << "- Prepare weights";

	int iwSize = this->inCount + 1;
	int cwSize = this->conCount + 1;
	int owSize = this->hidCount + 1;

	this->ihWeights = new double*[iwSize];
	for (int i = 0; i < iwSize; i++) {
		this->ihWeights[i] = new double[this->hidCount];
	}

	this->chWeights = new double*[cwSize];
	for (int i = 0; i < cwSize; i++) {
		this->chWeights[i] = new double[this->hidCount];
	}

	this->hoWeights = new double*[owSize];
	for (int i = 0; i < owSize; i++) {
		this->hoWeights[i] = new double[this->inCount];
	}
}

void RecurrentNetwork::initWeights() {
	int iwSize = this->inCount + 1;
	int cwSize = this->conCount + 1;
	int owSize = this->hidCount + 1;

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

	for (int i = 0; i < owSize; i++) {
		for (int j = 0; j < this->inCount; j++) {
			this->hoWeights[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::feedforward() {
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

double RecurrentNetwork::error() {
	double err = 0;

	for (int i = 0; i < this->inCount; i++) {
		err += pow(this->target[i] - this->actual[i], 2);
	}

	return err;
}

void RecurrentNetwork::backpropagation() {
	for (int i = 0; i < this->inCount; i++) {
		this->oError[i] = (this->target[i] - this->actual[i]) * this->derivative(this->actual[i]);
	}

	for (int i = 0; i < this->hidCount; i++) {
		this->hError[i] = 0;

		for (int j = 0; j < this->inCount; j++) {
			this->hError[i] += this->oError[j] * this->hoWeights[i][j];
		}

		this->hError[i] *= this->derivative(this->hidden[i]);
	}

	for (int j = 0; j < this->inCount; j++) {
		for (int i = 0; i < this->hidCount; i++) {
			this->hoWeights[i][j] += this->learnRate * this->oError[j] * this->hidden[i];
		}

		this->hoWeights[this->hidCount][j] += this->learnRate * this->oError[j];
	}

	for (int j = 0; j < this->hidCount; j++) {
		for (int i = 0; i < this->inCount; i++) {
			this->ihWeights[i][j] += this->learnRate * this->hError[j] * this->inputs[i];
		}

		this->ihWeights[this->inCount][j] += this->learnRate * this->hError[j];
	}
}

double RecurrentNetwork::activate(double S) {
	return log(S + sqrt(S * S + 1));
}

double RecurrentNetwork::derivative(double y) {
	return (1 + y / (y * y + 1)) / (y + sqrt(y * y + 1));
}

void RecurrentNetwork::training() {
	double e;
	double err;
	int iteration = 0;

	cout << "- Training:" << endl;

	do {
		e = 0;
		++iteration;

		for (int i = 0; i < this->L; i++) {
			this->inputs = this->trainingSample[i];
			this->target = this->trainingSample[i];

			this->feedforward();

			e += this->error();

			this->backpropagation();
		}

		cout << "[ " << iteration << " ] E = " << e << endl;

	} while (iteration < 1000);
}

double * RecurrentNetwork::process() {

}
