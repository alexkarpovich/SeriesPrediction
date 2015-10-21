#include <stdio.h>
#include <string.h>
#include <cmath>
#include "../headers/RecurrentNetwork.h"


RecurrentNetwork::RecurrentNetwork(double * sequence, int sequenceSize, int inCount, int hidCount, double minError) {
	this->inCount = inCount;
	this->hidCount = hidCount;
	this->minError = minError;

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
		trainingSample[i] = new double[inCount + 1];

		memcpy(trainingSample[i], sequence + i, inCount * sizeof(double));
		// Add bias
		trainingSample[i][inCount] = -1;
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::prepareLayers() {
	hidden = new double[hidCount + 1];
	context = new double[hidCount + 1];
}

void RecurrentNetwork::prepareWeights() {
	cout << "- Prepare weights";
	int incIn = inCount + 1;
	int incHid = hidCount + 1;

	wih = new double*[incIn];
	for (int i = 0; i < incIn; i++) {
		wih[i] = new double[hidCount];
	}

	wch = new double*[incHid];
	for (int i = 0; i < incHid; i++) {
		wch[i] = new double[hidCount];
		context[i] = 0;
	}

	hidden[hidCount] = -1;
	context[incHid] = 0;
	who = new double[incHid];
	woh = new double[hidCount];
}

void RecurrentNetwork::initWeights() {
	int incIn = inCount + 1;
	int incHid = hidCount + 1;

	for (int i = 0; i < incIn; i++) {
		for (int j = 0; j < hidCount; j++) {
			wih[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	for (int i = 0; i < incHid; i++) {
		for (int j = 0; j < hidCount; j++) {
			wch[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	for (int i = 0; i < incHid; i++) {
		who[i] = FunctionService::getRandom(-1, 1);
	}

	for (int i = 0; i < hidCount; i++) {
		woh[i] = FunctionService::getRandom(-1, 1);
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::feedForward() {
	double S = 0;
	int incIn = inCount + 1;
	int incHid = hidCount + 1;

	for (int j = 0; j < hidCount; j++) {
		S = 0;

		for (int i = 0; i < incIn; i++) {
			S += wih[i][j] * inputs[i];
		} // Calculation with input bias

		for (int i = 0; i < hidCount; i++) {
			S += wch[i][j] * context[i];
		}

		S += woh[j] * context[hidCount];

		hidden[j] = activate(S);
	}

	S = 0.0;

	for (int i = 0; i < incHid; i++) {
		S += who[i] * hidden[i];
	} // Calculation with hidden bias

	actual = activate(S);

	// Copy hidden to context
	for (int i = 0; i < hidCount; i++) {
		context[i] = hidden[i];
	}

	// Copy output to context
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
	double a = 0.0001; //adaptiveStep();
	double diff = a * (actual - target);

	for (int i = 0; i < hidCount; i++) {
		for (int j = 0; j < inCount; j++) {
			wih[j][i] -= diff * who[i] * derivative(hidden[i]) * inputs[j];
		}

		for (int j = 0; j < hidCount; j++) {
			wch[j][i] -= diff * who[i] * derivative(hidden[i]) * context[j];
		}

		woh[i] -= diff * who[i] * derivative(hidden[i]) * context[hidCount];
		wih[inCount][i] += diff * who[i] * derivative(hidden[i]);
		who[i] -= diff * hidden[i];
	}

	who[hidCount] += diff;

	//normalizeWeights();
}

void RecurrentNetwork::normalizeWeights() {
	int incIn = inCount + 1;
	int incHid = hidCount + 1;
	double s = 0,
		   s1 = 0;

	for (int i = 0; i < incIn; i++) {
		s = 0;

		for (int j = 0; j < incHid; j++) {
			s += pow(wih[i][j], 2);
		}

		s = sqrt(s);

		for (int j = 0; j < incHid; j++) {
			wih[i][j] /= s;
		}
	}

	for (int i = 0; i < incHid; i++) {
		s = 0;

		for (int j = 0; j < incHid; j++) {
			s += pow(wch[i][j], 2);
		}

		s = sqrt(s);

		for (int j = 0; j < incHid; j++) {
			wch[i][j] /= s;
		}
	}

	s = s1 = 0;

	for (int i = 0; i < incHid; i++) {
		s += pow(who[i], 2);
		s1 += pow(woh[i], 2);
	}

	s = sqrt(s);
	s1 = sqrt(s1);

	for (int i = 0; i < incHid; i++) {
		who[i] /= s;
		woh[i] /= s1;
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

	} while (e > minError);
}

double * RecurrentNetwork::process(int predictCount) {
	int iteration = 0;
	double * predictedSequence = new double[predictCount];

	for (int i = 0; i < L; i++) {
		inputs = trainingSample[i];

		feedForward();
	}

	predictedSequence[0] = actual;

	do {
		++iteration;

		memcpy(inputs, inputs + 1, (inCount - 1) * sizeof(double));
		inputs[inCount - 1] = actual;

		feedForward();

		predictedSequence[iteration] = actual;


	} while (iteration < predictCount);

	return predictedSequence;
}
