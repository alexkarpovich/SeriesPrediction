#include <stdio.h>
#include <string.h>
#include <cmath>
#include "../headers/RecurrentNetwork.h"

int BIAS = -1;
int incIn = 0;
int incHid = 0;


RecurrentNetwork::RecurrentNetwork(double * sequence, int sequenceSize, int inCount, int hidCount, double minError) {
	this->inCount = inCount;
	this->hidCount = hidCount;
	this->minError = activate(minError);
	incIn = inCount + 1;
	incHid = hidCount + 1;

	srand(0);

	prepareTrainingSample(sequence, sequenceSize);
	prepareLayers();
	prepareWeights();
	initWeights();
}

void RecurrentNetwork::prepareTrainingSample(double * sequence, int sequenceSize) {
	cout << "- Prepare training sample";

	L = sequenceSize - inCount + 1;

	double * preparedSequence = new double[sequenceSize];

	for (int i = 0; i < sequenceSize; i++) {
		preparedSequence[i] = activate(sequence[i]);
	}

	trainingSample = new double*[L];
	for (int i = 0; i < L; i++) {
		trainingSample[i] = new double[inCount + 1];

		memcpy(trainingSample[i], preparedSequence + i, inCount * sizeof(double));
		trainingSample[i][inCount] = BIAS;
	}

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::prepareLayers() {
	hidden = new double[hidCount + 1];
	context = new double[hidCount + 1];
}

void RecurrentNetwork::prepareWeights() {
	cout << "- Prepare weights";

	wih = new double*[incIn];
	for (int i = 0; i < incIn; i++) {
		wih[i] = new double[hidCount];
	}

	wch = new double*[hidCount];
	for (int i = 0; i < hidCount; i++) {
		wch[i] = new double[hidCount];
	}

	who = new double[incHid];
	woh = new double[hidCount];
}

void RecurrentNetwork::initWeights() {
	for (int i = 0; i <= inCount; i++) {
		for (int j = 0; j < hidCount; j++) {
			wih[i][j] = FunctionService::getRandom(-1, 1);
		}
	} // Init input to hidden weights

	for (int i = 0; i < hidCount; i++) {
		for (int j = 0; j < hidCount; j++) {
			wch[i][j] = FunctionService::getRandom(-1, 1);
		}
	} // Init context to hidden weights

	for (int i = 0; i <= hidCount; i++) {
		who[i] = FunctionService::getRandom(-1, 1);
	} // Init hidden to output weights

	for (int i = 0; i < hidCount; i++) {
		woh[i] = FunctionService::getRandom(-1, 1);
	} // Init context output to hidden

	for (int i = 0; i <= hidCount; i++) {
		context[i] = 0;
	} // Init context neurons as 0

	hidden[hidCount] = BIAS;

	cout << " [ DONE ]" << endl;
}

void RecurrentNetwork::feedForward() {
	double s = 0;

	for (int j = 0; j < hidCount; j++) {
		s = 0;

		for (int i = 0; i <= inCount; i++) {
			s += wih[i][j] * inputs[i];
		} // Sum inputs to hidden

		for (int i = 0; i < hidCount; i++) {
			s += wch[i][j] * context[i];
		} // Sum hidden-context to hidden

		// Sum hidden-output to hidden
		s += woh[j] * context[hidCount];

		hidden[j] = activate(s);
	} // Calculate hidden neurons

	s = 0;

	for (int i = 0; i <= hidCount; i++) {
		s += who[i] * hidden[i];
	} // Calculate output neuron

	actual = activate(s);
}

double RecurrentNetwork::error() {
	return pow(actual - target, 2) / 2;
}

double RecurrentNetwork::adaptiveStep() {
	double numerator = 0;
	double SI = 1;
	double SO = 0;

	for (int i = 0; i <= hidCount; i++) {
		numerator += pow(who[i], 2) * derivative(hidden[i]);
		SO += pow(who[i] * derivative(hidden[i]), 2);
	}

	for (int i = 0; i < inCount; i++) {
		SI += pow(inputs[i], 2);
	}

	return numerator / (SI * SO);
}

void RecurrentNetwork::backPropagation() {
	double a = 0.0005; //adaptiveStep();
	double diff = a * (actual - target);

	for (int j = 0; j < hidCount; j++) {
		for (int i = 0; i < inCount; i++) {
			wih[i][j] -= diff * who[j] * derivative(hidden[j]) * inputs[i];
		} // Correct input to hidden weights

		for (int i = 0; i < hidCount; i++) {
			wch[j][i] -= diff * who[j] * derivative(hidden[j]) * context[i];
		} // Correct hidden-context to hidden weights

		// Correct output-context to hidden weights
		woh[j] -= diff * who[j] * derivative(hidden[j]) * context[hidCount];

		// Correct input to hidden bias
		wih[inCount][j] += diff * who[j] * derivative(hidden[j]);

		who[j] -= diff * hidden[j];
	}

	// Correct hidden to output bias
	who[hidCount] += diff;

	//normalizeWeights();
}

void RecurrentNetwork::normalizeWeights() {
	double s = 0,
		   s1 = 0;

	for (int i = 0; i <= inCount; i++) {
		s = 0;

		for (int j = 0; j < hidCount; j++) {
			s += pow(wih[i][j], 2);
		}

		s = sqrt(s);

		for (int j = 0; j < hidCount; j++) {
			wih[i][j] /= s;
		}
	}

	for (int i = 0; i < hidCount; i++) {
		s = 0;

		for (int j = 0; j < hidCount; j++) {
			s += pow(wch[i][j], 2);
		}

		s = sqrt(s);

		for (int j = 0; j < hidCount; j++) {
			wch[i][j] /= s;
		}
	}

	s = s1 = 0;

	for (int i = 0; i <= hidCount; i++) {
		s += pow(who[i], 2);
		s1 += pow(woh[i], 2);
	}

	s = sqrt(s);
	s1 = sqrt(s1);

	for (int i = 0; i <= hidCount; i++) {
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

	predictedSequence[0] = sinh(actual);

	do {
		++iteration;

		memcpy(inputs, inputs + 1, (inCount - 1) * sizeof(double));
		inputs[inCount - 1] = actual;

		feedForward();

		predictedSequence[iteration] = sinh(actual);


	} while (iteration < predictCount);

	return predictedSequence;
}
