#ifndef RECURRENTNETWORK_H_
#define RECURRENTNETWORK_H_

#include "FunctionService.h"
#include <iostream>

using namespace std;

class RecurrentNetwork {
private:
	int inCount;
	int hidCount;
	int L;
	double minError;

	double ** trainingSample;

	double * inputs;
	double * hidden;
	double * context;
	double target;
	double actual;

	double ** wih;
	double ** wch;
	double * who;
	double * woh;

	void prepareTrainingSample(double * sequence, int sequenceSize);
	void prepareLayers();
	void prepareWeights();
	void initWeights();
	void feedForward();
	void backPropagation();
	void normalizeWeights();
	double error();
	double activate(double S);
	double derivative(double y);
	double adaptiveStep();
public:
	RecurrentNetwork(double * sequence, int sequenceSize, int inCount, int hidCount, double minError);
	void training();
	double * process(int predictCount);
};

#endif /* RECURRENTNETWORK_H_ */
