#include <stdio.h>
#include <string.h>
#include "../headers/RecurrentNetwork.h"


RecurrentNetwork::RecurrentNetwork(double * sequence, int p, int m, double minError) {
	this->p = p;
	this->m = m;
	this->size = this->p + this->m;
	this->minError = minError;

	this->prepareLayers();
	this->prepareInputLayer(sequence);
	this->prepareWeights();
	this->initWeights();
}

void RecurrentNetwork::prepareLayers() {
	this->inputLayer = new Neuron*[this->m];
	this->outputLayer = new Neuron*[this->m];
}

void RecurrentNetwork::prepareInputLayer(double * sequence) {
	for (int i=0; i<this->m; i++) {
		double * image = new double[this->p];
		memcpy(image, sequence + i, sizeof(double) * this->p);

		this->inputLayer[i] = new Neuron(image, this->p);
	}
}

void RecurrentNetwork::prepareWeights() {
	this->inputWeights = new double*[this->p + this->m];
	for (int i=0; i<this->size; i++) {
		this->inputWeights[i] = new double[this->p];
	}

	this->outputWeights =  new double[this->m];
}

void RecurrentNetwork::initWeights() {
	for (int i=0; i<this->size; i++) {
		for (int j=0; j<this->p; j++) {
			this->inputWeights[i][j] = FunctionService::getRandom(-1, 1);
		}
	}

	for (int i=0; i<this->m; i++) {
		this->outputWeights[i] = FunctionService::getRandom(-1, 1);
	}
}

void RecurrentNetwork::training() {

}

double * RecurrentNetwork::process() {

}
