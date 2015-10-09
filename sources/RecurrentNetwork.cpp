#include "../headers/RecurrentNetwork.h"

RecurrentNetwork::RecurrentNetwork(int * sequence, int p, int m, double minError) {
	this->p = p;
	this->m = m;
	this->minError = minError;

	this->prepareLayers();
	this->prepareInputLayer(sequence);
}

void RecurrentNetwork::prepareLayers() {
	this->inputLayer = new Neuron[this->m];
	this->outputLayer = new Neuron[this->m];
}

void RecurrentNetwork::prepareInputLayer(int * sequence) {

}
