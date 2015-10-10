#include "../headers/Neuron.h"

Neuron::Neuron(double * image, int p) {
	this->p = p;
	this->image = image;
}

Neuron::~Neuron() {
	delete image;
}

int Neuron::getImageSize() {
	return this->p;
}

double Neuron::get(int i) {
	if (i >= this->p) {
		cout << "Index out of range" << endl;
	}

	return this->image[i];
}
