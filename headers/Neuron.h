#ifndef NEURON_H_
#define NEURON_H_

#include <iostream>

using namespace std;

class Neuron {
private:
	double * image;
	int p;
public:
	Neuron(double * image, int p);
	int getImageSize();
	double get(int i);

	~Neuron();
};

#endif /* NEURON_H_ */
