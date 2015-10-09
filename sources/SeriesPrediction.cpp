//============================================================================
// Name        : SeriesPrediction.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "../headers/RecurrentNetwork.h"

using namespace std;

int main() {

	double ** inputImages = new double*[10];

	for (int i=0; i < 10; i++) {
		inputImages[i] = new double[10];
	}

	RecurrentNetwork network(inputImages, 10);
	return 0;
}
