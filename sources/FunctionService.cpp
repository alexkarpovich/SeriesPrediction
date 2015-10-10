#include "../headers/FunctionService.h"

double FunctionService::getRandom(int min, int max) {
	return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

double FunctionService::fibonacci(double n) {
	if (n < 3) {
		return 1;
	}

	return FunctionService::fibonacci(n-1) + FunctionService::fibonacci(n-2);
}

double FunctionService::factorial(double n) {
	if (n < 2) {
		return n;
	}

	return n * FunctionService::factorial(n-1);
}

double * FunctionService::getFibonacciSequence(int n) {
	double * fibonacciSeq = new double[n];

	for (int i=0; i<n; i++) {
		fibonacciSeq[i] = FunctionService::fibonacci(i+1);
	}

	return fibonacciSeq;
}

double * FunctionService::getFactorialSequence(int n) {
	double * factorialSeq = new double[n];

	for (int i=0; i<n; i++) {
		factorialSeq[i] = FunctionService::factorial(i+1);
	}

	return factorialSeq;
}
