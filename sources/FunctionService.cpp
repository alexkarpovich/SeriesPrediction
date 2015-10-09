#include "../headers/FunctionService.h"

int FunctionService::fibonacci(int n) {
	if (n < 3) {
		return 1;
	}

	return FunctionService::fibonacci(n-1) + FunctionService::fibonacci(n-2);
}

int FunctionService::factorial(int n) {
	if (n < 2) {
		return n;
	}

	return n * FunctionService::factorial(n-1);
}

int * FunctionService::getFibonacciSequence(int n) {
	int * fibonacciSeq = new int[n];

	for (int i=0; i<n; i++) {
		fibonacciSeq[i] = FunctionService::fibonacci(i+1);
	}

	return fibonacciSeq;
}

int * FunctionService::getFactorialSequence(int n) {
	int * factorialSeq = new int[n];

	for (int i=0; i<n; i++) {
		factorialSeq[i] = FunctionService::factorial(i+1);
	}

	return factorialSeq;
}
