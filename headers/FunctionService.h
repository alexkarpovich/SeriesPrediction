#ifndef FUNCTIONSERVICE_H_
#define FUNCTIONSERVICE_H_

#include <cstdlib>

class FunctionService {
public:
	static double getRandom(int min, int max);
	static double fibonacci(double n);
	static double factorial(double n);
	static double * getFibonacciSequence(int n);
	static double * getFactorialSequence(int n);
};

#endif /* FUNCTIONSERVICE_H_ */
