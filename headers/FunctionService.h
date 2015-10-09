#ifndef FUNCTIONSERVICE_H_
#define FUNCTIONSERVICE_H_

class FunctionService {
public:
	static int fibonacci(int n);
	static int factorial(int n);
	static int * getFibonacciSequence(int n);
	static int * getFactorialSequence(int n);
};

#endif /* FUNCTIONSERVICE_H_ */
