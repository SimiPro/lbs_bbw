#pragma once

#include "ObjectiveFunction.h"

class Minimizer
{
public:
	// Returns true if a minimum of the objective `function` has been found.
	// `x` is the initial/current candidate, and will also store the next
	// candidate once the method has returned.
	virtual bool minimize(const ObjectiveFunction *function, VectorXd &x) const = 0;
};
