#pragma once

#include "ObjectiveFunction.h"
#include "Minimizer.h"

class GradientDescentFixedStep : public Minimizer {
public:
	GradientDescentFixedStep(int maxIterations=100, double solveResidual=1e-5)
		: maxIterations(maxIterations), solveResidual(solveResidual) {
	}

	int getLastIterations() { return lastIterations; }

	bool minimize(const ObjectiveFunction *function, VectorXd &x) const override {

		bool optimizationConverged = false;

		VectorXd dx(x.size());

		int i=0;
		for(; i < maxIterations; i++) {
			dx.setZero();
			computeSearchDirection(function, x, dx);

			if (dx.norm() < solveResidual){
				optimizationConverged = true;
				break;
			}

			step(function, dx, x);
		}

		lastIterations = i;

		return optimizationConverged;
	}

protected:
	virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) const {
		dx = function->getGradient(x);
	}

	// Given the objective `function` and the search direction `x`, update the candidate `x`
	virtual void step(const ObjectiveFunction *function, const VectorXd& dx, VectorXd& x) const	{
		// for fixed-step gradient descent, the step size is independent of the objective function
		x -= stepSize * dx;
	}

public:
	double solveResidual = 1e-5;
	int maxIterations = 1;
	double stepSize = 0.001;

	// some stats about the last time `minimize` was called
	mutable int lastIterations = -1;
};


class GradientDescentVariableStep : public GradientDescentFixedStep {
public:
	GradientDescentVariableStep(int maxIterations=100, double solveResidual=1e-5, int maxLineSearchIterations=15)
		: GradientDescentFixedStep (maxIterations, solveResidual), maxLineSearchIterations(maxLineSearchIterations){
	}

protected:
	void step(const ObjectiveFunction *function, const VectorXd& dx, VectorXd& x) const override
	{
		// line search
		double alpha = 1.0; // initial step size
		VectorXd xc(x);
		double initialValue = function->evaluate(xc);

		for(int j = 0; j < maxLineSearchIterations; j++) {

			// let's take a step of size `alpha`
			x = xc - dx * alpha;

			// if the new function value is greater than initial,
			// we want to reduce alpha and take a smaller step
			double f =function->evaluate(x);
			if(!std::isfinite(f) || f > initialValue)
				alpha /= 2.0;
			else
				return;
		}
	}

protected:
	int maxLineSearchIterations = 15;
};

class GradientDescentMomentum : public GradientDescentVariableStep {
public:
	GradientDescentMomentum(int maxIterations=100, double solveResidual=1e-5, int maxLineSearchIterations=15)
		: GradientDescentVariableStep(maxIterations, solveResidual, maxLineSearchIterations){
	}


protected:
	void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) const override {

		// if gradient hasn't been set yet, set it to zero
		if(gradient.size() == 0){
			gradient.resize(x.size());
			gradient.setZero();
		}
		// compute new gradient
		VectorXd newGradient = function->getGradient(x);
		// search direction is augmented with old gradient
		dx = newGradient + alpha*gradient;
		// save gradient for next step
		gradient = newGradient;
	}

public:
	double alpha = 0.8;
	mutable VectorXd gradient;
};
