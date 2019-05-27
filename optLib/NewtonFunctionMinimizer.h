	#pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"

class NewtonFunctionMinimizer : public GradientDescentVariableStep {
public:
	NewtonFunctionMinimizer(int maxIterations = 100, double solveResidual = 0.0001, int maxLineSearchIterations = 15)
		: GradientDescentVariableStep(maxIterations, solveResidual, maxLineSearchIterations) {	}

	virtual ~NewtonFunctionMinimizer() {}

protected:
	void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) const override {

		// get hessian
		function->getHessian(x, hessian);

		// add regularization
		VectorXd r(x.size());
		r.setOnes();
		hessian += r.asDiagonal() * reg;

		// get gradient
		VectorXd gradient = function->getGradient(x);

		//dp = Hes^-1 * grad
		Eigen::SimplicialLDLT<SparseMatrixd, Eigen::Lower> solver;
		solver.compute(hessian);
		dx = solver.solve(gradient);
	}

public:
	mutable SparseMatrixd hessian;
	double reg = 1e-4;
};
