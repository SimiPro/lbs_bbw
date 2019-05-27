#pragma once

#include <vector>
#include <Eigen/Sparse>
using Eigen::VectorXd;
using Eigen::Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrixd;

class ObjectiveFunction{
public:
	// this should always return the current value of the objective function
	virtual double evaluate(const VectorXd& x) const = 0;

	VectorXd getGradient(const VectorXd &x) const {
		VectorXd gradient(x.size());
		gradient.setZero();
		addGradientTo(x, gradient);
		return gradient;
	}

	virtual void getHessian(const VectorXd &x, SparseMatrixd &hessian) const {
		hessian.resize(x.size(), x.size());
		hessian.setZero();
		std::vector<Triplet<double>> triplets;
		triplets.clear();
		addHessianEntriesTo(x, triplets);
		hessian.setFromTriplets(triplets.begin(), triplets.end());
	}

	virtual void addGradientTo(const VectorXd& x, VectorXd& grad) const {
		addFiniteDifferenceGradientTo(x, grad);
	}

	virtual void addHessianEntriesTo(const VectorXd& x, std::vector<Triplet<double>>& hessianEntries) const {
		addFiniteDifferenceHessianEntriesTo(x, hessianEntries);
	}

	void addFiniteDifferenceGradientTo(const VectorXd& x, VectorXd& grad) const {
		const double h = 1e-8;
		for (int i = 0; i < x.size(); ++i) {
			VectorXd dx(x.size());
			dx.setZero();
			dx[i] = h;
			grad[i] += (evaluate(x + dx) - evaluate(x - dx)) / (2.*h);
		}
	}

	void addFiniteDifferenceHessianEntriesTo(const VectorXd& x, std::vector<Triplet<double>>& hessianEntries) const {
		const double h = 1e-8;
		for (int i = 0; i < x.size(); ++i) {
			VectorXd dx(x.size());
			dx.setZero();
			dx[i] = h;

			VectorXd gp(x.size()), gm(x.size());
			gp.setZero(); gm.setZero();
			addGradientTo(x + dx, gp);
			addGradientTo(x - dx, gm);

			VectorXd hess = (gp - gm) / (2.*h);
			for (int j = i; j < x.size(); ++j) {
				if(abs(hess[j]) > 1e-12 || i == j)
					hessianEntries.push_back(Triplet<double>(i, j, hess[j]));
			}
		}
	}
};

