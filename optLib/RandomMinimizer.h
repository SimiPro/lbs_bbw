#pragma once

#include "Minimizer.h"

#include <random>

class RandomMinimizer : public Minimizer
{
public:
	RandomMinimizer(const VectorXd &upperLimit = VectorXd(), 
        const VectorXd &lowerLimit = VectorXd(), int max_iter = 10, double fBest = HUGE_VAL)
		: searchDomainMax(upperLimit), searchDomainMin(lowerLimit), fBest(fBest), iterations(max_iter) {
		fBest = HUGE_VAL;

		// initial random device and set uniform distribution to [0, 1]
		rng.seed(std::random_device()());
		dist = std::uniform_real_distribution<>(0.0,1.0);
	}

	virtual ~RandomMinimizer() {}

	bool minimize(const ObjectiveFunction *function, VectorXd &x) const override {
		for (int i = 0; i < iterations; ++i) {

			// for each element of `x`, generate a random variable in the search region
            
            for (int j = 0; j < x.rows(); j++) {
                VectorXd xr = x;
                xr[j] = dist(rng) * (searchDomainMax[j] - searchDomainMin[j]) + searchDomainMin[j];
                // if function value at new `x` is smaller, let's keep it
                double f = function->evaluate(xr);
                if(f < fBest){
                    x = xr;
                    fBest = f;
                }
            }
		}
		return false;
	}

public:
	int iterations;
	VectorXd searchDomainMax, searchDomainMin;

	mutable double fBest;
	mutable std::uniform_real_distribution<double> dist;
	mutable std::mt19937 rng;
};
