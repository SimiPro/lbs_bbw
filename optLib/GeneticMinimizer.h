#pragma once

#include "Minimizer.h"

#include <random>

class GeneticMinimizer : public Minimizer {
public:
    GeneticMinimizer(const VectorXd &upperLimit = VectorXd(), 
        const VectorXd &lowerLimit = VectorXd(), double fBest = HUGE_VAL)
        : searchDomainMax(upperLimit), searchDomainMin(lowerLimit), fBest(fBest) {
        fBest = HUGE_VAL;

        // initial random device and set uniform distribution to [0, 1]
        rng.seed(std::random_device()());
        dist = std::uniform_real_distribution<>(0.0,1.0);
        normal_dist = std::normal_distribution<>(0.0, 0.5);

    }


    bool minimize(const ObjectiveFunction *function, VectorXd &x) const override {
        for (int i = 0; i < iterations; ++i) {

            // for each element of `x`, generate a random variable in the search region
            
            for (int j = 0; j < x.rows(); j++) {
                VectorXd xr = x;
                xr[j] = rand_num(searchDomainMin[j], searchDomainMax[j]);
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

    struct Candidate {
        int k;
        Candidate(int k_): k(k_) {}

        double fitness = 0;

    };

    static bool cmpCand(const Candidate &cnd1, const Candidate &cnd2) {
        return cnd1.fitness > cnd2.fitness;
    }

    void genetic_algorithm(const ObjectiveFunction *function, const int popSize, VectorXd &x) {
        // gene size
        int geneSize = x.rows();
        // initialize population
        vector<Candidate> pop;
        vector<VectorXd> popK;
        for (int i = 0; i < popSize; i++) {
            VectorXd k; random_vector(geneSize, k );
            pop.push_back({i});
            popK.push_back(k);
        }

        
        // assess fitness
        for (Candidate &cand : pop) {
           cand.fitness = function->evaluate(popK[cand.k]);
        }

        // sort according to fit
        sort(pop.begin(), pop.end(), cmpCand);

        // Elitism 20% of fittest go on to live anyway
        vector<Candidate> new_pop;
        vector<VectorXd> new_popK;
        for (int i = 0; i < int((20*popSize)/100); i++) {
            new_popK.push_back(popK[pop[i].k]);
            new_pop.push_back({i});
        }

        // 50% of fittest pop, make children
        int prct50 = 50*popSize / 100;
        while (new_pop.size() < popSize) {
            int indv1 = rand_num(0, prct50);
            int indv2 = rand_num(0, prct50);

            Candidate cnd1 = pop[indv1];
            Candidate cnd2 = pop[indv2];

            VectorXd new_k = popK[cnd2.k];
            // crossover
            line_recomb(popK[cnd1.k], new_k);

            // mutate
            bounded_normal_convolution(new_k);

            new_popK.push_back(new_k);
            new_pop.push_back({int(new_popK.size() -1)});
        }

        pop = new_pop;
        popK = new_popK;


        x = popK[pop[0].k];

    }

    double rand_num(double lower, double upper) const {
        return dist(rng) * (upper - lower) + lower;
    }


    //// MUTATION
    // add random noise to vector
    void bounded_normal_convolution(VectorXd &k) {
        for (int i = 0; i < k.rows(); i++) {
            k[i] = checkDomain(k[i] + normal_dist(generator), i);
        }

    }

    double checkDomain(double val, int i) {
        val = min(val, searchDomainMax[i]);
        return max(val, searchDomainMin[i]);
    }

    //// CROSSOVER
    void line_recomb(const VectorXd &w, VectorXd &v) {
        double alpha = rand_num(0, 1);
        for (int i = 0; i < v.rows(); i++) {
            v[i] = checkDomain(alpha*v[i] + (1- alpha)*w[i], i);
        }
    
    }



    void random_vector(const int n, VectorXd &k) {
        k.resize(n);
        for (int j = 0; j < n; j++) {
            k[j] = rand_num(searchDomainMin[j], searchDomainMax[j]);
        }
    }

public:
    int iterations = 100;
    VectorXd searchDomainMax, searchDomainMin;

    mutable double fBest;
    mutable std::uniform_real_distribution<double> dist;
    mutable std::mt19937 rng;
    mutable std::default_random_engine generator;
    mutable std::normal_distribution<double> normal_dist;
};
