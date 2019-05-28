#pragma once

#include "optLib/ObjectiveFunction.h"
#include <igl/forward_kinematics.h>
#include "mass_props.h"
#include "kin.h"
#include "com_energy.h"

using namespace Eigen;
using namespace std;
using namespace igl;

typedef vector<Quaterniond, Eigen::aligned_allocator<Quaterniond> > RotationList;


class BossEnergyFunction : public ObjectiveFunction {
    public:
    CoMEnergyFunction comOptim;
    EnergyFunction kinOptim;
    double lambda;

    BossEnergyFunction(const MatrixXd &C, const MatrixXi &F, const MatrixXi &BE, const MatrixXd &M, 
        const VectorXi &P, const RowVector3d &CoM_target, const MatrixXd &C_target): 
            comOptim(C, F, BE, M, P, CoM_target), kinOptim(C, BE, M, P, C_target), lambda(1.0) {
    }
    virtual ~BossEnergyFunction(void){ }

    double evaluate(const VectorXd& a) const override {
        return comOptim.evaluate(a) + lambda*kinOptim.evaluate(a);
    }

    virtual void addGradientTo(const VectorXd& a, VectorXd& grad) const {
        comOptim.addGradientTo(a, grad);

        VectorXd grad1 = grad;
        grad.setZero();

        kinOptim.addGradientTo(a, grad);
        grad += lambda*grad1;
    }
};