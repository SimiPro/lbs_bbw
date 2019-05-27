#include "optLib/ObjectiveFunction.h"
#include <igl/forward_kinematics.h>

using namespace Eigen;
using namespace std;
using namespace igl;

typedef vector<Quaterniond, Eigen::aligned_allocator<Quaterniond> > RotationList;


class EnergyFunction : public ObjectiveFunction {
    public:
    const MatrixXd &C;
    const MatrixXi &BE;
    const MatrixXd &M;
    const VectorXi &P;
    const MatrixXd &C_target;
    EnergyFunction(const MatrixXd &C_, const MatrixXi &BE_, const MatrixXd &M_, 
        const VectorXi &P_, const MatrixXd &C_target_) : BE(BE_), M(M_), C(C_), P(P_), C_target(C_target_) {
    }
    virtual ~EnergyFunction(void){ }

    double evaluate(const VectorXd& a) const override {
        MatrixXd U, CBase;        
        forward2(a, U, CBase, false);
        double loss = (CBase - C_target).array().pow(2).sum();
        return loss;
    }

    virtual void addGradientTo(const VectorXd& a, VectorXd& grad) const {
        calc_dEda(a, grad);
    }


    void calc_dEda(const VectorXd &a, VectorXd &dEda) const {
        MatrixXd  U, CBase;
        forward2(a, U, CBase, false);

        MatrixXd dEdx = 2*(CBase - C_target); // m x 3

        int n = C.rows(), m = BE.rows();

        VectorXd dEdx_flat(n*3);
        dEdx_flat.segment(0, n) = dEdx.col(0);
        dEdx_flat.segment(n, n) = dEdx.col(1);
        dEdx_flat.segment(2*n, n) = dEdx.col(2);

        MatrixXd jakob; // dx(a)/da
        jacobian_finite_diff(a, jakob);

        dEda = jakob.transpose()*dEdx_flat;
    }

    // here we copy a because we later want to add and subtract from it 
    void jacobian_finite_diff(VectorXd a, MatrixXd &jakob) const {
        int n = C_target.rows(), m = BE.rows();
        jakob.resize(3*n, 3*m);


        MatrixXd  U, CBase;
        forward2(a, U, CBase, false);

        double EPS = 1e-7;
        for (int i = 0; i < 3*m; i++) {
            a[i] += EPS;
        
            MatrixXd CJ;
            forward2(a, U, CJ, false);

            a[i] -= EPS;

            jakob.block(0, i, n, 1) = (CJ.col(0) - CBase.col(0)) / EPS;
            jakob.block(n, i, n, 1) = (CJ.col(1) - CBase.col(1)) / EPS;
            jakob.block(2*n, i, n, 1) = (CJ.col(2) - CBase.col(2)) / EPS;

        }
        
    }



    void myDeform(const MatrixXd & T,   MatrixXd & CT) const {

        CT.resize(C.rows(), C.cols());
        // only transform each point once
        // since we are kin forward we should only have 
        // to have once
        for(int e = 0; e < BE.rows(); e++) {
            Matrix4d t;
            t << T.block(e*4,0,4,3).transpose(), 0,0,0,0;
            Affine3d a;
            a.matrix() = t;
            Vector3d c0 = C.row(BE(e, 0));
            Vector3d c1 = C.row(BE(e, 1));
            CT.row(BE(e, 0)) =   a * c0;
            CT.row(BE(e, 1)) =   a * c1;
        }
    }

    // a is a vector that holds all thetas stacked 
    // theta_11 = degree of theta_11 rotation around x axis of bone 1
    // , theta_12, degree of theta_12 rotation around y axis of bone 1
    // theta_13 
    void forward2(const VectorXd &a, MatrixXd &U, MatrixXd &CT_new, const bool calc_u) const {
        int m = BE.rows();
        assert(a.size() == m*3);

        RotationList dQ(m, Quaterniond::Identity()); //   dQ  #BE list of relative rotations
        for (int i = 0; i < m; i++) {
            const Quaterniond rotX(AngleAxisd(a[3*i + 0], Vector3d(1, 0, 0)));
            const Quaterniond rotY(AngleAxisd(a[3*i + 1], Vector3d(0, 1, 0)));
            const Quaterniond rotZ(AngleAxisd(a[3*i + 2], Vector3d(0, 0, 1)));
            dQ[i] = rotX*rotY*rotZ;
            //dQ[i] = rotX;
        }

        vector<Vector3d> dT(m, Vector3d(0,0,0));   // dT  #BE list of relative translations

        MatrixXd T;
        forward_kinematics(C, BE, P, dQ, dT, T);
        if (calc_u)
            U = M*T;
        myDeform(T, CT_new);

    }


};