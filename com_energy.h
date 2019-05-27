#include "optLib/ObjectiveFunction.h"
#include <igl/forward_kinematics.h>
#include "mass_props.h"

using namespace Eigen;
using namespace std;
using namespace igl;

typedef vector<Quaterniond, Eigen::aligned_allocator<Quaterniond> > RotationList;


class CoMEnergyFunction : public ObjectiveFunction {
    public:
    const MatrixXd &C;
    const MatrixXi &F;
    const MatrixXi &BE;
    const MatrixXd &M;
    const VectorXi &P;
    const RowVector3d &CoM_target;
    CoMEnergyFunction(const MatrixXd &C_, const MatrixXi &F_, const MatrixXi &BE_, const MatrixXd &M_, 
        const VectorXi &P_, const RowVector3d &CoM_target_) 
            : BE(BE_), M(M_), C(C_), P(P_), CoM_target(CoM_target_), F(F_) {
    }
    virtual ~CoMEnergyFunction(void){ }

    double evaluate(const VectorXd& a) const override {
        MatrixXd U, CBase;        
        forward2(a, U, CBase, true);

        RowVector3d com; setCom(U, com);

        double loss = (com - CoM_target).array().pow(2).sum();

        return loss;
    }

    virtual void addGradientTo(const VectorXd& a, VectorXd& grad) const {
        calc_dEda(a, grad);
    }

    void setCom(const MatrixXd &U, RowVector3d &com) const {
        VectorXd s10;
        props(U, F, 0.1,  s10);
        com = getCoM(s10).transpose();
    }


    void setComDv(const MatrixXd &U, MatrixXd &dCoMdu) const {
        // dCom_du = 3*|V|
        MatrixXd ds_dv;
        props_dv(U, F, 0.1, ds_dv);
        dCoMdu.resize(3, 3*U.rows());

        dCoMdu.row(0) = ds_dv.row(1);
        dCoMdu.row(1) = ds_dv.row(2);
        dCoMdu.row(2) = ds_dv.row(3);
    }


    void calc_dEda(const VectorXd &a, VectorXd &dEda) const {
        MatrixXd  U, CBase;
        forward2(a, U, CBase, true);

        RowVector3d com; setCom(U, com);
        RowVector3d dEdx = 2*(com - CoM_target); // 1 x 3

        MatrixXd dCoM_du; setComDv(U, dCoM_du); // 3 x (3*|V|)

        MatrixXd jakob; // dx(a) / da
        jacobian_finite_diff(a, jakob); // |T| x (3*m)

        MatrixXd du_da = M*jakob; // |V| x (3*m)

        dEda = du_da.transpose() * (dEdx*dCoM_du); // 1 x (3*m)
    }

    // here we copy a because we later want to add and subtract from it 
    void jacobian_finite_diff(VectorXd a, MatrixXd &jakob) const {
        MatrixXd base_T; setT(a, base_T);

        Map<VectorXd> T_base_flat(base_T.data(), base_T.size());

        jakob.resize(base_T.size(), a.rows());

        double EPS = 1e-7;
        for (int i = 0; i < a.rows(); i++) {
            a[i] += EPS;
        
            MatrixXd Td; setT(a, Td);
            Map<VectorXd> Td_flat(Td.data(), Td.size());
            a[i] -= EPS;

            jakob.col(i) = (Td_flat - T_base_flat) / EPS;
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

    void setT(const VectorXd &a, MatrixXd &T) const {
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
        forward_kinematics(C, BE, P, dQ, dT, T);
    }

    // a is a vector that holds all thetas stacked 
    // theta_11 = degree of theta_11 rotation around x axis of bone 1
    // , theta_12, degree of theta_12 rotation around y axis of bone 1
    // theta_13 
    void forward2(const VectorXd &a, MatrixXd &U, MatrixXd &CT_new, const bool calc_u) const {
        MatrixXd T;
        setT(a, T);

        if (calc_u)
            U = M*T;
        myDeform(T, CT_new);

    }


};