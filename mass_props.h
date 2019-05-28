#pragma once

using namespace Eigen;

double getMass(const Eigen::VectorXd &s10) {
    return s10[0];
}

Vector3d getCoM(const Eigen::VectorXd &s10) {
    double M = getMass(s10);
    Vector3d result;
    result << s10[1], s10[2], s10[3];
    if (M == 0) return Vector3d(0,0,0);
    return result / M;
}


void props(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const double phi,  Eigen::VectorXd &s10) {
    // triangle vertices counter clockwise: a, b, c
    s10.resize(10); s10.setZero();

    for (int i = 0; i < F.rows(); i++) {
        Vector3d a = V.row(F.row(i)[0]).transpose();
        Vector3d b = V.row(F.row(i)[1]).transpose();
        Vector3d c = V.row(F.row(i)[2]).transpose();

        Vector3d u = (b - a), v = (c - a);
        Vector3d n = u.cross(v);

        Vector3d h1 = a + b + c;
        Vector3d h2 = a.array()*a.array() + b.array()*(a + b).array();
        Vector3d h3 = h2.array() + c.array()*h1.array();
        Vector3d h4 = a.array()*a.array()*a.array() + b.array()*h2.array() + c.array()*h3.array();
        Vector3d h5 = h3.array() + a.array()*(h1.array() + a.array());
        Vector3d h6 = h3.array() + b.array()*(h1.array() + b.array());
        Vector3d h7 = h3.array() + c.array()*(h1.array() + c.array());
        Vector3d a_bar; a_bar << a[1], a[2], a[0];
        Vector3d b_bar; b_bar << b[1], b[2], b[0];
        Vector3d c_bar; c_bar << c[1], c[2], c[0];

        Vector3d h8 = a_bar.array()*h5.array() + b_bar.array()*h6.array() + c_bar.array()*h7.array();
        s10[0] += (n.array()*h1.array())[0];

        Vector3d stemp = n.array()*h3.array();
        s10[1] += stemp[0]; s10[2] += stemp[1]; s10[3] += stemp[2];

        stemp = n.array()*h8.array();
        s10[4] += stemp[0]; s10[5] += stemp[1]; s10[6] += stemp[2];

        stemp = n.array()*h4.array();
        s10[7] += stemp[0]; s10[8] += stemp[1]; s10[9] += stemp[2];
    }

    s10[0] *= 1./6.;
    s10[1] *= 1./24.; s10[2] *= 1./24.; s10[3] *= 1./24.;
    s10[4] *= 1./120.; s10[5] *= 1./120.; s10[6] *= 1./120.;
    s10[7] *= 1./60.; s10[8] *= 1./60.; s10[9] *= 1./60.;

    s10 *= phi;
}

Matrix3d makeSkew_cross(const Vector3d &x) {
    Matrix3d skew; 
    skew << 0, -x[2], x[1],
            x[2], 0, -x[0],
            -x[1], x[0], 0;
    return skew;
}

Matrix3d makeSkew_prod(const Vector3d &x) {
    Matrix3d skew; 
    skew << 0, x[0], 0,
            0, 0, x[1],
            x[2], 0, 0;
    return skew;
}

Matrix3d makeH(const Vector3d &x) {
    Matrix3d H;
    H.col(0) = x;
    H.col(1) = x;
    H.col(2) = x;
    return H;
}

void props_dv(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const double phi, MatrixXd &s_dv) {
    s_dv.resize(10, 3*V.rows()); s_dv.setZero();

    for (int i = 0; i < F.rows(); i++) {
        int ai = F.row(i)[0], bi = F.row(i)[1], ci = F.row(i)[2];
        Vector3d a = V.row(ai).transpose();
        Vector3d b = V.row(bi).transpose();
        Vector3d c = V.row(ci).transpose();

        Vector3d u = (b - a), v = (c - a);
        Vector3d n = u.cross(v);

        Vector3d h1 = a + b + c;
        Vector3d h2 = a.array()*a.array() + b.array()*(a + b).array();
        Vector3d h3 = h2.array() + c.array()*h1.array();
        Vector3d h4 = a.array()*a.array()*a.array() + b.array()*h2.array() + c.array()*h3.array();
        Vector3d h5 = h3.array() + a.array()*(h1.array() + a.array());
        Vector3d h6 = h3.array() + b.array()*(h1.array() + b.array());
        Vector3d h7 = h3.array() + c.array()*(h1.array() + c.array());
        Vector3d a_bar; a_bar << a[1], a[2], a[0];
        Vector3d b_bar; b_bar << b[1], b[2], b[0];
        Vector3d c_bar; c_bar << c[1], c[2], c[0];

        Vector3d h8 = a_bar.array()*h5.array() + b_bar.array()*h6.array() + c_bar.array()*h7.array();


        // here starts real derivative
        Matrix3d dn_da = makeSkew_cross(v) + makeSkew_cross(u).transpose();
        Matrix3d dn_db = makeSkew_cross(v).transpose();
        Matrix3d dn_dc = makeSkew_cross(u);

        Vector3d d3a = 2*a + b + c, d3b = a + 2*b + c, d3c = a + b + 2*c; 
        Vector3d d5a = 6*a + 2*b + 2*c, d6b = 2*a + 6*b + 2*c, d7c = 2*a + 2*b + 6*c; 
        Vector3d d5b = 2*a + 2*b + c;
        Vector3d d6a = d5b;
        Vector3d d5c = 2*a + b + 2*c;
        Vector3d d7a = d5c;
        Vector3d d6c = a + 2*b + 2*c;
        Vector3d d7b = d6c;
        Vector3d d8a = a_bar.array() * d5a.array() + b_bar.array() * d5b.array() + c_bar.array() * d5c.array();
        Vector3d d8b = a_bar.array() * d6a.array() + b_bar.array() * d6b.array() + c_bar.array() * d6c.array();
        Vector3d d8c = a_bar.array() * d7a.array() + b_bar.array() * d7b.array() + c_bar.array() * d7c.array();

        Matrix3d nMatrix = n.asDiagonal();

        // ds1/d{a,b,c}
        RowVector3d e3x; e3x << 1,0,0;
        s_dv.block(0, ai, 1, 3) += e3x * (dn_da*makeH(h1).transpose() + nMatrix);
        s_dv.block(0, bi, 1, 3) += e3x * (dn_db*makeH(h1).transpose() + nMatrix);
        s_dv.block(0, ci, 1, 3) += e3x * (dn_dc*makeH(h1).transpose() + nMatrix);

        //// d[sx, sy, sz] / {a,b,c}
        // d[sx, sy, sz] / a
        Vector3d nd3a = n.array() * d3a.array(); Matrix3d nd3aM = nd3a.asDiagonal();
        s_dv.block(1, ai, 3, 3) += dn_da*makeH(h3).transpose() + nd3aM; 
        // d[sx, sy, sz] / b
        Vector3d nd3b = n.array() * d3b.array(); Matrix3d nd3bM = nd3b.asDiagonal();
        s_dv.block(1, bi, 3, 3) += dn_db*makeH(h3).transpose() + nd3bM; 
        // d[sx, sy, sz] / c
        Vector3d nd3c = n.array() * d3c.array(); Matrix3d nd3cM = nd3c.asDiagonal();
        s_dv.block(1, ci, 3, 3) += dn_dc*makeH(h3).transpose() + nd3cM; 


        // d[sy, syz, sz] / a
        Vector3d nd8a = n.array() * d8a.array(); Matrix3d nd8aM = nd8a.asDiagonal();
        // TODO HBS: n.array() * h5.array() probably false? ask prof. bächler
        // but what alternatives would generate a Vector3d ? 
        s_dv.block(4, ai, 3, 3) += dn_da*makeH(h8).transpose() + nd8aM + makeSkew_prod(n.array() * h5.array()); 

        // d[sy, syz, sz] / b
        Vector3d nd8b = n.array() * d8b.array(); Matrix3d nd8bM = nd8b.asDiagonal();
        // TODO HBS: n.array() * h6.array() probably false? ask prof. bächler
        // but what alternatives would generate a Vector3d ? 
        s_dv.block(4, bi, 3, 3) += dn_db*makeH(h8).transpose() + nd8bM + makeSkew_prod(n.array() * h6.array()); 

        // d[sy, syz, sz] / c
        Vector3d nd8c = n.array() * d8c.array(); Matrix3d nd8cM = nd8c.asDiagonal();
        // TODO HBS: n.array() * h7.array() probably false? ask prof. bächler
        // but what alternatives would generate a Vector3d ? 
        s_dv.block(4, ci, 3, 3) += dn_dc*makeH(h8).transpose() + nd8cM + makeSkew_prod(n.array() * h7.array()); 
        
        // d[sx², sy², sz²] / d{a,b,c}
        // a
        Vector3d nh5 = n.array() * h5.array(); Matrix3d nh5M = nh5.asDiagonal();
        s_dv.block(7, ai, 3, 3) += dn_da * makeH(h4).transpose() + nh5M;

        // b
        Vector3d nh6 = n.array() * h6.array(); Matrix3d nh6M = nh6.asDiagonal();
        s_dv.block(7, bi, 3, 3) += dn_db * makeH(h4).transpose() + nh6M;

        // c
        Vector3d nh7 = n.array() * h7.array(); Matrix3d nh7M = nh7.asDiagonal();
        s_dv.block(7, ci, 3, 3) += dn_dc * makeH(h4).transpose() + nh7M;
    }

    s_dv.row(0) *= 1./6;
    s_dv.row(1) *= 1./24; s_dv.row(2) *= 1./24; s_dv.row(3) *= 1./24;
    s_dv.row(4) *= 1./120; s_dv.row(5) *= 1./120; s_dv.row(6) *= 1./120;
    s_dv.row(7) *= 1./60; s_dv.row(8) *= 1./60; s_dv.row(9) *= 1./60;

    s_dv *= phi;

}