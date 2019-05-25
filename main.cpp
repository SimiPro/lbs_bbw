#include <igl/opengl/glfw/Viewer.h>

#include <igl/read_triangle_mesh.h>
#include <igl/readTGF.h>
#include <igl/bone_parents.h>
#include <igl/directed_edge_parents.h>
#include <igl/slice.h>
#include <igl/sortrows.h>
#include "robust_bbw.h"
#include <igl/lbs_matrix.h>
#include <igl/jet.h>
#include <igl/normalize_row_sums.h>
#include <igl/setdiff.h>
#include <igl/forward_kinematics.h>
#include <igl/project.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/quat_conjugate.h>
#include <igl/trackball.h>
#include <igl/directed_edge_orientations.h>
#include <igl/deform_skeleton.h>

using namespace Eigen;
using namespace std;
using namespace igl;

typedef vector<Quaterniond, Eigen::aligned_allocator<Quaterniond> > RotationList;

typedef igl::opengl::glfw::Viewer Viewer;

// mesh
MatrixXi F;
MatrixXd V, N, W, M;

// skeleton
MatrixXd C; // list of control an joint positions
MatrixXi BE; // list of bone edges indexing C
VectorXi P; // list of point handles indexing C

RotationList rest_pose;

double bbd = 1.0;
int selected = 0;
int down_mouse_x = -1, down_mouse_y = -1;
bool picked_boned = false;

int moved = 0;


// else
const RowVector3d sea_green(70./255.,252./255.,167./255.);

int v_spec = -1;

void set_color(igl::opengl::glfw::Viewer &viewer) {
  MatrixXd C;
  jet(W.col(selected).eval(),true,C);

  if (v_spec != -1) {
    C.row(v_spec) = RowVector3d(1,0,0);
  }
  viewer.data().set_colors(C);
}

void new_mesh(Viewer &viewer) {
    // transformations
    RotationList dQ(BE.rows(), Quaterniond::Identity());   //   dQ  #BE list of relative rotations
    vector<Vector3d> dT(BE.rows(), Vector3d(0,0,0));   // dT  #BE list of relative translations


    const Quaterniond twist(AngleAxisd(igl::PI*0.3*(++moved), Vector3d(0,1,0)));
    const Quaterniond bend(AngleAxisd(-igl::PI*0.7,Vector3d(0,0,1)));

    dQ[selected] = rest_pose[selected]*twist*rest_pose[selected].conjugate();
    //dQ[3] = rest_pose[2]*bend*rest_pose[2].conjugate();
    //dQ[2] = rest_pose[2]*twist*rest_pose[2].conjugate();
    //dQ[3] = rest_pose[3]*twist*rest_pose[3].conjugate();

    MatrixXd T;
    forward_kinematics(C, BE, P, dQ, dT, T);

    // lbs
    MatrixXd U = M*T;
    MatrixXd UN;
    per_face_normals(U,F,UN);

    // Also deform skeleton edges
    MatrixXd CT;
    MatrixXi BET;
    igl::deform_skeleton(C, BE, T, CT, BET);

    viewer.data().clear();
    viewer.data().set_mesh(U, F);
    viewer.data().set_normals(UN);
    viewer.data().set_edges(CT, BET, sea_green);
    viewer.data().set_points(CT, RowVector3d(0,1,0.5));


}


bool key_down(Viewer &viewer, unsigned char key, int mods) {
  switch(key) {
    case '.':
      selected++;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      set_color(viewer);
      moved = 0;
      break;
    case ',':
      selected--;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      set_color(viewer);
      moved = 0;
      break;
    case ' ':
        new_mesh(viewer);
        break;
  }

  return true;
}

bool pre_draw(Viewer &viewer) {

    set_color(viewer);
}



Quaterniond computeRotation(Viewer &viewer, int mouse_x, int from_x, int mouse_y, int from_y) {

    Matrix4f modelview = viewer.core.view;// * viewer.data().model;

    //initialize a trackball around the handle that is being rotated
    //the trackball has (approximately) width w and height h
    double w = viewer.core.viewport[2]/8;
    double h = viewer.core.viewport[3]/8;

    Vector4f rotation;
    rotation.setZero();
    rotation[3] = 1.;
    //project the given point on the handle(centroid)
    RowVector3d selected_joint = C.row(selected);
    Vector3f proj = igl::project(selected_joint.transpose().cast<float>().eval(),
                             modelview, viewer.core.proj, viewer.core.viewport);
    proj[1] = viewer.core.viewport[3] - proj[1];

    //express the mouse points w.r.t the centroid
    from_x -= proj[0]; mouse_x -= proj[0];
    from_y -= proj[1]; mouse_y -= proj[1];

    //shift so that the range is from 0-w and 0-h respectively (similarly to a standard viewport)
    from_x += w/2; mouse_x += w/2;
    from_y += h/2; mouse_y += h/2;

    //get rotation from trackball
    Vector4f drot = viewer.core.trackball_angle.coeffs();
    Vector4f drot_conj;
    igl::quat_conjugate(drot.data(), drot_conj.data());
    igl::trackball(w, h, float(1.), rotation.data(), from_x, from_y, mouse_x, mouse_y, rotation.data());

    
    Quaterniond q(rotation[3], rotation[0], rotation[1], rotation[2]);

    //cout << rot << endl;
    std::cout << "This quaternion consists of a scalar " << q.w() << " and a vector " << std::endl << q.vec() << std::endl;


    return q;
}

bool mouse_move(Viewer& viewer, int mouse_x, int mouse_y) {
    if (picked_boned) {
        computeRotation(viewer, mouse_x, down_mouse_x,  mouse_y,  down_mouse_y);
    }

    return false;
}

bool mouse_up(Viewer& viewer, int button, int modifier) {
    picked_boned = false;
}

bool mouse_down(Viewer& viewer, int button, int modifier) {
    down_mouse_x = viewer.current_mouse_x;
    down_mouse_y = viewer.current_mouse_y;

    selected = 0;
    picked_boned = false;

    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;

    int fid, vi = -1;
    Vector3d baryC;
    if(unproject_onto_mesh(Eigen::Vector2f(down_mouse_x,y), viewer.core.view,
                              viewer.core.proj, viewer.core.viewport, V, F, fid, baryC)) {

        MatrixXd coord;
        igl::project(C, viewer.core.view, viewer.core.proj,viewer.core.viewport, coord);

        coord.col(2).setZero();



        cout << "x: " << x << " y: " << y << endl;
        cout << "coords: " << endl;
        cout << coord << endl;

        MatrixXd diff = coord.rowwise() - RowVector3d(x, y, 0);
        cout << diff << endl;

        (diff.rowwise().squaredNorm().minCoeff(&selected));

        picked_boned = true;

        return true;
  } else {
    return false;
  }

}


int main(int argc, char *argv[]) {
    string filename = argv[1];
    string skeleton_filename = argv[2];

    // read mesh
    string dir,_1,_2, name; // dir/name.tgf , dir/name.obj
    read_triangle_mesh(filename, V, F, dir,_1,_2,name);
    string output_weights_filename = dir + "/" + name + "-weights.dmat";

    // read skeleton
    
    MatrixXd Ctemp; 
    MatrixXi CE, PE, E, BEtemp; // CE = list of cage edges indexing P
    readTGF(skeleton_filename, Ctemp, E, P, BEtemp, CE, PE);

    // support legacy format
    if(E.rows() > 0 && (BEtemp.rows() == 0 && CE.rows() == 0)){
        cout<<"legacy"<<endl;
        // legacy format: all edges are bones, any points not touched by bones
        // are points
        BEtemp = E;
        VectorXi _;
        igl::setdiff(igl::LinSpaced<VectorXi>(C.rows(),0,C.rows()-1).eval(),BEtemp,P,_);
    }

    // Create small points for each point constraint
    BE.resize(P.rows()+BEtemp.rows(),2);
    BE.block(0,0,P.rows(),1) = P;
    BE.block(0,1,P.rows(),1) = 
    Ctemp.rows() + LinSpaced<VectorXi>(P.rows(),0,P.rows()-1).array();
    MatrixXi SBEtemp;
    
    VectorXi I;
    sortrows(BEtemp,true,SBEtemp,I);
    
    BE.block(P.rows(), 0, BEtemp.rows(), 2) = SBEtemp;
    MatrixXd CP;
    igl::slice(Ctemp, P, 1, CP);
    CP.col(2).array() += bbd*0.0001;
    C.resize(Ctemp.rows()+CP.rows(),3);
    C.block(0,0,Ctemp.rows(),3) = Ctemp;
    C.block(Ctemp.rows(),0,CP.rows(),3) = CP;

    // calculate weights 
    bone_parents(BE, P);
    robust_bbw(V, F, C, BE, W);

    //normalize W
    igl::normalize_row_sums(W,W);

    lbs_matrix(V,W,M);


    igl::directed_edge_orientations(C,BE,rest_pose);

    cout << C << endl;
    cout << BE << endl;


    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;

    // set callbacks 
    viewer.callback_key_down = &key_down;
    viewer.callback_mouse_down = mouse_down;
    viewer.callback_mouse_move = mouse_move;
    viewer.callback_pre_draw = &pre_draw;
    viewer.callback_mouse_up = &mouse_up;
    // set mesh
    viewer.data().set_mesh(V, F);
    set_color(viewer);
    viewer.data().set_edges(C, BE, sea_green);
    viewer.data().set_points(C, RowVector3d(0,1,0.5));
    viewer.data().show_lines = false;
    viewer.data().show_overlay_depth = false;
    viewer.data().line_width = 1;
    viewer.data().point_size = 10;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();


}
