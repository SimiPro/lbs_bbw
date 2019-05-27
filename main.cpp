#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

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

#include "kin.h"
#include "optLib/GradientDescentMinimizer.h"


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

// skeleton deformed
MatrixXd CT;
MatrixXi BET;

// 
RowVector3d moving_point;

RotationList rest_pose;

double bbd = 1.0;
int selected = 0;
int mouse_x = -1, mouse_y = -1; double mouse_z = -1;

int moved = 0;
bool dragging = false;

// else
const RowVector3d sea_green(70./255.,252./255.,167./255.);

int v_spec = -1;

int getSelectedBone() {
    // we convert the selected point index
    // into the bone whichs tip is the selected point
    for (int i = 0; i < BE.rows(); i++) {
        if (C.row(BE(i,1))  == C.row(selected))
            return i;
    }
    return 0;
    throw std::invalid_argument( "The selected point should always be of a joint");
}

void set_color(igl::opengl::glfw::Viewer &viewer) {
  MatrixXd C(0,3);

  int bone = getSelectedBone();
  jet(W.col(bone).eval(),true,C);

  if (v_spec != -1) {
    C.row(v_spec) = RowVector3d(1,0,0);
  }
  viewer.data().set_colors(C);
}


void new_mesh(Viewer &viewer, VectorXd &a) {
    // transformations

    MatrixXd  U, CEmpty;

    EnergyFunction kinematics(C, BE, M, P, CEmpty);
    kinematics.forward2(a, U, CT, true);
    
    // lbs
    MatrixXd UN;
    per_face_normals(U,F,UN);

    // Also deform skeleton edges
    // this basically just applies each transformation in T 
    // to the 2 joint endpoints it concerns in C
    //igl::deform_skeleton(C, BE, T, CT, BET);

    viewer.data().clear();
    viewer.data().set_mesh(U, F);
    viewer.data().set_normals(UN);
}

void optim(Viewer &viewer, VectorXd &a) {
//    int bone = getSelectedBone();

    MatrixXd  U, CTarget;
    EnergyFunction kinematics(C, BE, M, P, CTarget);
    kinematics.forward2(a, U, CTarget, false);
    CTarget.row(selected) = moving_point;

    EnergyFunction kinOptim(C, BE, M, P, CTarget);
    cout << "loss before: " << kinOptim.evaluate(a) << endl;

    GradientDescentVariableStep funcOpt(50, 1e-5, 15);
    funcOpt.minimize(&kinOptim, a);
    
    cout << "loss after: " << kinOptim.evaluate(a) << endl;

    new_mesh(viewer, a);

}



VectorXd a;

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
        if (a.rows() == 0)  {
            a.resize(BE.rows()*3); 
            a.setZero();
        }
        
        a[selected*3 + 2] = igl::PI*0.3*(++moved);
        new_mesh(viewer, a);
        break;
  }

  return true;
}

bool pre_draw(Viewer &viewer) {
    set_color(viewer);

    //clear points and lines
    viewer.data().set_points(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXd::Zero(0,3));
    viewer.data().set_edges(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXi::Zero(0,3), Eigen::MatrixXd::Zero(0,3));

    viewer.data().set_edges(CT, BET, sea_green);
    viewer.data().set_points(CT, RowVector3d(0,1,0.5));

    viewer.data().add_points(moving_point, RowVector3d(1,0,0));
//    MatrixXi edg(1, 2); edg.row(0) = BET.row(getSelectedBone());
 //   viewer.data().add_edges(CT, edg,  RowVector3d(1,0,0));
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

bool mouse_move(Viewer& viewer, int mouse_x_, int mouse_y_) {

    if (dragging) {
        //computeRotation(viewer, mouse_x_, mouse_x,  mouse_y_,  mouse_y);

        Vector3d mouse_now(mouse_x_, viewer.core.viewport(3) - mouse_y_, mouse_z);
        Vector3d mouse_before(mouse_x, viewer.core.viewport(3) - mouse_y, mouse_z);

        Vector3d before, now;
        unproject(mouse_before, viewer.core.view, viewer.core.proj, viewer.core.viewport, before);
        unproject(mouse_now, viewer.core.view, viewer.core.proj, viewer.core.viewport, now);

        moving_point += (now - before).transpose();

        if (a.rows() == 0) {
            a.resize(BE.rows()*3); a.setZero();
        }
        if (dragging)
            optim(viewer, a);  

    }




    mouse_x = mouse_x_;
    mouse_y = mouse_y_;

    return false;
}

bool mouse_up(Viewer& viewer, int button, int modifier) {
    if (a.rows() == 0) {
        a.resize(BE.rows()*3); a.setZero();
    }
    if (dragging && false)
        optim(viewer, a);  
    dragging = false;


}

bool mouse_down(Viewer& viewer, int button, int modifier) {
    selected = 0;

    double x = mouse_x;
    double y = viewer.core.viewport(3) - mouse_y;

    int fid, vi = -1;
    Vector3d baryC;
    if(unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
                              viewer.core.proj, viewer.core.viewport, V, F, fid, baryC)) {

        MatrixXd coord;
        igl::project(C, viewer.core.view, viewer.core.proj,viewer.core.viewport, coord);

        RowVector3d curr_mouse(x, y, 0);
        VectorXd diff = (coord.rowwise() - curr_mouse).rowwise().norm();
        
        diff.minCoeff(&selected);
        mouse_z = C(selected, 2);
        moving_point = CT.row(selected);
        dragging = true;
        return true;
  } else {
    return false;
  }

}

void menu(Viewer &viewer) {
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {
    // Draw parent menu content
    menu.draw_viewer_menu();

    // Add new group
    if (ImGui::CollapsingHeader("Deformation Controls", ImGuiTreeNodeFlags_DefaultOpen)) {

          if (ImGui::Button("Clear Selection", ImVec2(-1,0))) {

          }

          if (ImGui::Button("Apply Selection", ImVec2(-1,0))) {
            
          }

          if (ImGui::Button("Clear Constraints", ImVec2(-1,0))) {
            
          }
    }
  };

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

    CT = C;
    BET = BE;

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;

    //menu(viewer);

    // set callbacks 
    viewer.callback_key_down = &key_down;
    viewer.callback_mouse_down = mouse_down;
    viewer.callback_mouse_move = mouse_move;
    viewer.callback_pre_draw = &pre_draw;
    viewer.callback_mouse_up = &mouse_up;
    // set mesh
    viewer.data().set_mesh(V, F);
    set_color(viewer);
    viewer.data().show_lines = false;
    viewer.data().show_overlay_depth = false;
    viewer.data().line_width = 1;
    viewer.data().point_size = 10;
    viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
    viewer.launch();


}
