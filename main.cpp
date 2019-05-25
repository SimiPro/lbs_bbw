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

using namespace Eigen;
using namespace std;
using namespace igl;

// mesh
MatrixXi F;
MatrixXd V, N, W, M;

// skeleton
MatrixXd C; // list of control an joint positions
MatrixXi BE; // list of bone edges indexing C
VectorXi P; // list of point handles indexing C

double bbd = 1.0;
int selected = 0;

void set_color(igl::opengl::glfw::Viewer &viewer) {
  MatrixXd C;
  jet(W.col(selected).eval(),true,C);
  viewer.data().set_colors(C);
}


bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods) {
  switch(key) {
    case '.':
      selected++;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      set_color(viewer);
      break;
    case ',':
      selected--;
      selected = std::min(std::max(selected,0),(int)W.cols()-1);
      set_color(viewer);
      break;
  }
  return true;
}

bool pre_draw(igl::opengl::glfw::Viewer & viewer) {
    
}



// else
const RowVector3d sea_green(70./255.,252./255.,167./255.);

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

    
    cout << C << endl;
    cout << BE << endl;

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;

    // set callbacks 
    viewer.callback_key_down = &key_down;

    // set mesh
    viewer.data().set_mesh(V, F);
    set_color(viewer);
    viewer.data().set_edges(C, BE, sea_green);
    viewer.data().show_lines = false;
    viewer.data().show_overlay_depth = false;
    viewer.data().line_width = 1;
    viewer.launch();
}
