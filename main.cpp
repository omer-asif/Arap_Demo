
#include "Arap_mesh_deformation.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <igl/readOFF.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/viewer/Viewer.h>

#define ORIGINAL_MESH_PATH "../shared/bunny.off"

typedef Arap_mesh_deformation mesh_deformation;

std::mutex mtex;
mesh_deformation *deform_mesh = NULL;
Eigen::MatrixXd V, V_temp, C;
Eigen::MatrixXi F, F_temp;
igl::viewer::Viewer viewer;
int last_vid = -1;
bool flag = false;
bool first_time = true;
std::future<void> future;

bool setup_mesh_for_deformation(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
	// Create a deformation object
	if (deform_mesh != NULL) {
		delete deform_mesh;
		deform_mesh = NULL;
	}
	deform_mesh = new mesh_deformation(V, F);
	return true;
}

void perform_deformation() {
	bool stop_loop = false;
	mtex.lock();
	flag = true;
	mtex.unlock();
	while (!stop_loop)
	{
		deform_mesh->deform(1, 0.0);
		V = V_temp;
		viewer.data.set_vertices(V);
		if (flag == false)
			stop_loop = true;
	}
}

void stop_thread() {
	mtex.lock();
	flag = false;
	mtex.unlock();
	last_vid = -1;
	V = V_temp;
	viewer.data.set_vertices(V);

}

int main(int argc, char *argv[]) {
  float original_z = 0.0f;
  bool clicked_outside_mesh = true;
  std::thread *deformation_thread = NULL;
  std::map<int, Eigen::Vector3d> id_point_map;
  igl::readOFF(ORIGINAL_MESH_PATH, V, F);
  V_temp = V;
  F_temp = F;
  // Initialize with white color
  C= Eigen::MatrixXd::Constant(V.rows(),3,1);
  if (!setup_mesh_for_deformation(V_temp, F_temp))
  {
	  return 1; // error occured, exit the program
  }
  // handle mouse up event and do the deformation if mouse down was inside the mesh
  viewer.callback_mouse_up =
	  [&](igl::viewer::Viewer& viewer, int btn, int)->bool
  {
	  if (last_vid != -1 && !clicked_outside_mesh) {
		GLint x = viewer.current_mouse_x;
		GLfloat y = viewer.core.viewport(3) - viewer.current_mouse_y;
		Eigen::Vector3f win_d(x, y, original_z);
		Eigen::Vector3f ss;
		Eigen::Matrix4f mvv = viewer.core.view * viewer.core.model;
		ss = igl::unproject(win_d, mvv, viewer.core.proj, viewer.core.viewport);
		id_point_map[last_vid] = Eigen::Vector3d(ss.x(), ss.y(), ss.z());
		if (!first_time) {
			future.get();
		}
		deform_mesh->insert_control_vertex(last_vid);
		// The definition of the control vertex is done, call preprocess
		bool is_matrix_factorization_OK = deform_mesh->preprocess();
		if (!is_matrix_factorization_OK) {
			std::cerr << "Error in preprocessing" << std::endl;
			return 0;
		}
		deform_mesh->set_target_position(last_vid, id_point_map[last_vid]);
		future = std::async(std::launch::async, &perform_deformation);
		first_time = false;
		return true;
	  }
	  return false;
  };
// Setup Click event handler to catch the mouse click on the mesh.
  viewer.callback_mouse_down =
    [&](igl::viewer::Viewer& viewer, int btn, int)->bool
  {
	int face_id;
    Eigen::Vector3f bc;
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
	GLfloat zz;
	glReadPixels(x, int(y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &zz); // read the current z value, as mouse only returns x,y according to screen
	original_z = zz;
    if(igl::unproject_onto_mesh( // function to check if user clicked inside the mesh
      Eigen::Vector2f(x,y),
      viewer.core.view * viewer.core.model,
      viewer.core.proj,
      viewer.core.viewport,
      V,
      F,
      face_id,
      bc))
    {
		stop_thread();
		clicked_outside_mesh = false;
		int i;
		bc.maxCoeff(&i);
		last_vid = F(face_id, i); // retrieve the vertex id clicked, by using the retrieved face_id
		C.row(last_vid) = Eigen::RowVector3d(1, 0, 0);
		viewer.data.set_colors(C);
		return true;
    }
	else
	{
		clicked_outside_mesh = true;
	}
    return false;
  };
  // Show mesh
  viewer.data.set_mesh(V, F);
  viewer.data.set_colors(C);
  viewer.core.show_lines = false;
  viewer.launch();
}
