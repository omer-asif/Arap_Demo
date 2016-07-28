
#include "Arap_mesh_deformation.h"

#include <fstream>
#include <iostream>
#include <chrono>

#include <igl/readOFF.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/viewer/Viewer.h>
#include <nanogui/formhelper.h>
#include <nanogui/screen.h>


#define ORIGINAL_MESH_PATH "../shared/bunny.off"

#define DEFORMED_MESH_PATH "deformed_mesh.off"

typedef Arap_mesh_deformation mesh_deformation;

mesh_deformation *deform_mesh = NULL;
Eigen::MatrixXd V, V_temp, C;
Eigen::MatrixXi F, F_temp;
igl::viewer::Viewer viewer;
int last_vid = -1;
bool flag = false;

bool setup_mesh_for_deformation(const char* filename, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {

	// Create a deformation object
	if (deform_mesh != NULL) {
		delete deform_mesh;
		deform_mesh = NULL;
	}

	deform_mesh = new mesh_deformation(V, F);
	deform_mesh->insert_roi_vertices(0, V.rows()-1); // use whole mesh as ROI
	return true;
}

void update_ui()
{
	viewer.data.set_vertices(V_temp);
	viewer.data.set_colors(C);
}

void perform_deformation()
{
	Timer tmr;
	flag = true;
	std::vector<std::thread> lst;
	for (int l = 0; l < 20; l++)
	{
		deform_mesh->deform(1, 0.0);
		if(l%2 == 0 && l!=0)
			lst.push_back(std::thread(update_ui));
	}
	igl::writeOFF(DEFORMED_MESH_PATH, V, F);

	// Reset everything to start again using the deformed mesh
	last_vid = -1;
	V = V_temp;
	viewer.data.set_vertices(V);
	viewer.data.set_colors(C);
	deform_mesh->reset();
	double t = tmr.elapsed();
	for (auto&th : lst)
	{
		th.join();
	}
	std::cout << "Total Elapsed Time: " << t << std::endl; // Time Elapsed
	flag = false;
}


int main(int argc, char *argv[])
{
  float original_z = 0.0f;
  bool clicked_outside_mesh = true;
  
  std::thread *deformation_thread = NULL;
  std::map<int, Eigen::Vector3d> id_point_map;

  

  igl::readOFF(ORIGINAL_MESH_PATH, V, F);

  V_temp = V;
  F_temp = F;

  // Initialize with white color
  C= Eigen::MatrixXd::Constant(V.rows(),3,1); // Initialize Color matrix with size of Vertices, for per vertex coloring, white by default
  

  //Polyhedron mesh;
  if (!setup_mesh_for_deformation(ORIGINAL_MESH_PATH, V_temp, F_temp))
  {
	  return 1; // error occured, exit the program
  }
  
  // handle mouse up event and do the deformation if mouse down was inside the mesh
  viewer.callback_mouse_up =
	  [&](igl::viewer::Viewer& viewer, int btn, int)->bool 
  {
	  if (flag) {
		  return false;
	  }
	  if (last_vid != -1 && !clicked_outside_mesh)
	  {
		  GLint x = viewer.current_mouse_x; 
		  GLfloat y = viewer.core.viewport(3) - viewer.current_mouse_y;
		 		 
		  Eigen::Vector3f win_d(x, y, original_z); // using screen z value from initial mouse click down, current z gives wrong output
		  Eigen::Vector3f ss;
		  Eigen::Matrix4f mvv = viewer.core.view * viewer.core.model;
		  ss = igl::unproject(win_d, mvv, viewer.core.proj, viewer.core.viewport); // convert current mouse pos,from screen to object coordinates
		  
		  
		  // Set target positions of control vertices

		  id_point_map[last_vid] = Eigen::Vector3d(ss.x(), ss.y(), ss.z());
		  
		 // Have to set target positions for all ctrls coz during reset we lost these positions 
		 for (std::map<int, Eigen::Vector3d>::iterator ii = id_point_map.begin(); ii != id_point_map.end(); ++ii)
		  {
			 deform_mesh->set_target_position(ii->first,ii->second);
		  }
		 
		  Timer tmr;
		  deformation_thread = new std::thread(perform_deformation);

		  return true;
	  }
	  return false;
  };

// Setup Click event handler to catch the mouse click on the mesh.
  viewer.callback_mouse_down = 
    [&](igl::viewer::Viewer& viewer, int btn, int)->bool
  {
	
	if (flag) {
		return false;
	}

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
		  clicked_outside_mesh = false;
		  int i;
		  bc.maxCoeff(&i);
		  last_vid = F(face_id, i); // retrieve the vertex id clicked by using the retrieved face_id
		  Eigen::Vector3d point_1;

		  //control_vertex = *CGAL::cpp11::next(vb, last_vid);
		  deform_mesh->insert_control_vertex(last_vid, point_1);

		  C.row(last_vid) = Eigen::RowVector3d(1, 0, 0);
		  viewer.data.set_colors(C);
		  // The definition of the ROI and the control vertex is done, call preprocess
		  bool is_matrix_factorization_OK = deform_mesh->preprocess();
		  if (!is_matrix_factorization_OK) {
			  std::cerr << "Error in preprocessing, check documentation of preprocess()" << std::endl;
			  return 0;
		  }

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
