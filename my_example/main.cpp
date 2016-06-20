#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/boost/graph/properties_Polyhedron_3.h>

#include "Arap_mesh_deformation.h"

#include <fstream>
#include <iostream>

#include <igl/readOFF.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/viewer/Viewer.h>
#include <nanogui/formhelper.h>
#include <nanogui/screen.h>


#define ORIGINAL_MESH_PATH "../shared/bunny.off"

#define DEFORMED_MESH_PATH "deformed_mesh.off"


typedef CGAL::Simple_cartesian<double>                                   Kernel;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Polyhedron;
typedef Arap_mesh_deformation<Polyhedron> mesh_deformation;
typedef mesh_deformation::vertex_descriptor vertex_descriptor;
typedef mesh_deformation::vertex_iterator vertex_iterator;
Polyhedron mesh;
mesh_deformation *deform_mesh = NULL;
vertex_iterator vb, ve;



// CGAL related mesh setup
bool setup_mesh_for_deformation(const char* filename) {

	mesh.clear();
	std::ifstream input(filename);
	
	if (!input || !(input >> mesh) || mesh.empty()) {
		std::cerr << "Cannot open:  " << filename << std::endl;
		return false;
	}

	// Init the indices of the halfedges and the vertices.
	set_halfedgeds_items_id(mesh);

	// Create a deformation object
	if (deform_mesh != NULL) {
		delete deform_mesh;
		deform_mesh = NULL;
	}
	deform_mesh = new mesh_deformation(mesh);
	boost::tie(vb, ve) = vertices(mesh);
	return true;
}




int main(int argc, char *argv[])
{
  float original_z = 0.0f;
  bool clicked_outside_mesh = true;
  int last_vid = -1, roi_radius=3;

  Eigen::MatrixXd V, C;
  Eigen::MatrixXi F;
  // Load a mesh in OFF format
  igl::readOFF(ORIGINAL_MESH_PATH, V, F);

  // Initialize with white color
  C= Eigen::MatrixXd::Constant(V.rows(),3,1); // Initialize Color matrix with size of Vertices, for per vertex coloring, white by default
  igl::viewer::Viewer viewer;

  //Polyhedron mesh;
  if (!setup_mesh_for_deformation(ORIGINAL_MESH_PATH)) // Setup CGAL related mesh data structures
  {
	  return 1; // error occured, exit the program
  }
  
  // handle mouse up event and do the deformation if mouse down was inside the mesh
  viewer.callback_mouse_up =
	  [&](igl::viewer::Viewer& viewer, int btn, int)->bool 
  {
	  if (last_vid != -1 && !clicked_outside_mesh)
	  {
		  GLint x = viewer.current_mouse_x; 
		  GLfloat y = viewer.core.viewport(3) - viewer.current_mouse_y;
		 		 
		  Eigen::Vector3f win_d(x, y, original_z); // using screen z value from initial mouse click down, current z gives wrong output
		  Eigen::Vector3f ss;
		  Eigen::Matrix4f mvv = viewer.core.view * viewer.core.model;
		  ss = igl::unproject(win_d, mvv, viewer.core.proj, viewer.core.viewport); // convert current mouse pos,from screen to object coordinates
		  
		  
		  // The definition of the ROI and the control vertex is done, call preprocess
		  bool is_matrix_factorization_OK = deform_mesh->preprocess();
		  if (!is_matrix_factorization_OK) {
			  std::cerr << "Error in preprocessing, check documentation of preprocess()" << std::endl;
			  return 1;
		  }
		  // Set target positions of control vertices
		  vertex_descriptor vd = *CGAL::cpp11::next(vb, last_vid);
		  deform_mesh->set_target_position(vd, mesh_deformation::Point(ss.x(), ss.y(), ss.z()));
		  deform_mesh->deform(10, 0.0);
		  std::ofstream output(DEFORMED_MESH_PATH);
		  output << mesh;
		  output.close();
		  // Reset everything to start again using the deformed mesh
		  last_vid = -1;
		  igl::readOFF(DEFORMED_MESH_PATH, V, F); // Load the deformed mesh as current mesh
		  viewer.data.set_mesh(V, F);
		  viewer.data.set_colors(C);
		  setup_mesh_for_deformation(DEFORMED_MESH_PATH); // Setup CGAL code with new mesh

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
		  clicked_outside_mesh = false;
		  int i;
		  bc.maxCoeff(&i);
		  last_vid = F(face_id, i); // retrieve the vertex id clicked by using the retrieved face_id
		  vertex_descriptor control_vertex = *CGAL::cpp11::next(vb, last_vid);
		  std::vector<vertex_descriptor> ring_list;
		  deform_mesh->generate_roi_from_vertex(roi_radius, control_vertex, ring_list); // currently using radius == 3, for region of mesh to affect
		  // color the vertices in roi, red
		  for (int i = 0; i < ring_list.size(); i++)
		  {
			  C.row(ring_list[i]->id()) = Eigen::RowVector3d(1, 0, 0); 
		  }
		  viewer.data.set_colors(C);

		  mesh_deformation::Point point_1;
		  deform_mesh->insert_control_vertex(control_vertex, point_1);
		  return true;
    }
	else 
	{
		clicked_outside_mesh = true;
	}
    return false;
  };

  
  viewer.callback_init = [&](igl::viewer::Viewer& viewer)
  {
	  viewer.ngui->addGroup("Mesh Deformation");
	  viewer.ngui->addVariable("ROI radius", roi_radius); // you can change this from UI to change how much of neighbouring region should be affected.
	  // call to generate menu
	  viewer.screen->performLayout();
	  return false;
  };

  // Show mesh
  viewer.data.set_mesh(V, F);
  viewer.data.set_colors(C);
  viewer.core.show_lines = false;
  viewer.launch();
}
