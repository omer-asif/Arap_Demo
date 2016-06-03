//#include "tutorial_shared_path.h"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
// HalfedgeGraph adapters for Polyhedron_3
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/boost/graph/properties_Polyhedron_3.h>

#include "Arap_mesh_deformation.h"

#include <fstream>

#include <igl/readOFF.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/viewer/Viewer.h>
#include <nanogui/formhelper.h>
#include <nanogui/screen.h>
#include <iostream>


typedef CGAL::Simple_cartesian<double>                                   Kernel;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Polyhedron;

typedef boost::graph_traits<Polyhedron>::vertex_descriptor    vertex_descriptor;
typedef boost::graph_traits<Polyhedron>::vertex_iterator        vertex_iterator;

typedef Arap_mesh_deformation<Polyhedron> mesh_deformation;
Polyhedron mesh;
mesh_deformation *deform_mesh = NULL;
vertex_iterator vb, ve;
bool setup_mesh_for_deformation(const char* filename) {

	mesh.clear();
	std::ifstream input(filename);
	
	if (!input || !(input >> mesh) || mesh.empty()) {
		std::cerr << "Cannot open  ../shared/bunny.off" << std::endl;
		return 1;
	}

	// Init the indices of the halfedges and the vertices.
	set_halfedgeds_items_id(mesh);

	// Create a deformation object
	if (deform_mesh != NULL) {
		delete deform_mesh;
		deform_mesh = NULL;
	}
	deform_mesh = new mesh_deformation(mesh);

	// Definition of the region of interest (use the whole mesh)
	
	boost::tie(vb, ve) = vertices(mesh);
	deform_mesh->insert_roi_vertices(vb, ve);


}




int main(int argc, char *argv[])
{
  bool boolVariable = true;
  float floatVariable = 0.1f;
  bool done = false, btn_done=false;
  double posX=0.0, posY=0.0, posZ=0.0;

  // Mesh with per-face color
  Eigen::MatrixXd V, C;
  Eigen::MatrixXi F;
  // Load a mesh in OFF format
  igl::readOFF("../shared/bunny.off", V, F);

  // Initialize with white color
  C= Eigen::MatrixXd::Constant(V.rows(),3,1); // Initialize Color matrix with size of Vertices, for per vertex coloring
  igl::viewer::Viewer viewer;


  //Polyhedron mesh;
  std::vector<vertex_descriptor> ctrl_points;
  //vertex_descriptor last_control_vertex;
  int last_vid=-1;
  std::map<int, mesh_deformation::Point > vertex_point_map;
  setup_mesh_for_deformation("../shared/bunny.off");


  viewer.callback_mouse_down = 
    [&](igl::viewer::Viewer& viewer, int, int)->bool
  {
    int fid;
	int vid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(
      Eigen::Vector2f(x,y),
      viewer.core.view * viewer.core.model,
      viewer.core.proj,
      viewer.core.viewport,
      V,
      F,
      fid,
      bc))
    {
		int i;
		bc.maxCoeff(&i);
		vid = F(fid, i); // retrieve the vertex id clicked by using the face_id
		std::cout << "VID: " << vid << std::endl;
		Eigen::Vector3d p = V.row(vid);
		std::cout << "Original Position: X: " << p.x() << " Y: " << p.y() << " Z: " << p.z() << std::endl;
		viewer.data.add_points(V.row(vid),Eigen::RowVector3d(1,0,0)); // Add a RED Point over the vertex that user clicked.
		
      // paint hit red
     // C.row(vid)<<1,0,0; 
      //viewer.data.set_colors(C);
	  posX = p.x();
	  posY = p.y();
	  posZ = p.z();

	  vertex_descriptor last_control_vertex = *CGAL::cpp11::next(vb, vid);
	  last_vid = vid;
	  mesh_deformation::Point point_1;
	  mesh_deformation::Point point_2;
	  deform_mesh->insert_control_vertex(last_control_vertex, point_1);
	  vertex_point_map[vid] = point_1;
	  
	  if (!done) {
		  // Add an additional menu window
		  viewer.ngui->addWindow(Eigen::Vector2i(240, 40), " Mesh Deformation ");

		  // Expose the same variable directly ...
		  viewer.ngui->addVariable("Pos X", posX);
		  viewer.ngui->addVariable("Pos Y", posY);
		  viewer.ngui->addVariable("Pos Z", posZ);
		  // Add a button
		  viewer.ngui->addButton("Set New Pos", [&]() { 
			  std::cout << "NEW Position: X: " << posX << " Y: " << posY << " Z: " << posZ << std::endl; 
			  mesh_deformation::Point p(posX, posY, posZ);
			  vertex_point_map[last_vid] = p;
			  
			  
			  if (!btn_done) {
				  
				  viewer.ngui->addButton("Deform Mesh", [&]() {
					  // Deform Mesh Button Click event

					  if (vertex_point_map.size() > 0) {
						  // The definition of the ROI and the control vertices is done, call preprocess
						  bool is_matrix_factorization_OK = deform_mesh->preprocess();
						  if (!is_matrix_factorization_OK) {
							  std::cerr << "Error in preprocessing, check documentation of preprocess()" << std::endl;
							  return 1;
						  }
						  // Set target positions of control vertices
						  std::map<int, mesh_deformation::Point >::iterator itr = vertex_point_map.begin();
						  for (; itr != vertex_point_map.end(); itr++) {
							  std::cout << "VID GOT :" << itr->first << std::endl;
							  std::cout << "GOT Position: X: " << itr->second.x() << " Y: " << itr->second.y() << " Z: " << itr->second.z() << std::endl;
							  vertex_descriptor vd = *CGAL::cpp11::next(vb, itr->first);
							  deform_mesh->set_target_position(vd, itr->second); 
						  }
						  deform_mesh->deform(10, 0.0);
						  std::ofstream output("deform_1.off");
						  output << mesh;
						  output.close();
						  // Reset everything to start again using the deformed mesh
						  vertex_point_map.clear();
						  last_vid = -1;
						  posX = 0;
						  posY = 0;
						  posZ = 0;
						  igl::readOFF("deform_1.off", V, F); // Load the deformed mesh as current mesh
						  viewer.data.clear();
						  viewer.data.set_mesh(V, F);
						  C = Eigen::MatrixXd::Constant(V.rows(), 3, 1); // reset color to white
						  viewer.data.set_colors(C);
						  viewer.core.align_camera_center(V, F);
						  setup_mesh_for_deformation("deform_1.off"); // Setup CGAL code with new mesh
					  }

				  });
				  btn_done = true;
			  }
			  viewer.screen->performLayout();
		  });

		  // Generate menu
		  viewer.screen->performLayout();
		  done = true;

	  }




      return true;
    }
    return false;
  };
  std::cout<<R"(Usage: [click]  Pick face on shape )";
  // Show mesh
  viewer.data.set_mesh(V, F);
  viewer.data.set_colors(C);
  viewer.core.show_lines = false;
  viewer.launch();
}
