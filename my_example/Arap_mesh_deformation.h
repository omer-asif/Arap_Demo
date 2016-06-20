#ifndef ARAP_MESH_DEFORMATION_H
#define ARAP_MESH_DEFORMATION_H

#include <CGAL/config.h>
#include <CGAL/Default.h>
#include <CGAL/tuple.h>
#include <CGAL/Simple_cartesian.h>

#include <vector>
#include <list>
#include <utility>
#include <limits>
#include <boost/foreach.hpp>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Kernel/global_functions_3.h>
#include <CGAL/property_map.h>
#include <CGAL/Eigen_solver_traits.h>
#include <Eigen/Eigen>
#include <Eigen/SVD>


template < class Triangle_mesh >
class Arap_mesh_deformation
{

public:

	typedef typename CGAL::Eigen_solver_traits< Eigen::SparseLU< CGAL::Eigen_sparse_matrix<double>::EigenType, Eigen::COLAMDOrdering<int> > > Sparse_linear_solver;
	typedef typename boost::property_map<Triangle_mesh, CGAL::vertex_point_t>::type Vertex_point_map;
	typedef typename boost::graph_traits<Triangle_mesh>::vertex_descriptor vertex_descriptor;
	typedef typename boost::graph_traits<Triangle_mesh>::halfedge_descriptor halfedge_descriptor;
	typedef typename boost::property_traits<Vertex_point_map>::value_type Point;
	typedef typename boost::graph_traits<Triangle_mesh>::vertex_iterator vertex_iterator;
	typedef typename boost::graph_traits<Triangle_mesh>::halfedge_iterator halfedge_iterator;
	typedef typename boost::graph_traits<Triangle_mesh>::in_edge_iterator in_edge_iterator;
	typedef typename boost::graph_traits<Triangle_mesh>::out_edge_iterator out_edge_iterator;
	typedef typename Eigen::Matrix3d CR_matrix;
	typedef typename Eigen::Vector3d CR_vector;

private:

	Triangle_mesh& triangle_mesh;

	std::vector<Point> original;
	std::vector<Point> solution;

	std::vector<vertex_descriptor> roi; // region of interest
	std::vector<vertex_descriptor> ros; // region of solution

	std::vector<std::size_t> ros_id_map;
	std::vector<bool>        is_roi_map;
	std::vector<bool>        is_ctrl_map;

	std::vector<double> halfedge_weights;
	std::vector<CR_matrix> rotation_matrix;
	Sparse_linear_solver linear_solver;
	Vertex_point_map vertex_point_map;

	bool preprocess_successful;
	bool is_ids_cal_done;
	bool factorization_done;

public:

  Arap_mesh_deformation(Triangle_mesh& t_mesh): triangle_mesh(t_mesh)
  {
	  ros_id_map = std::vector<std::size_t>(num_vertices(t_mesh), (std::numeric_limits<std::size_t>::max)());
	  is_roi_map = std::vector<bool>(num_vertices(t_mesh), false);
	  is_ctrl_map = std::vector<bool>(num_vertices(t_mesh), false);
	  vertex_point_map = get(CGAL::vertex_point, t_mesh);// initialize a map of 3d_points
	  preprocess_successful = false;
	  is_ids_cal_done = false;
	  factorization_done = false;
	  halfedge_iterator eb, ee;
	  halfedge_weights.reserve(2 * num_edges(t_mesh)); // 2x because of halfedges
	  for (CGAL::cpp11::tie(eb, ee) = halfedges(t_mesh); eb != ee; ++eb)
	  {
		  halfedge_weights.push_back(calculate_weight(*eb));
	  }
	  
	   
  }

  bool insert_control_vertex(vertex_descriptor vd, CGAL::Simple_cartesian<double>::Point_3 &point)
  {
	  // factorization needs to be done again once this function is called.
	  if (is_control_vertex(vd)) 
	  { 
		  return false; 
	  }
	  factorization_done = false;
	  insert_roi_vertex(vd); // also insert it in region of interest

	  is_ctrl_map[get_vertex_id(vd)] = true;
	  point = get(vertex_point_map, vd);
	  return true;
  }

  bool insert_roi_vertex(vertex_descriptor vd)
  {
	  if (is_roi_vertex(vd)) 
	  { 
		  return false; 
	  }

	  is_roi_map[get_vertex_id(vd)] = true;
	  roi.push_back(vd);
	  return true;
  }


  bool preprocess()
  {
	  init_ids_and_rotations();
	  calculate_laplacian_and_factorize();
	  return preprocess_successful;
  }

  void set_target_position(vertex_descriptor vd, const Point& target_position)
  {	  
	  // ids should have been assigned before calling this function
	  init_ids_and_rotations();
	  if (!is_control_vertex(vd)) 
	  { 
		  return; 
	  }
	  solution[get_ros_id(vd)] = target_position; // target position directly goes to solution set, coz no calculation needed
  }

  void deform(unsigned int iterations, double tolerance)
  {
	  // preprocess();

	  if (!preprocess_successful) {
		  return;
	  }
	  double energy_this = 0;
	  double energy_last;
	  for (unsigned int ite = 0; ite < iterations; ++ite)
	  {
		  calculate_target_positions();
		  calculate_optimal_rotations();
		  if (tolerance > 0.0 && (ite + 1) < iterations)
		  {
			  energy_last = energy_this;
			  energy_this = compute_energy();
			  if (ite != 0)
			  {
				  double energy_dif = std::abs((energy_last - energy_this) / energy_this);
				  if (energy_dif < tolerance) 
				  { 
					  break; 
				  }
			  }
		  }
	  }

	  for (std::size_t i = 0; i < ros.size(); ++i) {
		  std::size_t v_id = get_ros_id(ros[i]);
		  if (is_roi_vertex(ros[i]))
		  {
			  put(vertex_point_map, ros[i], solution[v_id]); // assign new deformed positions to actual mesh vertices using vertex_descriptors
		  }
	  }
  }


  bool is_roi_vertex(vertex_descriptor vd) const
  {
	  return is_roi_map[get_vertex_id(vd)];
  }


  bool is_control_vertex(vertex_descriptor vd) const
  {
	  return is_ctrl_map[get_vertex_id(vd)];
  }

  void generate_roi_recursive(int curr_radius, int max_radius, vertex_descriptor vd, std::vector<vertex_descriptor>& list)
  {
	  
	  if (!is_roi_vertex(vd)) 
	  {
		  insert_roi_vertex(vd);
		  list.push_back(vd);
	  }

	  if (curr_radius < max_radius)// if max_radius reached, then don't make a call for vertex's one ring neighbours
	  {
		  curr_radius += 1;
		  in_edge_iterator e, e_end;
		  for (CGAL::cpp11::tie(e, e_end) = in_edges(vd, triangle_mesh); e != e_end; e++) // loop on one-ring neighbours
		  {
			  vertex_descriptor vt = source(*e, triangle_mesh);
			  generate_roi_recursive(curr_radius, max_radius, vt, list);
		  }
	  }
  }

  void generate_roi_from_vertex(int max_radius, vertex_descriptor vd, std::vector<vertex_descriptor>& list)
  {
	  generate_roi_recursive(0, max_radius, vd, list);
  }



private:


	double calculate_weight(halfedge_descriptor he)
	{
		vertex_descriptor v0 = target(he, triangle_mesh);
		vertex_descriptor v1 = source(he, triangle_mesh);

		if (is_border_edge(he, triangle_mesh))
		{

			halfedge_descriptor he_next = opposite(next(he, triangle_mesh), triangle_mesh);
			vertex_descriptor v2 = source(he_next, triangle_mesh);
			if (is_border_edge(he_next, triangle_mesh)) // if he and he_next both are border edges then they can't be part of same triangle, so check other end
			{
				halfedge_descriptor he_prev = prev(opposite(he, triangle_mesh), triangle_mesh);
				v2 = source(he_prev, triangle_mesh);
			}
			return (get_cotangent_value(v0, v2, v1) / 2.0);
		}
		else
		{
			// for non-border edge both alpha and beta are available
			halfedge_descriptor he_next = opposite(next(he, triangle_mesh), triangle_mesh);
			vertex_descriptor v2 = source(he_next, triangle_mesh);
			halfedge_descriptor he_prev = prev(opposite(he, triangle_mesh), triangle_mesh);
			vertex_descriptor v3 = source(he_prev, triangle_mesh);

			return ((get_cotangent_value(v0, v2, v1) + get_cotangent_value(v0, v3, v1)) / 2.0);
		}
	}


	// Using Cotangent formula from: http://people.eecs.berkeley.edu/~jrs/meshpapers/MeyerDesbrunSchroderBarr.pdf
	// Cot = cos/sin ==> using langrange's identity ==> a.b/sqrt(a^2 * b^2 - (a.b)^2)
	double get_cotangent_value(vertex_descriptor v0, vertex_descriptor v1, vertex_descriptor v2)
	{
		typedef CGAL::Simple_cartesian<double>::Vector_3 Vector;

		Vector a = get(vertex_point_map, v0) - get(vertex_point_map, v1);
		Vector b = get(vertex_point_map, v2) - get(vertex_point_map, v1);

		double dot_ab = a*b;

		Vector cross_ab = CGAL::cross_product(a, b);
		double divider = CGAL::sqrt(cross_ab*cross_ab);

		if (divider == 0)
		{
			return 0.0;
		}

		return (std::max)(0.0, (dot_ab / divider));
	}

	double compute_energy()
	{

		double sum_of_energy = 0;
		for (std::size_t k = 0; k < ros.size(); k++)
		{
			vertex_descriptor vi = ros[k];
			std::size_t vi_id = get_ros_id(vi);

			in_edge_iterator e, e_end;
			for (CGAL::cpp11::tie(e, e_end) = in_edges(vi, triangle_mesh); e != e_end; e++)
			{
				halfedge_descriptor he = halfedge(*e, triangle_mesh);
				vertex_descriptor vj = source(he, triangle_mesh);
				std::size_t vj_id = get_ros_id(vj);

				const CR_vector& pij = CR_vector(original[vi_id][0] - original[vj_id][0], original[vi_id][1] - original[vj_id][1], original[vi_id][2] - original[vj_id][2]);//sub_to_CR_vector(original[vi_id], original[vj_id]);
				const CR_vector& qij = CR_vector(solution[vi_id][0] - solution[vj_id][0], solution[vi_id][1] - solution[vj_id][1], solution[vi_id][2] - solution[vj_id][2]);//sub_to_CR_vector(solution[vi_id], solution[vj_id]);

				double wij = halfedge_weights[get_edge_id(he)];

				sum_of_energy += wij * (qij - rotation_matrix[vi_id] * pij).squaredNorm();

			}
		}
		return sum_of_energy;
	}


  void assign_ros_id_to_one_ring(vertex_descriptor vd, std::size_t& next_id, std::vector<vertex_descriptor>& push_vector)
  {
    in_edge_iterator e, e_end;
    for (CGAL::cpp11::tie(e,e_end) = in_edges(vd, triangle_mesh); e != e_end; e++)
    {
      vertex_descriptor vt = source(*e, triangle_mesh);
      if(get_ros_id(vt) == (std::numeric_limits<std::size_t>::max)())
      {
        get_ros_id(vt) = next_id++;
        push_vector.push_back(vt);
      }
    }
  }

 
  void init_ids_and_rotations()
  {

	if (is_ids_cal_done)
		return;
	
	is_ids_cal_done = true;
    ros.clear(); 
    ros.insert(ros.end(), roi.begin(), roi.end());

    ros_id_map.assign(num_vertices(triangle_mesh), (std::numeric_limits<std::size_t>::max)()); // assign max to represent invalid value

    for(std::size_t i = 0; i < roi.size(); i++)  
    { 
		get_ros_id(roi[i]) = i; 
	}

   
    std::size_t next_ros_index = roi.size();
    for(std::size_t i = 0; i < roi.size(); i++)
    { 
		assign_ros_id_to_one_ring(roi[i], next_ros_index, ros); 
	}

    std::vector<vertex_descriptor> outside_ros;
    
    for(std::size_t i = roi.size(); i < ros.size(); i++)
    { assign_ros_id_to_one_ring(ros[i], next_ros_index, outside_ros); }
    
    rotation_matrix.resize(ros.size());
    for(std::size_t i = 0; i < rotation_matrix.size(); i++)
    {
      std::size_t v_ros_id = get_ros_id(ros[i]);
	  rotation_matrix[v_ros_id] = CR_matrix().setIdentity();
    }

    solution.resize(ros.size() + outside_ros.size());
    original.resize(ros.size() + outside_ros.size());

    for(std::size_t i = 0; i < ros.size(); i++)
    {
      std::size_t v_ros_id = get_ros_id(ros[i]);
	  solution[v_ros_id] = get(vertex_point_map, ros[i]);
	  original[v_ros_id] = get(vertex_point_map, ros[i]);
    }
	
    for(std::size_t i = 0; i < outside_ros.size(); ++i)
    {
      std::size_t v_ros_id = get_ros_id(outside_ros[i]);
      original[v_ros_id] = get(vertex_point_map, outside_ros[i]);
      solution[v_ros_id] = get(vertex_point_map, outside_ros[i]);
    }
  }

  /// Construct matrix that corresponds to left-hand side of eq:lap_ber == LP' = b and 
  /// b == SumOfOneRing(Wij * ((Ri + Rj)/2) * (Vi-Vj)) and P'==Unknown, L==Weights
  void calculate_laplacian_and_factorize()
  {
	  if (factorization_done)
		  return;
	  factorization_done = true;
	  typename Sparse_linear_solver::Matrix A(ros.size());
	  for (std::size_t k = 0; k < ros.size(); k++)
	  {
		  vertex_descriptor vi = ros[k];
		  std::size_t vi_id = get_ros_id(vi);
		  if (is_roi_vertex(vi) && !is_control_vertex(vi))
		  {
			  double diagonal_val = 0;
			  in_edge_iterator e, e_end;
			  for (CGAL::cpp11::tie(e, e_end) = in_edges(vi, triangle_mesh); e != e_end; e++) // loop over 1-ring
			  {
				  halfedge_descriptor he = halfedge(*e, triangle_mesh);
				  vertex_descriptor vj = source(he, triangle_mesh);
				  double wij = halfedge_weights[get_edge_id(he)];  
				  double wji = halfedge_weights[get_edge_id(opposite(he, triangle_mesh))];
				  double total_weight = wij + wji;

				  A.set_coef(vi_id, get_ros_id(vj), -total_weight, true); // for non-diagonal entries, value is -(wij+wji)
				  diagonal_val += total_weight;
			  }
			  
			  A.set_coef(vi_id, vi_id, diagonal_val, true); // diagonal entries are sum of all others(1-ring) in this row
		  }
		  else
		  {
			  A.set_coef(vi_id, vi_id, 1.0, true); // for control vertices just set to 1 coz their target position comes from user
		  }
	  }

	  
	  double D;
	  preprocess_successful = linear_solver.factor(A, D);
  }
  
  // Covariance matrix S = SumOverOneRing(wij * pij * qij.transpose)
  void calculate_optimal_rotations()
  {

	  CR_matrix cov = CR_matrix().setZero();

	  for (std::size_t k = 0; k < ros.size(); k++)
	  {
		  vertex_descriptor vi = ros[k];
		  std::size_t vi_id = get_ros_id(vi);

		  cov = CR_matrix().setZero();

		  in_edge_iterator e, e_end;

		  for (CGAL::cpp11::tie(e, e_end) = in_edges(vi, triangle_mesh); e != e_end; e++)
		  {
			  halfedge_descriptor he = halfedge(*e, triangle_mesh);
			  vertex_descriptor vj = source(he, triangle_mesh);
			  std::size_t vj_id = get_ros_id(vj);

			  const CR_vector& pij = CR_vector(original[vi_id][0] - original[vj_id][0], original[vi_id][1] - original[vj_id][1], original[vi_id][2] - original[vj_id][2]);
			  const CR_vector& qij = CR_vector(solution[vi_id][0] - solution[vj_id][0], solution[vi_id][1] - solution[vj_id][1], solution[vi_id][2] - solution[vj_id][2]);
			  double wij = halfedge_weights[get_edge_id(he)];

			  cov += wij * (pij * qij.transpose());

		  }

		  compute_close_rotation(cov, rotation_matrix[vi_id]);
	  }
  }


  void calculate_target_positions()
  {
	  typename Sparse_linear_solver::Vector X(ros.size()), Bx(ros.size());
	  typename Sparse_linear_solver::Vector Y(ros.size()), By(ros.size());
	  typename Sparse_linear_solver::Vector Z(ros.size()), Bz(ros.size());


	  for (std::size_t k = 0; k < ros.size(); k++)
	  {
		  vertex_descriptor vi = ros[k];
		  std::size_t vi_id = get_ros_id(vi);

		  if (is_roi_vertex(vi) && !is_control_vertex(vi))
		  {

			  CR_vector xyz = CR_vector(0, 0, 0);

			  in_edge_iterator e, e_end;
			  for (CGAL::cpp11::tie(e, e_end) = in_edges(vi, triangle_mesh); e != e_end; e++)
			  {
				  halfedge_descriptor he = halfedge(*e, triangle_mesh);
				  vertex_descriptor vj = source(he, triangle_mesh);
				  std::size_t vj_id = get_ros_id(vj);

				  const CR_vector& pij = CR_vector(original[vi_id][0] - original[vj_id][0], original[vi_id][1] - original[vj_id][1], original[vi_id][2] - original[vj_id][2]);

				  double wij = halfedge_weights[get_edge_id(he)];
				  double wji = halfedge_weights[get_edge_id(opposite(he, triangle_mesh))];

				  xyz += (wij * rotation_matrix[vi_id] + wji * rotation_matrix[vj_id]) * pij;
			  }
			  Bx[vi_id] = xyz(0);
			  By[vi_id] = xyz(1);
			  Bz[vi_id] = xyz(2);
		  }
		  else
		  {		// control points's target positions are given by the user, so directly assign those values
			  Bx[vi_id] = solution[vi_id][0]; 
			  By[vi_id] = solution[vi_id][1]; 
			  Bz[vi_id] = solution[vi_id][2];
		  }
	  }
	  // Call solver for each dimension of the point seperately 
	  bool is_all_solved = linear_solver.linear_solver(Bx, X) && linear_solver.linear_solver(By, Y) && linear_solver.linear_solver(Bz, Z);
	  if (!is_all_solved) {
		  // could not solve all
		  return;
	  }
	  
	  for (std::size_t i = 0; i < ros.size(); i++)
	  {
		  std::size_t v_id = get_ros_id(ros[i]);
		  Point p(X[v_id], Y[v_id], Z[v_id]);
		  if (!is_ctrl_map[get_vertex_id(ros[i])])
			  solution[v_id] = p;
	  }
  }
  
  // Get original vertex id
  std::size_t get_vertex_id(vertex_descriptor vd) const
  { 
	  return vd->id();
  }

  // get ros_id as L value
  std::size_t& get_ros_id(vertex_descriptor vd)
  { 
	  return ros_id_map[get_vertex_id(vd)]; 
  }

  // get ros_id as R value
  std::size_t  get_ros_id(vertex_descriptor vd) const
  {
	  return ros_id_map[get_vertex_id(vd)];
  }
  // get half edge id
  std::size_t get_edge_id(halfedge_descriptor e) const
  {
	  return e->id();
  }

  // Calculate the closest rotation, which according to paper is : R = V*(identity_mtrix with last element == det(V*U.transpose()))*U.transpose()
  // det(V*U.transpose()) can be +1 or -1
  void compute_close_rotation(const CR_matrix& m, CR_matrix& R)
  {
	  Eigen::JacobiSVD<Eigen::Matrix3d> solver;
	  solver.compute(m, Eigen::ComputeFullU | Eigen::ComputeFullV);

	  const CR_matrix& u = solver.matrixU(); const CR_matrix& v = solver.matrixV();
	  R = v * u.transpose();

	  // If determinanat of rotation is < 0 , then multiply last column of U with -1 and calculate rotation again
	  if (R.determinant() < 0) {
		  CR_matrix u_copy = u;
		  u_copy.col(2) *= -1;        
		  R = v * u_copy.transpose(); 
	  }
  }

};
#endif  // ARAP_MESH_DEFORMATION_H
