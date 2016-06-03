/*
A modified version of Surface_mesh_deformation file from CGAL
*/

#ifndef ARAP_MESH_DEFORMATION_H
#define ARAP_MESH_DEFORMATION_H

#include <CGAL/config.h>
#include <CGAL/Default.h>
#include <CGAL/tuple.h>

#include <CGAL/Polygon_mesh_processing/Weights.h>
#include <CGAL/Simple_cartesian.h>

#include <vector>
#include <list>
#include <utility>
#include <limits>
#include <boost/foreach.hpp>

#include <CGAL/Eigen_solver_traits.h>  // for sparse linear system solver
#include <CGAL/Deformation_Eigen_polar_closest_rotation_traits_3.h>  // for 3x3 closest rotation computer



// property map that create a Simple_cartesian<double>::Point_3
// on the fly in order the deformation class to be used
// with points with minimal requirements
template <class Vertex_point_map>
struct SC_on_the_fly_pmap: public Vertex_point_map{
  typedef boost::readable_property_map_tag category;
  typedef CGAL::Simple_cartesian<double>::Point_3 value_type;
  typedef value_type reference;
  typedef typename boost::property_traits<Vertex_point_map>::key_type key_type;

  SC_on_the_fly_pmap(Vertex_point_map base):
    Vertex_point_map(base) {}

  friend value_type
  get(const SC_on_the_fly_pmap map, key_type k)
  {
    typename boost::property_traits<Vertex_point_map>::reference base=
      get(static_cast<const Vertex_point_map&>(map), k);
    return value_type(base[0], base[1], base[2]);
  }
};


template < class TM >
class Arap_mesh_deformation
{

public:
	
  typedef TM Triangle_mesh;
  typedef TM Halfedge_graph;

  typedef CGAL::internal::Cotangent_weight_impl<TM> Weight_calculator;
  
  // sparse linear solver
  
  typedef typename CGAL::Eigen_solver_traits< Eigen::SparseLU< CGAL::Eigen_sparse_matrix<double>::EigenType, Eigen::COLAMDOrdering<int> > > Sparse_linear_solver;
  
  // CR helper
  
  typedef typename CGAL::Deformation_Eigen_polar_closest_rotation_traits_3 Closest_rotation_traits;
  
  // vertex point pmap
 
  typedef typename boost::property_map<Triangle_mesh, CGAL::vertex_point_t>::type Vertex_point_map;
  
  
  /// The type for vertex descriptor
  typedef typename boost::graph_traits<Triangle_mesh>::vertex_descriptor vertex_descriptor;
  /// The type for halfedge descriptor
  typedef typename boost::graph_traits<Triangle_mesh>::halfedge_descriptor halfedge_descriptor;
  /// The 3D point type, model of `::RawPoint_3`
  typedef typename boost::property_traits<Vertex_point_map>::value_type Point;
  /// A constant iterator range over the vertices of the region-of-interest.
  /// It is a model of `ConstRange` with `vertex_descriptor` as iterator value type.
  typedef std::vector<vertex_descriptor> Roi_vertex_range;

private:
  typedef Arap_mesh_deformation< TM > Self;
  // Repeat Triangle_mesh types
  typedef typename boost::graph_traits<Triangle_mesh>::vertex_iterator     vertex_iterator;
  typedef typename boost::graph_traits<Triangle_mesh>::halfedge_iterator       halfedge_iterator;
  typedef typename boost::graph_traits<Triangle_mesh>::in_edge_iterator    in_edge_iterator;
  typedef typename boost::graph_traits<Triangle_mesh>::out_edge_iterator   out_edge_iterator;

  typedef typename Closest_rotation_traits::Matrix CR_matrix;
  typedef typename Closest_rotation_traits::Vector CR_vector;

// Data members.
  Triangle_mesh& m_triangle_mesh;                   /**< Source triangulated surface mesh for modeling */

  std::vector<Point> original;                        ///< original positions of roi (size: ros + boundary_of_ros)
  std::vector<Point> solution;                        ///< storing position of ros vertices during iterations (size: ros + boundary_of_ros)

  std::vector<vertex_descriptor> roi;                 ///< region of interest
  std::vector<vertex_descriptor> ros;                 ///< region of solution, including roi and hard constraints on boundary of roi

  std::vector<std::size_t> ros_id_map;                ///< (size: num vertices)
  std::vector<bool>        is_roi_map;                ///< (size: num vertices)
  std::vector<bool>        is_ctrl_map;               ///< (size: num vertices)

  std::vector<double> hedge_weight;                   ///< all halfedge weights
  std::vector<CR_matrix> rot_mtr;                     ///< rotation matrices of ros vertices (size: ros)

  Sparse_linear_solver m_solver;                      ///< linear sparse solver
  unsigned int m_iterations;                          ///< number of maximal iterations
  double m_tolerance;                                 ///< tolerance of convergence

  bool need_preprocess_factorization;                 ///< is there any need to compute L and factorize
  bool need_preprocess_region_of_solution;            ///< is there any need to compute region of solution

  bool last_preprocess_successful;                    ///< stores the result of last call to preprocess()

  Weight_calculator weight_calculator;

  Vertex_point_map vertex_point_map;


  Arap_mesh_deformation(const Self&) = delete; // no copy

public:

  Arap_mesh_deformation(Triangle_mesh& triangle_mesh)
    : m_triangle_mesh(triangle_mesh),
	  ros_id_map(std::vector<std::size_t>(num_vertices(triangle_mesh), (std::numeric_limits<std::size_t>::max)())),
	  is_roi_map(std::vector<bool>(num_vertices(triangle_mesh), false)),
	  is_ctrl_map(std::vector<bool>(num_vertices(triangle_mesh), false)),
	  m_iterations(5), m_tolerance(1e-4),
	  need_preprocess_factorization(true),
	  need_preprocess_region_of_solution(true),
	  last_preprocess_successful(false),
	  weight_calculator(Weight_calculator()),
	  vertex_point_map(get(CGAL::vertex_point, triangle_mesh))   
  {
    init();
  }


private:
  void init() {

	
    typedef SC_on_the_fly_pmap<Vertex_point_map> Wrapper;
    // compute halfedge weights
    halfedge_iterator eb, ee;
    hedge_weight.reserve(2*num_edges(m_triangle_mesh));
    for(CGAL::cpp11::tie(eb, ee) = halfedges(m_triangle_mesh); eb != ee; ++eb)
    {
      hedge_weight.push_back(
        this->weight_calculator(*eb, m_triangle_mesh, Wrapper(vertex_point_map)));
    }

  }

public:

  /**
   * Erases all the vertices from the region-of-interest (control vertices included).
   */
  void clear_roi_vertices(){
    need_preprocess_both();
    // clear roi vertices
    roi.clear();
    //set to false all bits
    is_roi_map.assign(num_vertices(m_triangle_mesh), false);
    is_ctrl_map.assign(num_vertices(m_triangle_mesh), false);
  }

  /**
   * Erases all the vertices from the set of control vertices.
   */
  void clear_control_vertices(){
    need_preprocess_factorization=true;
    //set to false all bits
    is_ctrl_map.assign(num_vertices(m_triangle_mesh), false);
  }

  
  bool insert_control_vertex(vertex_descriptor vd, CGAL::Simple_cartesian<double>::Point_3 &point)
  {
    if(is_control_vertex(vd)) { return false; }
    need_preprocess_factorization=true;

    insert_roi_vertex(vd); // also insert it as roi

    is_ctrl_map[id(vd)] = true;
	point = get(vertex_point_map, vd);
    return true;
  }

 
  template<class InputIterator>
  void insert_control_vertices(InputIterator begin, InputIterator end)
  {
    for( ;begin != end; ++begin)
    {
      insert_control_vertex(*begin);
    }
  }

  /**
   * Erases a vertex from the set of control vertices.
   * @param vd the vertex to be erased
   * @return `true` if `vd` was a control vertex.
   */
  bool erase_control_vertex(vertex_descriptor vd)
  {
    if(!is_control_vertex(vd)) { return false; }

    need_preprocess_factorization=true;
    is_ctrl_map[id(vd)] = false;
    return true;
  }

 
  template<class InputIterator>
  void insert_roi_vertices(InputIterator begin, InputIterator end)
  {
    for( ;begin != end; ++begin)
    {
      insert_roi_vertex(*begin);
    }
  }

  
  bool insert_roi_vertex(vertex_descriptor vd)
  {
    if(is_roi_vertex(vd)) { return false; }
    need_preprocess_both();

    is_roi_map[id(vd)] = true;
    roi.push_back(vd);
    return true;
  }

 
  bool erase_roi_vertex(vertex_descriptor vd)
  {
    if(!is_roi_vertex(vd)) { return false; }

    erase_control_vertex(vd); // also erase from being control

    typename std::vector<vertex_descriptor>::iterator it = std::find(roi.begin(), roi.end(), vd);
    if(it != roi.end())
    {
      is_roi_map[id(vd)] = false;
      roi.erase(it);

      need_preprocess_both();
      return true;
    }

    CGAL_assertion(false); // inconsistency between is_roi_map, and roi vector!
    return false;
  }

  /**
   * Preprocessing function that need to be called each time the region-of-interest or the set
   * of control vertices are changed before calling `deform()`.
   * If not already done, `deform()` first calls this function.
   * 
   * Collects the vertices not in the region-of-interest that are adjacent to a vertex from the
   * region-of-interest (these vertices are internally considered as fixed control vertices).
   * Then assembles and factorizes the Laplacian matrix used in the function `deform()`.
   *
   * \note A modification of the set of control vertices or the region-of-interest invalidates the
   * preprocessing data.
   * @return `true` if successful.
   */
  bool preprocess()
  {
    region_of_solution();// setup arrays for region of solution and id maps
    assemble_laplacian_and_factorize();
    return last_preprocess_successful; // which is set by assemble_laplacian_and_factorize()
  }

  /**
   * Sets the target position of a control vertex.
   * @param vd the control vertex the target position is set
   * @param target_position the new target position
   */
  void set_target_position(vertex_descriptor vd, const Point& target_position)
  {
    region_of_solution(); // we require ros ids, so if there is any need to preprocess of region of solution -do it.

    if(!is_control_vertex(vd)) { return; }
    solution[ros_id(vd)] = target_position;
  }

  
  const Point& target_position(vertex_descriptor vd)
  {
    region_of_solution();

    CGAL_precondition( is_control_vertex(vd) );
    return solution[ ros_id(vd) ];
  }

  /**
   * Deforms the region-of-interest according to the deformation algorithm, using the target positions of each control vertex set by using `rotate()`, `translate()`, or `set_target_position()`.
   * The points associated to each vertex of the input triangle mesh that are inside the region-of-interest are updated.
   * \note Nothing happens if `preprocess()` returns `false`.
   * @see set_iterations(unsigned int iterations), set_tolerance(double tolerance), deform(unsigned int iterations, double tolerance)
   */
  void deform()
  {
    deform(m_iterations, m_tolerance);
  }

  
  void deform(unsigned int iterations, double tolerance)
  {
    preprocess();

    if(!last_preprocess_successful) {
      CGAL_warning(false);
      return;
    }
    
    double energy_this = 0; 
    double energy_last;

    // iterations
    for ( unsigned int ite = 0; ite < iterations; ++ite)
    {
      // main steps of optimization
      update_solution();

      optimal_rotations();

      // energy based termination
      if(tolerance > 0.0 && (ite + 1) < iterations) // if tolerance <= 0 then don't compute energy
      {                                             // also no need compute energy if this iteration is the last iteration
        energy_last = energy_this;
        energy_this = energy();
        CGAL_warning(energy_this >= 0);

        if(ite != 0) // skip first iteration
        {
          double energy_dif = std::abs((energy_last - energy_this) / energy_this);
          if ( energy_dif < tolerance ) { break; }
        }
      }
    }
    // copy solution to the target surface mesh
    assign_solution();
  }

  
  void reset()
  {
    if(roi.empty()) { return; } // no ROI to reset

    region_of_solution(); // since we are using original vector

    //restore the current positions to be the original positions
    BOOST_FOREACH(vertex_descriptor vd, roi_vertices())
    {
      put(vertex_point_map, vd, original[ros_id(vd)]);
      solution[ros_id(vd)]=original[ros_id(vd)];
    }

    // also set rotation matrix to identity
    std::fill(rot_mtr.begin(), rot_mtr.end(),
              Closest_rotation_traits().identity_matrix());
  }



  /**
   * Gets the default number of iterations (5) or the value passed to the function `set_iterations()`
   */
  unsigned int iterations()
  { 
	  return m_iterations; 
  }

  /**
   * Gets the default tolerance parameter (1e-4) or the value passed to the function `set_tolerance()`
   */
  double tolerance()
  { 
	  return m_tolerance; 
  }

  /**
   * Sets the maximum number of iterations ran by `deform()`
   */
  void set_iterations(unsigned int iterations)
  { 
	  this->m_iterations = iterations; 
  }

   /// @brief Sets the tolerance of the convergence used in `deform()`.
  void set_tolerance(double tolerance)
  { 
	  this->m_tolerance = tolerance; 
  }

  /**
   * Returns the range of vertices in the region-of-interest.
   */
  const Roi_vertex_range& roi_vertices() const
  {
    return roi;
  }

  /**
   * Tests whether a vertex is inside the region-of-interest.
   * @param vd the query vertex
   * @return `true` if `vd` has been inserted to (and not erased from) the region-of-interest.
   */
  bool is_roi_vertex(vertex_descriptor vd) const
  { 
	  return is_roi_map[id(vd)]; 
  }

  
  bool is_control_vertex(vertex_descriptor vd) const
  { 
	  return is_ctrl_map[id(vd)]; 
  }

 
  const Triangle_mesh& triangle_mesh() const
  { 
	  return m_triangle_mesh; 
  }
  const Triangle_mesh& halfedge_graph() const
  { 
	  return m_triangle_mesh; 
  }

private:

  /// Assigns id to one ring neighbor of vd, and also push them into push_vector
  void assign_ros_id_to_one_ring(vertex_descriptor vd,
                             std::size_t& next_id,
                             std::vector<vertex_descriptor>& push_vector)
  {
    in_edge_iterator e, e_end;
    for (CGAL::cpp11::tie(e,e_end) = in_edges(vd, m_triangle_mesh); e != e_end; e++)
    {
      vertex_descriptor vt = source(*e, m_triangle_mesh);
      if(ros_id(vt) == (std::numeric_limits<std::size_t>::max)())  // neighboring vertex which is outside of roi and not visited previously (i.e. need an id)
      {
        ros_id(vt) = next_id++;
        push_vector.push_back(vt);
      }
    }
  }

 
  void region_of_solution()
  {
    if(!need_preprocess_region_of_solution) { return; }
    need_preprocess_region_of_solution = false;

    std::vector<std::size_t>  old_ros_id_map = ros_id_map;
    std::vector<CR_matrix>    old_rot_mtr    = rot_mtr;
    std::vector<Point>        old_solution   = solution;
    std::vector<Point>        old_original   = original;

    
    for(typename std::vector<vertex_descriptor>::iterator it = ros.begin(); it != ros.end(); ++it)
    {
      if(!is_roi_vertex(*it)) {
        put(vertex_point_map, *it, old_original[ old_ros_id_map[id(*it)] ]);
      }
    }

    ros.clear(); // clear ros
    ros.insert(ros.end(), roi.begin(), roi.end());

    ros_id_map.assign(num_vertices(m_triangle_mesh), (std::numeric_limits<std::size_t>::max)()); // use max as not assigned mark

    for(std::size_t i = 0; i < roi.size(); i++)  // assign id to all roi vertices
    { ros_id(roi[i]) = i; }

    // now assign an id to vertices on boundary of roi
    std::size_t next_ros_index = roi.size();
    for(std::size_t i = 0; i < roi.size(); i++)
    { assign_ros_id_to_one_ring(roi[i], next_ros_index, ros); }

    std::vector<vertex_descriptor> outside_ros;
    
    for(std::size_t i = roi.size(); i < ros.size(); i++)
    { assign_ros_id_to_one_ring(ros[i], next_ros_index, outside_ros); }
    

    // initialize the rotation matrices (size: ros)
    rot_mtr.resize(ros.size());
    for(std::size_t i = 0; i < rot_mtr.size(); i++)
    {
      std::size_t v_ros_id = ros_id(ros[i]);
      std::size_t v_id = id(ros[i]);

      // any vertex which is previously ROS has a rotation matrix
      // use that matrix to prevent jumping effects
      if(old_ros_id_map[v_id] != (std::numeric_limits<std::size_t>::max)()
          && old_ros_id_map[v_id] < old_rot_mtr.size()) {

        rot_mtr[v_ros_id] = old_rot_mtr[ old_ros_id_map[v_id] ];
      }
      else {
        rot_mtr[v_ros_id] = Closest_rotation_traits().identity_matrix();
      }
    }

    
    solution.resize(ros.size() + outside_ros.size());
    original.resize(ros.size() + outside_ros.size());

    for(std::size_t i = 0; i < ros.size(); i++)
    {
      std::size_t v_ros_id = ros_id(ros[i]);
      std::size_t v_id = id(ros[i]);

      if(is_roi_vertex(ros[i]) && old_ros_id_map[v_id] != (std::numeric_limits<std::size_t>::max)()) {

        original[v_ros_id] = old_original[old_ros_id_map[v_id]];
        solution[v_ros_id] = old_solution[old_ros_id_map[v_id]];
      }
      else {
        solution[v_ros_id] = get(vertex_point_map, ros[i]);
        original[v_ros_id] = get(vertex_point_map, ros[i]);
      }
    }

    for(std::size_t i = 0; i < outside_ros.size(); ++i)
    {
      std::size_t v_ros_id = ros_id(outside_ros[i]);
      original[v_ros_id] = get(vertex_point_map, outside_ros[i]);
      solution[v_ros_id] = get(vertex_point_map, outside_ros[i]);
    }
  }

  /// Assemble Laplacian matrix A of linear system A*X=B
  void assemble_laplacian_and_factorize()
  {
      assemble_laplacian_and_factorize_arap();
  }
  /// Construct matrix that corresponds to left-hand side of eq:lap_ber == LP' = b and 
  /// b == SumOfOneRing(Wij * ((Ri + Rj)/2) * (Vi-Vj)) and P'==Unknown, L==Weights
  /// 
  /// Also constraints are integrated as eq:lap_energy_system in user manual
  void assemble_laplacian_and_factorize_arap()
  {
    if(!need_preprocess_factorization) { return; }
    need_preprocess_factorization = false;

    typename Sparse_linear_solver::Matrix A(ros.size());

    /// assign cotangent Laplacian to ros vertices
    for(std::size_t k = 0; k < ros.size(); k++)
    {
      vertex_descriptor vi = ros[k];
      std::size_t vi_id = ros_id(vi);
      if ( is_roi_vertex(vi) && !is_control_vertex(vi) )          
      {
        double diagonal = 0;
        in_edge_iterator e, e_end;
        for (CGAL::cpp11::tie(e,e_end) = in_edges(vi, m_triangle_mesh); e != e_end; e++)
        {
          halfedge_descriptor he = halfedge(*e, m_triangle_mesh);
          vertex_descriptor vj = source(he, m_triangle_mesh);
          double wij = hedge_weight[id(he)];  // edge(pi - pj)
          double wji = hedge_weight[id(opposite(he, m_triangle_mesh))]; // edge(pi - pj)
          double total_weight = wij + wji;

          A.set_coef(vi_id, ros_id(vj), -total_weight, true); 
          diagonal += total_weight;
        }
        // diagonal coefficient
        A.set_coef(vi_id, vi_id, diagonal, true);
      }
      else
      {
        A.set_coef(vi_id, vi_id, 1.0, true);
      }
    }

    // now factorize
    double D;
    last_preprocess_successful = m_solver.factor(A, D);
    CGAL_warning(last_preprocess_successful);
  }
  
  /// Local step of iterations, computing optimal rotation matrices
  void optimal_rotations()
  {
      optimal_rotations_arap();
  }

  void optimal_rotations_arap()
  {
    Closest_rotation_traits cr_traits;
    CR_matrix cov = cr_traits.zero_matrix();

    for ( std::size_t k = 0; k < ros.size(); k++ )
    {
      vertex_descriptor vi = ros[k];
      std::size_t vi_id = ros_id(vi);
      
      cov = cr_traits.zero_matrix();

      in_edge_iterator e, e_end;

      for (CGAL::cpp11::tie(e,e_end) = in_edges(vi, m_triangle_mesh); e != e_end; e++)
      {
        halfedge_descriptor he=halfedge(*e, m_triangle_mesh);
        vertex_descriptor vj = source(he, m_triangle_mesh);
        std::size_t vj_id = ros_id(vj);

        const CR_vector& pij = sub_to_CR_vector(original[vi_id], original[vj_id]);
        const CR_vector& qij = sub_to_CR_vector(solution[vi_id], solution[vj_id]);
        double wij = hedge_weight[id(he)];

        cr_traits.add_scalar_t_vector_t_vector_transpose(cov, wij, pij, qij); // cov += wij * (pij * qij)
		
      }

      cr_traits.compute_close_rotation(cov, rot_mtr[vi_id]);
    }
  }
 
  /// Global step of iterations, updating solution
  void update_solution()
  {
      update_solution_arap();
  }
 
  void update_solution_arap()
  {
    typename Sparse_linear_solver::Vector X(ros.size()), Bx(ros.size());
    typename Sparse_linear_solver::Vector Y(ros.size()), By(ros.size());
    typename Sparse_linear_solver::Vector Z(ros.size()), Bz(ros.size());

    Closest_rotation_traits cr_traits;

    
    for ( std::size_t k = 0; k < ros.size(); k++ )
    {
      vertex_descriptor vi = ros[k];
      std::size_t vi_id = ros_id(vi);

      if ( is_roi_vertex(vi) && !is_control_vertex(vi) )
      {// free vertices
        
        CR_vector xyz = cr_traits.vector(0, 0, 0);

        in_edge_iterator e, e_end;
        for (CGAL::cpp11::tie(e,e_end) = in_edges(vi, m_triangle_mesh); e != e_end; e++)
        {
          halfedge_descriptor he = halfedge(*e, m_triangle_mesh);
          vertex_descriptor vj = source(he, m_triangle_mesh);
          std::size_t vj_id = ros_id(vj);

          const CR_vector& pij = sub_to_CR_vector(original[vi_id], original[vj_id]);

          double wij = hedge_weight[id(he)];
          double wji = hedge_weight[id(opposite(he, m_triangle_mesh))];

          cr_traits.add__scalar_t_matrix_p_scalar_t_matrix__t_vector(xyz, wij, rot_mtr[vi_id], wji, rot_mtr[vj_id], pij); 
          // corresponds xyz += (wij*rot_mtr[vi_id] + wji*rot_mtr[vj_id]) * pij // missed divided by 2 here why?
        }
        Bx[vi_id] = cr_traits.vector_coordinate(xyz, 0);
        By[vi_id] = cr_traits.vector_coordinate(xyz, 1);
        Bz[vi_id] = cr_traits.vector_coordinate(xyz, 2);
      }
      else
      {// constrained vertex
        Bx[vi_id] = solution[vi_id][0]; By[vi_id] = solution[vi_id][1]; Bz[vi_id] = solution[vi_id][2];
      }
    }

    // solve "A*X = B".
    bool is_all_solved = m_solver.linear_solver(Bx, X) && m_solver.linear_solver(By, Y) && m_solver.linear_solver(Bz, Z);
    if(!is_all_solved) {
      CGAL_warning(false);
      return;
    }
    // copy to solution
    for (std::size_t i = 0; i < ros.size(); i++)
    {
      std::size_t v_id = ros_id(ros[i]);
      Point p(X[v_id], Y[v_id], Z[v_id]);
      if (!is_ctrl_map[id(ros[i])])
        solution[v_id] = p;
    }
  }
  
  /// Assign solution to target surface mesh
  void assign_solution()
  {
    for(std::size_t i = 0; i < ros.size(); ++i){
      std::size_t v_id = ros_id(ros[i]);
      if(is_roi_vertex(ros[i]))
      {
        put(vertex_point_map, ros[i], solution[v_id]);
      }
    }
  }

  /// Compute modeling energy
  double energy() const
  {
      return energy_arap();
      return 0;
  }

  double energy_arap() const
  {
    Closest_rotation_traits cr_traits;

    double sum_of_energy = 0;
    // only accumulate ros vertices
    for( std::size_t k = 0; k < ros.size(); k++ )
    {
      vertex_descriptor vi = ros[k];
      std::size_t vi_id = ros_id(vi);

      in_edge_iterator e, e_end;
      for (CGAL::cpp11::tie(e,e_end) = in_edges(vi, m_triangle_mesh); e != e_end; e++)
      {
        halfedge_descriptor he = halfedge(*e, m_triangle_mesh);
        vertex_descriptor vj = source(he, m_triangle_mesh);
        std::size_t vj_id = ros_id(vj);

        const CR_vector& pij = sub_to_CR_vector(original[vi_id], original[vj_id]);
        const CR_vector& qij = sub_to_CR_vector(solution[vi_id], solution[vj_id]);

        double wij = hedge_weight[id(he)];

        sum_of_energy += wij * cr_traits.squared_norm_vector_scalar_vector_subs(qij, rot_mtr[vi_id], pij);
        // sum_of_energy += wij * ( qij - rot_mtr[vi_id]*pij )^2
      }
    }
    return sum_of_energy;
  }
  
  void need_preprocess_both()
  {
    need_preprocess_factorization = true;
    need_preprocess_region_of_solution = true;
  }

  /// p1 - p2, return CR_vector
  CR_vector sub_to_CR_vector(const Point& p1, const Point& p2) const
  {
    return Closest_rotation_traits().vector(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
  }

  template<class Vect>
  Point add_to_point(const Point& p, const Vect& v) {
    return Point(p[0] + v[0], p[1] + v[1], p[2] + v[2]);
  }

  template<class Vect>
  Vect sub_to_vect(const Point& p, const Vect& v) {
    return Vect(p[0] - v[0], p[1] - v[1], p[2] - v[2]);
  }

  /// shorthand of get(vertex_index_map, v)
  std::size_t id(vertex_descriptor vd) const
  { 
	  return vd->id();
  }

  std::size_t& ros_id(vertex_descriptor vd)
  { return ros_id_map[id(vd)]; }
  std::size_t  ros_id(vertex_descriptor vd) const
  { return ros_id_map[id(vd)]; }

  
  std::size_t id(halfedge_descriptor e) const
  {
	  return e->id();
  }
};
#endif  // ARAP_MESH_DEFORMATION_H
