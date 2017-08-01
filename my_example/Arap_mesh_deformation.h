#ifndef ARAP_MESH_DEFORMATION_H
#define ARAP_MESH_DEFORMATION_H

#include <iostream>
#include <thread>
#include <future>
#include <mutex>
#include <vector>
#include <list>
#include <utility>
#include <limits>
#include <Eigen/Eigen>
#include <Eigen/SVD>
#include<Eigen/SparseLU>

// Class to perform ARAP Mesh Deformation on triangular meshes
class Arap_mesh_deformation
{

public:
	typedef Eigen::SparseLU< Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > Sparse_linear_solver;
	double avg_solver_time;
	double avg_rotation_time;
private:
	Eigen::MatrixXd& vertices;
	Eigen::MatrixXi& triangles;
	Eigen::VectorXd prev_X;
	Eigen::VectorXd prev_Y;
	Eigen::VectorXd prev_Z;
	std::vector<Eigen::Vector3d> original;
	std::vector<Eigen::Vector3d> solution;
	std::vector<Eigen::Vector3d> prev_solution;
	std::vector<std::size_t> roi; // region of interest
	std::vector<std::size_t> ros; // region of solution
	std::vector<std::size_t> ros_id_map;
	std::vector<bool>        is_roi_map;
	std::vector<bool>        is_ctrl_map;
	std::vector<std::vector<int>> adjacent_vertices;
	std::vector<Eigen::Matrix3d> rotation_matrix;
	std::vector< Eigen::Triplet<double> > tripletList;
	std::map<std::size_t, std::size_t> ros_to_original_map;
	Eigen::MatrixXd e_weights;
	std::thread *weighting_thread;
	Sparse_linear_solver linear_solver;
	Eigen::SparseMatrix<double> *A;
	const typename Eigen::SparseMatrix<double> *tmp;
	bool preprocess_successful;
	bool is_ids_cal_done;
	bool factorization_done;
	double theta;


public:

	Arap_mesh_deformation(Eigen::MatrixXd &V, Eigen::MatrixXi &F) : vertices(V), triangles(F)
	{

		ros_id_map = std::vector<std::size_t>(vertices.rows(), (std::numeric_limits<std::size_t>::max)());
		is_roi_map = std::vector<bool>(vertices.rows(), false);
		is_ctrl_map = std::vector<bool>(vertices.rows(), false);
		preprocess_successful = false;
		is_ids_cal_done = false;
		factorization_done = false;
		avg_rotation_time = 0.0;
		avg_solver_time = 0.0;
		theta = 0.81; // for condition number=100, theta=0.81 and for condition number=1000, theta=0.938
		weighting_thread = new std::thread(&Arap_mesh_deformation::cal_weights, this);
	}

	bool insert_control_vertex(int vid)
	{
		// factorization needs to be done again once this function is called.
		if (is_control_vertex(vid))
		{
			return false;
		}
		factorization_done = false;
		insert_roi_vertex(vid); // also insert it in region of interest
		is_ctrl_map[vid] = true;
		return true;
	}

	void insert_roi_vertices(int begin, int end)
	{
		for (; begin <= end; ++begin)
		{
			insert_roi_vertex(begin);
		}
	}

	bool insert_roi_vertex(int vid)
	{
		if (is_roi_vertex(vid))
		{
			return false;
		}
		is_roi_map[vid] = true;
		roi.push_back(vid);
		return true;
	}
	
	bool preprocess()
	{
		init_ids_and_rotations();
		calculate_laplacian_and_factorize();
		return preprocess_successful;
	}

	void set_target_position(int vid, const Eigen::Vector3d& target_position)
	{
		// ids should have been assigned before calling this function
		init_ids_and_rotations();
		if (!is_control_vertex(vid))
		{
			return;
		}
		solution[get_ros_id(vid)] = target_position; // target position directly goes to solution set, coz no calculation needed
	}

	void deform(unsigned int iterations, double tolerance)
	{
		if (!preprocess_successful) {
			return;
		}
		double energy_this = 0;
		double energy_last;
		int itrs = 0;
		for (unsigned int ite = 0; ite < iterations; ++ite)
		{
			itrs += 1;
			if (ite != 0)
				calculate_target_positions(true); // apply acceleration
			else
				calculate_target_positions(false);
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
				vertices.row(ros[i]) = Eigen::RowVector3d(solution[v_id](0), solution[v_id](1), solution[v_id](2));
			}
		}
	}


	bool is_roi_vertex(int vid) const
	{
		return is_roi_map[vid];
	}


	bool is_control_vertex(int vid) const
	{
		return is_ctrl_map[vid];
	}

	void reset()
	{
		if (roi.empty()) { return; } // no ROI to reset
		avg_rotation_time = 0.0;
		avg_solver_time = 0.0;
	}

	bool double_equals(double a, double b, double epsilon = 0.001)
	{
		return std::abs(a - b) < epsilon;
	}

	void set_mesh_data(Eigen::MatrixXd &V, Eigen::MatrixXi &F)
	{
		vertices = V;
		triangles = F;
	}

	void cal_weights()
	{
		adjacent_vertices.resize(vertices.rows());
		e_weights.resize(vertices.rows(), vertices.rows());
		e_weights.setConstant(-1);
		Eigen::Matrix<int, 3, 3> edges;
		edges <<
			1, 2, 0,
			2, 0, 1,
			0, 1, 2;
		for (int i = 0; i < triangles.rows(); i++)
		{
			for (int e = 0; e<edges.rows(); e++)
			{
				int v0 = triangles(i, edges(e, 0)); // considering v0-v1 is the edge being considered
				int v1 = triangles(i, edges(e, 1));
				int v2 = triangles(i, edges(e, 2));
				double res = get_cot(v0, v2, v1, vertices);

				if (e_weights(v0, v1) == -1)
				{
					e_weights(v0, v1) = (res / 2.0);
					e_weights(v1, v0) = (res / 2.0);
				}
				else
				{
					e_weights(v0, v1) = (((e_weights(v0, v1) * 2.0) + res) / 2.0);
					e_weights(v1, v0) = (((e_weights(v1, v0) * 2.0) + res) / 2.0);
				}
				if (std::find(adjacent_vertices[v0].begin(), adjacent_vertices[v0].end(), v1) == adjacent_vertices[v0].end())
				{
					adjacent_vertices[v0].push_back(v1);
					adjacent_vertices[v1].push_back(v0);
				}
			}
		}
	}
	
private:
	// Using Cotangent formula from: http://people.eecs.berkeley.edu/~jrs/meshpapers/MeyerDesbrunSchroderBarr.pdf
	// Cot = cos/sin ==> using langrange's identity ==> a.b/sqrt(a^2 * b^2 - (a.b)^2)
	double get_cot(int v0, int v1, int v2, Eigen::MatrixXd &V)
	{
		typedef Eigen::Vector3d Vector;
		Vector a(V(v0, 0) - V(v1, 0), V(v0, 1) - V(v1, 1), V(v0, 2) - V(v1, 2));
		Vector b(V(v2, 0) - V(v1, 0), V(v2, 1) - V(v1, 1), V(v2, 2) - V(v1, 2));
		double dot_ab = a.dot(b);
		Vector cross_ab = a.cross(b);
		double divider = cross_ab.norm(); 

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
			std::size_t vi_id = get_ros_id(ros[k]);
			std::vector<int> vec = adjacent_vertices[ros[k]];
			double wij = 0;
			double total_weight = 0;
			for (int j = 0; j<vec.size(); j++)
			{
				wij = e_weights(ros[k], vec[j]);
				std::size_t vj_id = get_ros_id(vec[j]);
				const Eigen::Vector3d& pij = Eigen::Vector3d(original[vi_id](0) - original[vj_id](0), original[vi_id](1) - original[vj_id](1), original[vi_id](2) - original[vj_id](2));
				const Eigen::Vector3d& qij = Eigen::Vector3d(solution[vi_id](0) - solution[vj_id](0), solution[vi_id](1) - solution[vj_id](1), solution[vi_id](2) - solution[vj_id](2));
				sum_of_energy += wij * (qij - rotation_matrix[vi_id] * pij).squaredNorm();
			}

		}
		return sum_of_energy;
	}

	void init_ids_and_rotations() // Assuming we are using whole mesh in ROI
	{
		if (is_ids_cal_done)
			return;
		is_ids_cal_done = true;
		ros.clear();
		ros.insert(ros.end(), roi.begin(), roi.end());
		ros_id_map.assign(vertices.rows(), (std::numeric_limits<std::size_t>::max)()); // assign max to represent invalid value
		for (std::size_t i = 0; i < roi.size(); i++)
		{
			get_ros_id(roi[i]) = i;
		}
		rotation_matrix.resize(ros.size());
		for (std::size_t i = 0; i < rotation_matrix.size(); i++)
		{
			std::size_t v_ros_id = get_ros_id(ros[i]);
			rotation_matrix[v_ros_id] = Eigen::Matrix3d().setIdentity();
		}
		prev_solution.resize(ros.size());
		solution.resize(ros.size());
		original.resize(ros.size());
		for (std::size_t i = 0; i < ros.size(); i++)
		{
			std::size_t v_ros_id = get_ros_id(ros[i]);
			prev_solution[v_ros_id] = vertices.row(ros[i]);
			solution[v_ros_id] = vertices.row(ros[i]);
			original[v_ros_id] = vertices.row(ros[i]);
		}
	}

	/// Construct matrix that corresponds to left-hand side of eq:lap_ber == LP' = b and 
	/// b == SumOfOneRing(Wij * ((Ri + Rj)/2) * (Vi-Vj)) and P'==Unknown, L==Weights
	void calculate_laplacian_and_factorize()
	{
		if (factorization_done)
			return;
		factorization_done = true;
		if (weighting_thread->joinable())
			weighting_thread->join();
		size_t siz = ros.size();
		A = new Eigen::SparseMatrix<double>(static_cast<int>(siz), static_cast<int>(siz));
		tripletList.reserve(siz);
		std::vector<std::thread> threds;
		for (std::size_t k = 0; k < ros.size(); k++)
		{
			std::size_t vi_id = get_ros_id(ros[k]);
			if (is_roi_vertex(ros[k]) && !is_control_vertex(ros[k]))
			{
				double diagonal_val = 0;
				std::vector<int> vec = adjacent_vertices[ros[k]];
				double wij = 0;
				double total_weight = 0;
				for (int j = 0; j<vec.size(); j++)
				{
					wij = e_weights(ros[k], vec[j]);
					total_weight = wij + wij; // As wij == wji
					tripletList.push_back(Eigen::Triplet<double, int>(vi_id, get_ros_id(vec[j]), -total_weight));
					diagonal_val += total_weight;
				}
				tripletList.push_back(Eigen::Triplet<double, int>(vi_id, vi_id, diagonal_val));
			}
			else
			{
				tripletList.push_back(Eigen::Triplet<double, int>(vi_id, vi_id, 1.0));
			}
		}
		A->setFromTriplets(tripletList.begin(), tripletList.end());
		tripletList.clear();
		double D;
		A->makeCompressed();
		tmp = A;
		linear_solver.compute(*tmp);
		preprocess_successful = (linear_solver.info() == Eigen::Success);
	}

	void compute_rotation_subset(int index_from, int index_to) {
		Timer tmr;
		Eigen::Matrix3d cov = Eigen::Matrix3d().setZero();
		std::vector<std::thread> threds;
		for (std::size_t k = index_from; k <= index_to; k++)
		{
			std::size_t vi_id = get_ros_id(ros[k]);
			cov = Eigen::Matrix3d().setZero();
			std::vector<int> vec = adjacent_vertices[ros[k]];
			double wij = 0;
			double total_weight = 0;
			for (int j = 0; j<vec.size(); j++)
			{
				wij = e_weights(ros[k], vec[j]);
				std::size_t vj_id = get_ros_id(vec[j]);
				const Eigen::Vector3d& pij = Eigen::Vector3d(original[vi_id](0) - original[vj_id](0), original[vi_id](1) - original[vj_id](1), original[vi_id](2) - original[vj_id](2));
				const Eigen::Vector3d& qij = Eigen::Vector3d(solution[vi_id](0) - solution[vj_id](0), solution[vi_id](1) - solution[vj_id](1), solution[vi_id](2) - solution[vj_id](2));
				cov += wij * (pij * qij.transpose());
			}
			compute_close_rotation(cov, rotation_matrix[vi_id]);
		}
		for (auto& th : threds) {
			th.join();
		}
		avg_rotation_time += tmr.elapsed();
	}

	// Covariance matrix S = SumOverOneRing(wij * pij * qij.transpose)
	void calculate_optimal_rotations()
	{
		Eigen::Matrix3d cov = Eigen::Matrix3d().setZero();
		std::vector<std::thread> threds;
		unsigned int n = std::thread::hardware_concurrency();
		unsigned int subparts = ros.size()/n;
		int index = 0;
		if (subparts <= 0)
		{
			threds.push_back(std::thread(&Arap_mesh_deformation::compute_rotation_subset, this, index, ros.size()-1));
		}
		else
		{
			for (int k = 0; k < n; k += 1)
			{
				threds.push_back(std::thread(&Arap_mesh_deformation::compute_rotation_subset, this, index, index + subparts - 1));
				index += subparts;
			}
			
			if (ros.size() % n != 0)
			{
				threds.push_back(std::thread(&Arap_mesh_deformation::compute_rotation_subset, this, index, ros.size() - 1));
			}
		}
		for (auto& th : threds) {
			th.join();
		}
	}

	void calculate_target_positions(bool apply_acceleration)
	{
		Eigen::VectorXd X(ros.size()), Bx(ros.size());
		Eigen::VectorXd Y(ros.size()), By(ros.size());
		Eigen::VectorXd Z(ros.size()), Bz(ros.size());
		for (std::size_t k = 0; k < ros.size(); k++)
		{
			std::size_t vi_id = get_ros_id(ros[k]);
			if (is_roi_vertex(ros[k]) && !is_control_vertex(ros[k]))
			{
				Eigen::Vector3d xyz = Eigen::Vector3d(0, 0, 0);
				std::vector<int> vec = adjacent_vertices[ros[k]];
				double wij = 0;
				double total_weight = 0;
				for (int j = 0; j<vec.size(); j++)
				{
					wij = e_weights(ros[k], vec[j]);
					std::size_t vj_id = get_ros_id(vec[j]);
					const Eigen::Vector3d& pij = Eigen::Vector3d(original[vi_id](0) - original[vj_id](0), original[vi_id](1) - original[vj_id](1), original[vi_id](2) - original[vj_id](2));
					double wji = wij; // As wij == wji
					xyz += (wij * rotation_matrix[vi_id] + wji * rotation_matrix[vj_id]) * pij;
				}
				Bx[vi_id] = xyz(0);
				By[vi_id] = xyz(1);
				Bz[vi_id] = xyz(2);
			}
			else
			{		// control points's target positions are given by the user, so directly assign those values
				Bx[vi_id] = solution[vi_id](0);
				By[vi_id] = solution[vi_id](1);
				Bz[vi_id] = solution[vi_id](2);
			}
		}
		// Call solver for each dimension of the point seperately 
		Timer tmr;
		auto sol1 = std::async(std::launch::async, &Arap_mesh_deformation::solver, this, std::ref(Bx), std::ref(X));
		auto sol2 = std::async(std::launch::async, &Arap_mesh_deformation::solver, this, std::ref(By), std::ref(Y));
		auto sol3 = std::async(std::launch::async, &Arap_mesh_deformation::solver, this, std::ref(Bz), std::ref(Z));
		bool is_all_solved = sol1.get() && sol2.get() && sol3.get();
		avg_solver_time += tmr.elapsed();
		if (!is_all_solved) {
			// could not solve all
			return;
		}

		if (apply_acceleration)
		{
			X = X + (theta * (X - prev_X));
			Y = Y + (theta * (Y - prev_Y));
			Z = Z + (theta * (Z - prev_Z));
		}

		for (std::size_t i = 0; i < ros.size(); i++)
		{
			std::size_t v_id = get_ros_id(ros[i]);
			Eigen::Vector3d p(X[v_id], Y[v_id], Z[v_id]);
			if (!is_ctrl_map[ros[i]])
			{
				prev_solution[v_id] = solution[v_id];
				solution[v_id] = p;
			}
		}
		prev_X = X;
		prev_Y = Y;
		prev_Z = Z;
	}

	bool solver(Eigen::VectorXd& B, Eigen::VectorXd& X)
	{
		X = linear_solver.solve(B);
		return linear_solver.info() == Eigen::Success;
	}

	// get ros_id as L value
	std::size_t& get_ros_id(int vid)
	{
		return ros_id_map[vid];
	}

	// get ros_id as R value
	std::size_t  get_ros_id(int vid) const
	{
		return ros_id_map[vid];
	}

	// Calculate the closest rotation, using the equation : R = V*(identity_mtrix with last element == det(V*U.transpose()))*U.transpose()
	// det(V*U.transpose()) can be +1 or -1
	void compute_close_rotation(const Eigen::Matrix3d m, Eigen::Matrix3d& R)
	{
		bool done = compute_polar_decomposition(m, R);
		if (!done)
		{
			compute_rotation_svd(m, R);
		}
	}

	void compute_rotation_svd(const Eigen::Matrix3d& m, Eigen::Matrix3d& R)
	{
		Eigen::JacobiSVD<Eigen::Matrix3d> solver;
		solver.compute(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::Matrix3d& u = solver.matrixU(); const Eigen::Matrix3d& v = solver.matrixV();
		R = v * u.transpose();
		// If determinanat of rotation is < 0 , then multiply last column of U with -1 and calculate rotation again
		if (R.determinant() < 0) {
			Eigen::Matrix3d u_copy = u;
			u_copy.col(2) *= -1;
			R = v * u_copy.transpose();
		}

	}

	bool compute_polar_decomposition(const Eigen::Matrix3d& A, Eigen::Matrix3d& R)
	{
		if (A.determinant() < 0)
		{
			return false;
		}
		typedef Eigen::Matrix3d::Scalar Scalar;
		const Scalar th = std::sqrt(Eigen::NumTraits<Scalar>::dummy_precision());
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
		eig.computeDirect(A.transpose()*A);
		if (eig.eigenvalues()(0) / eig.eigenvalues()(2)<th)
		{
			return false;
		}
		Eigen::Vector3d S = eig.eigenvalues().cwiseSqrt();
		R = A  * eig.eigenvectors() * S.asDiagonal().inverse() * eig.eigenvectors().transpose();
		if (std::abs(R.squaredNorm() - 3.) > th || R.determinant() < 0)
		{
			return false;
		}
		R.transposeInPlace(); // the optimal rotation matrix should be transpose of decomposition result
		return true;
	}

};
#endif  // ARAP_MESH_DEFORMATION_H
