/*
As Rigid As Possible(ARAP) Mesh deformation algorithm implementation using MKL Pardiso solver.
*/

#ifndef ARAP_MESH_DEFORMATION_H
#define ARAP_MESH_DEFORMATION_H

#include <iostream>
#include <stdlib.h> 
#include <thread>
#include <future>
#include <mutex>
#include <vector>
#include <map>
#include <list>
#include <utility>
#include <limits>
#include <omp.h>
#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <Eigen/SparseLU>
#include <cmath>
#include "mkl_pardiso.h"
#include "mkl_types.h"


class Arap_mesh_deformation {

public:
	double avg_solver_time;
	double avg_rotation_time;
private:
	Eigen::MatrixXd& vertices;
	Eigen::MatrixXi& triangles;
	Eigen::VectorXd prev_X;
	Eigen::VectorXd prev_Y;
	Eigen::VectorXd prev_Z;

	std::vector<bool> is_ctrl_map;
	std::vector<std::vector<int>> adjacent_vertices;
	std::vector<int> control_vertices;
	std::vector<int> sorted_control_vertices;
	std::vector<Eigen::Matrix3d> rotation_matrix;
	std::vector<float> original_x;
	std::vector<float> original_y;
	std::vector<float> original_z;
	std::vector<float> solution_x;
	std::vector<float> solution_y;
	std::vector<float> solution_z;
	std::map<int, Eigen::VectorXf> control_columns;
	std::thread *weighting_thread;
	Eigen::SparseMatrix<float,Eigen::RowMajor> e_weights;
	bool preprocess_successful;
	bool is_ids_cal_done;
	bool factorization_done;
	double theta;

	Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT> *A;
	double *a;
	MKL_INT *ja;
	MKL_INT *ia;
	MKL_INT n;//rows;
	MKL_INT mtype;       // Real symmetric matrix //
	MKL_INT nrhs;     // Number of right hand sides. //
	// Internal solver memory pointer pt, //
	void *pt[64];
	// Pardiso control parameters. //
	MKL_INT iparm[64];
	MKL_INT maxfct, mnum, phase, error, msglvl;
	// Auxiliary variables. //
	MKL_INT i;
	double ddum;          // Double dummy //
	MKL_INT idum;         // Integer dummy. //

public:

	Arap_mesh_deformation(Eigen::MatrixXd &V, Eigen::MatrixXi &F) : vertices(V), triangles(F) {
		Eigen::initParallel();
		is_ctrl_map = std::vector<bool>(vertices.rows(), false);
		preprocess_successful = false;
		is_ids_cal_done = false;
		factorization_done = false;
		avg_rotation_time = 0.0;
		avg_solver_time = 0.0;
		theta = 0.81;
		weighting_thread = new std::thread(&Arap_mesh_deformation::cal_weights, this);

	}

	bool insert_control_vertex(int vid) {
		// factorization needs to be done again once this function is called.
		if (is_control_vertex(vid)) {
			return false;
		}
		if (factorization_done == true) {
			factorization_done = false;
			// Termination and release of memory.
			phase = -1;
			PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, &ddum, ia, ja, &idum, &nrhs,
					 iparm, &msglvl, &ddum, &ddum, &error);

			free(a);
			free(ja);
			free(ia);
		}
		control_vertices.push_back(vid);
		is_ctrl_map[vid] = true;
		return true;
	}

	bool preprocess() {
		init_ids_and_rotations();
		calculate_laplacian_and_factorize();
		return preprocess_successful;
	}

	void set_target_position(int vid, const Eigen::Vector3d& target_position) {
		// ids should have been assigned before calling this function
		init_ids_and_rotations();
		if (!is_control_vertex(vid)) {
			return;
		}
		solution_x[vid] = target_position(0);
		solution_y[vid] = target_position(1);
		solution_z[vid] = target_position(2);
	}

	void deform(unsigned int iterations, double tolerance)
	{
		if (!preprocess_successful) {
			return;
		}
		avg_rotation_time = 0;
		avg_solver_time = 0;
		double energy_this = 0;
		double energy_last;
		int itrs = 0;
		for (unsigned int ite = 0; ite < iterations; ++ite) {
			itrs += 1;
			calculate_optimal_rotations();
			if (ite != 0)
				calculate_target_positions(true); // apply acceleration
			else
				calculate_target_positions(false);

			if (tolerance > 0.0 && (ite + 1) < iterations) {
				energy_last = energy_this;
				energy_this = compute_energy();
				if (ite != 0) {
					double energy_dif = std::abs((energy_last - energy_this) / energy_this);
					if (energy_dif < tolerance) {
						break;
					}
				}
			}
		}
	}

	bool is_control_vertex(int vid) const
	{
		return is_ctrl_map[vid];
	}

	void reset() {
		avg_rotation_time = 0.0;
		avg_solver_time = 0.0;
	}

	bool double_equals(double a, double b, double epsilon = 0.001) {
		return std::abs(a - b) < epsilon;
	}

	void set_mesh_data(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
		vertices = V;
		triangles = F;
	}
	// calculate half edge weights
	void cal_weights() {
		int siz = vertices.rows();
		adjacent_vertices.resize(siz);
		e_weights = Eigen::SparseMatrix<float, Eigen::RowMajor>(siz, siz);
		e_weights.reserve(Eigen::VectorXi::Constant(siz, 15));// assuming at most 15 neighbors per vertex
		Eigen::Matrix<int, 3, 3> edges;
		edges <<
			1, 2, 0,
			2, 0, 1,
			0, 1, 2;
		for (int i = 0; i < triangles.rows(); i++) {
			for (int e = 0; e<edges.rows(); e++) {
				int v0 = triangles(i, edges(e, 0)); // considering v0-v1 is the edge being considered
				int v1 = triangles(i, edges(e, 1));
				int v2 = triangles(i, edges(e, 2));
				double res = get_cot(v0, v2, v1, vertices);
				if (e_weights.coeff(v0, v1) == 0) {
					e_weights.coeffRef(v0, v1) = (res / 2.0);
					e_weights.coeffRef(v1, v0) = (res / 2.0);
				}
				else {
					e_weights.coeffRef(v0, v1) = (((e_weights.coeff(v0, v1) * 2.0) + res) / 2.0);
					e_weights.coeffRef(v1, v0) = (((e_weights.coeff(v1, v0) * 2.0) + res) / 2.0);
				}
				if (std::find(adjacent_vertices[v0].begin(), adjacent_vertices[v0].end(), v1) == adjacent_vertices[v0].end()) {
					adjacent_vertices[v0].push_back(v1);
					adjacent_vertices[v1].push_back(v0);
				}
			}
		}
		e_weights.makeCompressed();
		
	}

private:

	// Using Cotangent formula from: http://people.eecs.berkeley.edu/~jrs/meshpapers/MeyerDesbrunSchroderBarr.pdf
	// Cot = cos/sin ==> using langrange's identity ==> a.b/sqrt(a^2 * b^2 - (a.b)^2)
	double get_cot(int v0, int v1, int v2, Eigen::MatrixXd &V) {
		typedef Eigen::Vector3d Vector;
		Vector a(V(v0, 0) - V(v1, 0), V(v0, 1) - V(v1, 1), V(v0, 2) - V(v1, 2));
		Vector b(V(v2, 0) - V(v1, 0), V(v2, 1) - V(v1, 1), V(v2, 2) - V(v1, 2));
		double dot_ab = a.dot(b);
		Vector cross_ab = a.cross(b);
		double divider = cross_ab.norm(); 
		if (divider == 0) {
			return 0.0;
		}
		return (std::max)(0.0, (dot_ab / divider));
	}

	double compute_energy() {
		double sum_of_energy = 0;
		for (std::size_t vi = 0; vi < vertices.rows(); vi++) {
			std::vector<int> vec = adjacent_vertices[vi];
			double wij = 0;
			double total_weight = 0;
			for (int j = 0; j<vec.size(); j++) {
				int vj = vec[j];
				wij = e_weights.coeff(vi, vj);
				const Eigen::Vector3d& pij = Eigen::Vector3d(original_x[vi] - original_x[vj], original_y[vi] - original_y[vj], original_z[vi] - original_z[vj]);
				const Eigen::Vector3d& qij = Eigen::Vector3d(solution_x[vi] - solution_x[vj], solution_y[vi] - solution_y[vj], solution_z[vi] - solution_z[vj]);
				sum_of_energy += wij * (qij - rotation_matrix[vi] * pij).squaredNorm();
			}
		}
		return sum_of_energy;
	}

	// initialize data structures
	void init_ids_and_rotations() {
		if (is_ids_cal_done)
			return;
		is_ids_cal_done = true;
		int ros_size = vertices.rows();
		rotation_matrix.resize(ros_size);

		for (int i = 0; i < ros_size; i++) {
			rotation_matrix[i] = Eigen::Matrix3d().setIdentity();
		}
		original_x.resize(ros_size);
		original_y.resize(ros_size);
		original_z.resize(ros_size);
		solution_x.resize(ros_size);
		solution_y.resize(ros_size);
		solution_z.resize(ros_size);

		for (int i = 0; i < ros_size; i++) {
			original_x[i] = vertices(i, 0);
			original_y[i] = vertices(i, 1);
			original_z[i] = vertices(i, 2);

			solution_x[i] = vertices(i, 0);
			solution_y[i] = vertices(i, 1);
			solution_z[i] = vertices(i, 2);
		}
	}

	int in_range(int vid) {
		for(int i=sorted_control_vertices.size()-1;i>=0;i--) {
			if(vid > sorted_control_vertices[i]) {
				return (i+1);
			}
		}
		return 0;
	}

	/// Construct matrix that corresponds to left-hand side of eq:lap_ber == LP' = b and 
	/// b == SumOfOneRing(Wij * ((Ri + Rj)/2) * (Vi-Vj)) and P'==Unknown, L==Weights
	void calculate_laplacian_and_factorize() {
		if (factorization_done)
			return;
		factorization_done = true;
		if (weighting_thread->joinable())
			weighting_thread->join();

		MKL_INT nnz=0;
		double d=1.0f;
		int siz=vertices.rows();
		int new_size = static_cast<int>((siz-control_vertices.size()));
		A = new Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT>(static_cast<int>(new_size), static_cast<int>(new_size));
		std::vector<Eigen::Triplet<double>> tripletList;
		tripletList.reserve(new_size*5);
		sorted_control_vertices = control_vertices;
		std::sort(sorted_control_vertices.begin(), sorted_control_vertices.end());
		for(int z=0;z<control_vertices.size();z++) {
			control_columns[control_vertices[z]] = Eigen::VectorXf::Zero(new_size);
		}
		MKL_INT row=0, col=0;
		float *weight_vals = e_weights.valuePtr();
		int *col_indices = e_weights.innerIndexPtr();
		int *row_ptrs = e_weights.outerIndexPtr();
		int nonzeros = e_weights.nonZeros();
		for (std::size_t vi = 0; vi < siz; vi++)
		{
			if(!(is_control_vertex(vi))) {
				col=0;
				double diagonal_val = 0;
				std::vector<int> vec = adjacent_vertices[vi];
				double wij = 0;
				double total_weight = 0;
				int strt = row_ptrs[vi]; // start index for adjacency list
				int end = (nonzeros) - 1; // last index for adjacency list
				if ((vi + 1) < siz) {
					end = row_ptrs[(vi + 1)] - 1;
				}

				for (; strt <=end; strt++) {
					int vj = col_indices[strt];
					wij = weight_vals[strt];
					total_weight = wij + wij; // As wij == wji
					if(!(is_control_vertex(vj))) {
						col = vj - in_range(vj);
						if(row<=col) {
							tripletList.push_back(Eigen::Triplet<double>(row, col, -total_weight));
						}
					}
					else {
						control_columns[vj][row] = -total_weight;
					}
					diagonal_val += total_weight;
				}
				tripletList.push_back(Eigen::Triplet<double>(row, row, diagonal_val));
				row++;
			}
		}
		A->setFromTriplets(tripletList.begin(), tripletList.end());
		A->makeCompressed();
		a = A->valuePtr();
		ja = A->innerIndexPtr();
		ia = A->outerIndexPtr();
		n = A->rows();	  //rows;
		mtype = 2;       // Real symmetric matrix
		nrhs = 3;       // Number of right hand sides.

		// Setup Pardiso control parameters.
		for ( i = 0; i < 64; i++ )
		{
			iparm[i] = 0;
		}
		iparm[0] = 1;         // No solver default
		iparm[1] = 2;         // Fill-in reordering from METIS;
		iparm[3] = 0;         // No iterative-direct algorithm
		iparm[4] = 0;         // No user fill-in reducing permutation
		iparm[5] = 0;         // Write solution into x
		iparm[6] = 0;         // Not in use
		iparm[7] = 2;         // Max numbers of iterative refinement steps
		iparm[8] = 0;         // Not in use
		iparm[9] = 13;        // Perturb the pivot elements with 1E-13
		iparm[10] = 1;        // Use nonsymmetric permutation and scaling MPS
		iparm[11] = 0;        // Not in use
		iparm[12] = 0;        // Maximum weighted matching algorithm is switched-off (default for symmetric).
		iparm[13] = 0;        // Output: Number of perturbed pivots
		iparm[14] = 0;        // Not in use
		iparm[15] = 0;        // Not in use
		iparm[16] = 0;        // Not in use
		iparm[17] = -1;       // Output: Number of nonzeros in the factor LU
		iparm[18] = -1;       // Output: Mflops for LU factorization
		iparm[19] = 0;        // Output: Numbers of CG Iterations
		iparm[34] = 1;		  // use c style indexing
		//iparm[59] = 1;		  // Let MKL choose in-core or out-core mode
		maxfct = 1;           // Maximum number of numerical factorizations.
		mnum = 1;         // Which factorization to use. //
		msglvl = 0;           // Print statistical information in file
		error = 0;            // Initialize error flag

		// .. Initialize the internal solver memory pointer. This is only //
		// necessary for the FIRST call of the PARDISO solver. //
		for ( i = 0; i < 64; i++ )
		{
			pt[i] = 0;
		}

		// .. Reordering and Symbolic Factorization. This step also allocates
		// all memory that is necessary for the factorization. //
		phase = 11;
		PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
				 &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
		if ( error != 0 )
		{
			printf ("\nERROR during symbolic factorization: %d", error);
			exit (1);
		}
		
		// Numerical factorization.
		phase = 22;
		PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
				 &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
		if ( error != 0 )
		{
			printf ("\nERROR during numerical factorization: %d", error);
			exit (2);
		}
		preprocess_successful = true;
	}

	void compute_rotation_subset_cpu(int index_from, int index_to) {
		Eigen::Matrix3d cov;
		for (int vi = index_from; vi <= index_to; vi++) {
			cov = Eigen::Matrix3d().setZero();
			std::vector<int> vec = adjacent_vertices[vi];
			double wij = 0;
			double total_weight = 0;
			for (int j = 0; j<vec.size(); j++) {
				int vj = vec[j];
				wij = e_weights.coeff(vi, vj);
				const Eigen::Vector3d& pij = Eigen::Vector3d(original_x[vi] - original_x[vj], original_y[vi] - original_y[vj], original_z[vi] - original_z[vj]);
				const Eigen::Vector3d& qij = Eigen::Vector3d(solution_x[vi] - solution_x[vj], solution_y[vi] - solution_y[vj], solution_z[vi] - solution_z[vj]);
				cov += wij * (pij * qij.transpose());
			}
			compute_rotation(cov, rotation_matrix[vi]);
		}
	}

	void calculate_optimal_rotations() {
		int siz = vertices.rows();
		Eigen::Matrix3d cov = Eigen::Matrix3d().setZero();
		std::vector<std::thread> threds;
		unsigned int n = std::thread::hardware_concurrency();
		unsigned int subparts = siz / n;
		int index = 0;
		//subparts = 0; // TEMP CHANGE
		if (subparts <= 0) {
			threds.push_back(std::thread(&Arap_mesh_deformation::compute_rotation_subset_cpu, this, index, siz - 1));
		}
		else {
			for (int k = 0; k < n; k += 1) {
				threds.push_back(std::thread(&Arap_mesh_deformation::compute_rotation_subset_cpu, this, index, index + subparts - 1));
				index += subparts;
			}
			if (siz % n != 0) {
				threds.push_back(std::thread(&Arap_mesh_deformation::compute_rotation_subset_cpu, this, index, siz - 1));
			}
		}
		for (auto& th : threds) {
			th.join();
		}
		
	}

	void calculate_target_positions(bool apply_acceleration) {
		calculate_target_positions_cpu(apply_acceleration);
	}

	void calculate_target_positions_cpu(bool apply_acceleration) {
		int v_count=0;
		int siz = vertices.rows();
		int a_siz = A->rows();
		float *weight_vals = e_weights.valuePtr();
		int *col_indices = e_weights.innerIndexPtr();
		int *row_ptrs = e_weights.outerIndexPtr();
		int nonzeros = e_weights.nonZeros();
		double *control_vals = (double *)calloc(3 * a_siz , sizeof(double));
		double *host_B = (double *)malloc(3 * a_siz * sizeof(double));
		double *host_X = (double *)malloc(3 * a_siz * sizeof(double));

		for(int i=0;i<control_vertices.size();i++) {
			Eigen::VectorXf vectr = control_columns[control_vertices[i]];
			for(int k=0;k<a_siz;k++){
				control_vals[k] += ((solution_x[control_vertices[i]])*vectr[k]);
				control_vals[k+a_siz] += ((solution_y[control_vertices[i]])*vectr[k]);
				control_vals[k+(a_siz*2)] += ((solution_z[control_vertices[i]])*vectr[k]);
			}
		}

		for (std::size_t vi = 0; vi < siz; vi++)
		{
			if(!(is_control_vertex(vi))) {
				Eigen::Vector3d xyz = Eigen::Vector3d(0, 0, 0);
				double wij = 0;
				double total_weight = 0;
				int strt = row_ptrs[vi]; // start index for adjacency list
				int end = (nonzeros) - 1; // last index for adjacency list
				if ((vi + 1) < siz) {
					end = row_ptrs[(vi + 1)] - 1;
				}
				for (; strt <=end; strt++) {
					int vj = col_indices[strt];
					wij = weight_vals[strt];
					const Eigen::Vector3d& pij = Eigen::Vector3d(original_x[vi] - original_x[vj], original_y[vi] - original_y[vj], original_z[vi] - original_z[vj]);
					double wji = wij;
					xyz += (wij * rotation_matrix[vi] + wji * rotation_matrix[vj]) * pij;

				}
				host_B[v_count] = xyz(0) - control_vals[v_count];
				host_B[v_count+a_siz] = xyz(1) - control_vals[v_count+a_siz];
				host_B[v_count+(a_siz*2)] = xyz(2) - control_vals[v_count+(a_siz*2)];
				v_count++;
			}
		}

		// .. Back substitution and iterative refinement.
		phase = 33;
		iparm[7] = 2;         // Max numbers of iterative refinement steps.
		
		PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
				 &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, host_B, host_X, &error);
		if ( error != 0 )
		{
			printf ("\nERROR during solution: %d", error);
			exit (3);
		}
		
		if (apply_acceleration) {
			/*X = X + (theta * (X - prev_X));
			Y = Y + (theta * (Y - prev_Y));
			Z = Z + (theta * (Z - prev_Z));*/
		}
		int indx=0;
		for (int vi = 0; vi < siz; vi++) {
			if (!is_ctrl_map[vi]) {
				solution_x[vi] = host_XX[indx];
				solution_y[vi] = host_XX[indx+a_siz];
				solution_z[vi] = host_XX[indx+(a_siz*2)];
				indx++;
			}
			vertices.row(vi) = Eigen::RowVector3d(solution_x[vi], solution_y[vi], solution_z[vi]);
		}
		//prev_X = X;
		//prev_Y = Y;
		//prev_Z = Z;

	}

	// Calculate the closest rotation, using the equation : R = V*(identity_mtrix with last element == det(V*U.transpose()))*U.transpose()
	// det(V*U.transpose()) can be +1 or -1
	void compute_rotation(const Eigen::Matrix3d m, Eigen::Matrix3d& R) {
		bool done = compute_polar_decomposition(m, R);
		if (!done) {
			compute_rotation_svd(m, R);
		}
	}

	void compute_rotation_svd(const Eigen::Matrix3d& m, Eigen::Matrix3d& R) {
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

	bool compute_polar_decomposition(const Eigen::Matrix3d& A, Eigen::Matrix3d& R) {
		if (A.determinant() < 0) {
			return false;
		}
		typedef Eigen::Matrix3d::Scalar Scalar;
		const Scalar th = std::sqrt(Eigen::NumTraits<Scalar>::dummy_precision());
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
		eig.computeDirect(A.transpose()*A);
		if (eig.eigenvalues()(0) / eig.eigenvalues()(2)<th) {
			return false;
		}
		Eigen::Vector3d S = eig.eigenvalues().cwiseSqrt();
		R = A  * eig.eigenvectors() * S.asDiagonal().inverse() * eig.eigenvectors().transpose();
		if (std::abs(R.squaredNorm() - 3.) > th || R.determinant() < 0) {
			return false;
		}
		R.transposeInPlace();
		return true;
	}
};
#endif  // ARAP_MESH_DEFORMATION_H
