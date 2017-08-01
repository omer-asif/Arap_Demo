/*
A Sparse triangular solver based on the paper

'Parallel Solution of Sparse Triangular Linear Systems in the Preconditioned Iterative Methods on GPU'

*/
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

int *dev_row_values_lower;
int *dev_row_values_upper;

float *dev_vals_lower;
int *dev_col_indices_lower;
int *dev_row_indices_lower;

float *dev_vals_upper;
int *dev_col_indices_upper;
int *dev_row_indices_upper;

float *dev_B_lower;
float *dev_X_lower;
float *dev_B_upper;
float *dev_X_upper;

/*
Function object to check if given integer is equal to -1
*/
struct isMinusOne {
	__host__ __device__
	bool operator()(const int x)
	{
		return x == -1;
	}
};

/*
CUDA Kernel to find the set of candidate rows which are not dependent on each other 
and their dependency on other rows is already accounted for, by previous round(s) of analysis.
It receives the matrix in CRS format and checks each row for dependencies, except the ones already processed.
*/
__global__
void dev_find_roots(int *col_indices, int *row_indices, int nnz, int rows, bool *is_processed, int *candidate_roots) {
	int row_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (row_id < rows) {
		if (is_processed[row_id] == false) {
			int strt = row_indices[row_id];
			int end = 0;
			if ((row_id + 1) == rows) {
				end = nnz;
			}
			else {
				end = row_indices[row_id + 1];
			}
			bool flag = false;
			for (int j = strt; j<end; j++) {
				if (row_id != col_indices[j]) {
					if (is_processed[col_indices[j]] != true) {
						flag = true;
						break;
					}
				}
			}
			if (flag == false) {
				candidate_roots[row_id] = row_id;
			}
		}
	}
}

/*
CUDA Kernel to assign the candidate rows found in the analysis phase, to the row_values array.
*/
__global__
void dev_analyze(int root_size, int row_val_size, int *candidate_roots, bool *is_processed, int *row_values) {
	int root_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (root_id < root_size) {
		row_values[(row_val_size + root_id)] = candidate_roots[root_id];
		is_processed[candidate_roots[root_id]] = true;
	}
}

/*
CUDA Kernel to solve a particular level, which represents a set of rows that can be solved in parallel.
*/
__global__
void dev_solve_level(float *vals, int *col_indices, int *row_indices, int nnz, int rows, int *row_values, int strt, int end, float *X, float *B, bool is_upper) {
	int row_indx = blockIdx.x * blockDim.x + threadIdx.x;
	float v = 0.0f;
	if (row_indx < end) {
		int row_id = row_values[row_indx];
		int _strt = row_indices[row_id];
		int _end = 0;
		if ((row_id + 1) == rows) {
			_end = nnz;
		}
		else {
			_end = row_indices[row_id + 1];
		}
		for (int j = _strt; j<(_end); j++) {
			if (row_id != col_indices[j]) {
				v = v + (vals[j] * X[col_indices[j]]);
			}
		}
		if (is_upper == true) {
			X[row_id] = (B[row_id] - v) / vals[_strt];
		}
		else {
			X[row_id] = (B[row_id] - v) / vals[(_end - 1)];
		}
	}
}

/*
C function to perform dependency analysis on rows of the matrix, to divide them into sets of Levels.
Each Level represents the set of rows that can be solved in parallel.
Levels have to be solved in sequence of their dependency
*/
void perform_analysis(int *dev_row_indices, int *dev_col_indices, int *dev_row_values, thrust::host_vector<int> &level_indices, int rows, int nnz) {
	bool *dev_is_processed;
	int *dev_candidates;
	thrust::device_vector<int> d_candidates;
	d_candidates.resize(rows, -1);
	dev_candidates = thrust::raw_pointer_cast(d_candidates.data());
	thrust::host_vector<bool> is_processed;
	is_processed.resize(rows, false);
	thrust::device_vector<bool> d_processed = is_processed;
	dev_is_processed = thrust::raw_pointer_cast(d_processed.data());
	int blocks = (rows / 256) + 1;
	int threads = 256;
	// find initial roots(rows) that do not depend on any other row, hence can be solved in parallel
	dev_find_roots << <blocks, threads >> >(dev_col_indices, dev_row_indices, nnz, rows, dev_is_processed, dev_candidates);
	int rows_processed = 0;
	int curr_level = 0;
	thrust::device_vector<int>::iterator new_end;
	int root_size = 0;
	while (rows_processed < rows) {
		new_end = thrust::remove_if(d_candidates.begin(), d_candidates.end(), isMinusOne());
		root_size = new_end - d_candidates.begin();
		level_indices.push_back(rows_processed);
		dev_analyze <<<1, root_size >>>(root_size, rows_processed, dev_candidates, dev_is_processed, dev_row_values);
		thrust::fill(d_candidates.begin(), d_candidates.end(), (int)-1);
		// find the currently possible candidate rows that are not dependent on any row still not processed.
		dev_find_roots <<<blocks, threads >>>(dev_col_indices, dev_row_indices, nnz, rows, dev_is_processed, dev_candidates);
		curr_level++;
		rows_processed += root_size;
	}
}

/*
Setup a random sparse lower/upper triangular matrix
*/
void setup_sparse(int rows, int cols, int &nnz, bool is_upper) {
	float *mat = (float *)calloc(rows*cols, sizeof(float));
	float d = 1.0f;
	nnz = 0;
	int t = 3 * rows;
	// Randomly assign values to the mat array, to generate a random sparse matrix
	for (int i = 0; i < t;) {
		int index = (int)((rows*cols) * ((double)rand() / (RAND_MAX + 1.0)));
		if (mat[index]) {
			continue;
		}
		mat[index] = i % 2 ? 2 : 4;
		++i;
		nnz++;
	}
	// Make the matrix upper or lower triangular depending on the parameter is_upper, by setting non-triangular elements to zero.
	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<cols; j++) {
			if (i == j) {
				if (!(mat[j + i*cols] == 2 || mat[j + i*cols] == 4)) {
					mat[j + i*cols] = d++;
					nnz++;
				}
			}
			if (is_upper) {
				// incase of upper triangular matrix, set the elements in lower triangular part of matrix to zero
				if (i>j) {
					if (mat[j + i*cols] == 2 || mat[j + i*cols] == 4) {
						nnz--;
					}
					mat[j + i*cols] = 0.0f;
				}
			}
			else {
				// incase of lower triangular matrix, set the elements in upper triangular part of matrix to zero
				if (i<j) {
					if (mat[j + i*cols] == 2 || mat[j + i*cols] == 4) {
						nnz--;
					}
					mat[j + i*cols] = 0.0f;
				}
			}
		}
	}

	// Convert the sparse matrix to Compressed Row Storage format.
	float *vals = (float *)malloc(sizeof(float)*nnz);
	int *col_indices = (int *)malloc(sizeof(int)*nnz);
	int *row_indices = (int *)malloc(sizeof(int)*(rows + 1));
	int col_count = 0;
	int row_count = 0;
	for (int i = 0; i<rows; i++) {
		row_indices[row_count++] = col_count;
		for (int j = 0; j<cols; j++) {
			if (mat[j + i*cols] != 0.0f) {
				vals[col_count] = mat[j + i*cols];
				col_indices[col_count] = j;
				col_count++;
			}
		}
	}
	if (is_upper) {
		cudaMalloc((void **)&dev_vals_upper, nnz * sizeof(float));
		cudaMalloc((void **)&dev_col_indices_upper, nnz * sizeof(int));
		cudaMalloc((void **)&dev_row_indices_upper, rows * sizeof(int));
		cudaMemcpy(dev_vals_upper, vals, nnz * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_col_indices_upper, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_row_indices_upper, row_indices, rows * sizeof(int), cudaMemcpyHostToDevice);
	}
	else {
		cudaMalloc((void **)&dev_vals_lower, nnz * sizeof(float));
		cudaMalloc((void **)&dev_col_indices_lower, nnz * sizeof(int));
		cudaMalloc((void **)&dev_row_indices_lower, rows * sizeof(int));
		cudaMemcpy(dev_vals_lower, vals, nnz * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_col_indices_lower, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_row_indices_lower, row_indices, rows * sizeof(int), cudaMemcpyHostToDevice);
	}

	free(vals);
	free(col_indices);
	free(row_indices);
	free(mat);
}

/*
Solve for values of X each level at a time because levels represent dependencies that have to be solved in sequence.
*/
void solve_X(thrust::host_vector<int>& level_indices, thrust::host_vector<int>& row_values, int nnz, int rows, float *dev_B, float *dev_X, bool upper) {
	for (int k = 0; k < level_indices.size(); k++) {
		int strt = level_indices[k];
		int end = 0;
		if (k == (level_indices.size() - 1)) {
			end = row_values.size();
		}
		else {
			end = level_indices[k + 1];
		}
		int threds = 32; // 32 is the minimum number of threads in a block
		if ((end - strt)>32)
			threds = (end - strt);
		if (upper){
			dev_solve_level <<<1, threds >>> (dev_vals_upper, dev_col_indices_upper, dev_row_indices_upper, nnz, rows, dev_row_values_upper, strt, end, dev_X, dev_B, true);
		}
		else {
			dev_solve_level <<<1, threds >>> (dev_vals_lower, dev_col_indices_lower, dev_row_indices_lower, nnz, rows, dev_row_values_lower, strt, end, dev_X, dev_B, false);
		}
	}
}
/*
Perform a triangular back substitution solve with parallel solver technique for a sample sparse matrix
*/
extern "C" void solve() {
	int rows = 40000;
	int cols = 40000;
	int nnz_lower = 0, nnz_upper = 0;
	// Setup random Upper and Lower Triangular matrices
	setup_sparse(rows, cols, nnz_lower, false);
	setup_sparse(rows, cols, nnz_upper, true);

	// Perform analysis for upper triangular matrix
	thrust::host_vector<int> level_indices_lower;
	thrust::host_vector<int> row_values_lower(rows);
	thrust::device_vector<int> d_row_values_lower = row_values_lower;
	dev_row_values_lower = thrust::raw_pointer_cast(d_row_values_lower.data());
	perform_analysis(dev_row_indices_lower, dev_col_indices_lower, dev_row_values_lower, level_indices_lower, rows, nnz_lower);
	row_values_lower = d_row_values_lower;// copy values from GPU to CPU

	// Perform analysis for upper triangular matrix
	thrust::host_vector<int> level_indices_upper;
	thrust::host_vector<int> row_values_upper(rows);
	thrust::device_vector<int> d_row_values_upper = row_values_upper;
	dev_row_values_upper = thrust::raw_pointer_cast(d_row_values_upper.data());

	perform_analysis(dev_row_indices_upper, dev_col_indices_upper, dev_row_values_upper, level_indices_upper, rows, nnz_upper);
	row_values_upper = d_row_values_upper;

	thrust::host_vector<float> X_lower(rows);
	thrust::device_vector<float> d_X_lower = X_lower;
	dev_X_lower = thrust::raw_pointer_cast(d_X_lower.data());
	thrust::host_vector<float> B;
	B.resize(rows, 1.0f);
	thrust::device_vector<float> d_B = B;
	dev_B_lower = thrust::raw_pointer_cast(d_B.data());

	// Solve lower triangular matrix for X vector
	solve_X(level_indices_lower, row_values_lower, nnz_lower, rows, dev_B_lower, dev_X_lower, false);

	thrust::host_vector<float> X_upper(rows);
	thrust::device_vector<float> d_X_upper = X_upper;
	dev_X_upper = thrust::raw_pointer_cast(d_X_upper.data());

	// Solve upper triangular matrix for X vector
	solve_X(level_indices_upper, row_values_upper, nnz_upper, rows, dev_X_lower, dev_X_upper, true);
	X_upper = d_X_upper; // copy results from GPU to CPU

}
