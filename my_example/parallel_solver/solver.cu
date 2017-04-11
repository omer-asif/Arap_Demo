#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
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

struct isMinusOne{
	__host__ __device__
	bool operator()(const int x)
	{
		return x==-1;
	}
};
__global__
void dev_find_roots(int *col_indices, int *row_indices,int nnz, int rows, bool *is_processed, int *rRoot){
	int row_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(row_id < rows){
		if(is_processed[row_id] == false){
			int strt = row_indices[row_id];
			int end = 0;
			if((row_id+1) == rows){
				end = nnz;
			}
			else{
				end = row_indices[row_id+1];
			}
			bool flag=false;
			for(int j=strt;j<end;j++){
				if(row_id != col_indices[j]){
					if(is_processed[col_indices[j]] != true){
						flag=true;
						break;
					}
				}
			}
			if(flag == false){
				rRoot[row_id] = row_id;
			}
		}
	}
}
__global__
void dev_analyze(int root_size, int row_val_size, int *candidateRoots, bool *is_processed, int *rowValues){
	int root_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(root_id < root_size){
		rowValues[(row_val_size+root_id)] = candidateRoots[root_id]; // Sort rows within a level, so sort roots
		is_processed[candidateRoots[root_id]] = true;
	}
}
__global__
void dev_solve_level(float *vals, int *col_indices, int *row_indices, int nnz, int rows, int *rowValues, int strt, int end, float *X, float *B, bool upper){
	int row_indx = blockIdx.x * blockDim.x + threadIdx.x;
	float v=0.0f;
	if(row_indx < end){
		int row_id = rowValues[row_indx];
		int _strt = row_indices[row_id];
		int _end = 0;
		if((row_id+1) == rows){
			_end = nnz;
		}
		else{
			_end = row_indices[row_id+1];
		}
		for(int j=_strt;j<(_end);j++){
			if(row_id != col_indices[j]){
				v = v + (vals[j]*X[col_indices[j]]);
			}
		}
		if(upper == true){
			X[row_id] = (B[row_id]-v)/vals[_strt];
		}
		else{
			X[row_id] = (B[row_id]-v)/vals[(_end-1)];
		}
	}
}
void perform_analysis(int *dev_row_indices, int *dev_col_indices, int *dev_row_values, thrust::host_vector<int> &levelIndices, int rows, int nnz){
	bool *dev_is_processed;
	int *dev_candidates;
	thrust::device_vector<int> d_candidates;
	d_candidates.resize(rows, -1);
	dev_candidates = thrust::raw_pointer_cast(d_candidates.data());
	thrust::host_vector<bool> isProcessed;
	isProcessed.resize(rows, false);
	thrust::device_vector<bool> d_processed = isProcessed;
	dev_is_processed = thrust::raw_pointer_cast(d_processed.data());
	int blocks = (rows/256)+1;
	int threads = 256;
	dev_find_roots<<<blocks, threads>>>(dev_col_indices, dev_row_indices, nnz, rows, dev_is_processed, dev_candidates);
	int rowsProcessed = 0;
	int currLevel = 0;
	thrust::device_vector<int>::iterator new_end;
	int root_size;
	while (rowsProcessed < rows) {
		new_end = thrust::remove_if(d_candidates.begin(), d_candidates.end(), isMinusOne());
		root_size = new_end - d_candidates.begin();
		levelIndices.push_back(rowsProcessed);
		dev_analyze<<<1, root_size>>>(root_size, rowsProcessed, dev_candidates, dev_is_processed, dev_row_values);
		thrust::fill(d_candidates.begin(), d_candidates.end(), (int)-1);// only fill until new_end/root_size
		dev_find_roots<<<blocks, threads>>>(dev_col_indices, dev_row_indices, nnz, rows, dev_is_processed, dev_candidates);
		currLevel++;
		rowsProcessed+=root_size;
	}
}

void setup_sparse(int rows, int cols, int &nnz, bool upper){
	float *mat = (float *)calloc(rows*cols, sizeof(float));
	float d=1.0f;
	nnz=0;
	int t = 3*rows;
	for (int i = 0; i < t;) {
	   int index = (int) ((rows*cols) * ((double) rand() / (RAND_MAX + 1.0)));
	   if (mat[index]) {
		  continue;
	   }
	   mat[index] = i % 2 ? 2 : 4;
	   ++i;
	   nnz++;
	}
	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<cols; j++) {
			if (i == j){
				if(!(mat[j+i*cols]==2 || mat[j+i*cols]==4)){
					mat[j+i*cols] = d++;
					nnz++;
				}
			}
			if(upper){
				if(i>j){
					if(mat[j+i*cols]==2 || mat[j+i*cols]==4){
						nnz--;
					}
					mat[j+i*cols] = 0.0f;
				}
			}
			else{
				if(i<j){
					if(mat[j+i*cols]==2 || mat[j+i*cols]==4){
						nnz--;
					}
					mat[j+i*cols] = 0.0f;
				}
			}
		}
	}
  
	float *vals = (float *)malloc(sizeof(float)*nnz);
	int *col_indices = (int *)malloc(sizeof(int)*nnz);
	int *row_indices = (int *)malloc(sizeof(int)*(rows+1));
	int col_count=0;
	int row_count=0;
	for(int i=0;i<rows;i++){
		row_indices[row_count++] = col_count;
		for(int j=0;j<cols;j++){
			if(mat[j+i*cols] != 0.0f){
				vals[col_count] = mat[j+i*cols];
				col_indices[col_count] = j;
				col_count++;
			}
		}
	}
	if(upper){
		cudaMalloc((void **)&dev_vals_upper, nnz * sizeof(float));
		cudaMalloc((void **)&dev_col_indices_upper, nnz * sizeof(int));
		cudaMalloc((void **)&dev_row_indices_upper, rows * sizeof(int));
		cudaMemcpy(dev_vals_upper, vals, nnz * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_col_indices_upper, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_row_indices_upper, row_indices, rows * sizeof(int), cudaMemcpyHostToDevice);
	}
	else{
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
void solve_X(thrust::host_vector<int>& levelIndices, thrust::host_vector<int>& rowValues, int nnz, int rows, float *dev_B, float *dev_X, bool upper){
	for (int k = 0; k < levelIndices.size(); k++) {
		int strt = levelIndices[k];
		int end = 0;
		if (k == (levelIndices.size() - 1)) {
			end = rowValues.size();
		}
		else {
			end = levelIndices[k + 1];
		}
		int threds = 32;
		if((end-strt)>32)
			threds = (end-strt);
		if(upper)
			dev_solve_level<<<1,threds>>>(dev_vals_upper, dev_col_indices_upper, dev_row_indices_upper, nnz, rows, dev_row_values_upper, strt, end, dev_X, dev_B, true);
		else
			dev_solve_level<<<1,threds>>>(dev_vals_lower, dev_col_indices_lower, dev_row_indices_lower, nnz, rows, dev_row_values_lower, strt, end, dev_X, dev_B, false);
	}
}

extern "C" void solve(){
	int rows=40000;
	int cols=40000;
	int nnz_lower=0, nnz_upper=0;;
	setup_sparse(rows, cols, nnz_lower, false);
	setup_sparse(rows, cols, nnz_upper, true);
  
	thrust::host_vector<int> levelIndices_lower;
	thrust::host_vector<int> rowValues_lower(rows);
	thrust::device_vector<int> d_rowValues_lower = rowValues_lower;
	dev_row_values_lower = thrust::raw_pointer_cast(d_rowValues_lower.data());
	perform_analysis(dev_row_indices_lower, dev_col_indices_lower, dev_row_values_lower, levelIndices_lower, rows, nnz_lower);
	rowValues_lower = d_rowValues_lower;

	thrust::host_vector<int> levelIndices_upper;
	thrust::host_vector<int> rowValues_upper(rows);
	thrust::device_vector<int> d_rowValues_upper = rowValues_upper;
	dev_row_values_upper = thrust::raw_pointer_cast(d_rowValues_upper.data());

	perform_analysis(dev_row_indices_upper, dev_col_indices_upper, dev_row_values_upper, levelIndices_upper, rows, nnz_upper);
	rowValues_upper = d_rowValues_upper;

	thrust::host_vector<float> X(rows);
	thrust::device_vector<float> d_X = X;
	dev_X_lower = thrust::raw_pointer_cast(d_X.data());
	thrust::host_vector<float> B;
	B.resize(rows, 1.0f);
	thrust::device_vector<float> d_B = B;
	dev_B_lower = thrust::raw_pointer_cast(d_B.data());

	thrust::host_vector<float> X_upper(rows);
	thrust::device_vector<float> d_X_upper = X_upper;
	dev_X_upper = thrust::raw_pointer_cast(d_X_upper.data());

	// Solve Lower
	solve_X( levelIndices_lower, rowValues_lower, nnz_lower, rows, dev_B_lower, dev_X_lower, false);
	// Solve Upper
	solve_X( levelIndices_upper, rowValues_upper, nnz_upper, rows, dev_X_lower, dev_X_upper, true);
	X_upper = d_X_upper; // copy back result

}
