//this file contains all the property related operations performed on the matrix
#pragma once


template<typename T>
__global__ void _non_zero_counter(T* d_arr,uint row,uint col,uint *counter){

	uint _row = threadIdx.x + blockIdx.x * blockDim.x;
	uint _col = threadIdx.y + blockIdx.y * blockDim.y;

	uint idx = _row * col + _col;

	if(d_arr[idx]!=0){
		atomicAdd(counter,1);
	}
}


template<typename T>
double density(T* h_arr,uint row,uint col,uint dim_x=32,uint dim_y=32){

	//creating space for device array
	T* d_arr;
	size_t size = row*col*sizeof(T);
	CHECK(cudaMalloc((T**)&d_arr,size));

	//copying the values to device array
	CHECK(cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice));

	size_t uint_size = sizeof(uint);

	//initializing counter variable
	uint* h_counter;
	h_counter = (uint*)malloc(uint_size):
	*h_counter = 0;

	uint* d_counter;
	CHECK(cudaMalloc((uint**)&d_counter,uint_size));
	
	CHECK(cudaMemcpy(d_counter,h_counter,uint_size,cudaMemcpyHostToDevice));

	dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);

	//calling the parallel funtion
	_non_zero_counter<<<grid,block>>>(d_arr,row,col,d_counter);

	//ensuring all the threads have completed there job
	__syncthreads();

	//copying the value to host counter
	CHECK(cudaMemcpy(h_counter,d_counter,uint_size,cudaMemcpyDeviceToHost));

	//calculating the density
	double _density = h_counter/((double)row*col);

	return _density;

}
template<typename T>
double sparisity(T* h_arr, uint row,uint col,uint dim_x=32,uint dim_y=32){

	double _sparisity = 1 - density(h_arr,row,col,dim_x,dim_y);

	return _sparisity;
}

template<typename T>
bool issparse(T* h_arr,uint row,uint col,double sparisity_threshold=0.45, uint dim_x=32,uint dim_y=32){

	double _sparisity = sparisity(h_arr,row,col,dim_x,dim_y);
	if( _sparisity < sparisity_threshold ){
		return false;
	}
	return true;
}

template<typename T>
__global__ void _sum_elements(T* d_arr,uint row,uint col,double summation){
	uint _row = threadIdx.x + blockIdx.x * blockDim.x;
	uint _col = threadIdx.y + blockIdx.y * blockDim.y;

	uint idx = _row * col + _col;

	//TODO __check atomicAdd works with double or not
	atomicAdd(summation,d_arr[idx]);
}

template<typename T>
void mean(T* h_arr, uint row, uint col, uint dim_x=32, uint dim_y=32){

	//creating space for device array
	T* d_arr;
	size_t size = row*col*sizeof(T);
	CHECK(cudaMalloc((T**)&d_arr,size));

	//copying the values to device array
	CHECK(cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice));

	size_t double_size = sizeof(double);

	//initializing sum variable
	double* h_sum;
	h_sum= (double*)malloc(double):
	*h_sum = 0;

	uint* d_sum;
	CHECK(cudaMalloc((double**)&d_sum,double_size));
	
	CHECK(cudaMemcpy(d_sum,h_sum,double_size,cudaMemcpyHostToDevice));

	dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);

	//calling the parallel funtion
	_sum_elements<<grid,block>>>(d_arr,row,col,d_sum);

	//ensuring all the threads have completed there job
	__syncthreads();

	//copying the value to host counter
	CHECK(cudaMemcpy(h_sum,d_sum,double_size,cudaMemcpyDeviceToHost));

	//calculating the density
	double _mean = h_sum/((double)row*col);

	return _mean;
}

template<typename T>
__global__ void _row_sum(T* d_arr,double *d_result,uint row,uint col){

	uint _row = threadIdx.x + blockIdx.x * blockDim.x;
	uint _col = threadIdx.y + blockIdx.y * blockDim.y;

	uint idx = _row * col + _col;

	//TODO __check atomicAdd works with double or not
	atomicAdd(d_result[_row],d_arr[idx]);

} 

template<typename T>
__global__ void _divider(double* d_result,uint size){
	uint idx = threadIdx.x;
	d[idx] = d[idx]/size; 
}

template<typename T>
void row_mean(T* h_arr,uint row,uint col,double *h_result,uint dim_x = 32, uint dim_y = 32){

	size_t size = row*col*sizeof(T);
	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));
	CHECK(cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice));

	size_t size_row = row*sizeof(double);
	double* d_result;
	CHECK(cudaMalloc((double**)d_result,size_row));
	memset(h_result, 0, size_row);
	CHECK(cudaMemcpy(d_result,h_result,size_row,cudaMemcpyHostToDevice));

	dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);
	//TODO
	_row_sum<<<grid,block>>>(d_arr,d_result,row,col);

	__syncthreads();
	//TODO  check for no. of bytes read at a time and accordingly modify grid value
	_divider<<<1,row>>>(d_result,row);

	__syncthreads();

	CHECK(cudaMemcpy(h_result,d_result,size_row,cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_result));
	CHECK(cudaFree(d_arr));
}



template<typename T>
__global__ void _col_sum(T* d_arr,double *d_result,uint row,uint col){

	uint _row = threadIdx.x + blockIdx.x * blockDim.x;
	uint _col = threadIdx.y + blockIdx.y * blockDim.y;

	uint idx = _row * col + _col;

	//TODO __check atomicAdd works with double or not
	atomicAdd(d_result[_col],d_arr[idx]);

} 

template<typename T>
void col_mean(T* h_arr,uint row,uint col,double *h_result,uint dim_x = 32, uint dim_y = 32){

	size_t size = row*col*sizeof(T);
	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));
	CHECK(cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice));

	size_t size_col = col*sizeof(double);
	double* d_result;
	CHECK(cudaMalloc((double**)d_result,size_col));
	memset(h_result, 0, size_col);
	CHECK(cudaMemcpy(d_result,h_result,size_col,cudaMemcpyHostToDevice));

	dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);
	//TODO
	_col_sum<<<grid,block>>>(d_arr,d_result,row,col);


	__syncthreads();
	
	//TODO  check for no. of bytes read at a time and accordingly modify grid value
	_divider<<<1,col>>>(d_result,col);

	__syncthreads();

	CHECK(cudaMemcpy(h_result,d_result,size_col,cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_result));
	CHECK(cudaFree(d_arr));
}





enum orientation{
	rowWise,
	colWise 
};

template<typename T>
void mean_vector(T* h_arr,uint row,uint col,double* h_result ,enum orientation _orientation=rowWise,uint dim_x = 32,dim_y = 32){

		if(_orientation == rowWise){
			row_mean(h_arr,row,col,h_result,dim_x,dim_y);
		}
		else{
			col_mean(h_arr,row,col,h_result,dim_x,dim_y);
		}
}

void std()

void issymmetric()

void isdiagonal()

void iseye()

template<typename T>
void iszero()

// AtomicMin(),AtomicXOR()