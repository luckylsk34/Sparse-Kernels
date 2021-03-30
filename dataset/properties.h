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
	CHECK(cudaMalloc((**T)&d_arr,size));

	//copying the values to device array
	CHECK(cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice));

	size_t uint_size = sizeof(uint);

	//initializing counter variable
	uint* h_counter;
	h_counter = (uint*)malloc(uint_size):
	*h_counter = 0;

	uint* d_counter;
	CHECK(cudaMalloc((**uint)&d_counter,uint_size));
	
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
	CHECK(cudaMalloc((**T)&d_arr,size));

	//copying the values to device array
	CHECK(cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice));

	size_t double_size = sizeof(double);

	//initializing sum variable
	double* h_sum;
	h_sum= (double*)malloc(double):
	*h_sum = 0;

	uint* d_sum;
	CHECK(cudaMalloc((**double)&d_sum,double_size));
	
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

void mean_vector()

void std()

void median()

void issymmetric()

void isdiagonal()

void iseye()

void iszero()

// AtomicMin(),AtomicXOR()