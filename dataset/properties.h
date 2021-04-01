//this file contains all the property related operations performed on the matrix
#pragma once

__device__ int count1 = 0;
__device__ int count2 = 0;
__device__ int count3 = 0;


template<typename T>
__global__ void _non_zero_counter1(T* d_arr,int row,int col){

	int i= threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = i *col +j;

	if(i==j && d_arr[idx]==1){
		atomicAdd(&count1,1);
	}
  
}

template<typename T>
__global__ void _non_zero_counter2(T* d_arr,int row,int col){

	int i= threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = i *col +j;

	if(i==j && d_arr[idx]!=0){
		atomicAdd(&count2,1);
	}
  
}

template<typename T>
__global__ void _non_zero_counter3(T* d_arr,int row,int col){

	int i= threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = i *col +j;

	if( d_arr[idx]==0){
		atomicAdd(&count3,1);
	}
  
}
template<typename T>
bool iseye(T* h_arr,int row,int col,int dim_x=16,int dim_y=16){
    //creating space for device array
	T* d_arr;
	size_t size = row*col*sizeof(T);
	cudaMalloc((T**)&d_arr,size);

	//copying the values to device array
cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice);

	size_t int_size = sizeof(int);

	//initializing counter variable
	
	dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);


	//calling the parallel funtion
	_non_zero_counter1<<<grid,block>>>(d_arr,row,col);

	//ensuring all the threads have completed there job
	//__syncthreads();

	//copying the value to host counter
//cudaMemcpy(h_counter,d_counter,int_size,cudaMemcpyDeviceToHost);
	bool result;
  int b;
   cudaMemcpyFromSymbol(&b, count1, sizeof(int));
  //cout<<"The value of B=  "<<b<<endl;
	if(b==min(row,col))
	result=true;
	else
	result=false;

	//calculating the density
	return result;

	

}
template<typename T>
bool isdiagonal(T* h_arr,int row,int col,int dim_x=16,int dim_y=16){
    //creating space for device array
	T* d_arr;
	size_t size = row*col*sizeof(T);
	cudaMalloc((T**)&d_arr,size);

	//copying the values to device array
cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice);

	size_t int_size = sizeof(int);

	//initializing counter variable
	
	dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);


	//calling the parallel funtion
	_non_zero_counter2<<<grid,block>>>(d_arr,row,col);

	//ensuring all the threads have completed there job
	//__syncthreads();

	//copying the value to host counter
//cudaMemcpy(h_counter,d_counter,int_size,cudaMemcpyDeviceToHost);
	bool result;
  int b;
   cudaMemcpyFromSymbol(&b, count2, sizeof(int));
  //cout<<"The value of B=  "<<b<<endl;
	if(b==min(row,col))
	result=true;
	else
	result=false;

	//calculating the density
	return result;
}
template<typename T>
bool iszero(T* h_arr,int row,int col,int dim_x=16,int dim_y=16){
    //creating space for device array
	T* d_arr;
	size_t size = row*col*sizeof(T);
	cudaMalloc((T**)&d_arr,size);

	//copying the values to device array
cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice);

	size_t int_size = sizeof(int);

	//initializing counter variable
	
	dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);


	//calling the parallel funtion
	_non_zero_counter3<<<grid,block>>>(d_arr,row,col);

	//ensuring all the threads have completed there job
	//__syncthreads();

	//copying the value to host counter
//cudaMemcpy(h_counter,d_counter,int_size,cudaMemcpyDeviceToHost);
	bool result;
  int b;
   cudaMemcpyFromSymbol(&b, count3, sizeof(int));
  cout<<"The value of B=  "<<b<<endl;
	if(b==(row*col))
	result=true;
	else
	result=false;

	//calculating the density
	return result;

	

}


	

	
	
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

	_divider<<<  >>>();

	__syncthreads();

	CHECK(cudaMemcpy(h_result,d_result,size_row,cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_result));
	CHECK(cudaFree(d_arr));
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

	_divider<<<  >>>();

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






template<typename T>
__global__ void _isSymmetric(T* arr,int size, int d_counter)
{

	int i= threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(arr[i*size + j] != arr[j*size + i])
	{
		atomicAdd(&d_counter,1);
	}
  
}

template<typename T>
bool issymmetric(T* h_arr, int size1, int dim_x=16,int dim_y=16)
{ 
    T* d_arr;
    size_t size = size1*size1*sizeof(T);

    CHECK(cudaMalloc((T**)&d_arr,size));
 
    CHECK(cudaMemcpy(d_arr,h_arr,size,cudaMemcpyHostToDevice));
 
 	int* h_ctr;
 	h_ctr = (int*) malloc(sizeof(int));
 	*h_ctr = 0;

 	int* d_ctr;
 	CHECK(cudaMalloc((int**)&d_ctr,sizeof(int)));

	CHECK(cudaMemcpy(d_ctr,h_ctr,sizeof(int),cudaMemcpyHostToDevice));

    dim3 block(dim_x,dim_y);
	dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);

    _isSymmetric<<<grid,block>>>(d_arr,size1,d_ctr);
    
 
    CHECK(cudaMemcpy(h_ctr,d_ctr,sizeof(int),cudaMemcpyDeviceToHost));

    return (*h_ctr)==0;
}




void std()







// AtomicMin(),AtomicXOR()
