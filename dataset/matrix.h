#pragma once
#include<stdlib.h>
#include<curand.h>
#include<curand_kernel.h>

#define CHECK(call){\
	const cudaError_t error = call;\
	if(error != cudaSuccess){\
		printf("Error : %s : %d, ",__FILE__,__LINE__);\
		printf("code : %d, reason %s \n",error,cudaGetErrorString(error));\
		exit(1);\
	}\
}


template<typename T>
__global__ void _matrix(T* d_arr,uint row,uint col,curandState_t* states,float sparsity,uint max_value,uint min_value){
    uint _row = threadIdx.x + blockIdx.x * blockDim.x;
	uint _col = threadIdx.y + blockIdx.y * blockDim.y;

	uint idx = _row * col + _col;
	//__TODO__ rand doesn't work on gpu
	// int val = 25 % 255;
	// val = val * (9 % 2);
	// d_arr[idx] = val;   

	int prob = curand_uniform(states);
	int val=0;
	if(prob>=sparsity){
		val = min_value + (int)(max_value-min_value)*curand_uniform(states);
	}
	d_arr[idx] = val;
}

__global__ void rand_init(unsigned int seed, curandState_t* states) {
  curand_init(seed, blockIdx.x, 5, states);
}


template<typename T>
void matrix(T* h_arr,uint row, uint col,float sparsity=0.7,uint max_value=10000,uint min_value=0,uint dim_x = 32, uint dim_y = 32){

	//initialization of randstate
	curandState_t* states;
    CHECK(cudaMalloc((curandState_t**) &states, sizeof(curandState_t)));
    rand_init<<<1, 1>>>(time(0), states);
    cudaDeviceSynchronize();

	// needs a (uninitialized) pointer to host and rows and columns in the required matrix

    size_t size = row*col*sizeof(T);

	// h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(dim_x,dim_y);
    dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);

    // printf("%d %d",grid.x,grid.y);

    _matrix<<<grid,block>>>(d_arr,row,col,states,sparsity,max_value,min_value);

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_arr));

}


template<typename T>
__global__ void _zeros(T* arr,int row,int col)
{
     

int blockNum = blockIdx .z * ( gridDim .x * gridDim .y) + blockIdx .y * gridDim .x +
blockIdx .x;
int threadNum = threadIdx .z * ( blockDim .x * blockDim .y) + threadIdx .y * ( blockDim .
x) + threadIdx .x;

int globalThreadId = blockNum * ( blockDim .x * blockDim .y * blockDim .z) + threadNum
;
arr[globalThreadId]=0;
}


template<typename T>
void zeros(T* h_arr, int row,int col)
{
size_t size = row*col*sizeof(T);

	//h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(2,2);
    dim3 grid(col/2,row/2);

    //printf("%d %d",grid.x,grid.y);

    _zeros<<<grid,block>>>(d_arr,row,col);

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_arr));
}

template<typename T>
__global__ void _eye(T* arr,int size)
{
     
int i= blockIdx .y* blockDim .y+ threadIdx .y;
int j= blockIdx .x* blockDim .x+ threadIdx .x;
if(i!=j)
arr[i*size+j]=0;
else
arr[i*size+j]=1;

}




template<typename T>
void eye(T* h_arr, int size1)
{
size_t size = size1*size1*sizeof(T);

	//h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(2,2);
    dim3 grid(size1/2,size1/2);

    //printf("%d %d",grid.x,grid.y);

    _eye<<<grid,block>>>(d_arr,size1);

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_arr));
}

template<typename T>
__global__ void _diagonal(T* arr,int size)
{
     
int i= blockIdx .y* blockDim .y+ threadIdx .y;
int j= blockIdx .x* blockDim .x+ threadIdx .x;
if(i!=j)
arr[i*size+j]=0;
else
arr[i*size+j]=i*size+j;

}



template<typename T>
void diagonal(T* h_arr, int size1)
{
size_t size = size1*size1*sizeof(T);

	//h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(2,2);
    dim3 grid(size1/2,size1/2);

    //printf("%d %d",grid.x,grid.y);

    _diagonal<<<grid,block>>>(d_arr,size1);

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_arr));
}







//#include <curand.h>
//#include <curand_kernel.h>

template<typename T>
_global_ void _upper(T* arr,int size)
{
     
	int i= blockIdx.y* blockDim.y+ threadIdx.y;
	int j= blockIdx.x* blockDim.x+ threadIdx.x;
	if(j>=i)
	{
		/* Generae random number between -2000 to 2000
		curandState_t state;
		curand_init(clock64(), //the seed controls the sequence of random values that are produced 
              blockIdx.x, //the sequence number is only important with multiple cores 
              0, //the offset is how much extra we advance in the sequence for each call, can be 0 
              &state);
		float myrandf = curand_uniform();
	    myrandf *= (4000+0.999999);
	    myrandf -=2000.00;
	    //

	    T myrand = (T) myrandf;
		arr[i*size+j]=myrand;
		*/
		arr[i*size+j]=i*size+j;
	}
	else
		arr[i*size+j]=0;

}


template<typename T>
void upper(T* h_arr,int size1)
{
	size_t size = size1*size1*sizeof(T);

	//h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(2,2);
    dim3 grid(size1/2,size1/2);

    //printf("%d %d",grid.x,grid.y);

    CHECK(_upper<<<grid,block>>>(d_arr,size1));

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_arr));
}




template<typename T>
_global_ void _lower(T* arr,int size)
{
     
	int i= blockIdx.y* blockDim.y+ threadIdx.y;
	int j= blockIdx.x* blockDim.x+ threadIdx.x;
	if(i>=j)
	{
		/* Generae random number between -2000 to 2000
		curandState_t state;
		curand_init(clock64(), //the seed controls the sequence of random values that are produced 
              blockIdx.x, //the sequence number is only important with multiple cores 
              0, //the offset is how much extra we advance in the sequence for each call, can be 0 
              &state);
		float myrandf = curand_uniform();
	    myrandf *= (4000+0.999999);
	    myrandf -=2000.00;
	    //

	    T myrand = (T) myrandf;
		arr[i*size+j]=myrand;
		*/
		arr[i*size+j]=i*size+j;
	}
	else
		arr[i*size+j]=0;

}


template<typename T>
void lower(T* h_arr, int size1)
{
	size_t size = size1*size1*sizeof(T);

	//h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(2,2);
    dim3 grid(size1/2,size1/2);

    //printf("%d %d",grid.x,grid.y);

    CHECK(_lower<<<grid,block>>>(d_arr,size1));

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_arr));
}



template<typename T>
_global_ void _tridiagonal(T* arr,int size)
{
     
	int i= blockIdx.y* blockDim.y+ threadIdx.y;
	int j= blockIdx.x* blockDim.x+ threadIdx.x;
	if(i-j<=1 && i-j>=-1)
	{
		/* Generae random number between -2000 to 2000
		curandState_t state;
		curand_init(clock64(), //the seed controls the sequence of random values that are produced 
              blockIdx.x, //the sequence number is only important with multiple cores 
              0, //the offset is how much extra we advance in the sequence for each call, can be 0 
              &state);
		float myrandf = curand_uniform();
	    myrandf *= (4000+0.999999);
	    myrandf -=2000.00;
	    //

	    T myrand = (T) myrandf;
		arr[i*size+j]=myrand;
		*/
		arr[i*size+j]=i*size+j;
	}
	else
		arr[i*size+j]=0;

}


template<typename T>
void tridiagonal(T* h_arr,int size1)
{
	size_t size = size1*size1*sizeof(T);

	//h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(2,2);
    dim3 grid(size1/2,size1/2);

    //printf("%d %d",grid.x,grid.y);

    CHECK(_tridiagonal<<<grid,block>>>(d_arr,size1));

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_arr));
}







//#include"properties.h"
