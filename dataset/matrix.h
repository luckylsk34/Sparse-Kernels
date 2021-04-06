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


// void matrix(int*,int,int);

template<typename T>
void zeros(T* arr, int row,int col);

template<typename T>
void eye(T* arr,int size);   // in identity matrix row=col

template<typename T>
void diagonal(T* arr,int size);

template<typename T>
void upper(T* arr,int size);

template<typename T>
void lower(T* arr, int size);

template<typename T>
void tridiagonal(T* arr,int size);

void test(int t);

//#include"properties.h"