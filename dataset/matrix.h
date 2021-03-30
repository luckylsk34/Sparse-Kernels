#pragma once
#include<stdlib.h>
#define CHECK(call){\
	const cudaError_t error = call;\
	if(error != cudaSuccess){\
		printf("Error : %s : %d, ",__FILE__,__LINE__);\
		printf("code : %d, reason %s \n",error,cudaGetErrorString(error));\
		exit(1);\
	}\
}


template<typename T>
__global__ void _matrix(T* d_arr,uint row,uint col){
    uint _row = threadIdx.x + blockIdx.x * blockDim.x;
	uint _col = threadIdx.y + blockIdx.y * blockDim.y;

	uint idx = _row * col + _col;

	int val = rand() % 255;
	val = val * (rand() % 2);
	d_arr[idx] = val;   
}



template<typename T>
void matrix(T* h_arr,uint row, uint col,uint dim_x = 32, uint dim_y = 32){
	// needs a (uninitialized) pointer to host and rows and columns in the required matrix

    size_t size = row*col*sizeof(T);

	// h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(dim_x,dim_y);
    dim3 grid((col + block.x - 1)/block.x, (row + block.y -1)/block.y);

    // printf("%d %d",grid.x,grid.y);

    _matrix<<<grid,block>>>(d_arr,row,col);

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

