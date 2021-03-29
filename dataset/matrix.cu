#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"

#define CHECK(call){\
	const cudaError_t error = call;\
	if(error != cudaSuccess){\
		printf("Error : %s : %d, ",__FILE__,__LINE__);\
		printf("code : %d, reason %s \n",error,cudaGetErrorString(error));\
		exit(1);\
	}\
}


template<typename T>
void matrix(T* h_arr,int row, int col){
	// needs a (uninitialized) pointer to host and rows and columns in the required matrix

	h_arr = (T*)malloc(row*col*sizeof(T));  // allocaating memory to host arr	

	T* d_arr;
	cudaMalloc((T**)&d_arr,row*col*sizeof(T));   // allocating memory to device arr



}

void test(int t){
    printf("from matrix.cu\n");
    printf("%d\n",t);
}