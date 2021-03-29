#pragma once
#define CHECK(call){\
	const cudaError_t error = call;\
	if(error != cudaSuccess){\
		printf("Error : %s : %d, ",__FILE__,__LINE__);\
		printf("code : %d, reason %s \n",error,cudaGetErrorString(error));\
		exit(1);\
	}\
}

template<typename T>
__global__ _matrix(T* d_arr,uint row,uint col){
    uint index =     
}



template<typename T>
void matrix(T* h_arr,uint row, uint col){
	// needs a (uninitialized) pointer to host and rows and columns in the required matrix

    size_t size = row*col*sizeof(T);

	h_arr = (T*)malloc(size);  // allocaating memory to host arr	

	T* d_arr;
	CHECK(cudaMalloc((T**)&d_arr,size));   // allocating memory to device arr


    dim3 block(32);
    dim3 grid((col + block.x -1)/block.x,row);

    printf("%d %d",grid.x,grid.y);

    CHECK(_matrix<<<grid,block>>>(d_arr,row,col));

    CHECK(cudaMemcpy(h_arr,d_arr,size,cudaMemcpyDeviceToHost));

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

