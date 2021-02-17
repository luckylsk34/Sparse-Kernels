#include "SpMM.h"
#include <cuda_runtime.h>
#include <iostream>


using namespace std;

__global__ void kernel() { }

void kernel_wrapper()
{
	kernel<<<1, 1>>>();
	cout << "kernel_wrapper called" << endl;
}
