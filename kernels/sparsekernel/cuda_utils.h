#ifndef THIRD_PARTY_SPARSEKERNEL_CUDA_UTILS_H_
#define THIRD_PARTY_SPARSEKERNEL_CUDA_UTILS_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sparsekernel {

typedef __half half;
typedef __half2 half2;

struct __align__(8) half4 {
  half2 x, y;
};

struct __align__(16) half8 {
  half2 x, y, z, w;
};

struct __align__(8) short4 {
  short2 x, y;
};

struct __align__(16) short8 {
  short2 x, y, z, w;
};

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_CUDA_UTILS_H_
