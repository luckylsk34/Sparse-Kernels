#ifndef THIRD_PARTY_SPARSEKERNEL_VECTOR_UTILS_H_
#define THIRD_PARTY_SPARSEKERNEL_VECTOR_UTILS_H_

/**
 * @file @brief Utilities for working with CUDA vector data types.
 */

#include "sparsekernel/cuda_utils.h"
#include "sparsekernel/type_utils.h"

namespace sparsekernel {

/**
 * @brief Functor for computing FMAs & MULs on mixes of vector and scalar
 * data types.
 */
template <typename Value> struct VectorCompute {
  typedef typename TypeUtils<Value>::Accumulator Accumulator;

  static __device__ __forceinline__ void FMA(float x1, Value x2,
                                             Accumulator *out);

  // Complementary index type to our load type.
  typedef typename Value2Index<Value>::Index Index;

  static __device__ __forceinline__ void Mul(int, Index x2, Index *out);

  static __device__ __forceinline__ void Dot(Value x1, Value x2,
                                             Accumulator *out);
};

template <> struct VectorCompute<float> {
  static __device__ __forceinline__ void FMA(float x1, float x2, float *out) {
    out[0] += x1 * x2;
  }

  static __device__ __forceinline__ void Mul(int x1, int x2, int *out) {
    out[0] = x1 * x2;
  }

  static __device__ __forceinline__ void Dot(float x1, float x2, float *out) {
    out[0] += x1 * x2;
  }
};

template <> struct VectorCompute<float4> {
  static __device__ __forceinline__ void FMA(float x1, float4 x2, float4 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
    out[0].z += x1 * x2.z;
    out[0].w += x1 * x2.w;
  }

  static __device__ __forceinline__ void Mul(int x1, int4 x2, int4 *out) {
    out[0].x = x1 * x2.x;
    out[0].y = x1 * x2.y;
    out[0].z = x1 * x2.z;
    out[0].w = x1 * x2.w;
  }

  static __device__ __forceinline__ void Dot(float4 x1, float4 x2, float *out) {
    out[0] += x1.x * x2.x;
    out[0] += x1.y * x2.y;
    out[0] += x1.z * x2.z;
    out[0] += x1.w * x2.w;
  }
};

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_VECTOR_UTILS_H_
