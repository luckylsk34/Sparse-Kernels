#ifndef THIRD_PARTY_SPARSEKERNEL_LOAD_STORE_H_
#define THIRD_PARTY_SPARSEKERNEL_LOAD_STORE_H_

/**
 * @file @brief Defines utilities for loading and storing data.
 */

#include "sparsekernel/cuda_utils.h"
#include <cstring>

namespace sparsekernel {

template <typename T>
__device__ __forceinline__ void Store(const T &value, T *ptr) {
  *ptr = value;
}

template <typename T> __device__ __forceinline__ T Load(const T *address) {
  return __ldg(address);
}

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_LOAD_STORE_H_
