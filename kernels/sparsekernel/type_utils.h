#ifndef THIRD_PARTY_SPARSEKERNEL_TYPE_UTILS_H_
#define THIRD_PARTY_SPARSEKERNEL_TYPE_UTILS_H_

/**
 * @file @brief Defines utilities for working with mixes of data types
 * for storage and computation.
 */

#include "sparsekernel/cuda_utils.h"

namespace sparsekernel {

/**
 * @brief Helper for mixed-precision computation meta-data.
 */
template <typename Value> struct TypeUtils {
  static constexpr int kElementsPerScalar = 1;

  static constexpr __device__ __forceinline__ bool IsMixed() { return false; }

  // The data type of our accumulators.
  typedef Value Accumulator;

  // The data type of a scalar value.
  typedef float ScalarValue;
};


/**
 * @brief Functor to translate vector data types to vector index types.
 */
template <typename Value> struct Value2Index { typedef int Index; };

template <> struct Value2Index<float4> { typedef int4 Index; };

/**
 * @brief Helper to convert between datatypes.
 */
template <typename To, typename From>
__device__ __forceinline__ void Convert(const From *in, To *out) {
  // In the default case, don't perform any conversion. Reinterpret.
  *out = *reinterpret_cast<const To *>(in);
}


} // namespace sparsekernel
#endif // THIRD_PARTY_SPARSEKERNEL_TYPE_UTILS_H_
