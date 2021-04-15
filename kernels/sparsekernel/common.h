#ifndef THIRD_PARTY_SPARSEKERNEL_COMMON_H_
#define THIRD_PARTY_SPARSEKERNEL_COMMON_H_

namespace sparsekernel {

/**
 * @brief Helper to round up to the nearest multiple of 'r'.
 */
constexpr __host__ __device__ __forceinline__ int RoundUpTo(int x, int r) {
  return (x + r - 1) / r * r;
}

/**
 * @brief Dividy x by y and round up.
 */
constexpr __host__ __device__ __forceinline__ int DivUp(int x, int y) {
  return (x + y - 1) / y;
}

/**
 * @brief Compute log base 2 statically. Only works when x
 * is a power of 2 and positive.
 *
 * TODO(tgale): GCC doesn't like this function being constexpr. Ensure
 * that this is evaluated statically.
 */
__host__ __device__ __forceinline__ int Log2(int x) {
  if (x >>= 1)
    return Log2(x) + 1;
  return 0;
}

/**
 * @brief Find the minimum statically.
 */
constexpr __host__ __device__ __forceinline__ int Min(int a, int b) {
  return a < b ? a : b;
}

/**
 * @brief Find the maximum statically.
 */
constexpr __host__ __device__ __forceinline__ int Max(int a, int b) {
  return a > b ? a : b;
}

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_COMMON_H_
