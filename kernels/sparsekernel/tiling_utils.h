#ifndef THIRD_PARTY_SPARSEKERNEL_TILING_UTILS_H_
#define THIRD_PARTY_SPARSEKERNEL_TILING_UTILS_H_

/**
 * @file @brief Utilities for mapping threads and thread blocks to matrix tiles.
 */

namespace sparsekernel {

/**
 * @brief Helper to simplify expressing indexing math efficiently.
 *
 * Offsets the input pointer in bytes.
 */
template <typename OutType, typename InType>
__device__ __forceinline__ OutType *OffsetCast(InType *ptr, int offset) {
  return reinterpret_cast<OutType *>(
      const_cast<char *>(reinterpret_cast<const char *>(ptr)) + offset);
}

/**
 * @brief Utilities for calculating M & N dimension tile mappings.
 */
template <int kBlockItemsY, int kBlockItemsK, int kBlockItemsX>
struct TilingUtils {
  static __device__ __forceinline__ int IndexM() {
    return blockIdx.x * kBlockItemsY + threadIdx.y;
  }

  static __device__ __forceinline__ int IndexN() {
    return blockIdx.y * kBlockItemsX;
  }

  template <typename T>
  static __device__ __forceinline__ T *MaybeOffset(T *ptr, int off) {
    return ptr + off;
  }
};

/**
 * @brief Specialization for 1-dimensional warp-level tiles.
 *
 * Skip some unescesary indexing math when all threads in a warp share
 * the same tile.
 */
template <int kBlockItemsK, int kBlockItemsX>
struct TilingUtils<1, kBlockItemsK, kBlockItemsX> {
  static __device__ __forceinline__ int IndexM() { return blockIdx.x; }

  static __device__ __forceinline__ int IndexN() {
    return blockIdx.y * kBlockItemsX;
  }

  template <typename T>
  static __device__ __forceinline__ T *MaybeOffset(T *ptr, int /* unused */) {
    return ptr;
  }
};

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_TILING_UTILS_H_
