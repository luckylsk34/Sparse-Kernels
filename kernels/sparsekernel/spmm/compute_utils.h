#ifndef THIRD_PARTY_SPARSEKERNEL_SPMM_COMPUTE_UTILS_H_
#define THIRD_PARTY_SPARSEKERNEL_SPMM_COMPUTE_UTILS_H_

/**
 * @file @brief Utilities for tile-level computation.
 */

#include "sparsekernel/type_utils.h"
#include "sparsekernel/vector_utils.h"

namespace sparsekernel {

/**
 * @brief Functor for managing sparse matrix-matrix product computation.
 */
template <typename Value, int kBlockItemsK, int kBlockItemsX, int kBlockWidth>
struct ComputeUtils {
  //
  /// Static members.
  //

  // The type of a single element of a Value.
  typedef typename TypeUtils<Value>::ScalarValue ScalarValue;

  // The number of float values per element of the rhs fragment.
  static constexpr int kValuesPerItem_ = sizeof(Value) / sizeof(ScalarValue);

  // The number of outputs each thread is responsbile for.
  static constexpr int kThreadItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerItem_;

  // The number of elements to compute on per-scalar rhs element.
  static constexpr int kElementsPerScalar_ =
      TypeUtils<Value>::kElementsPerScalar;

  // Data type of our accumulator registers.
  typedef typename TypeUtils<Value>::Accumulator Accumulator;

  //
  /// Member variables.
  //

  // Smem buffer storing the lhs tile values.
  const ScalarValue *lhs_tile_;

  // Register file fragment storing the rhs tile.
  const Value *rhs_fragment_;

  // Register file fragment to accumulate results into.
  float *output_fragment_;

  __device__ __forceinline__ ComputeUtils(const ScalarValue *lhs_tile,
                                          const ScalarValue *rhs_fragment,
                                          float *output_fragment)
      : lhs_tile_(lhs_tile),
        rhs_fragment_(reinterpret_cast<const Value *>(rhs_fragment)),
        output_fragment_(output_fragment) {}

  /**
   * @brief Compute a tile-level matrix product.
   */
  __device__ __forceinline__ void TileMAC() {
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kBlockItemsK; ++k_item_idx) {
      float lhs_values[kElementsPerScalar_];
      Convert(lhs_tile_ + k_item_idx, lhs_values);
#pragma unroll
      for (int elt_idx = 0; elt_idx < kElementsPerScalar_; ++elt_idx) {
#pragma unroll
        for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
          float *outputs = output_fragment_ +
                           x_item_idx * kValuesPerItem_ * kElementsPerScalar_;
          int rhs_offset = k_item_idx * kThreadItemsX_ * kElementsPerScalar_ +
                           elt_idx * kThreadItemsX_ + x_item_idx;
          VectorCompute<Value>::FMA(lhs_values[elt_idx],
                                    rhs_fragment_[rhs_offset],
                                    reinterpret_cast<Accumulator *>(outputs));
        }
      }
    }
  }
};

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_SPMM_COMPUTE_UTILS_H_
