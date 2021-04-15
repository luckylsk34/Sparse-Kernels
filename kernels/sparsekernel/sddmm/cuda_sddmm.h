#ifndef THIRD_PARTY_SPARSEKERNEL_SDDMM_CUDA_SDDMM_H_
#define THIRD_PARTY_SPARSEKERNEL_SDDMM_CUDA_SDDMM_H_

#include "sparsekernel/cuda_utils.h"

namespace sparsekernel {

/**
 * @brief Compute a sampled dense-dense matrix product.
 */
cudaError_t CudaSddmm(int m, int k, int n, int nonzeros,
                      const int *__restrict__ row_indices,
                      const int *__restrict__ row_offsets,
                      const int *__restrict__ column_indices,
                      const float *__restrict__ lhs_matrix,
                      const float *__restrict__ rhs_matrix,
                      float *__restrict__ output_values, cudaStream_t stream);

/**
 * @brief Compute a sampled dense-dense matrix product.
 */
template <typename LoadType, int kBlockItemsY, int kBlockItemsK,
          int kBlockItemsX, int kBlockWidth, int kPredicateK = true>
cudaError_t CudaSddmmEx(int m, int k, int n, int nonzeros,
                        const int *__restrict__ row_indices,
                        const int *__restrict__ row_offsets,
                        const int *__restrict__ column_indices,
                        const float *__restrict__ lhs_matrix,
                        const float *__restrict__ rhs_matrix,
                        float *__restrict__ output_values, cudaStream_t stream);

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_SDDMM_CUDA_SDDMM_H_
