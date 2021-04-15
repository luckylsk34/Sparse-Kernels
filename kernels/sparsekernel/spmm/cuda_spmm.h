#ifndef THIRD_PARTY_SPARSEKERNEL_SPMM_CUDA_SPMM_H_
#define THIRD_PARTY_SPARSEKERNEL_SPMM_CUDA_SPMM_H_

#include "sparsekernel/cuda_utils.h"

namespace sparsekernel {

/**
 * @brief Compute the product of a sparse matrix and a dense matrix.
 *
 * Computes `A * B = C`, where A is a sparse matrix stored in compressed
 * sparse row format, B is a row-major dense matrix, and C is a row-major
 * dense matrix.
 *
 * @param m The number of rows in the left-hand side sparse matrix and the
 * output dense matrix.
 * @param k The number of columns in the left-hand side sparse matrix and
 * the number of rows in the right-hand side dense matrix.
 * @param n The number of columns in the right-hand side dense matrix and
 * the output dense matrix.
 * @param nonzeros The number of nonzero values in the left-hand side sparse
 * matrix.
 * @param row_indices Device-side buffer of `m` ints. Reordering of row
 * indices to balance work across warps and SMs.
 * @param values The nonzero values in the left-hand-side sparse matrix.
 * Device-side buffer of `nonzeros` floats.
 * @param row_offsets The offsets of each row of nonzeros and column indices
 * in the left-hand side sparse matrix. Device-side buffer of `m + 1` ints.
 * @param column_indices The column indices of each nonzero values in the
 * left-hand side sparse matrix. Device-side buffer of `nonzeros` ints.
 * @params dense_matrix The dense right-hand side matrix. Device-side buffer
 * of `k * n` floats.
 * @param output_matrix The dense output matrix. Device-side buffer of `m * n`
 * floats.
 * @param stream The CUDA stream to launch the kernels in.
 */
cudaError_t CudaSpmm(int m, int k, int n, int nonzeros,
                     const int *__restrict__ row_indices,
                     const float *__restrict__ values,
                     const int *__restrict__ row_offsets,
                     const int *__restrict__ column_indices,
                     const float *__restrict__ dense_matrix,
                     float *__restrict__ output_matrix, cudaStream_t stream);

/**
 * @brief SpMM variant with hyperparameter template arguments exposed.
 *
 * Computes `A * B = C`, where A is a sparse matrix stored in compressed
 * sparse row format, B is a row-major dense matrix, and C is a row-major
 * dense matrix.
 *
 * This kernel is based on a 1-dimensional tiling scheme, and the size of
 * the tiles in each problem dimension can be configured through template
 * arguments. The number of threads that will collaboratively compute an
 * output tile is configurable through the `kBlockWidth` template argument.
 * The kernel can also be specialized to use vector datatypes to load the
 * sparse/dense matrices and store the results.
 *
 * @param m The number of rows in the left-hand side sparse matrix and the
 * output dense matrix.
 * @param k The number of columns in the left-hand side sparse matrix and
 * the number of rows in the right-hand side dense matrix.
 * @param n The number of columns in the right-hand side dense matrix and
 * the output dense matrix.
 * @param nonzeros The number of nonzero values in the left-hand side sparse
 * matrix.
 * @param row_indices Device-side buffer of `m` ints. Reordering of row
 * indices to balance work across warps and SMs.
 * @param values The nonzero values in the left-hand-side sparse matrix.
 * Device-side buffer of `nonzeros` floats.
 * @param row_offsets The offsets of each row of nonzeros and column indices
 * in the left-hand side sparse matrix. Device-side buffer of `m + 1` ints.
 * @param column_indices The column indices of each nonzero values in the
 * left-hand side sparse matrix. Device-side buffer of `nonzeros` ints.
 * @params dense_matrix The dense right-hand side matrix. Device-side buffer
 * of `k * n` floats.
 * @param bias The vector bias to add to the output. Device-side buffer of
 * `m` floats.
 * @param output_matrix The dense output matrix. Device-side buffer of `m * n`
 * floats.
 * @param stream The CUDA stream to launch the kernels in.
 */
template <typename Config>
cudaError_t
CudaSpmmEx(int m, int k, int n, int nonzeros,
           const int *__restrict__ row_indices,
           const typename Config::ScalarValue *__restrict__ values,
           const int *__restrict__ row_offsets,
           const typename Config::ScalarIndex *__restrict__ column_indices,
           const typename Config::ScalarValue *__restrict__ dense_matrix,
           typename Config::ScalarValue *__restrict__ output_matrix,
           cudaStream_t stream);

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_SPMM_CUDA_SPMM_H_
