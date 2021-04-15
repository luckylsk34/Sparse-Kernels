#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>

#include "sparsekernel/barrier.h"
#include "sparsekernel/cuda_utils.h"
#include "sparsekernel/load_store.h"
#include "sparsekernel/memory_aligner.h"
#include "sparsekernel/spmm/compute_utils.h"
#include "sparsekernel/spmm/cuda_spmm.h"
#include "sparsekernel/spmm/dense_tile.h"
#include "sparsekernel/spmm/output_tile.h"
#include "sparsekernel/spmm/predicate_utils.h"
#include "sparsekernel/spmm/sparse_tile.h"
#include "sparsekernel/spmm/spmm_config.h"
#include "sparsekernel/tiling_utils.h"
#include "sparsekernel/vector_utils.h"

namespace sparsekernel {

namespace {

template <typename Config> struct SpmmKernel {
  //
  /// Shortcuts for commonly used specialized types.
  //

  typedef TilingUtils<Config::kBlockItemsY, Config::kBlockItemsK,
                      Config::kBlockItemsX>
      Tiling;

  typedef PredicateVector<Config::kThreadItemsX> PredicateVector;

  typedef PredicatesN<typename Config::DenseValue, Config::kBlockItemsX,
                      Config::kBlockWidth>
      PredicatesN;

  typedef MemoryAligner<typename Config::SparseValue, Config::kBlockWidth>
      MemoryAligner;

  typedef SparseTile<typename Config::SparseValue, Config::kBlockItemsK,
                     Config::kBlockWidth>
      SparseTile;

  typedef DenseTile<typename Config::DenseValue, Config::kBlockItemsK,
                    Config::kBlockItemsX, Config::kBlockWidth,
                    Config::kResidueUnroll>
      DenseTile;

  typedef ComputeUtils<typename Config::DenseValue, Config::kBlockItemsK,
                       Config::kBlockItemsX, Config::kBlockWidth>
      Computer;

  typedef Barrier<Config::kBlockItemsY, Config::kBlockWidth> Barrier;

  typedef OutputTile<typename Config::DenseValue, Config::kBlockItemsX,
                     Config::kBlockWidth>
      OutputTile;

  typedef typename Config::ScalarValue ScalarValue;
  typedef typename Config::DenseValue DenseValue;
  typedef typename Config::SparseValue SparseValue;
  typedef typename Config::ScalarIndex ScalarIndex;
  typedef typename Config::Index Index;

  /**
   * @brief Main function for SpMM kernel.
   */
  static __device__ __forceinline__ void
  KernelFn(int m, int k, int n, const int *__restrict__ row_indices,
           const ScalarValue *__restrict__ values,
           const int *__restrict__ row_offsets,
           const ScalarIndex *__restrict__ column_indices,
           const ScalarValue *__restrict__ dense_matrix,
           const float *__restrict__ bias, ScalarValue *__restrict__ out) {
    // Calculate this thread block's indices into the M and N dimensions.
    int m_index = Tiling::IndexM(), n_index = Tiling::IndexN();

    // Threads that work on different m-dim indices are independent. If
    // we're out of bounds in the m-dimension we can just return.
    if (m_index >= m)
      return;
    m_index = Load(row_indices + m_index);

    // Divide some of our constant problem dimensions and indices by
    // the number of elements that are packed into each scalar.
    n /= Config::kElementsPerScalar;

    // Initialize the n-dimension predicates for this thread.
    PredicateVector predicates_n;
    if (Config::kPredicateLoads) {
      PredicatesN::Set(n_index, n, &predicates_n);
    }

    // Load the row offset and calculate the number of non-zeros in the row.
    int row_offset = Load(row_offsets + m_index);
    int nonzeros = Load(row_offsets + m_index + 1) - row_offset;

    // Divide some of our constant values by the number of elements that
    // are packed into a single scalar.
    nonzeros /= Config::kElementsPerScalar;
    row_offset /= Config::kElementsPerScalar;

    // Possibly align the row offset s.t. it's safe to use vector memory ops.
    //
    // Note that if we only have residue to process, we do not align the row
    // offset. This lets us not worry about masking in the residue handling,
    // where we use scalar memory operations anyways.
    MemoryAligner memory_aligner(row_offset, nonzeros);
    int aligned_nonzeros = memory_aligner.AlignedNonzeros();
    if (aligned_nonzeros >= Config::kBlockItemsK) {
      nonzeros = aligned_nonzeros;
      row_offset = memory_aligner.AlignedRowOffset();
    }

    // Shared memory tiles for the lhs values and indices.
    constexpr int kTileSize = Config::kBlockItemsK * Config::kBlockItemsY;
    __shared__ ScalarValue values_tile_array[kTileSize];
    __shared__ ScalarIndex column_indices_tile_array[kTileSize];

    // Possibly increment our tile pointers for 2D tiling schemes.
    ScalarValue *values_tile = Tiling::MaybeOffset(
        values_tile_array, Config::kBlockItemsK * threadIdx.y);
    ScalarIndex *column_indices_tile = Tiling::MaybeOffset(
        column_indices_tile_array, Config::kBlockItemsK * threadIdx.y);

    // Create a loader for the sparse lhs matrix.
    SparseTile sparse_tile_loader(n, row_offset, threadIdx.x, values,
                                  column_indices, values_tile,
                                  column_indices_tile);

    // Register fragment for the dense_matrix values.
    constexpr int kDenseFragmentSize =
        Config::kElementsPerScalar * Config::kBlockItemsK *
        Config::kBlockItemsX / Config::kBlockWidth;
    __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize];

    // Create a loader for the dense dense_matrix matrix.
    DenseTile dense_tile_loader(n, n_index, threadIdx.x, dense_matrix,
                                column_indices_tile, dense_matrix_fragment);

    // Accumulator registers for the output values. Initialize the
    // registers to zero s.t. we can always accumulate in-place.
    constexpr int kOutputFragmentSize =
        Config::kBlockItemsX / Config::kBlockWidth * Config::kElementsPerScalar;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};

    // Helper for computing tile-level partial matmuls.
    Computer computer(values_tile, dense_matrix_fragment, output_fragment);

    // Helper for managing synchronization between collaborating threads.
    Barrier barrier(threadIdx.y);

    //
    /// Begin kernel main loop.
    //

    // For the first iteration of our main loop, we need to possibly mask
    // the first few values from the sparse matrix in case we aligned our
    // values and column indices pointers.
    if (nonzeros >= Config::kBlockItemsK) {
      // Load a tile from the sparse lhs matrix and synchronize the cta.
      sparse_tile_loader.Load();
      barrier.Sync();

      // Mask any values we loaded that aren't from our row of the sparse
      // matrix. Threads could potentially mask values in smem that they
      // were not responsible for loading. Synchronize again to make sure
      // the masking occurs after the previous loads have completed.
      //
      // TODO(tgale): We don't need to synchronize here for the scalar
      // variants of the kernels. We also don't need to handle the first
      // iteration specially. This kernel has now become very complex. It
      // would be nice to break it out into an SpMM class where we can
      // break each of these sections out into helper functions.
      memory_aligner.MaskPrefix(values_tile, column_indices_tile);
      barrier.Sync();

      // Load a tile from the sparse dense_matrix matrix.
      dense_tile_loader.Load(predicates_n);

      // Multiply the tiles and accumulate the results.
      computer.TileMAC();
      nonzeros -= Config::kBlockItemsK;
    }

    // Loop over the tiles in the k-dimension of the dense_matrix/lhs matrices.
    for (; nonzeros >= Config::kBlockItemsK; nonzeros -= Config::kBlockItemsK) {
      // Synchronize s.t. we don't overwrite our shared memory tiles while
      // other warps have not completed using them in computation.
      barrier.Sync();

      // Load a tile from the sparse lhs matrix and synchronize the cta.
      sparse_tile_loader.Load();
      barrier.Sync();

      // Load a tile from the sparse dense_matrix matrix.
      dense_tile_loader.Load(predicates_n);

      // Multiply the tiles and accumulate the results.
      computer.TileMAC();
    }

    //
    /// Begin spmm residue computation.
    //

    // Synchronize s.t. we don't overwrite our shared memory tiles while
    // other warps have not completed using them in computation.
    barrier.Sync();

    // Zero the shared memory tiles s.t. we can operate on sets of 2/4
    // values safely in the dense tile loads and computation.
    if (Config::kResidueUnroll > 1) {
      sparse_tile_loader.ZeroTiles();
      barrier.Sync();
    }

    // Load a tile from the sparse lhs matrix and synchronize the cta.
    sparse_tile_loader.Residue(nonzeros);
    barrier.Sync();

    // Load a tile from the dense dense_matrix matrix and compute immediately.
    dense_tile_loader.ResidueLoadAndCompute(nonzeros, predicates_n, values_tile,
                                            output_fragment);

    //
    /// Write results to the output.
    //

    // Possibly apply the bias and RelU.
    if (bias != nullptr) {
      // Bias value is shared across all outputs.
      const float bias_value = Load(bias + m_index);
#pragma unroll
      for (int out_idx = 0; out_idx < kOutputFragmentSize; ++out_idx) {
        output_fragment[out_idx] += bias_value;
        output_fragment[out_idx] =
            output_fragment[out_idx] > 0 ? output_fragment[out_idx] : 0;
      }
    }

    // Create a storer for the output matrix.
    OutputTile output_tile_storer(m_index, n_index, n, threadIdx.x,
                                  output_fragment, out);
    output_tile_storer.Store(predicates_n);
  }
};

template <typename Config>
__global__ void __launch_bounds__(Config::kThreadsPerBlock)
    Kernel(int m, int k, int n, const int *__restrict__ row_indices,
           const typename Config::ScalarValue *__restrict__ values,
           const int *__restrict__ row_offsets,
           const typename Config::ScalarIndex *__restrict__ column_indices,
           const typename Config::ScalarValue *__restrict__ dense_matrix,
           const float *__restrict__ bias,
           typename Config::ScalarValue *__restrict__ out) {
  SpmmKernel<Config>::KernelFn(m, k, n, row_indices, values, row_offsets,
                               column_indices, dense_matrix, bias, out);
}

template <typename Config>
__global__ void __launch_bounds__(Config::kThreadsPerBlock,
                                  Config::kMinOccupancy)
    KernelWithBounds(
        int m, int k, int n, const int *__restrict__ row_indices,
        const typename Config::ScalarValue *__restrict__ values,
        const int *__restrict__ row_offsets,
        const typename Config::ScalarIndex *__restrict__ column_indices,
        const typename Config::ScalarValue *__restrict__ dense_matrix,
        const float *__restrict__ bias,
        typename Config::ScalarValue *__restrict__ out) {
  SpmmKernel<Config>::KernelFn(m, k, n, row_indices, values, row_offsets,
                               column_indices, dense_matrix, bias, out);
}

typedef std::function<cudaError_t(
    int,           // m: number of rows in lhs & output.
    int,           // k: number of cols in lhs and rows in rhs.
    int,           // n: number of cols in rhs/output.
    int,           // nonzeros: number of nonzero values in lhs.
    const int *,   // row_indices: ptr to row index swizzle map.
    const float *, // values: ptr to lhs values.
    const int *,   // row_offsets: ptr to lhs row offsets.
    const int *,   // column_indices: ptr to lhs column indices.
    const float *, // dense_matrix: ptr to rhs matrix.
    const float *, // bias: bias pointer.
    float *,       // output_matrix: ptr to output matrix.
    cudaStream_t)> // stream: stream to execute in.
    FloatSpmmFn;

} // namespace
cudaError_t CudaSpmm(int m, int k, int n, int nonzeros,
                     const int *__restrict__ row_indices,
                     const float *__restrict__ values,
                     const int *__restrict__ row_offsets,
                     const int *__restrict__ column_indices,
                     const float *__restrict__ dense_matrix,
                     float *__restrict__ output_matrix, cudaStream_t stream) {
  typedef SpmmConfig<float, float4, float4, 4, 32, 64, 8, 4, false, true, 8> Config;
  return CudaSpmmEx<Config>(m, k, n, nonzeros, row_indices, values,
                                row_offsets, column_indices, dense_matrix,
                                output_matrix, stream);
}

template <typename Config>
cudaError_t
CudaSpmmEx(int m, int k, int n, int nonzeros,
           const int *__restrict__ row_indices,
           const typename Config::ScalarValue *__restrict__ values,
           const int *__restrict__ row_offsets,
           const typename Config::ScalarIndex *__restrict__ column_indices,
           const typename Config::ScalarValue *__restrict__ dense_matrix,
           typename Config::ScalarValue *__restrict__ output_matrix,
           cudaStream_t stream) {
  dim3 grid_dim(ceil(static_cast<float>(m) / Config::kBlockItemsY),
                ceil(static_cast<float>(n) / Config::kBlockItemsX /
                     Config::kElementsPerScalar),
                1);
  dim3 block_dim(Config::kBlockWidth, Config::kBlockItemsY, 1);

  if (Config::kLaunchBounds) {
    KernelWithBounds<Config><<<grid_dim, block_dim, 0, stream>>>(
        m, k, n, row_indices, values, row_offsets, column_indices, dense_matrix,
        nullptr, output_matrix);
  } else {
    Kernel<Config><<<grid_dim, block_dim, 0, stream>>>(
        m, k, n, row_indices, values, row_offsets, column_indices, dense_matrix,
        nullptr, output_matrix);
  }
  return cudaGetLastError();
}


} // namespace sparsekernel
