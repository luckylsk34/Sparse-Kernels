#ifndef THIRD_PARTY_SPARSEKERNEL_TEST_UTILS_H_
#define THIRD_PARTY_SPARSEKERNEL_TEST_UTILS_H_

#include <cublas_v2.h>
#include <cusparse.h>

#include "glog/logging.h"

namespace sparsekernel {

#define CUDA_CALL(code)                                                        \
  do {                                                                         \
    cudaError_t status = code;                                                 \
    std::string err = cudaGetErrorString(status);                              \
    CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err;                    \
  } while (0)

#define CUSPARSE_CALL(code)                                                    \
  do {                                                                         \
    cusparseStatus_t status = code;                                            \
    CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS) << "CuSparse Error";             \
  } while (0)

#define CUBLAS_CALL(code)                                                      \
  do {                                                                         \
    cublasStatus_t status = code;                                              \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "CuBLAS Error";                 \
  } while (0)

} // namespace sparsekernel

#endif // THIRD_PARTY_SPARSEKERNEL_TEST_UTILS_H_
