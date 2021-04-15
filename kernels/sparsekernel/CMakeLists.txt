# For use in subdirs.
set(SPARSEKERNEL_SRCS)

set(SPARSEKERNEL_SPMM_TEST_SRCS)
set(SPARSEKERNEL_SPMM_BENCHMARK_SRCS)
add_subdirectory(spmm)

set(SPARSEKERNEL_SDDMM_TEST_SRCS)
set(SPARSEKERNEL_SDDMM_BENCHMARK_SRCS)
add_subdirectory(sddmm)



set(DIR_SRCS)

##
### Find all sources in this directory.
##

# Lib srcs.
file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND DIR_SRCS ${TMP})

##
### Filter files that we don't want in the main library.
##

set(FILTER_SRCS)

# Don't want test related code.
file(GLOB TMP test_utils*)
list(APPEND FILTER_SRCS ${TMP})

# Don't need matrix utilities.
file(GLOB TMP matrix_utils*)
list(APPEND FILTER_SRCS ${TMP})

foreach(FILE ${FILTER_SRCS})
  list(REMOVE_ITEM DIR_SRCS ${FILE})
endforeach(FILE)

# Add the sources to the build.
list(APPEND SPARSEKERNEL_SRCS ${DIR_SRCS})

# Set .cu.cc files to be compiled as CUDA.
set(SPARSEKERNEL_CUDA_SRCS ${SPARSEKERNEL_SRCS})
list(FILTER SPARSEKERNEL_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${SPARSEKERNEL_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Create libsparsekernel.
add_library(sparsekernel SHARED ${SPARSEKERNEL_SRCS})
target_link_libraries(sparsekernel ${SPARSEKERNEL_LIBS})

# Library installation.
install(TARGETS sparsekernel)

set(INSTALL_BASE "include/sparsekernel")
install(FILES "cuda_utils.h" DESTINATION ${INSTALL_BASE})
install(FILES "spmm/cuda_spmm.h" DESTINATION "${INSTALL_BASE}/spmm")
install(FILES "sddmm/cuda_sddmm.h" DESTINATION "${INSTALL_BASE}/sddmm")


# Optionally build the benchmark suite.
if (BUILD_BENCHMARK)
  # Benchmark sources for all targets.
  set(SPARSEKERNEL_BENCHMARK_SRCS)
  file(GLOB TMP matrix_utils*)
  list(APPEND SPARSEKERNEL_BENCHMARK_SRCS ${TMP})
  file(GLOB TMP test_utils*)
  list(APPEND SPARSEKERNEL_BENCHMARK_SRCS ${TMP})
  
  # SpMM benchmark build.
  list(APPEND SPARSEKERNEL_SPMM_BENCHMARK_SRCS ${SPARSEKERNEL_BENCHMARK_SRCS})
  
  set(SPARSEKERNEL_SPMM_BENCHMARK_CUDA_SRCS ${SPARSEKERNEL_SPMM_BENCHMARK_SRCS})
  list(FILTER SPARSEKERNEL_SPMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPARSEKERNEL_SPMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(spmm_benchmark ${SPARSEKERNEL_SPMM_BENCHMARK_SRCS})
  target_link_libraries(spmm_benchmark sparsekernel ${SPARSEKERNEL_BENCHMARK_LIBS})

  # SDDMM benchmark build.
  list(APPEND SPARSEKERNEL_SDDMM_BENCHMARK_SRCS ${SPARSEKERNEL_BENCHMARK_SRCS})
  
  set(SPARSEKERNEL_SDDMM_BENCHMARK_CUDA_SRCS ${SPARSEKERNEL_SDDMM_BENCHMARK_SRCS})
  list(FILTER SPARSEKERNEL_SDDMM_BENCHMARK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
  set_source_files_properties(${SPARSEKERNEL_SDDMM_BENCHMARK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)  
  add_executable(sddmm_benchmark ${SPARSEKERNEL_SDDMM_BENCHMARK_SRCS})
  target_link_libraries(sddmm_benchmark sparsekernel ${SPARSEKERNEL_BENCHMARK_LIBS})
   
endif()
