#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                        \
    cudaError_t e = (call);                                          \
    if (e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);          \
        exit(1);                                                     \
    } } while (0)

#define CHECK_CUBLAS(call) do {                                         \
    cublasStatus_t s = (call);                                          \
    if (s != CUBLAS_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(1);                                                        \
    } } while (0)