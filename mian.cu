#pragma once
#include "gemm_kernel.cuh"
#include "cuda_check.cuh"
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

int main(int argc, char** argv)
{
    size_t M = argc > 1 ? atoll(argv[1]) : 128;
    size_t N = argc > 2 ? atoll(argv[2]) : 128;
    size_t K = argc > 3 ? atoll(argv[3]) : 64;
    printf("GEMM: M=%zu N=%zu K=%zu\n", M, N, K);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    for (auto& v : hA) v = __float2bfloat16(dist(rng));
    for (auto& v : hB) v = __float2bfloat16(dist(rng));

    __nv_bfloat16* dA, * dB, * dC;
    float* dC_ref;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * 2));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * 2));
    CHECK_CUDA(cudaMalloc(&dC, M * N * 2));
    CHECK_CUDA(cudaMalloc(&dC_ref, M * N * 4));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));

    // 1.Check if the calculation is correct
    run_kernel(dA, dB, dC, M, N, K);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.f, beta = 0.f;

    CHECK_CUBLAS(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        dB, CUDA_R_16BF, (int)K,      
        dA, CUDA_R_16BF, (int)K,      
        &beta,
        dC_ref, CUDA_R_32F, (int)N,  
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hC(M * N);
    std::vector<float> hRef(M * N);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, M * N * 2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRef.data(), dC_ref, M * N * 4, cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (size_t i = 0; i < M * N; ++i) {
        double ref = hRef[i];
        double denom = fabs(ref) > 1e-3 ? fabs(ref) : 1e-3;
        max_err = fmax(max_err, fabs(__bfloat162float(hC[i]) - ref) / denom);
    }
    printf("max relative error = %e -> %s\n", max_err, max_err < 3e-2 ? "PASS" : "FAIL");
    if (max_err >= 3e-2) 
        return 1;

    // allocate a buffer much larger than L2 for flushing
    const size_t FLUSH_BYTES = 256ull << 20;  // 256MB > L2 cache size
    float* dFlush;
    CHECK_CUDA(cudaMalloc(&dFlush, FLUSH_BYTES));
    CHECK_CUDA(cudaMemset(dFlush, 0, FLUSH_BYTES)); 

    const int ITERS = 20;

    std::vector<float> times(ITERS);
    std::vector<cudaEvent_t> evBeg(ITERS), evEnd(ITERS);
    for (auto& e : evBeg) cudaEventCreate(&e);
    for (auto& e : evEnd) cudaEventCreate(&e);

    // 2.Preheating 
    for (int i = 0; i < 5; ++i) 
        run_kernel(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3.Timer
    for (int i = 0; i < ITERS; ++i) 
    {
        // flush outside the timing interval
        //first replace all of L2 with the flush buffer data
        cudaMemsetAsync(dFlush, 0, FLUSH_BYTES);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaEventRecord(evBeg[i]);
        run_kernel(dA, dB, dC, M, N, K);
        cudaEventRecord(evEnd[i]);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int i = 0; i < ITERS; ++i) {
        float ms;
        cudaEventElapsedTime(&ms, evBeg[i], evEnd[i]);
        times[i] = ms;
    }

    for (int i = 1; i < ITERS; ++i)
    {
        times[0] += times[i];
    }

    times[0] /= ITERS;
    printf("avg time: %.4f ms, throughput: %.2f TFLOPS\n",
        times[0], 2.0 * M * N * K / (times[0] * 1e6));

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
    return 0;
}