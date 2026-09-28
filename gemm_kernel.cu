#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_pipeline_primitives.h>
namespace cg = cooperative_groups;

// BF16_BATCH: number of bf16 elements loaded per cp.async transaction (8 * 2B = 16B)
#define BF16_BATCH 8

// Tensor Core MMA shape: m16n8k16 (bf16 inputs, f32 accumulate)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)

// ---------------------------------------------------------------------------
// Kernel computes C = A * B^T where:
//   A: M x K, row-major  (each row is contiguous along K)
//   B: N x K, row-major  (B's rows are B^T's columns, so no transpose needed
//                         in memory; both operands stream along K contiguously,
//                         which is the whole point of the B^T formulation)
//   C: M x N, row-major
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned smem_u32addr(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// Async-copy 16 bytes from global memory to shared memory (cp.async.cg bypasses L1)
__device__ __forceinline__ void cp_async_16B(unsigned smem_addr, const void *gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));
}

// wait_group 1: allow 1 outstanding group -> used for double-buffer pipelining
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;\n");
}

// ldmatrix: cooperative shared-memory -> register load of 8x8 b16 matrices.
// x4 = four matrices (full 16x16 A fragment), x2 = two matrices (8x16 B fragment)
__device__ __forceinline__ void ldmatrix_x4(unsigned &r0, unsigned &r1, unsigned &r2, unsigned &r3,unsigned addr) 
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// mma.sync m16n8k16: D += A * B, bf16 inputs, f32 accumulate (C/D are f32)
__device__ __forceinline__ void ldmatrix_x2(unsigned &r0, unsigned &r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16_f32(float *d, const unsigned *a, const unsigned *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// BM/BN/BK : CTA-level tile size
// PM/PN    : number of MMA fragments each warp computes along M/N
template <int BM, int BN, int BK, int PM, int PN>
__global__ void mult_kernel(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
    int64_t M, int64_t N, int64_t K,
    int grid_m, int grid_n, int GROUP) {

    // +8 elements of padding per row avoids shared-memory bank conflicts
    // when ldmatrix / cp.async access consecutive rows
    constexpr int BK_STRIDE   = BK + 8;
    constexpr int NUM_FRAG_M = BM / MMA_M;	 // total MMA fragments in a CTA tile (M dir)
    constexpr int NUM_FRAG_N = BN / MMA_N;	 // total MMA fragments in a CTA tile (N dir)
    constexpr int NWARP_M = NUM_FRAG_M / PM;	 // warp grid layout within CTA
    constexpr int NWARP_N = NUM_FRAG_N / PN;
    constexpr int CHUNKS = BK / BF16_BATCH;      // 16B-wide chunks along K per smem row    
    
    // Global->shared staging work decomposition for the A and B tiles
    constexpr int A_TASKS = BM * CHUNKS;
    constexpr int A_ITERS = A_TASKS / THREADS_PER_BLOCK;
    constexpr int B_TASKS = BN * CHUNKS;
    constexpr int B_ITERS = B_TASKS / THREADS_PER_BLOCK;

    // Number of threads cooperating on one smem row (row-major sweep over the tile)
    constexpr int ROWS_PER_PASS = THREADS_PER_BLOCK / CHUNKS;

    static_assert(PM * PN == NUM_FRAG_M * NUM_FRAG_N / NUM_WARPS);	// all frags covered by warps 
    static_assert(NUM_FRAG_M % PM == 0 && NUM_FRAG_N % PN == 0);
    static_assert(THREADS_PER_BLOCK % BF16_BATCH == 0);
    static_assert(BM * (BK / BF16_BATCH) % (NUM_WARPS * 32) == 0);	// each thread loads whole 16B chunks
    static_assert(BN * (BK / BF16_BATCH) % (NUM_WARPS * 32) == 0);
    static_assert(BK % BF16_BATCH == 0);          
    static_assert(BK % MMA_K == 0);               
    static_assert(A_TASKS >= THREADS_PER_BLOCK); 
    static_assert(B_TASKS >= THREADS_PER_BLOCK);    

    // ---- Threadblock swizzle (like Triton's grouped ordering) ----
    // Re-map linear blockIdx.x into a 2D (pid_m, pid_n) so that consecutive CTAs
    // form GROUP-sized column bands. This improves L2 reuse: CTAs running close
    // together in time share the same B (or A) tiles.
    // The "long" axis (larger grid dimension) is the one being grouped.
    int pid = blockIdx.x;
    int pid_m, pid_n;

    bool n_major = (grid_n > grid_m);		// group along the larger axis
    int g_long  = n_major ? grid_n : grid_m;    // tiles along the grouped axis
    int g_short = n_major ? grid_m : grid_n;	// tiles along the other axis

    int num_pid_in_group = GROUP * g_short;
    int group_id    = pid / num_pid_in_group;
    int first_pid   = group_id * GROUP;
    int group_size  = min(g_long - first_pid, GROUP);   // handle the last partial group

    int p_long  = first_pid + (pid % num_pid_in_group) % group_size;
    int p_short = (pid % num_pid_in_group) / group_size;

    pid_m = n_major ? p_short : p_long;
    pid_n = n_major ? p_long  : p_short;

    const int global_m_start = pid_m * BM;
    const int global_n_start = pid_n * BN;

    int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int group = lane >> 2;	// lane -> accumulator row (0..7)
    const int tig = lane & 3;		// lane -> accumulator column pair (0..3)

    // ---- Shared memory layout (double buffered) ----
    // [ A buf0 | A buf1 | B buf0 | B buf1 ]
    extern __shared__ __align__(16) __nv_bfloat16 smem_raw[];
    __nv_bfloat16 *A_value[2] = { smem_raw, smem_raw + BM * BK_STRIDE };
    __nv_bfloat16 *B_value[2] = { smem_raw + 2 * BM * BK_STRIDE,
                                  smem_raw + 2 * BM *BK_STRIDE + BN * BK_STRIDE
                                };

    // 2D warp grid: warp (wm, wn) owns a PM x PN patch of MMA fragments
    const int wm = warp_id / NWARP_N;
    const int wn = warp_id % NWARP_N;
    const int m_frag_base = wm * PM;
    const int n_frag_base = wn * PN;

    // ---- Per-thread ldmatrix source addresses (m16n8k16 fragment layout) ----
    // A fragment (row-major 16x16): each lane provides rows lane%8 and lane%8+8,
    // two 8-wide halves selected by lane bit 4 -> (a_row, a_col) below.
    // B fragment (col-major 8x16 for row.col MMA): row = lane%8, 8-wide half
    // selected by lane bit 3 -> (b_row, b_col).
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_col = ((lane >> 4) & 1) * 8;
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) * 8;

    // f32 accumulators: 4 values per MMA (2x2 quadrant of the 16x8 output tile)
    float acc[PM][PN][4];
#pragma unroll

    for (int mi = 0; mi < PM; ++mi)
#pragma unroll
        for (int ni = 0; ni < PN; ++ni)
#pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[mi][ni][r] = 0.f;

    // ---- ldmatrix address precomputation (at ko = 0) ----
    // Inside the k-loop only a constant byte offset (MMA_K * 2B) is added,
    // keeping the inner loop free of address arithmetic.
    unsigned a_addr[2][PM], b_addr[2][PN];
#pragma unroll

    for (int buf = 0; buf < 2; ++buf) {
#pragma unroll

        for (int mi = 0; mi < PM; ++mi) {
            a_addr[buf][mi] = smem_u32addr(A_value[buf]
                                           + (m_frag_base + mi) * MMA_M * BK_STRIDE + a_row * BK_STRIDE + a_col);
        }

#pragma unroll

        for (int ni = 0; ni < PN; ++ni) {
            b_addr[buf][ni] = smem_u32addr(B_value[buf]
                                           + (n_frag_base + ni) * MMA_N * BK_STRIDE + b_row * BK_STRIDE + b_col);
        }
    }

    constexpr unsigned KO_OFF = MMA_K * sizeof(__nv_bfloat16);


    // ---- Global -> shared staging for one K-tile into buffer `buf` ----
    // Key optimization: the boundary check is done ONCE per tile (a_full/b_full/
    // k_full) instead of per element. Full tiles take a pure cp.async fast path;
    // boundary tiles fall back to per-chunk checks with zero-padding so the MMA
    // on out-of-range data still computes garbage-free results.
    auto issue_tile = [&](int buf, int k_load_start) {
        const __nv_bfloat16 *Abase = A + (int64_t)global_m_start * K + k_load_start;
        const __nv_bfloat16 *Bbase = B + (int64_t)global_n_start * K + k_load_start;
        const bool a_full = global_m_start + BM <= M;	// tile fully inside M
        const bool b_full = global_n_start + BN <= N;	// tile fully inside N
        const bool k_full = k_load_start + BK <= K;	// tile fully inside K

        if (a_full && k_full) 
        {
	    // Fast path: contiguous rows, pure cp.async, no predicates
            int rA = tid / CHUNKS;
            const int cA = tid % CHUNKS;
            unsigned dst = smem_u32addr(A_value[buf] + rA * BK_STRIDE + cA * BF16_BATCH);
            constexpr unsigned dst_step = ROWS_PER_PASS * BK_STRIDE * sizeof(__nv_bfloat16);
#pragma unroll
            for (int it = 0; it < A_ITERS; ++it) 
            {
                cp_async_16B(dst, Abase + rA * K + cA * BF16_BATCH);
                dst += dst_step;
                rA += ROWS_PER_PASS;
            }
        } 
        else 
        {
	    // Boundary path: zero-fill anything out of range so the MMA
            // contributes 0 for padded elements
            const int cA = tid % CHUNKS;
            const bool k_safe = k_load_start + (cA + 1) * BF16_BATCH <= K;

            for (int rA = tid / CHUNKS; rA < BM; rA += ROWS_PER_PASS) 
            {
                __nv_bfloat16 *dst = A_value[buf] + rA * BK_STRIDE + cA * BF16_BATCH;
                if (global_m_start + rA < M && k_safe)
                    cp_async_16B(smem_u32addr(dst), Abase + rA * K + cA * BF16_BATCH);
                else
                    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
            }
            
        }

        if (b_full && k_full) 
        {
            int rB = tid / CHUNKS;
            const int cB = tid % CHUNKS;    
            //constexpr int rB_step = THREADS_PER_BLOCK >> 3;
            unsigned dst = smem_u32addr(B_value[buf] + rB * BK_STRIDE + cB * BF16_BATCH);
            constexpr unsigned dst_step = ROWS_PER_PASS * BK_STRIDE * sizeof(__nv_bfloat16);
#pragma unroll

            for (int it = 0; it < B_ITERS; ++it) 
            {
                cp_async_16B(dst, Bbase + rB * K + cB * BF16_BATCH);
                dst += dst_step;
                rB += ROWS_PER_PASS;
            }
        } 
        else 
        {

            const int cB = tid % CHUNKS;
            const bool k_safe = k_load_start + (cB + 1) * BF16_BATCH <= K;

            for (int rB = tid / CHUNKS; rB < BN; rB += ROWS_PER_PASS) 
            {
                __nv_bfloat16 *dst = B_value[buf] + rB * BK_STRIDE + cB * BF16_BATCH;

                if ((global_n_start + rB < N) && k_safe)
                    cp_async_16B(smem_u32addr(dst), Bbase + rB * K + cB * BF16_BATCH);
                else
                    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
            }

        }
    };

    // ---- MMA compute on shared-memory buffer `buf` for one K-tile ----
    // Software-pipelines the ldmatrix loads: fragment for step ko+1 is loaded
    // into the "other" register buffer while MMA consumes step ko, hiding
    // shared-memory latency behind Tensor Core work.
    auto compute_tile = [&](int buf) {
        unsigned a_buf[2][PM][4];	// double-buffered A fragments (4 regs each)
        unsigned b_buf[2][PN][2];	// double-buffered B fragments (2 regs each)
        auto load_frag = [&](int b, int ko) {
            const unsigned koff = ko * KO_OFF;
#pragma unroll
            for (int mi = 0; mi < PM; ++mi)
                ldmatrix_x4(a_buf[b][mi][0], a_buf[b][mi][1],
                            a_buf[b][mi][2], a_buf[b][mi][3], a_addr[buf][mi] + koff);

#pragma unroll
            for (int ni = 0; ni < PN; ++ni)
                ldmatrix_x2(b_buf[b][ni][0], b_buf[b][ni][1], b_addr[buf][ni] + koff);
        };

        load_frag(0, 0);	// prime the pipeline

#pragma unroll
        for (int ko = 0; ko + 1 < BK / MMA_K; ++ko) 
        {
            load_frag((ko & 1) ^ 1, ko + 1);	 // prefetch next fragment
#pragma unroll
            for (int ni = 0; ni < PN; ++ni)
#pragma unroll
                for (int mi = 0; mi < PM; ++mi)
                    mma_bf16_f32(acc[mi][ni], a_buf[ko & 1][mi], b_buf[ko & 1][ni]);
        }

        // Drain the last MMA step
        constexpr int ko = BK / MMA_K - 1;
#pragma unroll

        for (int ni = 0; ni < PN; ++ni)
#pragma unroll
            for (int mi = 0; mi < PM; ++mi)
                mma_bf16_f32(acc[mi][ni], a_buf[ko & 1][mi], b_buf[ko & 1][ni]);
    };

    const int num_k_tiles = (int)((K + BK - 1) / BK);

    // ---- Main loop: cp.async double buffering across K-tiles ----
    // Pipeline: while computing tile kt (buffer kt&1), tile kt+1 is being
    // staged asynchronously into the other buffer.
    if (num_k_tiles > 0) {
        issue_tile(0, 0);
        cp_async_commit_group();

        int kt = 0;

        for (; kt + 1 < num_k_tiles; ++kt) {
            issue_tile((kt + 1) & 1, (kt + 1) * BK);	// stage next tile
            cp_async_commit_group();
            cp_async_wait_group_1();	//wait until tile kt has landed (kt+1 may still be in flight)
            __syncthreads();
            compute_tile(kt & 1);
            __syncthreads();	// all warps done with buffer before it gets overwritten
        }

        cp_async_wait_group_0();
        __syncthreads();
        compute_tile(kt & 1);
        __syncthreads();
    }

    // ---- Epilogue: write accumulators back to C ----
    // Accumulator register layout for m16n8k16 (per thread):
    //   c0 -> (row0,     col),  c1 -> (row0,     col+1)
    //   c2 -> (row0 + 8, col),  c3 -> (row0 + 8, col+1)
    // Both paths use ABSOLUTE (global) coordinates to avoid mixing coordinate systems.
    const int row_base = global_m_start + m_frag_base * MMA_M + group;   // absolute row
    const int col_base = global_n_start + n_frag_base * MMA_N + tig * 2; // absolute col
    
    // Per-thread constant check: max row/col this thread will ever write.
    // If the whole 2x2 span is in-bounds for all fragments, take the branch-free
    // fast path (bfloat162 vectorized stores).
    const bool fast_epilogue =
        (row_base + (PM - 1) * MMA_M + 8 < M) &&
        (col_base + (PN - 1) * MMA_N + 1 < N);

    if (fast_epilogue) {
        // Fast path: no per-element bounds checks. N even guaranteed by host,
        // so __nv_bfloat162 (4B) stores are alignment-safe.
#pragma unroll
        for (int mi = 0; mi < PM; ++mi) {
            __nv_bfloat16 *Crow = C + (int64_t)(row_base + mi * MMA_M) * N + col_base;
#pragma unroll

            for (int ni = 0; ni < PN; ++ni) {
                __nv_bfloat16 *p = Crow + ni * MMA_N;
                *reinterpret_cast<__nv_bfloat162 *>(p) =
                    __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
                *reinterpret_cast<__nv_bfloat162 *>(p + 8 * N) =
                    __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    } else {
        // Boundary path: element-wise guards, semantically identical to a naive
        // per-element bounds-checked version
#pragma unroll
        for (int mi = 0; mi < PM; ++mi) {
#pragma unroll

            for (int ni = 0; ni < PN; ++ni) {
                const int row0 = row_base + mi * MMA_M;
                const int row1 = row0 + 8;
                const int col = col_base + ni * MMA_N;

                if (col + 1 < N) {
		    // Full bfloat162 store (two consecutive columns in-bounds)
                    if (row0 < M)
                        *reinterpret_cast<__nv_bfloat162 *>(&C[(int64_t)row0 * N + col]) =
                            __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);

                    if (row1 < M)
                        *reinterpret_cast<__nv_bfloat162 *>(&C[(int64_t)row1 * N + col]) =
                            __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
                } else if (col < N) {
		    // Only one column in-bounds: scalar store (odd N edge case)
                    if (row0 < M)
                        C[(int64_t)row0 * N + col] = __float2bfloat16(acc[mi][ni][0]);

                    if (row1 < M)
                        C[(int64_t)row1 * N + col] = __float2bfloat16(acc[mi][ni][2]);
                }
            }
        }
    }
}

// Launch wrapper: sets max dynamic smem (needed when > 48KB) once per template
// instantiation, computes the grid, and picks GROUP_M for the swizzle.
template <int BM, int BN, int BK, int PM, int PN>
void launch(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
            int64_t M, int64_t N, int64_t K) {
    // Double-buffered smem: 2 buffers each for A and B tiles.
    // The +8 padding is included in BK_STRIDE via (BK + 8).
    constexpr int smem_bytes = 2 * (BM + BN) * (BK + 8) * sizeof(__nv_bfloat16);

    static bool attr_set = false;   // per-instantiation static; set only once

    if (!attr_set) {
        cudaFuncSetAttribute(mult_kernel<BM, BN, BK, PM, PN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);          
	// Required when smem exceeds the default 48KB limit
        attr_set = true;
    }

    const int grid_m = (int)((M + BM - 1) / BM);
    const int grid_n = (int)((N + BN - 1) / BN);

    int GROUP_M = std::min(8, grid_m);

    mult_kernel<BM, BN, BK, PM, PN> <<< grid_m *grid_n, THREADS_PER_BLOCK, smem_bytes>>>(
        A, B, C, M, N, K, grid_m, grid_n, GROUP_M);
}

// Host entry: picks a tile configuration based on the aspect ratio M/N.
// Tall-skinny (M >> N): use wide-B tiles / more N-fragments per warp to keep
// the B-tile residency high; the symmetric case for wide matrices.
void run_kernel(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
    int64_t M, int64_t N, int64_t K) {
    float k = (float)M/(float)N;
    if (k>=8.f)          
        launch<256, 128, 64, 8, 4>(A, B, C, M, N, K);
    else if (k<4.f && k<=0.5f)       
        launch<128, 256, 64, 8, 4>(A, B, C, M, N, K);
    else                      
        launch<256, 128, 64, 4, 8>(A, B, C, M, N, K);
}