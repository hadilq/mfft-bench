/* B5 — GPU Boolean GEMM: bit-pack along k, AND + popcount (ballot optional).
 *
 * C_ij = |{k : A_ik == 1 && B_kj == 1}| = popcount(row_i(A) & col_j(B))
 * when A,B are 0-1.  Words are uint32 (32 k's per word) for wide GPU popc.
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BP_CK
#define BP_CK(stmt) do {                                       \
    cudaError_t _e = (stmt);                                   \
    if (_e != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                       \
        exit(1);                                               \
    }                                                          \
} while (0)
#endif

static inline int bp_nwords(int n) { return (n + 31) >> 5; }

/* Pack bit 0 of each entry along k. as_cols=0: rows of M; as_cols=1: cols as rows. */
__global__ void k_bp_pack(const signed char *__restrict__ M,
                          unsigned *__restrict__ out,
                          int n, int nwords, int as_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    unsigned *dst = out + (size_t)row * nwords;
    for (int w = 0; w < nwords; w++) {
        unsigned bits = 0;
        int k0 = w << 5;
        for (int t = 0; t < 32; t++) {
            int k = k0 + t;
            if (k >= n) break;
            signed char v = as_cols ? M[(size_t)k * n + row] : M[(size_t)row * n + k];
            if (v) bits |= 1u << t;
        }
        dst[w] = bits;
    }
}

/* Warp-cooperative pack of one row: each lane handles one k in a 32-wide chunk,
 * __ballot_sync packs the 0-1 predicates into a mask. */
__global__ void k_bp_pack_ballot(const signed char *__restrict__ M,
                                 unsigned *__restrict__ out,
                                 int n, int nwords, int as_cols)
{
    int row = blockIdx.x;          /* one block per row */
    if (row >= n) return;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0 && warp == 0) {
        /* only using 32 threads recommended */
    }
    if (threadIdx.x >= 32) return; /* one warp packs the row word-by-word */

    unsigned *dst = out + (size_t)row * nwords;
    for (int w = 0; w < nwords; w++) {
        int k = (w << 5) + lane;
        int pred = 0;
        if (k < n) {
            signed char v = as_cols ? M[(size_t)k * n + row] : M[(size_t)row * n + k];
            pred = (v != 0);
        }
        unsigned mask = __ballot_sync(0xffffffffu, pred);
        if (lane == 0)
            dst[w] = mask;
    }
    (void)nwarps; (void)warp;
}

/* One thread per C[i,j]: popcount of AND along k. */
__global__ void k_bp_gemm(const unsigned *__restrict__ A,
                          const unsigned *__restrict__ B,
                          int *__restrict__ C,
                          int n, int nwords)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    const unsigned *ar = A + (size_t)i * nwords;
    const unsigned *br = B + (size_t)j * nwords;
    int acc = 0;
    for (int w = 0; w < nwords; w++)
        acc += __popc(ar[w] & br[w]);
    C[(size_t)i * n + j] = acc;
}

/* Tiled: block computes TILE x TILE of C; shared memory holds A/B bit words. */
#ifndef BP_TILE
#define BP_TILE 16
#endif
__global__ void k_bp_gemm_tiled(const unsigned *__restrict__ A,
                                const unsigned *__restrict__ B,
                                int *__restrict__ C,
                                int n, int nwords)
{
    __shared__ unsigned As[BP_TILE][8];  /* up to 256 k in shared (8 words) */
    __shared__ unsigned Bs[BP_TILE][8];
    int i0 = blockIdx.y * BP_TILE;
    int j0 = blockIdx.x * BP_TILE;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    int i = i0 + ti;
    int j = j0 + tj;
    int acc = 0;

    for (int w0 = 0; w0 < nwords; w0 += 8) {
        int nw = nwords - w0;
        if (nw > 8) nw = 8;
        /* load A rows i0.. and B cols j0.. words w0.. */
        if (ti < BP_TILE && tj < nw) {
            int ii = i0 + ti;
            if (ii < n)
                As[ti][tj] = A[(size_t)ii * nwords + w0 + tj];
            else
                As[ti][tj] = 0;
        }
        if (tj < BP_TILE && ti < nw) {
            int jj = j0 + tj;
            if (jj < n)
                Bs[tj][ti] = B[(size_t)jj * nwords + w0 + ti];
            else
                Bs[tj][ti] = 0;
        }
        __syncthreads();
        if (i < n && j < n) {
            for (int t = 0; t < nw; t++)
                acc += __popc(As[ti][t] & Bs[tj][t]);
        }
        __syncthreads();
    }
    if (i < n && j < n)
        C[(size_t)i * n + j] = acc;
}

/* Host: allocate, pack, GEMM. A,B are host int8 0-1 row-major n*n.
 * Returns device ms for (pack optional + gemm). If include_pack=0, packs once outside. */
static float bp_gpu_boolgemm(int n, const signed char *hA, const signed char *hB,
                             int *hC, int include_pack, int use_ballot_pack,
                             int use_tiled, int reps)
{
    int nw = bp_nwords(n);
    size_t nn = (size_t)n * n;
    size_t plane = (size_t)n * nw;

    signed char *dA, *dB;
    unsigned *dAp, *dBp;
    int *dC;
    BP_CK(cudaMalloc(&dA, nn));
    BP_CK(cudaMalloc(&dB, nn));
    BP_CK(cudaMalloc(&dAp, plane * sizeof(unsigned)));
    BP_CK(cudaMalloc(&dBp, plane * sizeof(unsigned)));
    BP_CK(cudaMalloc(&dC, nn * sizeof(int)));
    BP_CK(cudaMemcpy(dA, hA, nn, cudaMemcpyHostToDevice));
    BP_CK(cudaMemcpy(dB, hB, nn, cudaMemcpyHostToDevice));

    dim3 packBlk(128);
    dim3 packGrd((n + 127) / 128);
    dim3 ballBlk(32);
    dim3 ballGrd(n);

    dim3 gemmBlk(16, 16);
    dim3 gemmGrd((n + 15) / 16, (n + 15) / 16);

    /* warm-up */
    if (use_ballot_pack) {
        k_bp_pack_ballot<<<ballGrd, ballBlk>>>(dA, dAp, n, nw, 0);
        k_bp_pack_ballot<<<ballGrd, ballBlk>>>(dB, dBp, n, nw, 1);
    } else {
        k_bp_pack<<<packGrd, packBlk>>>(dA, dAp, n, nw, 0);
        k_bp_pack<<<packGrd, packBlk>>>(dB, dBp, n, nw, 1);
    }
    if (use_tiled)
        k_bp_gemm_tiled<<<gemmGrd, dim3(BP_TILE, BP_TILE)>>>(dAp, dBp, dC, n, nw);
    else
        k_bp_gemm<<<gemmGrd, gemmBlk>>>(dAp, dBp, dC, n, nw);
    BP_CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    BP_CK(cudaEventCreate(&t0));
    BP_CK(cudaEventCreate(&t1));
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        BP_CK(cudaEventRecord(t0));
        if (include_pack) {
            if (use_ballot_pack) {
                k_bp_pack_ballot<<<ballGrd, ballBlk>>>(dA, dAp, n, nw, 0);
                k_bp_pack_ballot<<<ballGrd, ballBlk>>>(dB, dBp, n, nw, 1);
            } else {
                k_bp_pack<<<packGrd, packBlk>>>(dA, dAp, n, nw, 0);
                k_bp_pack<<<packGrd, packBlk>>>(dB, dBp, n, nw, 1);
            }
        }
        if (use_tiled)
            k_bp_gemm_tiled<<<gemmGrd, dim3(BP_TILE, BP_TILE)>>>(dAp, dBp, dC, n, nw);
        else
            k_bp_gemm<<<gemmGrd, gemmBlk>>>(dAp, dBp, dC, n, nw);
        BP_CK(cudaEventRecord(t1));
        BP_CK(cudaEventSynchronize(t1));
        float ms;
        BP_CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }

    BP_CK(cudaMemcpy(hC, dC, nn * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dA); cudaFree(dB); cudaFree(dAp); cudaFree(dBp); cudaFree(dC);
    return best;
}

/* CPU reference for 0-1 GEMM (exact). */
static void bp_cpu_ref(int *C, const signed char *A, const signed char *B, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++)
                acc += (A[(size_t)i * n + k] & 1) & (B[(size_t)k * n + j] & 1);
            C[(size_t)i * n + j] = acc;
        }
    }
}

static int bp_check(const int *got, const int *ref, int n)
{
    size_t nn = (size_t)n * n;
    for (size_t i = 0; i < nn; i++)
        if (got[i] != ref[i]) return 0;
    return 1;
}
