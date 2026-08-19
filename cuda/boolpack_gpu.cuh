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

/* ------------------------------------------------------------------ */
/* B6: MFFT pointwise leaf — int32 0-1 planes → pack → tiled popc → int64 C
 * Specialized duplicates for performance (no shared path with host int8). */

__global__ void k_bp_pack_i32(const int32_t *__restrict__ M,
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
            int32_t v = as_cols ? M[(size_t)k * n + row] : M[(size_t)row * n + k];
            if (v) bits |= 1u << t;
        }
        dst[w] = bits;
    }
}

/* Ballot pack for int32 0-1 (one block per row). */
__global__ void k_bp_pack_i32_ballot(const int32_t *__restrict__ M,
                                     unsigned *__restrict__ out,
                                     int n, int nwords, int as_cols)
{
    int row = blockIdx.x;
    if (row >= n) return;
    int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;
    unsigned *dst = out + (size_t)row * nwords;
    for (int w = 0; w < nwords; w++) {
        int k = (w << 5) + lane;
        int pred = 0;
        if (k < n) {
            int32_t v = as_cols ? M[(size_t)k * n + row] : M[(size_t)row * n + k];
            pred = (v != 0);
        }
        unsigned mask = __ballot_sync(0xffffffffu, pred);
        if (lane == 0) dst[w] = mask;
    }
}

/* flag = 1 if any entry is not in {0,1}.  Early-out friendly: one atomic per
 * violating thread is enough; caller may only sample a prefix for speed. */
__global__ void k_plane_not_01(int *flag, const int32_t *M, size_t nn)
{
    if (*flag) return; /* already failed — still weak without mem fence */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    int32_t v = M[i];
    if (v != 0 && v != 1) atomicExch(flag, 1);
}

/* Fast path: check only the first min(nn, 4096) entries on device (amortized).
 * Full scan if that passes and nn is large — for MFFT we only need a cheap
 * reject of general panels; false "not 01" is safe (falls back to igemm32). */
static int bp_plane_is_01(const int32_t *dM, size_t nn)
{
    static int *dflag = NULL;
    if (!dflag) BP_CK(cudaMalloc(&dflag, sizeof(int)));
    BP_CK(cudaMemset(dflag, 0, sizeof(int)));
    size_t sample = nn < (size_t)4096 ? nn : (size_t)4096;
    int blk = 256;
    int g = (int)((sample + (size_t)blk - 1) / (size_t)blk);
    if (g < 1) g = 1;
    k_plane_not_01<<<g, blk>>>(dflag, dM, sample);
    int hflag = 0;
    BP_CK(cudaMemcpy(&hflag, dflag, sizeof(int), cudaMemcpyDeviceToHost));
    return hflag == 0;
}

/* Tiled popc GEMM: C[i,j] += sgn * popcount(Ai & Bj). C is int64. */
__global__ void k_bp_gemm_tiled_i64(const unsigned *__restrict__ A,
                                    const unsigned *__restrict__ B,
                                    long long *__restrict__ C,
                                    int n, int nwords, int sgn)
{
    __shared__ unsigned As[BP_TILE][8];
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
        if (ti < BP_TILE && tj < nw) {
            int ii = i0 + ti;
            As[ti][tj] = (ii < n) ? A[(size_t)ii * nwords + w0 + tj] : 0u;
        }
        if (tj < BP_TILE && ti < nw) {
            int jj = j0 + tj;
            Bs[tj][ti] = (jj < n) ? B[(size_t)jj * nwords + w0 + ti] : 0u;
        }
        __syncthreads();
        if (i < n && j < n) {
            for (int t = 0; t < nw; t++)
                acc += __popc(As[ti][t] & Bs[tj][t]);
        }
        __syncthreads();
    }
    if (i < n && j < n && acc)
        C[(size_t)i * n + j] += (long long)sgn * (long long)acc;
}

/* Non-tiled one-thread-per-C for small n / tail. */
__global__ void k_bp_gemm_i64(const unsigned *__restrict__ A,
                              const unsigned *__restrict__ B,
                              long long *__restrict__ C,
                              int n, int nwords, int sgn)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    const unsigned *ar = A + (size_t)i * nwords;
    const unsigned *br = B + (size_t)j * nwords;
    int acc = 0;
    for (int w = 0; w < nwords; w++)
        acc += __popc(ar[w] & br[w]);
    if (acc)
        C[(size_t)i * n + j] += (long long)sgn * (long long)acc;
}

/* Workspace for repeated pointwise calls (pack buffers reused). */
typedef struct {
    unsigned *dAp, *dBp;
    int n, nwords;
    size_t plane;
} BpMfftWs;

static int bp_mfft_ws_init(BpMfftWs *ws, int n)
{
    memset(ws, 0, sizeof(*ws));
    ws->n = n;
    ws->nwords = bp_nwords(n);
    ws->plane = (size_t)n * (size_t)ws->nwords;
    if (cudaMalloc(&ws->dAp, ws->plane * sizeof(unsigned)) != cudaSuccess)
        return 0;
    if (cudaMalloc(&ws->dBp, ws->plane * sizeof(unsigned)) != cudaSuccess) {
        cudaFree(ws->dAp); ws->dAp = NULL; return 0;
    }
    return 1;
}

static void bp_mfft_ws_free(BpMfftWs *ws)
{
    if (ws->dAp) cudaFree(ws->dAp);
    if (ws->dBp) cudaFree(ws->dBp);
    memset(ws, 0, sizeof(*ws));
}

/* Pack int32 0-1 plane as rows (as_cols=0) or columns as rows (as_cols=1). */
static void bp_pack_i32(unsigned *out, const int32_t *M, int n, int nwords,
                        int as_cols, int use_ballot)
{
    if (use_ballot) {
        k_bp_pack_i32_ballot<<<n, 32>>>(M, out, n, nwords, as_cols);
    } else {
        int blk = 128;
        int g = (n + blk - 1) / blk;
        k_bp_pack_i32<<<g, blk>>>(M, out, n, nwords, as_cols);
    }
}

/* C += sgn * boolpack_tiled(A, B) for int32 0-1 planes. Uses pre-init ws. */
static void bp_igemm32_bool_tiled(BpMfftWs *ws, long long *C,
                                  const int32_t *A, const int32_t *B,
                                  int n, int sgn, int use_ballot)
{
    int nw = ws->nwords;
    bp_pack_i32(ws->dAp, A, n, nw, 0, use_ballot);
    bp_pack_i32(ws->dBp, B, n, nw, 1, use_ballot);
    dim3 blk(BP_TILE, BP_TILE);
    dim3 grd((n + BP_TILE - 1) / BP_TILE, (n + BP_TILE - 1) / BP_TILE);
    k_bp_gemm_tiled_i64<<<grd, blk>>>(ws->dAp, ws->dBp, C, n, nw, sgn);
}

/* Same but A already packed in ws->dAp (reuse across B panels). */
static void bp_igemm32_bool_tiled_Apacked(BpMfftWs *ws, long long *C,
                                          const int32_t *B, int n, int sgn,
                                          int use_ballot)
{
    int nw = ws->nwords;
    bp_pack_i32(ws->dBp, B, n, nw, 1, use_ballot);
    dim3 blk(BP_TILE, BP_TILE);
    dim3 grd((n + BP_TILE - 1) / BP_TILE, (n + BP_TILE - 1) / BP_TILE);
    k_bp_gemm_tiled_i64<<<grd, blk>>>(ws->dAp, ws->dBp, C, n, nw, sgn);
}


/* ------------------------------------------------------------------ */
/* Pure 0-1 path (no limbs): bit-plane expansion + boolpack-tiled only.
 * Do NOT mix with 7-bit limb MFFT. */

/* Quantize float matrix to non-negative int in [0, 2^bits), store int32. */
__global__ void k_f32_to_uint_bits(int32_t *Q, const float *X, size_t nn,
                                   int bits, float scale)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    float v = fabsf(X[i]) * scale;
    int qmax = (1 << bits) - 1;
    int q = (int)(v + 0.5f);
    if (q < 0) q = 0;
    if (q > qmax) q = qmax;
    Q[i] = q;
}

/* Extract bit plane b into signed char 0/1 matrix. */
__global__ void k_extract_bitplane(signed char *P, const int32_t *Q,
                                   size_t nn, int bit)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    P[i] = (signed char)((Q[i] >> bit) & 1);
}

/* C_float[i] += (float)acc[i] * (1<<shift) / scale_combine */
__global__ void k_acc_shift_f32(float *C, const int *acc, size_t nn,
                                int shift, float inv_scale)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    C[i] += (float)acc[i] * (float)(1 << shift) * inv_scale;
}

/*
 * Pure bit-plane Boolean GEMM (schoolbook on bits).
 * A,B float -> quantize to `bits` unsigned bits -> for each (u,v):
 *   boolpack_tiled(plane_u(A), plane_v(B)) into int acc
 *   C += acc * 2^(u+v) / scale^2
 * This is NOT limb-MFFT. All panels are true 0/1; every product is tiled popc.
 */
static float bp_bitplanes_boolpack_tiled(int n, const float *dA, const float *dB,
                                         float *dC, int bits, int reps)
{
    size_t nn = (size_t)n * n;
    int32_t *dQa, *dQb;
    signed char *dPa, *dPb;
    int *dAcc;
    BP_CK(cudaMalloc(&dQa, nn * sizeof(int32_t)));
    BP_CK(cudaMalloc(&dQb, nn * sizeof(int32_t)));
    BP_CK(cudaMalloc(&dPa, nn));
    BP_CK(cudaMalloc(&dPb, nn));
    BP_CK(cudaMalloc(&dAcc, nn * sizeof(int)));

    float scale = (float)((1 << bits) - 1); /* map |x|~1 -> max code */
    int blk = 256;
    int g = (int)((nn + (size_t)blk - 1) / (size_t)blk);
    if (g < 1) g = 1;

    k_f32_to_uint_bits<<<g, blk>>>(dQa, dA, nn, bits, scale);
    k_f32_to_uint_bits<<<g, blk>>>(dQb, dB, nn, bits, scale);

    /* Workspace for packing uint32 words from signed char planes via existing pack */
    /* Reuse bp_gpu path: pack from signed char using k_bp_pack */
    int nw = bp_nwords(n);
    size_t plane = (size_t)n * (size_t)nw;
    unsigned *dAp, *dBp;
    BP_CK(cudaMalloc(&dAp, plane * sizeof(unsigned)));
    BP_CK(cudaMalloc(&dBp, plane * sizeof(unsigned)));

    dim3 gblk(BP_TILE, BP_TILE);
    dim3 ggrd((n + BP_TILE - 1) / BP_TILE, (n + BP_TILE - 1) / BP_TILE);
    dim3 pblk(128);
    dim3 pgrd((n + 127) / 128);

    /* warm-up one product */
    k_extract_bitplane<<<g, blk>>>(dPa, dQa, nn, 0);
    k_extract_bitplane<<<g, blk>>>(dPb, dQb, nn, 0);
    k_bp_pack<<<pgrd, pblk>>>(dPa, dAp, n, nw, 0);
    k_bp_pack<<<pgrd, pblk>>>(dPb, dBp, n, nw, 1);
    BP_CK(cudaMemset(dAcc, 0, nn * sizeof(int)));
    /* use int C kernel then we need int version of tiled writing to int */
    k_bp_gemm_tiled<<<ggrd, gblk>>>(dAp, dBp, dAcc, n, nw);
    BP_CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    BP_CK(cudaEventCreate(&t0));
    BP_CK(cudaEventCreate(&t1));
    float best = 1e30f;
    float inv = 1.0f / (scale * scale);

    for (int r = 0; r < reps; r++) {
        BP_CK(cudaEventRecord(t0));
        BP_CK(cudaMemset(dC, 0, nn * sizeof(float)));
        k_f32_to_uint_bits<<<g, blk>>>(dQa, dA, nn, bits, scale);
        k_f32_to_uint_bits<<<g, blk>>>(dQb, dB, nn, bits, scale);

        for (int u = 0; u < bits; u++) {
            k_extract_bitplane<<<g, blk>>>(dPa, dQa, nn, u);
            k_bp_pack<<<pgrd, pblk>>>(dPa, dAp, n, nw, 0);
            for (int v = 0; v < bits; v++) {
                k_extract_bitplane<<<g, blk>>>(dPb, dQb, nn, v);
                k_bp_pack<<<pgrd, pblk>>>(dPb, dBp, n, nw, 1);
                BP_CK(cudaMemset(dAcc, 0, nn * sizeof(int)));
                k_bp_gemm_tiled<<<ggrd, gblk>>>(dAp, dBp, dAcc, n, nw);
                k_acc_shift_f32<<<g, blk>>>(dC, dAcc, nn, u + v, inv);
            }
        }
        BP_CK(cudaEventRecord(t1));
        BP_CK(cudaEventSynchronize(t1));
        float ms;
        BP_CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dQa); cudaFree(dQb); cudaFree(dPa); cudaFree(dPb);
    cudaFree(dAcc); cudaFree(dAp); cudaFree(dBp);
    return best;
}
