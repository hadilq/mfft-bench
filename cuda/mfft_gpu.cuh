/* cuda/mfft_gpu.cuh -- GPU MFFT plan, op-list, transform kernels, packing.
 *
 * Isolated from the benchmark driver so the algorithm can evolve without
 * touching cuBLAS baselines / Ozaki / Strassen / table printing.
 *
 * Depends on: CUDA runtime, <cstring>, and the host helpers
 *   grid_for(), CK()  (provided by the including TU before this include).
 */
#pragma once
#ifndef MFFT_GPU_CUH_
#define MFFT_GPU_CUH_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { MFFT_OP_START = 0, MFFT_OP_STEP = 1, MFFT_OP_END = 2 };


typedef struct { int32_t u, v; int16_t sign, mode; } MfftOp;

typedef struct {
    int L;           /* original (unpadded) limb count used for encode */
    int Lpad;        /* ncoeffs * S >= L; high limbs are zero */
    int S, NB, K, g, ncoeffs;
    long nfw, niv;
    MfftOp *fwops, *ivops;           /* host */
} MfftPlanGpu;

static int mfft_ilog2(int x) { int r = 0; while ((1 << r) < x) r++; return r; }
static int mfft_next_pow2(int x)
{
    if (x < 2) return 2;
    if ((x & (x - 1)) == 0) return x;
    return 1 << mfft_ilog2(x);
}

/* Choose S = 2^σ to minimise NB·K².  L need not be a power of two: we take
 * ncoeffs = ceil(L/S) coefficient blocks and set NB = next_pow2(2·ncoeffs)
 * so the cyclic transform covers the linear limb convolution.  Matrix size
 * n is independent — it is never padded here (already a multiple of 16 for
 * int8 alignment). */
static int mfft_plan_init_gpu(MfftPlanGpu *p, int L_raw)
{
    int L = L_raw < 2 ? 2 : L_raw;
    int best_prod = 0x7fffffff;
    int best_S = 0, best_NB = 0, best_K = 0, best_g = 0, best_nc = 0, best_Lpad = 0;
    int max_sigma = mfft_ilog2(L) + 2;
    for (int sigma = 1; sigma <= max_sigma; sigma++) {
        int S = 1 << sigma;
        if (2 * S * S < L) continue;          /* bit-growth guard from the CPU plan */
        int ncoeffs = (L + S - 1) / S;        /* ceil(L/S) */
        if (ncoeffs < 1) continue;
        int NB = mfft_next_pow2(2 * ncoeffs); /* room for linear convolution */
        if (NB < 2) continue;
        int K = 2 * S;
        /* omega = y^g with y^K = -1, order 2K; need NB | 2K for a subgroup */
        if ((2 * K) % NB) continue;
        int g = 2 * K / NB;
        long long prod = (long long)NB * K * K;
        if (prod > 0 && prod < best_prod) {
            best_prod = (int)prod;
            best_S = S; best_NB = NB; best_K = K; best_g = g;
            best_nc = ncoeffs; best_Lpad = ncoeffs * S;
        }
    }
    if (best_S == 0) return -1;
    p->L = L;
    p->Lpad = best_Lpad;
    p->S = best_S; p->NB = best_NB; p->K = best_K; p->g = best_g;
    p->ncoeffs = best_nc;
    p->nfw = p->niv = 0;
    p->fwops = p->ivops = NULL;
    return 0;
}

static MfftOp *mfft_build_ops(int NB, int K, int g, int inverse, long *nops_out)
{
    int mod = 2 * K;
    long cap = 0;
    for (int len = 2; len <= NB; len <<= 1) cap += (long)NB / 2 * (K + K);
    MfftOp *ops = (MfftOp *)malloc((size_t)cap * sizeof(MfftOp));
    char *seen = (char *)malloc((size_t)K);
    if (!ops || !seen) { free(ops); free(seen); *nops_out = 0; return NULL; }
    long m = 0;
    int lens[32], nl = 0;
    if (inverse) for (int len = 2; len <= NB; len <<= 1) lens[nl++] = len;
    else         for (int len = NB; len >= 2; len >>= 1) lens[nl++] = len;

    for (int li = 0; li < nl; li++) {
        int len = lens[li], h = len >> 1, step = NB / len;
        for (int j0 = 0; j0 < NB; j0 += len)
            for (int t = 0; t < h; t++) {
                int e = (int)(((long long)g * t * step) % mod);
                if (inverse) e = (mod - e) % mod;
                int pb = j0 + t, qb = j0 + t + h;
                memset(seen, 0, (size_t)K);
                for (int c0 = 0; c0 < K; c0++) {
                    if (seen[c0]) continue;
                    int c = c0;
                    seen[c] = 1;
                    ops[m].u = pb * K + c0; ops[m].v = qb * K + c0;
                    ops[m].sign = 1; ops[m].mode = MFFT_OP_START; m++;
                    for (;;) {
                        int tgt = c + e, w = 0;
                        while (tgt >= K) { tgt -= K; w++; }
                        int sg = (w & 1) ? -1 : 1;
                        if (tgt == c0) {
                            ops[m].u = pb * K + tgt; ops[m].v = qb * K + tgt;
                            ops[m].sign = (int16_t)sg; ops[m].mode = MFFT_OP_END; m++;
                            break;
                        }
                        seen[tgt] = 1;
                        ops[m].u = pb * K + tgt; ops[m].v = qb * K + tgt;
                        ops[m].sign = (int16_t)sg; ops[m].mode = MFFT_OP_STEP; m++;
                        c = tgt;
                    }
                }
            }
    }
    free(seen);
    *nops_out = m;
    return ops;
}

static int mfft_plan_build_ops(MfftPlanGpu *p)
{
    p->fwops = mfft_build_ops(p->NB, p->K, p->g, 0, &p->nfw);
    p->ivops = mfft_build_ops(p->NB, p->K, p->g, 1, &p->niv);
    return (p->fwops && p->ivops) ? 0 : -1;
}

static void mfft_plan_free_gpu(MfftPlanGpu *p)
{
    free(p->fwops); free(p->ivops);
    p->fwops = p->ivops = NULL;
}

/* One fused FFT op on nn-length int32 coefficient planes. */
__global__ void k_mfft_op32(int32_t *x, size_t nn, int32_t u, int32_t v,
                            int16_t sign, int16_t mode, int32_t *cur)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    int32_t *up = x + (size_t)u * nn;
    int32_t *vp = x + (size_t)v * nn;
    if (mode == MFFT_OP_START) {
        int32_t a = up[i], b = vp[i];
        up[i] = a + b; cur[i] = a - b;
    } else if (mode == MFFT_OP_STEP) {
        int32_t a = up[i], b = vp[i], nx = a - b;
        up[i] = a + b;
        vp[i] = (sign > 0) ? cur[i] : -cur[i];
        cur[i] = nx;
    } else {
        vp[i] = (sign > 0) ? cur[i] : -cur[i];
    }
}

/* Inverse ops on int64 (after pointwise). */
__global__ void k_mfft_op64(long long *x, size_t nn, int32_t u, int32_t v,
                            int16_t sign, int16_t mode, long long *cur)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    long long *up = x + (size_t)u * nn;
    long long *vp = x + (size_t)v * nn;
    if (mode == MFFT_OP_START) {
        cur[i] = vp[i];
    } else if (mode == MFFT_OP_STEP) {
        long long nx = vp[i], tv = (sign > 0) ? cur[i] : -cur[i], a = up[i];
        up[i] = a + tv; vp[i] = a - tv; cur[i] = nx;
    } else {
        long long tv = (sign > 0) ? cur[i] : -cur[i], a = up[i];
        up[i] = a + tv; vp[i] = a - tv;
    }
}

/* Fold inverse-FFT output: acc[w] += Ch[b,c] / NB for w = b*S + c. */
__global__ void k_mfft_fold(int *acc, const long long *Ch, size_t nn,
                            int NB, int K, int S, int planes)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    for (int b = 0; b < NB; b++) {
        for (int c = 0; c < K; c++) {
            int w = b * S + c;
            if (w >= planes) continue;
            long long v = Ch[((size_t)b * K + c) * nn + i];
            acc[(size_t)w * nn + i] += (int)(v / NB);
        }
    }
}

/* Integer GEMM for MFFT pointwise: C += sgn * A * B, int32 A/B, int64 C.
 * 32×32 tile, each thread owns a 4×4 register block (8×8 threads). */
__global__ void k_igemm32(long long *C, const int32_t *A, const int32_t *B,
                          int n, int sgn)
{
    const int TS = 32, RT = 4;
    __shared__ int As[32][32], Bs[32][32];
    int ty = (int)threadIdx.y, tx = (int)threadIdx.x;
    int row0 = (int)blockIdx.y * TS + ty * RT;
    int col0 = (int)blockIdx.x * TS + tx * RT;
    long long sum[4][4];
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < RT; j++) sum[i][j] = 0;

    for (int kk = 0; kk < n; kk += TS) {
        #pragma unroll
        for (int i = 0; i < RT; i++)
            #pragma unroll
            for (int j = 0; j < RT; j++) {
                int r = ty * RT + i, c = tx * RT + j;
                int ar = (int)blockIdx.y * TS + r, ac = kk + c;
                int br = kk + r, bc = (int)blockIdx.x * TS + c;
                As[r][c] = (ar < n && ac < n) ? A[(size_t)ar * n + ac] : 0;
                Bs[r][c] = (br < n && bc < n) ? B[(size_t)br * n + bc] : 0;
            }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TS; t++) {
            int a0 = As[ty*RT+0][t], a1 = As[ty*RT+1][t];
            int a2 = As[ty*RT+2][t], a3 = As[ty*RT+3][t];
            int b0 = Bs[t][tx*RT+0], b1 = Bs[t][tx*RT+1];
            int b2 = Bs[t][tx*RT+2], b3 = Bs[t][tx*RT+3];
            sum[0][0] += (long long)a0*b0; sum[0][1] += (long long)a0*b1;
            sum[0][2] += (long long)a0*b2; sum[0][3] += (long long)a0*b3;
            sum[1][0] += (long long)a1*b0; sum[1][1] += (long long)a1*b1;
            sum[1][2] += (long long)a1*b2; sum[1][3] += (long long)a1*b3;
            sum[2][0] += (long long)a2*b0; sum[2][1] += (long long)a2*b1;
            sum[2][2] += (long long)a2*b2; sum[2][3] += (long long)a2*b3;
            sum[3][0] += (long long)a3*b0; sum[3][1] += (long long)a3*b1;
            sum[3][2] += (long long)a3*b2; sum[3][3] += (long long)a3*b3;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < RT; j++) {
            int r = row0 + i, c = col0 + j;
            if (r < n && c < n) {
                if (sgn > 0) C[(size_t)r * n + c] += sum[i][j];
                else         C[(size_t)r * n + c] -= sum[i][j];
            }
        }
}

static void igemm32_rm(long long *C, const int32_t *A, const int32_t *B,
                       int n, int sgn)
{
    dim3 blk(8, 8);
    dim3 grd((n + 31) / 32, (n + 31) / 32);
    k_igemm32<<<grd, blk>>>(C, A, B, n, sgn);
}

/* Entire op list in one launch: each thread owns one matrix element and
 * walks all ops with a register-local carry (no global carry buffer). */
__global__ void k_mfft_run32(int32_t *x, size_t nn, const MfftOp *ops, int nops)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    int32_t cur = 0;
    for (int o = 0; o < nops; o++) {
        int32_t *u = x + (size_t)ops[o].u * nn + i;
        int32_t *v = x + (size_t)ops[o].v * nn + i;
        int16_t mode = ops[o].mode, sign = ops[o].sign;
        if (mode == MFFT_OP_START) {
            int32_t a = *u, b = *v;
            *u = a + b; cur = a - b;
        } else if (mode == MFFT_OP_STEP) {
            int32_t a = *u, b = *v, nx = a - b;
            *u = a + b;
            *v = (sign > 0) ? cur : -cur;
            cur = nx;
        } else {
            *v = (sign > 0) ? cur : -cur;
        }
    }
}
__global__ void k_mfft_run64(long long *x, size_t nn, const MfftOp *ops, int nops)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    long long cur = 0;
    for (int o = 0; o < nops; o++) {
        long long *u = x + (size_t)ops[o].u * nn + i;
        long long *v = x + (size_t)ops[o].v * nn + i;
        int16_t mode = ops[o].mode, sign = ops[o].sign;
        if (mode == MFFT_OP_START) {
            cur = *v;
        } else if (mode == MFFT_OP_STEP) {
            long long nx = *v, tv = (sign > 0) ? cur : -cur, a = *u;
            *u = a + tv; *v = a - tv; cur = nx;
        } else {
            long long tv = (sign > 0) ? cur : -cur, a = *u;
            *u = a + tv; *v = a - tv;
        }
    }
}
__global__ void k_zero_i32(int32_t *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0;
}
__global__ void k_zero_i64(long long *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0;
}
__global__ void k_copy_i8_to_i32(int32_t *dst, const signed char *src, size_t nn)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < nn) dst[i] = (int32_t)src[i];
}

/* Expand signed-char limb planes into the MFFT coefficient buffer layout.
 * ncoeffs blocks of S limbs; NB may be larger (zero-padded evaluation
 * points for the linear convolution). */
static void mfft_pack_limbs(int32_t *Ah, const signed char *pA, size_t nn,
                            int L, int S, int K, int NB, int ncoeffs)
{
    int blk = 256;
    size_t tot = (size_t)NB * K * nn;
    k_zero_i32<<<grid_for(tot, blk), blk>>>(Ah, tot);
    for (int b = 0; b < ncoeffs; b++) {
        for (int c = 0; c < S; c++) {
            int limb = b * S + c;
            if (limb >= L) continue;          /* zero beyond real limbs */
            size_t off = ((size_t)b * K + c) * nn;
            k_copy_i8_to_i32<<<grid_for(nn, blk), blk>>>(
                Ah + off, pA + (size_t)limb * nn, nn);
        }
    }
}

/* Pack only limbs [limb_lo, limb_lo + L_keep) as a degree-(L_keep-1)
 * polynomial (faithful / high-limb path). */
static void mfft_pack_limbs_hi(int32_t *Ah, const signed char *pA, size_t nn,
                               int limb_lo, int L_keep, int S, int K, int NB,
                               int ncoeffs)
{
    int blk = 256;
    size_t tot = (size_t)NB * K * nn;
    k_zero_i32<<<grid_for(tot, blk), blk>>>(Ah, tot);
    for (int b = 0; b < ncoeffs; b++) {
        for (int c = 0; c < S; c++) {
            int rel = b * S + c;
            if (rel >= L_keep) continue;
            int limb = limb_lo + rel;
            size_t off = ((size_t)b * K + c) * nn;
            k_copy_i8_to_i32<<<grid_for(nn, blk), blk>>>(
                Ah + off, pA + (size_t)limb * nn, nn);
        }
    }
}

/* Fold inverse-FFT output into acc starting at limb base_off (for high-limb
 * MFFT where the product polynomial sits at u0+v0 on the full grid). */
__global__ void k_mfft_fold_off(int *acc, const long long *Ch, size_t nn,
                                int NB, int K, int S, int planes, int base_off)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    for (int b = 0; b < NB; b++) {
        for (int c = 0; c < K; c++) {
            int w = base_off + b * S + c;
            if (w < 0 || w >= planes) continue;
            long long v = Ch[((size_t)b * K + c) * nn + i];
            acc[(size_t)w * nn + i] += (int)(v / NB);
        }
    }
}


#endif /* MFFT_GPU_CUH_ */
