/* lowprec.c -- exact matrix multiplication at ML precisions.
 *
 * The fp32 embedding in fpfixed.c needs many limbs because an fp32
 * significand is 24 bits wide.  Every precision below fp32 collapses the
 * problem, and the collapse is worth stating precisely.
 *
 * An entry needs  vbits = significand_bits + exponent_spread  bits on the
 * shared fixed-point grid.  A single GEMM computes the product exactly when
 *
 *      2 * vbits + ceil(log2 n) + 1  <=  accumulator width.
 *
 * With an int64 accumulator that allows vbits <= 26 at n = 1024:
 *
 *      fp32   24 + spread(~17) = 41 bits  -> 4 limbs, 9..16 products
 *      bf16    8 + spread(~17) = 25 bits  -> 1 limb,  ONE product
 *      int8                      8 bits   -> 1 limb,  ONE product
 *      int4                      4 bits   -> 1 limb,  ONE product
 *
 * So at bf16 and below, exact matrix multiplication costs a single integer
 * GEMM.  There is no convolution to accelerate: Karatsuba and MFFT have
 * nothing to do, because there is only one limb.  That is the whole result
 * of this file, and it is why the ML answer to MFFT is "not applicable"
 * rather than "too slow".
 *
 * Per the "optimise, accept duplication" rule, each precision gets its own
 * packed kernel rather than sharing a generic one: int8/int4 run int16
 * operands into int32 accumulators (twice the SIMD lanes of the int32/int64
 * path, and int32 adds instead of int64), bf16 runs int32 into int64.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mfftbench.h"

/* ------------------------------------------------------------------ *
 * SIMD shapes.  Deliberately separate from the ones in mlgemm.c: the
 * int16->int32 kernel wants twice the lanes of the int32->int64 kernel,
 * and sharing a shape would cost one of them half its throughput.
 * ------------------------------------------------------------------ */
#if defined(__AVX512F__)
#  define LPV 64
#elif defined(__AVX2__) || defined(__ARM_NEON) || defined(__aarch64__)
#  define LPV 32
#else
#  define LPV 16
#endif

typedef int32_t lp_i32 __attribute__((vector_size(LPV)));
typedef int32_t lp_i32u __attribute__((vector_size(LPV), aligned(4)));
typedef int16_t lp_i16u __attribute__((vector_size(LPV / 2), aligned(2)));
typedef int64_t lp_i64  __attribute__((vector_size(LPV)));
typedef int64_t lp_i64u __attribute__((vector_size(LPV), aligned(8)));
/* half-width int32 view: same lane count as lp_i64, for widening loads */
typedef int32_t lp_i32h __attribute__((vector_size(LPV / 2), aligned(4)));

#define LP_L32 ((int)(sizeof(lp_i32) / sizeof(int32_t)))
#define LP_L64 ((int)(sizeof(lp_i64) / sizeof(int64_t)))

#define LMC 192
#define LKC 256
#define LNC 512
#define LMR 6
#define LMIN(a,b) ((a) < (b) ? (a) : (b))

/* ================================================================== *
 * Kernel A: int16 operands, int32 accumulators.   int8 / int4 path.
 * ================================================================== */
#define NR32 (2 * LP_L32)

static void a_pack_a(int16_t *Ap, const int16_t *A, int n,
                     int ic, int mc, int pc, int kc)
{
    for (int pi = 0; pi * LMR < mc; pi++) {
        int16_t *d = Ap + (size_t)pi * LMR * kc;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < LMR; i++)
                d[p * LMR + i] = (pi * LMR + i < mc)
                    ? A[(size_t)(ic + pi * LMR + i) * n + pc + p] : 0;
    }
}

static void a_pack_b(int16_t *Bp, const int16_t *B, int n,
                     int jc, int nc, int pc, int kc)
{
    for (int pj = 0; pj * NR32 < nc; pj++) {
        int16_t *d = Bp + (size_t)pj * NR32 * kc;
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < NR32; j++)
                d[p * NR32 + j] = (pj * NR32 + j < nc)
                    ? B[(size_t)(pc + p) * n + jc + pj * NR32 + j] : 0;
    }
}

static void a_micro(const int16_t *restrict a, const int16_t *restrict b,
                    int32_t *c, int ldc, int kc, int mr, int nr)
{
    lp_i32 acc[LMR][2];
    for (int i = 0; i < LMR; i++) { acc[i][0] = (lp_i32){0}; acc[i][1] = (lp_i32){0}; }
    for (int p = 0; p < kc; p++) {
        const int16_t *bp = b + (size_t)p * NR32;
        lp_i32 b0 = __builtin_convertvector(*(const lp_i16u *)bp, lp_i32);
        lp_i32 b1 = __builtin_convertvector(*(const lp_i16u *)(bp + LP_L32), lp_i32);
        const int16_t *ap = a + (size_t)p * LMR;
        for (int i = 0; i < LMR; i++) {
            lp_i32 av = (lp_i32){0} + (int32_t)ap[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }
    int32_t tmp[LMR][NR32];
    for (int i = 0; i < LMR; i++) {
        *(lp_i32u *)&tmp[i][0]        = acc[i][0];
        *(lp_i32u *)&tmp[i][LP_L32]   = acc[i][1];
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++) c[(size_t)i * ldc + j] += tmp[i][j];
}

void lp_gemm_i16_i32(int32_t *C, const int16_t *A, const int16_t *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(int32_t));
    int16_t *Ap = malloc((size_t)(LMC / LMR + 1) * LMR * LKC * sizeof(int16_t));
    int16_t *Bp = malloc((size_t)(LNC / NR32 + 1) * NR32 * LKC * sizeof(int16_t));
    if (!Ap || !Bp) { free(Ap); free(Bp); return; }
    for (int jc = 0; jc < n; jc += LNC) {
        int nc = LMIN(LNC, n - jc);
        for (int pc = 0; pc < n; pc += LKC) {
            int kc = LMIN(LKC, n - pc);
            a_pack_b(Bp, B, n, jc, nc, pc, kc);
            for (int ic = 0; ic < n; ic += LMC) {
                int mc = LMIN(LMC, n - ic);
                a_pack_a(Ap, A, n, ic, mc, pc, kc);
                for (int j = 0; j < nc; j += NR32) {
                    int nr = LMIN(NR32, nc - j);
                    for (int i = 0; i < mc; i += LMR) {
                        int mr = LMIN(LMR, mc - i);
                        a_micro(Ap + (size_t)(i / LMR) * LMR * kc,
                                Bp + (size_t)(j / NR32) * NR32 * kc,
                                C + (size_t)(ic + i) * n + jc + j,
                                n, kc, mr, nr);
                    }
                }
            }
        }
    }
    free(Ap); free(Bp);
}

/* ================================================================== *
 * Kernel B: int32 operands, int64 accumulators.   bf16 path.
 * ================================================================== */
#define NR64 (2 * LP_L64)

static void b_pack_a(int32_t *Ap, const int32_t *A, int n,
                     int ic, int mc, int pc, int kc)
{
    for (int pi = 0; pi * LMR < mc; pi++) {
        int32_t *d = Ap + (size_t)pi * LMR * kc;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < LMR; i++)
                d[p * LMR + i] = (pi * LMR + i < mc)
                    ? A[(size_t)(ic + pi * LMR + i) * n + pc + p] : 0;
    }
}

static void b_pack_b(int32_t *Bp, const int32_t *B, int n,
                     int jc, int nc, int pc, int kc)
{
    for (int pj = 0; pj * NR64 < nc; pj++) {
        int32_t *d = Bp + (size_t)pj * NR64 * kc;
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < NR64; j++)
                d[p * NR64 + j] = (pj * NR64 + j < nc)
                    ? B[(size_t)(pc + p) * n + jc + pj * NR64 + j] : 0;
    }
}

static void b_micro(const int32_t *restrict a, const int32_t *restrict b,
                    int64_t *c, int ldc, int kc, int mr, int nr)
{
    lp_i64 acc[LMR][2];
    for (int i = 0; i < LMR; i++) { acc[i][0] = (lp_i64){0}; acc[i][1] = (lp_i64){0}; }
    for (int p = 0; p < kc; p++) {
        const int32_t *bp = b + (size_t)p * NR64;
        lp_i64 b0 = __builtin_convertvector(*(const lp_i32h *)bp, lp_i64);
        lp_i64 b1 = __builtin_convertvector(*(const lp_i32h *)(bp + LP_L64), lp_i64);
        const int32_t *ap = a + (size_t)p * LMR;
        for (int i = 0; i < LMR; i++) {
            lp_i64 av = (lp_i64){0} + (int64_t)ap[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }
    int64_t tmp[LMR][NR64];
    for (int i = 0; i < LMR; i++) {
        *(lp_i64u *)&tmp[i][0]      = acc[i][0];
        *(lp_i64u *)&tmp[i][LP_L64] = acc[i][1];
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++) c[(size_t)i * ldc + j] += tmp[i][j];
}

void lp_gemm_i32_i64(int64_t *C, const int32_t *A, const int32_t *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(int64_t));
    int32_t *Ap = malloc((size_t)(LMC / LMR + 1) * LMR * LKC * sizeof(int32_t));
    int32_t *Bp = malloc((size_t)(LNC / NR64 + 1) * NR64 * LKC * sizeof(int32_t));
    if (!Ap || !Bp) { free(Ap); free(Bp); return; }
    for (int jc = 0; jc < n; jc += LNC) {
        int nc = LMIN(LNC, n - jc);
        for (int pc = 0; pc < n; pc += LKC) {
            int kc = LMIN(LKC, n - pc);
            b_pack_b(Bp, B, n, jc, nc, pc, kc);
            for (int ic = 0; ic < n; ic += LMC) {
                int mc = LMIN(LMC, n - ic);
                b_pack_a(Ap, A, n, ic, mc, pc, kc);
                for (int j = 0; j < nc; j += NR64) {
                    int nr = LMIN(NR64, nc - j);
                    for (int i = 0; i < mc; i += LMR) {
                        int mr = LMIN(LMR, mc - i);
                        b_micro(Ap + (size_t)(i / LMR) * LMR * kc,
                                Bp + (size_t)(j / NR64) * NR64 * kc,
                                C + (size_t)(ic + i) * n + jc + j,
                                n, kc, mr, nr);
                    }
                }
            }
        }
    }
    free(Ap); free(Bp);
}

/* ================================================================== *
 * Rounding to the ML formats
 * ================================================================== */
float lp_to_bf16(float x)
{
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

/* ================================================================== *
 * bf16: exact via a single fixed-point GEMM
 * ================================================================== */
static int ilog2c_lp(int x) { int r = 0; while ((1 << r) < x) r++; return r; }

static void scan_e2(const float *X, size_t nn, int sig, int *lo, int *hi)
{
    int l = 1 << 30, h = -(1 << 30), seen = 0;
    for (size_t i = 0; i < nn; i++) {
        if (X[i] == 0.0f || !isfinite(X[i])) continue;
        int e; frexpf(X[i], &e);
        int e2 = e - sig;
        if (e2 < l) l = e2;
        if (e2 > h) h = e2;
        seen = 1;
    }
    if (!seen) { l = 0; h = 0; }
    *lo = l; *hi = h;
}

int lp_bf16_init(lp_ctx *c, const float *A, const float *B, int n)
{
    size_t nn = (size_t)n * n;
    int loA, hiA, loB, hiB;
    memset(c, 0, sizeof *c);
    c->n = n;
    c->Ar = malloc(nn * sizeof(float));
    c->Br = malloc(nn * sizeof(float));
    if (!c->Ar || !c->Br) return -1;
    for (size_t i = 0; i < nn; i++) c->Ar[i] = lp_to_bf16(A[i]);
    for (size_t i = 0; i < nn; i++) c->Br[i] = lp_to_bf16(B[i]);

    scan_e2(c->Ar, nn, 8, &loA, &hiA);
    scan_e2(c->Br, nn, 8, &loB, &hiB);
    c->SA = -loA; c->SB = -loB;
    int vA = 8 + (hiA - loA), vB = 8 + (hiB - loB);
    c->vbits = vA > vB ? vA : vB;

    /* One GEMM is exact iff |a|*|b| summed n times still fits.  Bounding
     * A and B separately rather than by their max matters here: it is
     * routinely the difference between one limb and two. */
    /* |a| < 2^vA and |b| < 2^vB strictly, so the length-n sum is strictly
     * below 2^(vA+vB+ceil(log2 n)); no extra slack bit is needed. */
    c->needbits = vA + vB + ilog2c_lp(n);
    if (c->needbits > 63) return -2;

    c->A32 = malloc(nn * sizeof(int32_t));
    c->B32 = malloc(nn * sizeof(int32_t));
    c->C64 = malloc(nn * sizeof(int64_t));
    if (!c->A32 || !c->B32 || !c->C64) return -1;

    for (int pass = 0; pass < 2; pass++) {
        const float *src = pass ? c->Br : c->Ar;
        int32_t *dst = pass ? c->B32 : c->A32;
        int S = pass ? c->SB : c->SA;
        for (size_t i = 0; i < nn; i++) {
            float x = src[i];
            if (x == 0.0f || !isfinite(x)) { dst[i] = 0; continue; }
            int e; float m = frexpf(x, &e);
            int32_t mi = (int32_t)(fabsf(m) * 256.0f);       /* 8-bit, exact */
            int off = e - 8 + S;
            dst[i] = (x < 0.0f ? -mi : mi) << off;
        }
    }
    return 0;
}

void lp_bf16_gemm(lp_ctx *c)
{
    lp_gemm_i32_i64(c->C64, c->A32, c->B32, c->n);
}

void lp_bf16_decode(const lp_ctx *c, float *C)
{
    size_t nn = (size_t)c->n * c->n;
    int S = c->SA + c->SB;
    for (size_t i = 0; i < nn; i++) C[i] = (float)ldexp((double)c->C64[i], -S);
}

/* ================================================================== *
 * int8 / int4: per-channel symmetric quantization.  int32 accumulation
 * is already exact, so the only error is the quantization itself.
 * ================================================================== */
int lp_intq_init(lp_ctx *c, const float *A, const float *B, int n, int bits)
{
    size_t nn = (size_t)n * n;
    memset(c, 0, sizeof *c);
    c->n = n; c->vbits = bits;
    int qmax = (1 << (bits - 1)) - 1;              /* 127 for int8, 7 for int4 */

    c->A16 = malloc(nn * sizeof(int16_t));
    c->B16 = malloc(nn * sizeof(int16_t));
    c->C32 = malloc(nn * sizeof(int32_t));
    c->sa  = malloc((size_t)n * sizeof(float));
    c->sb  = malloc((size_t)n * sizeof(float));
    if (!c->A16 || !c->B16 || !c->C32 || !c->sa || !c->sb) return -1;

    for (int i = 0; i < n; i++) {                  /* per-row scale for A */
        float mx = 0;
        for (int k = 0; k < n; k++) {
            float v = fabsf(A[(size_t)i * n + k]);
            if (v > mx) mx = v;
        }
        c->sa[i] = mx > 0 ? mx / (float)qmax : 1.0f;
        for (int k = 0; k < n; k++)
            c->A16[(size_t)i * n + k] =
                (int16_t)lrintf(A[(size_t)i * n + k] / c->sa[i]);
    }
    for (int j = 0; j < n; j++) {                  /* per-column scale for B */
        float mx = 0;
        for (int k = 0; k < n; k++) {
            float v = fabsf(B[(size_t)k * n + j]);
            if (v > mx) mx = v;
        }
        c->sb[j] = mx > 0 ? mx / (float)qmax : 1.0f;
        for (int k = 0; k < n; k++)
            c->B16[(size_t)k * n + j] =
                (int16_t)lrintf(B[(size_t)k * n + j] / c->sb[j]);
    }
    return 0;
}

void lp_intq_gemm(lp_ctx *c)
{
    lp_gemm_i16_i32(c->C32, c->A16, c->B16, c->n);
}

void lp_intq_decode(const lp_ctx *c, float *C)
{
    int n = c->n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[(size_t)i * n + j] =
                (float)c->C32[(size_t)i * n + j] * c->sa[i] * c->sb[j];
}

void lp_free(lp_ctx *c)
{
    free(c->Ar); free(c->Br);
    free(c->A16); free(c->B16); free(c->A32); free(c->B32);
    free(c->C32); free(c->C64); free(c->sa); free(c->sb);
    memset(c, 0, sizeof *c);
}
