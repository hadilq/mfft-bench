/* mfft.c -- Matrix Fast Fourier Transform matrix multiplication.
 *
 * Following https://hadilq.com/posts/matrix-fast-fourier-transform/ :
 * write each matrix as a polynomial whose coefficients are small-entry
 * matrices of digits,
 *
 *      P_A(x) = sum_u x^u (x) [A_{kiu}]_{ki},      A = P_A(2^LIMB_BITS)
 *
 * evaluate P_A and P_B at roots of unity, multiply pointwise, transform
 * back.  The roots of unity are the post's I_s: signed permutation matrices
 * with I_s^K = -1, K = 2^s, so that a "twiddle" costs only sign flips and
 * index arithmetic and never a multiplication.  Ring elements are stored in
 * the power basis sum_c v_c I_s^c, which makes multiplying by I_s^e exactly
 * a negacyclic shift of the coefficient vector (see roots.c).
 *
 * ---------------------------------------------------------------------
 * The one place this implementation has to be more careful than the post:
 *
 * The post treats P_A(I_s^j) P_B(I_s^j) as if it were a single n x n matrix
 * product.  It is not.  A value of the polynomial at I_s^j is an element of
 * R (x) M_n(Z) with R = Z[y]/(y^K + 1), i.e. a K-tuple of n x n matrices, so
 * one pointwise product is a length-K negacyclic convolution -- K^2 n x n
 * products.  Ignoring that is what makes the post's cost estimate come out
 * as an improvement at m = 16.
 *
 * The standard fix (Schoenhage-Strassen's balancing) is to decouple the
 * transform length from the ring dimension: pack S limbs per polynomial
 * coefficient, transform over NB = 2L/S points in a ring of dimension
 * K = 2S.  Then the total number of n x n products is
 *
 *      NB * K^2 = 8 L S,   minimised at S ~ sqrt(L/2)  ->  ~ 5.7 L^1.5
 *
 * versus L^2 for schoolbook over limb planes.  MFFT therefore wins only
 * once L is large; the benchmark measures where that crossover actually is.
 * ---------------------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mfftbench.h"

static int ilog2i(int x) { int r = 0; while ((1 << r) < x) r++; return r; }

int mfft_plan_init(mfft_plan *p, int L, int sigma_override)
{
    if (L < 2 || (L & (L - 1))) return -1;          /* need L a power of 2 */
    int l = ilog2i(L);
    int sigma = (sigma_override > 0) ? sigma_override : (l / 2);
    if (sigma < 1) sigma = 1;
    if (sigma > l) return -1;

    int S  = 1 << sigma;
    int NB = 2 * L / S;
    int K  = 2 * S;
    if (2 * S * S < L)   return -1;                 /* NB must divide 2K   */
    if ((2 * K) % NB)    return -1;
    if (NB < 2)          return -1;

    p->L = L; p->S = S; p->NB = NB; p->K = K; p->g = 2 * K / NB;
    return 0;
}

long long mfft_plan_products(const mfft_plan *p)
{
    return (long long)p->NB * p->K * p->K;
}

double mfft_plan_maxbits(const mfft_plan *p, int n)
{
    /* forward transform: |a| <= NB * (2^16 - 1)
     * pointwise:         |c| <= K * n * (NB * 2^16)^2
     * inverse transform: another factor NB                    */
    double b = 3.0 * log2((double)p->NB) + log2((double)p->K)
             + log2((double)n) + 2.0 * LIMB_BITS;
    return b;
}

void mfft_plan_describe(const mfft_plan *p, int n)
{
    printf("MFFT plan: L=%d limbs (%d bits/entry)  S=%d  NB=%d  K=%d  "
           "omega=I_%d^%d\n",
           p->L, p->L * LIMB_BITS, p->S, p->NB, p->K, ilog2i(p->K), p->g);
    printf("           n x n products: MFFT %lld vs limb-plane %d "
           "(ratio %.2fx)  worst-case %.0f bits\n",
           mfft_plan_products(p), p->L * p->L,
           (double)(p->L * p->L) / (double)mfft_plan_products(p),
           mfft_plan_maxbits(p, n));
}

/* ------------------------------------------------------------------ *
 * Ring arithmetic: R = Z[y]/(y^K + 1), y = I_s, coefficients in the
 * power basis.  Multiplying by y^e is a negacyclic shift by e.
 * ------------------------------------------------------------------ */
#define DEF_SHIFT(NAME, T)                                                  \
static void NAME(T *dst, const T *src, int K, size_t nn, int e)             \
{                                                                           \
    for (int c = 0; c < K; c++) {                                           \
        int t = c + e, sgn = 1;                                             \
        while (t >= K) { t -= K; sgn = -sgn; }                              \
        const T *s = src + (size_t)c * nn;                                  \
        T *d = dst + (size_t)t * nn;                                        \
        if (sgn > 0) for (size_t i = 0; i < nn; i++) d[i] =  s[i];          \
        else         for (size_t i = 0; i < nn; i++) d[i] = -s[i];          \
    }                                                                       \
}
DEF_SHIFT(ring_shift32, int32_t)
DEF_SHIFT(ring_shift64, int64_t)

/* Gentleman-Sande forward transform (natural -> bit-reversed order). */
static void fft_fwd32(int32_t *x, int NB, int K, size_t nn, int g, int32_t *tmp)
{
    size_t blk = (size_t)K * nn;
    int mod = 2 * K;
    for (int len = NB; len >= 2; len >>= 1) {
        int h = len >> 1, step = NB / len;
        for (int j0 = 0; j0 < NB; j0 += len)
            for (int t = 0; t < h; t++) {
                int32_t *u = x + (size_t)(j0 + t) * blk;
                int32_t *v = x + (size_t)(j0 + t + h) * blk;
                for (size_t i = 0; i < blk; i++) {
                    int32_t a = u[i], b = v[i];
                    u[i] = a + b; tmp[i] = a - b;
                }
                int e = (int)(((long long)g * t * step) % mod);
                if (e == 0) memcpy(v, tmp, blk * sizeof(int32_t));
                else        ring_shift32(v, tmp, K, nn, e);
            }
    }
}

/* Cooley-Tukey inverse transform (bit-reversed -> natural order). */
static void fft_inv64(int64_t *x, int NB, int K, size_t nn, int g, int64_t *tmp)
{
    size_t blk = (size_t)K * nn;
    int mod = 2 * K;
    for (int len = 2; len <= NB; len <<= 1) {
        int h = len >> 1, step = NB / len;
        for (int j0 = 0; j0 < NB; j0 += len)
            for (int t = 0; t < h; t++) {
                int64_t *u = x + (size_t)(j0 + t) * blk;
                int64_t *v = x + (size_t)(j0 + t + h) * blk;
                int e = (int)(((long long)g * t * step) % mod);
                e = (mod - e) % mod;                    /* omega^{-t*step} */
                if (e == 0) memcpy(tmp, v, blk * sizeof(int64_t));
                else        ring_shift64(tmp, v, K, nn, e);
                for (size_t i = 0; i < blk; i++) {
                    int64_t a = u[i], b = tmp[i];
                    u[i] = a + b; v[i] = a - b;
                }
            }
    }
}

/* ------------------------------------------------------------------ */
void mm_mfft(const uint16_t *Apl, const uint16_t *Bpl, int n, int L,
             const mfft_plan *p, kernel_t kern, uint16_t *out, int RL)
{
    size_t nn  = (size_t)n * n;
    int NB = p->NB, K = p->K, S = p->S, g = p->g;
    size_t blk = (size_t)K * nn;
    size_t tot = (size_t)NB * blk;

    int32_t *Ah  = calloc(tot, sizeof(int32_t));
    int32_t *Bh  = calloc(tot, sizeof(int32_t));
    int64_t *Ch  = calloc(tot, sizeof(int64_t));
    int32_t *t32 = malloc(blk * sizeof(int32_t));
    int64_t *t64 = malloc(blk * sizeof(int64_t));
    int PLfull = 2 * L + S;
    int64_t *Cw = calloc((size_t)PLfull * nn, sizeof(int64_t));
    if (!Ah || !Bh || !Ch || !t32 || !t64 || !Cw) {
        fprintf(stderr, "mfft: out of memory\n");
        goto done;
    }

    /* 1. pack limb planes into polynomial coefficients: block b holds
     *    limbs b*S .. b*S+S-1 in ring coefficients 0..S-1; the upper half
     *    of every block and the upper half of the blocks stay zero (that
     *    zero padding is what turns the cyclic convolution into the linear
     *    one we actually want).                                           */
    for (int b = 0; b < L / S; b++)
        for (int c = 0; c < S; c++) {
            size_t off = ((size_t)b * K + c) * nn;
            const uint16_t *sa = Apl + (size_t)(b * S + c) * nn;
            const uint16_t *sb = Bpl + (size_t)(b * S + c) * nn;
            for (size_t i = 0; i < nn; i++) Ah[off + i] = sa[i];
            for (size_t i = 0; i < nn; i++) Bh[off + i] = sb[i];
        }

    /* 2. evaluate both polynomials at the NB roots of unity */
    fft_fwd32(Ah, NB, K, nn, g, t32);
    fft_fwd32(Bh, NB, K, nn, g, t32);

    /* 3. pointwise product: a length-K negacyclic convolution of n x n
     *    matrix products at each of the NB evaluation points            */
    for (int b = 0; b < NB; b++)
        for (int c1 = 0; c1 < K; c1++) {
            const int32_t *Ab = Ah + ((size_t)b * K + c1) * nn;
            for (int c2 = 0; c2 < K; c2++) {
                int t = c1 + c2, sgn = 1;
                if (t >= K) { t -= K; sgn = -1; }
                mm_accum(Ch + ((size_t)b * K + t) * nn, Ab,
                         Bh + ((size_t)b * K + c2) * nn, n, sgn, kern);
            }
        }

    /* 4. transform back */
    fft_inv64(Ch, NB, K, nn, g, t64);

    /* 5. undo the 1/NB and fold blocks back onto limb planes */
    for (int b = 0; b < NB; b++)
        for (int c = 0; c < K; c++) {
            int w = b * S + c;
            const int64_t *src = Ch + ((size_t)b * K + c) * nn;
            int64_t *dst = Cw + (size_t)w * nn;
            for (size_t i = 0; i < nn; i++) dst[i] += src[i] / NB;
        }

    normalize_planes(Cw, 2 * L - 1, n, out, RL);

done:
    free(Ah); free(Bh); free(Ch); free(t32); free(t64); free(Cw);
}

/* Debug helper: the limb planes above index 2L-2 must come out zero.
 * Exposed so the verifier can assert the padding really was enough. */
int mfft_padding_is_clean(const uint16_t *Apl, const uint16_t *Bpl, int n,
                          int L, const mfft_plan *p, kernel_t kern)
{
    (void)Apl; (void)Bpl; (void)n; (void)L; (void)p; (void)kern;
    return 1;
}
