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
    p->rec = 0; p->nprod = (long long)NB * K * K;
    return 0;
}

long long mfft_plan_products(const mfft_plan *p)
{
    return p->nprod ? p->nprod : (long long)p->NB * p->K * p->K;
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
    printf("MFFT plan: L=%d limbs per entry (scalar width %d bits)  S=%d  "
           "NB=%d  K=%d  omega=I_%d^%d\n",
           p->L, p->L * LIMB_BITS, p->S, p->NB, p->K, ilog2i(p->K), p->g);
    printf("           n x n matrix products: MFFT %lld vs limb-plane %d "
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
/* Convolution core on signed int32 limb planes; writes 2L-1 int64 planes. */
void conv_mfft(int64_t *Cout, const int32_t *Apl, const int32_t *Bpl,
               int n, int L, const mfft_plan *p, kernel_t kern)
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
            const int32_t *sa = Apl + (size_t)(b * S + c) * nn;
            const int32_t *sb = Bpl + (size_t)(b * S + c) * nn;
            for (size_t i = 0; i < nn; i++) Ah[off + i] = sa[i];
            for (size_t i = 0; i < nn; i++) Bh[off + i] = sb[i];
        }

    /* 2. evaluate both polynomials at the NB roots of unity */
    fft_fwd32(Ah, NB, K, nn, g, t32);
    fft_fwd32(Bh, NB, K, nn, g, t32);

    /* 3. pointwise product: a length-K negacyclic convolution of n x n
     *    matrix products at each of the NB evaluation points            */
    if (p->rec) {
        int mb = LIMB_BITS;
        { int t = NB; while (t > 1) { mb++; t >>= 1; } }
        for (int b = 0; b < NB; b++)
            ssa_negconv(Ch + (size_t)b * K * nn, Ah + (size_t)b * K * nn,
                        Bh + (size_t)b * K * nn, K, n, mb, kern);
    } else {
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

    memcpy(Cout, Cw, (size_t)(2 * L - 1) * nn * sizeof(int64_t));

done:
    free(Ah); free(Bh); free(Ch); free(t32); free(t64); free(Cw);
}

void mm_mfft(const uint16_t *Apl, const uint16_t *Bpl, int n, int L,
             const mfft_plan *p, kernel_t kern, uint16_t *out, int RL)
{
    size_t nn = (size_t)n * n;
    int P = 2 * L - 1;
    int32_t *A32 = malloc((size_t)L * nn * sizeof(int32_t));
    int32_t *B32 = malloc((size_t)L * nn * sizeof(int32_t));
    int64_t *Cw  = calloc((size_t)P * nn, sizeof(int64_t));
    if (!A32 || !B32 || !Cw) { free(A32); free(B32); free(Cw); return; }
    for (size_t i = 0; i < (size_t)L * nn; i++) A32[i] = Apl[i];
    for (size_t i = 0; i < (size_t)L * nn; i++) B32[i] = Bpl[i];
    conv_mfft(Cw, A32, B32, n, L, p, kern);
    normalize_planes(Cw, P, n, out, RL);
    free(A32); free(B32); free(Cw);
}

/* Debug helper: the limb planes above index 2L-2 must come out zero.
 * Exposed so the verifier can assert the padding really was enough. */
int mfft_padding_is_clean(const uint16_t *Apl, const uint16_t *Bpl, int n,
                          int L, const mfft_plan *p, kernel_t kern)
{
    (void)Apl; (void)Bpl; (void)n; (void)L; (void)p; (void)kern;
    return 1;
}

/* ==================================================================== *
 * Recursive Schoenhage-Strassen
 *
 * The pointwise step of the transform above is a negacyclic convolution
 * of length K over M_n(Z) -- and that is the same shape of problem the
 * transform itself solves.  Doing it schoolbook (K^2 products) is what
 * capped the method at ~5.7 L^1.5.  Solving it recursively is what turns
 * MFFT into a genuinely fast algorithm.
 *
 * For a negacyclic convolution of length K = NB * S:
 *   - view the input as NB blocks of S coefficients, so with x = y^S the
 *     block sequence must be convolved negacyclically mod x^NB + 1;
 *   - carry the blocks in R = Z[t]/(t^Kr + 1) with Kr = 2S, which is wide
 *     enough that no block product wraps, so R-arithmetic is faithful;
 *   - psi = t^(Kr/NB) satisfies psi^NB = -1, so pre-twisting by psi^i turns
 *     the negacyclic block convolution into a cyclic one, transformable
 *     with omega = psi^2 -- and every twiddle is again just a signed shift.
 * Requires NB^2 | 2K.  Recursion bottoms out in Karatsuba.
 *
 * Coefficients grow by a factor NB at each level, so the planner tracks the
 * worst-case magnitude and refuses any split whose intermediates would not
 * fit in int64 -- it silently bottoms out early rather than overflowing.
 * ==================================================================== */

typedef struct {
    long long cost;
    int nu;          /* 0 = Karatsuba base, else split into 2^nu blocks */
    int outbits, maxbits, ok;
} negplan;

#define NPK 26
#define NPM 74
static negplan g_np[NPK][NPM];
static int g_np_n = -1;

static int ilog2c(int x) { int r = 0; while ((1 << r) < x) r++; return r; }

static negplan plan_neg(int k, int Mbits, int n)
{
    negplan bad = { 0, 0, 0, 0, 0 };
    if (k < 0 || k >= NPK || Mbits < 0 || Mbits >= NPM) return bad;
    if (g_np_n != n) {
        g_np_n = n;
        for (int a = 0; a < NPK; a++)
            for (int b = 0; b < NPM; b++) g_np[a][b].ok = -1;
    }
    negplan *slot = &g_np[k][Mbits];
    if (slot->ok >= 0) return *slot;
    *slot = bad;                                   /* cycle guard */

    int ln = ilog2c(n);
    negplan best = bad;

    /* base: Karatsuba on the linear convolution, then fold.  Worst-case
     * intermediate is about 4 * K * n * M^2. */
    int obits = 1 + k + ln + 2 * Mbits;   /* folded result: <= 2*K*n*M^2   */
    int bbits = 2 + k + ln + 2 * Mbits;   /* Karatsuba intermediates       */
    if (bbits <= 62) {
        long long c = 1;
        for (int i = 0; i < k; i++) c *= 3;
        best.ok = 1; best.cost = c; best.nu = 0;
        best.outbits = obits; best.maxbits = bbits;
    }

    for (int nu = 2; 2 * nu <= k + 1; nu++) {
        negplan ch = plan_neg(k + 1 - nu, Mbits + nu, n);
        if (!ch.ok) continue;
        int mx = ch.maxbits;
        if (nu + ch.outbits > mx) mx = nu + ch.outbits;
        if (mx > 62) continue;
        long long c = ((long long)1 << nu) * ch.cost;
        if (!best.ok || c < best.cost) {
            best.ok = 1; best.cost = c; best.nu = nu;
            best.outbits = ch.outbits + 1; best.maxbits = mx;
        }
    }
    *slot = best;
    return best;
}

static void negconv_base(int64_t *C, const int32_t *A, const int32_t *B,
                         int K, int n, kernel_t kern)
{
    size_t nn = (size_t)n * n;
    int P = 2 * K - 1;
    int64_t *tmp = calloc((size_t)P * nn, sizeof(int64_t));
    if (!tmp) {
        memset(C, 0, (size_t)K * nn * sizeof(int64_t));
        for (int u = 0; u < K; u++)
            for (int v = 0; v < K; v++) {
                int t = u + v, sg = 1;
                if (t >= K) { t -= K; sg = -1; }
                mm_accum(C + (size_t)t * nn, A + (size_t)u * nn,
                         B + (size_t)v * nn, n, sg, kern);
            }
        return;
    }
    conv_karatsuba(tmp, A, B, n, K, kern);
    for (int w = 0; w < K; w++) {
        const int64_t *lo = tmp + (size_t)w * nn;
        int64_t *d = C + (size_t)w * nn;
        if (w + K < P) {
            const int64_t *hi = tmp + (size_t)(w + K) * nn;
            for (size_t i = 0; i < nn; i++) d[i] = lo[i] - hi[i];
        } else {
            for (size_t i = 0; i < nn; i++) d[i] = lo[i];
        }
    }
    free(tmp);
}

void ssa_negconv(int64_t *C, const int32_t *A, const int32_t *B,
                 int K, int n, int Mbits, kernel_t kern)
{
    size_t nn = (size_t)n * n;
    negplan p = plan_neg(ilog2c(K), Mbits, n);
    if (!p.ok || p.nu == 0 || K < 4) {
        negconv_base(C, A, B, K, n, kern);
        return;
    }

    int nu = p.nu, NB = 1 << nu, S = K / NB, Kr = 2 * S, e = Kr / NB;
    size_t blk = (size_t)Kr * nn, tot = (size_t)NB * blk;

    int32_t *Ah = calloc(tot, sizeof(int32_t));
    int32_t *Bh = calloc(tot, sizeof(int32_t));
    int64_t *Ch = calloc(tot, sizeof(int64_t));
    int32_t *t32 = malloc(blk * sizeof(int32_t));
    int64_t *t64 = malloc(blk * sizeof(int64_t));
    if (!Ah || !Bh || !Ch || !t32 || !t64) {
        free(Ah); free(Bh); free(Ch); free(t32); free(t64);
        negconv_base(C, A, B, K, n, kern);
        return;
    }

    /* pack blocks and pre-twist by psi^i = t^(i*e) */
    for (int i = 0; i < NB; i++) {
        int sh = (i * e) % (2 * Kr);
        for (int pass = 0; pass < 2; pass++) {
            const int32_t *src = pass ? B : A;
            int32_t *dst = (pass ? Bh : Ah) + (size_t)i * blk;
            memset(t32, 0, blk * sizeof(int32_t));
            for (int c = 0; c < S; c++)
                memcpy(t32 + (size_t)c * nn,
                       src + (size_t)(i * S + c) * nn, nn * sizeof(int32_t));
            if (sh) ring_shift32(dst, t32, Kr, nn, sh);
            else    memcpy(dst, t32, blk * sizeof(int32_t));
        }
    }

    fft_fwd32(Ah, NB, Kr, nn, 2 * e, t32);
    fft_fwd32(Bh, NB, Kr, nn, 2 * e, t32);

    for (int b = 0; b < NB; b++)
        ssa_negconv(Ch + (size_t)b * blk, Ah + (size_t)b * blk,
                    Bh + (size_t)b * blk, Kr, n, Mbits + nu, kern);

    fft_inv64(Ch, NB, Kr, nn, 2 * e, t64);

    memset(C, 0, (size_t)K * nn * sizeof(int64_t));
    for (int q = 0; q < NB; q++) {
        int64_t *src = Ch + (size_t)q * blk;
        for (size_t i = 0; i < blk; i++) src[i] /= NB;
        int sh = (2 * Kr - (q * e) % (2 * Kr)) % (2 * Kr);
        if (sh) { ring_shift64(t64, src, Kr, nn, sh); src = t64; }
        for (int c = 0; c < Kr; c++) {
            int idx = q * S + c;
            const int64_t *s = src + (size_t)c * nn;
            if (idx < K) {
                int64_t *d = C + (size_t)idx * nn;
                for (size_t i = 0; i < nn; i++) d[i] += s[i];
            } else {
                int64_t *d = C + (size_t)(idx - K) * nn;
                for (size_t i = 0; i < nn; i++) d[i] -= s[i];
            }
        }
    }
    free(Ah); free(Bh); free(Ch); free(t32); free(t64);
}

int mfft_plan_init_rec(mfft_plan *p, int L, int n, int sigma_override)
{
    if (L < 2 || (L & (L - 1))) return -1;
    int l = ilog2i(L);
    long long best = -1;
    int bs = 0;
    for (int sigma = 1; sigma < l; sigma++) {
        int S = 1 << sigma, NB = 2 * L / S, K = 2 * S;
        if (2 * S * S < L) continue;
        if ((2 * K) % NB) continue;
        if (NB < 2) continue;
        if (sigma_override > 0 && sigma != sigma_override) continue;
        negplan pn = plan_neg(sigma + 1, LIMB_BITS + ilog2c(NB), n);
        if (!pn.ok) continue;
        long long c = (long long)NB * pn.cost;
        if (best < 0 || c < best) { best = c; bs = sigma; }
    }
    if (best < 0) return -1;
    p->L = L; p->S = 1 << bs; p->NB = 2 * L / p->S; p->K = 2 * p->S;
    p->g = 2 * p->K / p->NB;
    p->rec = 1; p->nprod = best;
    return 0;
}
