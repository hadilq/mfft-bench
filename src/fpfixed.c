/* fpfixed.c -- running MFFT on fp32 matrices by embedding floats in a
 * fixed-point integer grid.
 *
 * The idea (due to Hadi Lashkari Ghouchani): an fp32 number is a 24-bit
 * significand times a power of two, and the exponent only has 8 bits of
 * range.  So every fp32 in a matrix can be written as an integer on one
 * shared grid -- shift the significand left by (exponent + offset) -- and
 * once that is done the exponents are gone and the entries are just wide
 * integers.  Multiply them exactly, then round the exact result back to
 * fp32.  The wide integer is never materialised: the shift writes straight
 * into the limb planes the convolution already wants.
 *
 * Two things make this better than the "pick a big constant" version:
 *
 *  1. A and B get *separate* scales (SA, SB), and the product comes out at
 *     scale SA+SB.  Each matrix then only has to span its own exponent
 *     range instead of both.
 *
 *  2. The grid is sized from the data, not fixed at 512 bits.  The width
 *     needed is 24 + (max exponent - min exponent) + 1, because what costs
 *     limbs is the *spread* of exponents, not their absolute size.  A
 *     tensor whose values live within a factor of 2^20 needs ~64 bits, not
 *     512.  Since the cost of every method here is quadratic in the limb
 *     count, that is the difference between 16 and 1024 matrix products.
 *     --fp-width forces the fixed 512-bit grid for comparison.
 *
 * The result is bit-exact and independent of summation order: this is a
 * Kulisch-style exact accumulation, so it is *more* accurate than an fp32
 * or even fp64 dot product, not less.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mfftbench.h"

static int next_pow2(int x) { int r = 1; while (r < x) r <<= 1; return r; }

/* exponent statistics: every finite x is  mi * 2^e2  with mi in [2^23,2^24) */
static void scan(const float *X, size_t nn, int *mine2, int *maxe2, int *any)
{
    int lo = 1 << 30, hi = -(1 << 30), seen = 0;
    for (size_t i = 0; i < nn; i++) {
        float x = X[i];
        if (x == 0.0f || !isfinite(x)) continue;
        int e; frexpf(x, &e);
        int e2 = e - 24;
        if (e2 < lo) lo = e2;
        if (e2 > hi) hi = e2;
        seen = 1;
    }
    if (!seen) { lo = 0; hi = 0; }
    *mine2 = lo; *maxe2 = hi; *any = seen;
}

/* Shared sizing logic for both precisions: `sig` is the significand width
 * (24 for fp32, 53 for fp64) and `wr` is how many limbs a single entry's
 * shifted significand can touch. */
static int fpx_size(fpx_ctx *c, int loA, int hiA, int loB, int hiB,
                    int anyA, int anyB, int sig, int wr, int force_bits)
{
    c->spreadA = anyA ? hiA - loA : 0;
    c->spreadB = anyB ? hiB - loB : 0;
    if (force_bits > 0) {
        c->SA = c->SB = force_bits / 2;
        return next_pow2(force_bits / LIMB_BITS);
    }
    c->SA = -loA;
    c->SB = -loB;
    int bitsA = sig + c->spreadA + 1;
    int bitsB = sig + c->spreadB + 1;
    int bits = bitsA > bitsB ? bitsA : bitsB;
    int spread = c->spreadA > c->spreadB ? c->spreadA : c->spreadB;
    int need = (bits + LIMB_BITS - 1) / LIMB_BITS;
    int need2 = (spread / LIMB_BITS) + wr;
    int L = next_pow2(need > need2 ? need : need2);
    return L < 2 ? 2 : L;
}

static int fpx_alloc(fpx_ctx *c, int n, int L)
{
    size_t nn = (size_t)n * n;
    c->n = n; c->L = L;
    if ((double)n * L * 65535.0 * 65535.0 > 4.0e18) {
        fprintf(stderr, "fpx: n=%d L=%d would overflow int64 accumulators\n",
                n, L);
        return -1;
    }
    c->A32 = calloc((size_t)L * nn, sizeof(int32_t));
    c->B32 = calloc((size_t)L * nn, sizeof(int32_t));
    c->Cw  = calloc((size_t)(2 * L - 1) * nn, sizeof(int64_t));
    if (!c->A32 || !c->B32 || !c->Cw) { fpx_free(c); return -1; }
    return 0;
}

int fpx_init(fpx_ctx *c, const float *A, const float *B, int n, int force_bits)
{
    size_t nn = (size_t)n * n;
    int loA, hiA, loB, hiB, anyA, anyB;
    scan(A, nn, &loA, &hiA, &anyA);
    scan(B, nn, &loB, &hiB, &anyB);

    memset(c, 0, sizeof *c);
    c->sig = 24;
    /* a 24-bit significand shifted by up to 15 spans three limbs */
    int L = fpx_size(c, loA, hiA, loB, hiB, anyA, anyB, 24, 3, force_bits);
    return fpx_alloc(c, n, L);
}

/* ------------------------------------------------------------------ *
 * fp64 on the same grid.  A 53-bit significand shifted by up to 15 reaches
 * 68 bits, past what a uint64 can stage, so the shift goes through
 * __int128 and can touch five limbs.
 * ------------------------------------------------------------------ */
static void scan_d(const double *X, size_t nn, int *mine2, int *maxe2, int *any)
{
    int lo = 1 << 30, hi = -(1 << 30), seen = 0;
    for (size_t i = 0; i < nn; i++) {
        double x = X[i];
        if (x == 0.0 || !isfinite(x)) continue;
        int e; frexp(x, &e);
        int e2 = e - 53;
        if (e2 < lo) lo = e2;
        if (e2 > hi) hi = e2;
        seen = 1;
    }
    if (!seen) { lo = 0; hi = 0; }
    *mine2 = lo; *maxe2 = hi; *any = seen;
}

int fpx_init_d(fpx_ctx *c, const double *A, const double *B, int n,
               int force_bits)
{
    size_t nn = (size_t)n * n;
    int loA, hiA, loB, hiB, anyA, anyB;
    scan_d(A, nn, &loA, &hiA, &anyA);
    scan_d(B, nn, &loB, &hiB, &anyB);
    memset(c, 0, sizeof *c);
    c->sig = 53;
    int L = fpx_size(c, loA, hiA, loB, hiB, anyA, anyB, 53, 5, force_bits);
    return fpx_alloc(c, n, L);
}

static void encode_one_d(int32_t *plane, size_t nn, size_t idx,
                         const double *X, int L, int S)
{
    double x = X[idx];
    if (x == 0.0 || !isfinite(x)) return;
    int e; double m = frexp(x, &e);
    int64_t mi = (int64_t)(fabs(m) * 9007199254740992.0);   /* 2^53, exact */
    int off = e - 53 + S;
    if (off < 0) return;
    int lo = off >> 4, sh = off & 15;
    unsigned __int128 v = (unsigned __int128)mi << sh;       /* <= 68 bits */
    int sgn = (x < 0.0) ? -1 : 1;
    for (int t = 0; t < 5; t++) {
        int w = lo + t;
        if (w >= L) break;
        uint64_t chunk = (uint64_t)(v >> (16 * t));
        plane[(size_t)w * nn + idx] = sgn * (int32_t)(chunk & 0xFFFFu);
    }
}

void fpx_encode_d(fpx_ctx *c, const double *A, const double *B)
{
    size_t nn = (size_t)c->n * c->n;
    double t0 = now_sec();
    memset(c->A32, 0, (size_t)c->L * nn * sizeof(int32_t));
    memset(c->B32, 0, (size_t)c->L * nn * sizeof(int32_t));
    for (size_t i = 0; i < nn; i++) encode_one_d(c->A32, nn, i, A, c->L, c->SA);
    for (size_t i = 0; i < nn; i++) encode_one_d(c->B32, nn, i, B, c->L, c->SB);
    c->enc_secs = now_sec() - t0;
}

static void encode_one(int32_t *plane, size_t nn, size_t idx,
                       const float *X, int L, int S)
{
    float x = X[idx];
    if (x == 0.0f || !isfinite(x)) return;
    int e; float m = frexpf(x, &e);
    int32_t mi = (int32_t)(fabsf(m) * 16777216.0f);   /* exact: 24-bit int */
    int e2 = e - 24;
    int off = e2 + S;
    if (off < 0) return;                              /* below the grid */
    int lo = off >> 4, sh = off & 15;
    uint64_t v = (uint64_t)mi << sh;                  /* <= 39 bits */
    int sgn = (x < 0.0f) ? -1 : 1;
    for (int t = 0; t < 3; t++) {
        int w = lo + t;
        if (w >= L) break;
        plane[(size_t)w * nn + idx] = sgn * (int32_t)((v >> (16 * t)) & 0xFFFFu);
    }
}

void fpx_encode(fpx_ctx *c, const float *A, const float *B)
{
    size_t nn = (size_t)c->n * c->n;
    double t0 = now_sec();
    memset(c->A32, 0, (size_t)c->L * nn * sizeof(int32_t));
    memset(c->B32, 0, (size_t)c->L * nn * sizeof(int32_t));
    for (size_t i = 0; i < nn; i++) encode_one(c->A32, nn, i, A, c->L, c->SA);
    for (size_t i = 0; i < nn; i++) encode_one(c->B32, nn, i, B, c->L, c->SB);
    c->enc_secs = now_sec() - t0;
}

/* ------------------------------------------------------------------ *
 * Decode: carry-normalise one output entry into a signed big integer,
 * then round it to 24 (or 53) significant bits.  Correct rounding here is
 * the whole point -- the integer product is exact, so the only error in
 * the final fp32 is a single half-ulp.
 * ------------------------------------------------------------------ */
#define RLIMBS(L) (2 * (L) + 2)

static int extract(const fpx_ctx *c, size_t idx, int32_t *limb,
                   uint64_t *top, int *tb, int *shift, int *sticky)
{
    size_t nn = (size_t)c->n * c->n;
    int RL = RLIMBS(c->L), P = 2 * c->L - 1;
    int64_t carry = 0;
    for (int w = 0; w < RL; w++) {
        int64_t t = (w < P ? c->Cw[(size_t)w * nn + idx] : 0) + carry;
        limb[w] = (int32_t)(t & 0xFFFF);
        carry = t >> LIMB_BITS;                       /* arithmetic shift */
    }
    int neg = (carry < 0);
    if (neg) {                                        /* two's complement negate */
        int32_t borrow = 1;
        for (int w = 0; w < RL; w++) {
            int32_t t = (~limb[w] & 0xFFFF) + borrow;
            limb[w] = t & 0xFFFF;
            borrow = t >> LIMB_BITS;
        }
    }
    int h = -1;
    for (int w = RL - 1; w >= 0; w--) if (limb[w]) { h = w; break; }
    if (h < 0) return 0;                              /* exact zero */

    int bl = 0; { int32_t v = limb[h]; while (v) { bl++; v >>= 1; } }
    int bitlen = 16 * h + bl;

    int k = (h + 1 < 4) ? h + 1 : 4;                  /* limbs gathered */
    uint64_t acc = 0;
    for (int w = h; w > h - k; w--) acc = (acc << 16) | (uint64_t)limb[w];
    int base = 16 * (h - k + 1);

    int st = 0;
    for (int w = h - k; w >= 0; w--) if (limb[w]) { st = 1; break; }

    *top = acc; *tb = bitlen - base; *shift = base; *sticky = st;
    return neg ? -1 : 1;
}

static uint64_t round_to(uint64_t top, int tb, int bits, int sticky, int *shift)
{
    if (tb <= bits) return top;                       /* already exact */
    int drop = tb - bits;
    uint64_t sig  = top >> drop;
    uint64_t rem  = top & ((1ULL << drop) - 1);
    uint64_t half = 1ULL << (drop - 1);
    int up = (rem > half) ? 1
           : (rem < half) ? 0
           : (sticky ? 1 : (int)(sig & 1));
    sig += up;
    if (sig >> bits) { sig >>= 1; drop++; }
    *shift += drop;
    return sig;
}

void fpx_decode_f32(fpx_ctx *c, float *C)
{
    size_t nn = (size_t)c->n * c->n;
    int RL = RLIMBS(c->L);
    int32_t *limb = malloc((size_t)RL * sizeof(int32_t));
    int S = c->SA + c->SB;
    double t0 = now_sec();
    for (size_t i = 0; i < nn; i++) {
        uint64_t top; int tb, shift, sticky;
        int sgn = extract(c, i, limb, &top, &tb, &shift, &sticky);
        if (!sgn) { C[i] = 0.0f; continue; }
        uint64_t sig = round_to(top, tb, 24, sticky, &shift);
        C[i] = (float)sgn * ldexpf((float)sig, shift - S);
    }
    c->dec_secs = now_sec() - t0;
    free(limb);
}

/* Reference decode.  The error columns compare timed methods against the
 * exact product, so the reference must carry more precision than anything
 * it scores; long double gives 64 mantissa bits on x86, 11 more than the
 * fp64 methods being measured.  Elsewhere it degrades to double and the
 * fp64 rows should be read as an upper bound on their own error. */
void fpx_decode_ld(fpx_ctx *c, long double *C)
{
    size_t nn = (size_t)c->n * c->n;
    int RL = RLIMBS(c->L);
    int32_t *limb = malloc((size_t)RL * sizeof(int32_t));
    int S = c->SA + c->SB;
    for (size_t i = 0; i < nn; i++) {
        uint64_t top; int tb, shift, sticky;
        int sgn = extract(c, i, limb, &top, &tb, &shift, &sticky);
        if (!sgn) { C[i] = 0.0L; continue; }
        uint64_t sig = round_to(top, tb, 63, sticky, &shift);
        C[i] = (long double)sgn * ldexpl((long double)sig, shift - S);
    }
    free(limb);
}

void fpx_decode_f64(fpx_ctx *c, double *C)
{
    size_t nn = (size_t)c->n * c->n;
    int RL = RLIMBS(c->L);
    int32_t *limb = malloc((size_t)RL * sizeof(int32_t));
    int S = c->SA + c->SB;
    for (size_t i = 0; i < nn; i++) {
        uint64_t top; int tb, shift, sticky;
        int sgn = extract(c, i, limb, &top, &tb, &shift, &sticky);
        if (!sgn) { C[i] = 0.0; continue; }
        uint64_t sig = round_to(top, tb, 53, sticky, &shift);
        C[i] = (double)sgn * ldexp((double)sig, shift - S);
    }
    free(limb);
}

void fpx_free(fpx_ctx *c)
{
    free(c->A32); free(c->B32); free(c->Cw);
    c->A32 = NULL; c->B32 = NULL; c->Cw = NULL;
}
