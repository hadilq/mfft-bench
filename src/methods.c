/* methods.c -- the non-MFFT matrix multiplication methods.
 *
 *  mm_bigint_ijk / mm_bigint_ikj
 *      Textbook n^3 matmul where each scalar product is a schoolbook
 *      L x L limb multiplication.  This is the m^2 O(n^3) cost model the
 *      post starts from: n^3 * L^2 limb multiply-adds.
 *
 *  mm_limbplane
 *      The post's decomposition without the transform.  Write
 *          A = sum_u A_u B^u,  B = sum_v B_v B^v      (B = 2^LIMB_BITS)
 *      with A_u, B_v small-entry matrices, then
 *          AB = sum_w (sum_{u+v=w} A_u B_v) B^w.
 *      That is L^2 small n x n matrix products -- the same n^3 L^2 limb
 *      multiply-adds as schoolbook, but arranged as dense small-integer
 *      GEMMs, so it is the honest baseline MFFT has to beat.
 */
#include <stdlib.h>
#include <string.h>
#include "mfftbench.h"

void mm_bigint_ijk(const uint16_t *A, const uint16_t *B,
                   int n, int L, uint16_t *out, int RL)
{
    int P = 2 * L;
    uint64_t *acc = malloc((size_t)P * sizeof(uint64_t));
    size_t nn = (size_t)n * n;
    if (!acc) return;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            memset(acc, 0, (size_t)P * sizeof(uint64_t));
            for (int k = 0; k < n; k++) {
                const uint16_t *a = A + ((size_t)i * n + k) * L;
                const uint16_t *b = B + ((size_t)k * n + j) * L;
                for (int u = 0; u < L; u++) {
                    uint64_t au = a[u];
                    if (!au) continue;
                    uint64_t *ac = acc + u;
                    for (int v = 0; v < L; v++) ac[v] += au * b[v];
                }
            }
            uint64_t carry = 0;
            size_t idx = (size_t)i * n + j;
            for (int w = 0; w < RL; w++) {
                uint64_t v = (w < P ? acc[w] : 0) + carry;
                out[(size_t)w * nn + idx] = (uint16_t)(v & LIMB_MASK);
                carry = v >> LIMB_BITS;
            }
        }
    free(acc);
}

void mm_bigint_ikj(const uint16_t *A, const uint16_t *B,
                   int n, int L, uint16_t *out, int RL)
{
    int P = 2 * L;
    size_t nn = (size_t)n * n;
    uint64_t *row = calloc((size_t)n * P, sizeof(uint64_t));
    if (!row) return;

    for (int i = 0; i < n; i++) {
        memset(row, 0, (size_t)n * P * sizeof(uint64_t));
        for (int k = 0; k < n; k++) {
            const uint16_t *a = A + ((size_t)i * n + k) * L;
            for (int u = 0; u < L; u++) {
                uint64_t au = a[u];
                if (!au) continue;
                for (int j = 0; j < n; j++) {
                    const uint16_t *b = B + ((size_t)k * n + j) * L;
                    uint64_t *ac = row + (size_t)j * P + u;
                    for (int v = 0; v < L; v++) ac[v] += au * b[v];
                }
            }
        }
        for (int j = 0; j < n; j++) {
            uint64_t carry = 0;
            const uint64_t *acc = row + (size_t)j * P;
            size_t idx = (size_t)i * n + j;
            for (int w = 0; w < RL; w++) {
                uint64_t v = (w < P ? acc[w] : 0) + carry;
                out[(size_t)w * nn + idx] = (uint16_t)(v & LIMB_MASK);
                carry = v >> LIMB_BITS;
            }
        }
    }
    free(row);
}

/* Convolution core, on signed int32 limb planes.  Split out from
 * mm_limbplane so the fp32 path (src/fpfixed.c) can reuse it: that path
 * needs signed limbs and wants the raw int64 planes, not a normalised
 * unsigned big integer. */
void conv_limbplane(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                    int n, int L, kernel_t kern)
{
    size_t nn = (size_t)n * n;
    memset(Cw, 0, (size_t)(2 * L - 1) * nn * sizeof(int64_t));
    for (int u = 0; u < L; u++)
        for (int v = 0; v < L; v++)
            mm_accum(Cw + (size_t)(u + v) * nn,
                     A32 + (size_t)u * nn,
                     B32 + (size_t)v * nn, n, +1, kern);
}

void mm_limbplane(const uint16_t *Apl, const uint16_t *Bpl,
                  int n, int L, kernel_t kern, uint16_t *out, int RL)
{
    size_t nn = (size_t)n * n;
    int P = 2 * L - 1;
    int32_t *A32 = malloc((size_t)L * nn * sizeof(int32_t));
    int32_t *B32 = malloc((size_t)L * nn * sizeof(int32_t));
    int64_t *C   = calloc((size_t)P * nn, sizeof(int64_t));
    if (!A32 || !B32 || !C) { free(A32); free(B32); free(C); return; }

    for (size_t i = 0; i < (size_t)L * nn; i++) A32[i] = Apl[i];
    for (size_t i = 0; i < (size_t)L * nn; i++) B32[i] = Bpl[i];

    conv_limbplane(C, A32, B32, n, L, kern);

    normalize_planes(C, P, n, out, RL);
    free(A32); free(B32); free(C);
}

/* ------------------------------------------------------------------ *
 * Karatsuba convolution over limb planes.
 *
 * MFFT's balancing costs 8*L*S ~ 5.7*L^1.5 matrix products; schoolbook
 * costs L^2.  Karatsuba costs L^log2(3) = L^1.585.  Asymptotically MFFT
 * wins, but only past L ~ 10^8 limbs, because of that 5.7 constant --
 * so across every width anyone would actually multiply, Karatsuba is the
 * one to beat.  It matters most exactly where the fp32 embedding lands
 * (L = 2..32), which is the regime MFFT handles worst.
 *
 * Splits the limb polynomial in half and trades one of the four
 * sub-products for three additions, recursively:
 *   A = A0 + x^h A1,  B = B0 + x^h B1
 *   C = A0B0 + x^h ((A0+A1)(B0+B1) - A0B0 - A1B1) + x^2h A1B1
 * ------------------------------------------------------------------ */
static void kar_rec(int64_t *C, const int32_t *A, const int32_t *B,
                    int L, int n, kernel_t kern)
{
    size_t nn = (size_t)n * n;
    if (L == 1) {
        memset(C, 0, nn * sizeof(int64_t));
        mm_accum(C, A, B, n, +1, kern);
        return;
    }
    int h = L / 2, ph = 2 * h - 1;
    size_t psz = (size_t)ph * nn;

    int64_t *P0 = malloc(psz * sizeof(int64_t));
    int64_t *P2 = malloc(psz * sizeof(int64_t));
    int64_t *Ps = malloc(psz * sizeof(int64_t));
    int32_t *As = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Bs = malloc((size_t)h * nn * sizeof(int32_t));
    if (!P0 || !P2 || !Ps || !As || !Bs) {
        free(P0); free(P2); free(Ps); free(As); free(Bs);
        memset(C, 0, (size_t)(2 * L - 1) * nn * sizeof(int64_t));
        for (int u = 0; u < L; u++)
            for (int v = 0; v < L; v++)
                mm_accum(C + (size_t)(u + v) * nn, A + (size_t)u * nn,
                         B + (size_t)v * nn, n, +1, kern);
        return;
    }

    for (size_t i = 0; i < (size_t)h * nn; i++) As[i] = A[i] + A[(size_t)h*nn + i];
    for (size_t i = 0; i < (size_t)h * nn; i++) Bs[i] = B[i] + B[(size_t)h*nn + i];

    kar_rec(P0, A, B, h, n, kern);
    kar_rec(P2, A + (size_t)h*nn, B + (size_t)h*nn, h, n, kern);
    kar_rec(Ps, As, Bs, h, n, kern);

    memset(C, 0, (size_t)(2 * L - 1) * nn * sizeof(int64_t));
    for (size_t i = 0; i < psz; i++) C[i] += P0[i];
    for (size_t i = 0; i < psz; i++) C[(size_t)2*h*nn + i] += P2[i];
    for (size_t i = 0; i < psz; i++)
        C[(size_t)h*nn + i] += Ps[i] - P0[i] - P2[i];

    free(P0); free(P2); free(Ps); free(As); free(Bs);
}

void conv_karatsuba(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                    int n, int L, kernel_t kern)
{
    kar_rec(Cw, A32, B32, L, n, kern);
}


long long karatsuba_products(int L)
{
    long long p = 1;
    while (L > 1) { p *= 3; L >>= 1; }
    return p;
}

/* ------------------------------------------------------------------ *
 * Toom-3 convolution over limb planes (Bodrato / GMP-style).
 *
 * Splits into 3 parts of length h = ceil(L/3), evaluates at
 * {0, 1, -1, 2, inf}, multiplies (5 recursive products), interpolates
 * with exact divisions by 2 and 3.  Base cases: L=1 schoolbook, L=2
 * Karatsuba (Toom-2).
 *
 * Product count is ~L^log3(5) ≈ L^1.465, beating Karatsuba's L^1.585
 * and schoolbook's L^2 at the widths the float embeddings use
 * (see toom3_products).  The theoretical floor for a dense linear
 * convolution is 2L-1; one-level Toom-k for large k reaches it but
 * evaluation weights overflow int32 limb planes, so we stay recursive.
 * ------------------------------------------------------------------ */

static void toom3_rec(int64_t *C, const int32_t *A, const int32_t *B,
                      int L, int n, kernel_t kern)
{
    size_t nn = (size_t)n * n;
    if (L <= 0) return;
    if (L == 1) {
        memset(C, 0, nn * sizeof(int64_t));
        mm_accum(C, A, B, n, +1, kern);
        return;
    }
    if (L == 2) {
        kar_rec(C, A, B, 2, n, kern);
        return;
    }

    int h = (L + 2) / 3;          /* ceil(L/3) */
    int Lpad = 3 * h;
    int ph = 2 * h - 1;           /* planes per sub-product */
    size_t psz = (size_t)ph * nn;
    int fullP = 6 * h - 1;        /* planes in padded product */

    int32_t *Ap = calloc((size_t)Lpad * nn, sizeof(int32_t));
    int32_t *Bp = calloc((size_t)Lpad * nn, sizeof(int32_t));
    int32_t *As1 = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Asm = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *As2 = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Bs1 = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Bsm = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Bs2 = malloc((size_t)h * nn * sizeof(int32_t));
    int64_t *V0 = malloc(psz * sizeof(int64_t));
    int64_t *V1 = malloc(psz * sizeof(int64_t));
    int64_t *Vm1 = malloc(psz * sizeof(int64_t));
    int64_t *V2 = malloc(psz * sizeof(int64_t));
    int64_t *Vinf = malloc(psz * sizeof(int64_t));
    int64_t *c1 = malloc(psz * sizeof(int64_t));
    int64_t *c2 = malloc(psz * sizeof(int64_t));
    int64_t *c3 = malloc(psz * sizeof(int64_t));
    int64_t *Ct = calloc((size_t)fullP * nn, sizeof(int64_t));
    if (!Ap || !Bp || !As1 || !Asm || !As2 || !Bs1 || !Bsm || !Bs2 ||
        !V0 || !V1 || !Vm1 || !V2 || !Vinf || !c1 || !c2 || !c3 || !Ct) {
        free(Ap); free(Bp); free(As1); free(Asm); free(As2);
        free(Bs1); free(Bsm); free(Bs2);
        free(V0); free(V1); free(Vm1); free(V2); free(Vinf);
        free(c1); free(c2); free(c3); free(Ct);
        /* OOM fallback: schoolbook */
        memset(C, 0, (size_t)(2 * L - 1) * nn * sizeof(int64_t));
        for (int u = 0; u < L; u++)
            for (int v = 0; v < L; v++)
                mm_accum(C + (size_t)(u + v) * nn, A + (size_t)u * nn,
                         B + (size_t)v * nn, n, +1, kern);
        return;
    }

    memcpy(Ap, A, (size_t)L * nn * sizeof(int32_t));
    memcpy(Bp, B, (size_t)L * nn * sizeof(int32_t));
    int32_t *a0 = Ap, *a1 = Ap + (size_t)h * nn, *a2 = Ap + (size_t)2 * h * nn;
    int32_t *b0 = Bp, *b1 = Bp + (size_t)h * nn, *b2 = Bp + (size_t)2 * h * nn;

    for (size_t i = 0; i < (size_t)h * nn; i++) {
        int32_t a0i = a0[i], a1i = a1[i], a2i = a2[i];
        int32_t b0i = b0[i], b1i = b1[i], b2i = b2[i];
        As1[i] = a0i + a1i + a2i;
        Asm[i] = a0i - a1i + a2i;
        As2[i] = a0i + 2 * a1i + 4 * a2i;
        Bs1[i] = b0i + b1i + b2i;
        Bsm[i] = b0i - b1i + b2i;
        Bs2[i] = b0i + 2 * b1i + 4 * b2i;
    }

    toom3_rec(V0,   a0,  b0,  h, n, kern);
    toom3_rec(V1,   As1, Bs1, h, n, kern);
    toom3_rec(Vm1,  Asm, Bsm, h, n, kern);
    toom3_rec(V2,   As2, Bs2, h, n, kern);
    toom3_rec(Vinf, a2,  b2,  h, n, kern);

    /* Interpolation (verified on monomials): 
     *   s = (w1+wm)/2 = c0+c2+c4
     *   d = (w1-wm)/2 = c1+c3
     *   c2 = s - c0 - c4
     *   t = (w2 - c0 - 4*c2 - 16*c4)/2 = c1+4*c3
     *   c3 = (t-d)/3,  c1 = d - c3
     */
    for (size_t i = 0; i < psz; i++) {
        int64_t w0 = V0[i], w1 = V1[i], wm = Vm1[i], w2 = V2[i], wi = Vinf[i];
        int64_t s  = (w1 + wm) / 2;
        int64_t d  = (w1 - wm) / 2;
        int64_t c2v = s - w0 - wi;
        int64_t t  = (w2 - w0 - 4 * c2v - 16 * wi) / 2;
        int64_t c3v = (t - d) / 3;
        int64_t c1v = d - c3v;
        c1[i] = c1v;
        c2[i] = c2v;
        c3[i] = c3v;
    }

    /* Compose: c_k lives at offset k*h in the padded product. */
    for (size_t i = 0; i < psz; i++) {
        Ct[i]                                += V0[i];
        Ct[(size_t)h * nn + i]               += c1[i];
        Ct[(size_t)(2 * h) * nn + i]         += c2[i];
        Ct[(size_t)(3 * h) * nn + i]         += c3[i];
        Ct[(size_t)(4 * h) * nn + i]         += Vinf[i];
    }

    memcpy(C, Ct, (size_t)(2 * L - 1) * nn * sizeof(int64_t));

    free(Ap); free(Bp); free(As1); free(Asm); free(As2);
    free(Bs1); free(Bsm); free(Bs2);
    free(V0); free(V1); free(Vm1); free(V2); free(Vinf);
    free(c1); free(c2); free(c3); free(Ct);
}

long long toom3_products(int L)
{
    if (L <= 0) return 0;
    if (L == 1) return 1;
    if (L == 2) return 3;
    int h = (L + 2) / 3;
    return 5LL * toom3_products(h);
}

void conv_toom3(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                int n, int L, kernel_t kern)
{
    /* Prefer whichever of Toom-3 / Karatsuba issues fewer leaf products.
     * At power-of-two L Karatsuba often wins; Toom-3 wins for L=3,6,9,... */
    long long tp = toom3_products(L);
    long long kp = karatsuba_products(L);
    if (tp >= kp)
        kar_rec(Cw, A32, B32, L, n, kern);
    else
        toom3_rec(Cw, A32, B32, L, n, kern);
}

/* ------------------------------------------------------------------ *
 * Even/odd-index Karatsuba (Cooley–Tukey limb basis).
 *
 * High/low Karatsuba splits A = A_lo + x^h A_hi.  Even/odd splits
 *   A(x) = Ae(x^2) + x Ao(x^2)
 * so limbs are interleaved: Ae = (A0,A2,...), Ao = (A1,A3,...).
 * Same 3-product recurrence, different memory traffic:
 *   P0 = Ae*Be,  P2 = Ao*Bo,  P1 = (Ae+Ao)*(Be+Bo) - P0 - P2
 *   C(x) = P0(x^2) + x P1(x^2) + x^2 P2(x^2)
 * i.e. P0[k] -> C[2k], P1[k] -> C[2k+1], P2[k] -> C[2k+2].
 *
 * Product count matches karatsuba_products for power-of-two L; hybrid
 * may still prefer this layout on some microarchitectures (Phase B3).
 * ------------------------------------------------------------------ */
static void evenodd_rec(int64_t *C, const int32_t *A, const int32_t *B,
                        int L, int n, kernel_t kern)
{
    size_t nn = (size_t)n * n;
    if (L <= 0) return;
    if (L == 1) {
        memset(C, 0, nn * sizeof(int64_t));
        mm_accum(C, A, B, n, +1, kern);
        return;
    }
    /* Odd L: one high/low Karatsuba step avoids uneven even/odd counts. */
    if (L & 1) {
        kar_rec(C, A, B, L, n, kern);
        return;
    }

    int h = L / 2;
    int ph = 2 * h - 1;
    size_t psz = (size_t)ph * nn;
    int fullP = 2 * L - 1;

    int32_t *Ae = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Ao = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Be = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Bo = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *As = malloc((size_t)h * nn * sizeof(int32_t));
    int32_t *Bs = malloc((size_t)h * nn * sizeof(int32_t));
    int64_t *P0 = malloc(psz * sizeof(int64_t));
    int64_t *P2 = malloc(psz * sizeof(int64_t));
    int64_t *P1 = malloc(psz * sizeof(int64_t));
    if (!Ae || !Ao || !Be || !Bo || !As || !Bs || !P0 || !P2 || !P1) {
        free(Ae); free(Ao); free(Be); free(Bo); free(As); free(Bs);
        free(P0); free(P2); free(P1);
        kar_rec(C, A, B, L, n, kern);
        return;
    }

    for (int i = 0; i < h; i++) {
        memcpy(Ae + (size_t)i * nn, A + (size_t)(2 * i) * nn, nn * sizeof(int32_t));
        memcpy(Ao + (size_t)i * nn, A + (size_t)(2 * i + 1) * nn, nn * sizeof(int32_t));
        memcpy(Be + (size_t)i * nn, B + (size_t)(2 * i) * nn, nn * sizeof(int32_t));
        memcpy(Bo + (size_t)i * nn, B + (size_t)(2 * i + 1) * nn, nn * sizeof(int32_t));
    }
    for (size_t i = 0; i < (size_t)h * nn; i++) {
        As[i] = Ae[i] + Ao[i];
        Bs[i] = Be[i] + Bo[i];
    }

    /* Recurse with evenodd when subproblem is even-sized; else Karatsuba. */
    evenodd_rec(P0, Ae, Be, h, n, kern);
    evenodd_rec(P2, Ao, Bo, h, n, kern);
    evenodd_rec(P1, As, Bs, h, n, kern);

    memset(C, 0, (size_t)fullP * nn * sizeof(int64_t));
    for (int k = 0; k < ph; k++) {
        size_t base = (size_t)k * nn;
        for (size_t i = 0; i < nn; i++) {
            int64_t p0 = P0[base + i];
            int64_t p1 = P1[base + i] - p0 - P2[base + i];
            int64_t p2 = P2[base + i];
            C[(size_t)(2 * k) * nn + i]     += p0;
            C[(size_t)(2 * k + 1) * nn + i] += p1;
            C[(size_t)(2 * k + 2) * nn + i] += p2;
        }
    }

    free(Ae); free(Ao); free(Be); free(Bo); free(As); free(Bs);
    free(P0); free(P2); free(P1);
}

long long evenodd_products(int L)
{
    if (L <= 0) return 0;
    if (L == 1) return 1;
    /* Odd L falls back to kar_rec; count matches karatsuba. */
    if (L & 1) return karatsuba_products(L);
    long long p = 1;
    int ell = L;
    while (ell > 1) { p *= 3; ell >>= 1; }
    return p;
}

void conv_evenodd(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                  int n, int L, kernel_t kern)
{
    evenodd_rec(Cw, A32, B32, L, n, kern);
}

/* ------------------------------------------------------------------ *
 * Hybrid limb convolution (B3 Phase 4).
 *
 * At top-level L pick the strategy with the fewest estimated leaf
 * products among schoolbook, Karatsuba, Toom-3, and even/odd-index.
 * Ties: Toom < Karatsuba < evenodd < schoolbook (preserve existing
 * Toom/Kara preference; evenodd is experimental when counts equal).
 * Subproblems keep their own algorithms (kar_rec / toom3_rec /
 * evenodd_rec already recurse internally).
 * ------------------------------------------------------------------ */
long long hybrid_products(int L)
{
    if (L <= 0) return 0;
    long long best = (long long)L * (long long)L;
    long long kp = karatsuba_products(L);
    if (kp < best) best = kp;
    long long tp = toom3_products(L);
    if (tp < best) best = tp;
    long long ep = evenodd_products(L);
    if (ep < best) best = ep;
    return best;
}

void conv_hybrid(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                 int n, int L, kernel_t kern)
{
    long long sb = (long long)L * (long long)L;
    long long kp = karatsuba_products(L);
    long long tp = toom3_products(L);
    long long ep = evenodd_products(L);

    long long best = sb;
    int choice = 0; /* 0 schoolbook, 1 kara, 2 toom, 3 evenodd */
    if (tp < best) { best = tp; choice = 2; }
    if (kp < best) { best = kp; choice = 1; }
    if (ep < best) { best = ep; choice = 3; }

    if (choice == 0) {
        memset(Cw, 0, (size_t)(2 * L - 1) * (size_t)n * n * sizeof(int64_t));
        for (int u = 0; u < L; u++)
            for (int v = 0; v < L; v++)
                mm_accum(Cw + (size_t)(u + v) * (size_t)n * n,
                         A32 + (size_t)u * (size_t)n * n,
                         B32 + (size_t)v * (size_t)n * n, n, +1, kern);
    } else if (choice == 1) {
        kar_rec(Cw, A32, B32, L, n, kern);
    } else if (choice == 2) {
        toom3_rec(Cw, A32, B32, L, n, kern);
    } else {
        evenodd_rec(Cw, A32, B32, L, n, kern);
    }
}




void mm_karatsuba(const uint16_t *Apl, const uint16_t *Bpl,
                  int n, int L, kernel_t kern, uint16_t *out, int RL)
{
    size_t nn = (size_t)n * n;
    int32_t *A32 = malloc((size_t)L * nn * sizeof(int32_t));
    int32_t *B32 = malloc((size_t)L * nn * sizeof(int32_t));
    int64_t *C   = calloc((size_t)(2 * L - 1) * nn, sizeof(int64_t));
    if (!A32 || !B32 || !C) { free(A32); free(B32); free(C); return; }
    for (size_t i = 0; i < (size_t)L * nn; i++) A32[i] = Apl[i];
    for (size_t i = 0; i < (size_t)L * nn; i++) B32[i] = Bpl[i];
    conv_karatsuba(C, A32, B32, n, L, kern);
    normalize_planes(C, 2 * L - 1, n, out, RL);
    free(A32); free(B32); free(C);
}
