/* kernel.c -- the inner n x n small-integer matrix product.
 *
 * Every "plane" method (schoolbook over bit/limb planes, and MFFT) reduces
 * the big-integer matmul to a number of these.  Keeping the kernel identical
 * across methods means the benchmark measures the *algorithm*, not the
 * data type or the loop tuning.
 */
#include <stdlib.h>
#include <string.h>
#include "mfftbench.h"

long long g_kernel_calls = 0;
long long g_strassen_cutoff = 128;

static const char *knames[KERNEL__COUNT] = {
    "ikj", "blocked", "packed", "strassen", "winograd", "bitplane",
    "convk", "convkara", "boolpack", "bitpack"
};

const char *kernel_name(kernel_t k)
{
    if (k < 0 || k >= KERNEL__COUNT) return "?";
    return knames[k];
}

int kernel_from_name(const char *s)
{
    for (int i = 0; i < KERNEL__COUNT; i++)
        if (strcmp(s, knames[i]) == 0) return i;
    return -1;
}

/* ------------------------------------------------------------------ */
static void mm_ikj(int64_t *C, const int32_t *A, const int32_t *B,
                   int n, int sign)
{
    for (int i = 0; i < n; i++) {
        int64_t *Cr = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            int64_t a = A[(size_t)i * n + k];
            if (!a) continue;
            a *= sign;
            const int32_t *Br = B + (size_t)k * n;
            for (int j = 0; j < n; j++)
                Cr[j] += a * Br[j];
        }
    }
}

#define BS 64
static void mm_blocked(int64_t *C, const int32_t *A, const int32_t *B,
                       int n, int sign)
{
    for (int ii = 0; ii < n; ii += BS) {
        int imax = ii + BS < n ? ii + BS : n;
        for (int kk = 0; kk < n; kk += BS) {
            int kmax = kk + BS < n ? kk + BS : n;
            for (int jj = 0; jj < n; jj += BS) {
                int jmax = jj + BS < n ? jj + BS : n;
                for (int i = ii; i < imax; i++) {
                    int64_t *Cr = C + (size_t)i * n;
                    for (int k = kk; k < kmax; k++) {
                        int64_t a = A[(size_t)i * n + k];
                        if (!a) continue;
                        a *= sign;
                        const int32_t *Br = B + (size_t)k * n;
                        for (int j = jj; j < jmax; j++)
                            Cr[j] += a * Br[j];
                    }
                }
            }
        }
    }
}


/* ------------------------------------------------------------------ *
 * Packed micro-kernel GEMM -- the structure every production BLAS
 * (OpenBLAS, BLIS, oneDNN) actually uses: cache-block the loops, copy
 * each block into a contiguous panel so the inner loop streams, and
 * accumulate a small register tile.  This is the practical state of the
 * art; the sub-cubic algorithms sit on top of it, not instead of it.
 * ------------------------------------------------------------------ */
#define IMC 128
#define IKC 128
#define INC 256

#if defined(__AVX512F__)
#  define IVBYTES 64
#elif defined(__AVX2__) || defined(__ARM_NEON) || defined(__aarch64__)
#  define IVBYTES 32
#else
#  define IVBYTES 16
#endif
typedef int64_t vl  __attribute__((vector_size(IVBYTES)));
typedef int64_t vlu __attribute__((vector_size(IVBYTES), aligned(8)));
typedef int32_t vwu __attribute__((vector_size(IVBYTES/2), aligned(4)));
#define IVL ((int)(sizeof(vl) / sizeof(int64_t)))
#define IMR 6
#define INR (2 * IVL)
#define IMIN(a,b) ((a) < (b) ? (a) : (b))

static void ipack_a(int32_t *Ap, const int32_t *A, int n,
                    int ic, int mc, int pc, int kc)
{
    for (int pi = 0; pi * IMR < mc; pi++) {
        int32_t *d = Ap + (size_t)pi * IMR * kc;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < IMR; i++)
                d[p * IMR + i] = (pi * IMR + i < mc)
                    ? A[(size_t)(ic + pi * IMR + i) * n + pc + p] : 0;
    }
}

static void ipack_b(int32_t *Bp, const int32_t *B, int n,
                    int jc, int nc, int pc, int kc)
{
    for (int pj = 0; pj * INR < nc; pj++) {
        int32_t *d = Bp + (size_t)pj * INR * kc;
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < INR; j++)
                d[p * INR + j] = (pj * INR + j < nc)
                    ? B[(size_t)(pc + p) * n + jc + pj * INR + j] : 0;
    }
}

static void imicro(const int32_t *restrict a, const int32_t *restrict b,
                   int64_t *c, int ldc, int kc, int mr, int nr, int sign)
{
    vl acc[IMR][2];
    for (int i = 0; i < IMR; i++) { acc[i][0] = (vl){0}; acc[i][1] = (vl){0}; }

    for (int p = 0; p < kc; p++) {
        const int32_t *bp = b + (size_t)p * INR;
        vl b0 = __builtin_convertvector(*(const vwu *)bp, vl);
        vl b1 = __builtin_convertvector(*(const vwu *)(bp + IVL), vl);
        const int32_t *ap = a + (size_t)p * IMR;
        for (int i = 0; i < IMR; i++) {
            vl av = (vl){0} + (int64_t)ap[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }

    int64_t tmp[IMR][INR];
    for (int i = 0; i < IMR; i++) {
        *(vlu *)&tmp[i][0]   = acc[i][0];
        *(vlu *)&tmp[i][IVL] = acc[i][1];
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            c[(size_t)i * ldc + j] += sign > 0 ? tmp[i][j] : -tmp[i][j];
}

static void mm_packed(int64_t *C, const int32_t *A, const int32_t *B,
                      int n, int sign)
{
    size_t apsz = (size_t)(IMC / IMR + 1) * IMR * IKC;
    size_t bpsz = (size_t)(INC / INR + 1) * INR * IKC;
    int32_t *Ap = malloc(apsz * sizeof(int32_t));
    int32_t *Bp = malloc(bpsz * sizeof(int32_t));
    if (!Ap || !Bp) { free(Ap); free(Bp); mm_blocked(C, A, B, n, sign); return; }

    for (int jc = 0; jc < n; jc += INC) {
        int nc = IMIN(INC, n - jc);
        for (int pc = 0; pc < n; pc += IKC) {
            int kc = IMIN(IKC, n - pc);
            ipack_b(Bp, B, n, jc, nc, pc, kc);
            for (int ic = 0; ic < n; ic += IMC) {
                int mc = IMIN(IMC, n - ic);
                ipack_a(Ap, A, n, ic, mc, pc, kc);
                for (int j = 0; j < nc; j += INR) {
                    int nr = IMIN(INR, nc - j);
                    for (int i = 0; i < mc; i += IMR) {
                        int mr = IMIN(IMR, mc - i);
                        imicro(Ap + (size_t)(i / IMR) * IMR * kc,
                               Bp + (size_t)(j / INR) * INR * kc,
                               C + (size_t)(ic + i) * n + jc + j,
                               n, kc, mr, nr, sign);
                    }
                }
            }
        }
    }
    free(Ap); free(Bp);
}

/* ------------------------------------------------------------------ *
 * Strassen over int64.  Included because the post explicitly notes that
 * MFFT is "orthogonal to the other methods" -- here it can be stacked
 * underneath MFFT by picking --kernel strassen.
 * ------------------------------------------------------------------ */
/* Base case for the recursive schemes: a packed int64 GEMM, so Strassen
 * and Winograd are measured against a real kernel rather than a strawman
 * (bottoming out in a naive loop flatters them). */
static void b64_pack_a(int64_t *Ap, const int64_t *A, int n,
                       int ic, int mc, int pc, int kc)
{
    for (int pi = 0; pi * IMR < mc; pi++) {
        int64_t *d = Ap + (size_t)pi * IMR * kc;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < IMR; i++)
                d[p * IMR + i] = (pi * IMR + i < mc)
                    ? A[(size_t)(ic + pi * IMR + i) * n + pc + p] : 0;
    }
}
static void b64_pack_b(int64_t *Bp, const int64_t *B, int n,
                       int jc, int nc, int pc, int kc)
{
    for (int pj = 0; pj * INR < nc; pj++) {
        int64_t *d = Bp + (size_t)pj * INR * kc;
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < INR; j++)
                d[p * INR + j] = (pj * INR + j < nc)
                    ? B[(size_t)(pc + p) * n + jc + pj * INR + j] : 0;
    }
}
static void b64_micro(const int64_t *restrict a, const int64_t *restrict b,
                      int64_t *c, int ldc, int kc, int mr, int nr)
{
    vl acc[IMR][2];
    for (int i = 0; i < IMR; i++) { acc[i][0] = (vl){0}; acc[i][1] = (vl){0}; }
    for (int p = 0; p < kc; p++) {
        const int64_t *bp = b + (size_t)p * INR;
        vl b0 = *(const vlu *)bp, b1 = *(const vlu *)(bp + IVL);
        const int64_t *ap = a + (size_t)p * IMR;
        for (int i = 0; i < IMR; i++) {
            vl av = (vl){0} + ap[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }
    int64_t tmp[IMR][INR];
    for (int i = 0; i < IMR; i++) {
        *(vlu *)&tmp[i][0]   = acc[i][0];
        *(vlu *)&tmp[i][IVL] = acc[i][1];
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++) c[(size_t)i * ldc + j] += tmp[i][j];
}

static void st_base(int64_t *C, const int64_t *A, const int64_t *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(int64_t));
    int64_t *Ap = malloc((size_t)(IMC / IMR + 1) * IMR * IKC * sizeof(int64_t));
    int64_t *Bp = malloc((size_t)(INC / INR + 1) * INR * IKC * sizeof(int64_t));
    if (!Ap || !Bp) {
        free(Ap); free(Bp);
        for (int i = 0; i < n; i++) {
            int64_t *Cr = C + (size_t)i * n;
            for (int k = 0; k < n; k++) {
                int64_t a = A[(size_t)i * n + k];
                if (!a) continue;
                const int64_t *Br = B + (size_t)k * n;
                for (int j = 0; j < n; j++) Cr[j] += a * Br[j];
            }
        }
        return;
    }
    for (int jc = 0; jc < n; jc += INC) {
        int nc = IMIN(INC, n - jc);
        for (int pc = 0; pc < n; pc += IKC) {
            int kc = IMIN(IKC, n - pc);
            b64_pack_b(Bp, B, n, jc, nc, pc, kc);
            for (int ic = 0; ic < n; ic += IMC) {
                int mc = IMIN(IMC, n - ic);
                b64_pack_a(Ap, A, n, ic, mc, pc, kc);
                for (int j = 0; j < nc; j += INR) {
                    int nr = IMIN(INR, nc - j);
                    for (int i = 0; i < mc; i += IMR) {
                        int mr = IMIN(IMR, mc - i);
                        b64_micro(Ap + (size_t)(i / IMR) * IMR * kc,
                                  Bp + (size_t)(j / INR) * INR * kc,
                                  C + (size_t)(ic + i) * n + jc + j,
                                  n, kc, mr, nr);
                    }
                }
            }
        }
    }
    free(Ap); free(Bp);
}

static void q_get(int64_t *q, const int64_t *M, int n, int h, int r, int c)
{
    for (int i = 0; i < h; i++)
        memcpy(q + (size_t)i * h,
               M + (size_t)(r * h + i) * n + (size_t)c * h,
               (size_t)h * sizeof(int64_t));
}

static void q_put(int64_t *M, const int64_t *q, int n, int h, int r, int c)
{
    for (int i = 0; i < h; i++)
        memcpy(M + (size_t)(r * h + i) * n + (size_t)c * h,
               q + (size_t)i * h, (size_t)h * sizeof(int64_t));
}

static void vadd(int64_t *d, const int64_t *a, const int64_t *b, size_t n)
{ for (size_t i = 0; i < n; i++) d[i] = a[i] + b[i]; }
static void vsub(int64_t *d, const int64_t *a, const int64_t *b, size_t n)
{ for (size_t i = 0; i < n; i++) d[i] = a[i] - b[i]; }

static void st_mul(int64_t *C, const int64_t *A, const int64_t *B, int n)
{
    if (n <= g_strassen_cutoff || (n & 1)) { st_base(C, A, B, n); return; }

    int h = n / 2;
    size_t hs = (size_t)h * h;
    int64_t *buf = malloc(hs * sizeof(int64_t) * 15);
    if (!buf) { st_base(C, A, B, n); return; }

    int64_t *a11 = buf,          *a12 = buf + hs,      *a21 = buf + 2*hs;
    int64_t *a22 = buf + 3*hs,   *b11 = buf + 4*hs,    *b12 = buf + 5*hs;
    int64_t *b21 = buf + 6*hs,   *b22 = buf + 7*hs;
    int64_t *t1  = buf + 8*hs,   *t2  = buf + 9*hs;
    int64_t *m   = buf + 10*hs;              /* m[0..4] scratch results */

    q_get(a11, A, n, h, 0, 0); q_get(a12, A, n, h, 0, 1);
    q_get(a21, A, n, h, 1, 0); q_get(a22, A, n, h, 1, 1);
    q_get(b11, B, n, h, 0, 0); q_get(b12, B, n, h, 0, 1);
    q_get(b21, B, n, h, 1, 0); q_get(b22, B, n, h, 1, 1);

    int64_t *M1 = m, *M2 = m + hs, *M3 = m + 2*hs, *M4 = m + 3*hs,
            *M5 = m + 4*hs;
    int64_t *M6 = malloc(hs * sizeof(int64_t));
    int64_t *M7 = malloc(hs * sizeof(int64_t));
    int64_t *acc = malloc(hs * sizeof(int64_t));
    if (!M6 || !M7 || !acc) { free(buf); free(M6); free(M7); free(acc);
                              st_base(C, A, B, n); return; }

    vadd(t1, a11, a22, hs); vadd(t2, b11, b22, hs); st_mul(M1, t1, t2, h);
    vadd(t1, a21, a22, hs);                          st_mul(M2, t1, b11, h);
    vsub(t2, b12, b22, hs);                          st_mul(M3, a11, t2, h);
    vsub(t2, b21, b11, hs);                          st_mul(M4, a22, t2, h);
    vadd(t1, a11, a12, hs);                          st_mul(M5, t1, b22, h);
    vsub(t1, a21, a11, hs); vadd(t2, b11, b12, hs); st_mul(M6, t1, t2, h);
    vsub(t1, a12, a22, hs); vadd(t2, b21, b22, hs); st_mul(M7, t1, t2, h);

    vadd(acc, M1, M4, hs); vsub(acc, acc, M5, hs); vadd(acc, acc, M7, hs);
    q_put(C, acc, n, h, 0, 0);
    vadd(acc, M3, M5, hs);                 q_put(C, acc, n, h, 0, 1);
    vadd(acc, M2, M4, hs);                 q_put(C, acc, n, h, 1, 0);
    vsub(acc, M1, M2, hs); vadd(acc, acc, M3, hs); vadd(acc, acc, M6, hs);
    q_put(C, acc, n, h, 1, 1);

    free(buf); free(M6); free(M7); free(acc);
}


/* ------------------------------------------------------------------ *
 * Strassen-Winograd: the same 7 multiplications, but 15 additions per
 * level instead of Strassen's 18.  This -- not any of the newer tensor
 * decompositions -- is the fast algorithm production libraries ship,
 * because at practical cutoffs the additions dominate.
 * ------------------------------------------------------------------ */
static void sw_mul(int64_t *C, const int64_t *A, const int64_t *B, int n)
{
    if (n <= g_strassen_cutoff || (n & 1)) { st_base(C, A, B, n); return; }

    int h = n / 2;
    size_t hs = (size_t)h * h;
    int64_t *buf = malloc(hs * sizeof(int64_t) * 23);
    if (!buf) { st_base(C, A, B, n); return; }

    int64_t *a11=buf, *a12=buf+hs, *a21=buf+2*hs, *a22=buf+3*hs;
    int64_t *b11=buf+4*hs, *b12=buf+5*hs, *b21=buf+6*hs, *b22=buf+7*hs;
    int64_t *S1=buf+8*hs,  *S2=buf+9*hs,  *S3=buf+10*hs, *S4=buf+11*hs;
    int64_t *T1=buf+12*hs, *T2=buf+13*hs, *T3=buf+14*hs, *T4=buf+15*hs;
    int64_t *P1=buf+16*hs, *P2=buf+17*hs, *P3=buf+18*hs, *P4=buf+19*hs;
    int64_t *P5=buf+20*hs, *P6=buf+21*hs, *P7=buf+22*hs;

    q_get(a11,A,n,h,0,0); q_get(a12,A,n,h,0,1);
    q_get(a21,A,n,h,1,0); q_get(a22,A,n,h,1,1);
    q_get(b11,B,n,h,0,0); q_get(b12,B,n,h,0,1);
    q_get(b21,B,n,h,1,0); q_get(b22,B,n,h,1,1);

    vadd(S1, a21, a22, hs);          /* S1 = A21 + A22 */
    vsub(S2, S1,  a11, hs);          /* S2 = S1  - A11 */
    vsub(S3, a11, a21, hs);          /* S3 = A11 - A21 */
    vsub(S4, a12, S2,  hs);          /* S4 = A12 - S2  */
    vsub(T1, b12, b11, hs);          /* T1 = B12 - B11 */
    vsub(T2, b22, T1,  hs);          /* T2 = B22 - T1  */
    vsub(T3, b22, b12, hs);          /* T3 = B22 - B12 */
    vsub(T4, T2,  b21, hs);          /* T4 = T2  - B21 */

    sw_mul(P1, a11, b11, h);
    sw_mul(P2, a12, b21, h);
    sw_mul(P3, S4,  b22, h);
    sw_mul(P4, a22, T4,  h);
    sw_mul(P5, S1,  T1,  h);
    sw_mul(P6, S2,  T2,  h);
    sw_mul(P7, S3,  T3,  h);

    int64_t *U = S1, *V = S2;        /* scratch, no longer needed */
    vadd(U, P1, P2, hs);             q_put(C, U, n, h, 0, 0);  /* C11 */
    vadd(U, P1, P6, hs);             /* U2 */
    vadd(V, U,  P7, hs);             /* U3 */
    vadd(U, U,  P5, hs);             /* U4 */
    vadd(U, U,  P3, hs);             q_put(C, U, n, h, 0, 1);  /* C12 */
    vsub(U, V,  P4, hs);             q_put(C, U, n, h, 1, 0);  /* C21 */
    vadd(U, V,  P5, hs);             q_put(C, U, n, h, 1, 1);  /* C22 */

    free(buf);
}

static void mm_fast_rec(int64_t *C, const int32_t *A, const int32_t *B,
                        int n, int sign,
                        void (*rec)(int64_t *, const int64_t *,
                                    const int64_t *, int))
{
    size_t nn = (size_t)n * n;
    int64_t *A64 = malloc(nn * sizeof(int64_t));
    int64_t *B64 = malloc(nn * sizeof(int64_t));
    int64_t *T   = malloc(nn * sizeof(int64_t));
    if (!A64 || !B64 || !T) { free(A64); free(B64); free(T);
                              mm_blocked(C, A, B, n, sign); return; }
    for (size_t i = 0; i < nn; i++) A64[i] = A[i];
    for (size_t i = 0; i < nn; i++) B64[i] = B[i];
    rec(T, A64, B64, n);
    if (sign > 0) for (size_t i = 0; i < nn; i++) C[i] += T[i];
    else          for (size_t i = 0; i < nn; i++) C[i] -= T[i];
    free(A64); free(B64); free(T);
}


/* ------------------------------------------------------------------ *
 * Bit-plane leaf: expand A in binary and only touch set bits.
 *
 *   A = sum_b 2^b A^{(b)},  A^{(b)} in {0,1}
 *   (A B)_{ij} = sum_b 2^b sum_{k: A^{(b)}_{ik}=1} B_{kj}
 *
 * For each nonzero A_ik we walk set bits of |A_ik| (ctz loop) and
 * add/sub (B_k,: << b) into row i.  No general integer mul in the
 * inner loop — only shifts and adds.  Wins when limbs are narrow or
 * sparse in bits; dense 16-bit random limbs usually lose to packed
 * mul or Strassen.
 * ------------------------------------------------------------------ */
static void mm_bitplane(int64_t *C, const int32_t *A, const int32_t *B,
                        int n, int sign)
{
    for (int i = 0; i < n; i++) {
        int64_t *Cr = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            int32_t a = A[(size_t)i * n + k];
            if (!a) continue;
            int s = sign;
            uint32_t ua;
            if (a < 0) {
                /* avoid UB on INT32_MIN */
                ua = (uint32_t)(-(int64_t)a);
                s = -s;
            } else {
                ua = (uint32_t)a;
            }
            const int32_t *Br = B + (size_t)k * n;
            while (ua) {
                int b = __builtin_ctz(ua);
                ua &= ua - 1u;          /* clear lowest set bit */
                if (s > 0) {
                    for (int j = 0; j < n; j++)
                        Cr[j] += (int64_t)Br[j] << b;
                } else {
                    for (int j = 0; j < n; j++)
                        Cr[j] -= (int64_t)Br[j] << b;
                }
            }
        }
    }
}


/* ------------------------------------------------------------------ *
 * B4 — contraction index as convolution.
 *
 *   C_ij = sum_k A_ik B_kj
 *        = sum_k A_ik D_{(n-1-k)j}    where D_{tj} := B_{(n-1-t)j}
 *
 * So each entry is the middle coefficient of the 1D convolution of row i
 * of A with the reversed column j of B.  That is the same *shape* of sum
 * integer Karatsuba / SS accelerate — but here we need n² such middle
 * coefficients.  Full length-n polynomial products per (i,j) cost more
 * than a dot product unless further structure is shared (future work).
 *
 * KERNEL_CONVK:    schoolbook matmul (= ikj). The reversed-B form is an
 *                  identity, not a different algorithm; O(n³).
 * KERNEL_CONVKARA: full length-n Karatsuba polynomial product per (i,j),
 *                  then keep the middle coeff. Algebraically SS-style, but
 *                  a single middle coeff is only a dot product — computing
 *                  the other 2n-2 coeffs is wasted work. O(n² n^{log2 3}).
 *
 * Negative result (B4 phase 2): per-(i,j) convolution does not beat ikj
 * or matrix-Strassen. Beating them needs structure *shared* across many
 * (i,j) (classical matmul bilinear algorithms), not n² independent 1D muls.
 * ------------------------------------------------------------------ */

static void conv1d_school(int64_t *out, const int32_t *a, const int32_t *b,
                          int n)
{
    int P = 2 * n - 1;
    memset(out, 0, (size_t)P * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        int32_t ai = a[i];
        if (!ai) continue;
        for (int j = 0; j < n; j++)
            out[i + j] += (int64_t)ai * b[j];
    }
}

/* Karatsuba polynomial product, length-n coeffs -> length 2n-1. */
static void conv1d_kara(int64_t *out, const int32_t *a, const int32_t *b,
                        int n)
{
    if (n <= 16) {
        conv1d_school(out, a, b, n);
        return;
    }
    int h = n / 2;
    int n0 = h, n1 = n - h;          /* low size n0, high size n1 */
    int ph = 2 * h - 1;
    /* For odd n, pad high to h with zeros by using temp buffers of size h. */
    int hs = (n + 1) / 2;            /* ceil(n/2) */
    int64_t *z0 = NULL, *z1 = NULL, *z2 = NULL;
    int32_t *a0 = NULL, *a1 = NULL, *b0 = NULL, *b1 = NULL, *as = NULL, *bs = NULL;
    int pfull = 2 * n - 1;

    if (n & 1) {
        /* pad to even length n+1 */
        int np = n + 1;
        int32_t *ap = calloc((size_t)np, sizeof(int32_t));
        int32_t *bp = calloc((size_t)np, sizeof(int32_t));
        int64_t *op = calloc((size_t)(2 * np - 1), sizeof(int64_t));
        if (!ap || !bp || !op) {
            free(ap); free(bp); free(op);
            conv1d_school(out, a, b, n);
            return;
        }
        memcpy(ap, a, (size_t)n * sizeof(int32_t));
        memcpy(bp, b, (size_t)n * sizeof(int32_t));
        conv1d_kara(op, ap, bp, np);
        memcpy(out, op, (size_t)pfull * sizeof(int64_t));
        free(ap); free(bp); free(op);
        return;
    }

    size_t psz = (size_t)ph * sizeof(int64_t);
    z0 = malloc(psz);
    z1 = malloc(psz);
    z2 = malloc(psz);
    as = malloc((size_t)h * sizeof(int32_t));
    bs = malloc((size_t)h * sizeof(int32_t));
    if (!z0 || !z1 || !z2 || !as || !bs) {
        free(z0); free(z1); free(z2); free(as); free(bs);
        conv1d_school(out, a, b, n);
        return;
    }
    a0 = (int32_t *)a;
    a1 = (int32_t *)a + h;
    b0 = (int32_t *)b;
    b1 = (int32_t *)b + h;
    for (int i = 0; i < h; i++) {
        as[i] = a0[i] + a1[i];
        bs[i] = b0[i] + b1[i];
    }
    conv1d_kara(z0, a0, b0, h);
    conv1d_kara(z2, a1, b1, h);
    conv1d_kara(z1, as, bs, h);

    memset(out, 0, (size_t)pfull * sizeof(int64_t));
    for (int i = 0; i < ph; i++) out[i] += z0[i];
    for (int i = 0; i < ph; i++) out[2 * h + i] += z2[i];
    for (int i = 0; i < ph; i++) out[h + i] += z1[i] - z0[i] - z2[i];

    free(z0); free(z1); free(z2); free(as); free(bs);
    (void)n0; (void)n1; (void)hs;
}

static void mm_convk(int64_t *C, const int32_t *A, const int32_t *B,
                     int n, int sign)
{
    /* Algebra: with D_tj = B_(n-1-t)j,
     *   sum_k A_ik D_(n-1-k)j = sum_k A_ik B_kj
     * so the reversed-index form is exactly the schoolbook matmul (ikj).
     * No temporary D — same work as KERNEL_IKJ; kept as a named leaf so
     * tables show the B4 identity without a copy tax. */
    for (int i = 0; i < n; i++) {
        int64_t *Cr = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            int64_t a = (int64_t)sign * A[(size_t)i * n + k];
            if (!a) continue;
            const int32_t *Br = B + (size_t)k * n;
            for (int j = 0; j < n; j++)
                Cr[j] += a * Br[j];
        }
    }
}

static void mm_convkara(int64_t *C, const int32_t *A, const int32_t *B,
                        int n, int sign)
{
    int32_t *row = malloc((size_t)n * sizeof(int32_t));
    int32_t *rev = malloc((size_t)n * sizeof(int32_t));
    int64_t *conv = malloc((size_t)(2 * n - 1) * sizeof(int64_t));
    if (!row || !rev || !conv) {
        free(row); free(rev); free(conv);
        mm_convk(C, A, B, n, sign);
        return;
    }
    for (int i = 0; i < n; i++) {
        memcpy(row, A + (size_t)i * n, (size_t)n * sizeof(int32_t));
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++)
                rev[n - 1 - k] = B[(size_t)k * n + j];
            conv1d_kara(conv, row, rev, n);
            C[(size_t)i * n + j] += (int64_t)sign * conv[n - 1];
        }
    }
    free(row); free(rev); free(conv);
}


/* ------------------------------------------------------------------ *
 * B5 — bit-packed Boolean / multi-bit plane GEMM (AND + popcount).
 *
 * 0-1 case:
 *   pack row i of A along k into uint64 words;
 *   pack col j of B along k similarly;
 *   C_ij += sign * popcount(Arow_i AND Bcol_j).
 *
 * Multi-bit: A = sum_b 2^b A^(b), B = sum_c 2^c B^(c),
 *   AB = sum_{b,c} 2^{b+c} (A^(b) B^(c)) with each factor Boolean.
 * ------------------------------------------------------------------ */

static inline int bitpack_nwords(int n)
{
    return (n + 63) >> 6;
}

/* Pack bit `bit` of each entry along k.
 * If as_cols==0: out[i*nw + w] from row i of M (M[i,k]).
 * If as_cols==1: out[j*nw + w] from column j of M (M[k,j]).
 * Returns 1 if any bit was set. */
static int pack_bitplane(uint64_t *out, const int32_t *M, int n, int bit,
                         int as_cols)
{
    int nw = bitpack_nwords(n);
    memset(out, 0, (size_t)n * nw * sizeof(uint64_t));
    int any = 0;
    uint32_t mask = 1u << bit;
    if (!as_cols) {
        for (int i = 0; i < n; i++) {
            uint64_t *row = out + (size_t)i * nw;
            for (int k = 0; k < n; k++) {
                int32_t v = M[(size_t)i * n + k];
                uint32_t u = (v < 0) ? (uint32_t)(-(int64_t)v) : (uint32_t)v;
                if (u & mask) {
                    row[k >> 6] |= (uint64_t)1 << (k & 63);
                    any = 1;
                }
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            uint64_t *row = out + (size_t)j * nw;
            for (int k = 0; k < n; k++) {
                int32_t v = M[(size_t)k * n + j];
                uint32_t u = (v < 0) ? (uint32_t)(-(int64_t)v) : (uint32_t)v;
                if (u & mask) {
                    row[k >> 6] |= (uint64_t)1 << (k & 63);
                    any = 1;
                }
            }
        }
    }
    return any;
}

/* Boolean GEMM: C_ij += scale * popcount(Arow_i & Brow_j). A/B packed nw words. */
static void bool_gemm_accum(int64_t *C, const uint64_t *Apack,
                            const uint64_t *Bpack, int n, int64_t scale)
{
    int nw = bitpack_nwords(n);
    if (!scale) return;
    for (int i = 0; i < n; i++) {
        const uint64_t *ar = Apack + (size_t)i * nw;
        int64_t *Cr = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const uint64_t *br = Bpack + (size_t)j * nw;
            unsigned acc = 0;
            for (int w = 0; w < nw; w++)
                acc += (unsigned)__builtin_popcountll(ar[w] & br[w]);
            if (acc)
                Cr[j] += scale * (int64_t)acc;
        }
    }
}

static int matrix_is_01(const int32_t *M, int n)
{
    size_t nn = (size_t)n * n;
    for (size_t i = 0; i < nn; i++) {
        int32_t v = M[i];
        if (v != 0 && v != 1) return 0;
    }
    return 1;
}

/* Strict 0-1 planes: pack bits along k, AND+popcount. Otherwise → ikj. */
static void mm_boolpack(int64_t *C, const int32_t *A, const int32_t *B,
                        int n, int sign)
{
    if (!matrix_is_01(A, n) || !matrix_is_01(B, n)) {
        mm_ikj(C, A, B, n, sign);
        return;
    }
    int nw = bitpack_nwords(n);
    uint64_t *Ap = malloc((size_t)n * nw * sizeof(uint64_t));
    uint64_t *Bp = malloc((size_t)n * nw * sizeof(uint64_t));
    if (!Ap || !Bp) {
        free(Ap); free(Bp);
        mm_ikj(C, A, B, n, sign);
        return;
    }
    pack_bitplane(Ap, A, n, 0, 0);
    pack_bitplane(Bp, B, n, 0, 1);
    bool_gemm_accum(C, Ap, Bp, n, sign);
    free(Ap); free(Bp);
}

/* General small integers via all bit-plane pairs (magnitude), then sign fix
 * is absorbed per-entry in packing (magnitude bits only) — products of
 * magnitudes are unsigned; apply overall sign of A_ik*B_kj via...
 *
 * Careful: A_ik may be negative. Bit planes of |A| and |B| give |A||B|
 * contribution layout, but sign(A_ik)*sign(B_kj) varies per k, so we cannot
 * factor a global sign per plane pair.
 *
 * Correct approach for signed values: split into four 0-1-ish contributions
 * is hard. Practical approach used here:
 *   - Treat bit planes of the raw two's-complement pattern only for non-neg
 *     path: if any negative entry exists, fall back to mm_bitplane/ikj.
 *   - For non-negative A,B: plane pairs as above.
 */
static int matrix_nonneg(const int32_t *M, int n)
{
    size_t nn = (size_t)n * n;
    for (size_t i = 0; i < nn; i++)
        if (M[i] < 0) return 0;
    return 1;
}

static int max_bit_needed(const int32_t *M, int n)
{
    uint32_t mx = 0;
    size_t nn = (size_t)n * n;
    for (size_t i = 0; i < nn; i++) {
        uint32_t u = (uint32_t)M[i];
        if (u > mx) mx = u;
    }
    if (!mx) return -1;
    return 31 - __builtin_clz(mx);
}

static void mm_bitpack(int64_t *C, const int32_t *A, const int32_t *B,
                       int n, int sign)
{
    if (!matrix_nonneg(A, n) || !matrix_nonneg(B, n)) {
        /* Signed entries: per-k sign breaks plane factorisation; use bitplane. */
        mm_bitplane(C, A, B, n, sign);
        return;
    }
    int ba = max_bit_needed(A, n);
    int bb = max_bit_needed(B, n);
    if (ba < 0 || bb < 0) return;

    int nw = bitpack_nwords(n);
    uint64_t *Ap = malloc((size_t)n * nw * sizeof(uint64_t));
    uint64_t *Bp = malloc((size_t)n * nw * sizeof(uint64_t));
    if (!Ap || !Bp) {
        free(Ap); free(Bp);
        mm_ikj(C, A, B, n, sign);
        return;
    }

    for (int b = 0; b <= ba; b++) {
        if (!pack_bitplane(Ap, A, n, b, 0))
            continue;
        for (int c = 0; c <= bb; c++) {
            if (!pack_bitplane(Bp, B, n, c, 1))
                continue;
            int64_t scale = (int64_t)sign << (b + c);
            bool_gemm_accum(C, Ap, Bp, n, scale);
        }
    }
    free(Ap); free(Bp);
}

/* ------------------------------------------------------------------ */
void mm_accum(int64_t *C, const int32_t *A, const int32_t *B,
              int n, int sign, kernel_t k)
{
    g_kernel_calls++;
    switch (k) {
    case KERNEL_BLOCKED:  mm_blocked (C, A, B, n, sign); break;
    case KERNEL_PACKED:   mm_packed  (C, A, B, n, sign); break;
    case KERNEL_STRASSEN: mm_fast_rec(C, A, B, n, sign, st_mul); break;
    case KERNEL_WINOGRAD: mm_fast_rec(C, A, B, n, sign, sw_mul); break;
    case KERNEL_BITPLANE: mm_bitplane(C, A, B, n, sign); break;
    case KERNEL_CONVK:    mm_convk   (C, A, B, n, sign); break;
    case KERNEL_CONVKARA: mm_convkara(C, A, B, n, sign); break;
    case KERNEL_BOOLPACK: mm_boolpack(C, A, B, n, sign); break;
    case KERNEL_BITPACK:  mm_bitpack (C, A, B, n, sign); break;
    default:              mm_ikj     (C, A, B, n, sign); break;
    }
}
