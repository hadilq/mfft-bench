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
long long g_strassen_cutoff = 64;

static const char *knames[KERNEL__COUNT] = { "ikj", "blocked", "strassen" };

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
 * Strassen over int64.  Included because the post explicitly notes that
 * MFFT is "orthogonal to the other methods" -- here it can be stacked
 * underneath MFFT by picking --kernel strassen.
 * ------------------------------------------------------------------ */
static void st_base(int64_t *C, const int64_t *A, const int64_t *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        int64_t *Cr = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            int64_t a = A[(size_t)i * n + k];
            if (!a) continue;
            const int64_t *Br = B + (size_t)k * n;
            for (int j = 0; j < n; j++) Cr[j] += a * Br[j];
        }
    }
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

static void mm_strassen(int64_t *C, const int32_t *A, const int32_t *B,
                        int n, int sign)
{
    size_t nn = (size_t)n * n;
    int64_t *A64 = malloc(nn * sizeof(int64_t));
    int64_t *B64 = malloc(nn * sizeof(int64_t));
    int64_t *T   = malloc(nn * sizeof(int64_t));
    if (!A64 || !B64 || !T) { free(A64); free(B64); free(T);
                              mm_blocked(C, A, B, n, sign); return; }
    for (size_t i = 0; i < nn; i++) A64[i] = A[i];
    for (size_t i = 0; i < nn; i++) B64[i] = B[i];
    st_mul(T, A64, B64, n);
    if (sign > 0) for (size_t i = 0; i < nn; i++) C[i] += T[i];
    else          for (size_t i = 0; i < nn; i++) C[i] -= T[i];
    free(A64); free(B64); free(T);
}

/* ------------------------------------------------------------------ */
void mm_accum(int64_t *C, const int32_t *A, const int32_t *B,
              int n, int sign, kernel_t k)
{
    g_kernel_calls++;
    switch (k) {
    case KERNEL_BLOCKED:  mm_blocked (C, A, B, n, sign); break;
    case KERNEL_STRASSEN: mm_strassen(C, A, B, n, sign); break;
    default:              mm_ikj     (C, A, B, n, sign); break;
    }
}
