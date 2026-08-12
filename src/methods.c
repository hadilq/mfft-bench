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

    for (int u = 0; u < L; u++)
        for (int v = 0; v < L; v++)
            mm_accum(C + (size_t)(u + v) * nn,
                     A32 + (size_t)u * nn,
                     B32 + (size_t)v * nn, n, +1, kern);

    normalize_planes(C, P, n, out, RL);
    free(A32); free(B32); free(C);
}
