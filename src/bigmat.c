/* bigmat.c -- big-integer matrix storage and helpers. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mfftbench.h"

double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

uint16_t *bigmat_alloc(int n, int L)
{
    return calloc((size_t)n * n * L, sizeof(uint16_t));
}

/* splitmix64 -- deterministic, so every method sees identical inputs */
static uint64_t sm64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void bigmat_rand(uint16_t *m, int n, int L, uint64_t seed)
{
    uint64_t s = seed;
    size_t total = (size_t)n * n * L;
    for (size_t i = 0; i < total; i++)
        m[i] = (uint16_t)(sm64(&s) & LIMB_MASK);
}

uint16_t *bigmat_to_entry_major(const uint16_t *plane, int n, int L)
{
    uint16_t *e = bigmat_alloc(n, L);
    if (!e) return NULL;
    size_t nn = (size_t)n * n;
    for (int u = 0; u < L; u++)
        for (size_t idx = 0; idx < nn; idx++)
            e[idx * L + u] = plane[(size_t)u * nn + idx];
    return e;
}

void normalize_planes(const int64_t *acc, int nplanes, int n,
                      uint16_t *out, int RL)
{
    size_t nn = (size_t)n * n;
    for (size_t idx = 0; idx < nn; idx++) {
        int64_t carry = 0;
        for (int w = 0; w < RL; w++) {
            int64_t v = (w < nplanes ? acc[(size_t)w * nn + idx] : 0) + carry;
            out[(size_t)w * nn + idx] = (uint16_t)(v & LIMB_MASK);
            carry = v >> LIMB_BITS;
        }
    }
}

int bigres_equal(const uint16_t *a, const uint16_t *b, int n, int RL,
                 int *bad_i, int *bad_j, int *bad_w)
{
    size_t nn = (size_t)n * n;
    for (int w = 0; w < RL; w++)
        for (size_t idx = 0; idx < nn; idx++)
            if (a[(size_t)w * nn + idx] != b[(size_t)w * nn + idx]) {
                if (bad_w) *bad_w = w;
                if (bad_i) *bad_i = (int)(idx / n);
                if (bad_j) *bad_j = (int)(idx % n);
                return 0;
            }
    return 1;
}
