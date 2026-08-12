/* mfftbench.h -- shared declarations for the MFFT matmul benchmark.
 *
 * Reference: Hadi Lashkari Ghouchani, "Matrix Fast Fourier transform (MFFT)",
 * https://hadilq.com/posts/matrix-fast-fourier-transform/ (2024)
 */
#ifndef MFFTBENCH_H
#define MFFTBENCH_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ *
 * Number representation
 *
 * Matrix entries are unsigned integers of L*LIMB_BITS bits, stored as L
 * limbs in base 2^LIMB_BITS.  The post works in base 2 ("digits"); using a
 * wider limb is the same construction with a bigger radix and is strictly
 * better in practice.  Set LIMB_BITS to 1 and you get the post verbatim.
 * ------------------------------------------------------------------ */
#ifndef LIMB_BITS
#define LIMB_BITS 16
#endif
#if LIMB_BITS < 1 || LIMB_BITS > 16
#error "LIMB_BITS must be between 1 and 16"
#endif
#define LIMB_MASK ((uint32_t)((1u << LIMB_BITS) - 1u))
#define LIMB_MAX  LIMB_MASK

/* ------------------------------------------------------------------ *
 * Inner kernel: the small-integer n x n matrix product that every
 * plane/transform based method ends up calling many times.
 * ------------------------------------------------------------------ */
typedef enum {
    KERNEL_IKJ = 0,
    KERNEL_BLOCKED,
    KERNEL_PACKED,
    KERNEL_STRASSEN,
    KERNEL_WINOGRAD,
    KERNEL__COUNT
} kernel_t;

const char *kernel_name(kernel_t k);
int         kernel_from_name(const char *s);

/* C += sign * (A*B).  C is int64 n*n, A and B are int32 n*n, row major. */
void mm_accum(int64_t *C, const int32_t *A, const int32_t *B,
              int n, int sign, kernel_t k);

extern long long g_kernel_calls;   /* number of mm_accum() invocations */
extern long long g_strassen_cutoff;

/* ------------------------------------------------------------------ *
 * Big-integer matrices
 *
 *   plane-major : p[u*n*n + i*n + j]  == limb u of entry (i,j)
 *   entry-major : p[(i*n + j)*L + u]  == limb u of entry (i,j)
 *
 * Plane-major *is* the "matrix of digits" decomposition A_u from the post,
 * so MFFT and the plane methods get it for free.
 * ------------------------------------------------------------------ */
uint16_t *bigmat_alloc(int n, int L);
void      bigmat_rand(uint16_t *m, int n, int L, uint64_t seed);
uint16_t *bigmat_to_entry_major(const uint16_t *plane, int n, int L);

/* carry-propagate a plane-major int64 accumulator into RL uint16 limbs */
void normalize_planes(const int64_t *acc, int nplanes, int n,
                      uint16_t *out, int RL);
int  bigres_equal(const uint16_t *a, const uint16_t *b, int n, int RL,
                  int *bad_i, int *bad_j, int *bad_w);

/* ------------------------------------------------------------------ *
 * Methods under test
 * ------------------------------------------------------------------ */
void mm_bigint_ijk(const uint16_t *Aem, const uint16_t *Bem,
                   int n, int L, uint16_t *out, int RL);
void mm_bigint_ikj(const uint16_t *Aem, const uint16_t *Bem,
                   int n, int L, uint16_t *out, int RL);
void mm_limbplane (const uint16_t *Apl, const uint16_t *Bpl,
                   int n, int L, kernel_t k, uint16_t *out, int RL);

/* ------------------------------------------------------------------ *
 * MFFT plan
 *
 *   S  limbs per block          (block = one coefficient of the outer poly)
 *   NB outer transform length   = 2L/S  (half the blocks are zero padding)
 *   K  dimension of the root-of-unity ring R = Z[y]/(y^K + 1), K = 2S,
 *      i.e. y is represented by the post's I_s with 2^s = K
 *   g  omega = y^g is the primitive NB-th root of unity, g = 2K/NB
 *
 * Number of n x n kernel products = NB*K*K.
 * ------------------------------------------------------------------ */
typedef struct { int L, S, NB, K, g; } mfft_plan;

int    mfft_plan_init(mfft_plan *p, int L, int sigma_override);
long long mfft_plan_products(const mfft_plan *p);
double mfft_plan_maxbits(const mfft_plan *p, int n);  /* worst-case log2|x| */
void   mfft_plan_describe(const mfft_plan *p, int n);

void mm_mfft(const uint16_t *Apl, const uint16_t *Bpl, int n, int L,
             const mfft_plan *p, kernel_t k, uint16_t *out, int RL);

/* ------------------------------------------------------------------ *
 * Roots of unity: the post's H_{s,k} recursion
 * ------------------------------------------------------------------ */
int *roots_build_H(int s);          /* [(1<<(s+1)) * (1<<s)] signed 1-based */
int  roots_selftest(int max_s, int verbose);

/* ------------------------------------------------------------------ */
double now_sec(void);

/* ------------------------------------------------------------------ *
 * Machine-learning track: inexact GEMM at ML precisions.  Reports
 * throughput *and* accuracy, because in ML the two trade against each
 * other and speed alone is not a fair ranking.
 * ------------------------------------------------------------------ */
int ml_run(int n, int reps, int csv, int with_naive);

#endif /* MFFTBENCH_H */
