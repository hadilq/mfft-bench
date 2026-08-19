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
    KERNEL_BITPLANE,   /* binary expansion of A: set-bit row-adds of B */
    KERNEL_CONVK,      /* B4: C_ij via reversed-B convolution identity */
    KERNEL_CONVKARA,   /* B4: same, 1D Karatsuba conv per (i,j), middle coeff */
    KERNEL_BOOLPACK,   /* B5: 0-1 planes, AND+popcount along k */
    KERNEL_BITPACK,    /* B5: multi-bit via Boolean GEMMs of bit-planes */
    KERNEL__COUNT
} kernel_t;

const char *kernel_name(kernel_t k);
int         kernel_from_name(const char *s);

/* C += sign * (A*B).  C is int64 n*n, A and B are int32 n*n, row major. */
void mm_accum(int64_t *C, const int32_t *A, const int32_t *B,
              int n, int sign, kernel_t k);

extern long long g_kernel_calls;   /* number of mm_accum() invocations */

/* B1: optional phase timers for MFFT (enabled by --profile). */
typedef struct {
    double pack;        /* limb -> coefficient packing */
    double build_ops;   /* fused op-list construction */
    double transform;   /* forward + inverse FFT */
    double pointwise;   /* leaf mm_accum / recursive SSA */
    double fold;        /* /NB + scatter back to limb planes */
    long long n_build;  /* build_ops invocations */
    long long n_fft;    /* fft_run32/64 invocations */
} mfft_profile_t;
extern int g_mfft_profile;
extern mfft_profile_t g_mfft_prof;
void mfft_profile_reset(void);
void mfft_profile_print(const char *label);
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

/* Convolution cores on signed int32 limb planes, writing 2L-1 int64 planes.
 * These are what the fp32 path in src/fpfixed.c drives. */
void conv_limbplane(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                    int n, int L, kernel_t k);
void mm_karatsuba (const uint16_t *Apl, const uint16_t *Bpl,
                   int n, int L, kernel_t k, uint16_t *out, int RL);
void conv_karatsuba(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                    int n, int L, kernel_t k);
long long karatsuba_products(int L);
void conv_toom3(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                int n, int L, kernel_t k);
long long toom3_products(int L);
void conv_evenodd(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                  int n, int L, kernel_t k);
long long evenodd_products(int L);
void conv_hybrid(int64_t *Cw, const int32_t *A32, const int32_t *B32,
                 int n, int L, kernel_t k);
long long hybrid_products(int L);
void mm_toom3   (const uint16_t *Apl, const uint16_t *Bpl,
                int n, int L, kernel_t k, uint16_t *out, int RL);
void mm_evenodd (const uint16_t *Apl, const uint16_t *Bpl,
                int n, int L, kernel_t k, uint16_t *out, int RL);
void mm_hybrid  (const uint16_t *Apl, const uint16_t *Bpl,
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
typedef struct {
    int L, S, NB, K, g;
    int rec;            /* 1 = pointwise step is a recursive SSA convolution */
    long long nprod;    /* n x n products the plan will actually issue       */
} mfft_plan;

int    mfft_plan_init(mfft_plan *p, int L, int sigma_override);
/* Recursive plan: chooses the block size by minimising the *total* product
 * count, with the pointwise negacyclic convolutions solved recursively by
 * the same construction (Schoenhage-Strassen) rather than schoolbook. */
int    mfft_plan_init_rec(mfft_plan *p, int L, int n, int sigma_override);
void   ssa_negconv(int64_t *C, const int32_t *A, const int32_t *B,
                   int K, int n, int Mbits, kernel_t kern);
long long mfft_plan_products(const mfft_plan *p);
double mfft_plan_maxbits(const mfft_plan *p, int n);  /* worst-case log2|x| */
void   mfft_plan_describe(const mfft_plan *p, int n);

void mm_mfft(const uint16_t *Apl, const uint16_t *Bpl, int n, int L,
             const mfft_plan *p, kernel_t k, uint16_t *out, int RL);
void conv_mfft(int64_t *Cw, const int32_t *A32, const int32_t *B32,
               int n, int L, const mfft_plan *p, kernel_t k);

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
int ml_run(int n, int reps, int csv, int with_naive, int fp_width, int illcond,
           int fp64_mode, int data_narrow);

/* ------------------------------------------------------------------ *
 * fp32 -> fixed-point integer embedding, so the exact integer methods
 * (including MFFT) apply to float matrices.  See src/fpfixed.c.
 * ------------------------------------------------------------------ */
typedef struct {
    int n, L, sig;
    int SA, SB;             /* A is encoded at scale 2^SA, B at 2^SB     */
    int spreadA, spreadB;   /* exponent spread that set the limb count   */
    int32_t *A32, *B32;     /* signed limb planes, L*n*n each            */
    int64_t *Cw;            /* exact product, 2L-1 int64 planes          */
    double enc_secs, dec_secs;
} fpx_ctx;

int  fpx_init(fpx_ctx *c, const float *A, const float *B, int n, int force_bits);
void fpx_encode(fpx_ctx *c, const float *A, const float *B);
int  fpx_init_d(fpx_ctx *c, const double *A, const double *B, int n,
                int force_bits);
void fpx_encode_d(fpx_ctx *c, const double *A, const double *B);
void fpx_decode_ld(fpx_ctx *c, long double *C);
void fpx_decode_f32(fpx_ctx *c, float *C);
void fpx_decode_f64(fpx_ctx *c, double *C);
void fpx_free(fpx_ctx *c);

/* ------------------------------------------------------------------ *
 * Low-precision exact GEMM (src/lowprec.c).
 *
 * At bf16 and below a single integer GEMM is already exact -- see the
 * header comment in lowprec.c for the width argument.  Each precision has
 * its own packed kernel; the shapes differ because the accumulator widths
 * do, and sharing one would halve the lanes of the narrower path.
 * ------------------------------------------------------------------ */
typedef struct {
    int n, vbits, needbits;
    int SA, SB;             /* bf16: shared fixed-point scales           */
    float *sa, *sb;         /* int8/int4: per-channel scales             */
    float *Ar, *Br;         /* inputs rounded to the target format       */
    int16_t *A16, *B16;     /* int8/int4 operands                        */
    int32_t *A32, *B32;     /* bf16 operands                             */
    int32_t *C32;           /* int8/int4 exact accumulator               */
    int64_t *C64;           /* bf16 exact accumulator                    */
} lp_ctx;

float lp_to_bf16(float x);
void  lp_gemm_i16_i32(int32_t *C, const int16_t *A, const int16_t *B, int n);
void  lp_gemm_i32_i64(int64_t *C, const int32_t *A, const int32_t *B, int n);

int   lp_bf16_init(lp_ctx *c, const float *A, const float *B, int n);
void  lp_bf16_gemm(lp_ctx *c);
void  lp_bf16_decode(const lp_ctx *c, float *C);

int   lp_intq_init(lp_ctx *c, const float *A, const float *B, int n, int bits);
void  lp_intq_gemm(lp_ctx *c);
void  lp_intq_decode(const lp_ctx *c, float *C);
void  lp_free(lp_ctx *c);

#endif /* MFFTBENCH_H */
