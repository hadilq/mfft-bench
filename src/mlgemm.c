/* mlgemm.c -- the machine-learning side of the benchmark.
 *
 * ML matmul differs from the exact big-integer problem in one decisive
 * way: it is allowed to be wrong.  Every production speedup -- fp32 ->
 * bf16 -> int8 -> int4 -- buys throughput by spending accuracy, so timing
 * alone is not a fair ranking.  Each method here reports throughput *and*
 * the relative error against a float64 reference.
 *
 * Methods:
 *   sgemm-ijk       textbook triple loop, the "ordinary method"
 *   sgemm-ikj       same flops, cache-friendly loop order
 *   sgemm-blocked   cache tiling
 *   sgemm-packed    packed panels + register micro-kernel (what OpenBLAS,
 *                   BLIS and oneDNN actually do; the practical state of
 *                   the art on CPU)
 *   sgemm-strassen  Strassen recursion bottoming out in sgemm-packed
 *   bf16-packed     inputs rounded to bfloat16, fp32 accumulate -- the
 *                   default training precision on modern accelerators
 *   int8-packed     per-channel symmetric quantization, int32 accumulate,
 *                   dequantize -- the default inference path
 *   blas-sgemm      OpenBLAS reference (build with WITH_BLAS=1)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mfftbench.h"

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

#define FMC 192
#define FKC 256
#define FNC 512

/* Register-tile shape and SIMD width.
 *
 * Autovectorisation gives up on the 2-D accumulator here (gcc reports
 * "complicated access pattern"), so the micro-kernel is written with
 * explicit GCC/Clang vector types.  NR is two vectors wide and MR is
 * chosen so MR*2 accumulators fit the register file without spilling --
 * that ratio is the whole game in a packed GEMM. */
#if defined(__AVX512F__)
#  define VBYTES 64
#elif defined(__AVX2__) || defined(__ARM_NEON) || defined(__aarch64__)
#  define VBYTES 32
#else
#  define VBYTES 16
#endif

typedef float   vf  __attribute__((vector_size(VBYTES)));
typedef float   vfu __attribute__((vector_size(VBYTES), aligned(4)));
typedef int32_t vi  __attribute__((vector_size(VBYTES)));
typedef int32_t viu __attribute__((vector_size(VBYTES), aligned(4)));
typedef int16_t vsu __attribute__((vector_size(VBYTES/2), aligned(2)));

#define VL  ((int)(sizeof(vf) / sizeof(float)))
#ifndef FMR
#  define FMR 6
#endif
#define FNR (2 * VL)

#define FMIN(a,b) ((a) < (b) ? (a) : (b))

/* ------------------------------------------------------------------ *
 * fp32 packed GEMM
 * ------------------------------------------------------------------ */
static void fpack_a(float *Ap, const float *A, int lda,
                    int ic, int mc, int pc, int kc)
{
    for (int pi = 0; pi * FMR < mc; pi++) {
        float *d = Ap + (size_t)pi * FMR * kc;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < FMR; i++)
                d[p * FMR + i] = (pi * FMR + i < mc)
                    ? A[(size_t)(ic + pi * FMR + i) * lda + pc + p] : 0.0f;
    }
}

static void fpack_b(float *Bp, const float *B, int ldb,
                    int jc, int nc, int pc, int kc)
{
    for (int pj = 0; pj * FNR < nc; pj++) {
        float *d = Bp + (size_t)pj * FNR * kc;
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < FNR; j++)
                d[p * FNR + j] = (pj * FNR + j < nc)
                    ? B[(size_t)(pc + p) * ldb + jc + pj * FNR + j] : 0.0f;
    }
}

static void fmicro(const float *restrict a, const float *restrict b,
                   float *c, int ldc, int kc, int mr, int nr)
{
    vf acc[FMR][2];
    for (int i = 0; i < FMR; i++) { acc[i][0] = (vf){0}; acc[i][1] = (vf){0}; }

    for (int p = 0; p < kc; p++) {
        const float *bp = b + (size_t)p * FNR;
        vf b0 = *(const vfu *)bp;
        vf b1 = *(const vfu *)(bp + VL);
        const float *ap = a + (size_t)p * FMR;
        for (int i = 0; i < FMR; i++) {
            vf av = (vf){0} + ap[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }

    float tmp[FMR][FNR];
    for (int i = 0; i < FMR; i++) {
        *(vfu *)&tmp[i][0]  = acc[i][0];
        *(vfu *)&tmp[i][VL] = acc[i][1];
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++) c[(size_t)i * ldc + j] += tmp[i][j];
}

/* C = A*B, all row-major with the given leading dimensions */
static void sgemm_packed_ld(float *C, const float *A, const float *B,
                            int m, int nn, int k,
                            int lda, int ldb, int ldc)
{
    for (int i = 0; i < m; i++)
        memset(C + (size_t)i * ldc, 0, (size_t)nn * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int jc = 0; jc < nn; jc += FNC) {
        int nc = FMIN(FNC, nn - jc);
        float *Ap = malloc((size_t)(FMC / FMR + 1) * FMR * FKC * sizeof(float));
        float *Bp = malloc((size_t)(FNC / FNR + 1) * FNR * FKC * sizeof(float));
        if (!Ap || !Bp) { free(Ap); free(Bp); continue; }
        for (int pc = 0; pc < k; pc += FKC) {
            int kc = FMIN(FKC, k - pc);
            fpack_b(Bp, B, ldb, jc, nc, pc, kc);
            for (int ic = 0; ic < m; ic += FMC) {
                int mc = FMIN(FMC, m - ic);
                fpack_a(Ap, A, lda, ic, mc, pc, kc);
                for (int j = 0; j < nc; j += FNR) {
                    int nr = FMIN(FNR, nc - j);
                    for (int i = 0; i < mc; i += FMR) {
                        int mr = FMIN(FMR, mc - i);
                        fmicro(Ap + (size_t)(i / FMR) * FMR * kc,
                               Bp + (size_t)(j / FNR) * FNR * kc,
                               C + (size_t)(ic + i) * ldc + jc + j,
                               ldc, kc, mr, nr);
                    }
                }
            }
        }
        free(Ap); free(Bp);
    }
}

static void sgemm_packed(float *C, const float *A, const float *B, int n)
{ sgemm_packed_ld(C, A, B, n, n, n, n, n, n); }

/* ------------------------------------------------------------------ */
static void sgemm_ijk(float *C, const float *A, const float *B, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += A[(size_t)i*n+k] * B[(size_t)k*n+j];
            C[(size_t)i*n+j] = s;
        }
}

static void sgemm_ikj(float *C, const float *A, const float *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = A[(size_t)i*n+k];
            if (a == 0.0f) continue;
            const float *Br = B + (size_t)k*n;
            float *Cr = C + (size_t)i*n;
            for (int j = 0; j < n; j++) Cr[j] += a * Br[j];
        }
}

#define FBS 64
static void sgemm_blocked(float *C, const float *A, const float *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int ii = 0; ii < n; ii += FBS) {
        int imax = FMIN(ii + FBS, n);
        for (int kk = 0; kk < n; kk += FBS) {
            int kmax = FMIN(kk + FBS, n);
            for (int jj = 0; jj < n; jj += FBS) {
                int jmax = FMIN(jj + FBS, n);
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        float a = A[(size_t)i*n+k];
                        const float *Br = B + (size_t)k*n;
                        float *Cr = C + (size_t)i*n;
                        for (int j = jj; j < jmax; j++) Cr[j] += a * Br[j];
                    }
            }
        }
    }
}

/* ------------------------------------------------------------------ *
 * Strassen on top of the packed kernel.
 * ------------------------------------------------------------------ */


static void fq_get(float *q, const float *M, int n, int h, int r, int c)
{
    for (int i = 0; i < h; i++)
        memcpy(q + (size_t)i*h, M + (size_t)(r*h+i)*n + (size_t)c*h,
               (size_t)h * sizeof(float));
}
static void fq_put(float *M, const float *q, int n, int h, int r, int c)
{
    for (int i = 0; i < h; i++)
        memcpy(M + (size_t)(r*h+i)*n + (size_t)c*h, q + (size_t)i*h,
               (size_t)h * sizeof(float));
}
static void fadd(float *d, const float *a, const float *b, size_t n)
{ for (size_t i = 0; i < n; i++) d[i] = a[i] + b[i]; }
static void fsub(float *d, const float *a, const float *b, size_t n)
{ for (size_t i = 0; i < n; i++) d[i] = a[i] - b[i]; }

static void sgemm_strassen(float *C, const float *A, const float *B, int n)
{
    if (n <= (int)g_strassen_cutoff || (n & 1)) { sgemm_packed(C, A, B, n); return; }
    int h = n / 2;
    size_t hs = (size_t)h * h;
    float *buf = malloc(hs * sizeof(float) * 18);
    if (!buf) { sgemm_packed(C, A, B, n); return; }

    float *a11=buf,*a12=buf+hs,*a21=buf+2*hs,*a22=buf+3*hs;
    float *b11=buf+4*hs,*b12=buf+5*hs,*b21=buf+6*hs,*b22=buf+7*hs;
    float *t1=buf+8*hs,*t2=buf+9*hs,*acc=buf+10*hs;
    float *M1=buf+11*hs,*M2=buf+12*hs,*M3=buf+13*hs,*M4=buf+14*hs;
    float *M5=buf+15*hs,*M6=buf+16*hs,*M7=buf+17*hs;

    fq_get(a11,A,n,h,0,0); fq_get(a12,A,n,h,0,1);
    fq_get(a21,A,n,h,1,0); fq_get(a22,A,n,h,1,1);
    fq_get(b11,B,n,h,0,0); fq_get(b12,B,n,h,0,1);
    fq_get(b21,B,n,h,1,0); fq_get(b22,B,n,h,1,1);

    fadd(t1,a11,a22,hs); fadd(t2,b11,b22,hs); sgemm_strassen(M1,t1,t2,h);
    fadd(t1,a21,a22,hs);                      sgemm_strassen(M2,t1,b11,h);
    fsub(t2,b12,b22,hs);                      sgemm_strassen(M3,a11,t2,h);
    fsub(t2,b21,b11,hs);                      sgemm_strassen(M4,a22,t2,h);
    fadd(t1,a11,a12,hs);                      sgemm_strassen(M5,t1,b22,h);
    fsub(t1,a21,a11,hs); fadd(t2,b11,b12,hs); sgemm_strassen(M6,t1,t2,h);
    fsub(t1,a12,a22,hs); fadd(t2,b21,b22,hs); sgemm_strassen(M7,t1,t2,h);

    fadd(acc,M1,M4,hs); fsub(acc,acc,M5,hs); fadd(acc,acc,M7,hs);
    fq_put(C,acc,n,h,0,0);
    fadd(acc,M3,M5,hs);                       fq_put(C,acc,n,h,0,1);
    fadd(acc,M2,M4,hs);                       fq_put(C,acc,n,h,1,0);
    fsub(acc,M1,M2,hs); fadd(acc,acc,M3,hs); fadd(acc,acc,M6,hs);
    fq_put(C,acc,n,h,1,1);
    free(buf);
}

/* ------------------------------------------------------------------ *
 * bfloat16: 8 mantissa bits, same exponent range as fp32.  There is no
 * bf16 arithmetic path in portable C, so this measures the *accuracy*
 * cost of the format, not the speedup real hardware gets from it.
 * ------------------------------------------------------------------ */
static float to_bf16(float x)
{
    uint32_t u;
    memcpy(&u, &x, 4);
    uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);   /* round to nearest even */
    u += rounding;
    u &= 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

/* ------------------------------------------------------------------ *
 * int8 quantized GEMM, per-channel symmetric (row scales for A, column
 * scales for B) -- the arrangement production inference stacks use.
 * Operands are held as int16 so the inner loop maps onto widening
 * multiply-accumulate; accumulation is int32.
 * ------------------------------------------------------------------ */
static void qpack_a(int16_t *Ap, const int16_t *A, int n,
                    int ic, int mc, int pc, int kc)
{
    for (int pi = 0; pi * FMR < mc; pi++) {
        int16_t *d = Ap + (size_t)pi * FMR * kc;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < FMR; i++)
                d[p*FMR+i] = (pi*FMR+i < mc)
                    ? A[(size_t)(ic + pi*FMR + i)*n + pc + p] : 0;
    }
}
static void qpack_b(int16_t *Bp, const int16_t *B, int n,
                    int jc, int nc, int pc, int kc)
{
    for (int pj = 0; pj * FNR < nc; pj++) {
        int16_t *d = Bp + (size_t)pj * FNR * kc;
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < FNR; j++)
                d[p*FNR+j] = (pj*FNR+j < nc)
                    ? B[(size_t)(pc+p)*n + jc + pj*FNR + j] : 0;
    }
}
static void qmicro(const int16_t *restrict a, const int16_t *restrict b,
                   int32_t *c, int ldc, int kc, int mr, int nr)
{
    vi acc[FMR][2];
    for (int i = 0; i < FMR; i++) { acc[i][0] = (vi){0}; acc[i][1] = (vi){0}; }

    for (int p = 0; p < kc; p++) {
        const int16_t *bp = b + (size_t)p * FNR;
        /* panels stay int16 (a quarter of the fp32 footprint) and widen
         * in-register, so the cache benefit of quantization is preserved */
        vi b0 = __builtin_convertvector(*(const vsu *)bp, vi);
        vi b1 = __builtin_convertvector(*(const vsu *)(bp + VL), vi);
        const int16_t *ap = a + (size_t)p * FMR;
        for (int i = 0; i < FMR; i++) {
            vi av = (vi){0} + (int32_t)ap[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }

    int32_t tmp[FMR][FNR];
    for (int i = 0; i < FMR; i++) {
        *(viu *)&tmp[i][0]  = acc[i][0];
        *(viu *)&tmp[i][VL] = acc[i][1];
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++) c[(size_t)i * ldc + j] += tmp[i][j];
}

static void qgemm_packed(int32_t *C, const int16_t *A, const int16_t *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(int32_t));
    for (int jc = 0; jc < n; jc += FNC) {
        int nc = FMIN(FNC, n - jc);
        int16_t *Ap = malloc((size_t)(FMC/FMR+1)*FMR*FKC*sizeof(int16_t));
        int16_t *Bp = malloc((size_t)(FNC/FNR+1)*FNR*FKC*sizeof(int16_t));
        if (!Ap || !Bp) { free(Ap); free(Bp); return; }
        for (int pc = 0; pc < n; pc += FKC) {
            int kc = FMIN(FKC, n - pc);
            qpack_b(Bp, B, n, jc, nc, pc, kc);
            for (int ic = 0; ic < n; ic += FMC) {
                int mc = FMIN(FMC, n - ic);
                qpack_a(Ap, A, n, ic, mc, pc, kc);
                for (int j = 0; j < nc; j += FNR) {
                    int nr = FMIN(FNR, nc - j);
                    for (int i = 0; i < mc; i += FMR) {
                        int mr = FMIN(FMR, mc - i);
                        qmicro(Ap + (size_t)(i/FMR)*FMR*kc,
                               Bp + (size_t)(j/FNR)*FNR*kc,
                               C + (size_t)(ic+i)*n + jc + j, n, kc, mr, nr);
                    }
                }
            }
        }
        free(Ap); free(Bp);
    }
}

/* ------------------------------------------------------------------ */
typedef struct {
    const char *name;
    double secs;
    double err;
    int    exactish;      /* 1 = same arithmetic as fp32 reference path  */
} mlres_t;

static uint64_t sm(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double rel_err(const float *C, const double *R, size_t n)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)C[i] - R[i];
        num += d * d; den += R[i] * R[i];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

int ml_run(int n, int reps, int csv, int with_naive)
{
    size_t nn = (size_t)n * n;
    float  *A = malloc(nn * sizeof(float));
    float  *B = malloc(nn * sizeof(float));
    float  *C = malloc(nn * sizeof(float));
    float  *Ab = malloc(nn * sizeof(float));
    float  *Bb = malloc(nn * sizeof(float));
    double *R = malloc(nn * sizeof(double));
    if (!A || !B || !C || !Ab || !Bb || !R) { fprintf(stderr, "oom\n"); return 1; }

    uint64_t s = 987654321ULL;
    for (size_t i = 0; i < nn; i++)
        A[i] = (float)((double)(sm(&s) >> 11) / 9007199254740992.0 * 2.0 - 1.0);
    for (size_t i = 0; i < nn; i++)
        B[i] = (float)((double)(sm(&s) >> 11) / 9007199254740992.0 * 2.0 - 1.0);

    /* float64 reference */
    memset(R, 0, nn * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i*n+k];
            const float *Br = B + (size_t)k*n;
            double *Rr = R + (size_t)i*n;
            for (int j = 0; j < n; j++) Rr[j] += a * Br[j];
        }

    mlres_t res[10];
    int nr = 0;
    double t0, t;

#define TIME_IT(label, expr, exact)                                          \
    do {                                                                     \
        double best = 1e30;                                                  \
        for (int r = 0; r < reps; r++) {                                     \
            t0 = now_sec(); expr; t = now_sec() - t0;                        \
            if (t < best) best = t;                                          \
        }                                                                    \
        res[nr].name = label; res[nr].secs = best;                           \
        res[nr].err = rel_err(C, R, nn); res[nr].exactish = exact; nr++;     \
    } while (0)

    if (with_naive) TIME_IT("sgemm-ijk",   sgemm_ijk(C, A, B, n), 1);
    if (with_naive) TIME_IT("sgemm-ikj",   sgemm_ikj(C, A, B, n), 1);
    TIME_IT("sgemm-blocked",  sgemm_blocked(C, A, B, n), 1);
    TIME_IT("sgemm-packed",   sgemm_packed(C, A, B, n), 1);
    TIME_IT("sgemm-strassen", sgemm_strassen(C, A, B, n), 1);

#ifdef HAVE_CBLAS
    TIME_IT("blas-sgemm",
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                        1.0f, A, n, B, n, 0.0f, C, n), 1);
#endif

    for (size_t i = 0; i < nn; i++) Ab[i] = to_bf16(A[i]);
    for (size_t i = 0; i < nn; i++) Bb[i] = to_bf16(B[i]);
    TIME_IT("bf16-packed", sgemm_packed(C, Ab, Bb, n), 0);

    /* int8, per-channel symmetric */
    {
        int16_t *qa = malloc(nn * sizeof(int16_t));
        int16_t *qb = malloc(nn * sizeof(int16_t));
        int32_t *qc = malloc(nn * sizeof(int32_t));
        float *sa = malloc((size_t)n * sizeof(float));
        float *sb = malloc((size_t)n * sizeof(float));
        if (!qa || !qb || !qc || !sa || !sb) { fprintf(stderr, "oom\n"); return 1; }

        for (int i = 0; i < n; i++) {           /* per-row scale for A */
            float mx = 0;
            for (int k = 0; k < n; k++) { float v = fabsf(A[(size_t)i*n+k]);
                                          if (v > mx) mx = v; }
            sa[i] = mx > 0 ? mx / 127.0f : 1.0f;
            for (int k = 0; k < n; k++)
                qa[(size_t)i*n+k] = (int16_t)lrintf(A[(size_t)i*n+k] / sa[i]);
        }
        for (int j = 0; j < n; j++) {           /* per-column scale for B */
            float mx = 0;
            for (int k = 0; k < n; k++) { float v = fabsf(B[(size_t)k*n+j]);
                                          if (v > mx) mx = v; }
            sb[j] = mx > 0 ? mx / 127.0f : 1.0f;
            for (int k = 0; k < n; k++)
                qb[(size_t)k*n+j] = (int16_t)lrintf(B[(size_t)k*n+j] / sb[j]);
        }

        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            t0 = now_sec();
            qgemm_packed(qc, qa, qb, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    C[(size_t)i*n+j] = (float)qc[(size_t)i*n+j] * sa[i] * sb[j];
            t = now_sec() - t0;
            if (t < best) best = t;
        }
        res[nr].name = "int8-packed"; res[nr].secs = best;
        res[nr].err = rel_err(C, R, nn); res[nr].exactish = 0; nr++;
        free(qa); free(qb); free(qc); free(sa); free(sb);
    }

    double flops = 2.0 * (double)n * n * n;
    double base = 0;
    for (int i = 0; i < nr; i++)
        if (!strcmp(res[i].name, "sgemm-packed")) base = res[i].secs;

    if (csv) {
        printf("method,seconds,gflops,speedup,rel_error\n");
        for (int i = 0; i < nr; i++)
            printf("%s,%.6f,%.3f,%.3f,%.3e\n", res[i].name, res[i].secs,
                   flops / res[i].secs / 1e9,
                   base > 0 ? base / res[i].secs : 0.0, res[i].err);
    } else {
        printf("\n==== ML track: n = %d, fp32 inputs, float64 reference ====\n", n);
#ifdef _OPENMP
        printf("(OpenMP enabled)\n");
#endif
        printf("\n%-16s %10s %9s %9s %12s\n",
               "method", "seconds", "GFLOP/s", "vs packed", "rel error");
        printf("---------------------------------------------------------------\n");
        for (int i = 0; i < nr; i++)
            printf("%-16s %10.4f %9.2f %8.2fx %12.2e%s\n",
                   res[i].name, res[i].secs, flops / res[i].secs / 1e9,
                   base > 0 ? base / res[i].secs : 0.0, res[i].err,
                   res[i].exactish ? "" : "  <- lossy");
        printf("\nrel error is ||C - C_fp64||_F / ||C_fp64||_F.  fp32 methods\n"
               "differ from each other only by summation order.\n");
    }

    free(A); free(B); free(C); free(Ab); free(Bb); free(R);
    return 0;
}
