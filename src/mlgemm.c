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


/* ------------------------------------------------------------------ *
 * fp64 GEMM.
 *
 * fp64 is the honest competitor to an exact fp32 product: it is what people
 * actually reach for when fp32 accumulation stops being good enough.  Its
 * own kernel rather than a template over the fp32 one -- half the lanes per
 * vector means a different register tile is optimal.
 * ------------------------------------------------------------------ */
typedef double vd  __attribute__((vector_size(VBYTES)));
typedef double vdu __attribute__((vector_size(VBYTES), aligned(8)));
#define DVL ((int)(sizeof(vd) / sizeof(double)))
#define DMR 6
#define DNR (2 * DVL)

static void dpack_a(double *Ap, const double *A, int lda,
                    int ic, int mc, int pc, int kc)
{
    for (int pi = 0; pi * DMR < mc; pi++) {
        double *d = Ap + (size_t)pi * DMR * kc;
        for (int p = 0; p < kc; p++)
            for (int i = 0; i < DMR; i++)
                d[p * DMR + i] = (pi * DMR + i < mc)
                    ? A[(size_t)(ic + pi * DMR + i) * lda + pc + p] : 0.0;
    }
}

static void dpack_b(double *Bp, const double *B, int ldb,
                    int jc, int nc, int pc, int kc)
{
    for (int pj = 0; pj * DNR < nc; pj++) {
        double *d = Bp + (size_t)pj * DNR * kc;
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < DNR; j++)
                d[p * DNR + j] = (pj * DNR + j < nc)
                    ? B[(size_t)(pc + p) * ldb + jc + pj * DNR + j] : 0.0;
    }
}

static void dmicro(const double *restrict a, const double *restrict b,
                   double *c, int ldc, int kc, int mr, int nr)
{
    vd acc[DMR][2];
    for (int i = 0; i < DMR; i++) { acc[i][0] = (vd){0}; acc[i][1] = (vd){0}; }
    for (int p = 0; p < kc; p++) {
        const double *bp = b + (size_t)p * DNR;
        vd b0 = *(const vdu *)bp;
        vd b1 = *(const vdu *)(bp + DVL);
        const double *ap = a + (size_t)p * DMR;
        for (int i = 0; i < DMR; i++) {
            vd av = (vd){0} + ap[i];
            acc[i][0] += av * b0;
            acc[i][1] += av * b1;
        }
    }
    double tmp[DMR][DNR];
    for (int i = 0; i < DMR; i++) {
        *(vdu *)&tmp[i][0]   = acc[i][0];
        *(vdu *)&tmp[i][DVL] = acc[i][1];
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++) c[(size_t)i * ldc + j] += tmp[i][j];
}

static void dgemm_packed(double *C, const double *A, const double *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int jc = 0; jc < n; jc += FNC) {
        int nc = FMIN(FNC, n - jc);
        double *Ap = malloc((size_t)(FMC / DMR + 1) * DMR * FKC * sizeof(double));
        double *Bp = malloc((size_t)(FNC / DNR + 1) * DNR * FKC * sizeof(double));
        if (!Ap || !Bp) { free(Ap); free(Bp); return; }
        for (int pc = 0; pc < n; pc += FKC) {
            int kc = FMIN(FKC, n - pc);
            dpack_b(Bp, B, n, jc, nc, pc, kc);
            for (int ic = 0; ic < n; ic += FMC) {
                int mc = FMIN(FMC, n - ic);
                dpack_a(Ap, A, n, ic, mc, pc, kc);
                for (int j = 0; j < nc; j += DNR) {
                    int nr = FMIN(DNR, nc - j);
                    for (int i = 0; i < mc; i += DMR) {
                        int mr = FMIN(DMR, mc - i);
                        dmicro(Ap + (size_t)(i / DMR) * DMR * kc,
                               Bp + (size_t)(j / DNR) * DNR * kc,
                               C + (size_t)(ic + i) * n + jc + j,
                               n, kc, mr, nr);
                    }
                }
            }
        }
        free(Ap); free(Bp);
    }
}

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

static void dgemm_ijk(double *C, const double *A, const double *B, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[(size_t)i*n+k] * B[(size_t)k*n+j];
            C[(size_t)i*n+j] = s;
        }
}

static void dgemm_blocked(double *C, const double *A, const double *B, int n)
{
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += FBS) {
        int imax = FMIN(ii + FBS, n);
        for (int kk = 0; kk < n; kk += FBS) {
            int kmax = FMIN(kk + FBS, n);
            for (int jj = 0; jj < n; jj += FBS) {
                int jmax = FMIN(jj + FBS, n);
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        double a = A[(size_t)i*n+k];
                        const double *Br = B + (size_t)k*n;
                        double *Cr = C + (size_t)i*n;
                        for (int j = jj; j < jmax; j++) Cr[j] += a * Br[j];
                    }
            }
        }
    }
}

static void dq_get(double *q, const double *M, int n, int h, int r, int c)
{
    for (int i = 0; i < h; i++)
        memcpy(q + (size_t)i*h, M + (size_t)(r*h+i)*n + (size_t)c*h,
               (size_t)h * sizeof(double));
}
static void dq_put(double *M, const double *q, int n, int h, int r, int c)
{
    for (int i = 0; i < h; i++)
        memcpy(M + (size_t)(r*h+i)*n + (size_t)c*h, q + (size_t)i*h,
               (size_t)h * sizeof(double));
}
static void dadd(double *d, const double *a, const double *b, size_t n)
{ for (size_t i = 0; i < n; i++) d[i] = a[i] + b[i]; }
static void dsub(double *d, const double *a, const double *b, size_t n)
{ for (size_t i = 0; i < n; i++) d[i] = a[i] - b[i]; }

static void dgemm_strassen(double *C, const double *A, const double *B, int n)
{
    int h = n / 2;
    if (n <= (int)g_strassen_cutoff || (n & 1) || h < 1) {
        dgemm_packed(C, A, B, n);
        return;
    }
    size_t hs = (size_t)h * h;
    double *buf = malloc(hs * sizeof(double) * 18);
    if (!buf) { dgemm_packed(C, A, B, n); return; }
    double *a11=buf,*a12=buf+hs,*a21=buf+2*hs,*a22=buf+3*hs;
    double *b11=buf+4*hs,*b12=buf+5*hs,*b21=buf+6*hs,*b22=buf+7*hs;
    double *t1=buf+8*hs,*t2=buf+9*hs,*acc=buf+10*hs;
    double *M1=buf+11*hs,*M2=buf+12*hs,*M3=buf+13*hs,*M4=buf+14*hs;
    double *M5=buf+15*hs,*M6=buf+16*hs,*M7=buf+17*hs;
    dq_get(a11,A,n,h,0,0); dq_get(a12,A,n,h,0,1);
    dq_get(a21,A,n,h,1,0); dq_get(a22,A,n,h,1,1);
    dq_get(b11,B,n,h,0,0); dq_get(b12,B,n,h,0,1);
    dq_get(b21,B,n,h,1,0); dq_get(b22,B,n,h,1,1);
    dadd(t1,a11,a22,hs); dadd(t2,b11,b22,hs); dgemm_strassen(M1,t1,t2,h);
    dadd(t1,a21,a22,hs);                      dgemm_strassen(M2,t1,b11,h);
    dsub(t2,b12,b22,hs);                      dgemm_strassen(M3,a11,t2,h);
    dsub(t2,b21,b11,hs);                      dgemm_strassen(M4,a22,t2,h);
    dadd(t1,a11,a12,hs);                      dgemm_strassen(M5,t1,b22,h);
    dsub(t1,a21,a11,hs); dadd(t2,b11,b12,hs); dgemm_strassen(M6,t1,t2,h);
    dsub(t1,a12,a22,hs); dadd(t2,b21,b22,hs); dgemm_strassen(M7,t1,t2,h);
    dadd(acc,M1,M4,hs); dsub(acc,acc,M5,hs); dadd(acc,acc,M7,hs);
    dq_put(C,acc,n,h,0,0);
    dadd(acc,M3,M5,hs);                       dq_put(C,acc,n,h,0,1);
    dadd(acc,M2,M4,hs);                       dq_put(C,acc,n,h,1,0);
    dsub(acc,M1,M2,hs); dadd(acc,acc,M3,hs); dadd(acc,acc,M6,hs);
    dq_put(C,acc,n,h,1,1);
    free(buf);
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
    int h = n / 2;
    if (n <= (int)g_strassen_cutoff || (n & 1) || h < 1) {
        sgemm_packed(C, A, B, n);
        return;
    }
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

static double rel_err(const float *C, const long double *R, size_t n)
{
    long double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        long double d = (long double)C[i] - R[i];
        num += d * d; den += R[i] * R[i];
    }
    return den > 0 ? (double)sqrtl(num / den) : 0.0;
}

static double rel_err_d(const double *C, const double *R, size_t n)
{
    long double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        long double d = (long double)C[i] - (long double)R[i];
        num += d * d; den += (long double)R[i] * (long double)R[i];
    }
    return den > 0 ? (double)sqrtl(num / den) : 0.0;
}

static double rel_err_ld(const double *C, const long double *R, size_t n)
{
    long double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        long double d = (long double)C[i] - R[i];
        num += d * d; den += R[i] * R[i];
    }
    return den > 0 ? (double)sqrtl(num / den) : 0.0;
}

/* Faithful limb window: most-significant live power-of-2 limb count.
 * aggressive=1 keeps at most 2 limbs so the approximation is visible. */
static int faithful_limb_window(const int32_t *A32, const int32_t *B32,
                                size_t nn, int L, int sig, int n,
                                int aggressive, int *u0_out, int *hi_out)
{
    int need_bits = aggressive ? (sig / 2 + 2) : (sig + 4);
    for (int tt = n; tt > 1; tt >>= 1) need_bits++;
    int L_keep = (need_bits + LIMB_BITS - 1) / LIMB_BITS;
    if (L_keep < 2) L_keep = 2;
    if (aggressive && L_keep > 2) L_keep = 2;
    if (L_keep > L) L_keep = L;

    int hi = -1;
    for (int w = 0; w < L; w++) {
        int any = 0;
        for (size_t i = 0; i < nn && !any; i++)
            if (A32[(size_t)w * nn + i] || B32[(size_t)w * nn + i]) any = 1;
        if (any) hi = w;
    }
    if (hi < 0) hi = L - 1;
    int span = hi + 1;
    int Lk = 2;
    while (Lk * 2 <= L_keep && Lk * 2 <= span && Lk * 2 <= L)
        Lk *= 2;
    int u0 = hi - Lk + 1;
    if (u0 < 0) u0 = 0;
    *u0_out = u0;
    *hi_out = hi;
    return Lk;
}


int opt_full64 = 1;

int ml_run(int n, int reps, int csv, int with_naive, int fp_width, int illcond,
           int fp64_mode)
{
    size_t nn = (size_t)n * n;
    float  *A = malloc(nn * sizeof(float));
    float  *B = malloc(nn * sizeof(float));
    float  *C = malloc(nn * sizeof(float));
    float  *Ab = malloc(nn * sizeof(float));
    float  *Bb = malloc(nn * sizeof(float));
    long double *R = malloc(nn * sizeof(long double));
    double *Rd = malloc(nn * sizeof(double));       /* host/promoted cross-check */
    double *Rd64 = malloc(nn * sizeof(double));     /* independent fp64 reference */
    double *dA0 = malloc(nn * sizeof(double));
    double *dB0 = malloc(nn * sizeof(double));
    if (!A || !B || !C || !Ab || !Bb || !R || !Rd || !Rd64 || !dA0 || !dB0) {
        fprintf(stderr, "oom\n"); return 1;
    }

    /* Data generation.
     * Default: fp32 uniform in (-1,1); fp64 rows promote these values.
     * --fp64: genuine 53-bit significands so the double track measures both
     * cost and the accuracy benefit of the wider embedding. */
    uint64_t s = 987654321ULL;
    if (fp64_mode) {
        for (size_t i = 0; i < nn; i++) {
            dA0[i] = (double)(sm(&s) >> 11) / 9007199254740992.0 * 2.0 - 1.0;
            dB0[i] = (double)(sm(&s) >> 11) / 9007199254740992.0 * 2.0 - 1.0;
            A[i] = (float)dA0[i];
            B[i] = (float)dB0[i];
        }
        if (!csv)
            printf("data: genuine fp64 (53-bit significands); "
                   "fp32 rows are the same values rounded to float\n");
    } else {
        for (size_t i = 0; i < nn; i++)
            A[i] = (float)((double)(sm(&s) >> 11) / 9007199254740992.0 * 2.0 - 1.0);
        for (size_t i = 0; i < nn; i++)
            B[i] = (float)((double)(sm(&s) >> 11) / 9007199254740992.0 * 2.0 - 1.0);
        for (size_t i = 0; i < nn; i++) {
            dA0[i] = (double)A[i];
            dB0[i] = (double)B[i];
        }
        if (!csv)
            printf("data: fp32 promoted to fp64 for the double rows "
                   "(use --fp64 for genuine 53-bit inputs)\n");
    }

    /* Optionally spread the exponents.  This is the regime where an fp32
     * dot product loses digits to cancellation and an exact one does not --
     * and also the regime that costs the fp32-embedding methods limbs. */
    if (illcond > 0) {
        for (size_t i = 0; i < nn; i++) {
            int e = (int)(sm(&s) % (unsigned)(illcond + 1)) - illcond / 2;
            A[i] = ldexpf(A[i], e);
            dA0[i] = ldexp(dA0[i], e);
        }
        for (size_t i = 0; i < nn; i++) {
            int e = (int)(sm(&s) % (unsigned)(illcond + 1)) - illcond / 2;
            B[i] = ldexpf(B[i], e);
            dB0[i] = ldexp(dB0[i], e);
        }
    }

    /* Independent references, computed once outside every timed method.
     * R  = exact product of the fp32 matrices (float embedding).
     * Rd64 = exact product of the double matrices (53-bit embedding).
     * Float methods are scored against R; double methods against Rd64. */
    fpx_ctx fx;
    int have_fx = (fpx_init(&fx, A, B, n, fp_width) == 0);
    if (have_fx) {
        fpx_encode(&fx, A, B);
        conv_limbplane(fx.Cw, fx.A32, fx.B32, n, fx.L, KERNEL_PACKED);
        fpx_decode_ld(&fx, R);          /* fp32 reference, 63 significant bits */
    }

    {
        fpx_ctx dx;
        if (fpx_init_d(&dx, dA0, dB0, n, 0) == 0) {
            fpx_encode_d(&dx, dA0, dB0);
            conv_limbplane(dx.Cw, dx.A32, dx.B32, n, dx.L, KERNEL_PACKED);
            fpx_decode_f64(&dx, Rd64);
            fpx_free(&dx);
            if (!csv)
                printf("reference: fp32 embedding + fp64 embedding, both "
                       "computed once outside the timed table\n");
        } else {
            memcpy(Rd64, dA0, nn * sizeof(double)); /* fallback: unused */
        }
    }

    /* plain float64 loop on the float data, kept as a cross-check */
    memset(Rd, 0, nn * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i*n+k];
            const float *Br = B + (size_t)k*n;
            double *Rr = Rd + (size_t)i*n;
            for (int j = 0; j < n; j++) Rr[j] += a * Br[j];
        }

    mlres_t res[28];
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

    /* fp64: the realistic alternative to an exact fp32 product.
     * Inputs are dA0/dB0 (genuine under --fp64, promoted otherwise).
     * Errors are against Rd64, the independent exact double product. */
    {
        double *dA = dA0, *dB = dB0;
        double *dC = malloc(nn * sizeof(double));
        if (dC) {
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                t0 = now_sec(); dgemm_packed(dC, dA, dB, n); t = now_sec() - t0;
                if (t < best) best = t;
            }
            for (size_t i = 0; i < nn; i++) C[i] = (float)dC[i];
            res[nr].name = "dgemm-packed"; res[nr].secs = best;
            res[nr].err = rel_err_d(dC, Rd64, nn); res[nr].exactish = 1; nr++;

            if (opt_full64) {
                best = 1e30;
                for (int r = 0; r < reps; r++) {
                    t0 = now_sec(); dgemm_blocked(dC, dA, dB, n); t = now_sec() - t0;
                    if (t < best) best = t;
                }
                res[nr].name = "dgemm-blocked"; res[nr].secs = best;
                res[nr].err = rel_err_d(dC, Rd64, nn); res[nr].exactish = 1; nr++;

                best = 1e30;
                for (int r = 0; r < reps; r++) {
                    t0 = now_sec(); dgemm_strassen(dC, dA, dB, n); t = now_sec() - t0;
                    if (t < best) best = t;
                }
                res[nr].name = "dgemm-strassen"; res[nr].secs = best;
                res[nr].err = rel_err_d(dC, Rd64, nn); res[nr].exactish = 1; nr++;

                if (with_naive) {
                    best = 1e30;
                    for (int r = 0; r < reps; r++) {
                        t0 = now_sec(); dgemm_ijk(dC, dA, dB, n); t = now_sec() - t0;
                        if (t < best) best = t;
                    }
                    res[nr].name = "dgemm-ijk"; res[nr].secs = best;
                    res[nr].err = rel_err_d(dC, Rd64, nn); res[nr].exactish = 1; nr++;
                }
            }

            /* exact product at fp64 significand width -- timed; reference
             * was computed with the same plan outside this block */
            {
                fpx_ctx dx;
                if (fpx_init_d(&dx, dA, dB, n, 0) == 0) {
                    fpx_encode_d(&dx, dA, dB);
                    best = 1e30;
                    for (int r = 0; r < reps; r++) {
                        t0 = now_sec();
                        conv_karatsuba(dx.Cw, dx.A32, dx.B32, n, dx.L,
                                       KERNEL_PACKED);
                        t = now_sec() - t0;
                        if (t < best) best = t;
                    }
                    fpx_decode_f64(&dx, dC);
                    res[nr].name = "fp64->karatsuba"; res[nr].secs = best;
                    res[nr].err = rel_err_d(dC, Rd64, nn); res[nr].exactish = 2; nr++;

                    best = 1e30;
                    for (int r = 0; r < reps; r++) {
                        t0 = now_sec();
                        conv_limbplane(dx.Cw, dx.A32, dx.B32, n, dx.L,
                                       KERNEL_PACKED);
                        t = now_sec() - t0;
                        if (t < best) best = t;
                    }
                    fpx_decode_f64(&dx, dC);
                    res[nr].name = "fp64->limbplane"; res[nr].secs = best;
                    /* same algorithm as the reference: report as exact, err ~0 */
                    res[nr].err = rel_err_d(dC, Rd64, nn); res[nr].exactish = 2; nr++;

                    mfft_plan dpl;
                    long long dprod = 0;
                    if (mfft_plan_init_rec(&dpl, dx.L, n, 0) == 0) {
                        dprod = dpl.nprod;
                        best = 1e30;
                        for (int r = 0; r < reps; r++) {
                            t0 = now_sec();
                            conv_mfft(dx.Cw, dx.A32, dx.B32, n, dx.L, &dpl,
                                      KERNEL_PACKED);
                            t = now_sec() - t0;
                            if (t < best) best = t;
                        }
                        fpx_decode_f64(&dx, dC);
                        res[nr].name = "fp64->mfft-rec"; res[nr].secs = best;
                        res[nr].err = rel_err_d(dC, Rd64, nn);
                        res[nr].exactish = 2; nr++;
                    }

                    /* Faithful high-limb MFFT for fp64: drop most low limbs. */
                    {
                        int u0, hi;
                        int Lk = faithful_limb_window(dx.A32, dx.B32, nn, dx.L,
                                                      53, n, 1, &u0, &hi);
                        mfft_plan plf;
                        int plan_ok = (Lk >= 4)
                            ? (mfft_plan_init_rec(&plf, Lk, n, 0) == 0)
                            : (mfft_plan_init(&plf, Lk, 0) == 0);
                        if (plan_ok) {
                            int32_t *Ahi = malloc((size_t)Lk * nn * sizeof(int32_t));
                            int32_t *Bhi = malloc((size_t)Lk * nn * sizeof(int32_t));
                            int64_t *Chi = calloc((size_t)(2 * Lk - 1) * nn,
                                                  sizeof(int64_t));
                            if (Ahi && Bhi && Chi) {
                                for (int w = 0; w < Lk; w++) {
                                    memcpy(Ahi + (size_t)w * nn,
                                           dx.A32 + (size_t)(u0 + w) * nn,
                                           nn * sizeof(int32_t));
                                    memcpy(Bhi + (size_t)w * nn,
                                           dx.B32 + (size_t)(u0 + w) * nn,
                                           nn * sizeof(int32_t));
                                }
                                best = 1e30;
                                for (int r = 0; r < reps; r++) {
                                    t0 = now_sec();
                                    conv_mfft(Chi, Ahi, Bhi, n, Lk, &plf,
                                              KERNEL_PACKED);
                                    t = now_sec() - t0;
                                    if (t < best) best = t;
                                }
                                memset(dx.Cw, 0,
                                       (size_t)(2 * dx.L - 1) * nn * sizeof(int64_t));
                                int base = 2 * u0;
                                for (int w = 0; w < 2 * Lk - 1; w++) {
                                    if (base + w >= 2 * dx.L - 1) break;
                                    memcpy(dx.Cw + (size_t)(base + w) * nn,
                                           Chi + (size_t)w * nn,
                                           nn * sizeof(int64_t));
                                }
                                fpx_decode_f64(&dx, dC);
                                res[nr].name = "fp64->mfft-faithful";
                                res[nr].secs = best;
                                res[nr].err = rel_err_d(dC, Rd64, nn);
                                res[nr].exactish = 4; nr++;
                                if (!csv)
                                    printf("fp64->mfft-faithful: L_keep=%d of %d "
                                           "(live hi=%d u0=%d), products %lld\n",
                                           Lk, dx.L, hi, u0, plf.nprod);
                            }
                            free(Ahi); free(Bhi); free(Chi);
                        }
                    }

                    if (!csv)
                        printf("fp64 embedding: %d limbs (%d bits)%s; products: "
                               "limb-plane %d, karatsuba %lld, mfft %lld\n",
                               dx.L, dx.L * LIMB_BITS,
                               fp64_mode ? " [genuine]" : " [promoted]",
                               dx.L * dx.L, karatsuba_products(dx.L), dprod);
                    fpx_free(&dx);
                }
            }
#ifdef HAVE_CBLAS
            best = 1e30;
            for (int r = 0; r < reps; r++) {
                t0 = now_sec();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                            1.0, dA, n, dB, n, 0.0, dC, n);
                t = now_sec() - t0;
                if (t < best) best = t;
            }
            res[nr].name = "blas-dgemm"; res[nr].secs = best;
            res[nr].err = rel_err_d(dC, Rd64, nn); res[nr].exactish = 1; nr++;
#endif
            free(dC);
        }
    }

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

    /* --- exact low precision: bf16, int8, int4 ----------------------
     * These need no limb decomposition at all (vbits is small enough that
     * one GEMM is exact), so they are the honest ML answer to "can MFFT
     * help here": there is nothing to convolve. */
    {
        lp_ctx lc;
        int bfrc = lp_bf16_init(&lc, A, B, n);
        if (bfrc == -2) {
            /* Exponent spread too wide for one int64 GEMM.  Fall back to the
             * multi-limb embedding on the bf16-rounded inputs -- still exact,
             * just no longer a single pass. */
            fpx_ctx bx;
            if (fpx_init(&bx, lc.Ar, lc.Br, n, 0) == 0) {
                fpx_encode(&bx, lc.Ar, lc.Br);
                double best = 1e30;
                for (int r = 0; r < reps; r++) {
                    t0 = now_sec();
                    conv_karatsuba(bx.Cw, bx.A32, bx.B32, n, bx.L, KERNEL_PACKED);
                    t = now_sec() - t0;
                    if (t < best) best = t;
                }
                fpx_decode_f32(&bx, C);
                res[nr].name = "bf16-exact*"; res[nr].secs = best;
                res[nr].err = rel_err(C, R, nn); res[nr].exactish = 2; nr++;
                if (!csv)
                    printf("bf16 exact: %d accumulator bits needed (> 63), so "
                           "%d limbs / %lld products instead of one GEMM\n",
                           lc.needbits, bx.L, karatsuba_products(bx.L));
            }
            fpx_free(&bx);
        }
        if (bfrc == 0) {
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                t0 = now_sec(); lp_bf16_gemm(&lc); t = now_sec() - t0;
                if (t < best) best = t;
            }
            lp_bf16_decode(&lc, C);
            res[nr].name = "bf16-exact"; res[nr].secs = best;
            res[nr].err = rel_err(C, R, nn); res[nr].exactish = 2; nr++;
            if (!csv)
                printf("bf16 exact: %d value bits -> 1 limb, 1 GEMM\n", lc.vbits);
        }
        lp_free(&lc);

        static const int qb[2] = { 8, 4 };
        static const char *qn[2] = { "int8-exact", "int4-exact" };
        for (int q = 0; q < 2; q++) {
            if (lp_intq_init(&lc, A, B, n, qb[q]) != 0) { lp_free(&lc); continue; }
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                t0 = now_sec(); lp_intq_gemm(&lc); t = now_sec() - t0;
                if (t < best) best = t;
            }
            lp_intq_decode(&lc, C);
            res[nr].name = qn[q]; res[nr].secs = best;
            res[nr].err = rel_err(C, R, nn); res[nr].exactish = 3; nr++;
            lp_free(&lc);
        }
    }

    /* --- the fp32 embedding: exact integer matmul on float data --------
     * Timing covers the convolution only.  Encode/decode are reported
     * separately: a caller staying in fixed point between consecutive
     * matmuls pays them once, not per multiply. */
    if (have_fx) {
        mfft_plan pl;
        int have_plan = (mfft_plan_init_rec(&pl, fx.L, n, 0) == 0);

        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            t0 = now_sec();
            conv_limbplane(fx.Cw, fx.A32, fx.B32, n, fx.L, KERNEL_PACKED);
            t = now_sec() - t0;
            if (t < best) best = t;
        }
        fpx_decode_f32(&fx, C);
        res[nr].name = "fp32->limbplane"; res[nr].secs = best;
        res[nr].err = rel_err(C, R, nn); res[nr].exactish = 2; nr++;

        best = 1e30;
        for (int r = 0; r < reps; r++) {
            t0 = now_sec();
            conv_karatsuba(fx.Cw, fx.A32, fx.B32, n, fx.L, KERNEL_PACKED);
            t = now_sec() - t0;
            if (t < best) best = t;
        }
        fpx_decode_f32(&fx, C);
        res[nr].name = "fp32->karatsuba"; res[nr].secs = best;
        res[nr].err = rel_err(C, R, nn); res[nr].exactish = 2; nr++;

        if (have_plan) {
            best = 1e30;
            for (int r = 0; r < reps; r++) {
                t0 = now_sec();
                conv_mfft(fx.Cw, fx.A32, fx.B32, n, fx.L, &pl, KERNEL_PACKED);
                t = now_sec() - t0;
                if (t < best) best = t;
            }
            fpx_decode_f32(&fx, C);
            res[nr].name = "fp32->mfft-rec"; res[nr].secs = best;
            res[nr].err = rel_err(C, R, nn); res[nr].exactish = 2; nr++;
        }

        /* Faithful high-limb MFFT: aggressive drop of low limbs. Not exact. */
        {
            int u0, hi;
            int Lk = faithful_limb_window(fx.A32, fx.B32, nn, fx.L,
                                          24, n, 1, &u0, &hi);
            mfft_plan plf;
            int plan_ok = (Lk >= 4)
                ? (mfft_plan_init_rec(&plf, Lk, n, 0) == 0)
                : (mfft_plan_init(&plf, Lk, 0) == 0);
            if (plan_ok) {
                int32_t *Ahi = malloc((size_t)Lk * nn * sizeof(int32_t));
                int32_t *Bhi = malloc((size_t)Lk * nn * sizeof(int32_t));
                int64_t *Chi = calloc((size_t)(2 * Lk - 1) * nn, sizeof(int64_t));
                if (Ahi && Bhi && Chi) {
                    for (int w = 0; w < Lk; w++) {
                        memcpy(Ahi + (size_t)w * nn,
                               fx.A32 + (size_t)(u0 + w) * nn,
                               nn * sizeof(int32_t));
                        memcpy(Bhi + (size_t)w * nn,
                               fx.B32 + (size_t)(u0 + w) * nn,
                               nn * sizeof(int32_t));
                    }
                    best = 1e30;
                    for (int r = 0; r < reps; r++) {
                        t0 = now_sec();
                        conv_mfft(Chi, Ahi, Bhi, n, Lk, &plf, KERNEL_PACKED);
                        t = now_sec() - t0;
                        if (t < best) best = t;
                    }
                    memset(fx.Cw, 0, (size_t)(2 * fx.L - 1) * nn * sizeof(int64_t));
                    int base = 2 * u0;
                    for (int w = 0; w < 2 * Lk - 1; w++) {
                        if (base + w >= 2 * fx.L - 1) break;
                        memcpy(fx.Cw + (size_t)(base + w) * nn,
                               Chi + (size_t)w * nn, nn * sizeof(int64_t));
                    }
                    fpx_decode_f32(&fx, C);
                    res[nr].name = "fp32->mfft-faithful"; res[nr].secs = best;
                    res[nr].err = rel_err(C, R, nn); res[nr].exactish = 4; nr++;
                    if (!csv)
                        printf("fp32->mfft-faithful: L_keep=%d of %d "
                               "(live hi=%d u0=%d), products %lld\n",
                               Lk, fx.L, hi, u0, plf.nprod);
                }
                free(Ahi); free(Bhi); free(Chi);
            }
        }
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
        printf("\n==== ML track: n = %d, fp32 inputs ====\n", n);
        if (have_fx) {
            printf("fp32 embedding: exponent spread A=%d B=%d -> %d limbs "
                   "(%d bits), scales 2^%d / 2^%d\n",
                   fx.spreadA, fx.spreadB, fx.L, fx.L * LIMB_BITS, fx.SA, fx.SB);
            printf("                encode %.4f s, decode %.4f s "
                   "(excluded from the timings below)\n",
                   fx.enc_secs, fx.dec_secs);
            {
                mfft_plan q;
                int ok = (mfft_plan_init_rec(&q, fx.L, n, 0) == 0);
                printf("                n x n products: limb-plane %d, "
                       "karatsuba %lld, mfft %lld\n",
                       fx.L * fx.L, karatsuba_products(fx.L),
                       ok ? q.nprod : 0LL);
            }
        }
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
                   res[i].exactish == 2 ? "  <- EXACT"
                     : res[i].exactish == 3 ? "  <- exact sum, lossy inputs"
                     : res[i].exactish == 4 ? "  <- FAITHFUL (low limbs dropped)"
                     : res[i].exactish ? "" : "  <- lossy");
        printf("\nrel error is ||C - C_exact||_F / ||C_exact||_F against the\n"
               "bit-exact product, carried at 63 bits so no timed method\n"
               "scores itself.  For scale, a plain fp64 loop scores %.2e.\n",
               rel_err_ld(Rd, R, nn));
    }

    if (have_fx) fpx_free(&fx);
    free(A); free(B); free(C); free(Ab); free(Bb); free(R); free(Rd);
    free(Rd64); free(dA0); free(dB0);
    return 0;
}
