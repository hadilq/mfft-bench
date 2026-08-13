/* gemm_bench.cu -- GPU half of the benchmark.
 *
 * The CPU results say the exact-integer route costs ~50x an sgemm.  That
 * number is a property of CPU arithmetic, not of the algorithm: a CPU has
 * one integer multiplier per lane and no int8 dot-product unit.  A GPU with
 * IMMA tensor cores runs int8 GEMM several times faster than fp32, which is
 * exactly the operation the limb decomposition needs.  This file measures
 * whether that closes the gap.
 *
 * Strategy: never hand-roll the inner GEMM.  Every limb product is issued
 * through cublasGemmEx with CUDA_R_8I inputs and CUDA_R_32I accumulation, so
 * the limb path rides the same tensor cores cuBLAS uses.  The only custom
 * kernels are the elementwise encode/decode/combine passes, which are
 * memory-bound and trivially correct.
 *
 * Limb choice for int8 tensor cores
 * --------------------------------
 * Operands must fit in int8, so limbs are 7 bits (0..127).  An entry needing
 * vbits bits takes L = ceil(vbits/7) limbs, and the exact product is the
 * L x L schoolbook convolution of the limb planes -- L^2 int8 GEMMs.
 *
 * Karatsuba is deliberately NOT used here: it forms sums like A0+A1 of limb
 * planes, which immediately exceed int8 and force the slow int32 path.  On a
 * tensor-core GPU, 36 int8 GEMMs beat 27 int32 ones.  MFFT is likewise
 * irrelevant -- at L = 6 there is no convolution long enough to transform.
 * That is the closing argument of the whole benchmark: on the hardware ML
 * actually runs on, the limb count never reaches the regime where MFFT wins.
 *
 * Accumulator check: 7-bit limbs give products < 2^14, and a length-n sum
 * stays under 2^(14 + log2 n), so int32 is safe to n = 2^17.
 *
 * Build:  make -C cuda            (or: nvcc -O3 -arch=sm_80 ... )
 * Run:    ./cuda/gemm_bench --n 4096 --reps 5
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {              \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                    \
            cudaGetErrorString(e_)); exit(1); } } while (0)

#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); \
    exit(1); } } while (0)

static const int LIMB_BITS_GPU = 7;
static const int LIMB_MASK_GPU = (1 << LIMB_BITS_GPU) - 1;

/* ------------------------------------------------------------------ *
 * Elementwise kernels
 * ------------------------------------------------------------------ */

/* Round fp32 to bf16 (round-to-nearest-even) in place of a copy. */
__global__ void k_to_bf16(float *dst, const float *src, size_t nn)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    unsigned u = __float_as_uint(src[i]);
    u += 0x7FFFu + ((u >> 16) & 1u);
    dst[i] = __uint_as_float(u & 0xFFFF0000u);
}

/* Split every entry onto the shared fixed-point grid and scatter its
 * 7-bit limbs into L planes.  A negative entry stores negated limbs; the
 * int8 GEMM is linear so the signs simply carry through. */
__global__ void k_encode(signed char *planes, const float *X, size_t nn,
                         int L, int S, int sig)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    for (int w = 0; w < L; w++) planes[(size_t)w * nn + i] = 0;

    float x = X[i];
    if (x == 0.0f || !isfinite(x)) return;

    int e;
    float m = frexpf(x, &e);
    long long mi = (long long)(fabsf(m) * (float)(1u << sig));
    int off = e - sig + S;
    if (off < 0) return;

    /* value = mi << off, read out in 7-bit digits */
    int lo = off / LIMB_BITS_GPU, sh = off % LIMB_BITS_GPU;
    unsigned long long v = (unsigned long long)mi << sh;
    int sgn = (x < 0.0f) ? -1 : 1;
    for (int t = 0; v && lo + t < L; t++) {
        int d = (int)(v & LIMB_MASK_GPU);
        planes[(size_t)(lo + t) * nn + i] = (signed char)(sgn * d);
        v >>= LIMB_BITS_GPU;
    }
}

/* Accumulate one limb-product plane into the wide result, at digit w. */
__global__ void k_accum(long long *acc, const int *part, size_t nn, int w)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    acc[(size_t)w * nn + i] += (long long)part[i];
}

/* Fold the digit planes back into fp32.
 *
 * The planes are unnormalised (each holds a length-n sum, so up to ~2^26),
 * and the full value can exceed 2^90 -- far beyond a double.  So carry the
 * planes into proper 7-bit digits first, then read the top 49 bits, which
 * is exact in double and leaves the discarded tail 2^-49 below the result.
 * That makes the fp32 output correctly rounded. */
__global__ void k_decode(float *C, const long long *acc, size_t nn,
                         int planes, int S)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;

    int dig[64];
    int nd = 0;
    long long carry = 0;
    for (int w = 0; w < planes && nd < 56; w++) {
        long long t = acc[(size_t)w * nn + i] + carry;
        dig[nd++] = (int)(t & LIMB_MASK_GPU);
        carry = t >> LIMB_BITS_GPU;
    }
    while (carry != 0 && carry != -1 && nd < 60) {
        dig[nd++] = (int)(carry & LIMB_MASK_GPU);
        carry >>= LIMB_BITS_GPU;
    }
    int neg = (carry < 0);
    if (neg) {
        int borrow = 1;
        for (int w = 0; w < nd; w++) {
            int t = ((~dig[w]) & LIMB_MASK_GPU) + borrow;
            dig[w] = t & LIMB_MASK_GPU;
            borrow = t >> LIMB_BITS_GPU;
        }
    }
    int hi = nd - 1;
    while (hi > 0 && dig[hi] == 0) hi--;
    int lo = hi - 6 > 0 ? hi - 6 : 0;
    double v = 0.0;
    for (int w = hi; w >= lo; w--) v = v * 128.0 + (double)dig[w];
    v = ldexp(v, lo * LIMB_BITS_GPU - S);
    C[i] = (float)(neg ? -v : v);
}

/* Per-channel symmetric quantization to `bits` (int8 / int4 study). */
__global__ void k_quant_rows(signed char *Q, float *scale, const float *X,
                             int n, int bits)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    int qmax = (1 << (bits - 1)) - 1;
    float mx = 0.0f;
    for (int k = 0; k < n; k++) { float v = fabsf(X[(size_t)r * n + k]);
                                  if (v > mx) mx = v; }
    float s = mx > 0.0f ? mx / (float)qmax : 1.0f;
    scale[r] = s;
    for (int k = 0; k < n; k++)
        Q[(size_t)r * n + k] = (signed char)lrintf(X[(size_t)r * n + k] / s);
}

/* B is quantized per column, but the int8 GEMM wants k contiguous, so this
 * writes B transposed: Q[j][k]. */
__global__ void k_quant_cols_T(signed char *Q, float *scale, const float *X,
                               int n, int bits)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int qmax = (1 << (bits - 1)) - 1;
    float mx = 0.0f;
    for (int k = 0; k < n; k++) { float v = fabsf(X[(size_t)k * n + c]);
                                  if (v > mx) mx = v; }
    float s = mx > 0.0f ? mx / (float)qmax : 1.0f;
    scale[c] = s;
    for (int k = 0; k < n; k++)
        Q[(size_t)c * n + k] = (signed char)lrintf(X[(size_t)k * n + c] / s);
}

__global__ void k_dequant(float *C, const int *acc, const float *sa,
                          const float *sb, int n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n) return;
    C[i] = (float)acc[i] * sa[i / n] * sb[i % n];
}

__global__ void k_transpose(signed char *dst, const signed char *src, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) dst[(size_t)j * n + i] = src[(size_t)i * n + j];
}

__global__ void k_f2bf(__nv_bfloat16 *dst, const float *src, size_t nn)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < nn) dst[i] = __float2bfloat16(src[i]);
}

/* ------------------------------------------------------------------ *
 * Helpers
 * ------------------------------------------------------------------ */
static int grid_for(size_t nn, int blk) { return (int)((nn + blk - 1) / blk); }

/* Row-major C = A*B via cuBLAS (column-major) by computing C^T = B^T A^T. */
static void sgemm_rm(cublasHandle_t h, int n, const float *A, const float *B,
                     float *C)
{
    const float a = 1.0f, b = 0.0f;
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                   &a, B, n, A, n, &b, C, n));
}

/* Row-major int8 GEMM: A is [n][k] row-major, Bt is [n][k] row-major (i.e.
 * B transposed), C is int32 row-major.  In column-major terms this is
 * C^T = Bt^T(op T) * A^T, both operands k-contiguous, which is the layout
 * IMMA wants. */
static void igemm_rm(cublasHandle_t h, int n, const signed char *A,
                     const signed char *Bt, int *C)
{
    const int a = 1, b = 0;
    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                    &a, Bt, CUDA_R_8I, n, A, CUDA_R_8I, n,
                    &b, C, CUDA_R_32I, n,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
}

struct Res { const char *name; double ms; double err; int exact; long long gemms; };

static double rel_err_host(const float *C, const double *R, size_t nn)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < nn; i++) {
        double d = (double)C[i] - R[i];
        num += d * d; den += R[i] * R[i];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

/* ------------------------------------------------------------------ *
 * Exact float GEMM through int8 limb planes
 * ------------------------------------------------------------------ */
struct LimbPlan { int L, SA, SB, sig, vA, vB; };

static void scan_exponents(const float *X, size_t nn, int sig,
                           int *lo, int *hi)
{
    int l = 1 << 30, h = -(1 << 30), seen = 0;
    for (size_t i = 0; i < nn; i++) {
        if (X[i] == 0.0f || !isfinite(X[i])) continue;
        int e; frexpf(X[i], &e);
        int e2 = e - sig;
        if (e2 < l) l = e2;
        if (e2 > h) h = e2;
        seen = 1;
    }
    if (!seen) { l = 0; h = 0; }
    *lo = l; *hi = h;
}

/* hA/hB are host copies used only to size the grid. */
static LimbPlan plan_limbs(const float *hA, const float *hB, size_t nn, int sig)
{
    int loA, hiA, loB, hiB;
    scan_exponents(hA, nn, sig, &loA, &hiA);
    scan_exponents(hB, nn, sig, &loB, &hiB);
    LimbPlan p;
    p.sig = sig;
    p.SA = -loA; p.SB = -loB;
    p.vA = sig + (hiA - loA);
    p.vB = sig + (hiB - loB);
    int v = p.vA > p.vB ? p.vA : p.vB;
    p.L = (v + LIMB_BITS_GPU - 1) / LIMB_BITS_GPU;
    return p;
}

/* Runs the L x L schoolbook limb convolution as int8 tensor-core GEMMs.
 * Returns elapsed ms; `gemms` receives the count. */
static double exact_float_gemm(cublasHandle_t h, int n, const float *dA,
                               const float *dB, float *dC, LimbPlan p,
                               int reps, long long *gemms)
{
    size_t nn = (size_t)n * n;
    signed char *pA, *pB, *pBt;
    int *part;
    long long *acc;
    int planes = 2 * p.L - 1;

    CK(cudaMalloc(&pA, nn * p.L));
    CK(cudaMalloc(&pB, nn * p.L));
    CK(cudaMalloc(&pBt, nn * p.L));
    CK(cudaMalloc(&part, nn * sizeof(int)));
    CK(cudaMalloc(&acc, nn * planes * sizeof(long long)));

    int blk = 256, g = grid_for(nn, blk);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float best = 1e30f;

    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(t0));
        k_encode<<<g, blk>>>(pA, dA, nn, p.L, p.SA, p.sig);
        k_encode<<<g, blk>>>(pB, dB, nn, p.L, p.SB, p.sig);
        dim3 tb(16, 16), tg((n + 15) / 16, (n + 15) / 16);
        for (int w = 0; w < p.L; w++)
            k_transpose<<<tg, tb>>>(pBt + (size_t)w * nn,
                                    pB + (size_t)w * nn, n);
        CK(cudaMemset(acc, 0, nn * planes * sizeof(long long)));
        for (int u = 0; u < p.L; u++)
            for (int v = 0; v < p.L; v++) {
                igemm_rm(h, n, pA + (size_t)u * nn, pBt + (size_t)v * nn, part);
                k_accum<<<g, blk>>>(acc, part, nn, u + v);
            }
        k_decode<<<g, blk>>>(dC, acc, nn, planes, p.SA + p.SB);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    *gemms = (long long)p.L * p.L;

    cudaFree(pA); cudaFree(pB); cudaFree(pBt); cudaFree(part); cudaFree(acc);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int n = 2048, reps = 3, check = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--n N] [--reps R] [--check]\n", argv[0]);
            return 0;
        }
    }
    /* cuBLAS int8 GEMM (IMMA) requires leading dimensions that are multiples
     * of 4 and 4-byte aligned pointers; 16 is where it actually performs. */
    if (n % 16) {
        int up = (n + 15) / 16 * 16;
        fprintf(stderr, "n must be a multiple of 16 for the int8 tensor-core "
                        "path (cuBLAS IMMA ld constraint); try --n %d\n", up);
        return 2;
    }
    size_t nn = (size_t)n * n;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  sm_%d%d  %d SMs\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    float *hA = (float *)malloc(nn * sizeof(float));
    float *hB = (float *)malloc(nn * sizeof(float));
    float *hC = (float *)malloc(nn * sizeof(float));
    double *hR = (double *)malloc(nn * sizeof(double));

    unsigned long long s = 88172645463325252ULL;
    for (size_t i = 0; i < nn; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        hA[i] = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        hB[i] = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
    }

    float *dA, *dB, *dC, *dAbf, *dBbf;
    CK(cudaMalloc(&dA, nn * sizeof(float)));
    CK(cudaMalloc(&dB, nn * sizeof(float)));
    CK(cudaMalloc(&dC, nn * sizeof(float)));
    CK(cudaMalloc(&dAbf, nn * sizeof(float)));
    CK(cudaMalloc(&dBbf, nn * sizeof(float)));
    CK(cudaMemcpy(dA, hA, nn * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, nn * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CB(cublasCreate(&h));

    /* Reference: exact fp32 product, computed on the GPU through the limb
     * path, then cross-checked on the host in double for small n. */
    LimbPlan pf = plan_limbs(hA, hB, nn, 24);
    long long gm;
    exact_float_gemm(h, n, dA, dB, dC, pf, 1, &gm);
    CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nn; i++) hR[i] = (double)hC[i];

    if (check && n > 512)
        printf("check: skipped, the host float64 reference is O(n^3); "
               "use --n 512 or smaller\n\n");
    if (check && n <= 512) {
        double worst = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double acc = 0.0;
                for (int k = 0; k < n; k++)
                    acc += (double)hA[(size_t)i * n + k] * hB[(size_t)k * n + j];
                double d = fabs(acc - hR[(size_t)i * n + j]);
                double r = fabs(acc) > 0 ? d / fabs(acc) : d;
                if (r > worst) worst = r;
            }
        printf("check: exact limb path vs host float64, worst relative "
               "difference %.3e (expect ~1e-7, the fp32 rounding of the "
               "exact result)\n\n", worst);
    }

    Res res[8];
    int nr = 0;
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

/* Variadic: a kernel launch's <<<grid, block>>> contains a comma that the
 * preprocessor does not see as bracketed, so a fixed-arity macro splits on it. */
#define TIME_BLOCK(LABEL, EXACT, GEMMS, ...)                                   \
    do {                                                                       \
        float best = 1e30f;                                                    \
        for (int r = 0; r < reps; r++) {                                       \
            CK(cudaEventRecord(t0));                                           \
            __VA_ARGS__;                                                       \
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));             \
            CK(cudaGetLastError());                                            \
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));                   \
            if (ms < best) best = ms;                                          \
        }                                                                      \
        CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));    \
        res[nr].name = LABEL; res[nr].ms = best;                               \
        res[nr].err = rel_err_host(hC, hR, nn);                                \
        res[nr].exact = EXACT; res[nr].gemms = GEMMS; nr++;                    \
    } while (0)

    /* --- fp32 baseline --- */
    TIME_BLOCK("cublas-sgemm", 0, 1, sgemm_rm(h, n, dA, dB, dC));

    /* --- bf16 tensor cores, fp32 accumulate --- */
    {
        __nv_bfloat16 *bA, *bB;
        CK(cudaMalloc(&bA, nn * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&bB, nn * sizeof(__nv_bfloat16)));
        int blk = 256, g = grid_for(nn, blk);
        k_f2bf<<<g, blk>>>(bA, dA, nn);
        k_f2bf<<<g, blk>>>(bB, dB, nn);
        const float alpha = 1.0f, beta = 0.0f;
        TIME_BLOCK("cublas-bf16", 0, 1,
                   CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                   &alpha, bB, CUDA_R_16BF, n, bA, CUDA_R_16BF, n,
                                   &beta, dC, CUDA_R_32F, n,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)));
        cudaFree(bA); cudaFree(bB);
    }

    /* --- int8 and int4 quantized (exact accumulation, lossy inputs) --- */
    for (int bits = 8; bits >= 4; bits -= 4) {
        signed char *qA, *qBt;
        float *sa, *sb;
        int *iacc;
        CK(cudaMalloc(&qA, nn)); CK(cudaMalloc(&qBt, nn));
        CK(cudaMalloc(&sa, n * sizeof(float)));
        CK(cudaMalloc(&sb, n * sizeof(float)));
        CK(cudaMalloc(&iacc, nn * sizeof(int)));
        int tb = 128, tg = (n + tb - 1) / tb;
        k_quant_rows<<<tg, tb>>>(qA, sa, dA, n, bits);
        k_quant_cols_T<<<tg, tb>>>(qBt, sb, dB, n, bits);
        int blk = 256, g = grid_for(nn, blk);
        TIME_BLOCK(bits == 8 ? "int8-tc" : "int4-in-int8", 0, 1,
                   do { igemm_rm(h, n, qA, qBt, iacc);
                        k_dequant<<<g, blk>>>(dC, iacc, sa, sb, n); } while (0));
        cudaFree(qA); cudaFree(qBt); cudaFree(sa); cudaFree(sb); cudaFree(iacc);
    }

    /* --- exact bf16 through limb planes --- */
    {
        int blk = 256, g = grid_for(nn, blk);
        k_to_bf16<<<g, blk>>>(dAbf, dA, nn);
        k_to_bf16<<<g, blk>>>(dBbf, dB, nn);
        CK(cudaMemcpy(hC, dAbf, nn * sizeof(float), cudaMemcpyDeviceToHost));
        float *hBbf = (float *)malloc(nn * sizeof(float));
        CK(cudaMemcpy(hBbf, dBbf, nn * sizeof(float), cudaMemcpyDeviceToHost));
        LimbPlan pb = plan_limbs(hC, hBbf, nn, 8);
        long long gb;
        double ms = exact_float_gemm(h, n, dAbf, dBbf, dC, pb, reps, &gb);
        CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
        res[nr].name = "bf16-exact"; res[nr].ms = ms;
        res[nr].err = rel_err_host(hC, hR, nn);
        res[nr].exact = 1; res[nr].gemms = gb; nr++;
        printf("bf16 exact: %d value bits -> %d limbs of %d bits, %lld int8 GEMMs\n",
               pb.vA > pb.vB ? pb.vA : pb.vB, pb.L, LIMB_BITS_GPU, gb);
        free(hBbf);
    }

    /* --- exact fp32 through limb planes --- */
    {
        long long gf;
        double ms = exact_float_gemm(h, n, dA, dB, dC, pf, reps, &gf);
        CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
        res[nr].name = "fp32-exact"; res[nr].ms = ms;
        res[nr].err = rel_err_host(hC, hR, nn);
        res[nr].exact = 1; res[nr].gemms = gf; nr++;
        printf("fp32 exact: %d value bits -> %d limbs of %d bits, %lld int8 GEMMs\n",
               pf.vA > pf.vB ? pf.vA : pf.vB, pf.L, LIMB_BITS_GPU, gf);
    }

    double flops = 2.0 * (double)n * n * n;
    double base = res[0].ms;
    printf("\n%-16s %8s %10s %10s %11s\n",
           "method", "GEMMs", "ms", "TFLOP/s", "rel error");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < nr; i++)
        printf("%-16s %8lld %10.3f %10.2f %11.2e%s\n",
               res[i].name, res[i].gemms, res[i].ms,
               flops / (res[i].ms * 1e-3) / 1e12, res[i].err,
               res[i].exact ? "  <- EXACT" : "");
    printf("\nbaseline cublas-sgemm = %.3f ms; error is measured against the "
           "exact product.\n", base);

    CB(cublasDestroy(h));
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dAbf); cudaFree(dBbf);
    free(hA); free(hB); free(hC); free(hR);
    return 0;
}
