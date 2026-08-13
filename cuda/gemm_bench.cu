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
#include <cublasLt.h>
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

/* Fold the digit planes back into fp32.
 *
 * The planes are unnormalised (each holds a length-n sum, so up to ~2^26),
 * and the full value can exceed 2^90 -- far beyond a double.  So carry the
 * planes into proper 7-bit digits first, then read the top 49 bits, which
 * is exact in double and leaves the discarded tail 2^-49 below the result.
 * That makes the fp32 output correctly rounded. */
__global__ void k_decode(float *C, const int *acc, size_t nn,
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

/* fp64 input on the same grid: 53-bit significand instead of 24. */
__global__ void k_encode_d(signed char *planes, const double *X, size_t nn,
                           int L, int S)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    for (int w = 0; w < L; w++) planes[(size_t)w * nn + i] = 0;

    double x = X[i];
    if (x == 0.0 || !isfinite(x)) return;

    int e;
    double m = frexp(x, &e);
    long long mi = (long long)(fabs(m) * 9007199254740992.0);   /* 2^53 */
    int off = e - 53 + S;
    if (off < 0) return;

    int lo = off / LIMB_BITS_GPU, sh = off % LIMB_BITS_GPU;
    unsigned long long v = (unsigned long long)mi << sh;        /* <= 59 bits */
    int sgn = (x < 0.0) ? -1 : 1;
    for (int t = 0; v && lo + t < L; t++) {
        int d = (int)(v & LIMB_MASK_GPU);
        planes[(size_t)(lo + t) * nn + i] = (signed char)(sgn * d);
        v >>= LIMB_BITS_GPU;
    }
}

/* Same fold as k_decode but keeping 56 bits, so the result is a double
 * rather than an fp32.  Used for the reference and for fp64-exact. */
__global__ void k_decode_d(double *C, const int *acc, size_t nn,
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
    int lo = hi - 7 > 0 ? hi - 7 : 0;
    double v = 0.0;
    for (int w = hi; w >= lo; w--) v = v * 128.0 + (double)dig[w];
    v = ldexp(v, lo * LIMB_BITS_GPU - S);
    C[i] = neg ? -v : v;
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
 * Hand-written dp4a int8 GEMM.
 *
 * cuBLAS refuses CUBLAS_COMPUTE_32I on some architectures (RTX 50-series /
 * sm_120 returns CUBLAS_STATUS_NOT_SUPPORTED), and the int8 -> fp32 variant
 * is not a safe substitute here: our accumulators reach n * 2^14, which
 * exceeds the 24-bit fp32 significand past n = 1024 and would silently stop
 * being exact.  So the fallback keeps int32 accumulation and uses __dp4a,
 * available on every GPU from sm_61 onward.
 *
 * 64x64 block tile, 16x16 threads, 4x4 outputs each, K in chunks of 32.
 * A is [n][k] and Bt is [n][k], both k-contiguous, so all loads are the
 * 4-byte packed int8x4 that __dp4a consumes directly.
 * ------------------------------------------------------------------ */
/* Tile shape, k-chunk depth and blocking are all template parameters, and
 * the right combination is not predictable from first principles -- it
 * depends on the SM count, the shared-memory budget and how much of the
 * grid a given n fills.  So the kernel is instantiated several ways and
 * tune_dp4a() times them on the actual problem at startup and keeps the
 * winner.  That turns a guess into a measurement.
 *
 * Every instantiation uses 256 threads (TS/TH == 16).  Shared memory is
 * TS x (KW+1) ints per operand; the +1 padding keeps the per-thread strided
 * reads off a single bank.  Deeper KW means fewer __syncthreads and more
 * reuse per byte loaded, at the cost of shared memory.
 */
template <int TS, int TH, int KW>
__global__ __launch_bounds__(256)
void k_dp4a_gemm(const int *__restrict__ A4, const int *__restrict__ B4,
                 int *__restrict__ C, int n, int kw)
{
    __shared__ int As[TS][KW + 1];
    __shared__ int Bs[TS][KW + 1];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;   /* 0..255 */
    int r0 = blockIdx.y * TS, c0 = blockIdx.x * TS;
    int acc[TH][TH];
#pragma unroll
    for (int i = 0; i < TH; i++)
#pragma unroll
        for (int j = 0; j < TH; j++) acc[i][j] = 0;

    for (int k0 = 0; k0 < kw; k0 += KW) {
        /* coalesced in groups of KW along k, which is the contiguous axis */
        for (int t = tid; t < TS * KW; t += 256) {
            int rr = t / KW, cc = t % KW;
            As[rr][cc] = (r0 + rr < n && k0 + cc < kw)
                       ? A4[(size_t)(r0 + rr) * kw + k0 + cc] : 0;
            Bs[rr][cc] = (c0 + rr < n && k0 + cc < kw)
                       ? B4[(size_t)(c0 + rr) * kw + k0 + cc] : 0;
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < KW; q++) {
            int av[TH], bv[TH];
#pragma unroll
            for (int i = 0; i < TH; i++) av[i] = As[threadIdx.y * TH + i][q];
#pragma unroll
            for (int j = 0; j < TH; j++) bv[j] = Bs[threadIdx.x * TH + j][q];
#pragma unroll
            for (int i = 0; i < TH; i++)
#pragma unroll
                for (int j = 0; j < TH; j++)
                    acc[i][j] = __dp4a(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }

    /* accumulate, so a limb product lands straight on its digit plane */
    for (int i = 0; i < TH; i++) {
        int r = r0 + threadIdx.y * TH + i;
        if (r >= n) continue;
        for (int j = 0; j < TH; j++) {
            int c = c0 + threadIdx.x * TH + j;
            if (c < n) C[(size_t)r * n + c] += acc[i][j];
        }
    }
}

static int g_sms = 1;                 /* set from the device properties */

typedef void (*dp_launch_fn)(int, const signed char *, const signed char *, int *);

template <int TS, int TH, int KW>
static void dp_launch_t(int n, const signed char *A, const signed char *Bt,
                        int *C)
{
    dim3 blk(TS / TH, TS / TH);
    dim3 grd((n + TS - 1) / TS, (n + TS - 1) / TS);
    k_dp4a_gemm<TS, TH, KW><<<grd, blk>>>((const int *)A, (const int *)Bt,
                                          C, n, n / 4);
}

struct DpCfg { dp_launch_fn fn; const char *name; int ts; };
static const DpCfg g_dpcfgs[] = {
    { dp_launch_t<128, 8, 8>,  "128x128 tile, 8x8/thread, k=32",  128 },
    { dp_launch_t<128, 8, 16>, "128x128 tile, 8x8/thread, k=64",  128 },
    { dp_launch_t<128, 8, 32>, "128x128 tile, 8x8/thread, k=128", 128 },
    { dp_launch_t<64, 4, 16>,  "64x64 tile, 4x4/thread, k=64",     64 },
    { dp_launch_t<64, 4, 32>,  "64x64 tile, 4x4/thread, k=128",    64 },
    { dp_launch_t<32, 2, 32>,  "32x32 tile, 2x2/thread, k=128",    32 },
};
#define DP_NCFG ((int)(sizeof(g_dpcfgs) / sizeof(g_dpcfgs[0])))
static int g_dpcfg = 0;

static void dp4a_launch(int n, const signed char *A, const signed char *Bt,
                        int *C)
{
    g_dpcfgs[g_dpcfg].fn(n, A, Bt, C);
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
 * IMMA wants.
 *
 * g_i8mode is decided once by probe_i8(): 0 = cuBLAS int8 -> int32,
 * 1 = the dp4a kernel above.  Both accumulate in int32, so both are exact. */
static int g_i8mode = 1;   /* 0 = cuBLAS, 1 = dp4a, 2 = cuBLASLt */

static cublasStatus_t try_cublas_i8(cublasHandle_t h, int n, const signed char *A,
                                    const signed char *Bt, int *C,
                                    cublasGemmAlgo_t algo)
{
    const int a = 1, b = 0;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                        &a, Bt, CUDA_R_8I, n, A, CUDA_R_8I, n,
                        &b, C, CUDA_R_32I, n,
                        CUBLAS_COMPUTE_32I, algo);
}

/* ------------------------------------------------------------------ *
 * cuBLASLt int8 path.
 *
 * The legacy cublasGemmEx refuses CUBLAS_COMPUTE_32I on sm_120 with CUDA
 * 12.4.  cuBLASLt sometimes exposes kernels the legacy API will not, and
 * its heuristic query answers the question directly: ask for algorithms
 * matching the int8 -> int32 configuration and see whether any come back.
 * If none do, the capability genuinely is not there and dp4a stands.
 * ------------------------------------------------------------------ */
static cublasLtHandle_t g_lt = NULL;
static cublasLtMatmulDesc_t g_ltdesc = NULL;
static cublasLtMatrixLayout_t g_lta = NULL, g_ltb = NULL, g_ltc = NULL;
static cublasLtMatmulHeuristicResult_t g_ltheur;
static void *g_ltws = NULL;
static size_t g_ltwsz = 32u << 20;

static void lt_teardown(void)
{
    if (g_lta) cublasLtMatrixLayoutDestroy(g_lta);
    if (g_ltb) cublasLtMatrixLayoutDestroy(g_ltb);
    if (g_ltc) cublasLtMatrixLayoutDestroy(g_ltc);
    if (g_ltdesc) cublasLtMatmulDescDestroy(g_ltdesc);
    if (g_lt) cublasLtDestroy(g_lt);
    if (g_ltws) cudaFree(g_ltws);
    g_lta = NULL; g_ltb = NULL; g_ltc = NULL;
    g_ltdesc = NULL; g_lt = NULL; g_ltws = NULL;
}

/* Mirrors igemm_rm's mapping: column-major C^T = op(Bt)^T * op(A). */
static int lt_setup(int n)
{
    if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) { g_lt = NULL; return 0; }
    if (cublasLtMatmulDescCreate(&g_ltdesc, CUBLAS_COMPUTE_32I,
                                 CUDA_R_32I) != CUBLAS_STATUS_SUCCESS)
        { lt_teardown(); return 0; }

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(g_ltdesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &opT, sizeof opT);
    cublasLtMatmulDescSetAttribute(g_ltdesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &opN, sizeof opN);

    /* op(A) = T, so the stored operand is k x m */
    if (cublasLtMatrixLayoutCreate(&g_lta, CUDA_R_8I, n, n, n) != CUBLAS_STATUS_SUCCESS
     || cublasLtMatrixLayoutCreate(&g_ltb, CUDA_R_8I, n, n, n) != CUBLAS_STATUS_SUCCESS
     || cublasLtMatrixLayoutCreate(&g_ltc, CUDA_R_32I, n, n, n) != CUBLAS_STATUS_SUCCESS)
        { lt_teardown(); return 0; }

    if (cudaMalloc(&g_ltws, g_ltwsz) != cudaSuccess) { g_ltws = NULL; g_ltwsz = 0; }

    cublasLtMatmulPreference_t pref = NULL;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS)
        { lt_teardown(); return 0; }
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ltwsz, sizeof g_ltwsz);

    int got = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        g_lt, g_ltdesc, g_lta, g_ltb, g_ltc, g_ltc, pref, 1, &g_ltheur, &got);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != CUBLAS_STATUS_SUCCESS || got < 1) { lt_teardown(); return 0; }
    return 1;
}

static void igemm_lt(int n, const signed char *A, const signed char *Bt, int *C)
{
    (void)n;
    const int a = 1, b = 1;
    CB(cublasLtMatmul(g_lt, g_ltdesc, &a, Bt, g_lta, A, g_ltb,
                      &b, C, g_ltc, C, g_ltc, &g_ltheur.algo,
                      g_ltws, g_ltwsz, 0));
}

/* C += A * Bt^T.  Accumulating rather than overwriting lets each limb
 * product land directly on its digit plane: 49 GEMMs at n=4096 would
 * otherwise cost 49 extra read-modify-write passes over a 67 MB buffer. */
static void igemm_rm(cublasHandle_t h, int n, const signed char *A,
                     const signed char *Bt, int *C)
{
    if (g_i8mode == 0) {
        const int a = 1, b = 1;
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                        &a, Bt, CUDA_R_8I, n, A, CUDA_R_8I, n,
                        &b, C, CUDA_R_32I, n,
                        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        return;
    }
    if (g_i8mode == 2) { igemm_lt(n, A, Bt, C); return; }
    dp4a_launch(n, A, Bt, C);
}

/* Time every dp4a instantiation on the real problem and keep the fastest.
 * A tile that wins at n=8192 can lose badly at n=512 by leaving most of the
 * SMs idle, so this has to be decided against the actual n, not compiled in. */
static void tune_dp4a(int n, int verbose)
{
    size_t nn = (size_t)n * n;
    signed char *a, *b;
    int *c;
    if (cudaMalloc(&a, nn) != cudaSuccess) return;
    if (cudaMalloc(&b, nn) != cudaSuccess) { cudaFree(a); return; }
    if (cudaMalloc(&c, nn * sizeof(int)) != cudaSuccess) {
        cudaFree(a); cudaFree(b); return;
    }
    CK(cudaMemset(a, 1, nn));
    CK(cudaMemset(b, 1, nn));

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    double flops = 2.0 * (double)n * n * n;
    float best = 1e30f;
    int bestc = 0;

    if (verbose) printf("dp4a autotune (n=%d):\n", n);
    for (int i = 0; i < DP_NCFG; i++) {
        int blocks = ((n + g_dpcfgs[i].ts - 1) / g_dpcfgs[i].ts);
        blocks *= blocks;
        CK(cudaMemset(c, 0, nn * sizeof(int)));
        g_dpcfgs[i].fn(n, a, b, c);            /* warm up / catch failures */
        if (cudaDeviceSynchronize() != cudaSuccess) {
            if (verbose) printf("  %-34s  unavailable\n", g_dpcfgs[i].name);
            cudaGetLastError();
            continue;
        }
        float ms = 1e30f;
        for (int r = 0; r < 2; r++) {
            CK(cudaEventRecord(t0));
            g_dpcfgs[i].fn(n, a, b, c);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float e; CK(cudaEventElapsedTime(&e, t0, t1));
            if (e < ms) ms = e;
        }
        if (verbose)
            printf("  %-34s %8.3f ms  %7.2f TOP/s  %5d blocks (%.1f/SM)\n",
                   g_dpcfgs[i].name, ms, flops / (ms * 1e-3) / 1e12, blocks,
                   (double)blocks / (double)g_sms);
        if (ms < best) { best = ms; bestc = i; }
    }
    g_dpcfg = bestc;
    if (verbose)
        printf("  -> %s\n\n", g_dpcfgs[bestc].name);

    cudaFree(a); cudaFree(b); cudaFree(c);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* Pick the int8 path once, up front, and say which one won.  cuBLAS is
 * strongly preferred: it reaches the IMMA tensor cores, while the dp4a
 * kernel is plain integer SIMD and will be several times slower. */
static void probe_i8(cublasHandle_t h, int n)
{
    signed char *a, *b;
    int *c;
    CK(cudaMalloc(&a, (size_t)n * n));
    CK(cudaMalloc(&b, (size_t)n * n));
    CK(cudaMalloc(&c, (size_t)n * n * sizeof(int)));
    CK(cudaMemset(a, 1, (size_t)n * n));
    CK(cudaMemset(b, 1, (size_t)n * n));

    cublasStatus_t st = try_cublas_i8(h, n, a, b, c, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS)
        st = try_cublas_i8(h, n, a, b, c, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (st == CUBLAS_STATUS_SUCCESS && cudaDeviceSynchronize() == cudaSuccess) {
        g_i8mode = 0;
        printf("int8 path: cuBLAS CUBLAS_COMPUTE_32I (tensor cores)\n");
    } else if (lt_setup(n)) {
        /* verify the Lt path actually runs and agrees with dp4a before
         * trusting it: a heuristic hit is not a guarantee */
        CK(cudaMemset(c, 0, (size_t)n * n * sizeof(int)));
        igemm_lt(n, a, b, c);
        int ok = (cudaDeviceSynchronize() == cudaSuccess);
        int probe = 0;
        if (ok) CK(cudaMemcpy(&probe, c, sizeof(int), cudaMemcpyDeviceToHost));
        if (ok && probe == n) {
            g_i8mode = 2;
            printf("int8 path: cuBLAS refused CUBLAS_COMPUTE_32I (status %d), "
                   "but cuBLASLt accepts it -- using cuBLASLt\n"
                   "           (tensor cores).  This is the fast path.\n",
                   (int)st);
        } else {
            printf("int8 path: cuBLASLt heuristic matched but the matmul "
                   "returned %d instead of %d; falling back to __dp4a.\n",
                   probe, n);
            cudaGetLastError();
            lt_teardown();
            g_i8mode = 1;
        }
    } else {
        g_i8mode = 1;
        printf("int8 path: cuBLAS refused CUBLAS_COMPUTE_32I (status %d) and "
               "cuBLASLt offered no algorithm for\n"
               "           int8 -> int32 either, so the capability is not "
               "exposed on this toolkit/GPU pair.\n"
               "           Using the built-in __dp4a kernel: exactness is "
               "unaffected -- both accumulate in int32 --\n"
               "           but read the exact-path timings as an upper bound "
               "on cost, not as what tensor cores would give.\n", (int)st);
    }
    cudaFree(a); cudaFree(b); cudaFree(c);
}

struct Res { const char *name; double ms; double err; int exact; long long gemms; };

static double rel_err_host_d(const double *C, const double *R, size_t nn)
{
    double num = 0, den = 0;
    for (size_t i = 0; i < nn; i++) {
        double d = C[i] - R[i];
        num += d * d; den += R[i] * R[i];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

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
struct LimbPlan { int LA, LB, SA, SB, sig, vA, vB; };

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

static void scan_exponents_d(const double *X, size_t nn, int sig,
                             int *lo, int *hi)
{
    int l = 1 << 30, h = -(1 << 30), seen = 0;
    for (size_t i = 0; i < nn; i++) {
        if (X[i] == 0.0 || !isfinite(X[i])) continue;
        int e; frexp(X[i], &e);
        int e2 = e - sig;
        if (e2 < l) l = e2;
        if (e2 > h) h = e2;
        seen = 1;
    }
    if (!seen) { l = 0; h = 0; }
    *lo = l; *hi = h;
}

/* hA/hB are host copies used only to size the grid. */
static LimbPlan plan_limbs_d(const double *hA, const double *hB, size_t nn,
                             int sig)
{
    int loA, hiA, loB, hiB;
    scan_exponents_d(hA, nn, sig, &loA, &hiA);
    scan_exponents_d(hB, nn, sig, &loB, &hiB);
    LimbPlan p;
    p.sig = sig;
    p.SA = -loA; p.SB = -loB;
    p.vA = sig + (hiA - loA);
    p.vB = sig + (hiB - loB);
    p.LA = (p.vA + LIMB_BITS_GPU - 1) / LIMB_BITS_GPU;
    p.LB = (p.vB + LIMB_BITS_GPU - 1) / LIMB_BITS_GPU;
    return p;
}

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
    /* Size A and B independently: their exponent spreads differ, and the
     * GEMM count is LA*LB, so one wide matrix should not inflate both. */
    p.LA = (p.vA + LIMB_BITS_GPU - 1) / LIMB_BITS_GPU;
    p.LB = (p.vB + LIMB_BITS_GPU - 1) / LIMB_BITS_GPU;
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
    int *acc;
    int planes = p.LA + p.LB - 1;

    /* Digit planes stay int32: one plane accumulates at most
     * min(LA,LB) * n * 2^14, which is under 2^31 for every size here. */
    double bound = (double)(p.LA < p.LB ? p.LA : p.LB) * n * 16384.0;
    if (bound >= 2147483648.0) {
        fprintf(stderr, "digit planes would overflow int32 at n=%d (bound %.0f)\n",
                n, bound);
        exit(1);
    }

    CK(cudaMalloc(&pA, nn * p.LA));
    CK(cudaMalloc(&pB, nn * p.LB));
    CK(cudaMalloc(&pBt, nn * p.LB));
    CK(cudaMalloc(&acc, nn * planes * sizeof(int)));

    int blk = 256, g = grid_for(nn, blk);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float best = 1e30f;

    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(t0));
        k_encode<<<g, blk>>>(pA, dA, nn, p.LA, p.SA, p.sig);
        k_encode<<<g, blk>>>(pB, dB, nn, p.LB, p.SB, p.sig);
        dim3 tb(16, 16), tg((n + 15) / 16, (n + 15) / 16);
        for (int w = 0; w < p.LB; w++)
            k_transpose<<<tg, tb>>>(pBt + (size_t)w * nn,
                                    pB + (size_t)w * nn, n);
        CK(cudaMemset(acc, 0, nn * planes * sizeof(int)));
        for (int u = 0; u < p.LA; u++)
            for (int v = 0; v < p.LB; v++)
                igemm_rm(h, n, pA + (size_t)u * nn, pBt + (size_t)v * nn,
                         acc + (size_t)(u + v) * nn);
        k_decode<<<g, blk>>>(dC, acc, nn, planes, p.SA + p.SB);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    *gemms = (long long)p.LA * p.LB;

    cudaFree(pA); cudaFree(pB); cudaFree(pBt); cudaFree(acc);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best;
}

/* Exact fp64 product through the same int8 limb machinery.  A 53-bit
 * significand plus the exponent spread needs ~10 limbs, so ~100 int8 GEMMs
 * -- but on a consumer GPU, where fp64 runs at 1/64 of fp32, that can still
 * come out ahead of a single cublasDgemm.  That comparison is the point. */
static double exact_double_gemm(cublasHandle_t h, int n, const double *dA,
                                const double *dB, double *dC, LimbPlan p,
                                int reps, long long *gemms)
{
    size_t nn = (size_t)n * n;
    signed char *pA, *pB, *pBt;
    int *acc;
    int planes = p.LA + p.LB - 1;

    double bound = (double)(p.LA < p.LB ? p.LA : p.LB) * n * 16384.0;
    if (bound >= 2147483648.0) {
        fprintf(stderr, "fp64 digit planes would overflow int32 at n=%d\n", n);
        *gemms = 0;
        return 0.0;
    }

    CK(cudaMalloc(&pA, nn * p.LA));
    CK(cudaMalloc(&pB, nn * p.LB));
    CK(cudaMalloc(&pBt, nn * p.LB));
    CK(cudaMalloc(&acc, nn * planes * sizeof(int)));

    int blk = 256, g = grid_for(nn, blk);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float best = 1e30f;

    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(t0));
        k_encode_d<<<g, blk>>>(pA, dA, nn, p.LA, p.SA);
        k_encode_d<<<g, blk>>>(pB, dB, nn, p.LB, p.SB);
        dim3 tb(16, 16), tg((n + 15) / 16, (n + 15) / 16);
        for (int w = 0; w < p.LB; w++)
            k_transpose<<<tg, tb>>>(pBt + (size_t)w * nn,
                                    pB + (size_t)w * nn, n);
        CK(cudaMemset(acc, 0, nn * planes * sizeof(int)));
        for (int u = 0; u < p.LA; u++)
            for (int v = 0; v < p.LB; v++)
                igemm_rm(h, n, pA + (size_t)u * nn, pBt + (size_t)v * nn,
                         acc + (size_t)(u + v) * nn);
        k_decode_d<<<g, blk>>>(dC, acc, nn, planes, p.SA + p.SB);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    *gemms = (long long)p.LA * p.LB;

    cudaFree(pA); cudaFree(pB); cudaFree(pBt); cudaFree(acc);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int n = 2048, reps = 3, check = 0, tile = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--n N] [--reps R] [--check] [--tile I]\n"
                   "  --tile I  force dp4a config I instead of autotuning\n",
                   argv[0]);
            for (int c = 0; c < DP_NCFG; c++)
                printf("      %d: %s\n", c, g_dpcfgs[c].name);
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

    /* Rough device footprint, so an LLM-scale n fails with a number rather
     * than an out-of-memory abort partway through. */
    double need = (double)nn * (5 * 4          /* fp32 A,B,C + two bf16 stagings */
                             + 3 * 8           /* fp64 A,B,C                     */
                             + 20              /* limb planes, worst case        */
                             + 13 * 4)         /* digit planes, worst case       */
                / (1024.0 * 1024.0 * 1024.0);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int rtv = 0, drv = 0;
    cudaRuntimeGetVersion(&rtv);
    cudaDriverGetVersion(&drv);
    printf("device:  %s  sm_%d%d  %d SMs\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);
    printf("toolkit: runtime %d.%d, driver %d.%d\n",
           rtv / 1000, (rtv % 1000) / 10, drv / 1000, (drv % 1000) / 10);
    /* If the toolkit predates the GPU there is no native SASS in the binary
     * and the driver JITs from PTX on first launch; that costs startup time
     * but not throughput, and it is why timings are taken as a best-of-reps. */
    {
        size_t freeb = 0, totb = 0;
        cudaMemGetInfo(&freeb, &totb);
        printf("memory:  need ~%.2f GiB, free %.2f GiB of %.2f GiB\n",
               need, freeb / 1073741824.0, totb / 1073741824.0);
        if (need > freeb / 1073741824.0 * 0.9) {
            fprintf(stderr, "not enough device memory for n=%d; "
                            "try --n %d\n", n, n / 2);
            return 3;
        }
    }
    printf("\n");

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

    g_sms = prop.multiProcessorCount;
    cublasHandle_t h;
    CB(cublasCreate(&h));
    probe_i8(h, n);
    if (g_i8mode != 0) {
        if (tile >= 0 && tile < DP_NCFG) {
            g_dpcfg = tile;
            printf("dp4a tile forced: %s\n", g_dpcfgs[tile].name);
        } else {
            tune_dp4a(n, 1);
        }
    }
    printf("\n");

    /* Reference: the exact product, decoded to double rather than fp32 so
     * that fp64 methods can be scored meaningfully against it.  Rounding it
     * to fp32 would put a 1e-7 floor under every row. */
    LimbPlan pf = plan_limbs(hA, hB, nn, 24);
    {
        double *dRef;
        CK(cudaMalloc(&dRef, nn * sizeof(double)));
        double *hAd = (double *)malloc(nn * sizeof(double));
        double *hBd = (double *)malloc(nn * sizeof(double));
        for (size_t i = 0; i < nn; i++) { hAd[i] = hA[i]; hBd[i] = hB[i]; }
        double *dAd, *dBd;
        CK(cudaMalloc(&dAd, nn * sizeof(double)));
        CK(cudaMalloc(&dBd, nn * sizeof(double)));
        CK(cudaMemcpy(dAd, hAd, nn * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dBd, hBd, nn * sizeof(double), cudaMemcpyHostToDevice));
        LimbPlan pr = plan_limbs_d(hAd, hBd, nn, 53);
        long long gr;
        exact_double_gemm(h, n, dAd, dBd, dRef, pr, 1, &gr);
        CK(cudaMemcpy(hR, dRef, nn * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dRef); cudaFree(dAd); cudaFree(dBd);
        free(hAd); free(hBd);
    }

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

    Res res[12];
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

    /* --- fp64 baseline and the exact fp64 product ---
     * On a consumer card fp64 runs at a small fraction of fp32, so this is
     * the row that decides whether an exact integer product is competitive
     * with simply using more precision. */
    {
        double *dAd, *dBd, *dCd, *hCd;
        CK(cudaMalloc(&dAd, nn * sizeof(double)));
        CK(cudaMalloc(&dBd, nn * sizeof(double)));
        CK(cudaMalloc(&dCd, nn * sizeof(double)));
        hCd = (double *)malloc(nn * sizeof(double));
        double *tmpd = (double *)malloc(nn * sizeof(double));
        for (size_t i = 0; i < nn; i++) tmpd[i] = hA[i];
        CK(cudaMemcpy(dAd, tmpd, nn * sizeof(double), cudaMemcpyHostToDevice));
        for (size_t i = 0; i < nn; i++) tmpd[i] = hB[i];
        CK(cudaMemcpy(dBd, tmpd, nn * sizeof(double), cudaMemcpyHostToDevice));

        const double one = 1.0, zero = 0.0;
        float best = 1e30f;
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(t0));
            CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &one, dBd, n, dAd, n, &zero, dCd, n));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best) best = ms;
        }
        CK(cudaMemcpy(hCd, dCd, nn * sizeof(double), cudaMemcpyDeviceToHost));
        res[nr].name = "cublas-dgemm"; res[nr].ms = best;
        res[nr].err = rel_err_host_d(hCd, hR, nn);
        res[nr].exact = 0; res[nr].gemms = 1; nr++;

        double *hAd = (double *)malloc(nn * sizeof(double));
        double *hBd = (double *)malloc(nn * sizeof(double));
        for (size_t i = 0; i < nn; i++) { hAd[i] = hA[i]; hBd[i] = hB[i]; }
        LimbPlan p64 = plan_limbs_d(hAd, hBd, nn, 53);
        long long g64;
        double ms64 = exact_double_gemm(h, n, dAd, dBd, dCd, p64, reps, &g64);
        if (g64) {
            CK(cudaMemcpy(hCd, dCd, nn * sizeof(double), cudaMemcpyDeviceToHost));
            res[nr].name = "fp64-exact"; res[nr].ms = ms64;
            res[nr].err = rel_err_host_d(hCd, hR, nn);
            res[nr].exact = 1; res[nr].gemms = g64; nr++;
            printf("fp64 exact: %d/%d value bits -> %d x %d limbs of %d bits, "
                   "%lld int8 GEMMs\n", p64.vA, p64.vB, p64.LA, p64.LB,
                   LIMB_BITS_GPU, g64);
        }
        free(hAd); free(hBd); free(hCd); free(tmpd);
        cudaFree(dAd); cudaFree(dBd); cudaFree(dCd);
    }

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
        const char *ilbl = g_i8mode == 0
                        ? (bits == 8 ? "int8-cublas" : "int4-in-int8")
                        : (bits == 8 ? "int8-dp4a"   : "int4-in-dp4a");
        TIME_BLOCK(ilbl, 0, 1,
                   do { CK(cudaMemset(iacc, 0, nn * sizeof(int)));
                        igemm_rm(h, n, qA, qBt, iacc);
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
        printf("bf16 exact: %d/%d value bits -> %d x %d limbs of %d bits, "
               "%lld int8 GEMMs\n", pb.vA, pb.vB, pb.LA, pb.LB,
               LIMB_BITS_GPU, gb);
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
        printf("fp32 exact: %d/%d value bits -> %d x %d limbs of %d bits, "
               "%lld int8 GEMMs\n", pf.vA, pf.vB, pf.LA, pf.LB,
               LIMB_BITS_GPU, gf);
    }

    double flops = 2.0 * (double)n * n * n;
    double base = res[0].ms;
    printf("\n%-16s %8s %10s %10s %11s\n",
           "method", "GEMMs", "ms", "TFLOP/s", "rel error");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < nr; i++) {
            /* fp64-exact produced hR, so it cannot score itself */
        int is_ref = !strcmp(res[i].name, "fp64-exact");
        printf("%-16s %8lld %10.3f %10.2f ", res[i].name, res[i].gemms,
               res[i].ms, flops / (res[i].ms * 1e-3) / 1e12);
        if (is_ref) printf("%11s  <- EXACT (reference)\n", "-");
        else printf("%11.2e%s\n", res[i].err, res[i].exact ? "  <- EXACT" : "");
    }
    printf("\nbaseline cublas-sgemm = %.3f ms.  Error is against the exact "
           "product decoded to double,\nso an fp32 output cannot score below "
           "~3e-08 however exact its accumulation.\n", base);
    if (n < 2048)
        printf("n=%d is small enough that launch overhead dominates; "
               "use --n 4096 for meaningful throughput.\n", n);

    lt_teardown();
    CB(cublasDestroy(h));
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dAbf); cudaFree(dBbf);
    free(hA); free(hB); free(hC); free(hR);
    return 0;
}
