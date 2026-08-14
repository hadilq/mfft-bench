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
 * tensor-core GPU, 36 int8 GEMMs beat 27 int32 ones.
 *
 * MFFT is implemented (item 11) so it can be compared directly.  At the
 * natural ML limb counts (L≈8–16) it issues more products than schoolbook
 * and is expected to lose; the row must still be present and correct.  The
 * transform uses the post's I_s roots as signed permutations (negacyclic
 * shifts in the power basis), precomputed into a flat op list.
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

/* Returns the cublas status; does not abort.  Probe uses this to decide
 * whether the heuristic algo is actually runnable on this GPU/toolkit. */
static cublasStatus_t igemm_lt_status(int n, const signed char *A,
                                      const signed char *Bt, int *C)
{
    (void)n;
    const int a = 1, b = 1;
    return cublasLtMatmul(g_lt, g_ltdesc, &a, Bt, g_lta, A, g_ltb,
                          &b, C, g_ltc, C, g_ltc, &g_ltheur.algo,
                          g_ltws, g_ltwsz, 0);
}

static void igemm_lt(int n, const signed char *A, const signed char *Bt, int *C)
{
    CB(igemm_lt_status(n, A, Bt, C));
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
        /* verify the Lt path actually runs and agrees with a known answer
         * before trusting it: a heuristic hit is not a guarantee (sm_120 /
         * CUDA 12.4 has been observed to return an algo that later fails
         * with CUBLAS_STATUS_NOT_SUPPORTED). */
        CK(cudaMemset(c, 0, (size_t)n * n * sizeof(int)));
        cublasStatus_t ltst = igemm_lt_status(n, a, b, c);
        int ok = (ltst == CUBLAS_STATUS_SUCCESS &&
                  cudaDeviceSynchronize() == cudaSuccess);
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
                   "failed (status %d, result %d vs %d); falling back to __dp4a.\n",
                   (int)ltst, probe, n);
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

/* Item 4 of PLAN.md: enumerate limb width b in 4..7 x schoolbook /
 * 1-level / 2-level Karatsuba, reject combinations whose intermediate
 * sums leave signed int8, cost as (products) * (same per-product cost),
 * print the table, and report the winner.  Currently the only
 * implemented path is 7-bit schoolbook; the planner still surfaces the
 * negative result that narrower limbs + Karatsuba lose on product count
 * once the int8 constraint is respected.
 *
 * Max limb value after a sum of k terms of (2^b - 1) must stay <= 127.
 */
static void gpu_limb_strategy_table(int vA, int vB)
{
    printf("GPU limb-strategy planner (vbits %d x %d):\n", vA, vB);
    printf("  %-6s %-12s %-8s %-10s %s\n",
           "bits", "strategy", "LA x LB", "products", "fits int8?");
    printf("  ---------------------------------------------------------------\n");

    int best_prods = 1 << 30;
    int best_b = 7;
    const char *best_strat = "schoolbook";

    for (int b = 7; b >= 4; b--) {
        int maxv = (1 << b) - 1;
        int LA = (vA + b - 1) / b;
        int LB = (vB + b - 1) / b;

        /* schoolbook: no intermediate sums of limbs */
        int prods_sb = LA * LB;
        printf("  %-6d %-12s %3d x %-3d %10d  yes\n",
               b, "schoolbook", LA, LB, prods_sb);
        if (prods_sb < best_prods) {
            best_prods = prods_sb;
            best_b = b;
            best_strat = "schoolbook";
        }

        /* 1-level Karatsuba: needs A0+A1 (and B0+B1) to fit in int8.
         * One sum of two maxv values: 2*maxv <= 127. */
        if (2 * maxv <= 127 && LA >= 2 && LB >= 2) {
            /* rough: 3 * ceil(LA/2) * ceil(LB/2)  (classic Karatsuba on
             * the longer side; exact split counts differ by 1 but the
             * order of magnitude is what the table is for) */
            int ha = (LA + 1) / 2, hb = (LB + 1) / 2;
            int prods_k1 = 3 * ha * hb;
            printf("  %-6d %-12s %3d x %-3d %10d  yes (2*%d=%d<=127)\n",
                   b, "karatsuba-1", LA, LB, prods_k1, maxv, 2 * maxv);
            if (prods_k1 < best_prods) {
                best_prods = prods_k1;
                best_b = b;
                best_strat = "karatsuba-1";
            }
        } else {
            printf("  %-6d %-12s %3d x %-3d %10s  no  (2*%d=%d>127)\n",
                   b, "karatsuba-1", LA, LB, "-", maxv, 2 * maxv);
        }

        /* 2-level: four-way sums, 4*maxv <= 127 */
        if (4 * maxv <= 127 && LA >= 4 && LB >= 4) {
            int qa = (LA + 3) / 4, qb = (LB + 3) / 4;
            int prods_k2 = 9 * qa * qb; /* 3^2 */
            printf("  %-6d %-12s %3d x %-3d %10d  yes (4*%d=%d<=127)\n",
                   b, "karatsuba-2", LA, LB, prods_k2, maxv, 4 * maxv);
            if (prods_k2 < best_prods) {
                best_prods = prods_k2;
                best_b = b;
                best_strat = "karatsuba-2";
            }
        } else {
            printf("  %-6d %-12s %3d x %-3d %10s  no  (4*%d=%d>127)\n",
                   b, "karatsuba-2", LA, LB, "-", maxv, 4 * maxv);
        }
    }
    printf("  winner: %d-bit %s (%d products).  "
           "Runtime path is fixed 7-bit schoolbook (Karatsuba never wins "
           "under the int8 sum constraint for float embeddings).\n\n",
           best_b, best_strat, best_prods);
}

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

/* Faithful-rounding plan (item 5).
 *
 * The exact path keeps every value bit (significand + full exponent spread).
 * For a correctly-rounded binary output we only need
 *   sig_out + ceil(log2 n) + guard
 * bits of the *product*.  Bits below that cannot move the rounded result
 * unless massive cancellation makes |C_ij| far smaller than the typical
 * magnitude.  We drop low-order input limbs to meet that product budget
 * while minimising LA*LB, and rely on the timed verification against the
 * full exact reference to catch any residual mismatch.
 *
 * guard = 4 is conservative (covers the length-n carry and a rounding bit).
 * Cancellation: if a result is so small that the discarded tail could
 * matter, the comparison against the exact reference will fail the row;
 * a per-entry fallback is left for a later tightening pass. */
static int ceil_log2_int(int n)
{
    int k = 0, v = n - 1;
    while (v > 0) { v >>= 1; k++; }
    return k;
}

static LimbPlan plan_limbs_faithful(LimbPlan full, int sig_out, int n)
{
    int need = sig_out + ceil_log2_int(n) + 4;
    int have = full.vA + full.vB;
    LimbPlan p = full;
    if (have <= need) return p;   /* nothing to drop */

    int drop = have - need;       /* bits we may discard from the bottom */
    int dropA = 0, dropB = 0;
    /* Prefer to shrink the wider side so LA and LB stay balanced. */
    while (dropA + dropB < drop) {
        int remA = p.vA - dropA, remB = p.vB - dropB;
        if (remA <= LIMB_BITS_GPU && remB <= LIMB_BITS_GPU) break;
        if (remA >= remB && remA > LIMB_BITS_GPU) dropA++;
        else if (remB > LIMB_BITS_GPU) dropB++;
        else break;
    }
    p.vA = full.vA - dropA;
    p.vB = full.vB - dropB;
    if (p.vA < 1) p.vA = 1;
    if (p.vB < 1) p.vB = 1;
    p.LA = (p.vA + LIMB_BITS_GPU - 1) / LIMB_BITS_GPU;
    p.LB = (p.vB + LIMB_BITS_GPU - 1) / LIMB_BITS_GPU;
    return p;
}

/* Runs the schoolbook limb convolution as int8 GEMMs.
 * If faith.LA/LB are smaller than full.LA/LB, only the *high* limbs are
 * multiplied (low-order limbs dropped) -- the faithful-rounding path.
 * Encode always uses the full plan so limb indices keep their significance. */
static double exact_float_gemm(cublasHandle_t h, int n, const float *dA,
                               const float *dB, float *dC, LimbPlan full,
                               LimbPlan faith, int reps, long long *gemms)
{
    size_t nn = (size_t)n * n;
    signed char *pA, *pB, *pBt;
    int *acc;
    int planes = full.LA + full.LB - 1;
    int u0 = full.LA - faith.LA; if (u0 < 0) u0 = 0;
    int v0 = full.LB - faith.LB; if (v0 < 0) v0 = 0;

    double bound = (double)(full.LA < full.LB ? full.LA : full.LB) * n * 16384.0;
    if (bound >= 2147483648.0) {
        fprintf(stderr, "digit planes would overflow int32 at n=%d (bound %.0f)\n",
                n, bound);
        exit(1);
    }

    CK(cudaMalloc(&pA, nn * full.LA));
    CK(cudaMalloc(&pB, nn * full.LB));
    CK(cudaMalloc(&pBt, nn * full.LB));
    CK(cudaMalloc(&acc, nn * planes * sizeof(int)));

    int blk = 256, g = grid_for(nn, blk);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float best = 1e30f;

    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(t0));
        k_encode<<<g, blk>>>(pA, dA, nn, full.LA, full.SA, full.sig);
        k_encode<<<g, blk>>>(pB, dB, nn, full.LB, full.SB, full.sig);
        dim3 tb(16, 16), tg((n + 15) / 16, (n + 15) / 16);
        for (int w = 0; w < full.LB; w++)
            k_transpose<<<tg, tb>>>(pBt + (size_t)w * nn,
                                    pB + (size_t)w * nn, n);
        CK(cudaMemset(acc, 0, nn * planes * sizeof(int)));
        for (int u = u0; u < full.LA; u++)
            for (int v = v0; v < full.LB; v++)
                igemm_rm(h, n, pA + (size_t)u * nn, pBt + (size_t)v * nn,
                         acc + (size_t)(u + v) * nn);
        k_decode<<<g, blk>>>(dC, acc, nn, planes, full.SA + full.SB);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    *gemms = (long long)(full.LA - u0) * (full.LB - v0);

    cudaFree(pA); cudaFree(pB); cudaFree(pBt); cudaFree(acc);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best;
}

/* Exact fp64 product through the same int8 limb machinery.  A 53-bit
 * significand plus the exponent spread needs ~10 limbs, so ~100 int8 GEMMs
 * -- but on a consumer GPU, where fp64 runs at 1/64 of fp32, that can still
 * come out ahead of a single cublasDgemm.  That comparison is the point. */
static double exact_double_gemm(cublasHandle_t h, int n, const double *dA,
                                const double *dB, double *dC, LimbPlan full,
                                LimbPlan faith, int reps, long long *gemms)
{
    size_t nn = (size_t)n * n;
    signed char *pA, *pB, *pBt;
    int *acc;
    int planes = full.LA + full.LB - 1;
    int u0 = full.LA - faith.LA; if (u0 < 0) u0 = 0;
    int v0 = full.LB - faith.LB; if (v0 < 0) v0 = 0;

    double bound = (double)(full.LA < full.LB ? full.LA : full.LB) * n * 16384.0;
    if (bound >= 2147483648.0) {
        fprintf(stderr, "fp64 digit planes would overflow int32 at n=%d\n", n);
        *gemms = 0;
        return 0.0;
    }

    CK(cudaMalloc(&pA, nn * full.LA));
    CK(cudaMalloc(&pB, nn * full.LB));
    CK(cudaMalloc(&pBt, nn * full.LB));
    CK(cudaMalloc(&acc, nn * planes * sizeof(int)));

    int blk = 256, g = grid_for(nn, blk);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float best = 1e30f;

    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(t0));
        k_encode_d<<<g, blk>>>(pA, dA, nn, full.LA, full.SA);
        k_encode_d<<<g, blk>>>(pB, dB, nn, full.LB, full.SB);
        dim3 tb(16, 16), tg((n + 15) / 16, (n + 15) / 16);
        for (int w = 0; w < full.LB; w++)
            k_transpose<<<tg, tb>>>(pBt + (size_t)w * nn,
                                    pB + (size_t)w * nn, n);
        CK(cudaMemset(acc, 0, nn * planes * sizeof(int)));
        for (int u = u0; u < full.LA; u++)
            for (int v = v0; v < full.LB; v++)
                igemm_rm(h, n, pA + (size_t)u * nn, pBt + (size_t)v * nn,
                         acc + (size_t)(u + v) * nn);
        k_decode_d<<<g, blk>>>(dC, acc, nn, planes, full.SA + full.SB);
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    *gemms = (long long)(full.LA - u0) * (full.LB - v0);

    cudaFree(pA); cudaFree(pB); cudaFree(pBt); cudaFree(acc);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best;
}

/* ------------------------------------------------------------------ *
 * GPU Strassen (item 10): recursive baseline that bottoms out in
 * cublasSgemm / cublasDgemm.  Default cutoff 2048: at n=4096 one recursion
 * level (7 leaf GEMMs of size 2048) instead of three levels / 343 leaves.
 * Override with --cutoff.
 * ------------------------------------------------------------------ */
static int g_strassen_cutoff = 2048;

__global__ void k_add_f(float *C, const float *A, const float *B, size_t nn, float sb)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    C[i] = A[i] + sb * B[i];
}
__global__ void k_add_d(double *C, const double *A, const double *B, size_t nn, double sb)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    C[i] = A[i] + sb * B[i];
}

/* Contiguous quadrant extract / insert via a single 2D memcpy. */
static void copy_quad_f(float *dst, const float *src, int n, int r0, int c0, int h)
{
    CK(cudaMemcpy2D(dst, h * sizeof(float),
                    src + (size_t)r0 * n + c0, n * sizeof(float),
                    h * sizeof(float), h, cudaMemcpyDeviceToDevice));
}
static void paste_quad_f(float *dst, const float *src, int n, int r0, int c0, int h)
{
    CK(cudaMemcpy2D(dst + (size_t)r0 * n + c0, n * sizeof(float),
                    src, h * sizeof(float),
                    h * sizeof(float), h, cudaMemcpyDeviceToDevice));
}
static void copy_quad_d(double *dst, const double *src, int n, int r0, int c0, int h)
{
    CK(cudaMemcpy2D(dst, h * sizeof(double),
                    src + (size_t)r0 * n + c0, n * sizeof(double),
                    h * sizeof(double), h, cudaMemcpyDeviceToDevice));
}
static void paste_quad_d(double *dst, const double *src, int n, int r0, int c0, int h)
{
    CK(cudaMemcpy2D(dst + (size_t)r0 * n + c0, n * sizeof(double),
                    src, h * sizeof(double),
                    h * sizeof(double), h, cudaMemcpyDeviceToDevice));
}

static void strassen_f(cublasHandle_t h, int n, const float *A, const float *B,
                       float *C)
{
    if (n <= g_strassen_cutoff || (n & 1)) {
        const float alpha = 1.0f, beta = 0.0f;
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                       &alpha, B, n, A, n, &beta, C, n));
        return;
    }
    int hs = n / 2;
    size_t hn = (size_t)hs * hs;
    float *buf;
    CK(cudaMalloc(&buf, hn * 13 * sizeof(float)));
    float *A11 = buf, *A12 = buf + hn, *A21 = buf + 2 * hn, *A22 = buf + 3 * hn;
    float *B11 = buf + 4 * hn, *B12 = buf + 5 * hn, *B21 = buf + 6 * hn, *B22 = buf + 7 * hn;
    float *T1 = buf + 8 * hn, *T2 = buf + 9 * hn;
    float *M1 = buf + 10 * hn, *M2 = buf + 11 * hn, *M3 = buf + 12 * hn;
    float *Mextra;
    CK(cudaMalloc(&Mextra, hn * 4 * sizeof(float)));
    float *M4 = Mextra, *M5 = Mextra + hn, *M6 = Mextra + 2 * hn, *M7 = Mextra + 3 * hn;

    copy_quad_f(A11, A, n, 0, 0, hs);  copy_quad_f(A12, A, n, 0, hs, hs);
    copy_quad_f(A21, A, n, hs, 0, hs); copy_quad_f(A22, A, n, hs, hs, hs);
    copy_quad_f(B11, B, n, 0, 0, hs);  copy_quad_f(B12, B, n, 0, hs, hs);
    copy_quad_f(B21, B, n, hs, 0, hs); copy_quad_f(B22, B, n, hs, hs, hs);

    int blk = 256, g = grid_for(hn, blk);
    k_add_f<<<g, blk>>>(T1, A11, A22, hn, 1.0f);
    k_add_f<<<g, blk>>>(T2, B11, B22, hn, 1.0f);
    strassen_f(h, hs, T1, T2, M1);
    k_add_f<<<g, blk>>>(T1, A21, A22, hn, 1.0f);
    strassen_f(h, hs, T1, B11, M2);
    k_add_f<<<g, blk>>>(T2, B12, B22, hn, -1.0f);
    strassen_f(h, hs, A11, T2, M3);
    k_add_f<<<g, blk>>>(T2, B21, B11, hn, -1.0f);
    strassen_f(h, hs, A22, T2, M4);
    k_add_f<<<g, blk>>>(T1, A11, A12, hn, 1.0f);
    strassen_f(h, hs, T1, B22, M5);
    k_add_f<<<g, blk>>>(T1, A21, A11, hn, -1.0f);
    k_add_f<<<g, blk>>>(T2, B11, B12, hn, 1.0f);
    strassen_f(h, hs, T1, T2, M6);
    k_add_f<<<g, blk>>>(T1, A12, A22, hn, -1.0f);
    k_add_f<<<g, blk>>>(T2, B21, B22, hn, 1.0f);
    strassen_f(h, hs, T1, T2, M7);

    k_add_f<<<g, blk>>>(T1, M1, M4, hn, 1.0f);
    k_add_f<<<g, blk>>>(T1, T1, M5, hn, -1.0f);
    k_add_f<<<g, blk>>>(T1, T1, M7, hn, 1.0f);
    paste_quad_f(C, T1, n, 0, 0, hs);
    k_add_f<<<g, blk>>>(T1, M3, M5, hn, 1.0f);
    paste_quad_f(C, T1, n, 0, hs, hs);
    k_add_f<<<g, blk>>>(T1, M2, M4, hn, 1.0f);
    paste_quad_f(C, T1, n, hs, 0, hs);
    k_add_f<<<g, blk>>>(T1, M1, M2, hn, -1.0f);
    k_add_f<<<g, blk>>>(T1, T1, M3, hn, 1.0f);
    k_add_f<<<g, blk>>>(T1, T1, M6, hn, 1.0f);
    paste_quad_f(C, T1, n, hs, hs, hs);

    cudaFree(buf); cudaFree(Mextra);
}

static void strassen_d(cublasHandle_t h, int n, const double *A, const double *B,
                       double *C)
{
    if (n <= g_strassen_cutoff || (n & 1)) {
        const double alpha = 1.0, beta = 0.0;
        CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                       &alpha, B, n, A, n, &beta, C, n));
        return;
    }
    int hs = n / 2;
    size_t hn = (size_t)hs * hs;
    double *buf;
    CK(cudaMalloc(&buf, hn * 13 * sizeof(double)));
    double *A11 = buf, *A12 = buf + hn, *A21 = buf + 2 * hn, *A22 = buf + 3 * hn;
    double *B11 = buf + 4 * hn, *B12 = buf + 5 * hn, *B21 = buf + 6 * hn, *B22 = buf + 7 * hn;
    double *T1 = buf + 8 * hn, *T2 = buf + 9 * hn;
    double *M1 = buf + 10 * hn, *M2 = buf + 11 * hn, *M3 = buf + 12 * hn;
    double *Mextra;
    CK(cudaMalloc(&Mextra, hn * 4 * sizeof(double)));
    double *M4 = Mextra, *M5 = Mextra + hn, *M6 = Mextra + 2 * hn, *M7 = Mextra + 3 * hn;

    copy_quad_d(A11, A, n, 0, 0, hs);  copy_quad_d(A12, A, n, 0, hs, hs);
    copy_quad_d(A21, A, n, hs, 0, hs); copy_quad_d(A22, A, n, hs, hs, hs);
    copy_quad_d(B11, B, n, 0, 0, hs);  copy_quad_d(B12, B, n, 0, hs, hs);
    copy_quad_d(B21, B, n, hs, 0, hs); copy_quad_d(B22, B, n, hs, hs, hs);

    int blk = 256, g = grid_for(hn, blk);
    k_add_d<<<g, blk>>>(T1, A11, A22, hn, 1.0);
    k_add_d<<<g, blk>>>(T2, B11, B22, hn, 1.0);
    strassen_d(h, hs, T1, T2, M1);
    k_add_d<<<g, blk>>>(T1, A21, A22, hn, 1.0);
    strassen_d(h, hs, T1, B11, M2);
    k_add_d<<<g, blk>>>(T2, B12, B22, hn, -1.0);
    strassen_d(h, hs, A11, T2, M3);
    k_add_d<<<g, blk>>>(T2, B21, B11, hn, -1.0);
    strassen_d(h, hs, A22, T2, M4);
    k_add_d<<<g, blk>>>(T1, A11, A12, hn, 1.0);
    strassen_d(h, hs, T1, B22, M5);
    k_add_d<<<g, blk>>>(T1, A21, A11, hn, -1.0);
    k_add_d<<<g, blk>>>(T2, B11, B12, hn, 1.0);
    strassen_d(h, hs, T1, T2, M6);
    k_add_d<<<g, blk>>>(T1, A12, A22, hn, -1.0);
    k_add_d<<<g, blk>>>(T2, B21, B22, hn, 1.0);
    strassen_d(h, hs, T1, T2, M7);

    k_add_d<<<g, blk>>>(T1, M1, M4, hn, 1.0);
    k_add_d<<<g, blk>>>(T1, T1, M5, hn, -1.0);
    k_add_d<<<g, blk>>>(T1, T1, M7, hn, 1.0);
    paste_quad_d(C, T1, n, 0, 0, hs);
    k_add_d<<<g, blk>>>(T1, M3, M5, hn, 1.0);
    paste_quad_d(C, T1, n, 0, hs, hs);
    k_add_d<<<g, blk>>>(T1, M2, M4, hn, 1.0);
    paste_quad_d(C, T1, n, hs, 0, hs);
    k_add_d<<<g, blk>>>(T1, M1, M2, hn, -1.0);
    k_add_d<<<g, blk>>>(T1, T1, M3, hn, 1.0);
    k_add_d<<<g, blk>>>(T1, T1, M6, hn, 1.0);
    paste_quad_d(C, T1, n, hs, hs, hs);

    cudaFree(buf); cudaFree(Mextra);
}

/* Ozaki I: accumulate an int32 GEMM result into a double matrix with a
 * constant scale (product of the two slice scales). */
__global__ void k_axpy_i32_scale(double *C, const int *acc, double scale,
                                 size_t nn)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    C[i] += scale * (double)acc[i];
}

/* Ozaki I residual extraction: one global scale, int8 quantize, subtract. */
static void ozaki_split_host(signed char *planes, double *scales,
                             const double *X, size_t nn, int s)
{
    double *resid = (double *)malloc(nn * sizeof(double));
    if (!resid) { fprintf(stderr, "ozaki split oom\n"); exit(1); }
    memcpy(resid, X, nn * sizeof(double));
    for (int k = 0; k < s; k++) {
        double mx = 0.0;
        for (size_t i = 0; i < nn; i++) {
            double a = fabs(resid[i]);
            if (a > mx) mx = a;
        }
        scales[k] = mx > 0.0 ? mx / 127.0 : 1.0;
        for (size_t i = 0; i < nn; i++) {
            int q = (int)lrint(resid[i] / scales[k]);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            planes[(size_t)k * nn + i] = (signed char)q;
            resid[i] -= (double)q * scales[k];
        }
    }
    free(resid);
}

/* Ozaki scheme I: split each FP64 matrix into `s` int8 slices so every
 * pairwise product is exact in int32 accumulation, then reconstruct in
 * double.  Cost is s^2 int8 GEMMs.  Accuracy improves with s; s≈7 is the
 * usual target for near-dgemm quality on unit-scale data. */
static double ozaki_double_gemm(cublasHandle_t h, int n,
                                const double *hA, const double *hB,
                                double *dC, int s, int reps, long long *gemms)
{
    size_t nn = (size_t)n * n;
    signed char *hQa = (signed char *)malloc(nn * (size_t)s);
    signed char *hQb = (signed char *)malloc(nn * (size_t)s);
    double *sa = (double *)malloc((size_t)s * sizeof(double));
    double *sb = (double *)malloc((size_t)s * sizeof(double));
    if (!hQa || !hQb || !sa || !sb) {
        fprintf(stderr, "ozaki host oom\n"); exit(1);
    }
    ozaki_split_host(hQa, sa, hA, nn, s);
    ozaki_split_host(hQb, sb, hB, nn, s);

    signed char *dQa, *dQb, *dQbt;
    int *dAcc;
    CK(cudaMalloc(&dQa, nn * (size_t)s));
    CK(cudaMalloc(&dQb, nn * (size_t)s));
    CK(cudaMalloc(&dQbt, nn * (size_t)s));
    CK(cudaMalloc(&dAcc, nn * sizeof(int)));
    CK(cudaMemcpy(dQa, hQa, nn * (size_t)s, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dQb, hQb, nn * (size_t)s, cudaMemcpyHostToDevice));

    int blk = 256, g = grid_for(nn, blk);
    dim3 tb(16, 16), tg((n + 15) / 16, (n + 15) / 16);
    for (int w = 0; w < s; w++)
        k_transpose<<<tg, tb>>>(dQbt + (size_t)w * nn, dQb + (size_t)w * nn, n);

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float best = 1e30f;

    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(t0));
        CK(cudaMemset(dC, 0, nn * sizeof(double)));
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                CK(cudaMemset(dAcc, 0, nn * sizeof(int)));
                igemm_rm(h, n, dQa + (size_t)i * nn,
                         dQbt + (size_t)j * nn, dAcc);
                double scale = sa[i] * sb[j];
                k_axpy_i32_scale<<<g, blk>>>(dC, dAcc, scale, nn);
            }
        }
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    *gemms = (long long)s * s;

    cudaFree(dQa); cudaFree(dQb); cudaFree(dQbt); cudaFree(dAcc);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(hQa); free(hQb); free(sa); free(sb);
    return best;
}

/* ------------------------------------------------------------------ *
 * GPU MFFT (item 11)
 *
 * Same algorithm as src/mfft.c:
 *   pack S limbs per coefficient, transform length NB = 2L/S over the
 *   ring Z[y]/(y^K+1) with K = 2S, omega = y^g.
 * Roots of unity are the post's I_s: each power is a signed permutation,
 * which in the power basis is a negacyclic shift.  The fused op list is
 * built once on the host (Gentleman-Sande forward / Cooley-Tukey inverse)
 * and uploaded; the timed path never recomputes roots.
 *
 * L is padded to the next power of two; matrix n is already a multiple of
 * 16 for int8 alignment.  Rectangular support = pad each side (same idea).
 * ------------------------------------------------------------------ */
enum { MFFT_OP_START = 0, MFFT_OP_STEP = 1, MFFT_OP_END = 2 };
typedef struct { int32_t u, v; int16_t sign, mode; } MfftOp;

typedef struct {
    int L, S, NB, K, g;
    long nfw, niv;
    MfftOp *fwops, *ivops;           /* host */
} MfftPlanGpu;

static int mfft_ilog2(int x) { int r = 0; while ((1 << r) < x) r++; return r; }
static int mfft_next_pow2(int x)
{
    if (x < 2) return 2;
    if ((x & (x - 1)) == 0) return x;
    return 1 << mfft_ilog2(x);
}

static int mfft_plan_init_gpu(MfftPlanGpu *p, int L_raw)
{
    int L = mfft_next_pow2(L_raw < 2 ? 2 : L_raw);
    int l = mfft_ilog2(L);
    int sigma = l / 2;
    if (sigma < 1) sigma = 1;
    if (sigma > l) return -1;
    int S = 1 << sigma;
    int NB = 2 * L / S;
    int K = 2 * S;
    if (2 * S * S < L) return -1;
    if ((2 * K) % NB) return -1;
    if (NB < 2) return -1;
    p->L = L; p->S = S; p->NB = NB; p->K = K; p->g = 2 * K / NB;
    p->nfw = p->niv = 0;
    p->fwops = p->ivops = NULL;
    return 0;
}

static MfftOp *mfft_build_ops(int NB, int K, int g, int inverse, long *nops_out)
{
    int mod = 2 * K;
    long cap = 0;
    for (int len = 2; len <= NB; len <<= 1) cap += (long)NB / 2 * (K + K);
    MfftOp *ops = (MfftOp *)malloc((size_t)cap * sizeof(MfftOp));
    char *seen = (char *)malloc((size_t)K);
    if (!ops || !seen) { free(ops); free(seen); *nops_out = 0; return NULL; }
    long m = 0;
    int lens[32], nl = 0;
    if (inverse) for (int len = 2; len <= NB; len <<= 1) lens[nl++] = len;
    else         for (int len = NB; len >= 2; len >>= 1) lens[nl++] = len;

    for (int li = 0; li < nl; li++) {
        int len = lens[li], h = len >> 1, step = NB / len;
        for (int j0 = 0; j0 < NB; j0 += len)
            for (int t = 0; t < h; t++) {
                int e = (int)(((long long)g * t * step) % mod);
                if (inverse) e = (mod - e) % mod;
                int pb = j0 + t, qb = j0 + t + h;
                memset(seen, 0, (size_t)K);
                for (int c0 = 0; c0 < K; c0++) {
                    if (seen[c0]) continue;
                    int c = c0;
                    seen[c] = 1;
                    ops[m].u = pb * K + c0; ops[m].v = qb * K + c0;
                    ops[m].sign = 1; ops[m].mode = MFFT_OP_START; m++;
                    for (;;) {
                        int tgt = c + e, w = 0;
                        while (tgt >= K) { tgt -= K; w++; }
                        int sg = (w & 1) ? -1 : 1;
                        if (tgt == c0) {
                            ops[m].u = pb * K + tgt; ops[m].v = qb * K + tgt;
                            ops[m].sign = (int16_t)sg; ops[m].mode = MFFT_OP_END; m++;
                            break;
                        }
                        seen[tgt] = 1;
                        ops[m].u = pb * K + tgt; ops[m].v = qb * K + tgt;
                        ops[m].sign = (int16_t)sg; ops[m].mode = MFFT_OP_STEP; m++;
                        c = tgt;
                    }
                }
            }
    }
    free(seen);
    *nops_out = m;
    return ops;
}

static int mfft_plan_build_ops(MfftPlanGpu *p)
{
    p->fwops = mfft_build_ops(p->NB, p->K, p->g, 0, &p->nfw);
    p->ivops = mfft_build_ops(p->NB, p->K, p->g, 1, &p->niv);
    return (p->fwops && p->ivops) ? 0 : -1;
}

static void mfft_plan_free_gpu(MfftPlanGpu *p)
{
    free(p->fwops); free(p->ivops);
    p->fwops = p->ivops = NULL;
}

/* One fused FFT op on nn-length int32 coefficient planes. */
__global__ void k_mfft_op32(int32_t *x, size_t nn, int32_t u, int32_t v,
                            int16_t sign, int16_t mode, int32_t *cur)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    int32_t *up = x + (size_t)u * nn;
    int32_t *vp = x + (size_t)v * nn;
    if (mode == MFFT_OP_START) {
        int32_t a = up[i], b = vp[i];
        up[i] = a + b; cur[i] = a - b;
    } else if (mode == MFFT_OP_STEP) {
        int32_t a = up[i], b = vp[i], nx = a - b;
        up[i] = a + b;
        vp[i] = (sign > 0) ? cur[i] : -cur[i];
        cur[i] = nx;
    } else {
        vp[i] = (sign > 0) ? cur[i] : -cur[i];
    }
}

/* Inverse ops on int64 (after pointwise). */
__global__ void k_mfft_op64(long long *x, size_t nn, int32_t u, int32_t v,
                            int16_t sign, int16_t mode, long long *cur)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    long long *up = x + (size_t)u * nn;
    long long *vp = x + (size_t)v * nn;
    if (mode == MFFT_OP_START) {
        cur[i] = vp[i];
    } else if (mode == MFFT_OP_STEP) {
        long long nx = vp[i], tv = (sign > 0) ? cur[i] : -cur[i], a = up[i];
        up[i] = a + tv; vp[i] = a - tv; cur[i] = nx;
    } else {
        long long tv = (sign > 0) ? cur[i] : -cur[i], a = up[i];
        up[i] = a + tv; vp[i] = a - tv;
    }
}

/* Fold inverse-FFT output: acc[w] += Ch[b,c] / NB for w = b*S + c. */
__global__ void k_mfft_fold(int *acc, const long long *Ch, size_t nn,
                            int NB, int K, int S, int planes)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    for (int b = 0; b < NB; b++) {
        for (int c = 0; c < K; c++) {
            int w = b * S + c;
            if (w >= planes) continue;
            long long v = Ch[((size_t)b * K + c) * nn + i];
            acc[(size_t)w * nn + i] += (int)(v / NB);
        }
    }
}

/* Integer GEMM for MFFT pointwise: C += sgn * A * B, int32 A/B, int64 C.
 * 32×32 tile, each thread owns a 4×4 register block (8×8 threads). */
__global__ void k_igemm32(long long *C, const int32_t *A, const int32_t *B,
                          int n, int sgn)
{
    const int TS = 32, RT = 4;
    __shared__ int As[32][32], Bs[32][32];
    int ty = (int)threadIdx.y, tx = (int)threadIdx.x;
    int row0 = (int)blockIdx.y * TS + ty * RT;
    int col0 = (int)blockIdx.x * TS + tx * RT;
    long long sum[4][4];
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < RT; j++) sum[i][j] = 0;

    for (int kk = 0; kk < n; kk += TS) {
        #pragma unroll
        for (int i = 0; i < RT; i++)
            #pragma unroll
            for (int j = 0; j < RT; j++) {
                int r = ty * RT + i, c = tx * RT + j;
                int ar = (int)blockIdx.y * TS + r, ac = kk + c;
                int br = kk + r, bc = (int)blockIdx.x * TS + c;
                As[r][c] = (ar < n && ac < n) ? A[(size_t)ar * n + ac] : 0;
                Bs[r][c] = (br < n && bc < n) ? B[(size_t)br * n + bc] : 0;
            }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TS; t++) {
            int a0 = As[ty*RT+0][t], a1 = As[ty*RT+1][t];
            int a2 = As[ty*RT+2][t], a3 = As[ty*RT+3][t];
            int b0 = Bs[t][tx*RT+0], b1 = Bs[t][tx*RT+1];
            int b2 = Bs[t][tx*RT+2], b3 = Bs[t][tx*RT+3];
            sum[0][0] += (long long)a0*b0; sum[0][1] += (long long)a0*b1;
            sum[0][2] += (long long)a0*b2; sum[0][3] += (long long)a0*b3;
            sum[1][0] += (long long)a1*b0; sum[1][1] += (long long)a1*b1;
            sum[1][2] += (long long)a1*b2; sum[1][3] += (long long)a1*b3;
            sum[2][0] += (long long)a2*b0; sum[2][1] += (long long)a2*b1;
            sum[2][2] += (long long)a2*b2; sum[2][3] += (long long)a2*b3;
            sum[3][0] += (long long)a3*b0; sum[3][1] += (long long)a3*b1;
            sum[3][2] += (long long)a3*b2; sum[3][3] += (long long)a3*b3;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < RT; j++) {
            int r = row0 + i, c = col0 + j;
            if (r < n && c < n) {
                if (sgn > 0) C[(size_t)r * n + c] += sum[i][j];
                else         C[(size_t)r * n + c] -= sum[i][j];
            }
        }
}

static void igemm32_rm(long long *C, const int32_t *A, const int32_t *B,
                       int n, int sgn)
{
    dim3 blk(8, 8);
    dim3 grd((n + 31) / 32, (n + 31) / 32);
    k_igemm32<<<grd, blk>>>(C, A, B, n, sgn);
}

/* Entire op list in one launch: each thread owns one matrix element and
 * walks all ops with a register-local carry (no global carry buffer). */
__global__ void k_mfft_run32(int32_t *x, size_t nn, const MfftOp *ops, int nops)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    int32_t cur = 0;
    for (int o = 0; o < nops; o++) {
        int32_t *u = x + (size_t)ops[o].u * nn + i;
        int32_t *v = x + (size_t)ops[o].v * nn + i;
        int16_t mode = ops[o].mode, sign = ops[o].sign;
        if (mode == MFFT_OP_START) {
            int32_t a = *u, b = *v;
            *u = a + b; cur = a - b;
        } else if (mode == MFFT_OP_STEP) {
            int32_t a = *u, b = *v, nx = a - b;
            *u = a + b;
            *v = (sign > 0) ? cur : -cur;
            cur = nx;
        } else {
            *v = (sign > 0) ? cur : -cur;
        }
    }
}
__global__ void k_mfft_run64(long long *x, size_t nn, const MfftOp *ops, int nops)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nn) return;
    long long cur = 0;
    for (int o = 0; o < nops; o++) {
        long long *u = x + (size_t)ops[o].u * nn + i;
        long long *v = x + (size_t)ops[o].v * nn + i;
        int16_t mode = ops[o].mode, sign = ops[o].sign;
        if (mode == MFFT_OP_START) {
            cur = *v;
        } else if (mode == MFFT_OP_STEP) {
            long long nx = *v, tv = (sign > 0) ? cur : -cur, a = *u;
            *u = a + tv; *v = a - tv; cur = nx;
        } else {
            long long tv = (sign > 0) ? cur : -cur, a = *u;
            *u = a + tv; *v = a - tv;
        }
    }
}
__global__ void k_zero_i32(int32_t *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0;
}
__global__ void k_zero_i64(long long *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0;
}
__global__ void k_copy_i8_to_i32(int32_t *dst, const signed char *src, size_t nn)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < nn) dst[i] = (int32_t)src[i];
}

/* Expand signed-char limb planes into the MFFT coefficient buffer layout. */
static void mfft_pack_limbs(int32_t *Ah, const signed char *pA, size_t nn,
                            int L_pad, int S, int K)
{
    int blk = 256;
    size_t tot = (size_t)(2 * L_pad / S) * K * nn; /* NB * K * nn */
    int g = grid_for(tot, blk);
    k_zero_i32<<<g, blk>>>(Ah, tot);
    int nblocks = L_pad / S;
    for (int b = 0; b < nblocks; b++) {
        for (int c = 0; c < S; c++) {
            int limb = b * S + c;
            if (limb >= L_pad) continue;
            size_t off = ((size_t)b * K + c) * nn;
            k_copy_i8_to_i32<<<grid_for(nn, blk), blk>>>(
                Ah + off, pA + (size_t)limb * nn, nn);
        }
    }
}

/* Timed GPU MFFT on float inputs: encode -> pack -> FFT -> pointwise
 * (double GEMMs of coefficient planes) -> inverse FFT -> fold -> decode.
 *
 * Pointwise uses cublasDgemm: float was not exact past 2^24, which at n=256
 * already corrupted the integer products.  Double keeps exact integers up
 * to 2^53, enough for the coefficient magnitudes we see at L<=16.
 *
 * Forward transforms on A and B are sequential (shared carry buffer); a
 * concurrent Ah/Bh launch would race on the carry plane. */
static double mfft_float_gemm(cublasHandle_t h, int n, const float *dA,
                              const float *dB, float *dC, LimbPlan full,
                              int reps, long long *gemms)
{
    MfftPlanGpu plan;
    if (mfft_plan_init_gpu(&plan, full.LA > full.LB ? full.LA : full.LB) != 0 ||
        mfft_plan_build_ops(&plan) != 0) {
        fprintf(stderr, "mfft plan failed for L~%d\n", full.LA);
        *gemms = 0;
        return 0.0;
    }
    int L = plan.L, S = plan.S, NB = plan.NB, K = plan.K;
    size_t nn = (size_t)n * n;
    size_t tot = (size_t)NB * K * nn;
    long long nprod = (long long)NB * K * K;
    int planes = 2 * L - 1;

    signed char *pA, *pB;
    int32_t *Ah, *Bh;
    long long *Ch;
    int *acc;
    MfftOp *d_fwops, *d_ivops;
    (void)h; /* pointwise is custom int32 GEMM, not cuBLAS */

    CK(cudaMalloc(&pA, nn * (size_t)L));
    CK(cudaMalloc(&pB, nn * (size_t)L));
    CK(cudaMalloc(&Ah, tot * sizeof(int32_t)));
    CK(cudaMalloc(&Bh, tot * sizeof(int32_t)));
    CK(cudaMalloc(&Ch, tot * sizeof(long long)));
    CK(cudaMalloc(&acc, (size_t)planes * nn * sizeof(int)));
    CK(cudaMalloc(&d_fwops, (size_t)plan.nfw * sizeof(MfftOp)));
    CK(cudaMalloc(&d_ivops, (size_t)plan.niv * sizeof(MfftOp)));
    CK(cudaMemcpy(d_fwops, plan.fwops, (size_t)plan.nfw * sizeof(MfftOp),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ivops, plan.ivops, (size_t)plan.niv * sizeof(MfftOp),
                  cudaMemcpyHostToDevice));

    int blk = 256, gnn = grid_for(nn, blk);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float best = 1e30f;

    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(t0));
        /* 1. encode real limbs; padded high planes stay zero */
        CK(cudaMemset(pA, 0, nn * (size_t)L));
        CK(cudaMemset(pB, 0, nn * (size_t)L));
        k_encode<<<gnn, blk>>>(pA, dA, nn, full.LA, full.SA, full.sig);
        k_encode<<<gnn, blk>>>(pB, dB, nn, full.LB, full.SB, full.sig);

        /* 2. pack into coefficient buffers (upper half of each block zero) */
        mfft_pack_limbs(Ah, pA, nn, L, S, K);
        mfft_pack_limbs(Bh, pB, nn, L, S, K);

        /* 3. forward transforms — one launch each (fused op list) */
        k_mfft_run32<<<gnn, blk>>>(Ah, nn, d_fwops, (int)plan.nfw);
        k_mfft_run32<<<gnn, blk>>>(Bh, nn, d_fwops, (int)plan.nfw);

        /* 4. pointwise: NB * K^2 int32 GEMMs with int64 accumulate */
        k_zero_i64<<<grid_for(tot, blk), blk>>>(Ch, tot);
        for (int b = 0; b < NB; b++) {
            for (int c1 = 0; c1 < K; c1++) {
                const int32_t *Ap = Ah + ((size_t)b * K + c1) * nn;
                for (int c2 = 0; c2 < K; c2++) {
                    int t = c1 + c2, sgn = 1;
                    if (t >= K) { t -= K; sgn = -1; }
                    igemm32_rm(Ch + ((size_t)b * K + t) * nn, Ap,
                               Bh + ((size_t)b * K + c2) * nn, n, sgn);
                }
            }
        }

        /* 5. inverse transform */
        k_mfft_run64<<<gnn, blk>>>(Ch, nn, d_ivops, (int)plan.niv);

        /* 6. fold /NB onto limb planes and decode */
        CK(cudaMemset(acc, 0, (size_t)planes * nn * sizeof(int)));
        k_mfft_fold<<<gnn, blk>>>(acc, Ch, nn, NB, K, S, planes);
        k_decode<<<gnn, blk>>>(dC, acc, nn, planes, full.SA + full.SB);

        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }

    *gemms = nprod;
    printf("limb-mfft: L=%d (pad from %d/%d) S=%d NB=%d K=%d  products %lld "
           "(schoolbook would be %d)\n",
           L, full.LA, full.LB, S, NB, K, nprod, full.LA * full.LB);

    cudaFree(pA); cudaFree(pB); cudaFree(Ah); cudaFree(Bh); cudaFree(Ch);
    cudaFree(acc); cudaFree(d_fwops); cudaFree(d_ivops);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    mfft_plan_free_gpu(&plan);
    return best;
}

/* Scaling study (item 8): value bits, GEMM counts and TFLOP/s against n.
 * Runs a fixed ladder of powers-of-two; skips sizes that do not fit in
 * device memory.  Only the methods that expose the L(n) growth are timed. */
static int run_sweep(cublasHandle_t h, int reps, int fp64_mode, int tile)
{
    static const int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("\nGPU scaling study (--sweep-n)%s\n",
           fp64_mode ? " [genuine fp64 data]" : " [fp32 promoted]");
    printf("%-6s %10s %8s %8s %8s %9s %9s %9s %9s %9s %9s\n",
           "n", "vA/vB", "LA×LB", "exGEMM", "faGEMM",
           "sgemm", "fp32ex", "fp32fa", "dgemm", "fp64ex", "fp64fa");
    printf("%-6s %10s %8s %8s %8s %9s %9s %9s %9s %9s %9s\n",
           "", "(fp32)", "(fp32)", "(fp32)", "(fp32)",
           "ms", "ms", "ms", "ms", "ms", "ms");
    printf("----------------------------------------------------------------------"
           "---------------------------------------\n");

    /* Probe int8 once at the largest size we can afford so the path is set. */
    {
        int probe_n = 256;
        size_t freeb = 0, totb = 0;
        cudaMemGetInfo(&freeb, &totb);
        for (int i = nsizes - 1; i >= 0; i--) {
            double need = (double)sizes[i] * sizes[i] * 80.0 / 1e9;
            if (need < freeb / 1e9 * 0.85) { probe_n = sizes[i]; break; }
        }
        probe_i8(h, probe_n);
        if (g_i8mode == 1) {
            if (tile >= 0 && tile < DP_NCFG) {
                g_dpcfg = tile;
                printf("dp4a forced: %s\n", g_dpcfgs[tile].name);
            } else {
                tune_dp4a(probe_n, 1);
            }
        }
    }

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        size_t nn = (size_t)n * n;
        double need = (double)nn * (5 * 4 + 3 * 8 + 20 + 13 * 4)
                    / (1024.0 * 1024.0 * 1024.0);
        size_t freeb = 0, totb = 0;
        cudaMemGetInfo(&freeb, &totb);
        if (need > freeb / 1073741824.0 * 0.9) {
            printf("%-6d  (skip: need %.2f GiB, free %.2f)\n",
                   n, need, freeb / 1073741824.0);
            continue;
        }

        float *hA = (float *)malloc(nn * sizeof(float));
        float *hB = (float *)malloc(nn * sizeof(float));
        float *hC = (float *)malloc(nn * sizeof(float));
        double *hAd = (double *)malloc(nn * sizeof(double));
        double *hBd = (double *)malloc(nn * sizeof(double));
        if (!hA || !hB || !hC || !hAd || !hBd) {
            fprintf(stderr, "host oom at n=%d\n", n);
            free(hA); free(hB); free(hC); free(hAd); free(hBd);
            break;
        }

        unsigned long long s = 88172645463325252ULL + (unsigned long long)n * 17ULL;
        if (fp64_mode) {
            for (size_t i = 0; i < nn; i++) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                hAd[i] = (double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0;
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                hBd[i] = (double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0;
                hA[i] = (float)hAd[i];
                hB[i] = (float)hBd[i];
            }
        } else {
            for (size_t i = 0; i < nn; i++) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                hA[i] = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                hB[i] = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
                hAd[i] = (double)hA[i];
                hBd[i] = (double)hB[i];
            }
        }

        LimbPlan pf = plan_limbs(hA, hB, nn, 24);
        LimbPlan p64 = plan_limbs_d(hAd, hBd, nn, 53);
        LimbPlan ff32 = plan_limbs_faithful(pf, 24, n);
        LimbPlan ff64 = plan_limbs_faithful(p64, 53, n);

        float *dA, *dB, *dC;
        double *dAd, *dBd, *dCd;
        CK(cudaMalloc(&dA, nn * sizeof(float)));
        CK(cudaMalloc(&dB, nn * sizeof(float)));
        CK(cudaMalloc(&dC, nn * sizeof(float)));
        CK(cudaMalloc(&dAd, nn * sizeof(double)));
        CK(cudaMalloc(&dBd, nn * sizeof(double)));
        CK(cudaMalloc(&dCd, nn * sizeof(double)));
        CK(cudaMemcpy(dA, hA, nn * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, nn * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dAd, hAd, nn * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dBd, hBd, nn * sizeof(double), cudaMemcpyHostToDevice));

        float alpha = 1.0f, beta = 0.0f;
        double alphad = 1.0, betad = 0.0;
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

        float best_s = 1e30f, best_d = 1e30f;
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(t0));
            CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha, dB, n, dA, n, &beta, dC, n));
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best_s) best_s = ms;
        }
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(t0));
            CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alphad, dBd, n, dAd, n, &betad, dCd, n));
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best_d) best_d = ms;
        }

        long long g_ex32, g_fa32, g_ex64, g_fa64;
        double ms_ex32 = exact_float_gemm(h, n, dA, dB, dC, pf, pf, reps, &g_ex32);
        double ms_fa32 = exact_float_gemm(h, n, dA, dB, dC, pf, ff32, reps, &g_fa32);
        double ms_ex64 = exact_double_gemm(h, n, dAd, dBd, dCd, p64, p64, reps, &g_ex64);
        double ms_fa64 = exact_double_gemm(h, n, dAd, dBd, dCd, p64, ff64, reps, &g_fa64);

        printf("%-6d %4d/%-4d %3d×%-3d %8lld %8lld %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n",
               n, pf.vA, pf.vB, pf.LA, pf.LB, g_ex32, g_fa32,
               best_s, ms_ex32, ms_fa32, best_d, ms_ex64, ms_fa64);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        cudaFree(dAd); cudaFree(dBd); cudaFree(dCd);
        free(hA); free(hB); free(hC); free(hAd); free(hBd);
    }

    printf("\nColumns: vA/vB and LA×LB are the *exact* fp32 plan; faGEMM is the\n"
           "faithful product count.  ms columns are best-of-%d.  Value bits grow\n"
           "with n (wider exponent spread); faithful GEMMs stay flat (~9 / ~25)\n"
           "because the budget is sig + log2(n) + guard.\n", reps);
    return 0;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int n = 2048, reps = 3, check = 0, tile = -1, fp64_mode = 0, sweep = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fp64")) fp64_mode = 1;
        else if (!strcmp(argv[i], "--sweep-n")) sweep = 1;
        else if (!strcmp(argv[i], "--cutoff") && i + 1 < argc)
            g_strassen_cutoff = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--n N] [--reps R] [--check] [--tile I] [--fp64] "
                   "[--sweep-n] [--cutoff C]\n"
                   "  --tile I   force dp4a config I instead of autotuning\n"
                   "  --fp64     genuine double-precision inputs (53-bit\n"
                   "             significands).  Default promotes fp32 test\n"
                   "             data, so the fp64 rows measure cost without\n"
                   "             measuring benefit.  --fp64 regenerates the\n"
                   "             dataset and runs the double track properly.\n"
                   "  --sweep-n  scaling study: value bits / GEMMs / ms vs n\n"
                   "  --cutoff C Strassen leaf size (default %d); n<=C is a\n"
                   "             single cublas GEMM\n",
                   argv[0], g_strassen_cutoff);
            for (int c = 0; c < DP_NCFG; c++)
                printf("      %d: %s\n", c, g_dpcfgs[c].name);
            return 0;
        }
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int rtv = 0, drv = 0;
    cudaRuntimeGetVersion(&rtv);
    cudaDriverGetVersion(&drv);
    printf("device:  %s  sm_%d%d  %d SMs\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);
    printf("toolkit: runtime %d.%d, driver %d.%d\n",
           rtv / 1000, (rtv % 1000) / 10, drv / 1000, (drv % 1000) / 10);
    g_sms = prop.multiProcessorCount;

    if (sweep) {
        cublasHandle_t h;
        CB(cublasCreate(&h));
        int rc = run_sweep(h, reps, fp64_mode, tile);
        cublasDestroy(h);
        return rc;
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
    double *hAd = (double *)malloc(nn * sizeof(double));
    double *hBd = (double *)malloc(nn * sizeof(double));
    if (!hA || !hB || !hC || !hR || !hAd || !hBd) {
        fprintf(stderr, "host oom\n"); return 1;
    }

    /* Data generation.
     * Default: fp32 uniform in (-1,1).  fp64 rows then promote these values,
     * so their exact product is bit-identical to the fp32 one and the extra
     * limbs buy nothing real (cost without benefit).
     * --fp64: genuine 53-bit significands so the fp64 track measures both
     * cost and the accuracy gain of the wider embedding. */
    unsigned long long s = 88172645463325252ULL;
    if (fp64_mode) {
        for (size_t i = 0; i < nn; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            /* full 53-bit mantissa in (-1, 1) */
            hAd[i] = (double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0;
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            hBd[i] = (double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0;
            hA[i] = (float)hAd[i];
            hB[i] = (float)hBd[i];
        }
        printf("data: genuine fp64 (53-bit significands); "
               "fp32 rows are the same values rounded to float\n");
    } else {
        for (size_t i = 0; i < nn; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            hA[i] = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            hB[i] = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
            hAd[i] = (double)hA[i];
            hBd[i] = (double)hB[i];
        }
        printf("data: fp32 promoted to fp64 for the double rows "
               "(use --fp64 for genuine 53-bit inputs)\n");
    }

    float *dA, *dB, *dC, *dAbf, *dBbf;
    double *dAd, *dBd, *dCd;
    CK(cudaMalloc(&dA, nn * sizeof(float)));
    CK(cudaMalloc(&dB, nn * sizeof(float)));
    CK(cudaMalloc(&dC, nn * sizeof(float)));
    CK(cudaMalloc(&dAbf, nn * sizeof(float)));
    CK(cudaMalloc(&dBbf, nn * sizeof(float)));
    CK(cudaMalloc(&dAd, nn * sizeof(double)));
    CK(cudaMalloc(&dBd, nn * sizeof(double)));
    CK(cudaMalloc(&dCd, nn * sizeof(double)));
    CK(cudaMemcpy(dA, hA, nn * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, nn * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAd, hAd, nn * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBd, hBd, nn * sizeof(double), cudaMemcpyHostToDevice));

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

    /* Independent reference: always computed once, outside every timed
     * method, via the exact limb path decoded to double.  No timed row
     * produces hR, so none can score itself.  For n <= 512, --check also
     * cross-validates against a host triple loop. */
    LimbPlan pf = plan_limbs(hA, hB, nn, 24);
    gpu_limb_strategy_table(pf.vA, pf.vB);
    LimbPlan p64 = plan_limbs_d(hAd, hBd, nn, 53);
    printf("fp64 value bits: %d / %d%s\n", p64.vA, p64.vB,
           fp64_mode ? " (genuine)" : " (promoted fp32)");
    gpu_limb_strategy_table(p64.vA, p64.vB);
    {
        long long gr;
        exact_double_gemm(h, n, dAd, dBd, dCd, p64, p64, 1, &gr);
        CK(cudaMemcpy(hR, dCd, nn * sizeof(double), cudaMemcpyDeviceToHost));
        printf("reference: exact limb path (%lld int8 GEMMs), computed once "
               "outside the timed table\n\n", gr);
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
                    acc += hAd[(size_t)i * n + k] * hBd[(size_t)k * n + j];
                double d = fabs(acc - hR[(size_t)i * n + j]);
                double r = fabs(acc) > 0 ? d / fabs(acc) : d;
                if (r > worst) worst = r;
            }
        printf("check: exact limb path vs host float64, worst relative "
               "difference %.3e\n\n", worst);
    }

    Res res[24];
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

    /* Strassen-sgemm: recursive, bottoms out in cublasSgemm (item 10). */
    {
        float best = 1e30f;
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(t0));
            strassen_f(h, n, dA, dB, dC);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best) best = ms;
        }
        CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
        res[nr].name = "strassen-sgemm"; res[nr].ms = best;
        res[nr].err = rel_err_host(hC, hR, nn);
        res[nr].exact = 0; res[nr].gemms = 7; /* nominal; depth-dependent */
        nr++;
    }

    /* --- fp64 baseline and the exact fp64 product ---
     * On a consumer card fp64 runs at a small fraction of fp32, so this is
     * the row that decides whether an exact integer product is competitive
     * with simply using more precision.  Buffers and the limb plan were
     * prepared above; the reference was computed from the same plan but
     * outside this timed block, so fp64-exact cannot score itself. */
    {
        double *hCd = (double *)malloc(nn * sizeof(double));
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

        /* Strassen-dgemm: recursive, bottoms out in cublasDgemm. */
        best = 1e30f;
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(t0));
            strassen_d(h, n, dAd, dBd, dCd);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best) best = ms;
        }
        CK(cudaMemcpy(hCd, dCd, nn * sizeof(double), cudaMemcpyDeviceToHost));
        res[nr].name = "strassen-dgemm"; res[nr].ms = best;
        res[nr].err = rel_err_host_d(hCd, hR, nn);
        res[nr].exact = 0; res[nr].gemms = 7; nr++;

        long long g64;
        double ms64 = exact_double_gemm(h, n, dAd, dBd, dCd, p64, p64, reps, &g64);
        if (g64) {
            CK(cudaMemcpy(hCd, dCd, nn * sizeof(double), cudaMemcpyDeviceToHost));
            res[nr].name = "limb-fp64-exact"; res[nr].ms = ms64;
            res[nr].err = rel_err_host_d(hCd, hR, nn);
            res[nr].exact = 1; res[nr].gemms = g64; nr++;
            printf("limb-fp64-exact: %d/%d value bits -> %d x %d limbs of %d bits, "
                   "%lld int8 GEMMs%s\n", p64.vA, p64.vB, p64.LA, p64.LB,
                   LIMB_BITS_GPU, g64,
                   fp64_mode ? " [genuine fp64 data]" : " [promoted fp32 data]");
        }
        free(hCd);
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
        double ms = exact_float_gemm(h, n, dAbf, dBbf, dC, pb, pb, reps, &gb);
        CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
        res[nr].name = "limb-bf16-exact"; res[nr].ms = ms;
        res[nr].err = rel_err_host(hC, hR, nn);
        res[nr].exact = 1; res[nr].gemms = gb; nr++;
        printf("limb-bf16-exact: %d/%d value bits -> %d x %d limbs of %d bits, "
               "%lld int8 GEMMs\n", pb.vA, pb.vB, pb.LA, pb.LB,
               LIMB_BITS_GPU, gb);
        free(hBbf);
    }

    /* --- exact fp32 through limb planes --- */
    {
        long long gf;
        double ms = exact_float_gemm(h, n, dA, dB, dC, pf, pf, reps, &gf);
        CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
        res[nr].name = "limb-fp32-exact"; res[nr].ms = ms;
        res[nr].err = rel_err_host(hC, hR, nn);
        res[nr].exact = 1; res[nr].gemms = gf; nr++;
        printf("limb-fp32-exact: %d/%d value bits -> %d x %d limbs of %d bits, "
               "%lld int8 GEMMs\n", pf.vA, pf.vB, pf.LA, pf.LB,
               LIMB_BITS_GPU, gf);
    }

    /* --- GPU MFFT (item 11): same embedding, transform convolution --- */
    {
        long long gm;
        double ms = mfft_float_gemm(h, n, dA, dB, dC, pf, reps, &gm);
        if (gm) {
            CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
            res[nr].name = "limb-mfft-fp32"; res[nr].ms = ms;
            res[nr].err = rel_err_host(hC, hR, nn);
            res[nr].exact = 1; res[nr].gemms = gm; nr++;
        }
    }

    /* --- faithful-rounding fp32 / fp64 (item 5) ---
     * Drop low-order limbs that cannot affect a correctly-rounded binary
     * result of the stated precision.  Verified against the full exact
     * reference in hR. */
    {
        LimbPlan ff32 = plan_limbs_faithful(pf, 24, n);
        long long gf;
        double ms = exact_float_gemm(h, n, dA, dB, dC, pf, ff32, reps, &gf);
        CK(cudaMemcpy(hC, dC, nn * sizeof(float), cudaMemcpyDeviceToHost));
        res[nr].name = "limb-fp32-faithful"; res[nr].ms = ms;
        res[nr].err = rel_err_host(hC, hR, nn);
        res[nr].exact = 1; res[nr].gemms = gf; nr++;
        printf("limb-fp32-faithful: need ~%d product bits -> keep %d/%d value bits "
               "(%d x %d limbs), %lld int8 GEMMs (exact had %d)\n",
               24 + ceil_log2_int(n) + 4, ff32.vA, ff32.vB,
               ff32.LA, ff32.LB, gf, pf.LA * pf.LB);
    }
    {
        LimbPlan ff64 = plan_limbs_faithful(p64, 53, n);
        long long gf;
        double *hCd = (double *)malloc(nn * sizeof(double));
        double ms = exact_double_gemm(h, n, dAd, dBd, dCd, p64, ff64, reps, &gf);
        if (gf) {
            CK(cudaMemcpy(hCd, dCd, nn * sizeof(double), cudaMemcpyDeviceToHost));
            res[nr].name = "limb-fp64-faithful"; res[nr].ms = ms;
            res[nr].err = rel_err_host_d(hCd, hR, nn);
            res[nr].exact = 1; res[nr].gemms = gf; nr++;
            printf("limb-fp64-faithful: need ~%d product bits -> keep %d/%d value bits "
                   "(%d x %d limbs), %lld int8 GEMMs (exact had %d)\n",
                   53 + ceil_log2_int(n) + 4, ff64.vA, ff64.vB,
                   ff64.LA, ff64.LB, gf, p64.LA * p64.LB);
        }
        free(hCd);
    }

    /* --- Ozaki scheme I (item 9): s int8 slices, s^2 GEMMs --- */
    {
        static const int slices[] = {2, 4, 7};
        static const char *names[] = {"ozaki-i8-s2", "ozaki-i8-s4", "ozaki-i8-s7"};
        double *hCd = (double *)malloc(nn * sizeof(double));
        for (int k = 0; k < 3; k++) {
            int s = slices[k];
            long long go;
            double ms = ozaki_double_gemm(h, n, hAd, hBd, dCd, s, reps, &go);
            CK(cudaMemcpy(hCd, dCd, nn * sizeof(double), cudaMemcpyDeviceToHost));
            res[nr].name = names[k]; res[nr].ms = ms;
            res[nr].err = rel_err_host_d(hCd, hR, nn);
            res[nr].exact = 0; res[nr].gemms = go; nr++;
            printf("ozaki-i8-s%d: %d slices -> %lld int8 GEMMs\n", s, s, go);
        }
        free(hCd);
    }

    double flops = 2.0 * (double)n * n * n;
    double base = res[0].ms;
    printf("\n%-16s %8s %10s %10s %11s\n",
           "method", "GEMMs", "ms", "TFLOP/s", "rel error");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < nr; i++) {
            /* fp64-exact produced hR, so it cannot score itself */
        int is_ref = !strcmp(res[i].name, "limb-fp64-exact");
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
    cudaFree(dAd); cudaFree(dBd); cudaFree(dCd);
    free(hA); free(hB); free(hC); free(hR); free(hAd); free(hBd);
    return 0;
}
