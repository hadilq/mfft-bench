# mfft-bench

A C benchmark for **MFFT** (Matrix Fast Fourier Transform), the matrix
multiplication method described in
[hadilq.com/posts/matrix-fast-fourier-transform](https://hadilq.com/posts/matrix-fast-fourier-transform/),
compared against the methods people actually use.

There are two tracks, because they answer different questions:

* **exact track** (default) — `n x n` matrices of `B`-bit *integers*, every
  method producing the bit-identical product, checked against the textbook
  baseline. This is the setting MFFT is defined in.
* **ML track** (`--ml`) — `n x n` fp32 matrices at machine-learning
  precisions, reporting throughput *and* accuracy. In ML the two trade
  against each other, so speed alone is not a fair ranking.

## Quick start

```sh
nix develop                 # gcc, openblas, perf, hyperfine, valgrind
make && make check          # build + self-tests

./mfft-bench --test-roots                                # verify the post's math
./mfft-bench --n 64 --bits 8192 --no-naive --no-verify   # exact track
./mfft-bench --ml --n 1024                               # ML track
```

`make WITH_BLAS=1` adds OpenBLAS `sgemm`/`dgemm` references.
`make WITH_OPENMP=1` parallelises the packed fp32 kernel.
`make LIMB_BITS=1` builds the base-2 variant, i.e. the post's literal
"matrix of digits" model.

## Methods

### Exact integer track

Entries are `L` limbs in base `2^LIMB_BITS`. Writing each matrix as a
polynomial with small-entry matrix coefficients turns a big-integer matmul
into a **convolution of matrix-valued polynomials**:

```
A = sum_u A_u * beta^u          beta = 2^LIMB_BITS
AB = sum_w ( sum_{u+v=w} A_u B_v ) * beta^w
```

| method | convolution done by | `n x n` products |
| --- | --- | --- |
| `bigint-ijk`, `bigint-ikj` | not decomposed: textbook `n^3` matmul with schoolbook limb multiplies | — (`n^3 L^2` limb MACs) |
| `limbplane` | schoolbook | `L^2` |
| `mfft` | transform over roots of unity | `NB * K^2` |

Each is run under five interchangeable inner kernels, so the table separates
*algorithm* from *implementation quality*:

| kernel | what it is |
| --- | --- |
| `ikj` | cache-friendly triple loop |
| `blocked` | cache tiling |
| `packed` | packed panels + SIMD register micro-kernel — the structure OpenBLAS/BLIS/oneDNN actually use |
| `strassen` | Strassen recursion, 7 multiplies / 18 adds, bottoming out in `packed` |
| `winograd` | Strassen–Winograd, 7 multiplies / **15** adds — the variant fast libraries ship |

### ML track

`sgemm-ijk` (the ordinary method), `sgemm-ikj`, `sgemm-blocked`,
`sgemm-packed`, `sgemm-strassen`, optional `blas-sgemm`, plus the two
precisions that dominate production:

* `bf16-packed` — inputs rounded to bfloat16, fp32 accumulate (training)
* `int8-packed` — per-channel symmetric quantization, int32 accumulate,
  dequantize (inference). Panels stay int16 so the cache benefit of
  quantization survives, widening to int32 in-register.

## MFFT: verifying the post, and one correction

`--test-roots` implements the post's `H_{s,k}` recursion verbatim and checks
it against dense matrix powers: the order is `2K`, `I_s^K = -1`, the powers
`I_s^0..I_s^{K-1}` are linearly independent, and the sample arrays printed in
the post reproduce exactly. **All of it passes.** Because those powers are
independent, ring elements are stored in the power basis `sum_c v_c I_s^c`,
where multiplying by `I_s^e` is exactly a negacyclic shift — sign flips and
index arithmetic, no multiplications.

The cost analysis needs one fix. The post treats `P_A(I_s^j) P_B(I_s^j)` as a
single `n x n` product. It isn't: a value of the polynomial lives in
`R (x) M_n(Z)` with `R = Z[y]/(y^K + 1)`, i.e. a `K`-tuple of `n x n`
matrices, so one pointwise product is a length-`K` negacyclic convolution —
`K^2` matrix products. That missing factor is where the claimed 21% saving at
`m = 16` comes from.

The fix is Schönhage–Strassen balancing: decouple transform length from ring
dimension. Pack `S` limbs per coefficient, transform over `NB = 2L/S` points
in a ring of dimension `K = 2S`. Total products `NB*K^2 = 8LS`, minimised at
`S ~ sqrt(L/2)`, giving `~5.7 L^1.5` against `L^2` for schoolbook. So MFFT
does win asymptotically in `L` — just not at 16 bits.

## Results

Single core (shared vCPU, AVX-512), gcc 13.3 `-O3 -march=native`.

### Exact track, `n = 64`, best kernel per method

| entry bits | `L` | products mfft / plane | limb-plane | mfft | speedup |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 256 | 16 | 512 / 256 | 0.014 | 0.037 | 0.39x |
| 512 | 32 | 1024 / 1024 | 0.053 | 0.074 | 0.72x |
| 1024 | 64 | 4096 / 4096 | 0.202 | 0.248 | 0.81x |
| 2048 | 128 | 8192 / 16384 | 0.794 | 0.480 | **1.66x** |
| 4096 | 256 | 32768 / 65536 | 3.492 | 1.660 | **2.10x** |
| 8192 | 512 | 65536 / 262144 | 13.555 | 3.520 | **3.85x** |

Crossover is around 2048-bit entries, tracking the product-count ratio — so
the post's assumption that transform overhead is negligible does hold up.

Kernel breakdown at 8192 bits shows the two speedups are independent, exactly
as the post predicts ("orthogonal to the other methods"):

| kernel | limb-plane | mfft |
| --- | ---: | ---: |
| `ikj` | 24.02 | 7.90 |
| `blocked` | 21.85 | 5.77 |
| `packed` | 17.05 | 4.68 |
| `strassen` | 13.56 | **3.52** |
| `winograd` | 14.66 | 3.75 |

**The post's own example** (`m = 16`, base 2, `make LIMB_BITS=1`, `n = 128`):
MFFT needs 512 products against limb-plane's 256 and runs ~1.8x slower. The
predicted 21% improvement does not appear, and the corrected count says it
cannot at that width.

### ML track, `n = 1024`

| method | GFLOP/s | vs packed | rel error |
| --- | ---: | ---: | ---: |
| `sgemm-ijk` (ordinary) | 0.57 | 0.01x | 5.7e-07 |
| `sgemm-ikj` | 12.09 | 0.23x | 5.7e-07 |
| `sgemm-blocked` | 12.40 | 0.24x | 5.7e-07 |
| `sgemm-packed` | **52.39** | 1.00x | 2.9e-07 |
| `sgemm-strassen` | 33.80 | 0.65x | 1.7e-06 |
| `bf16-packed` | 51.29 | 0.98x | **2.1e-03** |
| `int8-packed` | 26.94 | 0.51x | **5.6e-03** |

Four things worth reading off this table:

1. **Data movement dominates everything else.** Packing plus a register
   micro-kernel is 92x the textbook loop and 4x cache blocking. No
   asymptotic trick in this repo comes close to that factor. If you are
   optimising ML matmul, this is where the wins are.
2. **Strassen loses in fp32** at ML shapes. It saves 12.5% of the
   multiplications per level and pays for it with `O(n^2)` copies and adds
   against a kernel that is already bandwidth-bound. It wins in the exact
   integer track (see above) because there the multiplies are genuinely
   expensive relative to memory. Tune with `--cutoff`.
3. **bf16 costs 4 orders of magnitude of accuracy and buys nothing here**,
   because portable C has no bf16 arithmetic path. On hardware with bf16
   units the same error buys 2–8x. The row measures the *accuracy* price of
   the format, so you can decide whether the hardware speedup is worth it.
4. **int8 is slower than fp32 on this CPU.** It needs a dot-product
   instruction (AVX-512 VNNI, ARM `sdot`) to pay off; without one, int32
   multiplies are slower than fp32 FMAs. The quantization error is real
   regardless. Quantization is a *hardware* bet, not an algorithmic one.

## Is MFFT useful for machine learning?

Short answer: no, and the benchmark shows why in three independent ways.

* **Wrong width by three orders of magnitude.** MFFT's crossover is around
  2048-bit entries. ML entries are fp32 (24-bit mantissa) at the widest and
  4–8 bits after quantization. In that regime MFFT does 2–4x *more* work.
* **Wrong direction on precision.** MFFT buys exactness at a cost. ML's
  entire optimisation history runs the other way — fp32 to bf16 to int8 to
  int4 — spending accuracy to buy throughput. A method whose selling point
  is bit-exactness is solving a problem ML does not have.
* **Wrong number type.** The post says it directly: the entries must be
  integers, and it could not be made to work with floating point.

Where this shape of math *does* matter for ML is **encrypted and verifiable
inference**. FHE schemes (BFV, BGV, CKKS) do their arithmetic in
`Z_q[y]/(y^K + 1)` — the exact negacyclic ring MFFT is built on — and their
ciphertext coefficients are hundreds to thousands of bits wide, right where
the crossover in the table above sits. Zero-knowledge proofs of inference
have the same profile. That is the honest home for this method, not a
training loop.

## State of the art, and what is not implemented here

| result | status here |
| --- | --- |
| `ω ≤ 2.371339` (Alman, Duan, Vassilevska Williams, Xu, Xu, Zhou 2025) | not implemented — galactic; the constants make it slower than schoolbook at any size that fits in a datacenter |
| Strassen 1969, 7 mults for 2×2 (`ω = 2.807`) | `strassen` |
| Strassen–Winograd, 7 mults / 15 adds | `winograd` |
| Laderman 1976, 23 mults for 3×3 | not implemented — `log_3 23 = 2.854`, *worse* than Strassen |
| AlphaTensor 2022, 47 mults for 4×4 | not implemented — valid only in characteristic 2, so not for integer or float matrices |
| AlphaEvolve 2025, 48 mults for 4×4 | not implemented — complex-valued coefficients |
| Dumas–Pernet–Sedoglavic 2025, 48 mults for 4×4 with rational coefficients | not implemented — needs `1/2`, so not exact over `Z` without rescaling |
| packed panels + register micro-kernel (BLIS/OpenBLAS/oneDNN) | `packed`, `sgemm-packed` — empirically the largest single factor |

The gap between the top and bottom rows is the point. The record for `ω` has
moved several times since 2020 and none of it has touched a production
kernel, while the unglamorous packing/blocking row is worth 92x.

## Options

```
--n N          matrix dimension (default 64)
--bits B       bits per entry, multiple of LIMB_BITS (default 256)
--sigma S      MFFT block exponent override (block size = 2^S limbs)
--reps R       repetitions, best time reported
--cutoff C     Strassen/Winograd base-case cutoff (default 128)
--seed X       PRNG seed
--ml           run the machine-learning GEMM track
--no-verify    skip exactness checks (needed at large n: reference is n^3 L^2)
--no-naive     skip the textbook methods
--sweep        bit-width sweep at the given --n
--test-roots   self-test the H_{s,k} recursion
--csv          machine-readable output
```

## Layout

```
src/mfftbench.h   shared declarations
src/roots.c       H_{s,k} recursion from the post + self-tests
src/mfft.c        transform, pointwise ring products, plan selection
src/methods.c     schoolbook and limb-plane baselines
src/kernel.c      integer kernels: ikj, blocked, packed, Strassen, Winograd
src/mlgemm.c      ML track: fp32 / bf16 / int8 GEMM with accuracy reporting
src/bigmat.c      big-integer storage, carry normalisation
src/main.c        CLI, verification, timing tables
```

## Caveats

* MFFT intermediates must fit in `int64`. `mfft_plan_maxbits()` reports the
  worst case and the driver warns past 62 bits.
* `L` must be a power of two; `n` is arbitrary (the recursive schemes fall
  back to the packed base case on odd sizes).
* Micro-kernel tile shape is picked from `__AVX512F__` / `__AVX2__` at
  compile time; override with `-DFMR=`. Rebuild on the target machine — a
  shape that spills the accumulators costs 4x.
* Only the fp32 packed kernel is parallelised, and only with
  `WITH_OPENMP=1`. Everything else is single-threaded. The MFFT pointwise
  stage is embarrassingly parallel over the `NB` evaluation points and is
  the obvious next step.
* Exact-track entries are unsigned; signed support needs only a
  sign-magnitude split.
