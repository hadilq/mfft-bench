# mfft-bench

A C benchmark for **MFFT** (Matrix Fast Fourier Transform), the matrix
multiplication method described in
[hadilq.com/posts/matrix-fast-fourier-transform](https://hadilq.com/posts/matrix-fast-fourier-transform/),
compared against the methods people actually use — including an **fp32
embedding** that lets the exact integer methods run on float matrices.

Three tracks:

* **exact track** (default) — `n x n` matrices of `B`-bit *integers*, every
  method producing the bit-identical product, checked against the textbook
  baseline. This is the setting MFFT is defined in.
* **ML track** (`--ml`) — `n x n` fp32 matrices at machine-learning
  precisions, reporting throughput *and* accuracy.
* **fp32 embedding** (inside `--ml`) — floats mapped onto a shared
  fixed-point integer grid, multiplied exactly, rounded back.

## Quick start

```sh
nix develop                 # gcc, openblas, perf, hyperfine, valgrind
make && make check          # build + self-tests

./mfft-bench --test-roots                                # verify the post's math
./mfft-bench --n 64 --bits 8192 --no-naive --no-verify   # exact track
./mfft-bench --ml --n 512                                # ML track + fp32 embedding
./mfft-bench --ml --n 256 --fp-width 512                 # the literal fixed grid
```

`make WITH_BLAS=1` adds OpenBLAS references, `WITH_OPENMP=1` parallelises the
packed fp32 kernel, `LIMB_BITS=1` builds the post's literal base-2 model.

## Methods

Entries are `L` limbs in base `2^LIMB_BITS`. Writing each matrix as a
polynomial with small-entry matrix coefficients turns a big-integer matmul
into a **convolution of matrix-valued polynomials**:

```
A = sum_u A_u * beta^u          beta = 2^LIMB_BITS
AB = sum_w ( sum_{u+v=w} A_u B_v ) * beta^w
```

| method | convolution done by | `n x n` products |
| --- | --- | --- |
| `bigint-ijk`, `bigint-ikj` | not decomposed: textbook `n^3` with schoolbook limb multiplies | — |
| `limbplane` | schoolbook | `L^2` |
| `karatsuba` | recursive halving | `L^1.585` |
| `mfft` | transform over roots of unity, schoolbook pointwise step | `NB * K^2 = 8LS ~ 5.7 L^1.5` |
| `mfft-rec` | same, with the pointwise step solved recursively (Schönhage–Strassen) | planner-chosen, `~L log L log log L` |

Each runs under five interchangeable inner kernels, so the tables separate
*algorithm* from *implementation quality*: `ikj`, `blocked`, `packed`
(packed panels + SIMD register micro-kernel, the OpenBLAS/BLIS structure),
`strassen`, and `winograd` (Strassen–Winograd, 7 multiplies / 15 adds).

The ML track adds `sgemm-ijk` (the ordinary method), `sgemm-ikj`,
`sgemm-blocked`, `sgemm-packed`, `sgemm-strassen`, optional `blas-sgemm`,
`bf16-packed` (bfloat16 inputs, fp32 accumulate) and `int8-packed`
(per-channel symmetric quantization, int32 accumulate).

## MFFT: verifying the post, and one correction

`--test-roots` implements the post's `H_{s,k}` recursion verbatim and checks
it against dense matrix powers: order `2K`, `I_s^K = -1`, powers
`I_s^0..I_s^{K-1}` linearly independent, and the sample arrays in the post
reproduced exactly. **All of it passes.** Ring elements are therefore stored
in the power basis `sum_c v_c I_s^c`, where multiplying by `I_s^e` is exactly
a negacyclic shift — sign flips and index arithmetic, no multiplications.

The cost analysis needs one fix. The post treats `P_A(I_s^j) P_B(I_s^j)` as a
single `n x n` product. It isn't: a value of the polynomial lives in
`R (x) M_n(Z)` with `R = Z[y]/(y^K + 1)`, i.e. a `K`-tuple of `n x n`
matrices, so one pointwise product is a length-`K` negacyclic convolution —
`K^2` matrix products. That missing factor is where the claimed 21% saving at
`m = 16` comes from.

The fix is Schönhage–Strassen balancing: pack `S` limbs per coefficient,
transform over `NB = 2L/S` points in a ring of dimension `K = 2S`. Total
`8LS`, minimised at `S ~ sqrt(L/2)`, so `~5.7 L^1.5` against `L^2`.

## Applying MFFT to fp32

An fp32 number is a 24-bit significand times a power of two, and the exponent
has only 8 bits of range. So every fp32 in a matrix can be placed on one
shared fixed-point grid — shift the significand left by (exponent + offset) —
and once that is done the exponents are gone and the entries are just wide
integers. Multiply exactly, round the exact result back to fp32. The wide
integer is never materialised: the shift writes straight into the limb planes
the convolution already wants.

Two changes to the naive version of this idea earn most of its performance:

**Separate scales.** `A` is encoded at scale `2^SA` and `B` at `2^SB`, with
the product landing at `2^(SA+SB)`. Each matrix only has to span its own
exponent range rather than both.

**Size the grid from the data.** The width needed is
`24 + (max exponent - min exponent) + 1`. What costs limbs is the *spread* of
exponents, not their absolute size. Since every method's cost is at least
quadratic-ish in `L`, this is the difference between 16 and 1024 matrix
products:

| grid | limbs | limb-plane | karatsuba | mfft | time (n=256, best) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed 512-bit (`--fp-width 512`) | 32 | 1024 | 243 | 1024 | 1.61 s |
| adaptive, from the data | 4 | 16 | 9 | 64 | 0.025 s |

Same bit-exact answer, **64x** less work.

The result is a Kulisch-style exact accumulation: bit-exact, independent of
summation order, hence perfectly reproducible across thread counts, and
correctly rounded to fp32 at the end. It is *more* accurate than fp32 or even
fp64 accumulation, not less.

### Does it pay off? (ML track, `n = 512`)

Exponent spread 17/19 → 4 limbs (64 bits). Encode 0.009 s and decode 0.016 s
are excluded from the timings, on the assumption a caller stays in fixed
point between consecutive matmuls.

| method | seconds | GFLOP/s | rel error |
| --- | ---: | ---: | ---: |
| `sgemm-packed` | 0.0050 | **53.29** | 2.9e-07 |
| `bf16-packed` | 0.0045 | 60.17 | 2.1e-03 |
| `int8-packed` | 0.0100 | 26.95 | 5.5e-03 |
| `fp32->limbplane` | 0.4187 | 0.64 | **2.5e-08** exact |
| `fp32->karatsuba` | 0.2636 | 1.02 | **2.5e-08** exact |
| `fp32->mfft` | 2.3790 | 0.11 | **2.5e-08** exact |

Error is measured against the bit-exact product; an fp64 loop scores 5.0e-16
on the same data, which cross-validates the embedding.

So: **11x more accurate than sgemm, and 53x slower.** The embedding works
and is exact, but it does not make MFFT competitive for ML, for a reason
that is structural rather than a matter of tuning:

* The embedding lands at `L = 2..8` limbs. MFFT's crossover is `L ~ 128`.
  At `L = 4` MFFT needs 64 products where schoolbook needs 16 — it is **9x
  slower than plain limb planes** here, the worst regime it has.
* The floor is not MFFT's fault either. Exactness needs `>= 24 + spread` bits
  per entry, so at minimum a handful of limb products per output. Even at
  `L = 2` with Karatsuba that is 3 small-integer GEMMs against 1 fp32 GEMM,
  and small-integer GEMM runs at ~10 GOP/s against fp32's 53 GFLOP/s. **An
  order of magnitude is the best case**, not 53x but not 1x either.

Where that trade is worth taking: reproducible/deterministic training and
inference, numerical debugging, ill-conditioned solves, and anything needing
order-independent reduction. Not throughput-bound training.

### Recursive MFFT

The pointwise step of the transform is a negacyclic convolution of length `K`
over `M_n(Z)` — the same shape of problem the transform itself solves. Doing
it schoolbook (`K^2` products) is what capped the method at `5.7 L^1.5`.
`mfft-rec` solves it recursively instead:

* view the length-`K` input as `NB` blocks of `S`, so with `x = y^S` the block
  sequence must be convolved negacyclically mod `x^NB + 1`;
* carry blocks in `R = Z[t]/(t^Kr + 1)`, `Kr = 2S`, wide enough that no block
  product wraps, so `R`-arithmetic is faithful;
* `psi = t^(Kr/NB)` satisfies `psi^NB = -1`, so pre-twisting by `psi^i` turns
  the negacyclic block convolution into a cyclic one, transformable with
  `omega = psi^2` — and every twiddle is again just a signed shift;
* recurse, bottoming out in Karatsuba. Requires `NB^2 | 2K`.

A DP planner picks the split at every level (including the top-level `S`) to
minimise the *total* product count, and tracks worst-case coefficient
magnitude so it refuses any split whose intermediates would not fit in
`int64` — it bottoms out early rather than overflowing.

The product counts fall by 3–7.5x:

| entry bits | `L` | limb-plane | karatsuba | mfft | **mfft-rec** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 4096 | 729 | 4096 | 1296 |
| 8192 | 512 | 262144 | 19683 | 65536 | **13824** |
| 32768 | 2048 | 4194304 | 177147 | 524288 | **93312** |
| 65536 | 4096 | 16777216 | 531441 | 2097152 | **279936** |

Wall clock, best kernel per method:

| entry bits | `n` | limb-plane | karatsuba | mfft | **mfft-rec** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 32 | 1.457 s | **0.179 s** | 0.412 s | 0.192 s |
| 32768 | 32 | — | 1.599 s | — | **1.002 s** |
| 65536 | 24 | — | 2.484 s | — | **1.814 s** |

So the recursion does what it was supposed to. Against the flat version it is
2.1–2.5x faster in wall time and up to 7.5x fewer products. Against Karatsuba
it draws at 8192-bit entries (fewer products, but the transform's `O(NB log NB
* K * n^2)` additions eat the margin at small `n`) and then pulls ahead:
**1.60x at 32768 bits, 1.37x at 65536 bits**, with the gap widening — as it
must, since `L log L log log L` beats `L^1.585`.

Two honest limits:

* **`int64` caps the depth at about two levels.** Coefficients grow by `NB`
  per level, so the planner runs out of headroom before the asymptotics fully
  kick in — at 65536-bit entries it issues 279936 products where an
  unconstrained planner would issue 165888. Real Schönhage–Strassen avoids
  this by working modulo `2^b + 1` so coefficients never grow; here that would
  mean modular reduction inside the matrix kernels plus CRT reconstruction.
  `__int128` accumulators would buy one more level more cheaply. Neither is
  implemented.
* **Smaller limbs do not help.** `LIMB_BITS=8` frees 16 bits of headroom but
  doubles `L`, which costs more than the extra recursion level buys.

The crossover story is now: schoolbook limb planes below ~256-bit entries,
Karatsuba from there to ~16000 bits, recursive MFFT above that.

**The post's own example** (`m = 16`, base 2, `make LIMB_BITS=1`, `n = 128`):
MFFT needs 512 products against limb-plane's 256 and runs ~1.8x slower.

## ML track observations

At `n = 1024`: `sgemm-ijk` 0.57 GFLOP/s, `sgemm-blocked` 12.40,
`sgemm-packed` **52.39**, `sgemm-strassen` 33.80.

1. **Data movement dominates.** Packing plus a register micro-kernel is 92x
   the textbook loop and 4x cache blocking. No asymptotic trick here comes
   close to that factor.
2. **Strassen loses in fp32** at ML shapes — it saves 12.5% of multiplies per
   level and pays `O(n^2)` copies against an already bandwidth-bound kernel.
   It wins in the exact integer track, where multiplies are genuinely
   expensive relative to memory. Tune with `--cutoff`.
3. **bf16 costs 4 orders of magnitude of accuracy and buys nothing here**,
   because portable C has no bf16 arithmetic path. The row measures the
   accuracy price of the format so you can judge the hardware speedup.
4. **int8 is slower than fp32 on this CPU.** It needs a dot-product
   instruction (AVX-512 VNNI, ARM `sdot`) to pay off. Quantization is a
   hardware bet, not an algorithmic one.

## State of the art, and what is not implemented here

| result | status |
| --- | --- |
| `ω ≤ 2.371339` (Alman, Duan, Vassilevska Williams, Xu, Xu, Zhou 2025) | not implemented — galactic |
| Strassen 1969, 7 mults for 2×2 | `strassen` |
| Strassen–Winograd, 7 mults / 15 adds | `winograd` |
| Karatsuba / Toom on the limb polynomial | `karatsuba` |
| Schönhage–Strassen recursion (Nussbaumer-style negacyclic transform) | `mfft-rec` |
| Laderman 1976, 23 mults for 3×3 | not implemented — `log_3 23 = 2.854`, worse than Strassen |
| AlphaTensor 2022, 47 mults for 4×4 | not implemented — characteristic 2 only |
| AlphaEvolve 2025, 48 mults for 4×4 | not implemented — complex-valued coefficients |
| Dumas–Pernet–Sedoglavic 2025, 48 mults, rational coefficients | not implemented — needs `1/2`, not exact over `Z` |
| packed panels + register micro-kernel | `packed`, `sgemm-packed` — the largest single factor measured |

## Options

```
--n N          matrix dimension (default 64)
--bits B       bits per entry, multiple of LIMB_BITS (default 256)
--sigma S      MFFT block exponent override (block size = 2^S limbs)
--reps R       repetitions, best time reported
--cutoff C     Strassen/Winograd base-case cutoff (default 128)
--seed X       PRNG seed
--ml           machine-learning GEMM track, including the fp32 embedding
--fp-width B   force a fixed B-bit fp32 grid instead of sizing from the data
--illcond E    widen the ML data exponent spread to E (costs limbs)
--no-verify    skip exactness checks (reference is n^3 L^2)
--no-naive     skip the textbook methods
--only LIST    comma-separated methods to run, e.g. karatsuba,mfft-rec
--sweep        bit-width sweep at the given --n
--test-roots   self-test the H_{s,k} recursion
--csv          machine-readable output
```

## Layout

```
src/mfftbench.h   shared declarations
src/roots.c       H_{s,k} recursion from the post + self-tests
src/mfft.c        transform, recursive SSA pointwise step, DP planner
src/methods.c     schoolbook, limb-plane and Karatsuba convolutions
src/fpfixed.c     fp32 <-> fixed-point embedding, correctly-rounded decode
src/kernel.c      integer kernels: ikj, blocked, packed, Strassen, Winograd
src/mlgemm.c      ML track: fp32 / bf16 / int8 GEMM with accuracy reporting
src/bigmat.c      big-integer storage, carry normalisation
src/main.c        CLI, verification, timing tables
```

## Caveats

* Intermediates must fit in `int64`; `mfft_plan_maxbits()` reports the worst
  case and the driver warns past 62 bits.
* `L` must be a power of two. The fp32 embedding rounds its computed width up
  to one, so it sometimes carries a few spare bits.
* The embedding skips entries whose exponent falls below the grid — with
  adaptive sizing that never happens, but `--fp-width` can be set too small.
* Karatsuba allocates `O(L n^2)` temporaries per level; watch memory at large
  `L` and `n` simultaneously.
* Micro-kernel tile shape is chosen from `__AVX512F__` / `__AVX2__` at compile
  time; rebuild on the target machine, a shape that spills costs 4x.
* Only the fp32 packed kernel is parallelised (`WITH_OPENMP=1`). MFFT's
  pointwise stage is embarrassingly parallel over the `NB` points and is the
  obvious next step.
