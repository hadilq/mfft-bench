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


## Source layout

CPU methods are already split by concern under `src/`:

| file | role |
| --- | --- |
| `src/mfft.c` | MFFT plan, transform ops, `conv_mfft` / `mm_mfft` |
| `src/methods.c` | schoolbook limb-plane, Karatsuba |
| `src/mlgemm.c` | ML track (sgemm/dgemm, faithful MFFT, low-prec) |
| `src/fpfixed.c` | float↔limb encode/decode |
| `src/roots.c` | signed-permutation roots of unity |
| `src/kernel.c` | packed integer GEMM micro-kernels |

GPU (work in progress toward the same shape):

| file | role |
| --- | --- |
| `cuda/mfft_gpu.cuh` | GPU MFFT plan, op-list, transform kernels, packing |
| `cuda/gemm_bench.cu` | driver, cuBLAS baselines, Ozaki, Strassen, limb exact, table |

Further splits (Ozaki / Strassen / limb-exact into their own `.cu` units) are
on the backlog.

## Glossary

| term | meaning |
| --- | --- |
| `n` | matrix dimension; all matrices are `n x n` |
| scalar width | bits per matrix **entry** (`--width`). Independent of `n` |
| limb | one 16-bit digit of an entry; `L` limbs per entry on CPU, 7-bit limbs on GPU so they fit int8 |
| plane | the `n x n` matrix formed by taking limb `u` of every entry. Multiplying entries becomes convolving planes |
| `sgemm` / `dgemm` | BLAS naming: single- (fp32) and double-precision (fp64) general matrix multiply |
| `packed` | the kernel style, not the precision: cache-block the loops, copy each block into a contiguous panel, accumulate a SIMD register tile. What OpenBLAS/BLIS/oneDNN do. `sgemm-packed` is that kernel in fp32, `dgemm-packed` in fp64 |
| `ikj` / `blocked` / `strassen` / `winograd` | the other inner-kernel styles, in the same slot as `packed` |
| exact | bit-exact: the integer product of the embedded values, correctly rounded once at the end. Not "more accurate" — *no* accumulation error |

## Two axes: `n` and scalar width

These are independent, and confusing them makes the tables unreadable:

* **`n`** is the matrix dimension — matrices are `n x n`. It sets the `O(n^3)`
  cost of **one** limb product, identically for every method.
* **scalar width** (`--width W`) is how many bits each matrix **entry** holds.
  `W = L * 16`, so `L` limbs per entry. It sets **how many** limb products a
  method needs — and that is the only thing the methods differ in.

So `time = products(L) x gemm_cost(n)`. The algorithmic comparison lives
along the width axis; `n` multiplies every row by the same factor. Sweeping
`n` is still worth doing (`--sweep-n`) because the `O(L n^2)` transform and
combination overheads shrink relative to `n^3`, so MFFT and Karatsuba get
slightly better as `n` grows — but the ordering is set by the width.

On the ML and GPU tracks it is the other way round: entries are a fixed
format, so `n` is the axis that matters.

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

### Roots of unity: where they live

There are **no complex roots of unity** in the timed path. The post’s
`I_s` matrices satisfy `I_s^K = -1` and order `2K`, so the ring
`Z[I_s] ≅ Z[y]/(y^K + 1)`. In the *power basis* `∑ v_c I_s^c`, multiplying
by a root `I_s^e` is only a **negacyclic shift** of the coefficient vector
(rotate + flip sign when you wrap past `K`).

| file | role |
|------|------|
| `src/roots.c` | Builds the dense `H_{s,k}` tables from the post’s recursion (for `--test-roots` only). Never used in matmul. |
| `src/mfft.c` `build_ops()` | Host: turns each FFT butterfly’s twiddle exponent `e` into a cycle of `(u, v, sign, mode)` ops. |
| `cuda/gemm_bench.cu` `mfft_build_ops()` | Same logic for the GPU plan. |

**How `e` is chosen (the only “root” arithmetic):**

```c
e = (g * t * step) % (2*K);     // forward
e = (2*K - e) % (2*K);          // inverse (conjugate)
```

with `omega = y^g`, `g = 2K / NB`. Applying `y^e` is:

```c
tgt = (c + e) % K;
sign = ((c + e) / K) is odd ? -1 : +1;   // negacyclic wrap
```

Those `(u, v, sign, mode)` triples are **built once** on the host when the
plan is created (`mfft_plan_build_ops`), uploaded to the device, and reused
for every timed repetition. The kernels (`k_mfft_run32` / `k_mfft_run64`)
only walk that list — they never recompute roots or look up `H_{s,k}`.

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
| `fp32->toom3 (hybrid Toom-3/Karatsuba), fp32->karatsuba` | 0.2636 | 1.02 | **2.5e-08** exact |
| `fp32->mfft` | 2.3790 | 0.11 | **2.5e-08** exact |

Error is measured against the bit-exact product; an fp64 loop scores 5.0e-16
on the same data, which cross-validates the embedding.

So on CPU: **11x more accurate than sgemm, and 53x slower.** (The GPU
answer is very different — see the GPU track below, where the same
embedding costs 9x an sgemm and exact fp64 costs 1.66x a dgemm.) The
embedding works and is exact, but on CPU it does not make MFFT competitive
for ML, for a reason that is structural rather than a matter of tuning:

* The embedding lands at `L = 2..8` limbs. MFFT only overtakes Karatsuba
  past `L ~ 512`. At `L = 4` MFFT needs 64 products where schoolbook needs
  16 — it is **9x slower than plain limb planes** here, the worst regime it
  has.
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

## Exact matrix multiplication below fp32

Generalising the embedding downward answers the ML question more sharply
than any timing does. An entry needs `vbits = significand + exponent spread`
bits on the shared grid, and a **single** GEMM is already exact when

```
vbits(A) + vbits(B) + ceil(log2 n)  <=  accumulator width
```

With an int64 accumulator at `n = 512`:

| format | significand | vbits | limbs | products |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 24 | 41 | 4 | 9 (Karatsuba) |
| bf16 | 8 | 26 | **1** | **1** |
| int8 | — | 8 | **1** | **1** |
| int4 | — | 4 | **1** | **1** |

At bf16 and below there is no convolution at all. Karatsuba and MFFT are not
slow there — they are *undefined*, because there is only one limb. That is
the structural reason MFFT has nothing to offer ML, and no amount of tuning
changes it.

Each precision gets its own packed kernel (`src/lowprec.c`): int8/int4 run
int16 operands into int32 accumulators, twice the SIMD lanes and cheaper
adds than the int32/int64 path bf16 needs.

Three findings from the CPU track:

* **Exact accumulation buys nothing at bf16.** `bf16-exact` and
  `bf16-packed` score the *same* 2.1e-03 — input rounding dominates
  completely, so paying 5.7x for an exact sum changes nothing measurable.
* **int8 accumulation was already exact.** int32 accumulation of int8
  products cannot overflow at these sizes, so the quantized path in every
  inference stack is bit-exact given its inputs. There is no accumulation
  error to remove.
* **int4 costs an order of magnitude of accuracy** (1.0e-01) for no CPU
  speedup — the win needs hardware that multiplies 4-bit operands natively.

## Is MFFT useful for machine learning?

**MFFT itself: no**, and the reason is structural rather than a matter of
constants.

* **Wrong width by two orders of magnitude.** MFFT only overtakes Karatsuba
  past `L ~ 512` limbs. Every ML embedding measured here lands at `L = 1`
  (bf16, int8, int4), `L = 4..8` (fp32) or `L = 12` (fp64). At those widths
  MFFT does *more* work than schoolbook.
* **Wrong direction on precision.** MFFT buys exactness. ML's whole
  optimisation history runs the other way — fp32 to bf16 to int8 to int4 —
  spending accuracy for throughput.
* **Wrong number type.** The post says it directly: the entries must be
  integers.

**Exact matmul via limb decomposition: it depends on the hardware, and the
answer changed once we measured a GPU.**

| | exact fp32 vs sgemm | exact fp64 vs dgemm | faithful fp32 | faithful fp64 |
| --- | ---: | ---: | ---: | ---: |
| CPU (AVX-512) | 53x slower | 55x slower | — | — |
| GPU (dp4a fallback) | 9x slower | **1.66x slower** | **1.8x** sgemm | **0.36x** dgemm |

On a CPU it is not worth it: fp64 costs only 2x fp32 and is
indistinguishable from exact at fp32 or fp64 output precision, so the 22-55x
buys only reproducibility. On a consumer GPU the arithmetic mix inverts —
int8 is 6x faster than fp32, fp64 is 12x slower — and exact fp64 lands
within 1.66x of `cublasDgemm` at ~5x the accuracy, on a *fallback* kernel.
With tensor-core int8 it would very likely win outright.

Faithful rounding (keep only the product bits that can affect a correctly
rounded binary result) cuts the GEMM count further: fp32 56→9, fp64
132→25. `limb-fp64-faithful` then **beats** native dgemm by ~2.8x at 1e-10–1e-12
relative error against the bit-exact product.

So the honest summary is: MFFT is the wrong algorithm for ML at every level
of the stack, but the fixed-point embedding it motivated is a live option
for exact and faithful fp64 on GPUs — which is a different and more useful
claim than the one this repository set out to test.

Where this shape of math *does* matter directly is **encrypted and
verifiable inference**. FHE schemes (BFV, BGV, CKKS) compute in
`Z_q[y]/(y^K + 1)` — the exact negacyclic ring MFFT is built on — with
ciphertext coefficients hundreds to thousands of bits wide, right where the
crossovers in this repository sit. Zero-knowledge proofs of inference have
the same profile.


## CPU ML faithful results (`n = 128`, `--ml --reps 2`)

Aggressive high-limb drop: fp32 keeps **one** top limb (1 GEMM); fp64 keeps a
**2-limb** MFFT window. Neither is bit-exact (`<- FAITHFUL`).

| method | products | seconds | vs packed | rel error | note |
| --- | ---: | ---: | ---: | ---: | --- |
| sgemm-packed | 1 | 0.0002 | 1.00x | 2e-07 | baseline |
| fp32->limbplane | 16 | 0.020 | 0.01x | 2.5e-08 | EXACT |
| fp32->karatsuba | 9 | 0.013 | 0.01x | 2.5e-08 | EXACT |
| fp32->mfft-rec | 36 | 0.044 | 0.00x | 2.5e-08 | EXACT |
| **fp32->mfft-faithful** | **1** | **0.0012** | **0.16x** | **3.2e-02** | top limb only |
| fp64->limbplane | 64 | 0.056 | 0.00x | 0 | EXACT |
| fp64->mfft-rec | 72 | 0.072 | 0.00x | 0 | EXACT |
| **fp64->mfft-faithful** | **32** | **0.029** | **0.01x** | **3.9e-06** | 2-limb MFFT |

Takeaway: dropping to one limb makes fp32 faithful **~15x faster** than exact
MFFT at this size, at the cost of ~3% relative error. fp64 faithful with two
limbs is **~2.5x faster** than exact MFFT with ~1e-6 error.

GPU uses the same policy (`limb-mfft-fp32-faithful` = 1 int8 GEMM of the top
7-bit limb; `limb-mfft-fp64-faithful` ≤ 2 limbs).


## GPU results (RTX 5070 Ti, n=4096, `--fp64 --reps 3`)

| method | GEMMs | ms | TFLOP/s | rel error | note |
| --- | ---: | ---: | ---: | ---: | --- |
| cublas-sgemm | 1 | 13.4 | 10.2 | 4e-07 | baseline |
| strassen-sgemm | 7 | 14.0 | 9.8 | 8e-07 | |
| cublas-dgemm | 1 | 168 | 0.82 | 8e-16 | |
| limb-fp64-exact | 64 | 140 | 0.98 | — | EXACT reference (skip-zero) |
| limb-mfft-fp64 | 512 | 13728 | 0.01 | 0 | EXACT, host-staged |
| limb-mfft-fp64-faithful | 128 | 1856 | 0.07 | 1.6e-10 | high limbs only |
| limb-fp32-exact | 56 | 122 | 1.13 | 4e-08 | EXACT |
| limb-mfft-fp32 | 128 | 1855 | 0.07 | 4e-08 | EXACT |
| limb-mfft-fp32-faithful | 64 | 901 | 0.15 | 5e-06 | high limbs only |
| limb-fp32-faithful | 9 | 25 | 5.6 | 5e-06 | schoolbook high limbs |
| limb-fp64-faithful | 25 | 60 | 2.3 | 1.6e-10 | schoolbook high limbs |
| ozaki-i8-s2 | 4 | 10.3 | 13.4 | 2e-05 | |
| ozaki-i8-s7 | 49 | 125 | 1.1 | 4e-16 | |

Schoolbook faithful still dominates MFFT faithful on GPU at these limb counts
(9 GEMMs / 25 ms vs 64 GEMMs / 901 ms for fp32). After the aggressive-faithful
fix (1 int8 GEMM for fp32 MFFT-faithful), the MFFT faithful row should move
much closer to `limb-fp32-faithful`.

n=256 is launch-overhead dominated; use `--n 4096` for throughput.

## GPU track (`cuda/`)

The penalty the CPU pays for exactness is a property of CPU arithmetic, not
of the algorithm: a CPU has one integer multiplier per lane, no int8
dot-product unit, and fp64 at only 2x the cost of fp32. A GPU inverts both
of those — int8 runs several times faster than fp32, and consumer fp64 runs
12x *slower* — which is exactly the trade a limb decomposition wants.
`cuda/gemm_bench.cu` measures the result, and it is much better than the CPU
number suggests: 9x an sgemm for exact fp32, 1.66x a dgemm for exact fp64,
and with faithful rounding **0.36x** a dgemm for a correctly-rounded-quality
fp64 product.

```sh
make -C cuda arch                          # what nvcc will target, and why
make cuda                                  # or: make -C cuda ARCH=sm_90
./cuda/gemm_bench --n 512 --reps 2 --check # correctness first
./cuda/gemm_bench --n 4096 --reps 5
./cuda/gemm_bench --n 4096 --reps 5 --fp64 # genuine 53-bit inputs
./cuda/gemm_bench --sweep-n --reps 3       # value bits / GEMMs / ms vs n
```

### What each GPU row is

These are **not** MFFT. MFFT is a transform of the limb *polynomial* and only
wins when the limb count `L` is large (hundreds). On the GPU track `L` is
6–12, so the limb convolution is plain **schoolbook** (`LA × LB` independent
int8 GEMMs). The names use the `limb-` prefix to make that explicit.

| row | algorithm | what it does |
| --- | --- | --- |
| `cublas-sgemm` / `cublas-dgemm` / `cublas-bf16` | vendor BLAS | Single cuBLAS call at the named precision. The ordinary baseline. |
| `strassen-sgemm` / `strassen-dgemm` | **standard Strassen** (1969) | Recursive 7-multiply scheme, cutoff 2048 (override with `--cutoff`). Leaf is the same cuBLAS GEMM. Not expected to beat cuBLAS on a GPU; kept as the recursive baseline. |
| `int8-dp4a` / `int4-in-dp4a` | quantised GEMM | Per-channel symmetric quantisation to int8/int4, accumulate with `__dp4a` (or cuBLAS int8 when available). Lossy inputs, exact int sum. |
| `limb-bf16-exact` / `limb-fp32-exact` / `limb-fp64-exact` | **fixed-point limb schoolbook** | Encode each entry into 7-bit digit planes (the post’s digit-plane idea), multiply every plane pair with an int8 GEMM, carry-normalise, decode. Bit-exact for the chosen significand width. Schoolbook over the limbs. |
| `limb-mfft-fp32` / `limb-mfft-fp64` | **MFFT** (post / this repo) | Same digit-plane embedding; limb polynomial evaluated via transform over the post’s `I_s` roots (signed permutations → negacyclic shifts), pointwise int32 GEMMs, inverse transform. `L` kept as-is; FFT length `NB = next_pow2(2·ceil(L/S))`. Product count `NB·K²`. Expected to *lose* to schoolbook at ML limb counts; present so the comparison is measured. |
| `limb-fp32-faithful` / `limb-fp64-faithful` | limb schoolbook + bit drop | Same as exact, but only the high limbs that can affect a correctly-rounded binary result (`sig + log₂ n + 4` product bits). |
| `ozaki-i8-s2/s4/s7` | **Ozaki scheme I** (2012) | Split each FP64 matrix into `s` int8 residual slices; run all `s²` pairwise int8 GEMMs; accumulate into double. |

CPU ML track uses different names for the same embedding idea with different
convolutions: `fp32->limbplane` (schoolbook), `fp32->karatsuba`,
`fp32->mfft-rec` (Schönhage–Strassen-style recursive plan). Only the CPU
rows that say `mfft` actually run MFFT.

The Makefile does not use `-arch=native`: that fails outright when the GPU is
newer than the toolkit (an RTX 50-series card reports `compute_120`, which
only CUDA 12.8+ can emit). It instead targets the newest architecture the
installed nvcc supports and also emits PTX, so the driver JITs forward onto
anything newer. JIT costs startup time, not throughput, and timings are
best-of-reps.

**The dp4a fallback autotunes.** When cuBLAS refuses int8, every limb
product runs through the built-in `__dp4a` kernel, so its throughput
multiplies through every exact row — at n=4096 the first version reached
20.5 TOP/s against a dp4a peak nearer 176. The kernel is now templated over
tile size, per-thread block and k-chunk depth, and `tune_dp4a()` times each
instantiation on the actual problem at startup and keeps the winner. That
matters because the right shape is not fixed: a 128x128 tile that wins at
n=8192 leaves 54 of 70 SMs idle at n=512 and loses by 2.8x. `--tile I`
forces a configuration, and the autotune table is printed so the choice is
inspectable.

**Three int8 paths, probed in order.** `cublasGemmEx` with
`CUBLAS_COMPUTE_32I` first; if that is refused, `cuBLASLt`, whose heuristic
query answers directly whether *any* algorithm exists for int8 -> int32 on
this toolkit/GPU pair (and whose result is then checked against a known
answer before being trusted, since a heuristic hit is not a guarantee);
failing both, the built-in `__dp4a` kernel. The active path is printed.

**cuBLAS first, with a fallback.** Every limb product goes through
`cublasGemmEx` with `CUDA_R_8I` inputs and `CUDA_R_32I` accumulation, so the
exact path rides the same tensor cores cuBLAS uses for inference. Some
architectures refuse that combination — RTX 50-series (sm_120) returns
`CUBLAS_STATUS_NOT_SUPPORTED` — so `probe_i8()` tests it once at startup and
falls back to a built-in `__dp4a` kernel, printing which path is in use.

The int8 -> fp32 cuBLAS variant is deliberately *not* used as a fallback: the
accumulators reach `n * 2^14`, which exceeds the 24-bit fp32 significand past
`n = 1024`, so it would silently stop being exact. Both supported paths
accumulate in int32. When the fallback is active the exact-path timings are an
upper bound on cost, not a measurement of what tensor cores would give.

**Limb choice.** Operands must fit in int8, so limbs are 7 bits and an entry
of `vbits` bits takes `L = ceil(vbits/7)` limbs, costing `LA x LB` int8
GEMMs. At `n = 4096`: fp32 is 8×7 = 56, bf16 is 6×5 = 30, fp64 is 12×11 =
132. The limb-strategy planner confirms 7-bit schoolbook minimises the
product count under the int8 sum constraint; every Karatsuba variant has
more products once operand sums must fit in signed int8.

**Faithful rounding** keeps only `sig_out + ceil(log2 n) + 4` product bits
by dropping low-order input limbs. At `n = 4096` that is 9 GEMMs for fp32
and 25 for fp64 — the numbers that let `limb-fp64-faithful` beat native dgemm.

**Karatsuba is deliberately not used on the GPU.** It forms sums like
`A0 + A1` of limb planes, which immediately exceed int8 and force the far
slower int32 path. 36 tensor-core int8 GEMMs beat 27 int32 ones. MFFT is
likewise inapplicable — at `L = 12` there is no convolution long enough to
transform.

That is the closing argument of the whole benchmark: **on the hardware ML
actually runs on, the limb count never reaches the regime where MFFT wins.**
It is not that MFFT is slow for ML; the ML parameter range never enters its
domain, at any level of the stack.

Both the limb arithmetic (encode, schoolbook, carry-normalising decode) and
the `__dp4a` kernel's tiling were validated on device: `--check --fp64` at
`n = 256` reports 4.5e-11 worst relative difference against a host float64
loop. Run `--check` first on a new GPU.

`limb-fp64-exact` runs the same machinery with a 53-bit significand: 12×11 = 132
int8 GEMMs at `n = 4096`. On a consumer card, where fp64 is throttled to a
small fraction of fp32, that against one `cublasDgemm` is a genuinely open
contest — and with faithful rounding the integer path wins it.

**LLM-scale sizes.** Current models sit well above the CPU track's range.
Llama-3 8B has hidden dimension 4096 with an MLP intermediate of 14336; 70B
is 8192; 405B is 16384. A training-step GEMM is therefore something like
`M = tokens_per_batch (4k-8k)`, `K = 4096`, `N = 14336` for the MLP
projection, and `K = N = 4096` for attention output. So square `n = 4096`
is a fair proxy and `n = 8192..16384` is the real upper end.

The CPU track runs at `n = 256..1024` for a mundane reason: the exact rows
cost `L^1.585` GEMMs, so `fp32->karatsuba` at `n = 4096` is 9 x 1.4e11
integer MACs -- minutes per measurement on one core. The GPU track is where
LLM shapes belong, and it defaults to `n = 2048` with `--n 4096` or higher
recommended. The benchmark is square; Device memory is checked up front
and roughly `n^2 x 110` bytes are needed (about 7 GiB at `n = 8192`, 29 GiB
at `n = 16384`), so a 16 GiB card tops out near `n = 8192`. Real LLM matmuls
are rectangular; only the square case is implemented.

Reported per method: GEMM count, milliseconds, TFLOP/s, and relative error
against the exact product, decoded to double so fp64 rows can be scored
meaningfully — an fp32 output cannot score below ~3e-08 however exact its
accumulation.

### GPU results

RTX 5070 Ti (sm_120, 70 SMs), CUDA 12.4, `n = 4096`, `--reps 5`. cuBLAS
rejects `CUBLAS_COMPUTE_32I` on this combination, so every limb GEMM runs on
the `__dp4a` fallback rather than tensor cores. The autotuner picked a 64x64
tile with 4x4 outputs per thread and a 128-deep k-chunk, at ~68 TOP/s.
Reference is the exact limb product, computed once outside the timed table.
`--fp64` regenerates the dataset with genuine 53-bit significands; the table
below is the default (fp32 promoted) run — timings are essentially identical
under `--fp64`.

| method | GEMMs | ms | TFLOP/s | rel error |
| --- | ---: | ---: | ---: | ---: |
| `cublas-sgemm` | 1 | 15.0 | 9.18 | 4.1e-07 |
| `strassen-sgemm` | 7† | 15.6 | 8.80 | 8.2e-07 |
| `cublas-dgemm` | 1 | 188 | 0.73 | 8.3e-16 |
| `strassen-dgemm` | 7† | **164** | **0.84** | 1.6e-15 |
| `cublas-bf16` | 1 | 6.4 | 21.3 | 2.1e-03 |
| `int8-dp4a` | 1 | 2.4 | 56.3 | 5.6e-03 |
| `int4-in-dp4a` | 1 | 2.5 | 55.7 | 1.0e-01 |
| `limb-bf16-exact` | 30 | 79.7 | 1.72 | 2.1e-03 |
| `limb-fp32-exact` | 56 | 151 | 0.91 | 4.1e-08 |
| `limb-fp64-exact` | 132 | 280 | 0.49 | reference |
| `limb-mfft-fp32` | 128 | 2244 | 0.06 | 4.1e-08 |
| `limb-mfft-fp64` | 512 | 16266 | 0.01 | 0.0e+00 |
| `limb-fp32-faithful` | **9** | **26.9** | **5.12** | 5.1e-06 |
| `limb-fp64-faithful` | **25** | **70.4** | **1.95** | 1.6e-10 |
| `ozaki-i8-s2` | 4 | 11.1 | 12.4 | 2.2e-05 |
| `ozaki-i8-s4` | 16 | 48.5 | 2.84 | 3.4e-10 |
| `ozaki-i8-s7` | **49** | **153** | **0.90** | **4.0e-16** |

† Standard Strassen, cutoff **2048** (one level → 7 leaf cuBLAS calls).
At this cutoff `strassen-dgemm` **beats** native dgemm; `strassen-sgemm`
is within ~4% of sgemm. With `--cutoff 512` (three levels / 343 leaves)
both are much slower — data movement dominates.

**MFFT rows** (`limb-mfft-fp32` / `limb-mfft-fp64`) are bit-exact (rel error
matches schoolbook) but slower at ML limb counts: more products
(128 vs 56 for fp32; 512 vs 132 for fp64) and each product is an int32 GEMM
rather than int8. That is the measured comparison the table was missing.
Roots of unity are never stored as complex numbers — see
[Roots of unity](#roots-of-unity-where-they-live) below.

Correctness: `--check --fp64` at `n = 256` reports a worst-case relative
difference of **4.5e-11** between the exact limb path and a host float64
loop. Table above is the `--fp64` (genuine 53-bit) run at `n = 4096`.

**Ozaki I at s=7 matches native dgemm accuracy (4e-16) and is faster**
(126 ms vs 169 ms) on the same dp4a fallback. At s=4 it undercuts
`limb-fp64-faithful` on wall-clock (41 vs 61 ms) at similar error (3e-10 vs
2e-10). The limb path’s advantage is bit-exact integer semantics and
predictable cost; Ozaki’s is fewer GEMMs when approximate accuracy is enough.

**Faithful rounding** keeps only `sig_out + ceil(log2 n) + 4` product bits:

| path | value bits kept | GEMMs | vs baseline |
| --- | ---: | ---: | --- |
| `limb-fp32-exact` | 54/48 | 56 | 9.0x sgemm |
| `limb-fp32-faithful` | 20/20 | **9** | **1.8x** sgemm |
| `limb-fp64-exact` | 82/76 | 132 | 1.66x dgemm |
| `limb-fp64-faithful` | 34/35 | **25** | **0.39x** dgemm (2.5x *faster*) |
| `ozaki-i8-s7` | — | **49** | **0.74x** dgemm, dgemm-level error |

`limb-fp64-faithful` at 60.8 ms beats native dgemm while staying within ~1e-10 of
the bit-exact product (Frobenius). `limb-fp32-faithful` is 1.8x an sgemm at
5e-06 relative error — coarser than exact's 2.5e-08 floor, but still far
tighter than bf16 (2e-03).

The reason exact fp64 competes at all is that consumer GPUs throttle fp64:
0.81 TFLOP/s against 10.16 for fp32 here, a 12.5x penalty the integer path
does not pay. The same comparison on a CPU, where fp64 costs only 2x fp32,
goes the other way entirely.

**Value bits grow with `n`.** Measured with `--sweep-n` (fp32-promoted,
best-of-3):

| n | vA/vB | LA×LB | exact | faith | sgemm | fp32ex | fp32fa | dgemm | fp64ex | fp64fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 40/40 | 6×6 | 36 | **9** | 0.01 | 0.27 | 0.10 | 0.05 | 0.68 | 0.22 |
| 512 | 39/41 | 6×6 | 36 | **9** | 0.04 | 0.46 | 0.18 | 0.38 | 1.16 | 0.39 |
| 1024 | 44/43 | 7×7 | 49 | **9** | 0.23 | 2.57 | 0.72 | 2.96 | 6.07 | 1.62 |
| 2048 | 48/45 | 7×7 | 49 | **9** | 1.70 | 16.2 | 4.03 | 23.6 | 43.5 | 10.0 |
| 4096 | 50/50 | 8×8 | 64 | **9** | 13.4 | 172 | 27.1 | 209 | 384 | 74.0 |
| 8192 | 55/53 | 8×8 | 64 | **9** | 135 | 1397 | 228 | 1722 | 3112 | 591 |

Exact GEMMs climb with the exponent spread; **faithful stays at 9** for
fp32 (`24 + log₂ n + 4`). From n=1024 up, `limb-fp64-faithful` beats native
dgemm by a growing margin (~2.9× at n=8192).

## Prior art

This repository sits on classical foundations; the post contributes a
*matrix* framing of those foundations, not a new transform.

**Classical (transform and multiprecision machinery):**

| piece | source |
| --- | --- |
| Cooley–Tukey FFT | Cooley & Tukey, *Math. Comp.* 1965 |
| Gentleman–Sande butterfly | Gentleman & Sande, 1966 |
| FFT multiplication over finite rings | Pollard, *Math. Comp.* 1971 |
| Schönhage–Strassen multiprecision multiplication | Schönhage & Strassen, 1971 |
| Nussbaumer negacyclic convolution | Nussbaumer, 1980 |
| Error-free transformation of matrix products (Ozaki scheme) | Ozaki, Ogita, Oishi, Rump, *Numer. Algorithms* 2012; INT8-TC realisations by Mukunoki, Ootomo, Uchino et al. |

The mid-sections of any modern NTT / multiprecision-FFT exposition will look
alike for the same reason: they share that textbook material. An independent
expository note on the NTT for lattice cryptography (e.g. arXiv:2509.05884)
covers cyclic and negacyclic convolution over `Z_q[x]/(x^n ± 1)`, roots
`ω` and `ψ`, CT/GS butterflies, and bit-reversed ordering — all classical.
It contains no matrix multiplication content.

**From the post (and implemented here):**

| piece | role |
| --- | --- |
| Digit-plane decomposition of *matrix* entries | turns a wide-scalar matmul into a convolution of narrow-scalar matmuls |
| Matrix-valued polynomial coefficients | the planes are themselves matrices |
| Roots of unity as signed permutation matrices `I_s` | the `H_{s,k}` recursion (`src/roots.c`, `--test-roots`) |
| Applying the transform along the *scalar-width* axis | not along the matrix dimensions |

None of that appears in the classical NTT literature or in the Ozaki line.
Ozaki is the closest *competitor* on the GPU track: it also reduces an
FP64 product to a sum of exact low-precision GEMMs, but by error-free
slicing rather than a fixed limb base. An Ozaki row is planned (item 9) so
the comparison is in-table.

## State of the art, and what is not implemented here

| result | status |
| --- | --- |
| `ω ≤ 2.371339` (Alman, Duan, Vassilevska Williams, Xu, Xu, Zhou 2025) | not implemented — galactic |
| Strassen 1969, 7 mults for 2×2 | `strassen` |
| Strassen–Winograd, 7 mults / 15 adds | `winograd` |
| Karatsuba / Toom on the limb polynomial | `karatsuba` |
| Schönhage–Strassen recursion (Nussbaumer-style negacyclic transform) | `mfft-rec` |
| Ozaki scheme I (error-free FP64 via int8 slices) | `ozaki-i8-s2/s4/s7` |
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
--fp64         genuine 53-bit double inputs (default promotes fp32)
--fp-width B   force a fixed B-bit fp32 grid instead of sizing from the data
--illcond E    widen the ML data exponent spread to E (costs limbs)
--tile I       (cuda) force a dp4a configuration instead of autotuning
--check        (cuda) cross-check exact limb path against a host float64 loop
--sweep-n      (cuda) scaling study: value bits / GEMMs / ms vs n
               (CPU) matrix-dimension sweep at the given --width
--cutoff C     (cuda) Strassen leaf size (default 2048)
--no-verify    skip exactness checks (reference is n^3 L^2)
--no-naive     skip the textbook methods
--only LIST    comma-separated methods to run, e.g. karatsuba,mfft-rec
--width W      bits per matrix entry (--bits is an alias)
--sweep-width  scalar-width sweep at the given --n (alias --sweep)
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
src/lowprec.c     exact bf16/int8/int4 GEMM, one specialised kernel each
src/kernel.c      integer kernels: ikj, blocked, packed, Strassen, Winograd
src/mlgemm.c      ML track: fp32 / bf16 / int8 GEMM with accuracy reporting
src/bigmat.c      big-integer storage, carry normalisation
src/main.c        CLI, verification, timing tables
cuda/gemm_bench.cu  GPU track: cuBLAS baselines + exact limb GEMMs on int8
                    tensor cores
```

## Conclusion — where MFFT is (and is not) practical

This repository started from a claim that an MFFT-style transform would make
*exact* matrix multiplication competitive with floating-point GEMM. After
CPU and GPU measurements across schoolbook, Karatsuba, Toom-3, MFFT, Ozaki,
Strassen, and quantised baselines, the picture is sharper.

### Where MFFT itself is the right tool

MFFT wins on **product count** only once the limb width is large. The measured
crossover against Karatsuba sits near **`L ≈ 512` limbs**. That regime is not
ML; it is:

| Domain | Why MFFT fits |
| --- | --- |
| **FHE / lattice crypto** (BFV, BGV, CKKS) | Ciphertexts live in the same ring MFFT is built on: \(\mathbb{Z}_q[y]/(y^K+1)\). Coefficients are hundreds–thousands of bits (`L` in the hundreds). Negacyclic NTT/MFFT is already the workhorse; matrix-valued coefficients (this repo’s “planes”) are the natural extension to *batched* linear algebra over ciphertexts. |
| **Multiprecision integer matmul** | Exact products of big integers (scientific computing, CAS, verifiable computation) with bit-widths past a few kilobits. Schoolbook is \(O(L^2)\); MFFT is \(\tilde O(L)\). |
| **Zero-knowledge proofs of linear algebra** | Same profile: wide integers, exact arithmetic, often already in a cyclotomic ring. |

In those settings the fused signed-permutation transform (one `±1` per
row/column, precomputed ops, no general multiplies for roots of unity) is a
real advantage over a classical complex FFT or a generic NTT with modular
multiplies.

### Where MFFT is the wrong tool

| Setting | Observed outcome |
| --- | --- |
| **ML embeddings (fp32 / fp64)** | `L = 4…12`. MFFT needs *more* products than schoolbook (e.g. 36 vs 16 at `L=4`). Pointwise matmul is ≥95% of runtime; transform fusion does not change the ranking. |
| **Quantised ML (bf16 / int8 / int4)** | `L = 1`. One GEMM is already exact for the integer sum; there is nothing for an FFT to reduce. |
| **Throughput-bound training** | Exactness is the opposite of the industry direction (fp32→bf16→int8). |

Karatsuba or hybrid Toom-3 are the better exact-convolution choices at ML
limb counts; both beat MFFT there on the CPU ML track.

### What *is* useful for “exact-ish” floating-point GEMM

The **limb embedding** that MFFT motivated is separable from MFFT itself:

* **Exact limb-schoolbook** on GPU int8 (dp4a): exact fp64 within ~1.7× of
  `cublasDgemm` on a consumer GPU (fallback kernel; tensor-core int8 would
  close the gap further).
* **Faithful high-limb truncation** (keep only product bits that can affect a
  correctly rounded binary result): fp32 56→9 GEMMs, fp64 132→25. GPU
  `limb-fp64-faithful` can **beat** native dgemm at ~1e-10–1e-12 relative
  error against the bit-exact product.
* **Ozaki residual slices**: fewer GEMMs when approximate accuracy is enough;
  a direct competitor to faithful limb plans.

So: **MFFT for crypto and multiprecision; limb embeddings (schoolbook /
Karatsuba / faithful) for exact or near-exact floating-point on GPUs.** The
transform is interesting engineering either way, but the efficiency claim
holds only in the first class of problems.


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

## Toom-Cook limb convolution (B2)

CPU: `conv_toom3` — recursive Toom-3 over int32 limb planes (5-point Bodrato
interpolation). Hybrid with Karatsuba so the row never loses on product count
at power-of-two L. Rows: `fp32->toom3`, `fp64->toom3` (both EXACT).

GPU: planner lists `toom3-1` and the `LA+LB-1` bound; runtime path remains
7-bit schoolbook (eval weights do not fit signed int8).


## Data modes (ML track)

| flag | distribution | exponent spread | notes |
| --- | --- | --- | --- |
| `--data uniform` (default) | U(-1,1) | wide | many limb bands; buckets lose |
| `--data narrow` | magnitudes in [0.5,1), random signs | ~0 | single band; buckets should win |
| `--illcond E` | multiply by 2^{U(-E/2,E/2)} | forced wide | stress exact vs fp32 |

Example: `./mfft-bench --ml --n 128 --data narrow --no-naive`
GPU: `./cuda/gemm_bench --n 4096 --data narrow --reps 3`

## MFFT phase profile (`--profile`)

Splits CPU MFFT time into pack / build_ops / transform / pointwise / fold.
At ML limb counts (L=4..8) pointwise dominates (~60–85%); the fused transform
is 5–30%. Op-list build is negligible.

```
./mfft-bench --ml --n 128 --reps 2 --no-naive --profile
```
