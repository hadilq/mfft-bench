# mfft-bench

A C benchmark for **MFFT** (Matrix Fast Fourier Transform), the matrix
multiplication method described in
[hadilq.com/posts/matrix-fast-fourier-transform](https://hadilq.com/posts/matrix-fast-fourier-transform/),
compared against schoolbook, limb-plane and Strassen multiplication.

Every method computes the **exact** product of two `n x n` matrices of
unsigned `B`-bit integers, and every result is checked against the textbook
baseline, so the timings compare algorithms rather than approximations.

## Quick start

```sh
nix develop                 # gcc, openblas, perf, hyperfine, valgrind
make && make check          # build + self-tests

nix build                   # or: nix run . -- --n 64 --bits 8192 --no-naive
./mfft-bench --test-roots
./mfft-bench --n 64 --bits 8192 --no-naive --no-verify
```

`make WITH_BLAS=1` adds a `cblas_dgemm` timing as a scale reference (it is
inexact 53-bit floating point, so it is reported separately, not ranked).
`make LIMB_BITS=1` builds the variant that works in base 2, i.e. the post's
literal "matrix of digits" model.

## What is implemented

Entries are `L` limbs in base `2^LIMB_BITS`. Writing each matrix as a
polynomial with small-entry matrix coefficients,

```
A = sum_u A_u * beta^u        beta = 2^LIMB_BITS
AB = sum_w ( sum_{u+v=w} A_u B_v ) * beta^w
```

so a big-integer matmul is a **convolution of matrix-valued polynomials**.

| method | what it does | `n x n` products |
| --- | --- | --- |
| `bigint-ijk` / `bigint-ikj` | textbook `n^3` matmul, each scalar product a schoolbook limb multiply | — (`n^3 L^2` limb MACs) |
| `limbplane` | the decomposition above, convolution done schoolbook | `L^2` |
| `mfft` | the decomposition above, convolution done by transform | `NB * K^2` |
| kernels `ikj` / `blocked` / `strassen` | the inner small-integer GEMM, shared by all plane methods | — |

The MFFT roots of unity are the post's `I_s`: signed permutation matrices
with `I_s^K = -1`, `K = 2^s`. `src/roots.c` implements the post's `H_{s,k}`
recursion verbatim and `--test-roots` verifies it against dense matrix
powers, checks the order is `2K`, checks `I_s^K = -1`, checks the powers
`I_s^0..I_s^{K-1}` are linearly independent, and regression-tests the sample
arrays printed in the post. All of that passes.

Because those powers are independent, ring elements are stored in the
**power basis** `sum_c v_c I_s^c`, where multiplying by `I_s^e` is exactly a
negacyclic shift of the coefficient vector — sign flips and index
arithmetic, no multiplications. So the transform never materialises an
`I_s^k` matrix.

## One correction to the post's cost analysis

The post treats `P_A(I_s^j) P_B(I_s^j)` as a single `n x n` matrix product.
It isn't. A value of the polynomial at `I_s^j` lives in `R (x) M_n(Z)` with
`R = Z[y]/(y^K + 1)`, i.e. it is a `K`-tuple of `n x n` matrices, so one
pointwise product is a length-`K` negacyclic convolution — `K^2` matrix
products, not one. That missing factor is where the claimed 21% saving at
`m = 16` comes from.

The fix is Schönhage–Strassen's balancing: decouple the transform length
from the ring dimension. Pack `S` limbs per polynomial coefficient,
transform over `NB = 2L/S` points in a ring of dimension `K = 2S`
(the `2` in each is the zero padding that turns the cyclic convolution into
the linear one we want). Total products:

```
NB * K^2 = 8 L S ,  minimised at S ~ sqrt(L/2)  ->  ~ 5.7 * L^1.5
```

against `L^2` for the schoolbook convolution. So MFFT is asymptotically
better in `L`, but only pays off past `L ~ 32` limbs. `mfft_plan_init()`
picks `S` automatically; `--sigma` overrides it.

## Measured results

Single core, gcc 13.3 `-O3 -march=native`, `LIMB_BITS=16`, `n = 64`.
Times in seconds, best of the three inner kernels for each method.

| entry bits | `L` | plan (`S`,`NB`,`K`) | products: mfft / plane | limb-plane | mfft | speedup |
| ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| 256 | 16 | 4, 8, 8 | 512 / 256 | 0.018 | 0.037 | 0.48x |
| 512 | 32 | 4, 16, 8 | 1024 / 1024 | 0.069 | 0.080 | 0.86x |
| 1024 | 64 | 8, 16, 16 | 4096 / 4096 | 0.289 | 0.301 | 0.96x |
| 2048 | 128 | 8, 32, 16 | 8192 / 16384 | 1.042 | 0.591 | **1.76x** |
| 4096 | 256 | 16, 32, 32 | 32768 / 65536 | 4.418 | 2.245 | **1.97x** |
| 8192 | 512 | 16, 64, 32 | 65536 / 262144 | 17.534 | 4.741 | **3.70x** |

Measured speedup tracks the product-count ratio closely, so the transform
overhead (`O(NB log NB * K * n^2)`) really is negligible next to the `n^3`
pointwise work at these sizes — the post's assumption there holds up.

**The post's own example (`m = 16`, base 2, `make LIMB_BITS=1`, n = 128):**

| method | kernel | products | seconds |
| --- | --- | ---: | ---: |
| limb-plane | blocked | 256 | 0.107 |
| mfft | ikj | 512 | 0.192 |

MFFT needs 2x the matrix products and runs ~1.8x slower. The predicted 21%
improvement does not appear; the corrected count says it cannot at that
width.

Two further notes on the practical picture:

* **Machine-word entries are hopeless for MFFT.** The whole method trades
  one wide multiply for many narrow ones, but a 64-bit `imul` costs the same
  as an 8-bit one on real hardware. MFFT only makes sense for entries wider
  than a word, which is why this benchmark is built around big-integer
  matrices.
* **Strassen composes with MFFT**, as the post predicts — `--kernel
  strassen` shaves a few percent on top at these `n`. It has little room to
  work with below `n = 128`; the win in this range is dominated by the
  reduction in the *number* of products, not their cost.

## Options

```
--n N          matrix dimension (default 64)
--bits B       bits per entry, multiple of LIMB_BITS (default 256)
--sigma S      MFFT block exponent override (block size = 2^S limbs)
--reps R       repetitions, best time reported
--cutoff C     Strassen base-case cutoff (default 64)
--seed X       PRNG seed
--no-verify    skip exactness checks (needed at large n: the reference is n^3 L^2)
--no-naive     skip the schoolbook big-integer methods
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
src/kernel.c      inner GEMM kernels: ikj, blocked, Strassen
src/bigmat.c      big-integer matrix storage, carry normalisation
src/main.c        CLI, verification, timing table
```

## Caveats

* Intermediates must fit in `int64`. `mfft_plan_maxbits()` reports the
  worst case and the driver warns past 62 bits; reduce `--n` or `--bits`, or
  build with a smaller `LIMB_BITS`.
* `L` must be a power of two, `n` is arbitrary (Strassen falls back to the
  blocked base case on odd sizes).
* Single-threaded throughout. The pointwise stage is embarrassingly
  parallel over the `NB` evaluation points, which is the obvious next step.
* Entries are unsigned. Signed support only needs a sign-magnitude split.
