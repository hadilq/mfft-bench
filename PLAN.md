# PLAN

Work queue arising from the `n = 4096` GPU run. Items are applied one per
commit, in order, each with a measurement that says whether it worked.

## What the n=4096 run actually said

```
cublas-sgemm       1 GEMM     13.386 ms   10.27 TFLOP/s   4.09e-07
cublas-dgemm       1 GEMM    168.552 ms    0.82 TFLOP/s   6.06e-16
fp64-exact       132 GEMMs   869.397 ms    0.16 TFLOP/s   (reference)
cublas-bf16        1 GEMM      5.984 ms   22.97 TFLOP/s   2.09e-03
int8-dp4a          1 GEMM      6.699 ms   20.52 TOP/s     5.57e-03
bf16-exact        30 GEMMs   197.386 ms    0.70 TFLOP/s   2.09e-03
fp32-exact        56 GEMMs   366.561 ms    0.37 TFLOP/s   2.53e-08
```

Four things to carry forward:

1. **fp64 is only 12.5x slower than fp32 here, not 64x.** So `cublas-dgemm`
   at 168 ms is the number `fp64-exact` has to beat, and at 869 ms it loses
   by 5.2x. With tensor-core int8 instead of the dp4a fallback (roughly 5-8x)
   the 132 GEMMs would land at 110-175 ms and the contest would be real. The
   whole fp64-exact case therefore rests on getting IMMA, or on cutting the
   GEMM count.
2. **The dp4a kernel is leaving most of the GPU on the table.** 20.5 TOP/s
   against a dp4a peak near 176 TOP/s is ~12%. Every exact row is 30-132
   invocations of this kernel, so its efficiency multiplies through
   everything. This is the single biggest lever in the file.
3. **Value bits grew with n**: 45/40 at n=512, 54/48 at n=4096 for fp32.
   More samples means a wider exponent spread, so `L` creeps up with matrix
   size. Worth measuring deliberately rather than noticing by accident.
4. **`fp64-exact` scored 0.00e+00** because it is now the reference. Same
   self-scoring flaw as `fp32-exact` had; the reference has to come from
   outside the table.

## On applying MFFT recursively to the int8 matrices

The proposal is to embed fp64 into a wide fixed-point value, convert to int8
limb planes, and then apply MFFT again to those int8 matrices. Two things
need separating here.

**Applying MFFT recursively *is* Schönhage-Strassen.** That is not an
alternative to it. MFFT reduces a wide-scalar matmul to a convolution of
narrow-scalar matmuls; the pointwise step of that convolution is itself a
convolution, and recursing there is exactly what `mfft-rec` does and exactly
what SSA is. There is no third option hiding between them.

**MFFT cannot be applied to the int8 planes themselves.** MFFT works along
the *scalar width* axis: it needs entries wide enough to split into limbs.
An int8 plane entry is 8 bits and already one limb — there is no width left
to decompose. Running MFFT there would be multiplying 8-bit numbers by
splitting them into 8-bit numbers.

**But there is a real second level, and it is worth exploring properly.** The
limb planes have to be *convolved*: `L_A x L_B` plane products, currently
schoolbook. That convolution is a polynomial product of length ~12, and
Karatsuba/MFFT do apply to it. The CPU track already does this and it is
where Karatsuba and `mfft-rec` win. The GPU does not, for one specific
reason: Karatsuba forms sums like `A0+A1` of limb planes, and with 7-bit
limbs those reach 254, which overflows the int8 operand format.

That is a constraint to be *measured*, not assumed. Narrowing the limbs to
6 bits keeps one Karatsuba level inside int8 but raises `L`, and the two
effects fight. A first estimate for fp64 at 83 value bits:

| limbs | width | Karatsuba levels | products |
| --- | ---: | ---: | ---: |
| 7-bit | 12 x 11 | 0 (sums overflow int8) | 132 |
| 6-bit | 14 x 13 | 1 (2x63 = 126 fits) | 192 |
| 5-bit | 17 x 16 | 2 (4x31 = 124 fits) | 576 |

so on this estimate schoolbook at 7 bits wins and Karatsuba is a trap on
int8 hardware. That is a genuinely interesting negative result if it holds
up, and item 4 below turns the estimate into a measurement.

MFFT proper needs `L` in the hundreds before it beats Karatsuba, and the
float embeddings top out at `L = 12`. So the honest expectation is that the
GPU stays schoolbook and the exploration produces a *bound* on how much
could ever be recovered there. The place to look for a real win is item 5:
cutting `L` itself.

## Prior art, and the arXiv:2509.05884 question

That paper ("Introduction to Number Theoretic Transform", Sengupta, Gupta,
Sengupta) is an expository note on the NTT for lattice cryptography. It
covers cyclic and negacyclic convolution in `Z_q[x]/(x^n +- 1)`, the roots
`omega` and `psi`, Cooley-Tukey and Gentleman-Sande butterflies, bit-reversed
ordering, and an NTT-multiplication example for Dilithium. It states in its
own introduction that it is "mostly taken from" Satriawan-Mareta-Lee's
beginner guide (2024) and Seiler (2018), and it cites both.

It contains no matrix multiplication content: no matrix-valued polynomial
coefficients, no signed-permutation matrix roots of unity, no bit-plane
decomposition of matrix entries. Its subject is polynomial multiplication
over `Z_q`.

So the overlap with the MFFT post is the shared classical foundation, not the
post's contribution. Negacyclic convolution over `Z[x]/(x^n+1)`, roots of
unity of order `2n`, and the CT/GS butterflies are all decades old --
Cooley-Tukey 1965, Gentleman-Sande 1966, Pollard 1971, Schonhage-Strassen
1971, Nussbaumer 1980 -- and the paper cites them. Two documents building on
the same textbook material will look alike in their middle sections without
either having copied the other.

What is distinctive about the post is the *matrix* framing: entries expanded
into digit planes, matrix-valued polynomial coefficients, and roots of unity
represented as signed permutation matrices `I_s`. None of that appears in
the paper. On the evidence, this is not plagiarism.

The useful action is the opposite one: **document the lineage explicitly.**
Item 7 adds a prior-art section to the README setting out which parts of
this repository are classical (SSA, Nussbaumer, CT/GS) and which come from
the post. That protects the post's contribution far better than a dispute
would, because it makes the boundary legible to a reader who arrives from
either direction.

## Items

### 1. fp64 beside fp32 everywhere, naming, and an independent reference

* Glossary: `sgemm`/`dgemm` (single/double GEMM, BLAS naming), `packed`
  (packed-panel micro-kernel), limb, plane, scalar width, `n`.
* CPU ML track: `dgemm-ijk`, `dgemm-blocked`, `dgemm-strassen` beside their
  fp32 counterparts, and `fp64->karatsuba` / `fp64->mfft` beside the fp32
  exact rows. Needs a 53-bit-significand encode (`__int128` staging, since
  `mi << sh` reaches 68 bits).
* One subtlety worth getting right: on the GPU the fp64 rows currently run
  on *doubles converted from the fp32 test data*, so their exact product is
  bit-identical to the fp32 one and the extra limbs buy nothing real -- the
  row measures cost without measuring benefit. The fp64 rows need genuinely
  double-precision inputs, which means a second dataset and a second
  reference. A `--fp64` mode that regenerates the data and runs the whole
  track in double is cleaner than interleaving two datasets in one table.
* Reference computed independently of every timed method, so no row scores
  itself. Mark the reference row explicitly.

*Measure:* every table has an fp64 row; no row reports 0.00e+00.

### 2. dp4a kernel throughput -- DONE (autotuned; awaiting numbers)

Templated the kernel over tile size, per-thread block and k-chunk depth,
padded the shared tiles, added `__launch_bounds__` and full unrolling, and
made `tune_dp4a()` time all six instantiations on the actual problem and
keep the winner. Shapes cannot be chosen analytically here -- the best tile
depends on SM count, shared budget and how much of the grid `n` fills -- so
this is measured rather than guessed. `--tile I` overrides.

All six instantiations were verified against a host reference.

*Measure:* the printed autotune table, `int8-dp4a` TOP/s at n=4096, and
`fp64-exact` against `cublas-dgemm`'s 168 ms. If the winner is still near
20 TOP/s the bottleneck is not the tile shape and the next step is double
buffering or `cublasLt` (item 3).

### 3. Try harder for tensor cores

cuBLAS refuses `CUBLAS_COMPUTE_32I` on sm_120 with CUDA 12.4. Probe
`cublasLtMatmul` and the `CUDA_R_32I`/`CUBLAS_COMPUTE_32I` combination
through cuBLASLt, which sometimes exposes kernels the legacy API does not.
Report clearly if it stays unavailable — a 5-8x factor rests on this.

*Measure:* whether `probe_i8` reports the cuBLAS path.

### 4. GPU limb-strategy planner

Enumerate limb width b in 4..7 crossed with schoolbook / 1-level / 2-level
Karatsuba, reject combinations whose operand sums leave int8, cost each as
`products x per-product cost`, pick the minimum and *print the table*. This
turns the argument above into a measurement and makes the negative result
inspectable instead of asserted.

*Measure:* the printed table, and whether the planner ever picks anything
other than 7-bit schoolbook.

### 5. Faithful-rounding mode

The exact path pays for bits far below the output ulp. `vbits = significand
+ spread`, and the spread comes from a few near-zero entries that cannot
move a correctly-rounded result. Drop limb planes that provably cannot
affect the rounded output, with a cancellation guard that falls back to the
exact path when the result is too small to certify.

This is the only item that attacks `L` itself, and `L` is quadratic in the
GEMM count: fp32 at 54 bits needs 56 GEMMs, at 30 bits it would need 20.

*Measure:* GEMM count and wall clock at equal output accuracy, plus a
verification that the rounded result matches the exact one on every entry.

### 6. Fuse the transform -- DONE

Correct observation from review: the transform currently nests four loops
(stage, block, butterfly, element) and then makes a *second* pass over the
same `K x nn` block inside `ring_shift32` to apply the twiddle, using a
`K x nn` temporary.

Because every twiddle is `I_s^e` -- a signed permutation with exactly one
`+-1` per row -- the whole twiddle is described by a shift `e`, and the pair
`(butterfly, ring coefficient)` determines destination and sign outright. So
precompute, once per plan, a flat per-stage table of
`(src_u, src_v, dst, sign)` and the loops collapse to three: stage, flat
pair index, element. The butterfly and the permuted store fuse into one
pass, and the temporary shrinks from `K x nn` to `nn` by walking the
rotation's cycles.

Expected: roughly half the transform's memory traffic, one less buffer, and
a contiguous innermost loop over `nn` that vectorises cleanly. This matters
most exactly where the transform is not negligible -- small `n`, large `L`,
and every level of `mfft-rec`, which pays transform cost at each recursion.

Implemented as a flat op list built at plan time, which collapses the four
nested loops to **two** -- one over ops, one over elements -- since the stage,
block and butterfly indices are all baked into the op. Cycle walking over
the rotation `c -> (c+e) mod K` keeps it correct in place, with one
`nn`-sized carry instead of a `K x nn` temporary.

Measured at `n = 32`, product count unchanged:

| width | mfft before | after | mfft-rec before | after |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.0592 s | 0.0575 s | 0.0331 s | 0.0313 s (-5%) |
| 8192 | 0.4359 s | 0.4321 s | 0.2138 s | **0.1673 s (-22%)** |

The split is exactly what the structure predicts. Flat `mfft` barely moves
(1%) because its cost is dominated by the `K^2` pointwise products, not the
transform. `mfft-rec` gains 22% because it pays transform cost at every
recursion level, which is where halving the passes compounds. The gain also
shrinks as `n` grows, since the transform is `O(L n^2)` against `O(n^3)` of
arithmetic -- so this matters most at small `n` and large width, which is
precisely the regime `mfft-rec` targets.

### 7. Prior-art section in the README

Set out plainly which parts are classical and which are the post's, with
citations: Cooley-Tukey, Gentleman-Sande, Schonhage-Strassen, Nussbaumer for
the transform machinery; the post for the matrix framing and the `H_{s,k}`
recursion. Cheap, and it is the right answer to the question above.

### 8. Scaling study

`--sweep-n` on the GPU: value bits, GEMM count and TFLOP/s against n, so the
`L` growth in observation 3 is quantified rather than anecdotal. Rectangular
`M/N/K` if time allows, since real LLM matmuls are not square.

## Not planned

* Recursive MFFT on the GPU limb convolution. `L = 12` is two orders of
  magnitude below where MFFT overtakes Karatsuba, let alone schoolbook.
  Item 4 will produce the numbers that justify leaving it out.
* fp16. It sits between bf16 and fp32 and would tell us nothing new.
