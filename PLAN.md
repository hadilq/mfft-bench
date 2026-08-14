# PLAN

Work queue arising from the `n = 4096` GPU run. Items are applied one per
commit, in order, each with a measurement that says whether it worked.

## What the n=4096 run actually said (updated after items 2/3/4)

```
cublas-sgemm            1     13.513 ms   10.17 TFLOP/s   4.09e-07
cublas-dgemm            1    169.483 ms    0.81 TFLOP/s   6.06e-16
fp64-exact            132    282.249 ms    0.49 TFLOP/s   (reference)
cublas-bf16             1      6.066 ms   22.66 TFLOP/s   2.09e-03
int8-dp4a               1      2.308 ms   59.56 TOP/s     5.57e-03
bf16-exact             30     65.571 ms    2.10 TFLOP/s   2.09e-03
fp32-exact             56    121.565 ms    1.13 TFLOP/s   2.53e-08
```

After the dp4a autotune (item 2) and the robust Lt fallback (item 3):

1. **fp64-exact is now 1.66x cublas-dgemm** (282 ms vs 169 ms), down from
   5.2x. The contest is real on a consumer card even without IMMA tensor
   cores. Cutting the GEMM count (items 4/5) is the remaining lever.
2. **dp4a reached 59–67 TOP/s** (winner: 64x64 tile, 4x4/thread, k=128).
   That is ~3x the previous 20.5 TOP/s and ~34% of a theoretical ~176 TOP/s
   peak; the exact rows scaled almost linearly with it.
3. **Value bits at n=4096**: fp32 54/48 → 8x7=56 GEMMs; fp64 83/77 →
   12x11=132 GEMMs. Confirmed growth with n (item 8 still to quantify).
4. **Reference self-scoring** is already special-cased in the printer for
   `fp64-exact`; item 1 still wants a fully independent host reference and
   genuine double inputs.

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
effects fight. Measured for fp64 at 83/77 value bits (item 4):

| limbs | width | Karatsuba levels | products |
| --- | ---: | ---: | ---: |
| 7-bit | 12 x 11 | 0 (sums overflow int8) | **132** |
| 6-bit | 14 x 13 | 1 (2x63 = 126 fits) | 147 |
| 5-bit | 17 x 16 | 2 (4x31 = 124 fits) | 180 |
| 4-bit | 21 x 20 | 2 (4x15 = 60 fits) | 270 |

Schoolbook at 7 bits wins; Karatsuba is a trap on int8 hardware for the
widths the float embeddings reach. The same holds for fp32 (56 vs 60/81).
No runtime Karatsuba path is required.

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

## README reconciliation (done alongside item 6)

Two analysis sections -- the low-precision collapse argument and the
"is MFFT useful for ML" conclusion -- were silently dropped by a
range-replacement in an earlier commit. Restored and rewritten against
current data. The headline that changed: exact matmul costs 53x an sgemm on
CPU but 9x on GPU, and exact fp64 costs 1.66x a dgemm on GPU against 55x on
CPU. The CPU number was being quoted as if it were a property of the
algorithm; it is a property of CPU arithmetic.

## Backlog

### B1. Why the transform fusion did not move the fp32/fp64 benchmarks

Raised on review: item 6 cut `mfft-rec` by 22% at `n = 32`, width 8192, yet
the fp32 and fp64 rows did not move at all. Measured, at `n = 256`:

| row | before | after |
| --- | ---: | ---: |
| `fp32->mfft` | 0.11438 s | 0.11320 s (-1%) |
| `fp32->karatsuba` | 0.02346 s | 0.02356 s (0%) |
| `fp64->karatsuba` | 0.088115 s | 0.088118 s (0%) |

The first-order answer is that **almost nothing in those benchmarks runs the
transform**:

* The **GPU** path has no transform at all. `exact_float_gemm` issues the
  `LA x LB` limb products schoolbook, because Karatsuba's operand sums leave
  int8 and MFFT needs `L` two orders of magnitude larger than the 8-12 the
  float embeddings reach. Nothing in `src/mfft.c` is on that path, so no
  change there could have moved `fp32-exact` or `fp64-exact`.
* On the **CPU**, the fp32/fp64 embeddings land at `L = 4..12`, where the
  planner picks Karatsuba. `fp32->karatsuba` and `fp64->karatsuba` call
  `conv_karatsuba`, which never touches the transform either.
* Only `fp32->mfft` uses it, and there `L = 4` gives `NB = 4, K = 4`: the
  transform is a small fraction of 64 pointwise `n^3` products, so 1% is
  about what the structure predicts.

That explains the observation but leaves a real question open, which is what
this item is for: **is the fused transform actually twice as fast per byte,
as halving the passes should give?** The 22% on `mfft-rec` is consistent
with a 2x transform that was ~44% of runtime, but also with a 1.3x transform
that was ~85%. Those imply very different next steps.

Method:

* Add `--profile` splitting each method's time into transform / pointwise /
  encode-decode, so the transform's share is measured rather than inferred.
* Time the transform alone across `(n, L)` to get its cost per byte before
  and after, independent of what fraction of some benchmark it is.
* Check the op-list build overhead. `build_ops` now runs on every
  `ssa_negconv` invocation, and the top level calls that `NB` times per
  level -- cheap in theory (`O(NB log NB * K)` against `O(NB log NB * K * n^2)`
  of transform work) but never measured, and it is pure overhead the old code
  did not have. If it shows up, cache the lists per `(NB, K, g)`.
* Sweep `n` at fixed `L` to find where the transform stops mattering, which
  also tells us the `(n, L)` region where `mfft-rec` is worth choosing at all.

### B2. Toom-Cook for the limb convolution

Karatsuba is not the floor. A linear convolution of an `LA`-term and an
`LB`-term sequence needs only `LA + LB - 1` multiplications by evaluation
and interpolation, against Karatsuba's `L^1.585` and schoolbook's `LA x LB`:

| case | schoolbook | Karatsuba | interpolation bound |
| --- | ---: | ---: | ---: |
| fp32 CPU, L=4 | 16 | 9 | **7** |
| fp64 CPU, L=8 | 64 | 27 | **15** |
| fp32 GPU, 8x7 | 56 | n/a (int8) | **14** |
| fp64 GPU, 12x11 | 132 | n/a (int8) | **22** |

That is a 4-6x reduction at exactly the widths the float embeddings occupy
-- the regime where MFFT cannot help and Karatsuba barely can. It is very
likely the largest remaining win in the repository.

The catch is the same one that rules out Karatsuba on the GPU: evaluation at
points `0, +-1, +-2, ...` forms weighted sums of limb planes that outgrow
int8, and interpolation needs exact divisions. Both are manageable in
principle -- smaller limbs buy headroom, and the divisions are by small
constants that divide exactly over Z -- but the trade has to be measured,
not assumed. Fold into item 4's planner rather than treating separately.

## Items

### 1. fp64 beside fp32 everywhere, naming, and an independent reference -- DONE

* Glossary: already in README (`sgemm`/`dgemm`, `packed`, limb, plane, …).
* CPU ML track: `dgemm-ijk` / `blocked` / `strassen` / `packed` beside the
  fp32 counterparts; `fp64->karatsuba` / `fp64->limbplane` / `fp64->mfft`
  beside the fp32 exact rows. 53-bit encode uses `__int128` staging in
  `fpfixed.c`.
* **GPU:** `--fp64` regenerates the dataset with genuine 53-bit
  significands. Default still promotes fp32 (cost-only). Reference is
  always computed once via the exact limb path *outside* every timed
  method.
* **CPU:** same `--fp64` flag on the ML track. Dual independent
  references: fp32 embedding → `R` for float rows; fp64 embedding →
  `Rd64` for double rows. Neither is produced by a timed method.
* Output labels which data mode is active on both tracks.

*Measure:* every table has an fp64 row; no row reports 0.00e+00 against
itself. Host check at n=256 on GPU: 4.5e-11 relative difference.

### 2. dp4a kernel throughput -- DONE

Templated the kernel over tile size, per-thread block and k-chunk depth,
padded the shared tiles, added `__launch_bounds__` and full unrolling, and
made `tune_dp4a()` time all six instantiations on the actual problem and
keep the winner. Shapes cannot be chosen analytically here -- the best tile
depends on SM count, shared budget and how much of the grid `n` fills -- so
this is measured rather than guessed. `--tile I` overrides.

All six instantiations were verified against a host reference.

*Measured at n=4096 (RTX 5070 Ti, sm_120):*
```
  128x128 tile, 8x8/thread, k=32        2.669 ms    51.49 TOP/s
  128x128 tile, 8x8/thread, k=64        2.370 ms    57.99 TOP/s
  128x128 tile, 8x8/thread, k=128       2.219 ms    61.93 TOP/s
  64x64 tile, 4x4/thread, k=64          2.151 ms    63.89 TOP/s
  64x64 tile, 4x4/thread, k=128         2.037 ms    67.47 TOP/s   <- winner
  32x32 tile, 2x2/thread, k=128         3.627 ms    37.89 TOP/s
int8-dp4a  2.308 ms  59.56 TOP/s
fp64-exact 282 ms (was 869 ms) vs cublas-dgemm 169 ms  → 1.66x
```

### 3. Try harder for tensor cores -- DONE

`probe_i8` now tries three paths in order: legacy `cublasGemmEx`, then
cuBLASLt, then dp4a. The cuBLASLt attempt uses
`cublasLtMatmulAlgoGetHeuristic`, which answers the question directly -- if
no algorithm comes back for int8 -> int32, the capability genuinely is not
exposed. A heuristic hit is not proof the kernel works, so the result is
checked against a known answer before being trusted. On sm_120 the heuristic
can return an algo that later fails with CUBLAS_STATUS_NOT_SUPPORTED; the
probe now catches that without aborting and falls back cleanly.

*Measured:* path is `__dp4a` (cuBLAS and cuBLASLt both refuse
CUBLAS_COMPUTE_32I / equivalent on this toolkit/GPU pair). No tensor-core
int8→int32. The dp4a path is therefore the production path for exact rows.

### 4. GPU limb-strategy planner -- DONE

Enumerate limb width b in 4..7 crossed with schoolbook / 1-level / 2-level
Karatsuba, reject combinations whose operand sums leave int8, cost each as
`products x per-product cost`, pick the minimum and *print the table*. This
turns the argument above into a measurement and makes the negative result
inspectable instead of asserted.

*Measured at n=4096:*
- fp32 (54/48 value bits): winner **7-bit schoolbook (56 products)**.
  6-bit karatsuba-1 = 60, 5-bit karatsuba-2 = 81; both lose.
- fp64 (83/77 value bits): winner **7-bit schoolbook (132 products)**.
  6-bit karatsuba-1 = 147, 5-bit karatsuba-2 = 180; both lose.

The planner never selects anything other than 7-bit schoolbook for the
float embeddings that appear in practice. No Karatsuba (or narrower-limb)
runtime path is required; the negative result holds. Runtime stays on the
existing 7-bit schoolbook int8 path.

### 5. Faithful-rounding mode -- DONE (GPU)

The exact path pays for bits far below the output ulp. `vbits = significand
+ spread`, and the spread comes from a few near-zero entries that cannot
move a correctly-rounded result. Drop limb planes that provably cannot
affect the rounded output, with a cancellation guard that falls back to the
exact path when the result is too small to certify.

This is the only item that attacks `L` itself, and `L` is quadratic in the
GEMM count: fp32 at 54 bits needs 56 GEMMs, at 30 bits it would need 20.

**GPU implementation:** `plan_limbs_faithful` keeps
`sig_out + ceil(log2 n) + 4` product bits by dropping low-order *input*
limbs (high limbs only enter the schoolbook). New rows `fp32-faithful` and
`fp64-faithful` are timed and scored against the full exact reference.
Per-entry cancellation fallback is deferred; mismatches against the exact
reference will show up in the rel-error column.

*Measure:* GEMM count and wall clock at equal output accuracy, plus a
verification that the rounded result matches the exact one on every entry.
*Run:* `./cuda/gemm_bench --n 4096 --reps 5` and compare
`fp32-faithful` / `fp64-faithful` GEMMs and error to the exact rows.

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

### 7. Prior-art section in the README -- DONE

Set out plainly which parts are classical and which are the post's, with
citations: Cooley-Tukey, Gentleman-Sande, Schonhage-Strassen, Nussbaumer for
the transform machinery; the post for the matrix framing and the `H_{s,k}`
recursion. Ozaki line cited as the closest GPU competitor (item 9).
Also refreshed the GPU results table with faithful-rounding numbers.

### 8. Scaling study -- DONE

`--sweep-n` on the GPU: value bits, GEMM count and wall-clock against n for
sgemm / fp32-exact / fp32-faithful / dgemm / fp64-exact / fp64-faithful over
{256, 512, 1024, 2048, 4096, 8192} (skips sizes that do not fit). Rectangular
`M/N/K` left for a later pass if needed.

*Run:* `./cuda/gemm_bench --sweep-n --reps 3`

### 9. Ozaki scheme as a competitor row (GPU) -- DONE (Ozaki I)

The limb/schoolbook path is one exact strategy. The literature standard for
FP64-accurate GEMM on low-precision matrix units is the Ozaki family
(Ozaki 2012; Mukunoki et al.; Ootomo/Ozaki/Yokota; Uchino/Ozaki/Imamura).

**Implemented:** Ozaki I (global-scale residual slicing) on the existing
`igemm_rm` / dp4a path. Each FP64 matrix is split into `s` int8 slices by
repeated residual extraction; all `s²` pairwise products accumulate into a
double matrix. Rows: `ozaki-i8-s2` (4 GEMMs), `ozaki-i8-s4` (16),
`ozaki-i8-s7` (49). Ozaki II (CRT) and per-row/col scaling left as
follow-ups if global-scale accuracy is weak on ill-conditioned data.

*Measure:* product count, wall-clock, and rel-error vs `fp64-exact` and
`cublas-dgemm` at n=4096. *Run:* `./cuda/gemm_bench --n 4096 --reps 5`.

### 10. Recursive fp32 / fp64 baselines (Strassen and friends)

The CPU ML track already has `sgemm-strassen` and `dgemm-strassen`. The GPU
table and the exact track do not yet expose recursive algorithms for the
inexact FP32/FP64 baselines, and the exact path has no recursive variant
either (MFFT-rec on the limb convolution remains Not planned because L≤12).

Plan of work:

1. **GPU Strassen (or Winograd) for `cublas-sgemm` / `cublas-dgemm`
   competitors** — a pure recursive baseline that bottoms out in the same
   cuBLAS (or dp4a) leaf. Useful for scale comparison and for the
   “ordinary method vs recursive” story the CPU track already tells.
2. **CPU completeness** — ensure `fp32->mfft-rec` / `fp64->mfft-rec` (or
   the existing `mfft-rec` under the ML embedding) appear as named rows
   when `--ml` is run, so the recursive exact algorithms are visible in
   the same table as the schoolbook / Karatsuba exact rows.
3. Do **not** add recursive MFFT on the GPU limb convolution (already
   ruled out by item 4).

*Measure:* Strassen crossover vs plain GEMM at the n values used in the
tables; presence of at least one recursive exact row on CPU and one
recursive inexact row on GPU.

## Not planned

* Recursive MFFT on the GPU limb convolution. `L = 12` is two orders of
  magnitude below where MFFT overtakes Karatsuba, let alone schoolbook.
  Item 4 produced the numbers that justify leaving it out.
* GPU Karatsuba / narrower-limb int8 path. Item 4 measured that 7-bit
  schoolbook has the fewest products for every float embedding that appears;
  implementing the other strategies would only make the exact rows slower.
* fp16. It sits between bf16 and fp32 and would tell us nothing new.
