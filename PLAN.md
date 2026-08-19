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

**Applying MFFT recursively is *SS-family* (FFT convolution on digits), not
a literal 1971 Schönhage–Strassen integer implementation.** See README.
That is not an
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

### B1. Why the transform fusion did not move the fp32/fp64 benchmarks -- DONE

**Measurement (`--profile`, n=64, L=4/8):**

| method | pack | build_ops | transform | pointwise | fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32->mfft-rec (L=4) | 0.3% | 0.1% | **5%** | **84%** | 11% |
| fp64->mfft-rec (L=8) | 0.5% | ~0% | **29%** | **60%** | 11% |

Conclusions:
1. At float-embedding widths the transform is a minority of runtime; a 2x
   faster transform would move fp32->mfft by ~2.5% and fp64->mfft by ~15%.
2. `build_ops` is noise (~0%); caching the op-list is not worth a special path.
3. Pointwise leaf GEMMs dominate — that is why Karatsuba/Toom beat MFFT here.
4. The original 22% gain of item 6 was on width-8192 `mfft-rec`, where the
   transform share is large; it never applied to the ML rows.

`--profile` enables the phase split on CPU MFFT rows.

### B2. Toom-Cook for the limb convolution -- DONE (CPU; GPU planner only)

**Status (item B2):** CPU `conv_toom3` is recursive Toom-3 (Bodrato 5-point
interpolation, exact divisions by 2 and 3) with a hybrid that falls back to
Karatsuba when it has fewer leaf products.  Wired as `fp32->toom3` /
`fp64->toom3` in the ML track; product counts print next to karatsuba/mfft.
Measured at n=64: both rows EXACT (err matches limbplane).  At L=4 hybrid
selects Karatsuba (9 < 15); at L=8 Toom-3 has 25 vs 27 products.

GPU planner now lists `toom3-1` and the `toom-opt = LA+LB-1` bound.  Runtime
path stays 7-bit schoolbook: eval weights (7·maxv) overflow signed int8 for
every float embedding, same reason Karatsuba never wins on GPU.

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

### B3. Limb-axis odd/even (DIT) split + hybrid with Karatsuba/Toom -- IN PROGRESS (CPU)

**Motivation.** Classical Cooley–Tukey even/odd index splitting applies to the
*limb polynomial*, not only to MFFT block coefficients. High/low Karatsuba and
Toom-3 already split the digit vector a different way; an explicit even/odd
index basis is a third option. Hybrid dispatch (pick min product count among
schoolbook / Karatsuba / Toom / evenodd / limb-FFT) is the practical end state.

**References (primary).**
- Cooley & Tukey (1965), Math. Comp. 19 — DIT even/odd time indices.
- Gentleman & Sande (1966), AFIPS — DIF; matches our forward `fft_run*` notes.
- Karatsuba & Ofman (1962/63) — 3 half-size products (high/low digits).
- Toom (1963), Cook (1966); Bodrato (2007 WAIFI/ISSAC) — Toom-3 evaluation order.
- Schönhage & Strassen (1971), Computing 7:281–292 — FFT multiprecision + packing.
- Harvey & van der Hoeven (2021), Ann. of Math. — O(n log n) integer mul.
- Knuth, TAOCP Vol. 2 §4.3.3 — textbook survey.

**Phases.**

0. **Metrics.** Limb-product count, wall time, rel error vs bit-exact.
   Regimes: ML `L∈{4,8,12}` (expect little gain); exact/crypto `L∈{64,256,512,1024}`.

1. **Even/odd index convolution (CPU).** `conv_evenodd` in `methods.c`:
   one-level then recursive. Wire exact-track / ML rows. Compare product
   counts to schoolbook and high/low Karatsuba (different bases).

2. **SSA limb-aware option.** Document ring-only recursion (current, profitable).
   Optional entry-splitting MFFT when pointwise entries are multi-limb
   (exact track, small LIMB_BITS, large L) — threshold study vs `mm_accum`.

3. **Limb-axis FFT for large L.** Reuse `build_ops` / `mfft_gpu.cuh` with
   transform length from limb count. Benchmark vs Karatsuba/Toom/MFFT-packed
   at L=256…1024. This is where \(\tilde O(L)\) should appear.

4. **Hybrid dispatcher (the interesting combination).**
   At each recursive width `L`, choose the strategy with the fewest *estimated*
   leaf products among:
   - schoolbook (`L²`)
   - Karatsuba high/low (`~3(L/2)^α`)
   - Toom-3 (`~5(L/3)^α` with Bodrato overhead)
   - evenodd index (`~3(L/2)^α` but different add structure)
   - limb-FFT / MFFT when `L ≥ L_fft_cutoff` (product count from plan)
   Same pattern as existing Toom↔Karatsuba hybrid: never lose on product count
   at power-of-two L. Optional cost model weights adds/shifts if product count
   ties.

5. **GPU** only if Phase 1–4 win at large L; port winner to int8 planes.

6. **Write-up.** Product-count table vs L; short blog note distinguishing
   CT even/odd indices vs Karatsuba high/low vs SSA ring recursion.

**Expectation.** Hybrid does **not** invent a new asymptotic; it picks the
best known D&C at each size (GMP/MPFR-style). Wins show up as a smoother
crossover curve and fewer bad L values, not as beating FFT asymptotically.

**Status (CPU):** Recursive hybrid live. `hybrid_rec` picks strategy at every
subproblem size; pure Kara/Toom/evenodd remain pure (`g_hyb_children`). Exact
track rows: toom3, evenodd, hybrid. Product count is recursive min (e.g. L=8:
hybrid 15 vs pure Toom 25 / Kara 27). Verified exact at L=16. Large-L FFT and
GPU still open.

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

### 10. Recursive fp32 / fp64 baselines (Strassen and friends) -- DONE

1. **GPU:** `strassen-sgemm` and `strassen-dgemm` recurse with cutoff 512,
   leafing into `cublasSgemm` / `cublasDgemm`. Materialise quadrants into
   contiguous buffers; 7 multiplies per level.
2. **CPU:** exact embedding rows renamed `fp32->mfft-rec` / `fp64->mfft-rec`
   (they already used `mfft_plan_init_rec`). `sgemm-strassen` /
   `dgemm-strassen` were already present.
3. Recursive MFFT on the GPU limb convolution remains Not planned (item 4).

*Measure:* Strassen vs plain GEMM at n=4096; recursive exact rows visible
on CPU `--ml` and recursive inexact rows on GPU.

### 11. GPU MFFT path (required for fair comparison) -- MOSTLY DONE

`limb-mfft-fp32` is correct (rel error matches `limb-fp32-exact`) and uses:

1. `L` padded to next power of two; extra limb planes are zero.
2. Host-built fused op list from signed-permutation / negacyclic-shift
   structure; uploaded once; timed path never recomputes roots.
3. Device fused FFT kernels with register-local carry — one launch each.
4. Pointwise: `NB·K²` tiled int32×int32→int64 GEMMs (bit-exact).
5. Device fold `/NB` + `k_decode`.

At natural ML limb counts (L≈8–16) MFFT issues *more* products than
schoolbook (128 vs 56 at L=8) and each product is denser (int32 vs int8),
so it loses on wall-clock — the measured comparison the table needed.

`limb-mfft-fp64` added (same pipeline, `k_encode_d` / `k_decode_d`).
Roots of unity are cached as a fused op list (negacyclic shifts + signs);
`src/roots.c` remains verification-only (`--test-roots`).

Plan no longer forces `L` to a power of two: `ncoeffs = ceil(L/S)`,
`NB = next_pow2(2·ncoeffs)`, σ chosen to minimise `NB·K²`. At L=12 the
best feasible plan is still 512 products (power-of-2 FFT constraint).

### 12. (Optional follow-ups after 11)

* Rectangular `M×N×K` end-to-end (pad each side).
* Recursive pointwise (SSA) on GPU once L is large enough to matter.
* Per-row/col Ozaki scaling; Ozaki II (CRT).
* Further int32 pointwise kernel tuning (MFFT).

### 13. Skip all-zero limb planes -- DONE

Implemented in `exact_float_gemm` / `exact_double_gemm`:

1. `k_plane_nonzero` — block-local OR + one atomic per plane.
2. After encode, detect live A/B planes; skip transpose and GEMM for zeros.
3. `*gemms` is the **launched** count; if less than LA×LB, print
   `skip-zero: active A a/A  B b/B  GEMMs launched (of full)`.
4. Applies to both exact and faithful (same functions, trimmed plan).

Random `U(-1,1)` usually fills the band → few/no skips (detection cost
only). Narrow-exponent data is where wall-clock should drop.

### 14. Skip-zero + k-buckets on MFFT -- DONE (MVP)

- Live L_eff plan rebuild in dense MFFT (fp32/fp64).
- limb-mfft-bucket-fp32 / limb-mfft-bucket-fp64: forward MFFT, per-band gather + rectangular pointwise.
- Standalone limb-bucket-fp32 remains schoolbook-only.
- Random data: buckets exact but slower; narrow-exponent is the intended win case.

**Narrow data mode (`--data narrow`):** magnitudes in [0.5, 1) with random signs
so every finite entry shares the same binary exponent (spread A=0 B=0 on CPU).
Default remains `uniform` = U(-1,1).  Available on CPU (`mfft-bench --ml`) and
GPU (`gemm_bench`).  Re-measure `limb-bucket-*` / `limb-mfft-bucket-*` under
`--data narrow` to see the intended wall-clock win.


### 17. Aggressive faithful + recursive pointwise MFFT -- IN PROGRESS

**Aggressive faithful (CPU):** `faithful_limb_window(..., aggressive=1)` keeps
at most 2 power-of-2 limbs of the highest live band so FAITHFUL rows show
non-zero error (fp64 ~1e-6, fp32 ~1e-7) instead of silently matching exact.

**Recursive pointwise MFFT (design):** each of the NB·K² pointwise products is
an n×n integer matrix multiply.  In principle those matrices can again be
written as polynomials in a smaller base and transformed with MFFT.  Wins only
when *entries* are multi-limb (large integers); with LIMB_BITS=16 float
embeddings the pointwise entries fit in int32 and a second MFFT (L=2) costs
more than one packed GEMM.  The existing `mfft_plan_init_rec` + `ssa_negconv`
already recurses on the *ring* dimension (negacyclic K-conv), which is the
profitable recursion for this embedding.  Optional future: entry-splitting
pointwise MFFT for the exact big-integer track with LIMB_BITS=1..8 and large L.

### 14-design (reference)

**Idea.** A single fp32 entry only occupies ~4 consecutive 7-bit limbs
(24-bit significand). Different entries sit in different bands because of
different exponents. Group entries by band, run a short schoolbook
(`≈4×4` products) *inside each bucket*, then scatter-add into C.

This is a **new method**, not a tweak of the dense-plane path. Closest
cousin in the table: Ozaki residual slices (value splitting rather than
exponent bucketing).

**Design.**

1. **Band definition.**
   - Limb index of the least-significant non-zero digit of an entry
     (or `floor((exponent - S) / LIMB_BITS)`).
   - Band width `W = ceil(sig_bits / LIMB_BITS)` (fp32: W=4; fp64: W=8).
   - Entry with base limb `b` lives in band bucket `b` and only needs
     limbs `[b, b+W)`.

2. **Bucket construction** (host or device prefix-sum):
   ```
   for each entry (r,c):
     b = base_limb(A[r,c])           // or joint key with B if paired
     bucket[b].append((r,c))
   ```
   For matmul `C = A·B` the natural split is on **rows of A** and **columns
   of B** by their base limb, then the product of bucket `(bA, bB)` writes
   into result limbs starting at `bA+bB`.

3. **Per-bucket GEMM.**
   - Extract (or gather) the sub-rows / sub-cols that belong to the bucket
     into a dense rectangular panel, **or** use a masked/compacted layout.
   - Run schoolbook with only `W×W` digit products on that panel.
   - Scatter-add the digit results into the global limb accumulators for C.

4. **Cost model (rough).**
   - Dense schoolbook: `LA × LB` full `n×n` GEMMs.
   - Buckets: `∑_{bA,bB} W² · gemm(n_{bA}, n, n_{bB})` roughly, with
     `∑ n_{bA} = n`. If exponents are uniform over R bands, each bucket
     is ~n/R rows → arithmetic drops like `W²/R` vs `LA·LB`, but panel
     extraction and many small GEMMs add overhead.
   - Wins when R is moderate and panels stay large enough for GPU efficiency
     (n_bucket ≳ 256).

5. **Benchmark rows.**
   - `limb-bucket-fp32` / `limb-bucket-fp64` (exact within the limb model).
   - Log: number of buckets, min/median/max bucket size, GEMMs launched,
     ms, rel error vs the same reference.

6. **Implementation order.**
   1. Host prototype on CPU `--ml` track (easier gather/scatter, validate
      counts and error).
   2. GPU: encode + base-limb kernel; histogram + compact; panel GEMMs
      via existing `igemm_rm`; scatter-add into `acc` planes; decode.
   3. Only then consider faithful + bucket (drop bands outside the bit budget).

7. **Risks.**
   - Gather/scatter can dominate at small buckets.
   - Random `U(-1,1)` has a fairly wide exponent spread → many thin buckets.
   - Normalized ML activations (tight range) are the best case — worth a
     dedicated data mode (`--data narrow` or similar) for the table.

*Measure:* n=4096, compare `limb-bucket-fp32` to `limb-fp32-exact` and
`limb-fp32-faithful` on both random and narrow-exponent data; correctness
against the independent limb reference.



## B4. Convolution along the contraction index (pointwise “SS-style” matmul) — DONE

**Idea.** Limb convolution attacks the *digit* sum \(C_w = \sum_{u+v=w} A_u B_v\).
The same algebraic pattern can attack the *standard* matmul sum along \(w\):

\[
C_{uv} = \sum_w A_{uw} B_{wv}.
\]

Index-reverse the first axis of \(B\) (or pad to length \(W\ge 2n-1\)):

\[
D_{(W-w)v} := B_{wv}
\qquad\Rightarrow\qquad
C_{uv} = \sum_w A_{uw}\, D_{(W-w)v}.
\]

For fixed \((u,v)\) the right-hand side is the length-\(W\) **1D convolution** of
row \(u\) of \(A\) with the \(v\)-th “column” sequence of \(D\), sampled at the
appropriate index. That is the same shape of sum SS/Karatsuba/Toom accelerate
for integers.

**Goal.** Derive a practical recursive (or FFT) algorithm for dense \(n\times n\)
matmul by treating the contraction index as a convolution, analogous to how
Karatsuba/Toom/MFFT treat the limb index — and implement it as a new **leaf**
(or standalone method) in the exact track.

**Phases (do not skip straight to a full FFT matmul claim).**

1. **Algebra note (README / short design doc).**
   - Formalise \(D\) reversal and the convolution identity.
   - Cost of *naive* “\(n^2\) independent length-\(n\) convolutions” = worse than
     \(O(n^3)\) unless structure is shared across \(u,v\).
   - Identify what must be shared (row polynomials, column polynomials, 2D
     structure) so the method is not trivially dominated by schoolbook.

2. **Karatsuba/Toom on the \(k\)-sum (bilinear splitting).**
   - Split the contraction range \(w = 0..n-1\) into halves (or three parts).
   - Form midpoint combinations of row-blocks of \(A\) and column-blocks of \(B\)
     (same bookkeeping as integer Kara/Toom, but blocks are matrix panels).
   - Compare product count to matrix-Strassen (7/8) and schoolbook.
   - This may rediscover known bilinear matmul algorithms; document overlap.

3. **FFT / MFFT along \(k\) (research).**
   - Evaluate rows of \(A\) and reversed columns of \(B\) at roots of unity;
     pointwise products; interpolate \(C\).
   - Ring choice: complex FFT (lossy for exact track), NTT, or MFFT-style
     integer roots — only the last stays in the exact integer model.
   - Exactness + carry/bit-growth analysis is mandatory before a table row.

4. **Prototype leaf `KERNEL_CONV_K` or method `matmul-conv-k`.**
   - Start with Karatsuba-on-\(k\) only (phase 2), exact track, verify vs
     limbplane reference for small \(n\).
   - Microbench against `strassen` / `packed` at \(n = 64, 128, 256\).

5. **Compose with limb methods.**
   - Outer: Kara/Toom/MFFT on limbs; inner: conv-\(k\) leaf (or the reverse).
   - Only keep combinations that beat `karatsuba+strassen` on some \((n,L)\).

**Status: DONE — negative result (written + measured).**

Delivered:
- Algebra: \(C_{ij}=\sum_k A_{ik}B_{kj}=\sum_k A_{ik}D_{(n-1-k)j}\) with
  \(D_{tj}=B_{(n-1-t)j}\) (README + `kernel.c` comments).
- `KERNEL_CONVK` — identity = schoolbook / `ikj` (exact, same speed).
- `KERNEL_CONVKARA` — per-(i,j) 1D Karatsuba, middle coeff (exact, ~10–14×
  slower at n=64 on karatsuba track).

Conclusion:
- Limb-axis Kara/Toom/MFFT need **all** convolution outputs (every product
  limb). The matmul sum along \(k\) only needs **one** coefficient per
  \((i,j)\) — a dot product. Independent 1D Kara/FFT per entry cannot beat
  \(O(n^3)\) schoolbook or matrix-Strassen.
- Shared bilinear structure across \((i,j)\) is already what **matrix**
  Strassen/Winograd implement as leaves. No separate FFT-along-\(k\) row.

Kernels remain in the bench for regression and pedagogy; they are not
recommended defaults.



## B5. Binary-digit planes + bit-packed AND/popcount pointwise GEMM — BACKLOG

**Motivation.** Limb methods reduce *how many* plane products run; each product
is still an \(O(n^3)\) (or Strassen \(O(n^{\log_2 7})\)) integer GEMM. The
original MFFT post goes all the way to **base-2 digits** (bits \(0/1\)). At that
granularity a plane product is Boolean:

\[
C_{ij} = \sum_k A_{ik} B_{kj}
       = \bigl|\{k : A_{ik}=1 \land B_{kj}=1\}\bigr|
       = \mathrm{popcount}\bigl(\mathrm{row}_i(A) \land \mathrm{col}_j(B)\bigr)
\]

when the contraction index \(k\) is packed into machine words. AND and popcount
are not general multiplies; theoretically \(\Theta(n^3 / w)\) bit operations
with word size \(w=64\) (or wider SIMD).

For general integer planes, expand into bit-planes (as `KERNEL_BITPLANE` already
does for \(A\) only) and combine:

\[
A=\sum_b 2^b A^{(b)},\quad
B=\sum_c 2^c B^{(c)},\quad
AB=\sum_{b,c} 2^{b+c}\,(A^{(b)} B^{(c)})
\]

with each \(A^{(b)} B^{(c)}\) a 0-1 matmul via AND/popcount.

**Relation to existing code.**
- `KERNEL_BITPLANE`: walks set bits of \(A\), does integer row-adds of \(B\) —
  still multiplies in spirit (shift-add of full rows), not packed AND/popcount.
- B5: **both** operands bit-packed along \(k\); inner kernel is word AND + popcount.
- Limb axis stays schoolbook / Kara / Toom / MFFT; B5 only replaces the **leaf**.

**Phases.**

1. **Algebra + cost model (doc only).**
   - Formalise packed layout: `uint64_t Abits[n][(n+63)/64]` for one 0-1 plane.
   - Cost: \(n^2 \cdot \lceil n/64\rceil\) ANDs + popcounts per Boolean GEMM.
   - Full int16 plane: up to \(16\times 16=256\) Boolean GEMMs (or \(\sim 8\times 8\)
     average if sparse bits) — compare to one int16 schoolbook GEMM.
   - When it can win: pure 0-1 planes; sparse bits; SIMD popcount (AVX-512 VPOPCNT).
   - When it loses: dense 16-bit random limbs (too many plane pairs).

2. **Boolean GEMM leaf (`KERNEL_BOOLPACK` or `mm_boolpack`).**
   - Input: int32 planes that are known 0-1 (or clamp/mask to bit 0).
   - Pack rows of \(A\) and columns of \(B\) once per call (or pack \(B^\top\) rows).
   - \(C_{ij} += \mathrm{sign} \cdot \mathrm{popcount}(Arow_i \land Brow_j)\).
   - Verify exact on random 0-1 matrices vs `ikj`.

3. **Bit-plane expansion leaf (`KERNEL_BITPACK`).**
   - For general int32 \(A,B\) with values in \([0,2^w)\) (handle signs like bitplane):
     for each bit \(b,c\): Boolean GEMM of plane \(b\) of \(A\) and plane \(c\) of \(B\);
     add result \(\ll (b+c)\) into int64 accumulator.
   - Optimisations: skip all-zero bit-planes; iterate only set-bit pairs of the
     *value range* present in the plane (scan max bits).
   - Exact track: wire as another kernel next to `bitplane` / `strassen`.

4. **MFFT / limb path integration.**
   - CPU exact track already multiplies int32 limb planes — drop in `bitpack`.
   - Optional: force `LIMB_BITS=1` build (already supported) so each limb plane
     *is* Boolean; then Boolean GEMM is the natural leaf (best case for B5).
   - GPU later: use ballot/popcount or tensor-core binary paths if available;
     not required for phase 1–3.

5. **Bench and decision.**
   - Table: `limbplane`/`karatsuba` × `{ikj,strassen,bitplane,boolpack,bitpack}`
     at \(n\in\{64,256,1024\}\), `LIMB_BITS=1` and `LIMB_BITS=16`.
   - Success: at `LIMB_BITS=1`, boolpack beats ikj by a clear factor; at 16-bit,
     either bitpack wins on sparse/faithful planes or we document a negative
     result and keep boolpack only for base-2.

**Risks.**
- Dense multi-bit planes: \(w^2\) Boolean GEMMs can exceed one integer GEMM.
- Packing/transpose overhead for columns of \(B\).
- Signed values and `mm_accum` sign must match existing leaves.
- Popcount throughput depends on CPU (builtin vs VPOPCNT).

**Status:** Phases 2–3 optimized in tree.

- Pack all bit-planes in one pass (`pack_all_bitplanes`); OpenMP on
  `bool_gemm_accum` for n≥64; popcount loop unrolled.
- `LIMB_BITS=1` n=64: boolpack beats ikj, competes with packed; may beat
  or trail matrix-Strassen depending on CPU popcount (user saw boolpack
  win Strassen; sandbox: strassen still slightly ahead).
- `LIMB_BITS=16` dense: boolpack falls back to ikj unless planes are 0-1;
  bitpack exact but usually slower (many plane pairs).
- Use `--reps 5+` when product count is small (e.g. L=4 → 9 GEMMs) — single
  reps are noisy.

Also: `KERNEL_BOOLSTRASSEN` — Strassen recursion with boolpack only on
panels that remain 0-1; after Kara/Strassen sums leave {0,1}, leaf is ikj.
Important: Karatsuba intermediates are *not* 0-1 even for LIMB_BITS=1
(A0+A1 ∈ {0,1,2}), so boolpack must is_01-check every call.

Complexity note: boolpack does n² outputs × (n/64) word AND+popcount
= Θ(n³/64) *word ops*, not Θ(n²). No general mul, but still cubic in n.
Strassen is ~n^2.81 integer muls — wins for large n.

SIMD: AVX512 VPOPCNT on CPU boolpack (done).
GPU: `cuda/boolpack_gpu.cuh` — pack (scalar + `__ballot_sync`), GEMM
(`__popc`), tiled variant; rows `boolpack-gpu` / `boolpack-ballot` /
`boolpack-gemm` / `boolpack-tiled` in `gemm_bench` (0-1 microbench).
Still open: n=256 table on more machines; fuse boolpack into GPU limb
path when planes are 0-1.
Phase 5: more (n, L) table optional. GPU ballot path deferred.


## Not planned

* GPU Karatsuba / narrower-limb int8 path. Item 4 measured that 7-bit
  schoolbook has the fewest products for every float embedding that appears;
  implementing the other strategies would only make the exact rows slower.
* fp16. It sits between bf16 and fp32 and would tell us nothing new.
