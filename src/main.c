/* main.c -- benchmark driver.
 *
 * Compares MFFT against schoolbook and limb-plane matrix multiplication on
 * n x n matrices of B-bit unsigned integers.  All methods compute the exact
 * product, and every result is checked against the textbook baseline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mfftbench.h"

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

typedef struct {
    const char *name;
    kernel_t    kern;
    int         has_kern;
    double      secs;
    long long   calls;
    long long   products;
    int         ok;
    int         ran;
} result_t;

static int    opt_n       = 64;
static int    opt_bits    = 256;
static int    opt_reps    = 1;
static int    opt_sigma   = 0;
static int    opt_verify  = 1;
static int    opt_naive   = 1;
static int    opt_quiet   = 0;
static int    opt_csv     = 0;
static uint64_t opt_seed  = 12345;
static int    opt_fpwidth  = 0;
static int    opt_illcond  = 0;
static const char *opt_only = NULL;

static int want(const char *name)
{
    if (!opt_only) return 1;
    size_t ln = strlen(name);
    for (const char *q = opt_only; *q; ) {
        const char *e = strchr(q, ',');
        size_t seg = e ? (size_t)(e - q) : strlen(q);
        if (seg == ln && !strncmp(q, name, ln)) return 1;
        if (!e) break;
        q = e + 1;
    }
    return 0;
}

static void usage(const char *p)
{
    printf(
"usage: %s [options]\n"
"\nThe exact track has two independent axes:\n"
"  n      the matrix dimension; matrices are n x n.  Sets the O(n^3) cost of\n"
"         ONE limb product -- identical for every method.\n"
"  width  the scalar width: how many bits each matrix ENTRY holds.  Sets how\n"
"         many limb products a method needs, which is what the methods differ\n"
"         in.  width = L * %d, so L limbs per entry.\n"
"So the algorithms are compared along width; n only scales everything.\n\n"
"  --n N            matrix dimension (default 64)\n"
"  --width W        bits per matrix entry (scalar width), a multiple of\n"
"                   the limb size (default 256); --bits is an alias\n"
"  --sigma S        MFFT block exponent override (block size = 2^S limbs)\n"
"  --reps R         repetitions, best time is reported (default 1)\n"
"  --seed X         PRNG seed (default 12345)\n"
"  --no-verify      skip exactness checks against the textbook baseline\n"
"  --no-naive       skip the O(n^3 L^2) schoolbook big-integer methods\n"
"  --only LIST      run only methods whose name appears in LIST\n"
"  --sweep-width    sweep the scalar width at the given --n (alias --sweep)\n"
"  --sweep-n        sweep the matrix dimension at the given --width\n"
"  --ml             run the machine-learning GEMM track (fp32/bf16/int8)\n"
"  --fp-width B     force a fixed B-bit fp32 embedding grid instead of adaptive\n"
"  --illcond E      widen the ML data exponent spread to E (costs limbs)\n"
"  --test-roots     self-test the H_{s,k} roots-of-unity recursion\n"
"  --csv            emit machine-readable csv\n"
"  --help\n", p, LIMB_BITS);
}

/* ------------------------------------------------------------------ */
static void run_one(result_t *r, int n, int L, int RL,
                    const uint16_t *Apl, const uint16_t *Bpl,
                    const uint16_t *Aem, const uint16_t *Bem,
                    const mfft_plan *plan, const mfft_plan *plan2,
                    const uint16_t *ref,
                    uint16_t *out, int which)
{
    double best = 1e30;
    long long calls = 0;
    for (int rep = 0; rep < opt_reps; rep++) {
        memset(out, 0, (size_t)RL * n * n * sizeof(uint16_t));
        g_kernel_calls = 0;
        double t0 = now_sec();
        switch (which) {
        case 0: mm_bigint_ijk(Aem, Bem, n, L, out, RL); break;
        case 1: mm_bigint_ikj(Aem, Bem, n, L, out, RL); break;
        case 2: mm_limbplane(Apl, Bpl, n, L, r->kern, out, RL); break;
        case 3: mm_mfft(Apl, Bpl, n, L, plan, r->kern, out, RL); break;
        case 4: mm_karatsuba(Apl, Bpl, n, L, r->kern, out, RL); break;
        case 5: mm_mfft(Apl, Bpl, n, L, plan2, r->kern, out, RL); break;
        }
        double t = now_sec() - t0;
        if (t < best) best = t;
        calls = g_kernel_calls;
    }
    r->secs = best;
    r->calls = calls;
    r->ran = 1;
    if (ref && opt_verify) {
        int bi, bj, bw;
        r->ok = bigres_equal(ref, out, n, RL, &bi, &bj, &bw);
        if (!r->ok)
            fprintf(stderr, "  MISMATCH in %s/%s at entry (%d,%d) limb %d\n",
                    r->name, r->has_kern ? kernel_name(r->kern) : "-",
                    bi, bj, bw);
    } else {
        r->ok = -1;
    }
}

static void print_table(result_t *R, int nr, double baseline)
{
    if (opt_csv) {
        printf("method,kernel,products,kernel_calls,seconds,speedup,exact\n");
        for (int i = 0; i < nr; i++) {
            if (!R[i].ran) continue;
            printf("%s,%s,%lld,%lld,%.6f,%.3f,%s\n",
                   R[i].name, R[i].has_kern ? kernel_name(R[i].kern) : "-",
                   R[i].products, R[i].calls, R[i].secs,
                   baseline > 0 ? baseline / R[i].secs : 0.0,
                   R[i].ok < 0 ? "unchecked" : (R[i].ok ? "yes" : "NO"));
        }
        return;
    }
    printf("\n%-14s %-9s %12s %10s %9s  %s\n",
           "method", "kernel", "n*n products", "seconds", "speedup", "exact");
    printf("-------------------------------------------------------------------------\n");
    for (int i = 0; i < nr; i++) {
        if (!R[i].ran) continue;
        char prod[32];
        if (R[i].products > 0) snprintf(prod, sizeof prod, "%lld", R[i].products);
        else                   snprintf(prod, sizeof prod, "-");
        printf("%-14s %-9s %12s %10.4f %8.2fx  %s\n",
               R[i].name, R[i].has_kern ? kernel_name(R[i].kern) : "-",
               prod, R[i].secs,
               baseline > 0 ? baseline / R[i].secs : 0.0,
               R[i].ok < 0 ? "-" : (R[i].ok ? "yes" : "NO!"));
    }
}

/* ------------------------------------------------------------------ */
static int run_case(int n, int bits)
{
    int L = bits / LIMB_BITS;
    int RL = 2 * L + 1;
    size_t nn = (size_t)n * n;

    mfft_plan plan, planr;
    int have_plan = (mfft_plan_init(&plan, L, opt_sigma) == 0);
    int have_rec  = (mfft_plan_init_rec(&planr, L, n, opt_sigma) == 0);

    if (!opt_quiet) {
        printf("\n==== matrices %d x %d  |  scalar width %d bits "
               "= %d limbs of %d bits  |  seed %llu ====\n",
               n, n, bits, L, LIMB_BITS, (unsigned long long)opt_seed);
        printf("     n scales every method equally (one limb product costs "
               "O(n^3)); the scalar width\n"
               "     decides how many limb products each method needs, which "
               "is where they differ.\n");
        if (have_plan) mfft_plan_describe(&plan, n);
        else printf("MFFT plan: unavailable for L=%d\n", L);
        if (have_rec)
            printf("MFFT-rec:  S=%d NB=%d K=%d -> %lld products "
                   "(%.2fx fewer than flat MFFT, %.2fx vs karatsuba)\n",
                   planr.S, planr.NB, planr.K, planr.nprod,
                   have_plan ? (double)plan.nprod / (double)planr.nprod : 0.0,
                   (double)karatsuba_products(L) / (double)planr.nprod);
    }
    if (have_plan) {
        double mb = mfft_plan_maxbits(&plan, n);
        if (mb > 62.0) {
            fprintf(stderr, "warning: MFFT intermediates may need %.0f bits "
                            "(> 62); reduce --n or --bits\n", mb);
        }
    }

    uint16_t *Apl = bigmat_alloc(n, L), *Bpl = bigmat_alloc(n, L);
    if (!Apl || !Bpl) { fprintf(stderr, "oom\n"); return 1; }
    bigmat_rand(Apl, n, L, opt_seed);
    bigmat_rand(Bpl, n, L, opt_seed ^ 0xABCDEF);
    uint16_t *Aem = bigmat_to_entry_major(Apl, n, L);
    uint16_t *Bem = bigmat_to_entry_major(Bpl, n, L);

    uint16_t *ref = calloc((size_t)RL * nn, sizeof(uint16_t));
    uint16_t *out = calloc((size_t)RL * nn, sizeof(uint16_t));
    if (!Aem || !Bem || !ref || !out) { fprintf(stderr, "oom\n"); return 1; }

    result_t R[2 + 4 * KERNEL__COUNT];
    memset(R, 0, sizeof R);
    int nr = 0;

    /* reference first, always computed when verification is on */
    double t0 = now_sec();
    if (opt_verify || opt_naive) mm_bigint_ijk(Aem, Bem, n, L, ref, RL);
    double tref = now_sec() - t0;

    if (opt_naive) {
        R[nr] = (result_t){"bigint", KERNEL_IKJ, 0, tref, 0,
                           (long long)0, 1, 1};
        R[nr].name = "bigint-ijk"; nr++;
        R[nr] = (result_t){"bigint-ikj", KERNEL_IKJ, 0, 0, 0, 0, 0, 0};
        run_one(&R[nr], n, L, RL, Apl, Bpl, Aem, Bem, &plan, &planr,
                opt_verify ? ref : NULL, out, 1);
        nr++;
    }

    if (want("limbplane")) for (int k = 0; k < KERNEL__COUNT; k++) {
        R[nr] = (result_t){"limbplane", (kernel_t)k, 1, 0, 0,
                           (long long)L * L, 0, 0};
        run_one(&R[nr], n, L, RL, Apl, Bpl, Aem, Bem, &plan, &planr,
                opt_verify ? ref : NULL, out, 2);
        nr++;
    }

    if (want("karatsuba")) for (int k = 0; k < KERNEL__COUNT; k++) {
        R[nr] = (result_t){"karatsuba", (kernel_t)k, 1, 0, 0,
                           karatsuba_products(L), 0, 0};
        run_one(&R[nr], n, L, RL, Apl, Bpl, Aem, Bem, &plan, &planr,
                opt_verify ? ref : NULL, out, 4);
        nr++;
    }

    if (have_plan && want("mfft"))
        for (int k = 0; k < KERNEL__COUNT; k++) {
            R[nr] = (result_t){"mfft", (kernel_t)k, 1, 0, 0,
                               mfft_plan_products(&plan), 0, 0};
            run_one(&R[nr], n, L, RL, Apl, Bpl, Aem, Bem, &plan, &planr,
                    opt_verify ? ref : NULL, out, 3);
            nr++;
        }

    if (have_rec && want("mfft-rec"))
        for (int k = 0; k < KERNEL__COUNT; k++) {
            R[nr] = (result_t){"mfft-rec", (kernel_t)k, 1, 0, 0,
                               planr.nprod, 0, 0};
            run_one(&R[nr], n, L, RL, Apl, Bpl, Aem, Bem, &plan, &planr,
                    opt_verify ? ref : NULL, out, 5);
            nr++;
        }

    /* baseline for the speedup column: limbplane/ikj */
    double baseline = 0;
    for (int i = 0; i < nr; i++)
        if (!strcmp(R[i].name, "limbplane") && R[i].kern == KERNEL_IKJ)
            baseline = R[i].secs;

    print_table(R, nr, baseline);

#ifdef HAVE_CBLAS
    {
        double *da = malloc(nn * sizeof(double));
        double *db = malloc(nn * sizeof(double));
        double *dc = malloc(nn * sizeof(double));
        for (size_t i = 0; i < nn; i++) { da[i] = (double)(i % 97);
                                          db[i] = (double)(i % 89); }
        double b0 = now_sec();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    1.0, da, n, db, n, 0.0, dc, n);
        double bt = now_sec() - b0;
        printf("\nreference: one BLAS dgemm at n=%d took %.5f s "
               "(inexact, 53-bit doubles -- shown only for scale)\n", n, bt);
        free(da); free(db); free(dc);
    }
#endif

    int allok = 1;
    for (int i = 0; i < nr; i++) if (R[i].ok == 0) allok = 0;
    if (opt_verify && !opt_csv)
        printf("\nexactness: %s\n", allok ? "all methods agree with the "
               "textbook baseline" : "*** MISMATCH ***");

    free(Apl); free(Bpl); free(Aem); free(Bem); free(ref); free(out);
    return allok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int sweep = 0, test_roots = 0, ml = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n")     && i + 1 < argc) opt_n = atoi(argv[++i]);
        else if (!strcmp(a, "--bits")  && i + 1 < argc) opt_bits = atoi(argv[++i]);
        else if (!strcmp(a, "--width") && i + 1 < argc) opt_bits = atoi(argv[++i]);
        else if (!strcmp(a, "--reps")  && i + 1 < argc) opt_reps = atoi(argv[++i]);
        else if (!strcmp(a, "--sigma") && i + 1 < argc) opt_sigma = atoi(argv[++i]);
        else if (!strcmp(a, "--seed")  && i + 1 < argc) opt_seed = strtoull(argv[++i], 0, 0);
        else if (!strcmp(a, "--cutoff")&& i + 1 < argc) g_strassen_cutoff = atoi(argv[++i]);
        else if (!strcmp(a, "--no-verify")) opt_verify = 0;
        else if (!strcmp(a, "--no-naive"))  opt_naive = 0;
        else if (!strcmp(a, "--csv"))       opt_csv = 1;
        else if (!strcmp(a, "--sweep"))     sweep = 1;
        else if (!strcmp(a, "--sweep-width")) sweep = 1;
        else if (!strcmp(a, "--sweep-n"))   sweep = 2;
        else if (!strcmp(a, "--ml"))        ml = 1;
        else if (!strcmp(a, "--fp-width") && i + 1 < argc) opt_fpwidth = atoi(argv[++i]);
        else if (!strcmp(a, "--illcond")  && i + 1 < argc) opt_illcond = atoi(argv[++i]);
        else if (!strcmp(a, "--only")     && i + 1 < argc) opt_only = argv[++i];
        else if (!strcmp(a, "--test-roots"))test_roots = 1;
        else if (!strcmp(a, "--help"))      { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return 2; }
    }

    if (test_roots) return roots_selftest(6, 1) ? 0 : 1;

    if (ml) return ml_run(opt_n, opt_reps, opt_csv, opt_naive, opt_fpwidth, opt_illcond);

    if (opt_bits % LIMB_BITS || opt_bits < 2 * LIMB_BITS) {
        fprintf(stderr, "--bits must be a multiple of %d and >= %d\n",
                LIMB_BITS, 2 * LIMB_BITS);
        return 2;
    }
    if (opt_n < 1) { fprintf(stderr, "--n must be positive\n"); return 2; }

    if (!sweep) return run_case(opt_n, opt_bits);

    int rc = 0;
    if (sweep == 1)
        for (int bits = 128; bits <= 8192; bits *= 2)
            rc |= run_case(opt_n, bits);
    else
        for (int n = 32; n <= opt_n; n *= 2)
            rc |= run_case(n, opt_bits);
    return rc;
}
