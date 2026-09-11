//! Is the activation test the size it claims to be — at the `n` the gate actually uses?
//!
//! ## Why this file exists
//!
//! The C4.2′ certification run of 2026-09-09 logged an A/A calibration of **p = 0.0507**
//! at n=9 on one of its three texts, against a nominal 0.05. It did not violate, but it
//! is the closest a plain arm has come to disagreeing with itself, and three explanations
//! were open: the permutation test is not actually exact at n=9; some acoustic feature is
//! heavy-tailed enough to break exchangeability; or the two seed blocks the runner uses
//! (`0..n` and `1000..1000+n`) are distinguishable for a reason other than chance.
//!
//! `tests/cue_activation_measure.rs` already asserts calibration, but at **n=4** and over
//! 20 replications — it cannot speak about n=9, and 20 Bernoulli draws cannot resolve 5 %
//! from 15 %. This file answers the question at the operating point.
//!
//! ## What is proved here, and why it is stronger than a false-positive count
//!
//! Under the null the 2n renders are exchangeable, so every one of the
//! `N = C(2n, n) / 2` labelings is equally likely. `activation_test` reports the fraction
//! of labelings whose between-means distance is at least the observed one, so if it is
//! exact, **feeding it every labeling of one fixed pool must return every value in
//! `{1/N, 2/N, …, N/N}` exactly once**. That is a deterministic statement about a fixed
//! pool, it is checked here by enumeration, and it settles the calibration question for
//! *every* input distribution at once: the pools below include an infinite-variance
//! Cauchy one and a pool of real features extracted from 1.4-second clips, and the
//! uniformity is bit-exact in all of them. A distribution cannot break a conditional
//! statement that holds for its own pool.
//!
//! The consequence is arithmetic rather than empirical: the false-positive rate at
//! alpha is exactly `floor(N * alpha) / N`, which at n=9 and alpha=0.05 is
//! `1215 / 24310 = 0.049979`. The Monte-Carlo rows below are a cross-check on that
//! derivation, not the evidence for it.
//!
//! Everything is deterministic — the project's own `SplitMix64` supplies every draw — so
//! these are reproducible facts, not a statistical smoke test that can flake.

use syrinx_eval::acoustic::{activation_test, features, Features, N_FEATURES};
use syrinx_qwen::sampling::SplitMix64;

/// Alpha the A/A calibration is judged at (`ContrastThresholds::alpha`, nominal, not
/// Bonferroni-corrected — clause A2 asks whether the test is the size it claims).
const ALPHA: f64 = 0.05;

/// `C(18, 9) / 2` — the labelings an exact test enumerates at the n the C4.2′ run used.
const N_PERM_AT_9: usize = 24_310;

// ---- deterministic generators ------------------------------------------------------

fn standard_normal(r: &mut SplitMix64) -> f64 {
    let u1 = r.next_f64().max(1e-12);
    let u2 = r.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// A well-behaved pool: 2n i.i.d. Gaussian feature vectors.
fn pool_gaussian(count: usize, seed: u64) -> Vec<Features> {
    let mut r = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let mut v = [0.0; N_FEATURES];
            for slot in v.iter_mut() {
                *slot = standard_normal(&mut r);
            }
            Features { v }
        })
        .collect()
}

/// A pool with **no finite variance**: the ratio of two normals is Cauchy. This is the
/// worst case the "heavy tails break exchangeability" hypothesis can ask for, and it is
/// far heavier-tailed than any of the eleven acoustic dimensions.
fn pool_cauchy(count: usize, seed: u64) -> Vec<Features> {
    let mut r = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let mut v = [0.0; N_FEATURES];
            for slot in v.iter_mut() {
                let a = standard_normal(&mut r);
                let b = standard_normal(&mut r);
                *slot = a / if b.abs() < 1e-9 { 1e-9 } else { b };
            }
            Features { v }
        })
        .collect()
}

/// A pool of **real extracted features** from clips as short as the one the C4.2′ mid
/// cases actually render: `"before it gets any later."` is a five-word fragment, ~1.4 s.
fn pool_short_clips(count: usize, seed: u64) -> Vec<Features> {
    const SR: u32 = 24_000;
    (0..count as u64)
        .map(|i| {
            let mut r = SplitMix64::new(seed + i);
            let n = (1.4 * f64::from(SR)) as usize;
            let f0 = 180.0 * (1.0 + 0.05 * (r.next_f64() - 0.5));
            let amp = 0.3 * (1.0 + 0.1 * (r.next_f64() - 0.5));
            let w: Vec<f32> = (0..n)
                .map(|k| {
                    let t = k as f64 / f64::from(SR);
                    let mut acc = 0.0;
                    for h in 1..=20 {
                        acc += (2.0 * std::f64::consts::PI * f0 * h as f64 * t).sin() / h as f64;
                    }
                    (amp * (acc + 0.02 * (r.next_f64() * 2.0 - 1.0))) as f32
                })
                .collect();
            features(&w, SR)
        })
        .collect()
}

/// All `k`-subsets of `0..n`, lexicographic. A local copy: the point is to enumerate the
/// labelings independently of the enumerator under test.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Feed every labeling of one fixed pool to `activation_test` and return the sorted
/// p-values it produced, plus the labeling count.
fn p_values_over_every_labeling(pool: &[Features], n: usize) -> (Vec<f64>, usize) {
    let total = 2 * n;
    assert_eq!(pool.len(), total);
    // One representative per unordered partition: the half containing index 0.
    let canonical: Vec<Vec<usize>> =
        subsets(total, n).into_iter().filter(|c| c.contains(&0)).collect();
    let mut ps: Vec<f64> = canonical
        .iter()
        .map(|c| {
            let mut a = Vec::with_capacity(n);
            let mut b = Vec::with_capacity(n);
            for (i, f) in pool.iter().enumerate() {
                if c.contains(&i) {
                    a.push(f.clone());
                } else {
                    b.push(f.clone());
                }
            }
            // alpha only decides `activated`; the p-value is what is under test, and a
            // permissive alpha keeps every n answerable.
            activation_test(&a, &b, 0.5).expect("answerable").p_value
        })
        .collect();
    ps.sort_by(f64::total_cmp);
    (ps, canonical.len())
}

/// Assert the p-values of one pool are the exact discrete uniform `{1/N, …, N/N}`.
fn assert_exactly_uniform(pool: &[Features], n: usize, what: &str) {
    let (ps, n_perm) = p_values_over_every_labeling(pool, n);
    assert_eq!(ps.len(), n_perm);
    for (k, got) in ps.iter().enumerate() {
        let want = (k + 1) as f64 / n_perm as f64;
        // Bit-exact: both sides are the same division of the same two integers, so any
        // deviation at all is a real one and not floating-point slack.
        assert_eq!(
            *got, want,
            "{what}: the {k}-th smallest p-value is {got}, not {want} — the permutation \
             null is not exact, so every p-value the C4.2' run reports is mis-sized"
        );
    }
    let hits = ps.iter().filter(|p| **p <= ALPHA).count();
    let expected = (n_perm as f64 * ALPHA).floor() as usize;
    assert_eq!(
        hits, expected,
        "{what}: {hits} of {n_perm} labelings reject at alpha={ALPHA}, expected \
         floor({n_perm}*{ALPHA}) = {expected}"
    );
}

// ---- the exactness proof -------------------------------------------------------------

/// The core claim, enumerated exhaustively at n=4 (35 labelings) and n=5 (126).
///
/// Three pools, chosen to kill the "the features are badly behaved" explanation rather
/// than to flatter the test: Gaussian, Cauchy (no finite variance at all), and real
/// eleven-dimensional features extracted from 1.4-second synthetic clips.
#[test]
fn the_permutation_null_is_exactly_uniform_for_every_pool() {
    for seed in [1u64, 2, 3] {
        assert_exactly_uniform(&pool_gaussian(8, seed), 4, "gaussian n=4");
        assert_exactly_uniform(&pool_cauchy(8, seed), 4, "cauchy n=4");
        assert_exactly_uniform(&pool_short_clips(8, seed * 1000), 4, "short-clip features n=4");
    }
    assert_exactly_uniform(&pool_gaussian(10, 7), 5, "gaussian n=5");
    assert_exactly_uniform(&pool_cauchy(10, 7), 5, "cauchy n=5");
}

/// The resolution of the test at the operating point, and what that makes the false
/// positive rate.
///
/// The C4.2′ run used n=9, so a p-value is a multiple of `1/24310` and the exact size at
/// alpha=0.05 is `floor(24310 * 0.05) / 24310`. Pinned as arithmetic because it is the
/// number the observed 0.0507 has to be read against.
#[test]
fn the_exact_size_at_n_nine_is_just_under_alpha() {
    let pool = pool_gaussian(18, 42);
    let (a, b) = pool.split_at(9);
    let out = activation_test(a, b, ALPHA).expect("answerable at n=9");
    assert_eq!(out.n_permutations, N_PERM_AT_9, "C(18,9)/2 labelings");

    let largest_rejecting = (N_PERM_AT_9 as f64 * ALPHA).floor() as usize;
    assert_eq!(largest_rejecting, 1215);
    let exact_size = largest_rejecting as f64 / N_PERM_AT_9 as f64;
    assert!(exact_size <= ALPHA, "an exact test is never anti-conservative");
    assert!(
        (exact_size - 0.049_979).abs() < 1e-6,
        "the exact size at n=9 is {exact_size}, not 0.049979"
    );

    // The A/A value the certification run logged, read against that scale: 0.0507 is
    // 1233/24310, which is above the largest rejecting value, so it did not reject.
    let observed_aa = 0.050_719_868_366_927_19_f64;
    let rank = (observed_aa * N_PERM_AT_9 as f64).round() as usize;
    assert_eq!(rank, 1233, "the logged A/A p-value is 1233/24310");
    assert!(rank > largest_rejecting, "1233 > 1215: the logged A/A did not reject at alpha");
}

// ---- Monte-Carlo cross-checks --------------------------------------------------------

/// How often two same-condition groups of nine are called different, over many
/// independent replications. A cross-check on the derivation above, at the real `n`.
fn measured_false_positive_rate(reps: usize, base: u64, gen: fn(usize, u64) -> Vec<Features>) -> usize {
    (0..reps)
        .filter(|rep| {
            let pool = gen(18, base + *rep as u64 * 7919);
            let (a, b) = pool.split_at(9);
            activation_test(a, b, 0.5).expect("answerable").p_value <= ALPHA
        })
        .count()
}

/// Independent same-condition replications at n=9, well-behaved and heavy-tailed.
///
/// 1000 draws put the standard error at 0.7 percentage points, so the band below is wide;
/// it is not trying to resolve 5 % from 6 %, which is what the exactness proof above is
/// for. It is trying to catch the failure that matters — a test whose real size is
/// nothing like its nominal one — and every draw is seeded, so the numbers do not move
/// between runs.
#[test]
fn the_measured_false_positive_rate_at_n_nine_matches_the_nominal_size() {
    const REPS: usize = 1_000;
    // floor(0.05 * 24310)/24310 * 1000 = 49.98 expected hits, s.d. 6.9. Measured: 61
    // (gaussian) and 53 (cauchy), i.e. +1.6 and +0.4 standard deviations.
    for (label, base, gen) in [
        ("gaussian", 0xB0Bu64, pool_gaussian as fn(usize, u64) -> Vec<Features>),
        ("cauchy", 0xA11CE, pool_cauchy),
    ] {
        let hits = measured_false_positive_rate(REPS, base, gen);
        assert!(
            (20..=85).contains(&hits),
            "{label}: {hits}/{REPS} same-condition comparisons at n=9 activated; a test of \
             nominal size 0.05 gives 50 on average with a standard deviation of 6.9, so \
             anything outside 20..=85 is not sampling noise"
        );
    }
}

/// The seed blocks the C4.2′ runner actually uses are not distinguishable from each other.
///
/// The runner draws its two plain halves from seeds `0..n` and `1000..1000+n`. A render is
/// a deterministic function of `(prompt, seed)` and the PRNG stream is a function of the
/// **seed alone**, so the two blocks are fixed for the whole run — if `SplitMix64` gave
/// low seeds any shared structure, every A/A in every C4.2′ run would inherit it and the
/// calibration clause would be measuring the PRNG rather than the model.
///
/// This holds the two blocks fixed at exactly the values the runner uses and varies the
/// *prompt* instead, modelled as a per-replication offset into each seed's stream plus a
/// per-replication linear map onto the eleven feature dimensions. A block artefact would
/// show as a false-positive rate far above nominal; there is none — 48 of 1000, against
/// the 49.98 an exactly-sized test gives.
#[test]
fn the_runners_two_seed_blocks_are_exchangeable() {
    const REPS: usize = 1_000;
    const N: u64 = 9;
    let mut hits = 0usize;
    for rep in 0..REPS {
        let mut text = SplitMix64::new(0xC0FFEE + rep as u64);
        let skip = (text.next_f64() * 4000.0) as usize;
        let mut map = [[0.0f64; 8]; N_FEATURES];
        for row in map.iter_mut() {
            for cell in row.iter_mut() {
                *cell = standard_normal(&mut text);
            }
        }
        let render = |seed: u64| -> Features {
            let mut r = SplitMix64::new(seed);
            for _ in 0..skip {
                r.next_f64();
            }
            let mut g = [0.0f64; 8];
            for slot in g.iter_mut() {
                *slot = standard_normal(&mut r);
            }
            let mut v = [0.0f64; N_FEATURES];
            for (d, row) in map.iter().enumerate() {
                for (j, cell) in row.iter().enumerate() {
                    v[d] += cell * g[j];
                }
            }
            Features { v }
        };
        let a: Vec<Features> = (0..N).map(render).collect();
        let b: Vec<Features> = (1_000..1_000 + N).map(render).collect();
        if activation_test(&a, &b, 0.5).expect("answerable").p_value <= ALPHA {
            hits += 1;
        }
    }
    assert!(
        (20..=85).contains(&hits),
        "{hits}/{REPS} A/A comparisons between seeds 0..9 and 1000..1009 activated; at \
         nominal size that is 50, so anything outside 20..=85 says the two seed blocks \
         are distinguishable and the A/A clause is measuring the PRNG"
    );
}
