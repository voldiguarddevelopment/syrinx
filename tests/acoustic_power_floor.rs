//! The permutation test's **power** floor, not just its feasibility floor.
//!
//! `min_n_for_alpha` answers whether the exact test can ever reject at `alpha`. It was used
//! as though it answered whether the test can reject a *real effect*, and those differ by a
//! lot at small n.
//!
//! The `[sad]` tuning round of 2026-09-08 is the worked example. It ran at n=6, alpha=0.01 —
//! `min_n_for_alpha(0.01) == 6`, so the check passed. The incumbent, a phrase independently
//! shown to work on 5 of 6 sentences in the sentence sweep, scored **exactly 0.0022** on its
//! best sentence. That is 1/462: the single most extreme labeling out of all of them. On the
//! other two sentences it scored 0.3377 and 0.0368 and failed. A gate only the most extreme
//! possible draw can clear rejects working phrases and reports "nothing is better" when the
//! truth is "the test could not tell".

use syrinx_eval::acoustic::{min_n_for_alpha, min_n_for_headroom};

/// The number of distinct labelings the exact test enumerates, C(2n,n)/2 — mirrored here
/// so the test does not just restate the implementation back to itself.
fn splits(n: u64) -> u64 {
    let mut c: u64 = 1;
    for i in 0..n {
        c = c * (2 * n - i) / (i + 1);
    }
    c / 2
}

#[test]
fn the_feasibility_floor_is_exactly_where_alpha_becomes_attainable() {
    for n in 4..=10u64 {
        let attainable = 1.0 / splits(n) as f64;
        // Just above the attainable p: this n suffices.
        assert!(min_n_for_alpha(attainable * 1.0001) <= n as usize, "n={n}");
        // Just below it: this n does not.
        assert!(min_n_for_alpha(attainable * 0.9999) > n as usize, "n={n}");
    }
}

/// The headroom floor is strictly stricter, and by the amount asked for.
#[test]
fn the_headroom_floor_is_stricter_than_the_feasibility_floor() {
    for alpha in [0.05, 0.01, 0.00417, 0.001] {
        let feas = min_n_for_alpha(alpha);
        for h in [10.0, 20.0, 50.0] {
            let pow = min_n_for_headroom(alpha, h);
            assert!(pow >= feas, "alpha={alpha} h={h}: {pow} < {feas}");
            let attainable = 1.0 / splits(pow as u64) as f64;
            assert!(
                attainable <= alpha / h,
                "alpha={alpha} h={h}: n={pow} attains {attainable:.2e}, needs <= {:.2e}",
                alpha / h
            );
        }
    }
}

/// A headroom of 1 must degenerate to the feasibility floor exactly — the two functions
/// have to agree where they mean the same thing.
#[test]
fn a_headroom_of_one_is_the_feasibility_floor() {
    for alpha in [0.05, 0.01, 0.005, 0.001] {
        assert_eq!(min_n_for_headroom(alpha, 1.0), min_n_for_alpha(alpha), "alpha={alpha}");
    }
    // Below 1 is clamped rather than making the floor *looser*, which would be worse than
    // useless: it would silently permit an n the test cannot reject at.
    for alpha in [0.05, 0.01] {
        assert_eq!(min_n_for_headroom(alpha, 0.1), min_n_for_alpha(alpha), "alpha={alpha}");
        assert_eq!(min_n_for_headroom(alpha, 0.0), min_n_for_alpha(alpha), "alpha={alpha}");
    }
}

/// The case that motivated this, pinned with its real numbers so the regression is named.
#[test]
fn the_sad_tuning_round_configuration_is_rejected_by_the_power_floor() {
    let alpha = 0.01; // 5 candidates, Bonferroni from 0.05
    // The feasibility floor is n=5 (attainable p 0.0079, only 1.3x headroom). The round
    // ran at n=6 — comfortably above the floor the driver checked, and still only 4.6x.
    // The floor being *lower* than the n actually used is what makes the point: passing
    // that check told us nothing about power.
    assert_eq!(min_n_for_alpha(alpha), 5, "the feasibility floor the driver checked");
    assert_eq!(splits(5), 126);
    assert_eq!(splits(6), 462);
    assert!(1.0 / splits(6) as f64 > alpha / 20.0, "n=6 has under 20x headroom");
    // ...and 0.0022, the incumbent's best score, IS the minimum attainable at n=6.
    assert!(f64::abs(1.0 / 462.0 - 0.002165) < 1e-5);
    // The power floor rejects n=6 and asks for 8.
    assert_eq!(min_n_for_headroom(alpha, 20.0), 8, "n=8 is the first with 20x headroom");
    assert!(1.0 / splits(8) as f64 <= alpha / 20.0);
    assert!(1.0 / splits(7) as f64 > alpha / 20.0, "n=7 must not suffice at 20x");
}

/// Monotonic in both arguments: a smaller alpha or a larger headroom never asks for less.
#[test]
fn the_floor_is_monotonic_in_alpha_and_in_headroom() {
    let mut prev = 0;
    for alpha in [0.05, 0.01, 0.005, 0.001, 0.0001] {
        let n = min_n_for_headroom(alpha, 20.0);
        assert!(n >= prev, "alpha={alpha} lowered the floor");
        prev = n;
    }
    let mut prev = 0;
    for h in [1.0, 5.0, 20.0, 100.0, 1000.0] {
        let n = min_n_for_headroom(0.01, h);
        assert!(n >= prev, "headroom={h} lowered the floor");
        prev = n;
    }
}
