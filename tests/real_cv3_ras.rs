//! Regression test for the CV3 repetition-aware-sampling (RAS) fallback — the root cause of
//! the live-AR degeneracy (live n=33 short/off-track vs reference 102).
//!
//! Reference `cosyvoice/utils/common.py`:
//!   * `ras_sampling` (:138): nucleus-sample; if the pick repeated `>= win_size*tau_r` (=1)
//!     times in the last `win_size` decoded tokens, REPLACE it with `random_sampling`.
//!   * `random_sampling` (:165): `weighted_scores.softmax(dim=0).multinomial(1)` — the PLAIN
//!     full distribution over ALL ids, NO top-p/top-k, and crucially NO masking of the
//!     repeated pick.
//!
//! The Rust port previously masked the repeated pick (`logp[top] = -inf`) before the
//! fallback, forcing an off-distribution token on every repeat — which fires constantly
//! (speech tokens repeat within a 10-window) and drove the trajectory short/off-track. This
//! test pins the corrected behaviour against the **production** `ras_sampling`:
//!
//!   (1) on a forced repeat, the guard `triggers`;
//!   (2) the repeated id is STILL drawable (would be impossible — frequency exactly 0 —
//!       under the old masking code: the regression lock);
//!   (3) the fallback is the FULL distribution: its draw frequencies match `softmax(logits)`,
//!       and low-probability ids OUTSIDE any top-k are reachable.
//!
//! Pure CPU, no weights, no box. Build/run:
//!   cargo test --features real --test real_cv3_ras -- --nocapture

#![cfg(feature = "real")]

use syrinx_lm::real_cv3::testkit::ras_draws;

/// `log_softmax(logits)` as a host vector — `ras_sampling` consumes log-probabilities.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&x| ((x - m) as f64).exp()).sum();
    let lse = m as f64 + sum.ln();
    logits.iter().map(|&x| (x as f64 - lse) as f32).collect()
}

#[test]
fn ras_fallback_is_full_distribution_and_keeps_the_repeated_id() {
    // A peaked categorical over 6 ids: id 0 dominates (~0.62), the rest taper, id 5 is tiny.
    let logits = [3.0f32, 1.5, 1.0, 0.5, 0.0, -2.0];
    let logp = log_softmax(&logits);
    let probs: Vec<f64> = {
        let denom: f64 = logits.iter().map(|&x| x as f64).map(f64::exp).sum();
        logits.iter().map(|&x| (x as f64).exp() / denom).collect()
    };

    // Force the repetition guard unconditionally: every id appears once in the last
    // `win_size` decoded tokens, so WHATEVER nucleus picks has `rep == 1 >= 1` → always
    // trigger. The fallback token distribution is then exactly `softmax(logits)`.
    let decoded = [0u32, 1, 2, 3, 4, 5];
    let n = 300_000usize;
    let draws = ras_draws(&logp, &decoded, false, 0xBADC0DE, n);

    // (1) Every draw must have triggered the fallback (the pick repeated once).
    let triggered = draws.iter().filter(|&&(_, t)| t).count();
    assert_eq!(triggered, n, "RAS guard did not trigger on a forced repeat ({triggered}/{n})");

    // Empirical frequencies of the fallback draws.
    let mut counts = [0usize; 6];
    for &(tok, _) in &draws {
        counts[tok as usize] += 1;
    }
    let emp: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    for i in 0..6 {
        eprintln!("id {i}: softmax={:.4}  fallback_empirical={:.4}", probs[i], emp[i]);
    }

    // (2) REGRESSION LOCK: the repeated id 0 must still be drawn at ~its softmax mass. Under
    //     the old `logp[top] = -inf` masking, this frequency would be exactly 0.
    assert!(
        emp[0] > 0.5,
        "repeated id 0 was suppressed (emp={:.4}) — the fallback is masking the repeat (the bug)",
        emp[0]
    );

    // (3) The fallback is the PLAIN full softmax: every id's frequency matches softmax(logits)
    //     within a tight statistical band, AND the low-prob tail id 5 (well outside any top-k)
    //     is reachable — proving no top-p/top-k truncation in the fallback.
    const BAND: f64 = 6e-3;
    for i in 0..6 {
        assert!(
            (emp[i] - probs[i]).abs() < BAND,
            "fallback id {i} freq {:.4} != softmax {:.4} (|Δ| >= {BAND}) — not the full distribution",
            emp[i],
            probs[i]
        );
    }
    assert!(counts[5] > 0, "low-prob tail id 5 never drawn — fallback is truncated (not full)");
}

#[test]
fn ras_no_repeat_returns_nucleus_pick_without_triggering() {
    // No prior decoded tokens ⇒ the pick cannot have repeated ⇒ the guard must NOT fire, and
    // the returned token is the nucleus pick (here the dominant id 0).
    let logits = [4.0f32, 0.0, 0.0, -1.0];
    let logp = log_softmax(&logits);
    let draws = ras_draws(&logp, &[], false, 7, 5_000);
    let triggered = draws.iter().filter(|&&(_, t)| t).count();
    assert_eq!(triggered, 0, "RAS triggered with no decoded history ({triggered} times)");
    // Dominant id should be the overwhelming nucleus pick.
    let id0 = draws.iter().filter(|&&(tok, _)| tok == 0).count();
    assert!(id0 as f64 / draws.len() as f64 > 0.9, "nucleus pick not dominant: {id0}/{}", draws.len());
}
