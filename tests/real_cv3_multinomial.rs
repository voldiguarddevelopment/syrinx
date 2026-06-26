//! Distribution unit test for the CV3 live-AR sampler's core draw, [`multinomial1`]
//! (suspect 3 of the CV3 live-token degeneracy investigation).
//!
//! The CV3 AR loop turns a logit vector into a token by `nucleus_sampling` →
//! `multinomial1` (an inverse-CDF draw on one SplitMix64 uniform). If that draw were biased
//! toward the high-probability end of the candidate set, the generated sequence would
//! collapse toward a few dominant ids — exactly the degeneracy shape under investigation.
//! This test exercises the **production** `multinomial1` + `SplitMix64` (via the
//! `real_cv3::testkit` seam, no copy) and asserts the empirical frequencies match the target
//! categorical to within a tight statistical band, over both normalised and UN-normalised
//! weight vectors (the sampler must internally normalise by the running total).
//!
//! Pure CPU, no model weights, no box — but `real_cv3` is behind the `real` feature (it
//! pulls Candle), so build/run with:
//!
//!   cargo test -p syrinx-workspace-scaffold-tests --features real \
//!       --test real_cv3_multinomial -- --nocapture
//!
//! (equivalently `cargo test --features real --test real_cv3_multinomial` from the repo
//! root). Skips nothing — it always runs once compiled.

#![cfg(feature = "real")]

use syrinx_lm::real_cv3::testkit::{multinomial1_draws, uniform_draws};

/// Empirical frequencies of `draws` over `k` categories.
fn freqs(draws: &[usize], k: usize) -> Vec<f64> {
    let mut c = vec![0usize; k];
    for &d in draws {
        c[d] += 1;
    }
    let n = draws.len() as f64;
    c.into_iter().map(|x| x as f64 / n).collect()
}

#[test]
fn multinomial1_is_unbiased_inverse_cdf() {
    // A deliberately uneven 5-way categorical (sums to 1.0).
    let target = [0.05f32, 0.10, 0.20, 0.50, 0.15];
    let n = 400_000usize;

    // 400k draws ⇒ per-category std ≈ sqrt(p(1-p)/n) ≤ ~8e-4; a 5e-3 band is ~6σ.
    const BAND: f64 = 5e-3;

    let draws = multinomial1_draws(&target, 0xC0FFEE, n);
    let emp = freqs(&draws, target.len());
    for (i, (&p, &q)) in target.iter().zip(emp.iter()).enumerate() {
        let diff = (p as f64 - q).abs();
        eprintln!("cat {i}: target={p:.4}  empirical={q:.4}  |Δ|={diff:.4}");
        assert!(
            diff < BAND,
            "multinomial1 biased at category {i}: target {p:.4} vs empirical {q:.4} (|Δ|={diff:.4} >= {BAND})"
        );
    }

    // Every category must actually be reachable (no dead index from an off-by-one in the
    // inverse-CDF scan) — including the first (smallest u) and last (the fallthrough).
    let mut seen = vec![false; target.len()];
    for &d in &draws {
        seen[d] = true;
    }
    assert!(seen.iter().all(|&s| s), "some category was never drawn: {seen:?}");

    // UN-normalised weights (same shape × 7.3) must yield the SAME distribution: the draw
    // normalises by its own running total, as `torch.multinomial` does. A draw that assumed
    // pre-normalised inputs would skew — this is the exact property the nucleus path relies
    // on (candidate probs sum to ~top_p, not 1.0).
    let scaled: Vec<f32> = target.iter().map(|&p| p * 7.3).collect();
    let draws2 = multinomial1_draws(&scaled, 0xC0FFEE, n);
    let emp2 = freqs(&draws2, target.len());
    for (i, (&p, &q)) in target.iter().zip(emp2.iter()).enumerate() {
        let diff = (p as f64 - q).abs();
        assert!(
            diff < BAND,
            "multinomial1 not scale-invariant at category {i}: target {p:.4} vs empirical {q:.4} (|Δ|={diff:.4})"
        );
    }

    // A peaky distribution must concentrate (~99.7% on the dominant id) — confirms the draw
    // does NOT spread mass and is not, conversely, *over*-collapsing low-mass ids to zero.
    let peaky = [0.001f32, 0.001, 0.997, 0.001];
    let dp = multinomial1_draws(&peaky, 42, 100_000);
    let ep = freqs(&dp, peaky.len());
    assert!(
        (ep[2] - 0.997).abs() < BAND,
        "peaky draw mis-concentrated: dominant empirical {:.4} vs 0.997",
        ep[2]
    );
}

#[test]
fn splitmix64_uniform_is_flat() {
    // The PRNG feeding multinomial1 must itself be uniform on [0,1): a biased uniform would
    // bias every draw. Bucket 200k draws into 10 bins; each ≈ 0.1.
    let n = 200_000usize;
    let u = uniform_draws(0x5EED, n);
    let mut bins = [0usize; 10];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in &u {
        assert!((0.0..1.0).contains(&x), "uniform out of [0,1): {x}");
        lo = lo.min(x);
        hi = hi.max(x);
        bins[(x * 10.0) as usize] += 1;
    }
    for (i, &b) in bins.iter().enumerate() {
        let f = b as f64 / n as f64;
        eprintln!("bin {i} [{:.1},{:.1}): {f:.4}", i as f64 / 10.0, (i + 1) as f64 / 10.0);
        assert!((f - 0.1).abs() < 5e-3, "uniform bin {i} skewed: {f:.4} vs 0.1");
    }
    assert!(lo < 0.01 && hi > 0.99, "uniform does not span [0,1): lo={lo:.4} hi={hi:.4}");
}
