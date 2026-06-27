//! Model-free unit test for the CV3 HiFT **quality source** excitation math.
//!
//! Drives the pure f0 → source helpers behind `Cv3Synthesizer::quality_source` /
//! `deterministic_source` ([`quality_source_from_f0`] / [`det_source_from_f0`], the
//! `#[doc(hidden)] pub` test seam) on a synthetic f0 vector — no model weights, no
//! Candle device — so it runs everywhere (not env-gated). It asserts:
//!
//!   * **seed reproducibility** — the same `(f0, merge, seed)` yields a bit-identical
//!     source (all randomness is the seeded SplitMix64, never the system RNG);
//!   * **seed sensitivity** — a different seed yields a different source (the random
//!     per-harmonic phase + Gaussian breath actually depend on the seed);
//!   * **it differs from the deterministic source** — the random-phase NSF source is not
//!     the buzzy single-harmonic smoke source;
//!   * **shape + finiteness** — length is `f0.len() * 480` and every sample is finite.
//!
//! The real `quality_source` / `deterministic_source` only add the model `f0_predict` +
//! merge-weight fetch + tensor wrap around these helpers; the math here is byte-identical.

#![cfg(feature = "real")]

use syrinx_serve::synth_cv3::{det_source_from_f0, quality_source_from_f0};

/// The HiFT f0 → source upsample factor (mel frame -> 480 source samples).
const F0_UPSAMPLE: usize = 480;

/// A synthetic f0 frame vector: a voiced ramp (well above the 10 Hz voiced threshold),
/// a couple of unvoiced (0 Hz) frames, then voiced again — exercises both the voiced and
/// unvoiced branches of the NSF source.
fn synthetic_f0() -> Vec<f32> {
    let mut f0 = Vec::new();
    for i in 0..20 {
        f0.push(110.0 + i as f32 * 2.0); // voiced 110..148 Hz
    }
    f0.extend_from_slice(&[0.0, 0.0, 0.0]); // unvoiced
    for i in 0..10 {
        f0.push(160.0 + i as f32); // voiced again
    }
    f0
}

/// A plausible learned `m_source.l_linear` merge: 9 weights (fundamental + 8 overtones)
/// plus a bias. Values are arbitrary but fixed — the test asserts structure, not parity.
fn merge() -> (Vec<f32>, f32) {
    let w = vec![0.9, -0.3, 0.25, -0.2, 0.15, -0.1, 0.08, -0.05, 0.03];
    (w, 0.01)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn quality_source_is_seed_reproducible() {
    let f0 = synthetic_f0();
    let (w, b) = merge();

    let a = quality_source_from_f0(&f0, &w, b, 42);
    let a2 = quality_source_from_f0(&f0, &w, b, 42);

    assert_eq!(a.len(), f0.len() * F0_UPSAMPLE, "source length must be f0.len()*480");
    assert!(a.iter().all(|v| v.is_finite()), "source has a non-finite sample");
    // Same seed -> bit-identical (seeded SplitMix64, no system RNG).
    assert_eq!(a, a2, "same (f0, merge, seed) must reproduce the source exactly");
}

#[test]
fn quality_source_depends_on_seed() {
    let f0 = synthetic_f0();
    let (w, b) = merge();

    let a = quality_source_from_f0(&f0, &w, b, 42);
    let c = quality_source_from_f0(&f0, &w, b, 43);

    assert_eq!(a.len(), c.len());
    // A different seed must change the random phases/noise, so the source differs.
    assert!(
        max_abs_diff(&a, &c) > 1e-6,
        "a different seed must produce a different source"
    );
}

#[test]
fn quality_source_differs_from_deterministic() {
    let f0 = synthetic_f0();
    let (w, b) = merge();

    let q = quality_source_from_f0(&f0, &w, b, 7);
    let d = det_source_from_f0(&f0);

    assert_eq!(d.len(), f0.len() * F0_UPSAMPLE, "det source length must be f0.len()*480");
    assert_eq!(q.len(), d.len(), "both sources cover the same span");
    assert!(d.iter().all(|v| v.is_finite()), "det source has a non-finite sample");
    // The random-phase NSF source (multi-harmonic + noise + learned merge + tanh) is not
    // the buzzy single-harmonic deterministic smoke source.
    assert!(
        max_abs_diff(&q, &d) > 1e-3,
        "the quality source must differ from the deterministic source"
    );
}
