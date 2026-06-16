//! Frozen RED tests for T-03.04 — scale speech rate.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must add to `syrinx-prosody::plan`, on top of the T-03.01 data model:
//!
//!   * `ProsodyPlan::scale_rate(&self, r: f32) -> Result<ProsodyPlan, PlanError>`
//!     — the utterance-level rate scaler. For a positive `r`, it returns a NEW
//!     plan whose every `durations_ms[i]` is exactly `r * original[i]` (so the
//!     summed total scales by `r` and per-phoneme proportions are preserved) and
//!     whose `pitch_hz` is element-for-element identical to the input — rate
//!     scaling is a uniform multiply of `durations_ms` that never touches pitch.
//!     `r == 1.0` is duration-identity. `r <= 0.0` (e.g. `0.0`, `-1.0`) returns
//!     `Err(PlanError::InvalidRate)` and yields no scaled plan; any `r > 0.0`
//!     (down to `0.001`) returns `Ok`. Nothing panics.
//!
//! Scope (list.md / T-03.04): a whole-plan duration scale only — no pitch shift,
//! no per-phoneme rate, no prediction, no defaults. Whether the time-scaled audio
//! is perceptually correct and pitch-preserved on rendered output is a deferred
//! perceptual eval against the real model, not gated here.
//!
//! RED: `ProsodyPlan` exposes no `scale_rate` and `PlanError` has no `InvalidRate`
//! variant yet, so these symbols do not resolve and the test target fails to
//! build — every criterion is unmet. GREEN adds the method and the variant so each
//! assertion holds.

use syrinx_prosody::plan::{PlanError, ProsodyPlan};

/// A populated, schema-current plan with three phonemes. The `durations_ms` are
/// chosen so that 2.0×, 0.5×, 3.0×, and 1.0× all land on exact `f32` values, and
/// `pitch_hz` values are disjoint from the durations so a stray scale of the pitch
/// array (or a write into it) would observe a wrong number. Shared by the tests.
fn sample() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![100.0, 200.0, 300.0])
        .expect("equal-length arrays of length n must construct")
}

/// Exact sum of `sample()`'s durations: 10 + 20 + 30 == 60.
fn duration_sum(plan: &ProsodyPlan) -> f32 {
    plan.durations_ms.iter().sum()
}

// ----------------------------------------------------------------------------
// C1 — uniform multiply: 2.0× exactly doubles the sum and every entry, 0.5×
//      exactly halves, and a third factor 3.0 (≠ 2.0, 0.5) pins the scale to r.
// ----------------------------------------------------------------------------

/// `scale_rate(2.0)` returns a plan whose summed `durations_ms` is exactly twice
/// the original sum and whose every per-phoneme entry is exactly `2.0×` the
/// original.
#[test]
fn test_scale_rate_two_doubles_sum_and_each_entry() {
    let plan = sample(); // durations [10, 20, 30], sum 60
    let original_sum = duration_sum(&plan); // 60.0

    let scaled = plan.scale_rate(2.0).expect("r == 2.0 is positive => Ok");

    // Summed durations are exactly twice the original sum.
    assert_eq!(duration_sum(&scaled), 2.0 * original_sum); // 120.0
    // Every per-phoneme duration is exactly 2.0× the original.
    assert_eq!(scaled.durations_ms, vec![20.0, 40.0, 60.0]);
}

/// `scale_rate(0.5)` returns a plan whose summed `durations_ms` is exactly half
/// the original sum and whose every per-phoneme entry is exactly `0.5×`.
#[test]
fn test_scale_rate_half_halves_sum_and_each_entry() {
    let plan = sample(); // durations [10, 20, 30], sum 60
    let original_sum = duration_sum(&plan); // 60.0

    let scaled = plan.scale_rate(0.5).expect("r == 0.5 is positive => Ok");

    // Summed durations are exactly half the original sum.
    assert_eq!(duration_sum(&scaled), 0.5 * original_sum); // 30.0
    // Every per-phoneme duration is exactly 0.5× the original.
    assert_eq!(scaled.durations_ms, vec![5.0, 10.0, 15.0]);
}

/// A third factor `3.0`, distinct from `2.0` and `0.5`, scales every entry by
/// exactly `3.0×` and the sum by `3.0×` — so an implementation that hardcoded a
/// doubling/halving (or otherwise ignored `r`) is caught. This pins the scale to
/// the supplied `r`.
#[test]
fn test_scale_rate_three_scales_by_factor_distinct_from_two_and_half() {
    let plan = sample(); // durations [10, 20, 30], sum 60
    let original_sum = duration_sum(&plan); // 60.0

    let scaled = plan.scale_rate(3.0).expect("r == 3.0 is positive => Ok");

    assert_eq!(scaled.durations_ms, vec![30.0, 60.0, 90.0]);
    assert_eq!(duration_sum(&scaled), 3.0 * original_sum); // 180.0

    // The 3.0× result is distinct from both the 2.0× and 0.5× results, pinning
    // the scale factor to r rather than a fixed constant.
    assert_ne!(scaled.durations_ms, vec![20.0, 40.0, 60.0]); // != 2.0×
    assert_ne!(scaled.durations_ms, vec![5.0, 10.0, 15.0]); // != 0.5×
}

// ----------------------------------------------------------------------------
// C2 — pitch is never touched (for 2.0× and 0.5×), and r == 1.0 is a duration
//      identity (element-for-element equal to the input).
// ----------------------------------------------------------------------------

/// `scale_rate(2.0)` leaves `pitch_hz` element-for-element identical to the input
/// — rate scaling does not touch pitch.
#[test]
fn test_scale_rate_two_preserves_pitch() {
    let plan = sample(); // pitch [100, 200, 300]

    let scaled = plan.scale_rate(2.0).expect("r == 2.0 is positive => Ok");

    assert_eq!(scaled.pitch_hz, vec![100.0, 200.0, 300.0]);
    // And identical to the (untouched) input plan's pitch.
    assert_eq!(scaled.pitch_hz, plan.pitch_hz);
}

/// `scale_rate(0.5)` likewise leaves `pitch_hz` element-for-element identical to
/// the input — the other scale direction also never touches pitch.
#[test]
fn test_scale_rate_half_preserves_pitch() {
    let plan = sample(); // pitch [100, 200, 300]

    let scaled = plan.scale_rate(0.5).expect("r == 0.5 is positive => Ok");

    assert_eq!(scaled.pitch_hz, vec![100.0, 200.0, 300.0]);
    assert_eq!(scaled.pitch_hz, plan.pitch_hz);
}

/// `scale_rate(1.0)` returns `durations_ms` element-for-element equal to the input
/// (identity) and leaves `pitch_hz` identical too.
#[test]
fn test_scale_rate_one_is_duration_identity() {
    let plan = sample(); // durations [10, 20, 30], pitch [100, 200, 300]

    let scaled = plan.scale_rate(1.0).expect("r == 1.0 is positive => Ok");

    // Durations element-for-element equal to the input.
    assert_eq!(scaled.durations_ms, vec![10.0, 20.0, 30.0]);
    assert_eq!(scaled.durations_ms, plan.durations_ms);
    // Pitch untouched.
    assert_eq!(scaled.pitch_hz, vec![100.0, 200.0, 300.0]);
}

// ----------------------------------------------------------------------------
// C3 — positive-factor boundary: r <= 0.0 rejects with InvalidRate and produces
//      no scaled plan; any r > 0.0 (down to 0.001) is Ok; nothing panics.
// ----------------------------------------------------------------------------

/// `scale_rate(0.0)` returns `Err(PlanError::InvalidRate)` and produces no scaled
/// plan — pinning the threshold exactly at zero (a `<` rather than `<=` check
/// would wrongly admit `0.0`).
#[test]
fn test_scale_rate_zero_is_invalid_rate() {
    let plan = sample();

    let err = plan
        .scale_rate(0.0)
        .expect_err("r == 0.0 must be rejected as InvalidRate");
    assert!(matches!(err, PlanError::InvalidRate));
}

/// `scale_rate(-1.0)` returns `Err(PlanError::InvalidRate)` — the negative side of
/// the non-positive rejection.
#[test]
fn test_scale_rate_negative_is_invalid_rate() {
    let plan = sample();

    let err = plan
        .scale_rate(-1.0)
        .expect_err("r == -1.0 must be rejected as InvalidRate");
    assert!(matches!(err, PlanError::InvalidRate));
}

/// Any `r > 0.0`, down to a tiny `0.001`, returns `Ok` and scales by exactly that
/// factor — pinning the just-past-zero positive side of the boundary so that the
/// rejection is `r <= 0.0`, not `r <= some_larger_value`.
#[test]
fn test_scale_rate_tiny_positive_is_ok() {
    let plan = sample(); // durations [10, 20, 30]

    let scaled = plan
        .scale_rate(0.001)
        .expect("r == 0.001 is positive => Ok");

    // Scaled by exactly 0.001× (these products are exact in f32).
    assert_eq!(scaled.durations_ms, vec![0.01, 0.02, 0.03]);
    // Pitch still untouched.
    assert_eq!(scaled.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// No input panics: a non-finite-adjacent extreme (`f32::MAX`) on an empty plan
/// and a huge positive factor both return cleanly rather than panicking, and the
/// non-positive extreme `f32::MIN` (a large negative) is rejected as InvalidRate.
#[test]
fn test_scale_rate_never_panics() {
    // A huge positive factor on a populated plan returns Ok (no panic).
    let plan = sample();
    assert!(plan.scale_rate(f32::MAX).is_ok());

    // A large negative factor is non-positive => InvalidRate (no panic).
    assert!(matches!(
        plan.scale_rate(f32::MIN),
        Err(PlanError::InvalidRate)
    ));

    // An empty plan scales to an empty plan without panicking.
    let empty = ProsodyPlan::new(0, vec![], vec![]).expect("empty plan constructs");
    let scaled = empty.scale_rate(2.0).expect("positive r on empty plan is Ok");
    assert_eq!(scaled.durations_ms, Vec::<f32>::new());
    assert_eq!(scaled.pitch_hz, Vec::<f32>::new());
}
