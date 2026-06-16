//! Frozen RED tests for T-03.07 — applying intonation contours.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must add to `syrinx-prosody` (a new `contour` module), on top of the
//! T-03.01 data model and alongside the T-03.03 pitch-override edits. Two edits on
//! an existing [`ProsodyPlan`], both touching only `pitch_hz` and never
//! `durations_ms`:
//!
//!   * `ProsodyPlan::apply_contour(&mut self, contour: Contour) -> Result<(), PlanError>`
//!     — apply a named preset to the pitch array as a deterministic additive linear
//!     ramp over the `N` phonemes. The preset's specified delta is **30.0 Hz**:
//!       - `Contour::Rising` adds a ramp `0 → +30.0` across the array, so the first
//!         entry is unchanged and the last entry ends `+30.0` Hz above where it
//!         started — strictly greater than the (unchanged) first by the delta on a
//!         flat input — with interior phonemes linearly interpolated.
//!       - `Contour::Falling` adds the mirror ramp `0 → -30.0`, so the last entry
//!         ends strictly below the first by the delta on a flat input.
//!       - `Contour::Flat` adds an all-zero ramp: every `pitch_hz` entry is left
//!         exactly unchanged (identity), even when the input pitch is non-uniform.
//!     For a flat-`B` plan of `N == 3` the concrete results are:
//!       Rising → `[B, B+15, B+30]`, Falling → `[B, B-15, B-30]`, Flat → identity.
//!   * `ProsodyPlan::apply_curve(&mut self, curve: &[f32]) -> Result<(), PlanError>`
//!     — apply a manual per-phoneme F0 curve. A curve of length `N` sets `pitch_hz`
//!     element-for-element equal to the supplied curve; a curve whose length `!= N`
//!     (longer or shorter) yields `Err(PlanError::LengthMismatch)` and mutates
//!     nothing.
//!
//! Edges: an empty plan (`N == 0`) is a no-op `Ok` for every preset and for an
//! (empty) manual curve, leaving `pitch_hz` an unchanged empty vec; `durations_ms`
//! is untouched by every contour application; nothing panics.
//!
//! Scope (list.md / T-03.07): contour shape over the pitch array only — no emotion
//! semantics and no per-phoneme/per-word pitch API (that is T-03.03). Whether the
//! applied contour is perceptually the intended intonation on rendered audio is a
//! deferred perceptual eval against the real model, not gated here.
//!
//! RED: `syrinx-prosody` exposes neither a `contour` module / `Contour` type nor
//! `apply_contour` / `apply_curve` yet, so these symbols do not resolve and the
//! test target fails to build — every criterion is unmet. GREEN adds the module,
//! the enum, and the two methods so each assertion holds.

use syrinx_prosody::contour::Contour;
use syrinx_prosody::plan::{PlanError, ProsodyPlan};

/// The preset's specified delta, in Hz: Rising raises the last entry this far above
/// the first (on a flat input), Falling lowers it this far below.
const DELTA_HZ: f32 = 30.0;

/// Absolute tolerance for the "within tolerance" contour comparisons.
const TOL: f32 = 1e-3;

/// Assert two `f32` slices are equal element-for-element within [`TOL`].
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {actual:?} vs {expected:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "index {i}: {a} not within {TOL} of {e} ({actual:?} vs {expected:?})"
        );
    }
}

/// A flat-pitch plan of `N == 3` (every `pitch_hz` entry `100.0`) with distinct,
/// non-uniform `durations_ms` — so a flat input makes the rising/falling deltas
/// exact, while a stray touch of the duration array would be observed.
fn flat_plan() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![100.0, 100.0, 100.0])
        .expect("equal-length arrays of length n must construct")
}

/// A non-uniform plan of `N == 3` whose `pitch_hz` entries are all distinct — used
/// by the manual-curve and flat-identity tests so a wrong-slot write or a spurious
/// ramp would be observed.
fn varied_plan() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![110.0, 95.0, 130.0])
        .expect("equal-length arrays of length n must construct")
}

// ----------------------------------------------------------------------------
// C1 — preset direction: Rising raises last above first by the delta, Falling
//      lowers last below first by the delta, Flat is identity. Both sides pinned.
// ----------------------------------------------------------------------------

/// `Contour::Rising` adds a `0 → +DELTA` ramp: first unchanged, last strictly
/// greater than first by `DELTA` within tolerance, interior linearly interpolated.
/// `durations_ms` is untouched.
#[test]
fn test_rising_preset_raises_last_above_first_by_delta() {
    let mut plan = flat_plan(); // pitch [100, 100, 100]

    plan.apply_contour(Contour::Rising)
        .expect("applying a preset to a non-empty plan is Ok");

    // Concrete additive ramp 0 -> +30 over N == 3: [100, 115, 130].
    assert_close(&plan.pitch_hz, &[100.0, 115.0, 130.0]);

    let first = plan.pitch_hz[0];
    let last = plan.pitch_hz[2];
    // Direction: last strictly greater than first ...
    assert!(last > first, "rising: last {last} must exceed first {first}");
    // ... by the preset's specified delta within tolerance.
    assert!(
        (last - first - DELTA_HZ).abs() <= TOL,
        "rising: last-first {} must be {DELTA_HZ} within {TOL}",
        last - first
    );

    // Durations untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

/// `Contour::Falling` adds a `0 → -DELTA` ramp: first unchanged, last strictly less
/// than first by `DELTA` within tolerance. `durations_ms` is untouched.
#[test]
fn test_falling_preset_lowers_last_below_first_by_delta() {
    let mut plan = flat_plan(); // pitch [100, 100, 100]

    plan.apply_contour(Contour::Falling)
        .expect("applying a preset to a non-empty plan is Ok");

    // Concrete additive ramp 0 -> -30 over N == 3: [100, 85, 70].
    assert_close(&plan.pitch_hz, &[100.0, 85.0, 70.0]);

    let first = plan.pitch_hz[0];
    let last = plan.pitch_hz[2];
    // Direction: last strictly less than first ...
    assert!(last < first, "falling: last {last} must be below first {first}");
    // ... by the preset's specified delta within tolerance.
    assert!(
        (last - first + DELTA_HZ).abs() <= TOL,
        "falling: last-first {} must be {} within {TOL}",
        last - first,
        -DELTA_HZ
    );

    // Durations untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

/// `Contour::Flat` leaves every `pitch_hz` entry exactly unchanged — even on a
/// non-uniform input, so a ramp anchored at the first entry would be caught. This
/// pins flat as identity. `durations_ms` is untouched.
#[test]
fn test_flat_preset_is_identity() {
    let mut plan = varied_plan(); // pitch [110, 95, 130]

    plan.apply_contour(Contour::Flat)
        .expect("applying a preset to a non-empty plan is Ok");

    // Every entry bit-identical: Flat adds an all-zero ramp.
    assert_eq!(plan.pitch_hz, vec![110.0, 95.0, 130.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

// ----------------------------------------------------------------------------
// C2 — manual curve: length-N curve sets pitch pointwise; length != N (longer or
//      shorter) errors with LengthMismatch and leaves the plan unchanged.
// ----------------------------------------------------------------------------

/// A manual curve of length `N` sets `pitch_hz` element-for-element equal to the
/// supplied curve, and leaves `durations_ms` untouched.
#[test]
fn test_manual_curve_sets_pitch_pointwise() {
    let mut plan = varied_plan(); // pitch [110, 95, 130], durations [10, 20, 30]

    let curve = [111.0, 222.0, 333.0];
    plan.apply_curve(&curve)
        .expect("a length-N curve applies");

    // pitch_hz is exactly the supplied curve, point-for-point.
    assert_eq!(plan.pitch_hz, vec![111.0, 222.0, 333.0]);
    // Durations untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

/// A manual curve whose length `!= N` — both a longer (`N+1`) and a shorter (`N-1`)
/// curve — returns `Err(PlanError::LengthMismatch)` and mutates nothing: neither
/// `pitch_hz` nor `durations_ms`, and never a panic. Pins the length boundary on
/// both sides.
#[test]
fn test_manual_curve_wrong_length_errs_and_unchanged() {
    // Longer than N: length 4 vs N == 3.
    let mut plan = varied_plan();
    let too_long = [1.0, 2.0, 3.0, 4.0];
    let err = plan
        .apply_curve(&too_long)
        .expect_err("a curve longer than N must be LengthMismatch");
    assert!(matches!(err, PlanError::LengthMismatch));
    assert_eq!(plan.pitch_hz, vec![110.0, 95.0, 130.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);

    // Shorter than N: length 2 vs N == 3.
    let mut plan = varied_plan();
    let too_short = [1.0, 2.0];
    let err = plan
        .apply_curve(&too_short)
        .expect_err("a curve shorter than N must be LengthMismatch");
    assert!(matches!(err, PlanError::LengthMismatch));
    assert_eq!(plan.pitch_hz, vec![110.0, 95.0, 130.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

// ----------------------------------------------------------------------------
// C3 — empty plan no-op for every preset and a manual curve; durations untouched
//      by every contour application; nothing panics.
// ----------------------------------------------------------------------------

/// Applying any preset to an empty plan (`N == 0`) is a no-op that returns `Ok`
/// with an unchanged empty `pitch_hz` and an unchanged empty `durations_ms`.
#[test]
fn test_contour_empty_plan_is_noop() {
    for contour in [Contour::Rising, Contour::Falling, Contour::Flat] {
        let mut empty = ProsodyPlan::new(0, vec![], vec![]).expect("empty plan constructs");
        empty
            .apply_contour(contour)
            .expect("a preset on an empty plan is a no-op Ok");
        assert_eq!(empty.pitch_hz, Vec::<f32>::new());
        assert_eq!(empty.durations_ms, Vec::<f32>::new());
    }
}

/// Applying a (matching, empty) manual curve to an empty plan (`N == 0`) is a no-op
/// that returns `Ok` with an unchanged empty `pitch_hz` and empty `durations_ms`.
#[test]
fn test_curve_empty_plan_is_noop() {
    let mut empty = ProsodyPlan::new(0, vec![], vec![]).expect("empty plan constructs");
    let curve: [f32; 0] = [];
    empty
        .apply_curve(&curve)
        .expect("an empty curve on an empty plan is a no-op Ok");
    assert_eq!(empty.pitch_hz, Vec::<f32>::new());
    assert_eq!(empty.durations_ms, Vec::<f32>::new());
}

/// Every contour application — each preset and a manual curve — leaves the whole
/// `durations_ms` array element-for-element identical to the input.
#[test]
fn test_durations_untouched_by_every_contour() {
    let original_durations = vec![10.0, 20.0, 30.0];

    for contour in [Contour::Rising, Contour::Falling, Contour::Flat] {
        let mut plan = flat_plan();
        plan.apply_contour(contour).expect("preset applies");
        assert_eq!(plan.durations_ms, original_durations);
    }

    let mut plan = flat_plan();
    plan.apply_curve(&[1.0, 2.0, 3.0]).expect("length-N curve applies");
    assert_eq!(plan.durations_ms, original_durations);
}
