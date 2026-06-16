//! Frozen RED tests for T-03.09 — editing a phoneme in the plan.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must add to `syrinx-prosody::plan`, on top of the T-03.01 data model and
//! unifying the field-specific edits of T-03.02 (duration) and T-03.03 (pitch) into
//! one typed single-phoneme editor:
//!
//!   * `PhonemeEdit { duration_ms: Option<f32>, pitch_hz: Option<f32> }` — a typed
//!     edit carrying an OPTIONAL new duration and/or pitch for one phoneme index.
//!     A `Some` field requests writing that field; a `None` field leaves the
//!     corresponding entry of the plan equal to the original. The two fields are
//!     written independently.
//!   * `ProsodyPlan::edit_phoneme(&self, i: usize, edit: PhonemeEdit)
//!     -> Result<ProsodyPlan, PlanError>` — returns a NEW plan in which index `i`
//!     carries exactly the edited values and every other entry of both arrays is
//!     bit-identical to the original. `i == N-1` (the last valid index) is `Ok` and
//!     applies; `i == N` (one past the last) and any larger `usize` yield
//!     `Err(PlanError::IndexOutOfRange)` and mutate nothing — never a panic. The
//!     receiver is `&self`, so the source plan is never mutated.
//!
//! Scope (list.md / T-03.09): duration and pitch only — no volume editing, no
//! batch/scripted edit language (single-phoneme edits). Values are caller-supplied
//! per T-03.01. Whether a renderer audibly honors the edit is a deferred perceptual
//! eval, not gated here.
//!
//! RED: `ProsodyPlan` exposes no `edit_phoneme` method and `PhonemeEdit` does not
//! exist yet, so these symbols do not resolve and the test target fails to build —
//! every criterion is unmet. GREEN adds the type and method so each assertion holds.

use syrinx_prosody::plan::{PhonemeEdit, PlanError, ProsodyPlan};

/// A populated, schema-current plan with three phonemes whose `durations_ms` and
/// `pitch_hz` values are all distinct — so a write to the wrong slot, a write to
/// the wrong array, or a stray touch of an untargeted entry would observe a wrong
/// number. Shared by the tests.
fn sample() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![100.0, 200.0, 300.0])
        .expect("equal-length arrays of length n must construct")
}

// ----------------------------------------------------------------------------
// C1 — a full edit (both fields Some) writes durations_ms[i] and pitch_hz[i] to
//      exactly the edited values, with every other index of both arrays
//      bit-identical to the original and the source plan itself never mutated.
// ----------------------------------------------------------------------------

/// `edit_phoneme(i, PhonemeEdit { duration_ms: Some(d), pitch_hz: Some(p) })`
/// returns a new plan with `durations_ms[i] == d` and `pitch_hz[i] == p` exactly,
/// every other entry of both arrays bit-identical to the original, the schema
/// version carried over, and the source plan left completely unmutated (`&self`).
#[test]
fn test_edit_phoneme_full_edit_writes_both_fields_at_index() {
    let plan = sample(); // durations [10, 20, 30], pitch [100, 200, 300]

    let edited = plan
        .edit_phoneme(
            1,
            PhonemeEdit {
                duration_ms: Some(99.0),
                pitch_hz: Some(440.0),
            },
        )
        .expect("i == 1 is in range");

    // Index 1 carries exactly the edited values; indices 0 and 2 are unchanged in
    // BOTH arrays — the returned plan reflects exactly that edit and nothing else.
    assert_eq!(edited.durations_ms, vec![10.0, 99.0, 30.0]);
    assert_eq!(edited.pitch_hz, vec![100.0, 440.0, 300.0]);
    // The schema version is preserved on the returned plan.
    assert_eq!(edited.schema_version, plan.schema_version);

    // The source plan was not mutated — `edit_phoneme` takes `&self` and returns a
    // fresh plan.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// A full edit at index 0 lands in slot 0 of both arrays (and only slot 0),
/// pinning the low end of the index so a write that is offset by one would be
/// caught.
#[test]
fn test_edit_phoneme_full_edit_writes_first_index() {
    let plan = sample();

    let edited = plan
        .edit_phoneme(
            0,
            PhonemeEdit {
                duration_ms: Some(7.0),
                pitch_hz: Some(77.0),
            },
        )
        .expect("i == 0 is in range");

    assert_eq!(edited.durations_ms, vec![7.0, 20.0, 30.0]);
    assert_eq!(edited.pitch_hz, vec![77.0, 200.0, 300.0]);
}

// ----------------------------------------------------------------------------
// C2 — the two fields are written independently: a duration-only edit
//      (pitch_hz: None) leaves pitch_hz[i] equal to the original, and a
//      pitch-only edit (duration_ms: None) leaves durations_ms[i] equal to the
//      original.
// ----------------------------------------------------------------------------

/// A duration-only edit (`pitch_hz: None`) changes `durations_ms[i]` to the new
/// value while leaving `pitch_hz[i]` — and every other entry of both arrays —
/// equal to the original. Pins that the pitch field is not touched when its edit
/// field is `None`.
#[test]
fn test_edit_phoneme_duration_only_leaves_pitch_unchanged() {
    let plan = sample(); // durations [10, 20, 30], pitch [100, 200, 300]

    let edited = plan
        .edit_phoneme(
            1,
            PhonemeEdit {
                duration_ms: Some(99.0),
                pitch_hz: None,
            },
        )
        .expect("i == 1 is in range");

    // Duration at index 1 changed...
    assert_eq!(edited.durations_ms, vec![10.0, 99.0, 30.0]);
    // ...while pitch at index 1 (and everywhere) is exactly the original.
    assert_eq!(edited.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// A pitch-only edit (`duration_ms: None`) changes `pitch_hz[i]` to the new value
/// while leaving `durations_ms[i]` — and every other entry of both arrays — equal
/// to the original. Pins that the duration field is not touched when its edit
/// field is `None`.
#[test]
fn test_edit_phoneme_pitch_only_leaves_duration_unchanged() {
    let plan = sample(); // durations [10, 20, 30], pitch [100, 200, 300]

    let edited = plan
        .edit_phoneme(
            1,
            PhonemeEdit {
                duration_ms: None,
                pitch_hz: Some(440.0),
            },
        )
        .expect("i == 1 is in range");

    // Pitch at index 1 changed...
    assert_eq!(edited.pitch_hz, vec![100.0, 440.0, 300.0]);
    // ...while duration at index 1 (and everywhere) is exactly the original.
    assert_eq!(edited.durations_ms, vec![10.0, 20.0, 30.0]);
}

// ----------------------------------------------------------------------------
// C3 — boundaries & totality: Ok at N-1 and applies; IndexOutOfRange at N (and
//      beyond) mutating nothing; no usize index panics.
// ----------------------------------------------------------------------------

/// `edit_phoneme(N-1, ..)` is `Ok` and applies the write to the last slot of both
/// arrays — pinning the `Ok` side of the index-vs-N boundary.
#[test]
fn test_edit_phoneme_at_last_index_applies() {
    let plan = sample(); // N == 3, last valid index 2

    let edited = plan
        .edit_phoneme(
            2,
            PhonemeEdit {
                duration_ms: Some(42.0),
                pitch_hz: Some(43.0),
            },
        )
        .expect("i == N-1 must be Ok");

    assert_eq!(edited.durations_ms, vec![10.0, 20.0, 42.0]);
    assert_eq!(edited.pitch_hz, vec![100.0, 200.0, 43.0]);
}

/// `edit_phoneme(N, ..)` (one past the last) returns
/// `Err(PlanError::IndexOutOfRange)` and mutates nothing — the source plan's
/// arrays are unchanged and there is no panic. Pins the `Err` side of the
/// index-vs-N boundary.
#[test]
fn test_edit_phoneme_at_n_errors_and_mutates_nothing() {
    let plan = sample(); // N == 3

    let err = plan
        .edit_phoneme(
            3, // i == N == 3
            PhonemeEdit {
                duration_ms: Some(42.0),
                pitch_hz: Some(43.0),
            },
        )
        .expect_err("i == N must be IndexOutOfRange");
    assert!(matches!(err, PlanError::IndexOutOfRange));

    // The source plan is untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// Index access is total: an empty plan errors at `i == 0`, and the maximum
/// `usize` errors rather than panicking — no `usize` index ever panics, and the
/// source plan is untouched in each case.
#[test]
fn test_edit_phoneme_never_panics_on_any_index() {
    // Empty plan: i == 0 == N is out of range. (A fresh edit per call so the API
    // need not be `Copy`.)
    let empty = ProsodyPlan::new(0, vec![], vec![]).expect("empty plan constructs");
    assert!(matches!(
        empty.edit_phoneme(
            0,
            PhonemeEdit {
                duration_ms: Some(1.0),
                pitch_hz: Some(2.0),
            }
        ),
        Err(PlanError::IndexOutOfRange)
    ));
    assert_eq!(empty.durations_ms, Vec::<f32>::new());
    assert_eq!(empty.pitch_hz, Vec::<f32>::new());

    // A wildly out-of-range index returns an error instead of panicking.
    let plan = sample(); // N == 3
    assert!(matches!(
        plan.edit_phoneme(
            usize::MAX,
            PhonemeEdit {
                duration_ms: Some(1.0),
                pitch_hz: Some(2.0),
            }
        ),
        Err(PlanError::IndexOutOfRange)
    ));
    // And the source plan is untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}
