//! Frozen RED tests for T-03.02 — overriding phoneme durations.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must add to `syrinx-prosody::plan`, on top of the T-03.01 data model:
//!
//!   * `ProsodyPlan::set_duration(&mut self, i: usize, v: f32) -> Result<(), PlanError>`
//!     — a single-index duration write. On success it sets `durations_ms[i] == v`
//!     exactly and changes nothing else: every other `durations_ms` entry and the
//!     entire `pitch_hz` array are left untouched. `i == N-1` (the last valid
//!     index) is `Ok`; `i == N` (one past the last) and any larger `usize` yield
//!     `Err(PlanError::IndexOutOfRange)` and mutate nothing — never a panic.
//!   * `ProsodyPlan::override_durations(&mut self, new: Vec<f32>) -> Result<(), PlanError>`
//!     — a whole-array replacement. It succeeds iff `new.len() == N`, after which
//!     `durations_ms == new` (and `pitch_hz` is untouched). A `new` of length
//!     `N+1` or `N-1` returns `Err(PlanError::LengthMismatch)` and leaves the
//!     plan's `durations_ms` unchanged — atomic, no partial write, never a panic.
//!
//! Scope (list.md / T-03.02): duration-array editing only — no pitch, volume, or
//! rate control, no prediction, no defaults. Values are caller-supplied per
//! T-03.01. Whether the overridden timing sounds right on rendered audio is a
//! deferred perceptual eval, not gated here.
//!
//! RED: `ProsodyPlan` exposes neither `set_duration` nor `override_durations`
//! yet, so these symbols do not resolve and the test target fails to build —
//! every criterion is unmet. GREEN adds the two methods so each assertion holds.

use syrinx_prosody::plan::{PlanError, ProsodyPlan};

/// A populated, schema-current plan with three phonemes whose `durations_ms` and
/// `pitch_hz` values are all distinct — so a write to the wrong slot, or a stray
/// touch of the pitch array, would observe a wrong number. Shared by the tests.
fn sample() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![100.0, 200.0, 300.0])
        .expect("equal-length arrays of length n must construct")
}

// ----------------------------------------------------------------------------
// C1 — single-index write: exactly one duration entry changes, pitch untouched,
//      and a second distinct value at the same index pins the write to it.
// ----------------------------------------------------------------------------

/// `set_duration(i, v)` sets `durations_ms[i] == v` exactly, leaves every other
/// duration entry and the entire `pitch_hz` array unchanged, and a second
/// distinct value `V2 != V` written at the same `i` pins the write to that index.
#[test]
fn test_set_duration_writes_exact_index_only() {
    let mut plan = sample(); // durations [10, 20, 30], pitch [100, 200, 300]

    // First write V == 99.0 at i == 1.
    plan.set_duration(1, 99.0).expect("i == 1 is in range");
    // Exactly slot 1 changed; slots 0 and 2 are untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 99.0, 30.0]);
    // The whole pitch array is untouched.
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);

    // A second, distinct value V2 == 55.0 (!= 99.0) at the SAME index i == 1
    // overwrites only slot 1 — pinning the write to index 1, not 0 or 2.
    plan.set_duration(1, 55.0).expect("i == 1 is still in range");
    assert_eq!(plan.durations_ms, vec![10.0, 55.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// Writing at index 0 lands in slot 0 (and only slot 0), pinning the low end of
/// the index so a write that is offset by one would be caught.
#[test]
fn test_set_duration_writes_first_index() {
    let mut plan = sample();

    plan.set_duration(0, 7.0).expect("i == 0 is in range");
    assert_eq!(plan.durations_ms, vec![7.0, 20.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

// ----------------------------------------------------------------------------
// C2 — bulk replacement: replace iff new.len() == N; N+1 and N-1 both reject
//      atomically with LengthMismatch, leaving durations unchanged.
// ----------------------------------------------------------------------------

/// `override_durations(new)` with `new.len() == N` returns `Ok` and makes
/// `durations_ms == new`; `pitch_hz` is left untouched.
#[test]
fn test_override_durations_replaces_when_len_equals_n() {
    let mut plan = sample(); // N == 3

    plan.override_durations(vec![1.5, 2.5, 3.5])
        .expect("a replacement of length N must be Ok");
    assert_eq!(plan.durations_ms, vec![1.5, 2.5, 3.5]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// A `new` of length `N+1` returns `Err(PlanError::LengthMismatch)` and leaves
/// the plan's `durations_ms` unchanged — atomic, no partial write, no panic.
#[test]
fn test_override_durations_too_long_rejects_atomically() {
    let mut plan = sample(); // N == 3

    let err = plan
        .override_durations(vec![1.0, 2.0, 3.0, 4.0]) // len N+1 == 4
        .expect_err("a replacement of length N+1 must be rejected");
    assert!(matches!(err, PlanError::LengthMismatch));
    // Unchanged — not partially written.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// A `new` of length `N-1` returns `Err(PlanError::LengthMismatch)` and leaves
/// the plan's `durations_ms` unchanged — the other side of the length check.
#[test]
fn test_override_durations_too_short_rejects_atomically() {
    let mut plan = sample(); // N == 3

    let err = plan
        .override_durations(vec![1.0, 2.0]) // len N-1 == 2
        .expect_err("a replacement of length N-1 must be rejected");
    assert!(matches!(err, PlanError::LengthMismatch));
    // Unchanged — not partially written.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

// ----------------------------------------------------------------------------
// C3 — single-index boundary: Ok at N-1, IndexOutOfRange at N (and beyond),
//      mutating nothing; no usize index panics.
// ----------------------------------------------------------------------------

/// `set_duration(N-1, v)` is `Ok` and applies the write to the last slot.
#[test]
fn test_set_duration_at_last_index_applies() {
    let mut plan = sample(); // N == 3, last valid index 2

    plan.set_duration(2, 42.0).expect("i == N-1 must be Ok");
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 42.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// `set_duration(N, v)` (one past the last) returns
/// `Err(PlanError::IndexOutOfRange)` and mutates nothing — not durations, not
/// pitch, and not a panic.
#[test]
fn test_set_duration_past_end_errors_and_mutates_nothing() {
    let mut plan = sample(); // N == 3

    let err = plan
        .set_duration(3, 42.0) // i == N == 3
        .expect_err("i == N must be IndexOutOfRange");
    assert!(matches!(err, PlanError::IndexOutOfRange));
    // Nothing changed.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// Index access is total: an empty plan errors at `i == 0`, and the maximum
/// `usize` errors rather than panicking — no `usize` index ever panics.
#[test]
fn test_set_duration_never_panics_on_any_index() {
    // Empty plan: i == 0 == N is out of range.
    let mut empty = ProsodyPlan::new(0, vec![], vec![]).expect("empty plan constructs");
    assert!(matches!(
        empty.set_duration(0, 1.0),
        Err(PlanError::IndexOutOfRange)
    ));
    assert_eq!(empty.durations_ms, Vec::<f32>::new());

    // A wildly out-of-range index returns an error instead of panicking.
    let mut plan = sample(); // N == 3
    assert!(matches!(
        plan.set_duration(usize::MAX, 1.0),
        Err(PlanError::IndexOutOfRange)
    ));
    // And the plan is untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}
