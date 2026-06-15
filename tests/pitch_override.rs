//! Frozen RED tests for T-03.03 — overriding phoneme pitch.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must add to `syrinx-prosody::plan`, on top of the T-03.01 data model and
//! mirroring the duration-override edits of T-03.02 — but for `pitch_hz`, and with
//! an additional per-word span edit:
//!
//!   * `ProsodyPlan::set_pitch(&mut self, i: usize, hz: f32) -> Result<(), PlanError>`
//!     — a single-index pitch write. On success it sets `pitch_hz[i] == hz`
//!     exactly and changes nothing else: every other `pitch_hz` entry and the
//!     entire `durations_ms` array are left untouched. `i == N-1` (the last valid
//!     index) is `Ok`; `i == N` (one past the last) and any larger `usize` yield
//!     `Err(PlanError::IndexOutOfRange)` and mutate nothing — never a panic.
//!   * `ProsodyPlan::set_word_pitch(&mut self, span: Range<usize>, hz: f32)
//!     -> Result<(), PlanError>` — a per-word edit, where a word maps to the
//!     contiguous phoneme span `[start, end)`. On success it sets `pitch_hz[k] ==
//!     hz` for every `k` in the span and leaves every entry outside the span (and
//!     the whole `durations_ms` array) untouched. A span whose `end` exceeds `N`
//!     yields `Err(PlanError::IndexOutOfRange)` and mutates nothing; a span whose
//!     `end == N` (covering the last index) applies — never a panic.
//!
//! Scope (list.md / T-03.03): pitch-array editing only — no duration, volume, or
//! rate control, and no intonation presets (those build on this in T-03.07).
//! Values are caller-supplied per T-03.01. Whether the overridden F0 sounds right
//! on rendered audio is a deferred perceptual eval, not gated here.
//!
//! RED: `ProsodyPlan` exposes neither `set_pitch` nor `set_word_pitch` yet, so
//! these symbols do not resolve and the test target fails to build — every
//! criterion is unmet. GREEN adds the two methods so each assertion holds.

use syrinx_prosody::plan::{PlanError, ProsodyPlan};

/// A populated, schema-current plan with three phonemes whose `durations_ms` and
/// `pitch_hz` values are all distinct — so a write to the wrong slot, or a stray
/// touch of the duration array, would observe a wrong number.
fn sample() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![100.0, 200.0, 300.0])
        .expect("equal-length arrays of length n must construct")
}

/// A wider plan (`N == 5`) with all-distinct entries, used by the word-span tests
/// so a span strictly inside the array leaves entries on BOTH sides untouched.
fn wide_sample() -> ProsodyPlan {
    ProsodyPlan::new(
        5,
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![100.0, 200.0, 300.0, 400.0, 500.0],
    )
    .expect("equal-length arrays of length n must construct")
}

// ----------------------------------------------------------------------------
// C1 — single-index write: exactly one pitch entry changes, durations untouched,
//      and a second distinct value at the same index pins the write to it.
// ----------------------------------------------------------------------------

/// `set_pitch(i, hz)` sets `pitch_hz[i] == hz` exactly, leaves every other pitch
/// entry and the entire `durations_ms` array unchanged, and a second distinct
/// value `HZ2 != HZ` written at the same `i` pins the write to that index.
#[test]
fn test_set_pitch_writes_exact_index_only() {
    let mut plan = sample(); // durations [10, 20, 30], pitch [100, 200, 300]

    // First write HZ == 440.0 at i == 1.
    plan.set_pitch(1, 440.0).expect("i == 1 is in range");
    // Exactly slot 1 changed; slots 0 and 2 are untouched.
    assert_eq!(plan.pitch_hz, vec![100.0, 440.0, 300.0]);
    // The whole duration array is untouched.
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);

    // A second, distinct value HZ2 == 523.0 (!= 440.0) at the SAME index i == 1
    // overwrites only slot 1 — pinning the write to index 1, not 0 or 2.
    plan.set_pitch(1, 523.0).expect("i == 1 is still in range");
    assert_eq!(plan.pitch_hz, vec![100.0, 523.0, 300.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

/// Writing at index 0 lands in slot 0 (and only slot 0), pinning the low end of
/// the index so a write that is offset by one would be caught.
#[test]
fn test_set_pitch_writes_first_index() {
    let mut plan = sample();

    plan.set_pitch(0, 77.0).expect("i == 0 is in range");
    assert_eq!(plan.pitch_hz, vec![77.0, 200.0, 300.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

// ----------------------------------------------------------------------------
// C2 — per-word span write: every index in [start, end) takes hz, every index
//      outside is unchanged; width-1 and width-3 spans each apply exactly.
// ----------------------------------------------------------------------------

/// A width-1 span `[2, 3)` sets exactly index 2 and nothing else — pitch entries
/// 0, 1, 3, 4 stay put and the whole `durations_ms` array is untouched.
#[test]
fn test_set_word_pitch_width_one_applies_to_single_index() {
    let mut plan = wide_sample(); // pitch [100, 200, 300, 400, 500]

    plan.set_word_pitch(2..3, 440.0)
        .expect("span [2, 3) is within N == 5");
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 440.0, 400.0, 500.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
}

/// A width-3 span `[1, 4)` sets exactly indices 1, 2, 3 to `hz` and leaves the
/// entries on BOTH sides of the span (indices 0 and 4) untouched, as well as the
/// whole `durations_ms` array. The edit never bleeds past either span boundary.
#[test]
fn test_set_word_pitch_width_three_applies_across_span() {
    let mut plan = wide_sample(); // pitch [100, 200, 300, 400, 500]

    plan.set_word_pitch(1..4, 440.0)
        .expect("span [1, 4) is within N == 5");
    assert_eq!(plan.pitch_hz, vec![100.0, 440.0, 440.0, 440.0, 500.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
}

// ----------------------------------------------------------------------------
// C3 — boundaries & totality: per-phoneme Ok at N-1 / IndexOutOfRange at N (and
//      beyond); word span Ok when end == N / IndexOutOfRange when end > N; both
//      mutate nothing and never panic on any index.
// ----------------------------------------------------------------------------

/// `set_pitch(N-1, hz)` is `Ok` and applies the write to the last slot.
#[test]
fn test_set_pitch_at_last_index_applies() {
    let mut plan = sample(); // N == 3, last valid index 2

    plan.set_pitch(2, 42.0).expect("i == N-1 must be Ok");
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 42.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

/// `set_pitch(N, hz)` (one past the last) returns `Err(PlanError::IndexOutOfRange)`
/// and mutates nothing — not pitch, not durations, and not a panic.
#[test]
fn test_set_pitch_past_end_errors_and_mutates_nothing() {
    let mut plan = sample(); // N == 3

    let err = plan
        .set_pitch(3, 42.0) // i == N == 3
        .expect_err("i == N must be IndexOutOfRange");
    assert!(matches!(err, PlanError::IndexOutOfRange));
    // Nothing changed.
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0]);
}

/// Per-phoneme index access is total: an empty plan errors at `i == 0`, and the
/// maximum `usize` errors rather than panicking — no `usize` index ever panics.
#[test]
fn test_set_pitch_never_panics_on_any_index() {
    // Empty plan: i == 0 == N is out of range.
    let mut empty = ProsodyPlan::new(0, vec![], vec![]).expect("empty plan constructs");
    assert!(matches!(
        empty.set_pitch(0, 1.0),
        Err(PlanError::IndexOutOfRange)
    ));
    assert_eq!(empty.pitch_hz, Vec::<f32>::new());

    // A wildly out-of-range index returns an error instead of panicking.
    let mut plan = sample(); // N == 3
    assert!(matches!(
        plan.set_pitch(usize::MAX, 1.0),
        Err(PlanError::IndexOutOfRange)
    ));
    // And the plan is untouched.
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0]);
}

/// A word span whose `end == N` (covering the last index) applies — pinning the
/// `Ok` side of the span's end-vs-N boundary.
#[test]
fn test_set_word_pitch_end_at_n_applies() {
    let mut plan = wide_sample(); // N == 5

    // Span [3, 5): end == N, covers indices 3 and 4 (the last index).
    plan.set_word_pitch(3..5, 440.0)
        .expect("a span with end == N must be Ok");
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0, 440.0, 440.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
}

/// A word span whose `end` exceeds `N` returns `Err(PlanError::IndexOutOfRange)`
/// and mutates nothing — neither pitch nor durations, and never a panic. This
/// pins the `Err` side of the span's end-vs-N boundary.
#[test]
fn test_set_word_pitch_end_past_n_errors_and_mutates_nothing() {
    let mut plan = wide_sample(); // N == 5

    // Span [3, 6): end == N+1 == 6 > N == 5.
    let err = plan
        .set_word_pitch(3..6, 440.0)
        .expect_err("a span with end > N must be IndexOutOfRange");
    assert!(matches!(err, PlanError::IndexOutOfRange));
    // Nothing changed — not partially written.
    assert_eq!(plan.pitch_hz, vec![100.0, 200.0, 300.0, 400.0, 500.0]);
    assert_eq!(plan.durations_ms, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
}
