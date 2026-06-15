//! Frozen RED tests for T-01.09 — window cross-sentence context.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `syrinx_frontend::context::window(sentences: &[&str], current: usize,
//!     radius: usize) -> ContextWindow` — assembles a bounded conditioning
//!     window around the sentence at `current`. The returned
//!     `ContextWindow { before, current, after }` exposes:
//!       - `before: Vec<&str>` — up to `radius` sentences immediately preceding
//!         `current`, in source order, clamped at index 0.
//!       - `current: &str` — the target sentence itself.
//!       - `after:  Vec<&str>` — up to `radius` sentences immediately following
//!         `current`, in source order, clamped at the last index.
//!
//! Contract (task T-01.09 / plan.md / spec.md): the window is positional only —
//! no tokenization, no sentence splitting (the input is a pre-split slice), and
//! no semantic relevance weighting. The window is clamped at BOTH ends so it
//! never indexes out of range, and `radius == 0` yields only the current
//! sentence (both `before` and `after` empty).
//!
//! The standing invariant — `before.len() <= radius` and `after.len() <= radius`
//! — is asserted alongside every case below, including the clamped ones where
//! the realized length is strictly less than `radius`.
//!
//! The four criteria are pinned against one another: the interior radius-1
//! window (C1) fixes the un-clamped shape; the first/last index cases (C2) fix
//! both clamp boundaries; the over-radius case (C3) fixes that the window never
//! exceeds the bounded radius even when more sentences exist on one side; and
//! the zero-radius case (C4) fixes the degenerate current-only window.
//!
//! RED: `syrinx-frontend` exposes no `context` module yet, so `window` and
//! `ContextWindow` do not exist and this test target fails to build — every
//! criterion is unmet. GREEN adds the module so each assertion holds.

use syrinx_frontend::context::{window, ContextWindow};

/// The shared, pre-split input used across every criterion. Four single-letter
/// "sentences" make the before/after slices unambiguous to read.
fn sample() -> [&'static str; 4] {
    ["a", "b", "c", "d"]
}

// ----------------------------------------------------------------------------
// C1 — interior radius-1 window. `window(&["a","b","c","d"], 2, 1)` centers on
// "c", reaching exactly one sentence to each side: `before == ["b"]`,
// `after == ["d"]`. This pins the un-clamped shape (both sides realize the full
// radius) and the source ordering. The length checks pin the invariant at the
// boundary `len == radius`.
// ----------------------------------------------------------------------------

#[test]
fn test_interior_radius1_current_is_c() {
    let s = sample();
    let w: ContextWindow = window(&s, 2, 1);
    assert_eq!(w.current, "c", "current must be the sentence at index 2");
}

#[test]
fn test_interior_radius1_before_is_b() {
    let s = sample();
    let w = window(&s, 2, 1);
    assert_eq!(w.before, vec!["b"], "before must be exactly the one prior sentence");
}

#[test]
fn test_interior_radius1_after_is_d() {
    let s = sample();
    let w = window(&s, 2, 1);
    assert_eq!(w.after, vec!["d"], "after must be exactly the one following sentence");
}

#[test]
fn test_interior_radius1_lengths_within_radius() {
    let s = sample();
    let w = window(&s, 2, 1);
    // Both sides realize the full radius here: len == radius (the invariant's
    // upper edge), never more.
    assert_eq!(w.before.len(), 1, "before realizes exactly radius=1 here");
    assert_eq!(w.after.len(), 1, "after realizes exactly radius=1 here");
    assert!(w.before.len() <= 1, "before.len() <= radius");
    assert!(w.after.len() <= 1, "after.len() <= radius");
}

// ----------------------------------------------------------------------------
// C2 — both clamp boundaries. At the FIRST index, `window(&s, 0, 1)` has nothing
// before it, so `before` is empty and `after == ["b"]`. At the LAST index,
// `window(&s, 3, 1)` has nothing after it, so `after` is empty and
// `before == ["c"]`. Together these pin the lower clamp (index 0) and the upper
// clamp (last index) — the window never reads out of range at either edge.
// ----------------------------------------------------------------------------

#[test]
fn test_first_index_before_empty() {
    let s = sample();
    let w = window(&s, 0, 1);
    assert!(w.before.is_empty(), "no sentence precedes index 0");
    assert_eq!(w.before.len(), 0, "before clamps to empty at the first index");
}

#[test]
fn test_first_index_after_is_b() {
    let s = sample();
    let w = window(&s, 0, 1);
    assert_eq!(w.current, "a", "current must be the sentence at index 0");
    assert_eq!(w.after, vec!["b"], "after at index 0 is the single following sentence");
}

#[test]
fn test_last_index_after_empty() {
    let s = sample();
    let w = window(&s, 3, 1);
    assert!(w.after.is_empty(), "no sentence follows the last index");
    assert_eq!(w.after.len(), 0, "after clamps to empty at the last index");
}

#[test]
fn test_last_index_before_is_c() {
    let s = sample();
    let w = window(&s, 3, 1);
    assert_eq!(w.current, "d", "current must be the sentence at the last index");
    assert_eq!(w.before, vec!["c"], "before at the last index is the single prior sentence");
}

// ----------------------------------------------------------------------------
// C3 — over-radius clamp. With `radius=2` on the same four-sentence input,
// `window(&s, 1, 2)` would reach two sentences to each side, but only one exists
// before index 1: `before == ["a"]` (clamped to 1, NOT 2), while two exist
// after: `after == ["c","d"]` (exactly 2). This pins that the window is bounded
// by the available range on the short side yet realizes the full radius on the
// long side — it never exceeds `radius` and never reads out of range.
// ----------------------------------------------------------------------------

#[test]
fn test_over_radius_before_clamped_to_one() {
    let s = sample();
    let w = window(&s, 1, 2);
    assert_eq!(
        w.before,
        vec!["a"],
        "before must clamp to the single available prior sentence, not radius=2"
    );
    assert_eq!(w.before.len(), 1, "before clamps to 1 (< radius)");
    assert!(w.before.len() <= 2, "before.len() <= radius even when clamped");
}

#[test]
fn test_over_radius_after_is_two() {
    let s = sample();
    let w = window(&s, 1, 2);
    assert_eq!(
        w.after,
        vec!["c", "d"],
        "after must realize exactly the two following sentences, in order"
    );
    assert_eq!(w.after.len(), 2, "after realizes the full radius=2 here");
    assert!(w.after.len() <= 2, "after.len() <= radius");
}

#[test]
fn test_over_radius_current_is_b() {
    let s = sample();
    let w = window(&s, 1, 2);
    assert_eq!(w.current, "b", "current must be the sentence at index 1");
}

// ----------------------------------------------------------------------------
// C4 — zero-radius boundary. `window(&s, 1, 0)` yields only the current
// sentence: `before` and `after` are both empty and `current == "b"`. This pins
// the degenerate radius-0 window where the invariant collapses to len == 0 on
// both sides.
// ----------------------------------------------------------------------------

#[test]
fn test_zero_radius_current_only() {
    let s = sample();
    let w = window(&s, 1, 0);
    assert_eq!(w.current, "b", "current must be the sentence at index 1");
    assert!(w.before.is_empty(), "radius 0 yields no preceding sentences");
    assert!(w.after.is_empty(), "radius 0 yields no following sentences");
    assert_eq!(w.before.len(), 0, "before.len() == 0 at radius 0");
    assert_eq!(w.after.len(), 0, "after.len() == 0 at radius 0");
}
