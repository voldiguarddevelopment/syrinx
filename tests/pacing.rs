//! Frozen RED tests for T-01.10 — compute paragraph pacing.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `syrinx_frontend::pacing::breath_markers(text: &str,
//!     interval_words: usize) -> Vec<usize>` — scans `text` and returns the
//!     ordered list of breath-marker positions. A position is the count of words
//!     that precede the marker (a global, 1-based word index): a marker at
//!     position `N` sits immediately after the N-th word.
//!
//! Contract (task T-01.10 / plan.md / spec.md): insertion is deterministic and
//! positional only — no prosodic duration is assigned to breaths, and no
//! language-specific breathing model is applied (a uniform word-interval policy).
//! Two rules generate the positions:
//!
//!   1. Interior interval: within a paragraph, a breath marker falls strictly
//!      AFTER each completed interval of `interval_words` words — but only when
//!      the running word count EXCEEDS the interval (the boundary is `>`, not
//!      `>=`). A paragraph whose word count exactly equals the interval reaches
//!      the boundary without exceeding it and therefore emits no interior marker.
//!   2. Paragraph break: every boundary BETWEEN two paragraphs forces a breath
//!      marker, regardless of either paragraph's length. (The end of the whole
//!      text is not a paragraph break and forces nothing.)
//!
//! The four criteria are pinned against one another: the 25-word single
//! paragraph (C1) fixes the interior-interval positions [10, 20]; the 10-vs-11
//! word pair (C2) fixes the off-by-one boundary at `> interval`; the repeated
//! call (C3) fixes determinism; and the two short paragraphs (C4) fix that a
//! paragraph break forces a marker even below the interval.
//!
//! RED: `syrinx-frontend` exposes no `pacing` module yet, so `breath_markers`
//! does not exist and this test target fails to build — every criterion is
//! unmet. GREEN adds the module so each assertion holds.

use syrinx_frontend::pacing::breath_markers;

/// Build a single paragraph of `n` distinct space-separated words ("w1 w2 …").
/// Distinct tokens keep the word count unambiguous and contain no paragraph
/// breaks, so only the interior-interval rule can apply.
fn paragraph(n: usize) -> String {
    (1..=n)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ----------------------------------------------------------------------------
// C1 — interior interval. A 25-word single paragraph at interval 10 inserts a
// breath marker after word 10 and after word 20: exactly two markers, and the
// positions are exactly those word indices, in order.
// ----------------------------------------------------------------------------

#[test]
fn test_interior_interval_two_markers_at_10_and_20() {
    let text = paragraph(25);
    let markers = breath_markers(&text, 10);
    assert_eq!(
        markers,
        vec![10, 20],
        "25 words at interval 10 → markers after word 10 and word 20"
    );
    assert_eq!(markers.len(), 2, "exactly two interior markers, no more");
}

// ----------------------------------------------------------------------------
// C2 — the `> interval` (not `>=`) boundary, pinned from both sides. A paragraph
// of exactly 10 words reaches the interval without exceeding it → ZERO markers.
// A paragraph of 11 words exceeds it by one → exactly ONE marker, at position 10.
// ----------------------------------------------------------------------------

#[test]
fn test_boundary_exactly_ten_words_no_marker() {
    let text = paragraph(10);
    let markers = breath_markers(&text, 10);
    assert_eq!(
        markers,
        Vec::<usize>::new(),
        "10 words exactly equals the interval (not >) → no breath marker"
    );
}

#[test]
fn test_boundary_eleven_words_one_marker() {
    let text = paragraph(11);
    let markers = breath_markers(&text, 10);
    assert_eq!(
        markers,
        vec![10],
        "11 words exceeds the interval → one marker, after word 10"
    );
    assert_eq!(markers.len(), 1, "exactly one marker just past the boundary");
}

// ----------------------------------------------------------------------------
// C3 — determinism. Calling `breath_markers` twice on identical input returns
// identical marker positions. The equality-to-[10,20] check guards against a
// trivially-deterministic empty result.
// ----------------------------------------------------------------------------

#[test]
fn test_deterministic_repeated_calls() {
    let text = paragraph(25);
    let first = breath_markers(&text, 10);
    let second = breath_markers(&text, 10);
    assert_eq!(first, second, "identical input must yield identical positions");
    assert_eq!(
        first,
        vec![10, 20],
        "the deterministic result is the interior-interval positions, not empty"
    );
}

// ----------------------------------------------------------------------------
// C4 — paragraph break forces a marker. Two paragraphs of three words each
// (both below the interval of 10) are separated by a blank-line break. The break
// forces exactly one marker, at the paragraph boundary after word 3. No interior
// marker fires (neither paragraph exceeds the interval) and the text end forces
// nothing.
// ----------------------------------------------------------------------------

#[test]
fn test_paragraph_break_forces_single_marker() {
    let text = "aa bb cc\n\ndd ee ff";
    let markers = breath_markers(text, 10);
    assert_eq!(
        markers,
        vec![3],
        "the break between two 3-word paragraphs forces one marker, after word 3"
    );
    assert_eq!(markers.len(), 1, "exactly one marker — the paragraph break only");
}
