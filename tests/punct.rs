//! Frozen RED tests for T-01.08 — map punctuation to prosody.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `syrinx_frontend::punct::hints(input: &str) -> Vec<ProsodyHint>`
//!     — a total mapping from a *normalized* `&str` to an ordered list of typed
//!     prosody markers, one per recognized punctuation mark, in source order. It
//!     never panics and never returns markers for unpunctuated text.
//!
//! Contract (task T-01.08 / plan.md / spec.md): exactly four punctuation marks
//! are recognized, each mapping to one typed `ProsodyHint`:
//!
//!   * `.` (period)      -> `Boundary { tone: Falling, strength: Full }`
//!   * `,` (comma)       -> `Break    { kind: Short }`
//!   * `?` (question)    -> `Boundary { tone: Rising,  strength: Full }`
//!   * `!` (exclamation) -> `Boundary { tone: Falling, strength: Exclamatory }`
//!
//! These four are pinned *against one another*: period↔comma is a `Boundary`
//! vs a `Break` (C1/C2); `?`↔`!` is a `Rising` vs a `Falling` terminal tone
//! (C3); and period↔exclamation are both falling boundaries distinguished by
//! `strength` (`Full` vs `Exclamatory`, C1/C3). Unpunctuated text yields an
//! empty marker list (C4). The invariant — marker count equals the count of
//! recognized punctuation marks — is asserted alongside each case.
//!
//! Out of scope: no semicolon/colon/dash handling; no acoustic realization —
//! the markers are typed metadata only.
//!
//! RED: `syrinx-frontend` exposes no `punct` module yet, so `hints`,
//! `ProsodyHint`, `Tone`, `Strength`, and `BreakKind` do not exist and this
//! test target fails to build — every criterion is unmet. GREEN adds the
//! module so each assertion holds.

use syrinx_frontend::punct::{hints, BreakKind, ProsodyHint, Strength, Tone};

// ----------------------------------------------------------------------------
// C1 — `hints("Stop. Go")` emits a single falling/full `Boundary` at the period,
// and that marker is distinct from any comma `Break`. The exact-vector equality
// pins both the variant and both its fields; the `!= Break` assertion pins the
// "distinct from any comma marker" clause; the length pins the count invariant
// (one period in -> exactly one marker out).
// ----------------------------------------------------------------------------

#[test]
fn test_period_is_full_falling_boundary() {
    let got = hints("Stop. Go");
    assert_eq!(
        got,
        vec![ProsodyHint::Boundary {
            tone: Tone::Falling,
            strength: Strength::Full,
        }],
        "period must map to a full falling boundary"
    );
}

#[test]
fn test_period_count_invariant() {
    // Exactly one recognized punctuation mark (the period) -> exactly one marker.
    assert_eq!(hints("Stop. Go").len(), 1, "one period yields one marker");
}

#[test]
fn test_period_marker_is_not_a_comma_break() {
    let got = hints("Stop. Go");
    let comma_break = ProsodyHint::Break {
        kind: BreakKind::Short,
    };
    assert_ne!(
        got[0], comma_break,
        "the period marker must be distinct from a comma Break"
    );
}

// ----------------------------------------------------------------------------
// C2 — `hints("Wait, now")` emits a single `Break { kind: Short }` at the comma,
// and the comma↔period distinction is pinned directly: the comma is a `Break`
// (not a `Boundary`) while the period is a `Boundary` (not a `Break`). The
// length pins the count invariant.
// ----------------------------------------------------------------------------

#[test]
fn test_comma_is_short_break() {
    let got = hints("Wait, now");
    assert_eq!(
        got,
        vec![ProsodyHint::Break {
            kind: BreakKind::Short,
        }],
        "comma must map to a short break"
    );
}

#[test]
fn test_comma_count_invariant() {
    assert_eq!(hints("Wait, now").len(), 1, "one comma yields one marker");
}

#[test]
fn test_comma_period_distinction() {
    let comma = &hints("Wait, now")[0];
    let period = &hints("Stop. Go")[0];

    // The comma is a Short break, not a Full boundary...
    assert_eq!(comma, &ProsodyHint::Break { kind: BreakKind::Short });
    // ...and the period is a Full falling boundary, not a Short break.
    assert_eq!(
        period,
        &ProsodyHint::Boundary {
            tone: Tone::Falling,
            strength: Strength::Full,
        }
    );
    // The two markers are not equal: a comma Break != a period Boundary.
    assert_ne!(comma, period, "comma Break must differ from period Boundary");
}

// ----------------------------------------------------------------------------
// C3 — `hints("Really?")` emits a `Boundary { tone: Rising }` for the question
// mark, while `hints("Stop!")` emits a falling/exclamatory boundary. This pins
// rising vs falling terminal tone, and pins the exclamation apart from the
// period (both falling) via `strength` (Full vs Exclamatory).
// ----------------------------------------------------------------------------

#[test]
fn test_question_is_rising_boundary() {
    let got = hints("Really?");
    assert_eq!(
        got,
        vec![ProsodyHint::Boundary {
            tone: Tone::Rising,
            strength: Strength::Full,
        }],
        "question mark must map to a rising boundary"
    );
}

#[test]
fn test_exclamation_is_falling_exclamatory_boundary() {
    let got = hints("Stop!");
    assert_eq!(
        got,
        vec![ProsodyHint::Boundary {
            tone: Tone::Falling,
            strength: Strength::Exclamatory,
        }],
        "exclamation must map to a falling, exclamatory boundary"
    );
}

#[test]
fn test_question_count_invariant() {
    assert_eq!(hints("Really?").len(), 1, "one '?' yields one marker");
}

#[test]
fn test_exclamation_count_invariant() {
    assert_eq!(hints("Stop!").len(), 1, "one '!' yields one marker");
}

#[test]
fn test_question_rising_vs_exclamation_falling() {
    // Pull the terminal tone out of each boundary and assert they are the two
    // distinct tone variants — rising for '?', falling for '!'.
    let q = &hints("Really?")[0];
    let e = &hints("Stop!")[0];

    match q {
        ProsodyHint::Boundary { tone, .. } => assert_eq!(*tone, Tone::Rising),
        other => panic!("expected a Boundary for '?', got {other:?}"),
    }
    match e {
        ProsodyHint::Boundary { tone, .. } => assert_eq!(*tone, Tone::Falling),
        other => panic!("expected a Boundary for '!', got {other:?}"),
    }
    assert_ne!(q, e, "a rising '?' boundary must differ from a falling '!'");
}

#[test]
fn test_exclamation_distinct_from_period() {
    // Both are falling boundaries; the `strength` field keeps them distinct.
    let period = &hints("Stop. Go")[0];
    let exclam = &hints("Stop!")[0];
    assert_ne!(
        period, exclam,
        "exclamation (Exclamatory) must differ from period (Full) despite both falling"
    );
}

// ----------------------------------------------------------------------------
// C4 — `hints("hello world")` emits zero markers: unpunctuated text produces an
// empty marker list. The count invariant degenerates to zero recognized marks.
// ----------------------------------------------------------------------------

#[test]
fn test_no_punctuation_yields_no_markers() {
    let got = hints("hello world");
    assert!(got.is_empty(), "unpunctuated text must yield no markers");
    assert_eq!(got.len(), 0, "zero recognized marks -> zero markers");
}

// ----------------------------------------------------------------------------
// Cross-criterion — a mixed string pins the ordering and the count invariant
// together: the markers come out in source order and number exactly the
// recognized marks (comma, period, question = 3), exercising C1, C2 and C3 in
// one pass.
// ----------------------------------------------------------------------------

#[test]
fn test_mixed_punctuation_ordered_and_counted() {
    let got = hints("Hi, there. Ok?");
    assert_eq!(
        got,
        vec![
            ProsodyHint::Break {
                kind: BreakKind::Short,
            },
            ProsodyHint::Boundary {
                tone: Tone::Falling,
                strength: Strength::Full,
            },
            ProsodyHint::Boundary {
                tone: Tone::Rising,
                strength: Strength::Full,
            },
        ],
        "markers must be emitted in source order, one per recognized mark"
    );
    assert_eq!(got.len(), 3, "three recognized marks -> three markers");
}
