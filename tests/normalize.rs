//! Frozen RED tests for T-01.01 — the deterministic text normalizer.
//!
//! These pin criteria C1 (NFC composition), C2 (whitespace collapse + trim),
//! C4 (casing preserved), and the idempotence invariant against the real public
//! API the green phase must build:
//!
//!   * `syrinx_frontend::normalize::normalize(&str) -> String`
//!
//! Contract (DESIGN / list.md): take arbitrary `&str`, return a `String` in
//! Unicode NFC with runs of whitespace collapsed to single ASCII U+0020 spaces
//! and the ends trimmed, with casing left untouched. Empty input returns empty;
//! lone combining marks and mixed CR/LF/tab normalize without panic.
//!
//! RED: `syrinx-frontend` exposes no `normalize` module yet, so the symbol does
//! not resolve and the test target fails to build — every criterion is unmet.
//! GREEN implements `normalize` so each assertion below holds.
//!
//! Note on the C1 example: list.md illustrates C1 with `normalize("Café\u{0301}")`
//! → "café". Its falsifiable claims are the precomposed U+00E9, an output byte
//! length of 5, and `is_nfc`, all with casing untouched (C4). A self-consistent
//! input exercising those claims is `"cafe\u{0301}"` (base 'e' + COMBINING ACUTE)
//! → the documented output "café" (= "caf\u{e9}"), kept all-lowercase so casing
//! is provably preserved per C4. We assert that exact output below.

use syrinx_frontend::normalize::normalize;
use unicode_normalization::UnicodeNormalization;

// ----------------------------------------------------------------------------
// C1 — Unicode NFC composition.
// ----------------------------------------------------------------------------

/// Base 'e' + U+0301 COMBINING ACUTE ACCENT composes to the precomposed U+00E9,
/// shrinking the 6-byte decomposed form to the 5-byte precomposed form, and the
/// result is in NFC (criterion C1).
#[test]
fn nfc_composes_combining_acute_to_precomposed() {
    let input = "cafe\u{0301}"; // 'c','a','f','e', U+0301 — 6 bytes
    assert_eq!(input.len(), 6, "decomposed input is 6 bytes");

    let out = normalize(input);

    // Exactly the documented output "café", precomposed.
    assert_eq!(out, "caf\u{e9}");
    // The precomposed code point U+00E9 is present...
    assert!(out.chars().any(|c| c == '\u{e9}'), "precomposed é present");
    // ...and the lone combining mark is gone.
    assert!(!out.contains('\u{0301}'), "no leftover combining acute");
    // 5 bytes, not 6.
    assert_eq!(out.len(), 5);
    // `is_nfc` holds on the output.
    assert!(unicode_normalization::is_nfc(&out), "output is NFC");
}

/// A lone leading combining mark normalizes without panic and yields NFC output
/// (edge: "lone combining marks ... normalize without panic").
#[test]
fn nfc_handles_lone_combining_mark_without_panic() {
    let out = normalize("\u{0301}");
    assert!(unicode_normalization::is_nfc(&out), "output is NFC");
    // Idempotent on this edge input.
    assert_eq!(normalize(&out), out);
}

// ----------------------------------------------------------------------------
// C2 — whitespace collapse + trim.
// ----------------------------------------------------------------------------

/// Leading/trailing whitespace is trimmed and every interior run of whitespace
/// (spaces, tab, CR, LF) collapses to a single U+0020 (criterion C2).
#[test]
fn collapses_whitespace_runs_and_trims_ends() {
    let out = normalize("  Hello\tWorld \r\n");
    assert_eq!(out, "Hello World");
}

/// The collapse target is exactly one ASCII space — no tab, CR, or LF survives,
/// no run remains doubled, and neither end carries whitespace (criterion C2,
/// pinning both sides of "single space" vs "more than one").
#[test]
fn collapsed_output_has_no_other_whitespace_or_runs() {
    let out = normalize("  Hello\tWorld \r\n");
    assert!(!out.contains('\t'), "no tab");
    assert!(!out.contains('\r'), "no CR");
    assert!(!out.contains('\n'), "no LF");
    assert!(!out.contains("  "), "no doubled space");
    assert!(!out.starts_with(' '), "no leading space");
    assert!(!out.ends_with(' '), "no trailing space");
    // The single interior separator is one U+0020.
    assert_eq!(out.matches(' ').count(), 1);
}

/// A single interior space between two words is preserved as-is (boundary: a run
/// of length one stays length one, it is not dropped).
#[test]
fn single_interior_space_is_preserved() {
    assert_eq!(normalize("Hello World"), "Hello World");
}

/// Input that is only whitespace trims to the empty string (boundary: nothing
/// survives, no stray separator is emitted).
#[test]
fn all_whitespace_trims_to_empty() {
    assert_eq!(normalize(" \t\r\n "), "");
}

// ----------------------------------------------------------------------------
// C4 — casing preserved (no lowercasing).
// ----------------------------------------------------------------------------

/// Intra-word casing is preserved; `normalize` does not lowercase (criterion C4).
#[test]
fn preserves_intra_word_casing() {
    assert_eq!(normalize("iPhone XR"), "iPhone XR");
}

/// Casing is preserved across the whitespace-collapse path too: an uppercase
/// letter stays uppercase rather than being folded (criterion C4, distinct from
/// the would-be-lowercased form).
#[test]
fn does_not_lowercase_mixed_case() {
    let out = normalize("  HELLO\tWorld ");
    assert_eq!(out, "HELLO World");
    assert_ne!(out, "hello world", "must not lowercase");
}

// ----------------------------------------------------------------------------
// Edges + idempotence invariant.
// ----------------------------------------------------------------------------

/// Empty input returns the empty string (edge from the contract).
#[test]
fn empty_input_returns_empty() {
    assert_eq!(normalize(""), "");
}

/// `normalize` is idempotent: `normalize(normalize(x)) == normalize(x)` for a
/// spread of inputs covering accents, whitespace, casing, and combining marks.
#[test]
fn normalize_is_idempotent() {
    let inputs = [
        "",
        "cafe\u{0301}",
        "  Hello\tWorld \r\n",
        "iPhone XR",
        "\u{0301}",
        "A\u{0300}ngstro\u{0308}m",
        "a  b\t\tc\n\nd",
    ];
    for x in inputs {
        let once = normalize(x);
        let twice = normalize(&once);
        assert_eq!(twice, once, "normalize must be idempotent for {x:?}");
        // A normalized string is already NFC.
        assert!(unicode_normalization::is_nfc(&once), "{x:?} -> NFC");
    }
}

/// A clean string already in NFC with no redundant whitespace is returned
/// unchanged — equal to its own NFC form (sanity that normalize does not mangle
/// already-normal text).
#[test]
fn already_normal_text_is_unchanged() {
    let s = "The quick brown fox";
    let nfc: String = s.nfc().collect();
    assert_eq!(normalize(s), nfc);
    assert_eq!(normalize(s), s);
}
