//! Deterministic text normalization — the frontend's entry point (T-01.01).
//!
//! [`normalize`] takes arbitrary user text and returns a `String` in Unicode NFC
//! with runs of whitespace collapsed to single ASCII spaces and the ends trimmed,
//! leaving casing untouched. Casing folding, number/date expansion, and
//! transliteration are deliberately out of scope here (separate tasks).

use unicode_normalization::UnicodeNormalization;

/// Normalize `input` to Unicode NFC with collapsed, trimmed whitespace.
///
/// The string is first composed to NFC, then split on whitespace runs and
/// rejoined with a single ASCII U+0020 between tokens — which trims both ends and
/// drops every interior run to one space. Casing is preserved. The result is
/// already NFC, so `normalize` is idempotent.
pub fn normalize(input: &str) -> String {
    let composed: String = input.nfc().collect();
    composed.split_whitespace().collect::<Vec<&str>>().join(" ")
}
