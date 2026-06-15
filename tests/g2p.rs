//! Frozen RED tests for T-01.04 — phonemize a single word to IPA.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `syrinx_frontend::g2p::Phonemizer` — a trait exposing
//!     `fn phonemize(&self, word: &str) -> String`.
//!   * `syrinx_frontend::g2p::DefaultPhonemizer` — the default backend, a concrete
//!     `Phonemizer` constructed with `DefaultPhonemizer::new()`.
//!
//! Contract (DESIGN / list.md): `phonemize` is a *total* function `&str -> String`.
//! Known words hit a fixed labeled table exactly ("cat"→"kæt", "the"→"ðə", plus
//! the rest of the golden set under `tests/golden/g2p/`). An out-of-vocabulary
//! word takes a deterministic fallback that ALWAYS yields a non-empty IPA string
//! whose every character is a defined IPA symbol — it never panics. The empty
//! string maps to the empty string, pinning the empty-input boundary apart from
//! the OOV path. Out of scope: no stress/syllable marks, no per-word overrides
//! (T-01.05), no heteronym disambiguation (T-01.06).
//!
//! RED: `syrinx-frontend` exposes no `g2p` module yet, so `Phonemizer` /
//! `DefaultPhonemizer` do not resolve and this test target fails to build — every
//! criterion is unmet. GREEN adds the module so each assertion below holds.

use std::path::PathBuf;

use syrinx_frontend::g2p::{DefaultPhonemizer, Phonemizer};

/// The closed set of IPA symbols a phonemization may draw from. The fallback for
/// out-of-vocabulary words must emit ONLY characters from this set (criterion C3).
/// A generous English-IPA inventory: consonants, vowels, and the length mark.
const IPA_SYMBOLS: &str =
    "pbtdkɡgfvθðszʃʒhmnŋlrɹwjiɪeɛæəʌɑɒɔoʊuyɜɝɚː";

/// Whether `c` is a member of the defined IPA symbol set.
fn is_ipa_symbol(c: char) -> bool {
    IPA_SYMBOLS.contains(c)
}

/// The repo-root directory holding the golden `(.in, .expected)` word→IPA pairs.
fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("g2p")
}

/// Collect every `<word>.in` file under the golden dir, sorted for determinism.
fn input_files() -> Vec<PathBuf> {
    let dir = golden_dir();
    let mut ins: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read golden dir {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("in"))
        .collect();
    ins.sort();
    ins
}

/// Read a golden file's UTF-8 contents, trimming surrounding whitespace so a
/// trailing newline in the fixture never leaks into the compared word/IPA.
fn read_trimmed(path: &PathBuf) -> String {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    String::from_utf8(bytes)
        .unwrap_or_else(|e| panic!("{} is not UTF-8: {e}", path.display()))
        .trim()
        .to_string()
}

// ----------------------------------------------------------------------------
// C1 — the trait method exists and the known word "cat" maps exactly to "kæt".
// ----------------------------------------------------------------------------

/// `DefaultPhonemizer::new().phonemize("cat")` returns exactly "kæt" through the
/// `Phonemizer` trait method (criterion C1).
#[test]
fn cat_maps_to_known_ipa() {
    let p = DefaultPhonemizer::new();
    assert_eq!(p.phonemize("cat"), "kæt");
}

// ----------------------------------------------------------------------------
// C2 — a second known word "the"→"ðə", and the full golden set round-trips.
// ----------------------------------------------------------------------------

/// The second labeled word "the" maps exactly to "ðə" (criterion C2).
#[test]
fn the_maps_to_known_ipa() {
    let p = DefaultPhonemizer::new();
    assert_eq!(p.phonemize("the"), "ðə");
}

/// The labeled golden corpus must be non-empty — an empty directory must NOT
/// vacuously pass the round-trip suite (criterion C2 requires a real fixed set).
#[test]
fn golden_set_is_non_empty() {
    let count = input_files().len();
    assert!(count >= 2, "expected at least two golden word cases, found {count}");
}

/// Every golden case: `phonemize(<word>.in)` equals `<word>.expected` exactly.
/// The full fixed labeled set under `tests/golden/g2p/` round-trips word→IPA with
/// every entry matching (criterion C2).
#[test]
fn golden_labeled_set_round_trips() {
    let p = DefaultPhonemizer::new();
    let inputs = input_files();
    assert!(!inputs.is_empty(), "no golden word cases discovered");

    for in_path in inputs {
        let expected_path = in_path.with_extension("expected");
        let word = read_trimmed(&in_path);
        let expected = read_trimmed(&expected_path);

        assert_eq!(
            p.phonemize(&word),
            expected,
            "golden word {:?} did not phonemize to {:?}",
            word,
            expected,
        );
    }
}

// ----------------------------------------------------------------------------
// C3 — an OOV word yields a non-empty IPA string of only defined symbols, no panic.
// ----------------------------------------------------------------------------

/// The out-of-vocabulary word "zorptquax" returns a NON-EMPTY string via the
/// fallback path — OOV is always covered, never an empty result (criterion C3).
#[test]
fn oov_word_is_non_empty() {
    let p = DefaultPhonemizer::new();
    let out = p.phonemize("zorptquax");
    assert!(!out.is_empty(), "OOV fallback produced an empty string");
}

/// Every character of the OOV fallback output is a member of the defined IPA
/// symbol set (criterion C3). Guards against a fallback that leaks raw graphemes
/// or out-of-inventory symbols. Calling it at all also pins "does not panic".
#[test]
fn oov_word_chars_all_in_ipa_set() {
    let p = DefaultPhonemizer::new();
    let out = p.phonemize("zorptquax");
    assert!(!out.is_empty(), "OOV fallback produced an empty string");
    for c in out.chars() {
        assert!(
            is_ipa_symbol(c),
            "OOV output char {c:?} is not in the defined IPA symbol set",
        );
    }
}

// ----------------------------------------------------------------------------
// C4 — the empty input maps to the empty string, pinned apart from the OOV path.
// ----------------------------------------------------------------------------

/// `phonemize("")` returns the empty string and does not panic, pinning the
/// empty-input boundary against the always-non-empty OOV fallback (criterion C4).
#[test]
fn empty_input_maps_to_empty() {
    let p = DefaultPhonemizer::new();
    assert_eq!(p.phonemize(""), "");
}
