//! Rule-based heteronym resolution (T-01.06).
//!
//! [`resolve`] splits a sentence into whitespace-separated words and returns one
//! IPA string per word, in source order. Each non-heteronym word takes the base
//! phonemization from the T-01.04 [`DefaultPhonemizer`]; each word in the fixed
//! heteronym set {read, lead, bow} is resolved to the candidate pronunciation its
//! surrounding context selects.
//!
//! Disambiguation is rule-based context only — no statistical/ML POS tagging — and
//! covers exactly the fixed test set. Both readings of each heteronym are pinned:
//!
//! - read — past "rɛd" when the sentence carries a past-time adverb ("yesterday");
//!   present "riːd" otherwise.
//! - lead — noun "lɛd" when immediately preceded by the article "a"; verb "liːd"
//!   otherwise.
//! - bow — noun "boʊ" when immediately preceded by its noun modifier "violin";
//!   verb "baʊ" otherwise.
//!
//! A sentence with no heteronym passes through unchanged. Resolution is a pure
//! deterministic function of the sentence: identical input yields identical output
//! every call.
//!
//! Out of scope: no statistical/ML POS tagging; no coverage beyond the fixed
//! heteronym test set (read/lead/bow and the listed words).

use crate::g2p::{DefaultPhonemizer, Phonemizer};

/// Resolve `sentence` to a per-word IPA sequence, one entry per
/// whitespace-separated word in source order, with each heteronym disambiguated
/// by its surrounding context.
pub fn resolve(sentence: &str) -> Vec<String> {
    let phonemizer = DefaultPhonemizer::new();
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut out = Vec::with_capacity(words.len());
    let mut prev = "";
    for &word in &words {
        out.push(resolve_word(word, prev, &words, &phonemizer));
        prev = word;
    }
    out
}

/// Resolve a single `word` given the word immediately before it (`prev`), the full
/// sentence (`words`, for sentence-scope cues), and the base `phonemizer` for the
/// non-heteronym passthrough.
fn resolve_word(word: &str, prev: &str, words: &[&str], phonemizer: &DefaultPhonemizer) -> String {
    match word {
        "read" => {
            if words.contains(&"yesterday") {
                "rɛd".to_string()
            } else {
                "riːd".to_string()
            }
        }
        "lead" => {
            if prev == "a" {
                "lɛd".to_string()
            } else {
                "liːd".to_string()
            }
        }
        "bow" => {
            if prev == "violin" {
                "boʊ".to_string()
            } else {
                "baʊ".to_string()
            }
        }
        _ => phonemizer.phonemize(word),
    }
}
