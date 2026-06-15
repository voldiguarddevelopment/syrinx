//! Frozen RED tests for T-01.06 — resolve heteronyms.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `syrinx_frontend::hetero::resolve(sentence: &str) -> Vec<String>` — a pure
//!     function that splits `sentence` into whitespace-separated words and returns
//!     one IPA string per word, in source order. Each non-heteronym word takes the
//!     base phonemization (the T-01.04 `DefaultPhonemizer`); each heteronym is
//!     resolved to the candidate pronunciation its surrounding rule/POS context
//!     selects.
//!
//! Contract (DESIGN / spec.md): rule-based context disambiguation only — no
//! statistical/ML POS tagging — over the fixed heteronym set {read, lead, bow}.
//! Both readings of each are pinned on both sides of their disambiguating context:
//!
//!   * read  — past "rɛd"  (e.g. "I read the book yesterday")
//!             present "riːd" (e.g. "I read books daily")
//!   * lead  — verb "liːd" (e.g. "lead the way")
//!             noun "lɛd"  (e.g. "a lead pipe")
//!   * bow   — verb "baʊ"  (e.g. "take a bow")
//!             noun "boʊ"  (e.g. "a violin bow")
//!
//! A sentence with no heteronym passes through unchanged: every word keeps its
//! base phonemization, so "the cat sat" leaves "cat" as "kæt" (and "the" as "ðə").
//! Resolution is a pure deterministic function of the sentence — identical input
//! yields identical output every call.
//!
//! Out of scope: no statistical/ML POS tagging; no coverage beyond the fixed
//! heteronym test set (read/lead/bow and the listed words).
//!
//! RED: `syrinx-frontend` exposes no `hetero` module yet, so `resolve` does not
//! resolve and this test target fails to build — every criterion is unmet. GREEN
//! adds the module so each assertion below holds.

use syrinx_frontend::hetero::resolve;

/// Resolve `sentence` and return the IPA of the word at `idx`, asserting the
/// per-word sequence is long enough first so a short result fails loudly rather
/// than panicking on an out-of-range index.
fn word_at(sentence: &str, idx: usize) -> String {
    let seq = resolve(sentence);
    assert!(
        idx < seq.len(),
        "resolve({sentence:?}) returned {} words; need index {idx}: {seq:?}",
        seq.len(),
    );
    seq[idx].clone()
}

// ----------------------------------------------------------------------------
// C1 — read: past-tense "rɛd" vs present-tense "riːd", disambiguated by context.
// Both sides of the read boundary are pinned.
// ----------------------------------------------------------------------------

/// Past-tense context ("...yesterday") selects "rɛd" for "read" (criterion C1).
/// "read" is the second word (index 1) of "I read the book yesterday".
#[test]
fn read_past_tense_selects_red() {
    assert_eq!(word_at("I read the book yesterday", 1), "rɛd");
}

/// Present-tense context ("...daily") selects "riːd" for "read" (criterion C1).
/// "read" is the second word (index 1) of "I read books daily".
#[test]
fn read_present_tense_selects_reed() {
    assert_eq!(word_at("I read books daily", 1), "riːd");
}

/// The two read sentences must resolve "read" DIFFERENTLY — pinning that the
/// disambiguation actually flips, not that both happen to land on one reading
/// (criterion C1).
#[test]
fn read_two_contexts_differ() {
    let past = word_at("I read the book yesterday", 1);
    let present = word_at("I read books daily", 1);
    assert_ne!(past, present, "read did not disambiguate between the two tenses");
}

// ----------------------------------------------------------------------------
// C2 — lead: verb "liːd" vs noun "lɛd". Both sides of the lead boundary pinned.
// ----------------------------------------------------------------------------

/// Verb context ("lead the way") selects "liːd" for "lead" (criterion C2).
/// "lead" is the first word (index 0).
#[test]
fn lead_verb_selects_liid() {
    assert_eq!(word_at("lead the way", 0), "liːd");
}

/// Noun context ("a lead pipe", preceded by an article) selects "lɛd" for "lead"
/// (criterion C2). "lead" is the second word (index 1).
#[test]
fn lead_noun_selects_led() {
    assert_eq!(word_at("a lead pipe", 1), "lɛd");
}

/// The two lead sentences must resolve "lead" DIFFERENTLY, pinning that the
/// verb/noun split actually flips the choice (criterion C2).
#[test]
fn lead_two_contexts_differ() {
    let verb = word_at("lead the way", 0);
    let noun = word_at("a lead pipe", 1);
    assert_ne!(verb, noun, "lead did not disambiguate between verb and noun");
}

// ----------------------------------------------------------------------------
// C3 — bow: "baʊ" vs "boʊ" on the fixed set, deterministically.
// ----------------------------------------------------------------------------

/// "take a bow" selects "baʊ" for "bow" (criterion C3). "bow" is the third word
/// (index 2).
#[test]
fn bow_take_selects_bau() {
    assert_eq!(word_at("take a bow", 2), "baʊ");
}

/// "a violin bow" selects "boʊ" for "bow" (criterion C3). "bow" is the third word
/// (index 2).
#[test]
fn bow_violin_selects_bou() {
    assert_eq!(word_at("a violin bow", 2), "boʊ");
}

/// The two bow sentences must resolve "bow" DIFFERENTLY, pinning the flip
/// (criterion C3).
#[test]
fn bow_two_contexts_differ() {
    let a = word_at("take a bow", 2);
    let b = word_at("a violin bow", 2);
    assert_ne!(a, b, "bow did not disambiguate between its two readings");
}

/// Resolution is a pure deterministic function of the sentence: calling `resolve`
/// twice on the same input yields the identical per-word sequence (criterion C3).
#[test]
fn resolution_is_deterministic() {
    for sentence in [
        "take a bow",
        "a violin bow",
        "I read the book yesterday",
        "lead the way",
        "the cat sat",
    ] {
        assert_eq!(
            resolve(sentence),
            resolve(sentence),
            "resolve({sentence:?}) was not deterministic across calls",
        );
    }
}

// ----------------------------------------------------------------------------
// C4 — a sentence with no heteronym passes through to the base phonemization.
// ----------------------------------------------------------------------------

/// "the cat sat" has no heteronym, so "cat" keeps its base phonemization "kæt"
/// — no substitution is applied (criterion C4). "cat" is the second word (index 1).
#[test]
fn no_heteronym_leaves_cat_as_base() {
    assert_eq!(word_at("the cat sat", 1), "kæt");
}

/// The non-heteronym passthrough resolves the WHOLE sentence to the base
/// phonemization in order: "the" → "ðə", "cat" → "kæt", "sat" → its base form.
/// This pins that passthrough touches no word, and that the per-word sequence has
/// exactly one entry per input word (criterion C4).
#[test]
fn no_heteronym_passthrough_full_sequence() {
    let seq = resolve("the cat sat");
    assert_eq!(seq.len(), 3, "expected one IPA per word, got {seq:?}");
    assert_eq!(seq[0], "ðə", "base phonemization of \"the\" changed");
    assert_eq!(seq[1], "kæt", "base phonemization of \"cat\" changed");
}
