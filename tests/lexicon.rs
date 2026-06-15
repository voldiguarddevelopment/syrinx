//! Frozen RED tests for T-01.03 — override pronunciations via a two-tier lexicon.
//!
//! These pin criteria C1 (user precedence over default on a key collision),
//! C2 (default-only fallthrough and the total miss), C3 (case-folded key with the
//! stored value's casing returned verbatim), and C4 (an empty user lexicon leaves
//! every default entry reachable) against the real public API the green phase
//! must build:
//!
//!   * `syrinx_frontend::lexicon::Lexicon`
//!   * `Lexicon::with_user(HashMap<String, String>) -> Lexicon`
//!   * `Lexicon::lookup(&self, word: &str) -> Option<String>`
//!
//! Contract (DESIGN / list.md): a two-tier override table consulted before
//! phonemization. A fixed built-in *default* lexicon maps "tomato"→"tom-ah-to"
//! and "data"→"day-ta"; `with_user` layers an optional user map on top. `lookup`
//! folds the query key to lowercase and returns the winning replacement as an
//! owned `String`, or `None` when neither tier holds the key. Precedence is total
//! and deterministic — user ∪ default with the user winning every tie. Case
//! folding applies to the *key* only; the stored replacement value is returned
//! byte-for-byte, never re-cased. Out of scope: no fuzzy/stemmed matching, no IPA
//! validation of the replacement string.
//!
//! RED: `syrinx-frontend` exposes no `lexicon` module yet, so `Lexicon` does not
//! resolve and the test target fails to build — every criterion is unmet. GREEN
//! adds the module so each assertion below holds.

use std::collections::HashMap;

use syrinx_frontend::lexicon::Lexicon;

/// Build a user lexicon map from `(key, value)` literal pairs.
fn user_map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// An empty user lexicon — exercises the "default only" tier.
fn empty_user() -> HashMap<String, String> {
    HashMap::new()
}

// ----------------------------------------------------------------------------
// C1 — user precedence over the default on a key collision, pinned on both sides.
// ----------------------------------------------------------------------------

/// With the built-in default mapping "tomato"→"tom-ah-to" and a user lexicon
/// mapping "tomato"→"tom-ay-to", the user value wins (criterion C1).
#[test]
fn user_overrides_default_on_collision() {
    let lex = Lexicon::with_user(user_map(&[("tomato", "tom-ay-to")]));
    assert_eq!(lex.lookup("tomato").as_deref(), Some("tom-ay-to"));
}

/// The other side of the precedence boundary: with NO user override the same key
/// resolves to the *default* value "tom-ah-to". Together with the test above this
/// pins that the user tier — not the default — supplies the winning value when
/// both hold the key (criterion C1).
#[test]
fn default_value_wins_when_no_user_override() {
    let lex = Lexicon::with_user(empty_user());
    assert_eq!(lex.lookup("tomato").as_deref(), Some("tom-ah-to"));
    // And it is genuinely a *different* string from the user override, so a tie is
    // a real tie that precedence must break.
    assert_ne!(lex.lookup("tomato").as_deref(), Some("tom-ay-to"));
}

// ----------------------------------------------------------------------------
// C2 — default-only fallthrough, and the total miss returning None.
// ----------------------------------------------------------------------------

/// A key present ONLY in the default lexicon resolves through to its default
/// value even when a non-empty user lexicon exists that does not hold it
/// (criterion C2, the fallthrough side).
#[test]
fn default_only_key_falls_through() {
    let lex = Lexicon::with_user(user_map(&[("tomato", "tom-ay-to")]));
    assert_eq!(lex.lookup("data").as_deref(), Some("day-ta"));
}

/// A key in NEITHER tier returns `None` — the miss (criterion C2, the miss side).
#[test]
fn missing_key_returns_none() {
    let lex = Lexicon::with_user(user_map(&[("tomato", "tom-ay-to")]));
    assert_eq!(lex.lookup("zzqx"), None);
}

/// Even with an empty user lexicon, an unknown key misses both tiers and is
/// `None` — distinguishing "absent" from "default present" (criterion C2).
#[test]
fn missing_key_with_empty_user_returns_none() {
    let lex = Lexicon::with_user(empty_user());
    assert_eq!(lex.lookup("zzqx"), None);
}

// ----------------------------------------------------------------------------
// C3 — case-folded key lookup; stored value casing returned unaltered.
// ----------------------------------------------------------------------------

/// `lookup("Tomato")` and `lookup("tomato")` resolve to the SAME `Some` entry —
/// the key is folded to lowercase before matching (criterion C3, the key side).
#[test]
fn lookup_is_case_folded_on_key() {
    let lex = Lexicon::with_user(user_map(&[("tomato", "tom-ay-to")]));
    let mixed = lex.lookup("Tomato");
    let lower = lex.lookup("tomato");
    assert_eq!(mixed.as_deref(), Some("tom-ay-to"));
    assert_eq!(mixed, lower, "mixed- and lower-case keys resolve identically");
}

/// Case folding applies to the default tier too: `lookup("DATA")` reaches the
/// default "data"→"day-ta" entry (criterion C3, key side over the default tier).
#[test]
fn default_key_lookup_is_case_folded() {
    let lex = Lexicon::with_user(empty_user());
    assert_eq!(lex.lookup("DATA").as_deref(), Some("day-ta"));
}

/// The stored replacement value's casing is returned UNALTERED — a value with
/// upper-case letters comes back byte-for-byte even though the key was folded
/// (criterion C3, the value side). Guards against an implementation that lowers
/// the value alongside the key.
#[test]
fn stored_value_casing_is_returned_unaltered() {
    let lex = Lexicon::with_user(user_map(&[("jalapeno", "hah-lah-PEN-yo")]));
    assert_eq!(lex.lookup("JALAPENO").as_deref(), Some("hah-lah-PEN-yo"));
    // Explicitly not the lower-cased form.
    assert_ne!(lex.lookup("JALAPENO").as_deref(), Some("hah-lah-pen-yo"));
}

// ----------------------------------------------------------------------------
// C4 — an empty user lexicon leaves all default entries reachable.
// ----------------------------------------------------------------------------

/// `Lexicon::with_user(empty).lookup("data")` still returns the default
/// "day-ta" — an empty user tier shadows nothing (criterion C4).
#[test]
fn empty_user_keeps_defaults_reachable() {
    let lex = Lexicon::with_user(empty_user());
    assert_eq!(lex.lookup("data").as_deref(), Some("day-ta"));
}

/// The empty user tier also leaves the *other* default entry reachable, pinning
/// that an empty user map disables the override tier entirely rather than the
/// table as a whole (criterion C4).
#[test]
fn empty_user_keeps_all_defaults_reachable() {
    let lex = Lexicon::with_user(empty_user());
    assert_eq!(lex.lookup("tomato").as_deref(), Some("tom-ah-to"));
}
