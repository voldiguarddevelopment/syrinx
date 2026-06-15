//! Frozen RED tests for T-01.05 — map custom pronunciations.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `syrinx_frontend::g2p::OverridingPhonemizer` — a decorator over any
//!     [`Phonemizer`] that substitutes per-word IPA from an override map.
//!   * `OverridingPhonemizer::new(base, map)` — construct it from a base
//!     phonemizer and a `word -> IPA` override map (`HashMap<String, String>`).
//!   * It is itself a `Phonemizer`, so `phonemize(&str) -> String` resolves
//!     through the trait.
//!
//! Contract (DESIGN / list.md): the decorator's output equals
//! `map.get(fold(word)).unwrap_or_else(|| base.phonemize(word))`. A hit replaces
//! the base output exactly; a miss delegates to the base untouched; keys are
//! case-folded so an override registered under "Syrinx" matches the query
//! "syrinx"; and an empty override map is a transparent passthrough — the
//! decorator behaves identically to its base for every input. Out of scope: no
//! validation that override values are well-formed IPA; single-word keys only,
//! no multi-word/phrase overrides.
//!
//! The base used throughout is [`DefaultPhonemizer`] (T-01.04), whose known
//! words map exactly ("cat"→"kæt", "the"→"ðə") and whose out-of-vocabulary
//! fallback for "syrinx" is a string of `ə`s — distinct from any real override
//! value, so a hit that returns the mapped IPA is provably a replacement and not
//! a pass-through.
//!
//! RED: `syrinx-frontend` exposes no `OverridingPhonemizer` yet, so the type and
//! its constructor do not resolve and this test target fails to build — every
//! criterion is unmet. GREEN adds the decorator so each assertion below holds.

use std::collections::HashMap;

use syrinx_frontend::g2p::{DefaultPhonemizer, OverridingPhonemizer, Phonemizer};

/// Build a `word -> IPA` override map from `(key, value)` string-slice pairs.
fn overrides(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

// ----------------------------------------------------------------------------
// C1 — an override hit returns the mapped IPA exactly, replacing the base output.
// ----------------------------------------------------------------------------

/// With `{"syrinx" → "ˈsɪrɪŋks"}`, `phonemize("syrinx")` returns exactly the
/// mapped IPA (criterion C1).
#[test]
fn override_hit_returns_mapped_ipa_exactly() {
    let map = overrides(&[("syrinx", "ˈsɪrɪŋks")]);
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), map);
    assert_eq!(p.phonemize("syrinx"), "ˈsɪrɪŋks");
}

/// The override genuinely REPLACES the base G2P output: the decorator's result
/// for "syrinx" differs from what the bare base produces for the same word
/// (criterion C1). Kills an impl that ignores the map and always delegates.
#[test]
fn override_hit_replaces_base_output() {
    let base_out = DefaultPhonemizer::new().phonemize("syrinx");
    let map = overrides(&[("syrinx", "ˈsɪrɪŋks")]);
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), map);
    let decorated = p.phonemize("syrinx");

    assert_eq!(decorated, "ˈsɪrɪŋks");
    assert_ne!(
        decorated, base_out,
        "override hit must not equal the base output it replaces",
    );
}

// ----------------------------------------------------------------------------
// C2 — a miss delegates to the base, unchanged (override consulted only on a hit).
// ----------------------------------------------------------------------------

/// A word NOT in the map ("cat") returns the base phonemizer's output "kæt"
/// unchanged, even when the map holds an unrelated override (criterion C2).
#[test]
fn override_miss_delegates_to_base() {
    let map = overrides(&[("syrinx", "ˈsɪrɪŋks")]);
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), map);
    assert_eq!(p.phonemize("cat"), "kæt");
}

/// On a miss the decorator's output equals exactly what the bare base produces
/// for the same word — proving delegation is untouched (criterion C2). Kills an
/// impl that always returns a fixed override value regardless of the key.
#[test]
fn override_miss_matches_bare_base() {
    let map = overrides(&[("syrinx", "ˈsɪrɪŋks")]);
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), map);
    let base_out = DefaultPhonemizer::new().phonemize("cat");

    assert_eq!(p.phonemize("cat"), base_out);
    assert_ne!(
        p.phonemize("cat"),
        "ˈsɪrɪŋks",
        "a miss must not return the override value of a different key",
    );
}

// ----------------------------------------------------------------------------
// C3 — matching is case-folded: an override under "Syrinx" applies to "syrinx".
// ----------------------------------------------------------------------------

/// An override registered under the capitalized key "Syrinx" still applies to the
/// lowercase query "syrinx", returning the mapped IPA exactly (criterion C3).
#[test]
fn override_key_is_case_folded() {
    let map = overrides(&[("Syrinx", "ˈsɪrɪŋks")]);
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), map);
    assert_eq!(p.phonemize("syrinx"), "ˈsɪrɪŋks");
}

/// Folding applies to the query as well as the key: a lowercase key "syrinx"
/// matches the capitalized query "Syrinx" (criterion C3). Together with the
/// previous test this pins case-folding on both sides of the lookup.
#[test]
fn override_query_is_case_folded() {
    let map = overrides(&[("syrinx", "ˈsɪrɪŋks")]);
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), map);
    assert_eq!(p.phonemize("Syrinx"), "ˈsɪrɪŋks");
}

// ----------------------------------------------------------------------------
// C4 — an empty map is a transparent passthrough: identical to the base.
// ----------------------------------------------------------------------------

/// With an empty override map, `phonemize("the")` returns the base output "ðə"
/// (criterion C4).
#[test]
fn empty_map_passes_known_word_through() {
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), HashMap::new());
    assert_eq!(p.phonemize("the"), "ðə");
}

/// With an empty override map the decorator is indistinguishable from its base
/// for every kind of input — a known word, a second known word, and an
/// out-of-vocabulary word all match the bare base exactly (criterion C4).
#[test]
fn empty_map_is_transparent_for_every_input() {
    let base = DefaultPhonemizer::new();
    let p = OverridingPhonemizer::new(DefaultPhonemizer::new(), HashMap::new());

    for word in ["the", "cat", "zorptquax"] {
        assert_eq!(
            p.phonemize(word),
            base.phonemize(word),
            "empty-map decorator diverged from base on {word:?}",
        );
    }
}
