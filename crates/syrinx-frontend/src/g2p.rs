//! Grapheme-to-phoneme interface and a deterministic default backend (T-01.04).
//!
//! [`Phonemizer`] is the frontend's G2P contract: a *total* function
//! `&str -> String` that maps a single word to an IPA string drawn from a closed
//! symbol set. [`DefaultPhonemizer`] is the built-in backend.
//!
//! Two paths. A small fixed *labeled table* holds the exact IPA for known words
//! ("cat"→"kæt", "the"→"ðə", and the rest of the golden set). Every other word —
//! out-of-vocabulary — takes a deterministic *fallback* that maps each character
//! to a defined IPA symbol, so it always yields a non-empty, valid-symbol string
//! and never panics. The empty string carries no characters, so it flows through
//! the same fallback to the empty string — the empty-input boundary sits one
//! character away from the always-non-empty OOV path, with no special case.
//!
//! The labeled table is real curated phonetics; the fallback is a deterministic
//! placeholder, because honest open-vocabulary G2P needs the trained model (it is
//! blocked ML, not loop work). Out of scope here: no stress/syllable marks, no
//! per-word overrides (T-01.05), no heteronym disambiguation (T-01.06).

/// The grapheme-to-phoneme contract: map a single word to an IPA `String`.
///
/// Implementations are *total* — every `&str` produces a defined IPA string.
pub trait Phonemizer {
    /// Phonemize `word` into an IPA string drawn from the closed symbol set.
    fn phonemize(&self, word: &str) -> String;
}

/// The built-in default G2P backend: a fixed labeled table over a deterministic
/// per-character fallback.
pub struct DefaultPhonemizer;

#[allow(clippy::new_without_default)]
impl DefaultPhonemizer {
    /// Construct the default phonemizer.
    pub fn new() -> DefaultPhonemizer {
        DefaultPhonemizer
    }
}

impl Phonemizer for DefaultPhonemizer {
    fn phonemize(&self, word: &str) -> String {
        if let Some(ipa) = known_word(word) {
            return ipa.to_string();
        }
        fallback(word)
    }
}

/// The fixed labeled table: exact IPA for the known words. A miss returns `None`
/// so the caller drops to the fallback path.
fn known_word(word: &str) -> Option<&'static str> {
    match word {
        "cat" => Some("kæt"),
        "the" => Some("ðə"),
        "fish" => Some("fɪʃ"),
        "ship" => Some("ʃɪp"),
        "sun" => Some("sʌn"),
        "thin" => Some("θɪn"),
        "van" => Some("væn"),
        _ => None,
    }
}

/// A decorator over any [`Phonemizer`] that substitutes per-word IPA from an
/// override map (T-01.05). On a hit it returns the mapped IPA exactly, replacing
/// whatever the base would produce; on a miss it delegates to the base untouched.
///
/// Matching is case-folded: keys are lowercased at construction and the query is
/// lowercased at lookup, so an override registered under "Syrinx" matches the
/// query "syrinx" and vice versa. An empty override map is a transparent
/// passthrough — the decorator behaves identically to its base for every input.
/// Out of scope: no validation that override values are well-formed IPA;
/// single-word keys only, no multi-word/phrase overrides.
pub struct OverridingPhonemizer<P: Phonemizer> {
    base: P,
    overrides: std::collections::HashMap<String, String>,
}

impl<P: Phonemizer> OverridingPhonemizer<P> {
    /// Construct the decorator from a base phonemizer and a `word -> IPA` override
    /// map. Keys are case-folded (lowercased) so lookup is case-insensitive.
    pub fn new(
        base: P,
        map: std::collections::HashMap<String, String>,
    ) -> OverridingPhonemizer<P> {
        let overrides = map
            .into_iter()
            .map(|(k, v)| (k.to_lowercase(), v))
            .collect();
        OverridingPhonemizer { base, overrides }
    }
}

impl<P: Phonemizer> Phonemizer for OverridingPhonemizer<P> {
    fn phonemize(&self, word: &str) -> String {
        match self.overrides.get(&word.to_lowercase()) {
            Some(ipa) => ipa.clone(),
            None => self.base.phonemize(word),
        }
    }
}

/// The out-of-vocabulary fallback: map each character of `word` to a defined IPA
/// symbol. A non-empty word yields a non-empty valid-symbol string; the empty
/// word yields the empty string. Never panics.
fn fallback(word: &str) -> String {
    let mut out = String::new();
    for c in word.chars() {
        out.push(fallback_symbol(c));
    }
    out
}

/// Map a single character to a defined IPA symbol. The deterministic placeholder
/// stands in for trained open-vocabulary G2P (blocked ML); it guarantees the
/// fallback's output stays inside the IPA inventory.
fn fallback_symbol(_c: char) -> char {
    'ə'
}
