//! Lip-sync timeline export (T-07.06).
//!
//! A deterministic table mapping over a caller-supplied, ordered, contiguous
//! list of phoneme timings — no audio alignment, no model output. Each input
//! entry becomes exactly one output segment over the identical `[start_ms,
//! end_ms)` span, with the viseme resolved through the fixed `phoneme_to_viseme`
//! table. There is no interpolation, smoothing, or merging of adjacent equal
//! visemes, so the timeline covers the full input span with no gaps and no
//! overlaps. An empty input yields an empty timeline; any phoneme outside the
//! table (including the empty string) resolves to `Viseme::Rest` and never
//! panics.

/// The closed set of viseme classes a phoneme can resolve to. `Viseme::Rest` is
/// the neutral/rest class used for silence and for any phoneme outside the fixed
/// table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Viseme {
    /// Lips fully closed (p, b, m).
    Bilabial,
    /// Lower lip to upper teeth (f, v).
    LabioDental,
    /// Tongue to teeth (θ, ð).
    Dental,
    /// Tongue tip to ridge (t, d, s, z, n, l).
    Alveolar,
    /// Back of tongue to soft palate (k, ɡ, ŋ).
    Velar,
    /// Rounded, slightly protruded (ʃ, ʒ).
    PostAlveolar,
    /// Front (spread) vowels (i, ɪ, e, ɛ, æ).
    FrontVowel,
    /// Rounded vowels and the rounded glide (u, ʊ, o, ɔ, w).
    RoundVowel,
    /// Open / central vowels (ɑ, ʌ, ə).
    OpenVowel,
    /// Neutral/rest class — silence and any phoneme outside the table.
    Rest,
}

/// One typed input entry: a phoneme label over its `[start_ms, end_ms)` span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhonemeTiming {
    pub phoneme: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

/// One output segment: the resolved viseme class over the same span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisemeSegment {
    pub viseme: Viseme,
    pub start_ms: u32,
    pub end_ms: u32,
}

/// The fixed, deterministic phoneme→viseme table. IPA phonemes map to their
/// articulatory viseme class; anything not in the table (an unknown label, the
/// empty string) maps to `Viseme::Rest`. Total — never panics.
pub fn phoneme_to_viseme(phoneme: &str) -> Viseme {
    match phoneme {
        "p" | "b" | "m" => Viseme::Bilabial,
        "f" | "v" => Viseme::LabioDental,
        "θ" | "ð" => Viseme::Dental,
        "t" | "d" | "s" | "z" | "n" | "l" => Viseme::Alveolar,
        "k" | "ɡ" | "ŋ" => Viseme::Velar,
        "ʃ" | "ʒ" => Viseme::PostAlveolar,
        "i" | "ɪ" | "e" | "ɛ" | "æ" => Viseme::FrontVowel,
        "u" | "ʊ" | "o" | "ɔ" | "w" => Viseme::RoundVowel,
        "ɑ" | "ʌ" | "ə" => Viseme::OpenVowel,
        _ => Viseme::Rest,
    }
}

/// Map a contiguous list of phoneme timings to a viseme timeline. Each entry
/// becomes exactly one segment over the identical span, with its viseme resolved
/// through `phoneme_to_viseme`. An empty input yields an empty timeline.
pub fn lip_sync_timeline(entries: &[PhonemeTiming]) -> Vec<VisemeSegment> {
    entries
        .iter()
        .map(|entry| VisemeSegment {
            viseme: phoneme_to_viseme(&entry.phoneme),
            start_ms: entry.start_ms,
            end_ms: entry.end_ms,
        })
        .collect()
}
