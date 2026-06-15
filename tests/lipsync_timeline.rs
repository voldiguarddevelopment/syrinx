//! Frozen RED tests for T-07.06 — export the lip-sync timeline.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must build in `syrinx-prosody::lipsync`:
//!
//!   * `Viseme` — the closed set of viseme classes a phoneme can resolve to. It
//!     is `Copy + Eq + Debug`, and `Viseme::Rest` is the neutral/rest class used
//!     for silence and for any phoneme outside the fixed table.
//!   * `PhonemeTiming { phoneme: String, start_ms: u32, end_ms: u32 }` — one
//!     typed input entry: a phoneme label and its `[start_ms, end_ms)` span.
//!   * `VisemeSegment { viseme: Viseme, start_ms: u32, end_ms: u32 }` — one output
//!     segment: the resolved viseme class over the same span.
//!   * `phoneme_to_viseme(phoneme: &str) -> Viseme` — the fixed, deterministic
//!     phoneme→viseme table. IPA phonemes (the project's phoneme alphabet, see
//!     `tests/g2p.rs`) map to their articulatory viseme class; anything not in the
//!     table (an unknown label, the empty string) maps to `Viseme::Rest`. Total —
//!     never panics.
//!   * `lip_sync_timeline(entries: &[PhonemeTiming]) -> Vec<VisemeSegment>` — maps
//!     a contiguous list of phoneme timings to a viseme timeline. Each input entry
//!     becomes exactly one output segment over the identical `[start_ms, end_ms)`
//!     span, with `viseme = phoneme_to_viseme(&entry.phoneme)`. No interpolation,
//!     no smoothing, no merging of adjacent equal visemes.
//!
//! Contract (list.md / DESIGN §T7.6): a deterministic table mapping over a
//! caller-supplied, ordered, contiguous timestamp list — no audio alignment, no
//! model output. The output covers the full input span with no gaps and no
//! overlaps (each segment's start equals the previous segment's end; the last end
//! equals the input's last end). An empty input yields an empty timeline; an
//! unknown phoneme yields the rest viseme and never panics.
//!
//! RED: `syrinx-prosody` exposes no `lipsync` module yet, so none of these
//! symbols resolve and the test target fails to build — every criterion is unmet.
//! GREEN adds the module so each assertion below holds.

use syrinx_prosody::lipsync::{
    lip_sync_timeline, phoneme_to_viseme, PhonemeTiming, Viseme, VisemeSegment,
};

/// The fixed phoneme→viseme table the implementation must reproduce exactly.
/// Each `(IPA phoneme, expected viseme class)` pair is asserted in both
/// directions (every listed phoneme resolves to its class, and the class is not
/// the rest default), so a deleted or altered arm leaves a killed mutant.
const TABLE: &[(&str, Viseme)] = &[
    // Bilabial — lips fully closed.
    ("p", Viseme::Bilabial),
    ("b", Viseme::Bilabial),
    ("m", Viseme::Bilabial),
    // Labiodental — lower lip to upper teeth.
    ("f", Viseme::LabioDental),
    ("v", Viseme::LabioDental),
    // Dental — tongue to teeth.
    ("θ", Viseme::Dental),
    ("ð", Viseme::Dental),
    // Alveolar — tongue tip to ridge.
    ("t", Viseme::Alveolar),
    ("d", Viseme::Alveolar),
    ("s", Viseme::Alveolar),
    ("z", Viseme::Alveolar),
    ("n", Viseme::Alveolar),
    ("l", Viseme::Alveolar),
    // Velar — back of tongue to soft palate.
    ("k", Viseme::Velar),
    ("ɡ", Viseme::Velar),
    ("ŋ", Viseme::Velar),
    // Post-alveolar — rounded, slightly protruded.
    ("ʃ", Viseme::PostAlveolar),
    ("ʒ", Viseme::PostAlveolar),
    // Front (spread) vowels.
    ("i", Viseme::FrontVowel),
    ("ɪ", Viseme::FrontVowel),
    ("e", Viseme::FrontVowel),
    ("ɛ", Viseme::FrontVowel),
    ("æ", Viseme::FrontVowel),
    // Rounded vowels / the rounded glide.
    ("u", Viseme::RoundVowel),
    ("ʊ", Viseme::RoundVowel),
    ("o", Viseme::RoundVowel),
    ("ɔ", Viseme::RoundVowel),
    ("w", Viseme::RoundVowel),
    // Open / central vowels.
    ("ɑ", Viseme::OpenVowel),
    ("ʌ", Viseme::OpenVowel),
    ("ə", Viseme::OpenVowel),
];

/// Convenience: an owned input entry from a phoneme and its span.
fn timing(phoneme: &str, start_ms: u32, end_ms: u32) -> PhonemeTiming {
    PhonemeTiming {
        phoneme: phoneme.to_string(),
        start_ms,
        end_ms,
    }
}

// ----------------------------------------------------------------------------
// C1 — every phoneme resolves to its viseme class per the fixed table.
// ----------------------------------------------------------------------------

/// Each phoneme in the fixed table resolves to exactly its listed viseme class,
/// and that class is never the rest default (so a collapsed/deleted arm is
/// caught). Pins the whole table in one sweep.
#[test]
fn test_each_phoneme_maps_to_its_table_viseme() {
    for (phoneme, expected) in TABLE {
        let got = phoneme_to_viseme(phoneme);
        assert_eq!(
            got, *expected,
            "phoneme {phoneme:?} expected {expected:?}, got {got:?}"
        );
        assert_ne!(
            got,
            Viseme::Rest,
            "phoneme {phoneme:?} is in the table and must not resolve to Rest"
        );
    }
}

/// Phonemes from *different* articulatory groups land in *distinct* viseme
/// classes — a single representative per class, all required to differ. This
/// kills a mutant that funnels every phoneme to one class while still being
/// non-Rest.
#[test]
fn test_distinct_groups_map_to_distinct_visemes() {
    let representatives = [
        phoneme_to_viseme("b"), // Bilabial
        phoneme_to_viseme("f"), // LabioDental
        phoneme_to_viseme("θ"), // Dental
        phoneme_to_viseme("t"), // Alveolar
        phoneme_to_viseme("k"), // Velar
        phoneme_to_viseme("ʃ"), // PostAlveolar
        phoneme_to_viseme("i"), // FrontVowel
        phoneme_to_viseme("u"), // RoundVowel
        phoneme_to_viseme("ɑ"), // OpenVowel
    ];
    for (i, a) in representatives.iter().enumerate() {
        for (j, b) in representatives.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "representatives at {i} and {j} must differ");
            }
        }
    }
}

/// In a full timeline each segment carries the viseme its phoneme maps to under
/// the same table — ties the table mapping (C1) into `lip_sync_timeline`'s
/// output, not just the bare `phoneme_to_viseme` helper.
#[test]
fn test_timeline_segments_carry_mapped_viseme() {
    let entries = vec![
        timing("m", 0, 50),   // Bilabial
        timing("ɑ", 50, 120), // OpenVowel
        timing("t", 120, 160), // Alveolar
    ];
    let timeline = lip_sync_timeline(&entries);
    assert_eq!(timeline.len(), 3);
    assert_eq!(timeline[0].viseme, Viseme::Bilabial);
    assert_eq!(timeline[1].viseme, Viseme::OpenVowel);
    assert_eq!(timeline[2].viseme, Viseme::Alveolar);
}

// ----------------------------------------------------------------------------
// C2 — the timeline covers the input span contiguously: no gaps, no overlaps.
// ----------------------------------------------------------------------------

/// One segment per input entry, each over the identical span; consecutive
/// segments meet exactly (`seg[i].start == seg[i-1].end`), the first start equals
/// the input's first start, and the last end equals the input's last end. Spans
/// are distinct and non-empty so a swapped or shifted bound leaves a killed
/// mutant.
#[test]
fn test_timeline_covers_span_contiguously() {
    let entries = vec![
        timing("h", 0, 100),    // unknown-ish but spans are what matter here
        timing("ə", 100, 250),
        timing("l", 250, 400),
        timing("o", 400, 530),
    ];
    let timeline = lip_sync_timeline(&entries);

    // Exactly one output segment per input entry.
    assert_eq!(timeline.len(), entries.len());

    // Each segment spans exactly its source entry's [start_ms, end_ms).
    for (seg, entry) in timeline.iter().zip(entries.iter()) {
        assert_eq!(seg.start_ms, entry.start_ms, "segment start matches entry");
        assert_eq!(seg.end_ms, entry.end_ms, "segment end matches entry");
    }

    // No gaps and no overlaps: each segment begins where the previous ended.
    for i in 1..timeline.len() {
        assert_eq!(
            timeline[i].start_ms,
            timeline[i - 1].end_ms,
            "segment {i} must start exactly at segment {} end",
            i - 1
        );
    }

    // The timeline covers the full input span end to end.
    assert_eq!(
        timeline.first().unwrap().start_ms,
        entries.first().unwrap().start_ms,
        "timeline starts at the input's first start"
    );
    assert_eq!(
        timeline.last().unwrap().end_ms,
        entries.last().unwrap().end_ms,
        "timeline ends at the input's last end"
    );
}

/// A single-entry input produces a single segment over the whole span — the
/// degenerate boundary of the coverage invariant (first start == last end's
/// entry, exactly one segment).
#[test]
fn test_single_entry_spans_whole_input() {
    let entries = vec![timing("s", 30, 95)];
    let timeline = lip_sync_timeline(&entries);
    assert_eq!(timeline.len(), 1);
    assert_eq!(timeline[0].start_ms, 30);
    assert_eq!(timeline[0].end_ms, 95);
    assert_eq!(timeline[0].viseme, Viseme::Alveolar);
}

// ----------------------------------------------------------------------------
// C3 — empty input -> empty timeline; unknown phoneme -> rest, never panics.
// ----------------------------------------------------------------------------

/// An empty input list yields an empty timeline (not a one-element or panicking
/// result).
#[test]
fn test_empty_input_yields_empty_timeline() {
    let entries: Vec<PhonemeTiming> = Vec::new();
    let timeline = lip_sync_timeline(&entries);
    assert!(timeline.is_empty(), "empty input must yield an empty timeline");
}

/// Unknown labels and the empty string resolve to the neutral/rest viseme via the
/// bare helper — pins the table's default arm without going through a timeline.
#[test]
fn test_unknown_phoneme_maps_to_rest() {
    assert_eq!(phoneme_to_viseme("zzz"), Viseme::Rest);
    assert_eq!(phoneme_to_viseme(""), Viseme::Rest);
    assert_eq!(phoneme_to_viseme("ʔ"), Viseme::Rest); // a real IPA symbol, not in the table
}

/// An unknown phoneme inside a timeline produces a `Rest` segment over its span
/// without panicking, and the surrounding known segments still map and stay
/// contiguous — the rest viseme is woven in, not dropped.
#[test]
fn test_unknown_phoneme_in_timeline_is_rest_segment() {
    let entries = vec![
        timing("p", 0, 40),      // Bilabial
        timing("Q", 40, 90),     // unknown -> Rest
        timing("i", 90, 140),    // FrontVowel
    ];
    let timeline = lip_sync_timeline(&entries);

    assert_eq!(timeline.len(), 3);
    assert_eq!(timeline[0].viseme, Viseme::Bilabial);
    assert_eq!(timeline[1].viseme, Viseme::Rest);
    assert_eq!(timeline[2].viseme, Viseme::FrontVowel);

    // The rest segment still covers its span and keeps the timeline contiguous.
    assert_eq!(timeline[1].start_ms, 40);
    assert_eq!(timeline[1].end_ms, 90);
    assert_eq!(timeline[1].start_ms, timeline[0].end_ms);
    assert_eq!(timeline[2].start_ms, timeline[1].end_ms);
}

/// Construction sanity: the public segment type is comparable so callers (and
/// these tests) can assert exact segments. Pins the `VisemeSegment` shape and its
/// `PartialEq`.
#[test]
fn test_viseme_segment_is_constructible_and_comparable() {
    let a = VisemeSegment {
        viseme: Viseme::Rest,
        start_ms: 5,
        end_ms: 10,
    };
    let b = VisemeSegment {
        viseme: Viseme::Rest,
        start_ms: 5,
        end_ms: 10,
    };
    assert_eq!(a, b);
}
