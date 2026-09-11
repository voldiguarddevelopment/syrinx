//! The canonical cue vocabulary, loaded from `vocab.toml`.
//!
//! Per ADR-0001 the vocabulary is **data, not Rust literals**, so it can be extended
//! without a recompile. That only holds if the schema is enforced, so every invariant the
//! lowering passes rely on is checked by a test here rather than assumed:
//! ids unique across all kinds, synonyms unique across all entries (an ambiguous synonym
//! would make parsing non-deterministic), scalars in range, and — the one that matters for
//! honest degradation — **every entry expressible on Fish S2**, whose vocabulary the
//! survey confirmed is open.

use serde::Deserialize;
use std::collections::HashMap;

/// Which family of cue an entry describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Emotion,
    Style,
    Event,
}

/// One canonical vocabulary entry.
#[derive(Debug, Clone, Deserialize)]
pub struct Entry {
    pub id: String,
    pub arousal: f32,
    /// Only emotions carry valence; styles and events are unvalenced.
    #[serde(default)]
    pub valence: Option<f32>,
    pub synonyms: Vec<String>,
    /// Native spellings. Absence is meaningful: it drives `Dropped`/`Hoisted` outcomes.
    #[serde(default)]
    pub fish_s1: Option<String>,
    #[serde(default)]
    pub fish_s2: Option<String>,
    #[serde(default)]
    pub cosyvoice: Option<String>,
    #[serde(default)]
    pub step_audio: Option<String>,
    /// Chatterbox Turbo's native `[tag]`, brackets included. A CLOSED nineteen-token set
    /// (`added_tokens.json`, ids 50257-50275), so absence here means "no counterpart",
    /// not "we did not get round to it" — see the column's note in `vocab.toml`.
    #[serde(default)]
    pub chatterbox_turbo: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawVocab {
    #[serde(default)]
    emotion: Vec<Entry>,
    #[serde(default)]
    style: Vec<Entry>,
    #[serde(default)]
    event: Vec<Entry>,
}

/// The loaded vocabulary with its lookup indexes.
#[derive(Debug, Clone)]
pub struct Vocab {
    entries: Vec<(Kind, Entry)>,
    by_id: HashMap<String, usize>,
    by_synonym: HashMap<String, usize>,
}

/// The vocabulary shipped with the crate, embedded at build time so a deployed binary
/// needs no data file alongside it.
pub const EMBEDDED_VOCAB: &str = include_str!("../vocab.toml");

#[derive(Debug)]
pub enum VocabError {
    Parse(String),
    Schema(String),
}

impl std::fmt::Display for VocabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(m) => write!(f, "vocab parse: {m}"),
            Self::Schema(m) => write!(f, "vocab schema: {m}"),
        }
    }
}

impl std::error::Error for VocabError {}

impl Vocab {
    /// Load and validate the embedded vocabulary.
    pub fn embedded() -> Result<Self, VocabError> {
        Self::from_toml(EMBEDDED_VOCAB)
    }

    /// Parse and validate a vocabulary document.
    pub fn from_toml(src: &str) -> Result<Self, VocabError> {
        let raw: RawVocab = toml::from_str(src).map_err(|e| VocabError::Parse(e.to_string()))?;
        let mut entries = Vec::new();
        for (kind, list) in [
            (Kind::Emotion, raw.emotion),
            (Kind::Style, raw.style),
            (Kind::Event, raw.event),
        ] {
            for e in list {
                entries.push((kind, e));
            }
        }

        let mut by_id = HashMap::new();
        let mut by_synonym = HashMap::new();
        for (i, (kind, e)) in entries.iter().enumerate() {
            if e.id.is_empty() {
                return Err(VocabError::Schema("entry with empty id".into()));
            }
            if !e
                .id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                return Err(VocabError::Schema(format!(
                    "id `{}` must be lowercase snake_case",
                    e.id
                )));
            }
            if by_id.insert(e.id.clone(), i).is_some() {
                return Err(VocabError::Schema(format!("duplicate id `{}`", e.id)));
            }
            if !(0.0..=1.0).contains(&e.arousal) {
                return Err(VocabError::Schema(format!(
                    "`{}` arousal {} out of 0..1",
                    e.id, e.arousal
                )));
            }
            match (kind, e.valence) {
                (Kind::Emotion, None) => {
                    return Err(VocabError::Schema(format!("emotion `{}` needs valence", e.id)))
                }
                (Kind::Emotion, Some(v)) if !(0.0..=1.0).contains(&v) => {
                    return Err(VocabError::Schema(format!(
                        "`{}` valence {v} out of 0..1",
                        e.id
                    )))
                }
                _ => {}
            }
            if e.synonyms.is_empty() {
                return Err(VocabError::Schema(format!("`{}` has no synonyms", e.id)));
            }
            for s in &e.synonyms {
                if s != &s.to_lowercase() {
                    return Err(VocabError::Schema(format!(
                        "synonym `{s}` on `{}` must be lowercase",
                        e.id
                    )));
                }
                // An ambiguous synonym would make parsing non-deterministic, so this is a
                // hard error rather than a first-wins resolution.
                if let Some(prev) = by_synonym.insert(s.clone(), i) {
                    return Err(VocabError::Schema(format!(
                        "synonym `{s}` claimed by both `{}` and `{}`",
                        entries[prev].1.id, e.id
                    )));
                }
            }
        }
        Ok(Self { entries, by_id, by_synonym })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Kind, &Entry)> {
        self.entries.iter().map(|(k, e)| (*k, e))
    }

    pub fn by_id(&self, id: &str) -> Option<(Kind, &Entry)> {
        self.by_id.get(id).map(|&i| (self.entries[i].0, &self.entries[i].1))
    }

    /// Resolve an author's free text to a canonical entry. Case-insensitive; whitespace
    /// trimmed. Returns `None` for anything unrecognised — the caller makes it a `Free`
    /// cue rather than guessing, per ADR-0001 §2.2.
    pub fn resolve(&self, text: &str) -> Option<(Kind, &Entry)> {
        let key = text.trim().to_lowercase();
        self.by_synonym
            .get(&key)
            .map(|&i| (self.entries[i].0, &self.entries[i].1))
    }

    pub fn count_of(&self, kind: Kind) -> usize {
        self.entries.iter().filter(|(k, _)| *k == kind).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v() -> Vocab {
        Vocab::embedded().expect("embedded vocab must load and validate")
    }

    /// C1.1 AC: the shipped vocabulary is validated by a test, and meets the ≥40 floor.
    #[test]
    fn embedded_vocab_loads_and_meets_the_size_floor() {
        let v = v();
        assert!(
            v.len() >= 40,
            "C1.1 requires >=40 canonical labels, found {}",
            v.len()
        );
        // All three kinds must be represented or the lowering passes have nothing to
        // exercise for that branch.
        for k in [Kind::Emotion, Kind::Style, Kind::Event] {
            assert!(v.count_of(k) > 0, "no entries of kind {k:?}");
        }
    }

    #[test]
    fn ids_and_synonyms_are_unique_and_resolvable() {
        let v = v();
        for (_, e) in v.iter() {
            assert!(v.by_id(&e.id).is_some(), "`{}` not indexed by id", e.id);
            for s in &e.synonyms {
                let (_, got) = v.resolve(s).unwrap_or_else(|| panic!("synonym `{s}` unresolved"));
                assert_eq!(got.id, e.id, "synonym `{s}` resolved to the wrong entry");
            }
        }
    }

    #[test]
    fn resolution_is_case_and_whitespace_insensitive() {
        let v = v();
        assert_eq!(v.resolve("  WhIsPeRiNg ").unwrap().1.id, "whisper");
        assert!(v.resolve("definitely not a cue").is_none());
    }

    /// Fish S2's vocabulary is open (survey: "15,000+ unique tags"), so every canonical
    /// label must be expressible there. If this fails we have invented a label with no
    /// backend that can render it, which would make honest degradation impossible to test.
    #[test]
    fn every_entry_is_expressible_on_fish_s2() {
        let v = v();
        let missing: Vec<&str> = v
            .iter()
            .filter(|(_, e)| e.fish_s2.is_none())
            .map(|(_, e)| e.id.as_str())
            .collect();
        assert!(missing.is_empty(), "no fish_s2 spelling for: {missing:?}");
    }

    /// Fish S1 is a CLOSED set of 65 tags. Any `fish_s1` spelling we claim must be one the
    /// model actually knows, so these are cross-checked against the enumerated card list.
    #[test]
    fn fish_s1_spellings_are_in_the_documented_closed_set() {
        // Verbatim from ~/models/openaudio-s1-mini/README.md, recorded in C0.1.
        const S1: &[&str] = &[
            "angry","sad","disdainful","excited","surprised","satisfied","unhappy","anxious",
            "hysterical","delighted","scared","worried","indifferent","upset","impatient",
            "nervous","guilty","scornful","frustrated","depressed","panicked","furious",
            "empathetic","embarrassed","reluctant","disgusted","keen","moved","proud",
            "relaxed","grateful","confident","interested","curious","confused","joyful",
            "disapproving","negative","denying","astonished","serious","sarcastic",
            "conciliative","comforting","sincere","sneering","hesitating","yielding",
            "painful","awkward","amused",
            "in a hurry tone","shouting","screaming","whispering","soft tone",
            "laughing","chuckling","sobbing","crying loudly","sighing","panting","groaning",
            "crowd laughing","background laughter","audience laughing",
        ];
        let v = v();
        let bad: Vec<String> = v
            .iter()
            .filter_map(|(_, e)| e.fish_s1.as_ref().map(|s| (e.id.clone(), s.clone())))
            .filter(|(_, s)| !S1.contains(&s.as_str()))
            .map(|(id, s)| format!("{id} -> `{s}`"))
            .collect();
        assert!(bad.is_empty(), "fish_s1 spellings outside the closed 65-tag set: {bad:?}");
    }

    /// CosyVoice's inline set is closed and known exactly (tokenizer
    /// `additional_special_tokens`). Same argument as above.
    #[test]
    fn cosyvoice_spellings_are_in_the_documented_closed_set() {
        const CV: &[&str] = &[
            "[breath]", "[noise]", "[laughter]", "[cough]", "[clucking]", "[accent]",
            "[quick_breath]",
        ];
        let v = v();
        let bad: Vec<String> = v
            .iter()
            .filter_map(|(_, e)| e.cosyvoice.as_ref().map(|s| (e.id.clone(), s.clone())))
            .filter(|(_, s)| !CV.contains(&s.as_str()))
            .map(|(id, s)| format!("{id} -> `{s}`"))
            .collect();
        assert!(bad.is_empty(), "cosyvoice spellings outside the closed set: {bad:?}");
    }

    #[test]
    fn schema_rejects_duplicate_ids_and_ambiguous_synonyms() {
        let dup_id = r#"
[[emotion]]
id = "x"
arousal = 0.5
valence = 0.5
synonyms = ["a"]
fish_s2 = "x"
[[event]]
id = "x"
arousal = 0.5
synonyms = ["b"]
fish_s2 = "x"
"#;
        assert!(matches!(Vocab::from_toml(dup_id), Err(VocabError::Schema(m)) if m.contains("duplicate id")));

        let dup_syn = r#"
[[emotion]]
id = "p"
arousal = 0.5
valence = 0.5
synonyms = ["same"]
fish_s2 = "p"
[[emotion]]
id = "q"
arousal = 0.5
valence = 0.5
synonyms = ["same"]
fish_s2 = "q"
"#;
        assert!(matches!(Vocab::from_toml(dup_syn), Err(VocabError::Schema(m)) if m.contains("claimed by both")));
    }

    #[test]
    fn schema_rejects_out_of_range_scalars_and_missing_valence() {
        let bad_arousal = "[[event]]\nid = \"e\"\narousal = 1.5\nsynonyms = [\"e\"]\nfish_s2 = \"e\"\n";
        assert!(matches!(Vocab::from_toml(bad_arousal), Err(VocabError::Schema(m)) if m.contains("arousal")));
        let no_valence = "[[emotion]]\nid = \"e\"\narousal = 0.5\nsynonyms = [\"e\"]\nfish_s2 = \"e\"\n";
        assert!(matches!(Vocab::from_toml(no_valence), Err(VocabError::Schema(m)) if m.contains("needs valence")));
    }
}
