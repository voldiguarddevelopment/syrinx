//! The curated instruct phrasing for each cue label — [`instruct.toml`](../instruct.toml).
//!
//! A backend with [`Inline::None`](crate::caps::Inline) carries no cue markup in its text
//! stream; its one expressive channel is an utterance-scoped instruction in plain prose.
//! [`hoist::instruct_for`](crate::hoist::instruct_for) turns a label into that prose and
//! this table is what it reads.
//!
//! It replaces `legacy_emotion::DEFAULT_EMOTIONS`, which was wrong in two ways at once:
//! it lived in the module CLAUDE.md quarantines as the deprecated CosyVoice-era parser,
//! and it was keyed on legacy CosyVoice tag names while `instruct_for` looks up canonical
//! `vocab.toml` ids — so 38 of 51 labels missed and fell through to a generic phrasing
//! that produced "Speak in a anxious tone" and "Speak in a quick_breath tone".

use std::collections::BTreeMap;

use crate::legacy_emotion::InstructLang;

/// The embedded phrase table. One source of truth, compiled in like `vocab.toml`.
pub const EMBEDDED_INSTRUCT: &str = include_str!("../instruct.toml");

/// Why a phrase table would not load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructError {
    Parse(String),
    /// A label carries an empty phrase in one of the languages.
    Empty { label: String, lang: &'static str },
}

impl std::fmt::Display for InstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructError::Parse(e) => write!(f, "instruct table parse error: {e}"),
            InstructError::Empty { label, lang } => {
                write!(f, "instruct phrase for {label:?} is empty in {lang}")
            }
        }
    }
}

impl std::error::Error for InstructError {}

/// One label's phrasing in both languages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Phrasing {
    pub en: String,
    pub zh: String,
}

impl Phrasing {
    /// The phrase in one language.
    pub fn get(&self, lang: InstructLang) -> &str {
        match lang {
            InstructLang::En => &self.en,
            InstructLang::Zh => &self.zh,
        }
    }
}

/// A phrasing found by the tuning loop rather than authored by hand.
///
/// It carries its whole provenance because it has a genuinely new KIND of provenance for
/// this crate. `vocab.toml`'s header promises its content is derived from verified upstream
/// sources and "NOT invented"; a tuned phrase is neither — it was *found by automated
/// search against a measurement, on one box, on one checkpoint, on one date, and signed off
/// by a person*. That is why tuned rows live in their own table with these fields and not
/// beside the curated ones.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TunedPhrase {
    pub label: String,
    pub lang: String,
    /// Which checkpoint it was tuned for. Instruct semantics are per checkpoint (ADR-0001
    /// §2.3): CustomVoice's instruct describes delivery, VoiceDesign's describes the voice.
    pub backend: String,
    pub phrase: String,
    /// **The gate.** A row without a human signature is INERT — present in the file,
    /// returned by no lookup. The loop proposes; a person listens and signs. Per CLAUDE.md,
    /// "intended emotion" is not expressible as a frozen-test gate, and no measurement
    /// changes that.
    #[serde(default)]
    pub accepted_by: Option<String>,
    pub measured_on: String,
    /// The phrase this one beat, and by how much, in the judge's units.
    pub incumbent: String,
    pub margin: f64,
    /// The judge that measured it, and its recall on THIS class. A verdict quoted without
    /// the second number is not a verdict.
    pub judge: String,
    pub judge_recall_on_class: f64,
    /// Which holdout partition confirmed it, and how many decisions that partition had
    /// already served. A holdout used repeatedly stops being one.
    pub holdout_id: String,
    pub holdout_uses: usize,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, serde::Deserialize)]
struct RawTable {
    #[serde(default)]
    emotion: BTreeMap<String, Phrasing>,
    #[serde(default)]
    style: BTreeMap<String, Phrasing>,
    /// Tuned rows, in a flat list because they are keyed on (label, lang, backend).
    #[serde(default)]
    tuned: Vec<TunedPhrase>,
}

/// Canonical-id keyed instruct phrasings.
#[derive(Debug, Clone, PartialEq)]
pub struct InstructTable {
    entries: BTreeMap<String, Phrasing>,
    /// Keyed `(backend, lang, label)`. Only ACCEPTED rows reach this map; an unsigned row
    /// is parsed, validated, and then deliberately dropped.
    tuned: BTreeMap<(String, String, String), TunedPhrase>,
    /// Every tuned row as written, accepted or not, for reporting.
    all_tuned: Vec<TunedPhrase>,
}

impl InstructTable {
    /// The compiled-in table, parsed once per call. Prefer [`InstructTable::shared`].
    pub fn embedded() -> Result<Self, InstructError> {
        Self::from_toml(EMBEDDED_INSTRUCT)
    }

    /// The compiled-in table, parsed exactly once for the life of the process.
    ///
    /// `instruct_for` is called once per cue per request, and the previous implementation
    /// rebuilt a 19-entry `BTreeMap` on every one of those calls.
    pub fn shared() -> &'static InstructTable {
        static TABLE: std::sync::OnceLock<InstructTable> = std::sync::OnceLock::new();
        TABLE.get_or_init(|| {
            InstructTable::embedded().expect("the embedded instruct.toml must be valid")
        })
    }

    /// Parse and validate a phrase table.
    pub fn from_toml(src: &str) -> Result<Self, InstructError> {
        let raw: RawTable =
            toml::from_str(src).map_err(|e| InstructError::Parse(e.to_string()))?;
        let mut entries = BTreeMap::new();
        for (label, p) in raw.emotion.into_iter().chain(raw.style) {
            if p.en.trim().is_empty() {
                return Err(InstructError::Empty { label, lang: "en" });
            }
            if p.zh.trim().is_empty() {
                return Err(InstructError::Empty { label, lang: "zh" });
            }
            entries.insert(label, p);
        }

        let mut tuned = BTreeMap::new();
        for t in &raw.tuned {
            if t.phrase.trim().is_empty() {
                return Err(InstructError::Empty { label: t.label.clone(), lang: "tuned" });
            }
            // An unsigned row is inert. It is still validated, so a malformed proposal
            // cannot sit in the file unnoticed until the day someone signs it.
            match t.accepted_by.as_deref().map(str::trim) {
                Some(sig) if !sig.is_empty() => {
                    tuned.insert(
                        (t.backend.clone(), t.lang.clone(), t.label.clone()),
                        t.clone(),
                    );
                }
                _ => {}
            }
        }
        Ok(InstructTable { entries, tuned, all_tuned: raw.tuned })
    }

    /// The curated phrase for a canonical label, if there is one.
    pub fn phrase(&self, label: &str, lang: InstructLang) -> Option<&str> {
        self.entries.get(label).map(|p| p.get(lang))
    }

    /// The phrase to use for a label on a specific backend: an **accepted** tuned row if
    /// one exists, otherwise the curated one.
    ///
    /// Tuned rows are strictly additive — one never deletes or overwrites a curated phrase,
    /// so removing a tuned row restores the previous behaviour exactly. Reversibility is
    /// what makes the loop safe to run at all.
    pub fn phrase_for_backend(
        &self,
        label: &str,
        lang: InstructLang,
        backend: &str,
    ) -> Option<&str> {
        let key = (backend.to_string(), lang_code(lang).to_string(), label.to_string());
        if let Some(t) = self.tuned.get(&key) {
            return Some(t.phrase.as_str());
        }
        self.phrase(label, lang)
    }

    /// Every tuned row in the file, accepted or not.
    pub fn tuned_rows(&self) -> &[TunedPhrase] {
        &self.all_tuned
    }

    /// How many tuned rows are live (signed).
    pub fn accepted_tuned(&self) -> usize {
        self.tuned.len()
    }

    /// Every label this table covers.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// How many labels the table covers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the table empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The `lang` key a [`TunedPhrase`] row uses.
pub fn lang_code(lang: InstructLang) -> &'static str {
    match lang {
        InstructLang::En => "en",
        InstructLang::Zh => "zh",
    }
}

/// The English indefinite article for a word — "a" or "an".
///
/// Used only by the deterministic fallback in
/// [`instruct_for`](crate::hoist::instruct_for), for labels with no curated phrase. It is
/// a vowel-letter test, which is what the fallback's inputs need: every `vocab.toml` id is
/// a plain lower-case ASCII word. It is deliberately not a general English article
/// function — "an hour" and "a user" are both wrong here and neither can arise.
pub fn indefinite_article(word: &str) -> &'static str {
    match word.chars().next() {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

