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

#[derive(Debug, serde::Deserialize)]
struct RawTable {
    #[serde(default)]
    emotion: BTreeMap<String, Phrasing>,
    #[serde(default)]
    style: BTreeMap<String, Phrasing>,
}

/// Canonical-id keyed instruct phrasings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructTable {
    entries: BTreeMap<String, Phrasing>,
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
        Ok(InstructTable { entries })
    }

    /// The curated phrase for a canonical label, if there is one.
    pub fn phrase(&self, label: &str, lang: InstructLang) -> Option<&str> {
        self.entries.get(label).map(|p| p.get(lang))
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

