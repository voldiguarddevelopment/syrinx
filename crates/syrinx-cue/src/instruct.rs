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
    /// A `[[tuned]]` row is present but not usable as written: a provenance field that is
    /// blank or out of range, a `lang` no lookup can ever produce, or a phrase that would
    /// carry cue markup back into a backend.
    ///
    /// Rejected at LOAD rather than defaulted, because a tuned row's whole justification is
    /// its provenance (ADR-0004 §1): a row whose `judge` is `""` is not a row that records
    /// which judge measured it, and a row nobody notices is broken is one signature away
    /// from shipping.
    InvalidTuned { label: String, field: &'static str, why: String },
    /// Two accepted `[[tuned]]` rows claim the same `(backend, lang, label)`. The file does
    /// not then say what will be spoken, and silently keeping the last one written makes
    /// the answer depend on row order.
    DuplicateTuned { backend: String, lang: String, label: String },
}

impl std::fmt::Display for InstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructError::Parse(e) => write!(f, "instruct table parse error: {e}"),
            InstructError::Empty { label, lang } => {
                write!(f, "instruct phrase for {label:?} is empty in {lang}")
            }
            InstructError::InvalidTuned { label, field, why } => {
                write!(f, "tuned row for {label:?}: field {field} is {why}")
            }
            InstructError::DuplicateTuned { backend, lang, label } => {
                write!(f, "two accepted tuned rows for ({backend}, {lang}, {label})")
            }
        }
    }
}

impl std::error::Error for InstructError {}

/// A label's grammatical role, which decides what sentence frames can hold it.
///
/// Exists because a generator that treats every label as a predicate adjective emits
/// "Speak in a sharply narrator tone". Only 27 of 38 labels are adjectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Form {
    /// `sad`, `angry`, `calm` — fits "Speak in a {label} tone".
    Adjective,
    /// `whisper`, `shout`, `fast` — a way of speaking; takes an imperative frame and has
    /// no adjective slot at all.
    Manner,
    /// `narrator`, `child` — someone to sound like; "Speak like a {noun}".
    Persona,
    /// `accented` — something to speak *with*; "Speak with a {noun}". Neither a manner nor
    /// a persona: "Speak like a noticeable accent" is not a sentence.
    With,
    /// Not declared. Generates **nothing** — an inert default rather than a plausible
    /// wrong one. A frozen test asserts no shipped row is left here.
    #[default]
    Unspecified,
}

/// One label's phrasing in both languages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Phrasing {
    /// The noun phrase the non-adjective frames need. The label id is often unusable
    /// there: `fast` and `elderly` are adjectives even though the cue means "a hurried
    /// delivery" and "an elderly person", and the first version of this grammar emitted
    /// "Use a fast throughout" and "Speak like an elderly" as a result.
    #[serde(default)]
    pub noun: Option<String>,
    /// Defaulted so a table written before `form` existed still parses — but the default
    /// is inert, so forgetting it costs candidates rather than producing bad ones.
    #[serde(default)]
    pub form: Form,
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
            // Validation applies to EVERY row, signed or not — ADR-0004 §3: an unsigned row
            // is "present in the file, validated on load, and returned by no lookup". A
            // malformed proposal must not be able to sit in the file unnoticed until the
            // day someone signs it, because signing is the moment nobody re-reads the
            // provenance.
            let bad = |field, why: &str| {
                Err(InstructError::InvalidTuned {
                    label: t.label.clone(),
                    field,
                    why: why.to_string(),
                })
            };

            // The CLAUDE.md hard invariant. A tuned phrase is machine-generated prose that
            // reaches a model, and no human reads it between the search and the file, so a
            // `[` in it is exactly the cue-markup leak the invariant forbids.
            if let Err(why) = phrase_is_safe(&t.phrase) {
                return bad("phrase", &why);
            }
            // A blank provenance field is a missing one that got past serde.
            for (field, v) in [
                ("label", &t.label),
                ("backend", &t.backend),
                ("lang", &t.lang),
                ("measured_on", &t.measured_on),
                ("incumbent", &t.incumbent),
                ("judge", &t.judge),
                ("holdout_id", &t.holdout_id),
            ] {
                if v.trim().is_empty() {
                    return bad(field, "blank");
                }
            }
            // A `lang` outside the closed set is a row no lookup can ever return: the key
            // is built from `lang_code`, which is total over the two `InstructLang`
            // variants. Silently inert is the failure mode this validation exists to stop,
            // so it is an error. `backend` gets no equivalent check on purpose — it is an
            // OPEN set (a new checkpoint appears before `caps.toml` lists it), so a row
            // naming an unknown backend is not provably unreachable the way this one is.
            if !known_lang(&t.lang) {
                return bad("lang", "not a known instruct language");
            }
            // The number that makes the verdict quotable at all (ADR-0004 §1). NaN and
            // out-of-range are both nonsense, and both would otherwise travel with the row
            // as though they were measurements.
            let recall = t.judge_recall_on_class;
            if !recall.is_finite() || recall < 0.0 || recall > 1.0 {
                return bad("judge_recall_on_class", "not a recall in [0, 1]");
            }
            // ADR-0004 §4(v): the margin is over the re-measured incumbent and "a tie keeps
            // the incumbent", so a row recording a margin of zero or less records a
            // candidate that should never have been written down.
            if !t.margin.is_finite() || t.margin <= 0.0 {
                return bad("margin", "not a positive margin over the incumbent");
            }

            // An unsigned row is inert: validated above, and then deliberately dropped.
            match t.accepted_by.as_deref().map(str::trim) {
                Some(sig) if !sig.is_empty() => {
                    let key = (t.backend.clone(), t.lang.clone(), t.label.clone());
                    if tuned.insert(key, t.clone()).is_some() {
                        return Err(InstructError::DuplicateTuned {
                            backend: t.backend.clone(),
                            lang: t.lang.clone(),
                            label: t.label.clone(),
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(InstructTable { entries, tuned, all_tuned: raw.tuned })
    }

    /// A label's grammatical form, if the table covers it.
    pub fn form(&self, label: &str) -> Option<Form> {
        self.entries.get(label).map(|p| p.form)
    }

    /// The noun phrase a non-adjective frame should use for this label.
    pub fn noun(&self, label: &str) -> Option<&str> {
        self.entries.get(label).and_then(|p| p.noun.as_deref())
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

/// Every `lang` key a lookup can ever produce.
///
/// Derived from [`lang_code`] rather than restated as a literal list, so a third language
/// cannot leave this stale — which would turn a valid row into a load error.
pub const KNOWN_LANGS: [InstructLang; 2] = [InstructLang::En, InstructLang::Zh];

/// Is this the `lang` key of a language the table can be looked up in?
///
/// A `[[tuned]]` row with any other `lang` is unreachable by construction: the lookup key
/// is built from [`lang_code`], which is total over [`KNOWN_LANGS`].
pub fn known_lang(code: &str) -> bool {
    KNOWN_LANGS.iter().any(|l| lang_code(*l) == code)
}

/// Is this string safe to put in front of a backend as an instruction?
///
/// **This is the crate that owns the CLAUDE.md hard invariant**, and an instruct string is
/// the one piece of cue-derived text that reaches a model as prose rather than being
/// stripped from it — so a `[` here is precisely the leak the invariant forbids, whether it
/// came from a hand-authored row or from a tuning search.
///
/// The rules are deliberately identical to `syrinx_eval::tune::phrase_is_safe`, which gates
/// a phrase at *proposal* time; this one gates it at *load* time. Two gates, one rule: the
/// eval crate is optional and model-gated, so it cannot be the only place the invariant is
/// enforced, and the two must never diverge. `tests/instruct_tuned_path.rs` asserts they
/// agree case by case.
pub fn phrase_is_safe(phrase: &str) -> Result<(), String> {
    let p = phrase.trim();
    if p.is_empty() {
        return Err("empty".into());
    }
    if p != phrase {
        return Err("leading or trailing whitespace".into());
    }
    if p.contains('\n') {
        return Err("multi-line".into());
    }
    if p.chars().count() > 120 {
        return Err(format!("too long ({} chars, limit 120)", p.chars().count()));
    }
    for bad in ['[', ']', '<', '>', '|'] {
        if p.contains(bad) {
            return Err(format!("contains {bad:?}"));
        }
    }
    if p.contains("endofprompt") {
        return Err("carries a prompt delimiter".into());
    }
    Ok(())
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

