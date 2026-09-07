//! Candidate instruct phrasings for the tuning loop — a deterministic, **enumerable**
//! grammar.
//!
//! # The defect this replaces
//!
//! The first generator was four `format!` templates with a `{label}` slot, all of which
//! assumed the label was a predicate adjective:
//!
//! ```text
//! Speak in a sharply sad tone        <- odd
//! Speak in a sharply narrator tone   <- nonsense
//! Speak in a sharply quick_breath tone
//! ```
//!
//! Only 27 of the vocabulary's 38 non-event labels are adjectives, so **32% of that
//! generator's output space was malformed**. It went unnoticed not because the templates
//! were subtle but because *nothing ever enumerated the output*. That is the same bug that
//! put "Speak in a anxious tone" into the shipping path, and it has the same fix: make the
//! output space finite, walk all of it, and assert on every element.
//!
//! # Why not an LLM proposer
//!
//! It would write grammatical phrases and would not fix this. The property that matters is
//! not fluency, it is that the output space can be **enumerated and gated**; a generator
//! whose range you cannot list cannot be frozen-tested. ADR-0004 defers an LLM proposer to
//! v2 and requires it to write into a reviewed candidates file as *data* for exactly this
//! reason.
//!
//! # The grammar
//!
//! Frames are chosen by the label's [`Form`], and adjective slots are filled from
//! `vocab.toml`'s `synonyms` — which came from verified upstream sources, so the generated
//! phrase inherits ADR-0001's "NOT invented" property instead of inventing vocabulary.
//! Intensifiers are chosen by the `arousal` the vocabulary already records, because
//! "sharply" suits anger and not sadness.

use syrinx_cue::instruct::{indefinite_article, Form, InstructTable};
use syrinx_cue::vocab::{Entry, Vocab};

/// Above this arousal a label takes an intensifier of energy rather than of depth.
///
/// 0.55 sits in the gap between the vocabulary's two clusters (`angry`/`excited` 0.90 vs
/// `sad` 0.25, `calm` 0.15) rather than at a round number chosen for looking like one.
const AROUSAL_SPLIT: f32 = 0.55;

/// Above this valence a high-arousal label is *bright* rather than *sharp*.
///
/// Arousal alone is not the axis, and the first version of this file proved it by emitting
/// "fiercely happy" — `happy` is arousal 0.70, above the split, but valence 0.90. English
/// intensifiers track both: high-and-negative is sharp, high-and-positive is bright.
const VALENCE_SPLIT: f32 = 0.5;

/// High arousal, negative valence: `angry` 0.90/0.10, `afraid` 0.85/0.15.
const SHARP_ADVERBS: [&str; 2] = ["sharply", "intensely"];
/// High arousal, positive valence: `happy` 0.70/0.90, `excited` 0.90/0.85.
const BRIGHT_ADVERBS: [&str; 2] = ["brightly", "openly"];
/// Low arousal, either valence: `sad` 0.25, `calm` 0.15, `soft` 0.20.
const DEEP_ADVERBS: [&str; 2] = ["deeply", "quietly"];

/// Which intensifier family a label takes.
///
/// Styles carry no valence (`vocab.toml` gives it only to emotions), and the two adjective
/// styles — `soft` 0.20, `warm` 0.35 — are both low-arousal, so the absent case never
/// reaches the valence test. It falls to the depth family if it ever does, which is the
/// safe direction: "deeply X" is odd at worst, "fiercely X" is wrong.
fn adverbs_for(e: &Entry) -> [&'static str; 2] {
    match (e.arousal > AROUSAL_SPLIT, e.valence) {
        (true, Some(v)) if v >= VALENCE_SPLIT => BRIGHT_ADVERBS,
        (true, Some(_)) => SHARP_ADVERBS,
        (true, None) => SHARP_ADVERBS,
        (false, _) => DEEP_ADVERBS,
    }
}

/// Candidate phrasings for one label, in a stable order.
///
/// Empty when the label is unknown, is an event, or has [`Form::Unspecified`] — an inert
/// default beats a plausible wrong one.
pub fn candidates_for(label: &str, vocab: &Vocab, table: &InstructTable) -> Vec<String> {
    let Some(form) = table.form(label) else { return Vec::new() };
    let Some((_, entry)) = vocab.by_id(label) else { return Vec::new() };

    // Non-adjective frames need the label's NOUN PHRASE, not its id: `fast` means "a
    // hurried delivery" and `elderly` means "an elderly person". A row that declares a
    // non-adjective form without one generates nothing rather than guessing.
    let noun = table.noun(label);
    let mut out = match (form, noun) {
        (Form::Unspecified, _) => Vec::new(),
        (Form::Adjective, _) => adjective_frames(label, entry),
        (_, None) => Vec::new(),
        (Form::Manner, Some(n)) => manner_frames(n),
        (Form::Persona, Some(n)) => persona_frames(n),
        (Form::With, Some(n)) => with_frames(n),
    };
    out.sort();
    out.dedup();
    out
}

/// Synonyms other than the id itself, longest-first so the more specific word leads.
fn other_synonyms(label: &str, e: &Entry) -> Vec<String> {
    let mut v: Vec<String> = e
        .synonyms
        .iter()
        .filter(|s| s.as_str() != label && !s.contains('_') && !s.contains(' '))
        .cloned()
        .collect();
    v.sort();
    v
}

fn adjective_frames(label: &str, e: &Entry) -> Vec<String> {
    let syns = other_synonyms(label, e);
    let adverbs = adverbs_for(e);
    let art = indefinite_article(label);

    let mut v = vec![
        format!("Speak as if you are genuinely {label}"),
        format!("Say this the way someone who is {label} would say it"),
    ];
    for adv in adverbs {
        // "a deeply sad tone" — the article agrees with the ADVERB here, not the label.
        v.push(format!("Speak in {} {adv} {label} tone", indefinite_article(adv)));
    }
    // Pair with a synonym, which is what the curated phrases already do
    // ("Speak in a sad, sorrowful tone" is exactly sad + sorrowful).
    if let Some(s) = syns.first() {
        v.push(format!("Speak in {art} {label}, {s} tone"));
    }
    v
}

/// No adjective slot exists for these: "a whisper tone" is not English. Every frame is
/// imperative and takes the noun phrase.
///
/// **Synonyms are deliberately NOT used here.** A label's synonyms are not guaranteed to
/// share its part of speech — `whisper`'s include `hushed`, an adjective — and filling a
/// noun slot from them produced "Say this in a hushed".
fn manner_frames(noun: &str) -> Vec<String> {
    let art = indefinite_article(noun);
    vec![
        format!("Say this in {art} {noun}"),
        format!("Use {art} {noun} throughout"),
    ]
}

fn persona_frames(noun: &str) -> Vec<String> {
    let art = indefinite_article(noun);
    vec![
        format!("Speak like {art} {noun}"),
        format!("Speak the way {art} {noun} would"),
        format!("Use the voice of {art} {noun}"),
    ]
}

fn with_frames(noun: &str) -> Vec<String> {
    let art = indefinite_article(noun);
    vec![
        format!("Speak with {art} {noun}"),
        format!("Speak throughout with {art} {noun}"),
    ]
}

/// Is this phrase well formed enough to render?
///
/// Deliberately mechanical: it catches the defect class that actually occurred — a label
/// dropped into a slot its part of speech does not fit — and makes no attempt to judge
/// fluency, which is not testable and is the human's job at the accept step.
pub fn is_well_formed(p: &str) -> Result<(), String> {
    if p.trim() != p || p.is_empty() {
        return Err("empty or untrimmed".into());
    }
    if p.contains('_') {
        return Err(format!("snake_case leaked: {p:?}"));
    }
    if p.contains('\n') {
        return Err("multi-line".into());
    }
    for bad in ['[', ']', '<', '>', '|'] {
        if p.contains(bad) {
            return Err(format!("contains {bad:?}"));
        }
    }
    // Article agreement, on every "a"/"an" in the phrase.
    let w: Vec<&str> = p.split_whitespace().collect();
    for i in 0..w.len().saturating_sub(1) {
        let (art, next) = (w[i].trim_matches(','), w[i + 1].trim_matches(','));
        if art.eq_ignore_ascii_case("a") || art.eq_ignore_ascii_case("an") {
            let want = indefinite_article(&next.to_ascii_lowercase());
            if !art.eq_ignore_ascii_case(want) {
                return Err(format!("article disagreement: {art:?} before {next:?} in {p:?}"));
            }
        }
    }
    Ok(())
}
