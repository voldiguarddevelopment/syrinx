//! The instruct phrase table (`crates/syrinx-cue/instruct.toml`) and the fallback.
//!
//! A backend with `Inline::None` — every Qwen3-TTS checkpoint — has exactly one expressive
//! channel: an utterance-scoped instruction in prose. `hoist::instruct_for` produces it.
//! Before this test existed it produced, for 38 of the vocabulary's 51 labels:
//!
//!     [anxious]       -> "Speak in a anxious tone"
//!     [authoritative] -> "Speak in a authoritative tone"
//!     [narrator]      -> "Speak in a narrator tone"
//!     [quick_breath]  -> "Speak in a quick_breath tone"
//!
//! — because the curated table was keyed on legacy CosyVoice tag names while the lookup
//! used canonical `vocab.toml` ids. Nine had the wrong article and several were not
//! delivery instructions at all. That string is a prompt to an LLM-conditioned TTS, so it
//! is not cosmetic.
//!
//! What is pinned here: every emotion and style label has an authored phrase in BOTH
//! languages; the phrases are well-formed prose; the fallback that remains for events
//! agrees with its article on both sides of the vowel test; and no phrase can carry cue
//! markup back into a backend (the CLAUDE.md hard invariant — an instruct string is
//! prose that reaches the model, so a stray `[` in it would be exactly the leak the
//! invariant forbids).

use syrinx_cue::hoist::instruct_for;
use syrinx_cue::instruct::{indefinite_article, InstructTable};
use syrinx_cue::ir::{Cue, CueKind};
use syrinx_cue::legacy_emotion::InstructLang;
use syrinx_cue::vocab::{Kind, Vocab};

const LANGS: [(InstructLang, &str); 2] = [(InstructLang::En, "en"), (InstructLang::Zh, "zh")];

fn cue(kind: CueKind) -> Cue {
    Cue { kind, raw: String::new(), span: 0..0, source: 0..0 }
}

fn cue_of(kind: Kind, label: &str) -> Cue {
    cue(match kind {
        Kind::Emotion => CueKind::Emotion { label: label.into(), intensity: 1.0 },
        Kind::Style => CueKind::Style { label: label.into() },
        Kind::Event => CueKind::Event { label: label.into() },
    })
}

// ---------------------------------------------------------------- coverage

/// The defect itself: every emotion and style label must have an authored phrase. This is
/// the assertion that would have failed before the fix, for 25 of 38 labels.
#[test]
fn every_emotion_and_style_label_has_a_curated_phrase_in_both_languages() {
    let v = Vocab::embedded().expect("vocab");
    let t = InstructTable::shared();
    let mut missing = Vec::new();
    for (kind, e) in v.iter() {
        if matches!(kind, Kind::Event) {
            continue;
        }
        for (lang, name) in LANGS {
            if t.phrase(&e.id, lang).is_none() {
                missing.push(format!("{kind:?}/{}/{name}", e.id));
            }
        }
    }
    assert!(missing.is_empty(), "labels with no curated instruct phrase: {missing:?}");
}

/// The other side of that boundary: events are deliberately NOT in the table. "Speak in a
/// cough tone" is not an instruction, and `pass_hoist` filters events out before
/// `instruct_for` on every backend that cannot express them. If someone adds one, this
/// fails and they have to justify it.
#[test]
fn event_labels_are_deliberately_absent_from_the_table() {
    let v = Vocab::embedded().expect("vocab");
    let t = InstructTable::shared();
    let mut present = Vec::new();
    for (kind, e) in v.iter() {
        if matches!(kind, Kind::Event) && t.phrase(&e.id, InstructLang::En).is_some() {
            present.push(e.id.clone());
        }
    }
    assert!(present.is_empty(), "event labels must not carry instruct phrases: {present:?}");
}

/// The table covers exactly the non-event vocabulary — no orphan rows for labels that do
/// not exist, which would be dead prose nobody would ever notice.
#[test]
fn the_table_has_no_rows_for_labels_outside_the_vocabulary() {
    let v = Vocab::embedded().expect("vocab");
    let orphans: Vec<&str> = InstructTable::shared()
        .labels()
        .filter(|l| v.by_id(l).is_none())
        .collect();
    assert!(orphans.is_empty(), "instruct.toml rows with no vocab entry: {orphans:?}");
}

// ---------------------------------------------------------------- well-formedness

/// Every phrase in both languages must be non-empty, single-line, and free of cue markup.
#[test]
fn every_phrase_is_well_formed_prose() {
    let t = InstructTable::shared();
    for label in t.labels() {
        for (lang, name) in LANGS {
            let p = t.phrase(label, lang).expect("covered");
            assert!(!p.trim().is_empty(), "{label}/{name}: empty");
            assert_eq!(p.trim(), p, "{label}/{name}: untrimmed whitespace: {p:?}");
            assert!(!p.contains('\n'), "{label}/{name}: multi-line: {p:?}");
            // The hard invariant, applied to the one string that reaches a model as prose.
            for bad in ['[', ']', '<', '>', '|'] {
                assert!(!p.contains(bad), "{label}/{name}: contains {bad:?}: {p:?}");
            }
            assert!(
                !p.contains("endofprompt"),
                "{label}/{name}: carries a prompt delimiter: {p:?}"
            );
        }
    }
}

/// English phrases must read as instructions, not as a label pasted into a frame. The
/// specific regression: a phrase must never contain its own snake_case id.
#[test]
fn english_phrases_are_instructions_not_pasted_labels() {
    let t = InstructTable::shared();
    for label in t.labels() {
        let p = t.phrase(label, InstructLang::En).expect("covered");
        assert!(!p.contains('_'), "{label}: snake_case leaked into the phrase: {p:?}");
        assert!(
            p.starts_with("Speak") || p.starts_with("Say") || p.starts_with("Shout")
                || p.starts_with("Scream"),
            "{label}: does not read as an instruction: {p:?}"
        );
    }
}

/// The two languages must be genuinely different strings — a copy-paste that left English
/// in the `zh` column would otherwise pass every test above.
#[test]
fn the_two_languages_are_distinct_and_zh_is_actually_chinese() {
    let t = InstructTable::shared();
    for label in t.labels() {
        let (en, zh) = (
            t.phrase(label, InstructLang::En).expect("covered"),
            t.phrase(label, InstructLang::Zh).expect("covered"),
        );
        assert_ne!(en, zh, "{label}: en and zh are the same string");
        assert!(
            zh.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)),
            "{label}: zh phrase has no Han characters: {zh:?}"
        );
        assert!(
            !en.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)),
            "{label}: en phrase contains Han characters: {en:?}"
        );
    }
}

// ---------------------------------------------------------------- the fallback

/// Both sides of the vowel boundary, which is the comparison the fallback turns on.
#[test]
fn indefinite_article_agrees_on_both_sides_of_the_vowel_test() {
    for vowel in ["anxious", "elderly", "impatient", "octave", "uhm"] {
        assert_eq!(indefinite_article(vowel), "an", "{vowel}");
    }
    for consonant in ["cough", "sigh", "breath", "groan", "tsk"] {
        assert_eq!(indefinite_article(consonant), "a", "{consonant}");
    }
    // The degenerate input, so the match arm's fall-through is pinned too.
    assert_eq!(indefinite_article(""), "a");
}

/// The fallback fires only for labels with no curated phrase, and when it fires the
/// article agrees. Both branches asserted, so neither can be mutated away.
#[test]
fn the_fallback_fires_only_for_uncurated_labels_and_agrees_with_its_article() {
    // Curated: the table wins, and the fallback frame does NOT appear.
    let got = instruct_for(&cue_of(Kind::Emotion, "anxious"), InstructLang::En).expect("total");
    assert_eq!(got, "Speak in an anxious, uneasy tone");
    assert!(!got.contains("Speak in a anxious"));

    // Uncurated (an event): the fallback fires, with the right article on both sides.
    let vowel = instruct_for(&cue_of(Kind::Event, "uhm"), InstructLang::En).expect("total");
    assert_eq!(vowel, "Speak in an uhm tone");
    let consonant = instruct_for(&cue_of(Kind::Event, "cough"), InstructLang::En).expect("total");
    assert_eq!(consonant, "Speak in a cough tone");
}

/// **The wiring, not just the table.** The coverage test above asserts `instruct.toml` has
/// a row for every label; this asserts `instruct_for` actually RETURNS it. Without this,
/// disconnecting the lookup and falling back to the generic frame for all 38 labels — the
/// exact pre-fix state — leaves the coverage test green, which was true of the first draft
/// of this file and was caught by reverting the fix and watching only one assertion move.
#[test]
fn instruct_for_returns_the_curated_phrase_for_every_covered_label() {
    let v = Vocab::embedded().expect("vocab");
    let t = InstructTable::shared();
    for (kind, e) in v.iter() {
        if matches!(kind, Kind::Event) {
            continue;
        }
        for (lang, name) in LANGS {
            let want = t.phrase(&e.id, lang).expect("covered");
            let got = instruct_for(&cue_of(kind, &e.id), lang).expect("total");
            assert_eq!(got, want, "{kind:?}/{}/{name}: instruct_for ignored the table", e.id);
        }
    }
}

/// `instruct_for` is documented as total. Every vocabulary label, in both languages, must
/// return something non-empty — the property the fallback exists to guarantee.
#[test]
fn instruct_for_is_total_over_the_whole_vocabulary() {
    let v = Vocab::embedded().expect("vocab");
    for (kind, e) in v.iter() {
        for (lang, name) in LANGS {
            let got = instruct_for(&cue_of(kind, &e.id), lang);
            let got = got.unwrap_or_else(|| panic!("{kind:?}/{}/{name}: returned None", e.id));
            assert!(!got.trim().is_empty(), "{kind:?}/{}/{name}: empty", e.id);
        }
    }
}

/// Free text is passed through untouched, and empty free text yields nothing. Both sides
/// of that emptiness test.
#[test]
fn free_text_passes_through_and_empty_free_text_yields_none() {
    let mut c = cue(CueKind::Free);
    c.raw = "  Speak like a tired lighthouse keeper  ".into();
    assert_eq!(
        instruct_for(&c, InstructLang::En).as_deref(),
        Some("Speak like a tired lighthouse keeper")
    );

    let mut empty = cue(CueKind::Free);
    empty.raw = "   ".into();
    assert_eq!(instruct_for(&empty, InstructLang::En), None);
}

/// Language selection actually selects. Asserted on a label whose two phrasings differ, in
/// both directions, so a mutant that ignores `lang` dies.
#[test]
fn the_language_argument_selects_the_language() {
    let c = cue_of(Kind::Emotion, "angry");
    assert_eq!(instruct_for(&c, InstructLang::En).as_deref(), Some("Speak in an angry tone"));
    assert_eq!(instruct_for(&c, InstructLang::Zh).as_deref(), Some("用愤怒生气的语气说"));
}

// ---------------------------------------------------------------- loading

/// A table with an empty phrase is rejected rather than silently shipping a blank
/// instruction — both languages checked, since they are separate branches.
#[test]
fn an_empty_phrase_is_a_load_error_in_either_language() {
    let ok = r#"[emotion]
happy = { en = "Speak happily", zh = "开心地说" }"#;
    assert!(InstructTable::from_toml(ok).is_ok());

    let blank_en = r#"[emotion]
happy = { en = "  ", zh = "开心地说" }"#;
    assert!(InstructTable::from_toml(blank_en).is_err(), "empty en must be rejected");

    let blank_zh = r#"[emotion]
happy = { en = "Speak happily", zh = "" }"#;
    assert!(InstructTable::from_toml(blank_zh).is_err(), "empty zh must be rejected");
}

/// The embedded table loads, and `shared()` returns the same content as a fresh parse.
#[test]
fn the_embedded_table_loads_and_is_shared() {
    let fresh = InstructTable::embedded().expect("embedded table must parse");
    assert_eq!(&fresh, InstructTable::shared());
    assert!(!fresh.is_empty());
    assert_eq!(fresh.len(), 38, "25 emotions + 13 styles");
}

// ---------------------------------------------------------------- tuned rows (ADR-0004)

/// **The human gate.** A tuned row without a signature is inert: parsed, validated, and
/// then not returned by any lookup. This is the single assertion standing between "the
/// loop proposed a phrase" and "the product says it", and CLAUDE.md is explicit that
/// intended emotion is not expressible as a frozen-test gate.
#[test]
fn an_unsigned_tuned_row_is_inert() {
    const ROW: &str = r#"
[emotion]
angry = { en = "Speak in an angry tone", zh = "用愤怒生气的语气说" }

[[tuned]]
label = "angry"
lang = "en"
backend = "qwen3-1.7b-customvoice"
phrase = "TUNED PHRASE"
measured_on = "2026-09-06"
incumbent = "Speak in an angry tone"
margin = 3.0
judge = "emotion2vec+ large"
judge_recall_on_class = 1.0
holdout_id = "h1"
holdout_uses = 0
"#;
    let t = InstructTable::from_toml(ROW).expect("must parse");
    assert_eq!(t.tuned_rows().len(), 1, "the row is present in the file");
    assert_eq!(t.accepted_tuned(), 0, "but it is not live");
    assert_eq!(
        t.phrase_for_backend("angry", InstructLang::En, "qwen3-1.7b-customvoice"),
        Some("Speak in an angry tone"),
        "an unsigned row must not shadow the curated phrase"
    );

    // The same row, signed, IS live. Both sides of the gate.
    let signed = ROW.replace("holdout_uses = 0", "holdout_uses = 0\naccepted_by = \"floofy\"");
    let t2 = InstructTable::from_toml(&signed).expect("must parse");
    assert_eq!(t2.accepted_tuned(), 1);
    assert_eq!(
        t2.phrase_for_backend("angry", InstructLang::En, "qwen3-1.7b-customvoice"),
        Some("TUNED PHRASE")
    );

    // An empty signature is not a signature.
    let blank = ROW.replace("holdout_uses = 0", "holdout_uses = 0\naccepted_by = \"  \"");
    assert_eq!(InstructTable::from_toml(&blank).expect("parses").accepted_tuned(), 0);
}

/// A tuned row is scoped to one backend and one language, because instruct semantics are
/// per checkpoint (ADR-0001 §2.3) — CustomVoice's instruct describes delivery, VoiceDesign's
/// describes the voice. A row must not leak across either axis.
#[test]
fn a_tuned_row_does_not_leak_across_backend_or_language() {
    const ROW: &str = r#"
[emotion]
angry = { en = "Speak in an angry tone", zh = "用愤怒生气的语气说" }

[[tuned]]
label = "angry"
lang = "en"
backend = "qwen3-1.7b-customvoice"
phrase = "TUNED"
accepted_by = "floofy"
measured_on = "2026-09-06"
incumbent = "Speak in an angry tone"
margin = 3.0
judge = "emotion2vec+ large"
judge_recall_on_class = 1.0
holdout_id = "h1"
holdout_uses = 0
"#;
    let t = InstructTable::from_toml(ROW).expect("parses");
    // Its own (backend, lang): tuned.
    assert_eq!(
        t.phrase_for_backend("angry", InstructLang::En, "qwen3-1.7b-customvoice"),
        Some("TUNED")
    );
    // A different backend: curated.
    assert_eq!(
        t.phrase_for_backend("angry", InstructLang::En, "qwen3-voicedesign"),
        Some("Speak in an angry tone")
    );
    // A different language: curated.
    assert_eq!(
        t.phrase_for_backend("angry", InstructLang::Zh, "qwen3-1.7b-customvoice"),
        Some("用愤怒生气的语气说")
    );
}

/// Tuned rows are additive: with none present, `phrase_for_backend` is exactly `phrase`.
/// Deleting a tuned row therefore restores the previous behaviour precisely, which is what
/// makes running the loop reversible and therefore safe.
#[test]
fn with_no_tuned_rows_the_backend_lookup_equals_the_curated_lookup() {
    let t = InstructTable::shared();
    assert_eq!(t.accepted_tuned(), 0, "the shipped table has no tuned rows yet");
    for label in t.labels() {
        for lang in [InstructLang::En, InstructLang::Zh] {
            assert_eq!(
                t.phrase_for_backend(label, lang, "qwen3-1.7b-customvoice"),
                t.phrase(label, lang),
                "{label}: additive lookup must fall through to curated"
            );
        }
    }
}

/// An empty tuned phrase is a load error, signed or not — a blank instruction must never
/// be shippable by signing it.
#[test]
fn an_empty_tuned_phrase_is_rejected_even_when_signed() {
    const ROW: &str = r#"
[[tuned]]
label = "angry"
lang = "en"
backend = "b"
phrase = "   "
accepted_by = "floofy"
measured_on = "2026-09-06"
incumbent = "x"
margin = 3.0
judge = "j"
judge_recall_on_class = 1.0
holdout_id = "h1"
holdout_uses = 0
"#;
    assert!(InstructTable::from_toml(ROW).is_err());
}
