//! The candidate grammar's **entire output space**, walked and asserted.
//!
//! This file is the fix. The generator it replaces was four `format!` templates with a
//! `{label}` slot, and 32% of what it could emit was malformed — "Speak in a sharply
//! narrator tone", "Speak in a sharply quick_breath tone". The templates were not subtle;
//! the problem was that **nothing ever asked the generator to show all its output at
//! once**, so the defect was invisible until one phrase happened to appear in a log.
//!
//! Better templates would not prevent a recurrence. Enumeration does. Every test below
//! walks all 51 vocabulary ids rather than sampling, so a frame added later that breaks on
//! one label fails here rather than shipping.

use std::collections::BTreeSet;

use syrinx_eval::candidates::{candidates_for, is_well_formed};
use syrinx_cue::instruct::{Form, InstructTable};
use syrinx_cue::vocab::{Kind, Vocab};

fn setup() -> (Vocab, &'static InstructTable) {
    (Vocab::embedded().expect("vocab"), InstructTable::shared())
}

/// THE test: every candidate for every label is well formed. This is the assertion the old
/// generator would have failed on 48 of its 152 outputs.
#[test]
fn every_candidate_for_every_label_is_well_formed() {
    let (v, t) = setup();
    let mut bad = Vec::new();
    let mut total = 0;
    for (_, e) in v.iter() {
        for c in candidates_for(&e.id, &v, t) {
            total += 1;
            if let Err(why) = is_well_formed(&c) {
                bad.push(format!("{}: {why}", e.id));
            }
        }
    }
    assert!(total > 100, "only {total} candidates generated — the grammar is not doing its job");
    assert!(bad.is_empty(), "{} of {total} candidates malformed:\n  {}", bad.len(), bad.join("\n  "));
}

/// Article agreement across the whole space, stated separately because it is the specific
/// error that shipped ("Speak in a anxious tone") and deserves to fail with its own name.
#[test]
fn no_candidate_anywhere_disagrees_with_its_article() {
    let (v, t) = setup();
    for (_, e) in v.iter() {
        for c in candidates_for(&e.id, &v, t) {
            let w: Vec<&str> = c.split_whitespace().collect();
            for i in 0..w.len().saturating_sub(1) {
                let art = w[i].trim_matches(',').to_ascii_lowercase();
                if art == "a" || art == "an" {
                    let next = w[i + 1].trim_matches(',').to_ascii_lowercase();
                    let vowel = matches!(next.chars().next(), Some('a' | 'e' | 'i' | 'o' | 'u'));
                    assert_eq!(art == "an", vowel, "{}: {c:?}", e.id);
                }
            }
        }
    }
}

/// Events generate nothing. An event is a point sound, not a way of speaking, and the old
/// generator happily produced "Speak in a sharply cough tone" for all thirteen.
#[test]
fn event_labels_generate_no_candidates() {
    let (v, t) = setup();
    for (kind, e) in v.iter() {
        if matches!(kind, Kind::Event) {
            assert!(
                candidates_for(&e.id, &v, t).is_empty(),
                "{} is an event and must generate nothing",
                e.id
            );
        }
    }
}

/// Every non-event label generates something. A form that silently produced nothing would
/// remove a cue from tuning without anyone noticing.
#[test]
fn every_non_event_label_generates_candidates() {
    let (v, t) = setup();
    for (kind, e) in v.iter() {
        if !matches!(kind, Kind::Event) {
            assert!(
                !candidates_for(&e.id, &v, t).is_empty(),
                "{} generates no candidates",
                e.id
            );
        }
    }
}

/// No shipped row may be left `Unspecified`. The default is inert on purpose, and this is
/// what stops "inert" from quietly becoming "38 labels generate nothing".
#[test]
fn no_shipped_row_is_left_without_a_form() {
    let t = InstructTable::shared();
    let missing: Vec<&str> = t
        .labels()
        .filter(|l| t.form(l) == Some(Form::Unspecified) || t.form(l).is_none())
        .collect();
    assert!(missing.is_empty(), "rows with no grammatical form: {missing:?}");
}

/// A non-adjective label must never land in an adjective frame. This is the defect class
/// itself, asserted directly rather than only through its symptoms.
#[test]
fn non_adjective_labels_never_appear_in_an_adjective_frame() {
    let (v, t) = setup();
    for (kind, e) in v.iter() {
        if matches!(kind, Kind::Event) {
            continue;
        }
        let form = t.form(&e.id).expect("covered");
        if form == Form::Adjective {
            continue;
        }
        for c in candidates_for(&e.id, &v, t) {
            assert!(
                !c.contains(&format!(" {} tone", e.id)),
                "{:?} is {form:?}, not an adjective, but got an adjective frame: {c:?}",
                e.id
            );
            for adv in ["sharply", "intensely", "brightly", "openly", "deeply", "quietly"] {
                assert!(
                    !c.contains(&format!("{adv} {}", e.id)),
                    "{:?} is {form:?} but was modified by {adv:?}: {c:?}",
                    e.id
                );
            }
        }
    }
}

/// **The gap that the first version of this file left open.** Enumeration alone is not
/// enough — the first pass asserted article agreement and well-formedness and still let
/// through "Say this in a hushed", "Use a fast throughout" and "Speak like an elderly",
/// because those are article-correct and only *semantically* wrong. Reading the output is
/// what caught them, which means the assertions had only encoded failure modes already
/// seen.
///
/// This encodes the class: a frame ending in a noun slot must not be filled by a bare
/// adjective. It is checked structurally — every non-adjective frame must use the row's
/// declared `noun`, never the label id, unless the two are the same word.
#[test]
fn non_adjective_frames_use_the_declared_noun_and_never_a_bare_adjective() {
    let (v, t) = setup();
    for (kind, e) in v.iter() {
        if matches!(kind, Kind::Event) {
            continue;
        }
        let form = t.form(&e.id).expect("covered");
        if form == Form::Adjective {
            continue;
        }
        let noun = t
            .noun(&e.id)
            .unwrap_or_else(|| panic!("{} is {form:?} but declares no noun", e.id));
        for c in candidates_for(&e.id, &v, t) {
            assert!(
                c.contains(noun),
                "{}: {form:?} frame does not use its declared noun {noun:?}: {c:?}",
                e.id
            );
            // Every occurrence of the label must lie INSIDE an occurrence of the noun
            // phrase. Stated that way after two weaker spellings each rejected correct
            // English: `ends_with` rejected "Speak like a young child" (the label is the
            // head of its own noun phrase) and article-adjacency rejected "Speak like an
            // elderly person" (the noun phrase begins with the label). Deleting the noun
            // phrase and asking whether the label survives is exact.
            if noun != e.id {
                let residue = c.replace(noun, "");
                assert!(
                    !residue.contains(&e.id),
                    "{}: the label appears outside the declared noun phrase {noun:?}: {c:?}",
                    e.id
                );
            }
        }
    }
}

/// No **non-adjective** frame may end on a word that is only ever an adjective. Restricted
/// to non-adjective forms deliberately: "Speak as if you are genuinely soft" ends on an
/// adjective and is correct English, because that frame has a predicate-adjective slot.
/// A first draft of this test omitted the restriction and rejected it.
#[test]
fn no_non_adjective_frame_ends_on_a_bare_adjective() {
    let (v, t) = setup();
    const ADJ_ONLY: [&str; 8] =
        ["hushed", "loud", "fast", "quick", "elderly", "soft", "gentle", "accented"];
    for (kind, e) in v.iter() {
        if matches!(kind, Kind::Event) || t.form(&e.id) == Some(Form::Adjective) {
            continue;
        }
        for c in candidates_for(&e.id, &v, t) {
            let last = c.rsplit(' ').next().unwrap_or("").trim_matches(',');
            assert!(
                !ADJ_ONLY.contains(&last),
                "{}: candidate ends on the bare adjective {last:?}: {c:?}",
                e.id
            );
        }
    }
}

/// Generation is deterministic and stable — the same inputs give the same list in the same
/// order. A tuning round's provenance is meaningless if the candidate set drifts.
#[test]
fn generation_is_deterministic_and_ordered() {
    let (v, t) = setup();
    for (_, e) in v.iter() {
        let a = candidates_for(&e.id, &v, t);
        assert_eq!(a, candidates_for(&e.id, &v, t), "{}", e.id);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(a, sorted, "{}: not in stable order", e.id);
        let uniq: BTreeSet<&String> = a.iter().collect();
        assert_eq!(uniq.len(), a.len(), "{}: duplicate candidates", e.id);
    }
}

/// Intensifiers track BOTH axes. All three bands asserted, and each excluded from the
/// others, so a mutant that drops either test dies.
///
/// The `happy` case is why the valence axis exists at all: arousal 0.70 puts it above the
/// split, and the first version of this grammar emitted "fiercely happy" as a result.
#[test]
fn intensifiers_follow_arousal_and_valence() {
    let (v, t) = setup();
    let joined = |l: &str| candidates_for(l, &v, t).join(" | ");
    let has = |s: &str, set: [&str; 2]| set.iter().any(|w| s.contains(w));

    const SHARP: [&str; 2] = ["sharply", "intensely"];
    const BRIGHT: [&str; 2] = ["brightly", "openly"];
    const DEEP: [&str; 2] = ["deeply", "quietly"];

    // high arousal, negative valence
    let angry = joined("angry"); // 0.90 / 0.10
    assert!(has(&angry, SHARP), "{angry}");
    assert!(!has(&angry, BRIGHT) && !has(&angry, DEEP), "{angry}");

    // high arousal, POSITIVE valence -- the case that motivated the second axis
    let happy = joined("happy"); // 0.70 / 0.90
    assert!(has(&happy, BRIGHT), "{happy}");
    assert!(!has(&happy, SHARP), "\"fiercely happy\" was the bug: {happy}");

    // low arousal
    let sad = joined("sad"); // 0.25 / 0.10
    assert!(has(&sad, DEEP), "{sad}");
    assert!(!has(&sad, SHARP) && !has(&sad, BRIGHT), "{sad}");
}

/// The specific strings that motivated this file must not be reachable from anywhere.
#[test]
fn the_phrases_that_prompted_this_are_unreachable() {
    let (v, t) = setup();
    let all: Vec<String> = v.iter().flat_map(|(_, e)| candidates_for(&e.id, &v, t)).collect();
    for banned in [
        "Speak in a sharply sad tone",
        "Speak in a fiercely happy tone",
        "Say this in a hushed",
        "Say this in a loud",
        "Use a fast throughout",
        "Speak like an elderly",
        "Speak in a sharply narrator tone",
        "Speak in a sharply whisper tone",
        "Speak in a sharply child tone",
        "Speak in a sharply elderly tone",
        "Speak in a sharply quick_breath tone",
    ] {
        assert!(!all.iter().any(|c| c == banned), "still reachable: {banned:?}");
    }
}

/// `is_well_formed` has teeth — it must reject what it claims to.
#[test]
fn the_well_formedness_check_rejects_what_it_claims_to() {
    for bad in [
        "Speak in a anxious tone",
        "Speak in a quick_breath tone",
        "say [happy] now",
        "two\nlines",
        " padded",
        "",
    ] {
        assert!(is_well_formed(bad).is_err(), "{bad:?} must be rejected");
    }
    for ok in ["Speak in an angry tone", "Speak like a narrator", "Say this in a whisper"] {
        assert!(is_well_formed(ok).is_ok(), "{ok:?}: {:?}", is_well_formed(ok));
    }
}

/// Not an assertion — a way to READ the whole space, which is how three defects the
/// assertions missed were actually found ("Say this in a hushed", "Use a fast throughout",
/// "Speak like an elderly"). Ignored by default so a test that cannot fail never sits on
/// the board pretending to be a gate:
///
///     cargo test --test cue_candidate_grammar show_all -- --ignored --nocapture
#[test]
#[ignore = "a viewer, not a gate — run explicitly to read the generated space"]
fn show_all() {
    let (v, t) = setup();
    for (kind, e) in v.iter() {
        let c = candidates_for(&e.id, &v, t);
        if c.is_empty() {
            continue;
        }
        eprintln!("{:?}/{} [{:?}] arousal {:.2}", kind, e.id, t.form(&e.id).unwrap(), e.arousal);
        for x in c {
            eprintln!("    {x}");
        }
    }
}
