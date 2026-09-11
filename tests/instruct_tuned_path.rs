//! The `[[tuned]]` row path, end to end — frozen.
//!
//! ADR-0004 defines a storage mechanism for phrasings found by the tuning loop: their own
//! table in `crates/syrinx-cue/instruct.toml`, a mandatory provenance block, a human
//! `accepted_by` signature that alone makes a row live, a strictly additive
//! tuned → curated → deterministic-fallback lookup, and scoping per checkpoint and per
//! language. The shipped file has **zero** `[[tuned]]` rows and must keep having zero —
//! signing a phrase is a human act and no test may forge one — so until this file existed
//! the mechanism had never run against a realistic row.
//!
//! Everything here uses FIXTURE rows built inside the test. Nothing is written to a shipped
//! file, and the one signature that appears is `"fixture — not a real acceptance"`, which
//! is deliberately not a person.
//!
//! # What driving it end to end found
//!
//! **The lookup was connected to nothing.** `InstructTable::phrase_for_backend` — the
//! tuned → curated tier — was called from tests and from no production code. The pipeline
//! resolved every prefix through `hoist::instruct_for`, which is backend-blind, so a row a
//! human had signed would have stayed inert with no diagnostic. ADR-0004 §2's three-tier
//! order did not exist anywhere: tiers 1 and 2 were in `instruct.rs`, tier 3 was in
//! `hoist.rs`, and nothing joined them. Fixed 2026-09-11 by `instruct_with` /
//! `pass_hoist_with`, which resolve against `caps.id` — the per-checkpoint key ADR-0004 §1
//! writes a row under.
//!
//! **Provenance was accepted blank or absurd.** `judge = ""`, `measured_on = ""`,
//! `judge_recall_on_class = 3.0`, `margin = -1.0` and a `lang` no lookup can produce all
//! loaded silently. §1 makes those fields mandatory *because* a verdict quoted without them
//! is not a verdict; a blank one defeats the requirement while satisfying serde.
//!
//! **A tuned phrase was exempt from the hard invariant.** The frozen well-formedness test
//! walks the CURATED rows only, so a signed tuned row containing `[sad]` would have been
//! handed to a backend as literal text — the leak CLAUDE.md calls a release blocker, from
//! the one source that no human reads before it ships.
//!
//! **Two accepted rows for one key were resolved by row order.** Silently.

use syrinx_cue::caps::BackendId;
use syrinx_cue::hoist::{instruct_with, pass_hoist_with, SplitOptions, UtteranceSegment};
use syrinx_cue::instruct::{
    known_lang, phrase_is_safe, InstructTable, TunedPhrase, EMBEDDED_INSTRUCT,
};
use syrinx_cue::ir::{Cue, CueKind};
use syrinx_cue::legacy_emotion::InstructLang;
use syrinx_cue::lower::{lower, LoweringReport};
use syrinx_cue::parse::ParseOptions;
use syrinx_cue::vocab::{Kind, Vocab};
use syrinx_cue::parse;

/// The fixture's key. `qwen3-1.7b-customvoice` is the checkpoint every tuning round has
/// actually run on, and `sad` is the label of the round that motivated this work.
const BACKEND: &str = "qwen3-1.7b-customvoice";
const LABEL: &str = "sad";
/// The curated English phrase for `sad`, i.e. what the incumbent is.
const CURATED_EN: &str = "Speak in a sad, sorrowful tone";
const CURATED_ZH: &str = "用悲伤难过的语气说";
/// A plausible winner from the 2026-09-09 `[sad]` grammar. It did not win; it is a fixture.
const TUNED_EN: &str = "Speak in a quietly sad tone";
/// Not a person. A test must never write something that reads like a real acceptance.
const FIXTURE_SIG: &str = "\"fixture — not a real acceptance\"";

// ---------------------------------------------------------------- fixtures

/// A realistic proposal row, in the exact shape `examples/tune_instruct.rs` writes.
///
/// An override with an empty value REMOVES the field, which is how the missing-field cases
/// are built; any other value is spliced in as raw TOML, so a caller can write a string, a
/// number or `nan`.
fn tuned_row(overrides: &[(&str, &str)]) -> String {
    let mut f: Vec<(&str, String)> = vec![
        ("label", format!("{LABEL:?}")),
        ("lang", "\"en\"".into()),
        ("backend", format!("{BACKEND:?}")),
        ("phrase", format!("{TUNED_EN:?}")),
        ("measured_on", "\"2026-09-09\"".into()),
        ("incumbent", format!("{CURATED_EN:?}")),
        ("margin", "1.42".into()),
        ("judge", "\"emotion2vec/emotion2vec_plus_large\"".into()),
        // The measured CREMA-D recall on `sad` — see renders/2026-09-06-sad-sentences/.
        ("judge_recall_on_class", "0.867".into()),
        ("holdout_id", "\"sad-2026-09-09-a\"".into()),
        ("holdout_uses", "0".into()),
        ("notes", "\"fixture row defined inside tests/instruct_tuned_path.rs\"".into()),
    ];
    for (k, v) in overrides {
        if v.is_empty() {
            f.retain(|(n, _)| n != k);
        } else if let Some(e) = f.iter_mut().find(|(n, _)| n == k) {
            e.1 = (*v).to_string();
        } else {
            f.push((k, (*v).to_string()));
        }
    }
    let mut s = String::from("\n[[tuned]]\n");
    for (k, v) in f {
        s.push_str(&format!("{k} = {v}\n"));
    }
    s
}

/// The signed variant of [`tuned_row`].
fn signed_row(overrides: &[(&str, &str)]) -> String {
    let mut o = overrides.to_vec();
    o.push(("accepted_by", FIXTURE_SIG));
    tuned_row(&o)
}

/// The SHIPPED curated table with fixture rows appended — the realistic shape, rather than
/// a two-line table that would not exercise fallthrough to 38 real curated phrases.
fn table_of(rows: &str) -> Result<InstructTable, syrinx_cue::instruct::InstructError> {
    InstructTable::from_toml(&format!("{EMBEDDED_INSTRUCT}\n{rows}"))
}

fn table(rows: &str) -> InstructTable {
    table_of(rows).unwrap_or_else(|e| panic!("fixture must load: {e}"))
}

fn cue_of(kind: Kind, label: &str) -> Cue {
    let kind = match kind {
        Kind::Emotion => CueKind::Emotion { label: label.into(), intensity: 1.0 },
        Kind::Style => CueKind::Style { label: label.into() },
        Kind::Event => CueKind::Event { label: label.into() },
    };
    Cue { kind, raw: String::new(), span: 0..0, source: 0..0 }
}

/// The WHOLE path: source text → parse → lower → hoist, against an explicit table.
fn hoist(
    src: &str,
    id: BackendId,
    lang: InstructLang,
    t: &InstructTable,
) -> Vec<UtteranceSegment> {
    let v = Vocab::embedded().expect("vocab");
    let d = parse(src, &v, &ParseOptions::default());
    let caps = id.caps().expect("caps");
    let low = lower(&d, &caps, &v);
    let mut rep: LoweringReport = low.report.clone();
    let opts = SplitOptions { lang, ..Default::default() };
    pass_hoist_with(&low, &caps, &opts, t, &mut rep)
}

fn prefix(src: &str, id: BackendId, lang: InstructLang, t: &InstructTable) -> Option<String> {
    hoist(src, id, lang, t).first().and_then(|s| s.instruct.clone())
}

// ---------------------------------------------------------------- the shipped file

/// **The shipped file carries no tuned rows, and no test may add one.** Signing a phrase is
/// a human act (ADR-0004 §3), and a fixture that leaked into `instruct.toml` would be a
/// forged signature with a provenance block to make it look researched.
#[test]
fn the_shipped_instruct_table_has_no_tuned_rows() {
    let t = InstructTable::shared();
    assert_eq!(t.tuned_rows().len(), 0, "crates/syrinx-cue/instruct.toml must have no [[tuned]]");
    assert_eq!(t.accepted_tuned(), 0);
    assert!(
        !EMBEDDED_INSTRUCT.contains("accepted_by"),
        "no shipped file may contain an acceptance signature"
    );
}

// ---------------------------------------------------------------- the complete path

/// **The headline.** An accepted row reaches the segment prefix through parse → lower →
/// hoist, and the same pipeline against the shipped table says the curated phrase.
///
/// This is the assertion that was impossible before 2026-09-11: the pipeline never
/// consulted a tuned row, so the two halves of this test were equal.
#[test]
fn an_accepted_row_reaches_the_segment_prefix_through_the_whole_pipeline() {
    let src = "[sad] I waited by the window until the last light went out.";

    let tuned = table(&signed_row(&[]));
    assert_eq!(tuned.accepted_tuned(), 1);
    assert_eq!(
        prefix(src, BackendId::Qwen17bCustomVoice, InstructLang::En, &tuned).as_deref(),
        Some(TUNED_EN),
        "an accepted row must reach the backend prefix"
    );

    assert_eq!(
        prefix(src, BackendId::Qwen17bCustomVoice, InstructLang::En, InstructTable::shared())
            .as_deref(),
        Some(CURATED_EN),
        "and the shipped table must still say the curated phrase"
    );
}

/// The prefix a tuned row produces is still a prefix on a real segment with real text —
/// the split machinery is unaffected, and the instruction is the only thing that moved.
#[test]
fn a_tuned_prefix_does_not_disturb_the_segmentation() {
    let src = "[happy] good morning [sad] but not for long";
    let tuned = table(&signed_row(&[]));

    let plain = hoist(src, BackendId::Qwen17bCustomVoice, InstructLang::En, InstructTable::shared());
    let with = hoist(src, BackendId::Qwen17bCustomVoice, InstructLang::En, &tuned);

    assert_eq!(with.len(), plain.len(), "the number of requests must not change");
    let texts: Vec<&str> = with.iter().map(|s| s.text.as_str()).collect();
    let plain_texts: Vec<&str> = plain.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, plain_texts, "the text split must be byte-identical");

    assert_eq!(with[0].instruct.as_deref(), Some("Speak in a happy, cheerful tone"));
    assert_eq!(with[1].instruct.as_deref(), Some(TUNED_EN), "only the tuned label moved");
    assert_eq!(plain[1].instruct.as_deref(), Some(CURATED_EN));
}

// ---------------------------------------------------------------- scoping

/// A row shadows its EXACT `(backend, lang, label)` and nothing else. Instruct semantics
/// are per checkpoint (ADR-0001 §2.3) — CustomVoice's instruct describes delivery,
/// VoiceDesign's describes the voice — so a row that leaked across a checkpoint would be
/// applying a measurement made about one thing to another.
///
/// Asserted through the full pipeline, not just the lookup, on all three axes.
#[test]
fn an_accepted_row_shadows_its_exact_key_and_nothing_else() {
    let t = table(&signed_row(&[]));

    // Its own key.
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bCustomVoice, InstructLang::En, &t).as_deref(),
        Some(TUNED_EN)
    );
    // Different backend — a different checkpoint entirely.
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bVoiceDesign, InstructLang::En, &t).as_deref(),
        Some(CURATED_EN)
    );
    // The 0.6B CustomVoice ACCEPTS an instruct string and silently discards it, so
    // `caps.toml` marks emotion `accepted` and lowering drops the cue — a tuned row must
    // not resurrect a cue the capability table says has no effect.
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen06bCustomVoice, InstructLang::En, &t),
        None,
        "a tuned row must not override the capability table"
    );
    // Different language.
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bCustomVoice, InstructLang::Zh, &t).as_deref(),
        Some(CURATED_ZH)
    );
    // Different label.
    assert_eq!(
        prefix("[angry] x", BackendId::Qwen17bCustomVoice, InstructLang::En, &t).as_deref(),
        Some("Speak in an angry tone")
    );
}

/// Exhaustively: an accepted row changes the lookup for exactly one `(backend, lang, label)`
/// triple out of every triple the vocabulary and the caps table can form. A scoping bug
/// that leaked along any axis would show up here as a second differing triple.
#[test]
fn exactly_one_triple_in_the_whole_space_changes() {
    let v = Vocab::embedded().expect("vocab");
    let shipped = InstructTable::shared();
    let t = table(&signed_row(&[]));

    let mut differing = Vec::new();
    for id in BackendId::ALL {
        for lang in [InstructLang::En, InstructLang::Zh] {
            for (kind, e) in v.iter() {
                let c = cue_of(kind, &e.id);
                let a = instruct_with(shipped, &c, lang, Some(id.as_str()));
                let b = instruct_with(&t, &c, lang, Some(id.as_str()));
                if a != b {
                    differing.push((id.as_str(), lang, e.id.clone(), b));
                }
            }
        }
    }
    assert_eq!(differing.len(), 1, "exactly one triple may move: {differing:#?}");
    assert_eq!(differing[0].0, BACKEND);
    assert_eq!(differing[0].1, InstructLang::En);
    assert_eq!(differing[0].2, LABEL);
    assert_eq!(differing[0].3.as_deref(), Some(TUNED_EN));
}

/// Two accepted rows on different backends coexist, each scoped to its own. Without this,
/// a "one accepted row at a time" bug would hide behind every test above.
///
/// VoiceDesign is the right second checkpoint precisely because its instruct means
/// something else — it describes the VOICE, not the delivery — so a row leaking from
/// CustomVoice to it would be applying a delivery measurement to a timbre channel.
#[test]
fn two_rows_on_different_backends_are_independently_scoped() {
    let rows = format!(
        "{}{}",
        signed_row(&[]),
        signed_row(&[("backend", "\"qwen3-1.7b-voicedesign\""), ("phrase", "\"A sad voice\"")])
    );
    let t = table(&rows);
    assert_eq!(t.accepted_tuned(), 2);
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bCustomVoice, InstructLang::En, &t).as_deref(),
        Some(TUNED_EN)
    );
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bVoiceDesign, InstructLang::En, &t).as_deref(),
        Some("A sad voice")
    );
}

// ---------------------------------------------------------------- the human gate

/// **An unsigned row is completely inert** — through the pipeline, not merely in the map.
/// This is the single assertion standing between "the loop proposed a phrase" and "the
/// product says it" (ADR-0004 §3), so it is asserted from both sides on the same row.
#[test]
fn an_unsigned_row_is_inert_through_the_whole_pipeline() {
    let unsigned = table(&tuned_row(&[]));
    assert_eq!(unsigned.tuned_rows().len(), 1, "present in the file");
    assert_eq!(unsigned.accepted_tuned(), 0, "and not live");
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bCustomVoice, InstructLang::En, &unsigned).as_deref(),
        Some(CURATED_EN),
        "an unsigned row must not reach a backend"
    );

    // The same row, signed, does reach it.
    let signed = table(&signed_row(&[]));
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bCustomVoice, InstructLang::En, &signed).as_deref(),
        Some(TUNED_EN)
    );
}

/// A signature that is only whitespace is not a signature, and neither is an empty one.
/// Both sides of the emptiness test, because it is one mutation from accepting everything.
#[test]
fn a_blank_signature_is_not_a_signature() {
    for blank in ["\"\"", "\"   \"", "\"\\t\""] {
        let t = table(&tuned_row(&[("accepted_by", blank)]));
        assert_eq!(t.accepted_tuned(), 0, "{blank} must not sign anything");
        assert_eq!(
            prefix("[sad] x", BackendId::Qwen17bCustomVoice, InstructLang::En, &t).as_deref(),
            Some(CURATED_EN)
        );
    }
    // One non-blank character is a signature. The gate is presence, not plausibility — a
    // machine cannot judge whether a name is a real person's, and pretending otherwise
    // would be a fake gate.
    let t = table(&tuned_row(&[("accepted_by", "\"f\"")]));
    assert_eq!(t.accepted_tuned(), 1);
}

// ---------------------------------------------------------------- additivity / reversibility

/// **Removing a tuned row restores the previous behaviour byte for byte.** ADR-0004 §2
/// rests reversibility on this: it is why running the loop at all is safe.
///
/// Asserted over the entire cross-product — every backend, both languages, every label in
/// the vocabulary — at the pipeline level, comparing whole `UtteranceSegment`s rather than
/// just the prefix string, so a difference in text or attached cues would fail too.
#[test]
fn deleting_a_tuned_row_restores_the_previous_behaviour_exactly() {
    let v = Vocab::embedded().expect("vocab");
    let shipped = InstructTable::shared();
    // The row is present in the file and validated, but unsigned — the state a proposal
    // sits in. It must be indistinguishable from the row not being there at all.
    let with_unsigned = table(&tuned_row(&[]));

    for id in BackendId::ALL {
        for lang in [InstructLang::En, InstructLang::Zh] {
            for (_, e) in v.iter() {
                let src = format!("[{}] the quick brown fox", e.id);
                assert_eq!(
                    hoist(&src, *id, lang, shipped),
                    hoist(&src, *id, lang, &with_unsigned),
                    "{}/{lang:?}/{}: an unsigned row changed behaviour",
                    id.as_str(),
                    e.id
                );
            }
        }
    }
}

/// And with the row deleted from the source text entirely — the literal "remove the row"
/// operation an operator would perform — the table is equal to the shipped one.
#[test]
fn a_table_with_the_row_removed_equals_the_shipped_table() {
    let signed = table(&signed_row(&[]));
    assert_ne!(&signed, InstructTable::shared(), "the fixture must actually differ");

    let removed = table("");
    assert_eq!(&removed, InstructTable::shared(), "removal must restore the shipped table");
}

// ---------------------------------------------------------------- provenance: missing

/// Every provenance field ADR-0004 §1 requires is MANDATORY at load. Omitting one is a load
/// error, not a silent default — the fields are what make a tuned row auditable, and a row
/// that has lost them is indistinguishable from an invented phrase.
#[test]
fn a_missing_provenance_field_is_a_load_error() {
    for field in [
        "label",
        "lang",
        "backend",
        "phrase",
        "measured_on",
        "incumbent",
        "margin",
        "judge",
        "judge_recall_on_class",
        "holdout_id",
        "holdout_uses",
    ] {
        assert!(
            table_of(&signed_row(&[(field, "")])).is_err(),
            "a row with no {field} must not load"
        );
    }
    // The complete row loads, so the assertions above are about the missing field and not
    // about the fixture being broken.
    assert!(table_of(&signed_row(&[])).is_ok());
}

/// Exactly two fields are optional, and deliberately: the signature (its absence is the
/// inert state, ADR-0004 §3) and free-text notes.
#[test]
fn only_the_signature_and_the_notes_are_optional() {
    assert!(table_of(&tuned_row(&[("accepted_by", "")])).is_ok(), "unsigned is a valid state");
    assert!(table_of(&signed_row(&[("notes", "")])).is_ok(), "notes are free text");
}

// ---------------------------------------------------------------- provenance: malformed

/// A blank string in a provenance field is a missing field that got past serde. Each field
/// on both sides: blank rejected, present accepted.
#[test]
fn a_blank_provenance_field_is_a_load_error() {
    for field in ["label", "lang", "backend", "measured_on", "incumbent", "judge", "holdout_id"] {
        assert!(
            table_of(&signed_row(&[(field, "\"\"")])).is_err(),
            "{field} = \"\" must not load"
        );
        assert!(
            table_of(&signed_row(&[(field, "\"   \"")])).is_err(),
            "{field} = whitespace must not load"
        );
    }
    assert!(table_of(&signed_row(&[])).is_ok());
}

/// A `lang` outside the closed set is a row no lookup can ever return, because the key is
/// built from `lang_code`. Silently-unreachable is exactly the failure this validation
/// exists to prevent, so it is an error — and both known codes still load.
#[test]
fn an_unknown_language_is_a_load_error_and_the_known_ones_are_not() {
    assert!(known_lang("en") && known_lang("zh"));
    assert!(!known_lang("fr") && !known_lang("EN") && !known_lang(""));

    for good in ["\"en\"", "\"zh\""] {
        assert!(table_of(&signed_row(&[("lang", good)])).is_ok(), "{good} must load");
    }
    for bad in ["\"fr\"", "\"EN\"", "\"en-GB\""] {
        assert!(table_of(&signed_row(&[("lang", bad)])).is_err(), "{bad} must not load");
    }
}

/// `judge_recall_on_class` is mandatory because a verdict quoted without it is not a
/// verdict (ADR-0004 §1). A value outside [0, 1] is not a recall. Both bounds are asserted
/// AT the boundary and just past it, so `<` cannot become `<=`.
#[test]
fn a_recall_outside_zero_to_one_is_a_load_error() {
    for ok in ["0.0", "1.0", "0.667", "0.911"] {
        assert!(
            table_of(&signed_row(&[("judge_recall_on_class", ok)])).is_ok(),
            "recall {ok} must load"
        );
    }
    for bad in ["-0.001", "-1.0", "1.001", "2.0", "nan", "inf", "-inf"] {
        assert!(
            table_of(&signed_row(&[("judge_recall_on_class", bad)])).is_err(),
            "recall {bad} must not load"
        );
    }
}

/// ADR-0004 §4(v): the margin is over the RE-MEASURED incumbent and "a tie keeps the
/// incumbent". A row recording a margin of zero or less records a candidate that should
/// never have been written down. Both sides of zero.
#[test]
fn a_non_positive_margin_is_a_load_error() {
    for ok in ["0.001", "1.42", "12.0"] {
        assert!(table_of(&signed_row(&[("margin", ok)])).is_ok(), "margin {ok} must load");
    }
    for bad in ["0.0", "-0.001", "-3.0", "nan", "-inf"] {
        assert!(table_of(&signed_row(&[("margin", bad)])).is_err(), "margin {bad} must not load");
    }
}

/// Validation applies to UNSIGNED rows too (ADR-0004 §3: "present in the file, validated on
/// load"). A malformed proposal must not be able to sit in the file until the day somebody
/// signs it, because signing is the moment nobody re-reads the provenance.
#[test]
fn an_unsigned_row_is_validated_just_as_hard_as_a_signed_one() {
    assert!(table_of(&tuned_row(&[("judge", "\"\"")])).is_err());
    assert!(table_of(&tuned_row(&[("margin", "0.0")])).is_err());
    assert!(table_of(&tuned_row(&[("lang", "\"fr\"")])).is_err());
    assert!(table_of(&tuned_row(&[("phrase", "\"Speak [sadly]\"")])).is_err());
    assert!(table_of(&tuned_row(&[])).is_ok());
}

/// `holdout_uses` is NOT checked against the K = 5 budget of ADR-0004 §5, on purpose: the
/// budget belongs to the tuning driver, which knows how many decisions the round is about
/// to serve, and duplicating it here would put the same constant in two crates.
///
/// Pinned so that the absence is a decision on record rather than an oversight a reader has
/// to infer. A future pass that wants the check has to change this test deliberately.
#[test]
fn the_holdout_budget_is_deliberately_not_enforced_at_load() {
    assert!(table_of(&signed_row(&[("holdout_uses", "0")])).is_ok());
    assert!(
        table_of(&signed_row(&[("holdout_uses", "99")])).is_ok(),
        "the table stores what was measured; the budget is the tuner's gate"
    );
}

// ---------------------------------------------------------------- the hard invariant

/// **The CLAUDE.md hard invariant, applied to the one string that reaches a model as
/// prose.** A tuned phrase is machine-generated and no human reads it between the search
/// and the file, so it is the likeliest source of a cue-markup leak in the tree.
#[test]
fn a_tuned_phrase_carrying_cue_markup_is_a_load_error() {
    for bad in [
        "\"Speak [sadly]\"",
        "\"Speak <prosody rate='slow'>\"",
        "\"a<|endofprompt|>b\"",
        "\"pipe | delimited\"",
        "\"two\\nlines\"",
        "\" padded \"",
        "\"\"",
    ] {
        assert!(
            table_of(&signed_row(&[("phrase", bad)])).is_err(),
            "phrase {bad} must not load"
        );
    }
    for ok in ["\"Speak in a quietly sad tone\"", "\"用悲伤难过的语气说\"", "\"Shout this loudly\""] {
        assert!(table_of(&signed_row(&[("phrase", ok)])).is_ok(), "phrase {ok} must load");
    }
    // Both sides of the length bound.
    let at = format!("\"{}\"", "a".repeat(120));
    let past = format!("\"{}\"", "a".repeat(121));
    assert!(table_of(&signed_row(&[("phrase", &at)])).is_ok(), "120 chars is the limit");
    assert!(table_of(&signed_row(&[("phrase", &past)])).is_err(), "121 chars is past it");
}

/// The invariant holds at the OUTPUT of the pipeline as well as at load — the property a
/// reviewer actually cares about is that nothing with markup in it reaches a backend.
#[test]
fn no_prefix_a_tuned_table_can_produce_carries_cue_markup() {
    let v = Vocab::embedded().expect("vocab");
    let t = table(&signed_row(&[]));
    for id in BackendId::ALL {
        for lang in [InstructLang::En, InstructLang::Zh] {
            for (_, e) in v.iter() {
                let src = format!("[{}] some text", e.id);
                for seg in hoist(&src, *id, lang, &t) {
                    let Some(p) = seg.instruct else { continue };
                    for bad in ['[', ']', '<', '>', '|'] {
                        assert!(!p.contains(bad), "{}/{}: prefix {p:?} carries {bad:?}", id.as_str(), e.id);
                    }
                    assert!(!p.contains("endofprompt"), "{}: {p:?}", e.id);
                }
            }
        }
    }
}

/// The load-time rule and the proposal-time rule are ONE rule in two places.
/// `syrinx_eval::tune::phrase_is_safe` gates a phrase when the loop proposes it;
/// `syrinx_cue::instruct::phrase_is_safe` gates it when the file loads. The eval crate is
/// optional and model-gated so it cannot be the only enforcement point, and two copies that
/// drift would mean a phrase rejected at one end and accepted at the other.
#[test]
fn the_load_time_and_proposal_time_safety_rules_agree() {
    for case in [
        "Speak in a quietly sad tone",
        "用悲伤难过的语气说",
        "say [happy] now",
        "a<|endofprompt|>b",
        "two\nlines",
        " padded",
        "",
        "   ",
        "pipe | here",
        "<tag>",
        &"a".repeat(120),
        &"a".repeat(121),
    ] {
        assert_eq!(
            phrase_is_safe(case).is_ok(),
            syrinx_eval::tune::phrase_is_safe(case).is_ok(),
            "the two safety rules disagree on {case:?}"
        );
    }
}

// ---------------------------------------------------------------- ambiguity

/// Two ACCEPTED rows for one `(backend, lang, label)` mean the file does not say what will
/// be spoken. Keeping the last one written makes the answer depend on row order, which is
/// not something a reader of the file can see.
#[test]
fn two_accepted_rows_for_one_key_are_a_load_error() {
    let dup = format!("{}{}", signed_row(&[]), signed_row(&[("phrase", "\"Speak very sadly\"")]));
    assert!(table_of(&dup).is_err(), "a duplicate key must not resolve by row order");

    // Not a duplicate: same label and lang, different checkpoint.
    let scoped = format!(
        "{}{}",
        signed_row(&[]),
        signed_row(&[("backend", "\"qwen3-0.6b-customvoice\"")])
    );
    assert!(table_of(&scoped).is_ok());

    // Not a duplicate either: only one of them is signed, so only one is live.
    let one_live = format!("{}{}", signed_row(&[]), tuned_row(&[("phrase", "\"Speak very sadly\"")]));
    let t = table(&one_live);
    assert_eq!(t.accepted_tuned(), 1);
    assert_eq!(t.tuned_rows().len(), 2);
    assert_eq!(
        prefix("[sad] x", BackendId::Qwen17bCustomVoice, InstructLang::En, &t).as_deref(),
        Some(TUNED_EN),
        "the signed row wins over an unsigned one for the same key"
    );
}

// ---------------------------------------------------------------- reporting

/// Every row stays visible for reporting whether or not it is live, so a reviewer can see
/// what the loop proposed. The full provenance survives the round trip — a row whose
/// fields were dropped on load could not be audited later.
#[test]
fn every_row_is_reported_with_its_provenance_intact() {
    let t = table(&format!("{}{}", signed_row(&[]), tuned_row(&[("label", "\"angry\"")])));
    let rows: &[TunedPhrase] = t.tuned_rows();
    assert_eq!(rows.len(), 2, "signed and unsigned alike");
    assert_eq!(t.accepted_tuned(), 1, "but only one is live");

    let r = rows.iter().find(|r| r.label == LABEL).expect("the sad row");
    assert_eq!(r.lang, "en");
    assert_eq!(r.backend, BACKEND);
    assert_eq!(r.phrase, TUNED_EN);
    assert_eq!(r.measured_on, "2026-09-09");
    assert_eq!(r.incumbent, CURATED_EN);
    assert_eq!(r.margin, 1.42);
    assert_eq!(r.judge, "emotion2vec/emotion2vec_plus_large");
    assert_eq!(r.judge_recall_on_class, 0.867);
    assert_eq!(r.holdout_id, "sad-2026-09-09-a");
    assert_eq!(r.holdout_uses, 0);
    assert_eq!(r.accepted_by.as_deref(), Some("fixture — not a real acceptance"));

    let unsigned = rows.iter().find(|r| r.label == "angry").expect("the angry row");
    assert_eq!(unsigned.accepted_by, None, "an unsigned row records no signature");
}
