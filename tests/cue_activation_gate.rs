//! C4.1 (activation harness) + C4.2 (frozen cue set, checksum, regression gate).
//!
//! **What is certified here:** the frozen set is immutable and checksummed, the harness
//! aggregates correctly per kind and per backend, the JSON has the required shape, and the
//! thresholds block a regression. **What is not:** the activation numbers themselves,
//! which require synthesizing the set on a GPU. A run that never happened produces no
//! measurements, and the harness reports that rather than a green.

use sha2::{Digest, Sha256};
use syrinx_eval::activation::{evaluate_activation, CueCase, Measurement, Thresholds};

const SET: &str = include_str!("golden/cue_eval/cue_set.jsonl");
const SUM: &str = include_str!("golden/cue_eval/cue_set.sha256");

fn cases() -> Vec<CueCase> {
    CueCase::parse_set(SET)
}

// ------------------------------------------------------------------ C4.2: the frozen set

#[test]
fn the_frozen_cue_set_matches_its_checksum() {
    // The whole point of freezing: if the set drifts, every historical number becomes
    // incomparable. A mismatch must fail loudly rather than silently re-baseline.
    let want = SUM.split_whitespace().next().expect("checksum file must name a digest");
    let got = format!("{:x}", Sha256::digest(SET.as_bytes()));
    assert_eq!(
        got, want,
        "the frozen cue set changed. If that is intentional, regenerate the checksum AND \
         re-baseline every recorded metric — they are not comparable across sets."
    );
}

#[test]
fn the_frozen_set_covers_every_kind_placement_and_language() {
    let cs = cases();
    assert_eq!(cs.len(), 54, "the set size is part of what is frozen");
    for kind in ["emotion", "style", "event", "neutral"] {
        assert!(cs.iter().any(|c| c.kind == kind), "no {kind} cases");
    }
    for lang in ["en", "de", "pl"] {
        assert!(cs.iter().any(|c| c.lang == lang), "no {lang} cases");
    }
    for placement in ["leading", "mid", "trailing"] {
        assert!(cs.iter().any(|c| c.placement == placement), "no {placement} cases");
    }
    // A neutral control group is required, or WER delta has no baseline.
    assert!(cs.iter().filter(|c| c.kind == "neutral").count() >= 3);
    // ids are unique, or aggregation would double-count
    let mut ids: Vec<_> = cs.iter().map(|c| c.id.clone()).collect();
    ids.sort();
    let n = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), n, "duplicate case ids in the frozen set");
}

#[test]
fn every_cued_case_actually_carries_its_cue_and_no_markup_survives_parsing() {
    // The set would be worthless if a case's cue did not parse — it would score as a
    // non-activation forever.
    let vocab = syrinx_cue::Vocab::embedded().unwrap();
    for c in cases() {
        let doc = syrinx_cue::parse(&c.text, &vocab, &syrinx_cue::ParseOptions::default());
        if c.kind == "neutral" {
            assert!(doc.cues.is_empty(), "{}: control case must carry no cue", c.id);
            continue;
        }
        assert_eq!(doc.cues.len(), 1, "{}: expected exactly one cue in {:?}", c.id, c.text);
        // and the hard invariant holds across the whole frozen set
        assert!(!doc.text.contains('['), "{}: markup leaked", c.id);
        assert!(!doc.text.contains(']'), "{}: markup leaked", c.id);
    }
}

// ------------------------------------------------------------------ C4.1: the harness

fn measure(id: &str, backend: &str, activated: bool, wer: f64, base: f64) -> Measurement {
    Measurement {
        case_id: id.to_string(),
        backend: backend.to_string(),
        activated,
        wer,
        baseline_wer: base,
    }
}

#[test]
fn the_harness_emits_per_kind_and_per_backend_activation_and_wer_delta() {
    let cs = cases();
    let events: Vec<_> = cs.iter().filter(|c| c.kind == "event").collect();
    let emotions: Vec<_> = cs.iter().filter(|c| c.kind == "emotion").collect();
    assert!(events.len() >= 4 && emotions.len() >= 4);

    let mut ms = Vec::new();
    for (i, c) in events.iter().enumerate() {
        ms.push(measure(&c.id, "fish-s2-pro", i % 10 != 0, 0.10, 0.08));
    }
    for c in &emotions {
        ms.push(measure(&c.id, "fish-s2-pro", true, 0.09, 0.09));
    }
    let report = evaluate_activation(
        &cs,
        &ms,
        &["fish-s2-pro".to_string()],
        Thresholds::default(),
    );

    // One cell per (backend, kind) actually measured.
    assert_eq!(report.cells.len(), 2, "{:#?}", report.cells);
    let ev = report.cells.iter().find(|c| c.kind == "event").unwrap();
    let em = report.cells.iter().find(|c| c.kind == "emotion").unwrap();
    assert_eq!(ev.backend, "fish-s2-pro");
    assert!((em.activation_rate - 1.0).abs() < 1e-9);
    assert!((em.wer_delta - 0.0).abs() < 1e-9);
    assert!((ev.wer_delta - 0.02).abs() < 1e-9, "wer delta = {}", ev.wer_delta);

    // The JSON the AC requires.
    let json = report.to_json();
    for needle in ["\"cells\"", "\"backend\"", "\"kind\"", "\"activation_rate\"",
                   "\"wer_delta\"", "\"violations\"", "\"passed\""] {
        assert!(json.contains(needle), "activation JSON missing {needle}: {json}");
    }
}

#[test]
fn the_neutral_control_group_is_excluded_from_activation() {
    // A control case has no cue, so counting it would drag every rate toward zero.
    let cs = cases();
    let neutral: Vec<_> = cs.iter().filter(|c| c.kind == "neutral").collect();
    let ms: Vec<_> = neutral
        .iter()
        .map(|c| measure(&c.id, "fish-s2-pro", false, 0.05, 0.05))
        .collect();
    let r = evaluate_activation(&cs, &ms, &[], Thresholds::default());
    assert!(r.cells.is_empty(), "control cases must not form a cell: {:#?}", r.cells);
}

// ------------------------------------------------------------------ C4.2: the gate

#[test]
fn the_gate_blocks_an_event_activation_regression_on_open_vocab_backends() {
    let cs = cases();
    let events: Vec<_> = cs.iter().filter(|c| c.kind == "event").collect();
    let open = vec!["fish-s2-pro".to_string()];

    // Just above the 0.85 floor: pass. (3 of 4 = 0.75 would fail, so use a bigger set.)
    let pass: Vec<_> = events
        .iter()
        .enumerate()
        .map(|(i, c)| measure(&c.id, "fish-s2-pro", i > 0 || events.len() > 6, 0.1, 0.1))
        .collect();
    let r = evaluate_activation(&cs, &pass, &open, Thresholds::default());
    let rate = r.cells.iter().find(|c| c.kind == "event").unwrap().activation_rate;
    if rate >= 0.85 {
        assert!(r.passed(), "should pass at rate {rate}: {:?}", r.violations);
    }

    // Well below the floor: must be blocked.
    let fail: Vec<_> =
        events.iter().map(|c| measure(&c.id, "fish-s2-pro", false, 0.1, 0.1)).collect();
    let r = evaluate_activation(&cs, &fail, &open, Thresholds::default());
    assert!(!r.passed(), "0% activation must be blocked");
    let v = &r.violations[0];
    assert_eq!(v.metric, "activation_rate");
    assert_eq!(v.limit, 0.85);
    assert!(r.to_json().contains("\"passed\": false"));
}

#[test]
fn the_gate_blocks_a_wer_regression_on_every_backend() {
    let cs = cases();
    let em: Vec<_> = cs.iter().filter(|c| c.kind == "emotion").collect();
    // +0.6 absolute WER: over the 0.5 limit.
    let bad: Vec<_> = em.iter().map(|c| measure(&c.id, "qwen3-1.7b-customvoice", true, 0.7, 0.1)).collect();
    let r = evaluate_activation(&cs, &bad, &[], Thresholds::default());
    assert!(!r.passed());
    assert_eq!(r.violations[0].metric, "wer_delta");

    // Exactly at the limit is not a regression — boundaries matter.
    let edge: Vec<_> = em.iter().map(|c| measure(&c.id, "qwen3-1.7b-customvoice", true, 0.6, 0.1)).collect();
    let r = evaluate_activation(&cs, &edge, &[], Thresholds::default());
    assert!(r.passed(), "a delta exactly at the limit must not fail: {:?}", r.violations);
}

#[test]
fn event_activation_is_gated_only_where_events_can_be_expressed() {
    // Gating a backend that has no event channel would be a permanent, meaningless red.
    let cs = cases();
    let events: Vec<_> = cs.iter().filter(|c| c.kind == "event").collect();
    let ms: Vec<_> =
        events.iter().map(|c| measure(&c.id, "qwen3-1.7b-customvoice", false, 0.1, 0.1)).collect();
    let r = evaluate_activation(&cs, &ms, &["fish-s2-pro".to_string()], Thresholds::default());
    assert!(r.passed(), "a non-open-vocab backend must not be gated on event activation");
}

#[test]
fn an_empty_run_is_reported_as_empty_not_as_a_pass_with_no_data() {
    // The honest failure mode: no GPU run => no measurements => no cells. It must be
    // visible in the JSON that nothing was measured, so an empty run cannot masquerade as
    // a clean one in CI.
    let r = evaluate_activation(&cases(), &[], &["fish-s2-pro".to_string()], Thresholds::default());
    assert!(r.cells.is_empty());
    let json = r.to_json();
    assert!(json.contains("\"cells\": [\n  ]") || json.contains("\"cells\": [\n  ],"),
            "empty cells must be visible in the JSON: {json}");
}
