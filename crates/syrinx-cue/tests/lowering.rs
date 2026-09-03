//! C2.2. **AC: unit tests per pass; a Fish-style backend receives byte-identical tags for
//! `Free` cues.**

use syrinx_cue::caps::BackendId;
use syrinx_cue::ir::{Cue, CueDoc, CueKind};
use syrinx_cue::lower::{lower, pass_normalize, pass_passthrough_and_map, Action, DropReason,
                        LoweringReport};
use syrinx_cue::parse::ParseOptions;
use syrinx_cue::{parse, Vocab};

fn doc(src: &str) -> CueDoc {
    parse(src, &Vocab::embedded().unwrap(), &ParseOptions::default())
}

fn caps(id: BackendId) -> syrinx_cue::ControlCaps {
    id.caps().unwrap()
}

// ------------------------------------------------------------------ pass 1: normalize

#[test]
fn the_parser_already_canonicalises_so_pass1_is_the_defensive_net() {
    // Where canonicalisation actually happens: `parse` resolves labels against the vocab,
    // so a synonym is already `happy` by the time lowering runs. Pinning this keeps the
    // two from silently swapping responsibility.
    let d = doc("[Cheerful] hi");
    match &d.cues[0].kind {
        CueKind::Emotion { label, .. } => assert_eq!(label, "happy"),
        k => panic!("{k:?}"),
    }
    // The author's text is never destroyed by canonicalisation.
    assert_eq!(d.cues[0].raw, "Cheerful");

    let v = Vocab::embedded().unwrap();
    let mut d2 = d.clone();
    let mut r = LoweringReport::default();
    pass_normalize(&mut d2, &v, &mut r);
    assert!(r.entries.is_empty(), "already canonical: pass 1 must be a no-op");
}

#[test]
fn pass1_canonicalises_an_ir_that_did_not_come_from_the_parser() {
    // The case pass 1 exists for: IR built by an API caller, or by the SSML path, which
    // never went through the parser's vocabulary resolution.
    let v = Vocab::embedded().unwrap();
    let mut d = CueDoc {
        text: " hi".into(),
        cues: vec![Cue {
            span: 0..3,
            kind: CueKind::Emotion { label: "Cheerful".into(), intensity: 0.6 },
            raw: "Cheerful".into(),
            source: 0..10,
        }],
        speakers: vec![],
    };
    let mut r = LoweringReport::default();
    pass_normalize(&mut d, &v, &mut r);
    match &d.cues[0].kind {
        CueKind::Emotion { label, .. } => assert_eq!(label, "happy", "synonym must canonicalise"),
        k => panic!("{k:?}"),
    }
    assert!(matches!(&r.entries[0].action,
                     Action::Normalized { from, to } if from == "Cheerful" && to == "happy"));
    assert_eq!(d.cues[0].raw, "Cheerful", "raw is never rewritten");
}

#[test]
fn pass1_leaves_an_already_canonical_label_alone_and_reports_nothing() {
    let v = Vocab::embedded().unwrap();
    let mut d = doc("[happy] hi");
    let mut r = LoweringReport::default();
    pass_normalize(&mut d, &v, &mut r);
    assert!(r.entries.is_empty(), "a no-op must not generate a report line");
}

#[test]
fn pass1_leaves_free_cues_free() {
    let v = Vocab::embedded().unwrap();
    let mut d = doc("[like a distant foghorn] hi");
    let mut r = LoweringReport::default();
    pass_normalize(&mut d, &v, &mut r);
    assert_eq!(d.cues[0].kind, CueKind::Free);
    assert!(r.entries.is_empty());
}

// ------------------------------------------------------------------ pass 2: pass-through

/// The headline AC.
#[test]
fn pass2_fish_s2_receives_free_cues_byte_identical() {
    let v = Vocab::embedded().unwrap();
    let s2 = caps(BackendId::FishS2Pro);
    for src in [
        "[whisper in a small voice] secret",
        "[professional broadcast tone] the news",
        "[like a distant foghorn, slowly] oooo",
        "[  odd   spacing  ] x",
        "[MiXeD CaSe] x",
    ] {
        let d = doc(src);
        let raw = d.cues[0].raw.clone();
        let out = lower(&d, &s2, &v);
        let e = &out.report.entries[0];
        match &e.action {
            Action::PassedThrough { rendered } => assert_eq!(
                rendered, &raw,
                "Fish S2 must receive {raw:?} byte-identical, got {rendered:?}"
            ),
            a => panic!("expected pass-through for {src:?}, got {a:?}"),
        }
        // and the cue itself is unchanged
        assert_eq!(out.cues[0].raw, raw);
    }
}

#[test]
fn pass2_does_not_canonicalise_away_expressive_range_on_an_open_vocabulary() {
    let v = Vocab::embedded().unwrap();
    let d = doc("[Cheerful] hi");
    let out = lower(&d, &caps(BackendId::FishS2Pro), &v);
    // Pass 1 canonicalises the KIND, but what Fish S2 receives is still the author's text.
    let rendered = out.report.entries.iter().find_map(|e| match &e.action {
        Action::PassedThrough { rendered } => Some(rendered.clone()),
        _ => None,
    });
    assert_eq!(rendered.as_deref(), Some("Cheerful"));
}

// ------------------------------------------------------------------ pass 3: vocab map

#[test]
fn pass3_maps_a_canonical_label_to_the_backends_native_spelling() {
    let v = Vocab::embedded().unwrap();
    let out = lower(&doc("[happy] hi"), &caps(BackendId::FishS1Mini), &v);
    let mapped = out.report.entries.iter().find_map(|e| match &e.action {
        Action::Mapped { to } => Some(to.clone()),
        _ => None,
    });
    let expected = v.by_id("happy").unwrap().1.fish_s1.clone().unwrap();
    assert_eq!(mapped, Some(expected.clone()));
    // the surviving cue carries the NATIVE spelling, which is what the backend will emit
    match &out.cues[0].kind {
        CueKind::Emotion { label, .. } => assert_eq!(label, &expected),
        k => panic!("{k:?}"),
    }
}

#[test]
fn pass3_reports_unmapped_when_a_label_has_no_native_spelling() {
    let v = Vocab::embedded().unwrap();
    // Find a label the closed S1 set genuinely cannot spell; if every label maps, this
    // test would be vacuous, so assert we found one.
    let orphan = v
        .iter()
        .find(|(_, e)| e.fish_s1.is_none() && e.fish_s2.is_some())
        .map(|(_, e)| e.id.clone());
    let Some(id) = orphan else {
        panic!("no label lacks a fish_s1 spelling — this test cannot prove anything");
    };
    let out = lower(&doc(&format!("[{id}] hi")), &caps(BackendId::FishS1Mini), &v);
    assert!(
        out.report.entries.iter().any(|e| e.action == Action::Unmapped),
        "expected Unmapped for {id:?}, got {:?}",
        out.report.entries
    );
}

#[test]
fn pass3_free_cues_cannot_survive_on_a_closed_vocabulary() {
    let v = Vocab::embedded().unwrap();
    let out = lower(&doc("[like a distant foghorn] hi"), &caps(BackendId::FishS1Mini), &v);
    assert!(matches!(
        out.report.entries[0].action,
        Action::Dropped { reason: DropReason::Unsupported }
    ));
    assert!(out.cues.is_empty());
}

// ------------------------------------------------------------------ the report

#[test]
fn accepted_but_ignored_is_reported_distinctly_from_unsupported() {
    let v = Vocab::embedded().unwrap();
    let d = doc("[happy] hi");

    // 0.6B CustomVoice ACCEPTS the instruction and discards it.
    let small = lower(&d, &caps(BackendId::Qwen06bCustomVoice), &v);
    assert!(
        matches!(small.report.entries[0].action,
                 Action::Dropped { reason: DropReason::AcceptedButIgnored }),
        "got {:?}",
        small.report.entries[0].action
    );

    // The Base checkpoint has no channel at all — a different reason, not a different
    // outcome, and the user must be able to tell them apart.
    let base = lower(&d, &caps(BackendId::Qwen06bBase), &v);
    assert!(matches!(base.report.entries[0].action,
                     Action::Dropped { reason: DropReason::Unsupported }));

    assert_ne!(small.report.entries[0].action, base.report.entries[0].action);
    assert!(small.report.explain().contains("does not act on it"));
    assert!(base.report.explain().contains("no channel"));
}

#[test]
fn nothing_is_ever_dropped_silently() {
    let v = Vocab::embedded().unwrap();
    let src = "[happy] a [whisper] b [laughs] c [like a foghorn] d";
    let d = doc(src);
    let n = d.cues.len();
    assert!(n >= 4, "fixture must exercise several cues, got {n}");
    for id in BackendId::ALL {
        let out = lower(&d, &caps(*id), &v);
        // Every cue that entered lowering has at least one report line...
        for cue in &d.cues {
            assert!(
                out.report.entries.iter().any(|e| e.source == cue.source),
                "cue {:?} vanished with no report line on {}",
                cue.raw,
                id.as_str()
            );
        }
        // ...and every surviving cue was reported as delivered.
        assert_eq!(
            out.cues.len(),
            out.report.delivered().filter(|e| !matches!(e.action, Action::Normalized { .. })).count(),
            "delivered count disagrees with surviving cues on {}",
            id.as_str()
        );
    }
}

#[test]
fn lowering_never_alters_the_clean_text_in_passes_1_to_3() {
    // Text rewriting belongs to C2.3 (splitting) and C2.4 (strip). If it starts happening
    // here, spans silently stop matching the text and every downstream offset is wrong.
    let v = Vocab::embedded().unwrap();
    let d = doc("[happy] hello [sad] world");
    for id in BackendId::ALL {
        let out = lower(&d, &caps(*id), &v);
        assert_eq!(out.text, d.text, "text changed on {}", id.as_str());
    }
}

#[test]
fn the_hard_invariant_survives_lowering_on_every_backend() {
    let v = Vocab::embedded().unwrap();
    for src in ["[happy] a [whisper] b", "[unknown thing] c", "<|speaker:1|> d [laughs] e"] {
        let d = doc(src);
        for id in BackendId::ALL {
            let out = lower(&d, &caps(*id), &v);
            assert!(!out.text.contains('['), "'[' leaked on {} from {src:?}", id.as_str());
            assert!(!out.text.contains(']'), "']' leaked on {} from {src:?}", id.as_str());
            assert!(
                !out.text.contains("<|speaker"),
                "speaker token leaked on {} from {src:?}",
                id.as_str()
            );
        }
    }
}

#[test]
fn passes_are_independently_runnable_and_compose_to_lower() {
    let v = Vocab::embedded().unwrap();
    let c = caps(BackendId::FishS1Mini);
    let d = doc("[Cheerful] hi");

    let mut manual = d.clone();
    let mut r = LoweringReport::default();
    pass_normalize(&mut manual, &v, &mut r);
    pass_passthrough_and_map(&mut manual, &c, &v, &mut r);

    let auto = lower(&d, &c, &v);
    assert_eq!(manual.cues, auto.cues);
    assert_eq!(r, auto.report);
}

#[test]
fn a_cue_with_no_label_needs_no_spelling() {
    let v = Vocab::embedded().unwrap();
    let d = CueDoc {
        text: "ab".into(),
        cues: vec![Cue {
            span: 1..1,
            kind: CueKind::Pause { ms: 300 },
            raw: "pause 300ms".into(),
            source: 0..0,
        }],
        speakers: vec![],
    };
    // CV2 does not support pause at all -> dropped, with a reason.
    let cv2 = lower(&d, &caps(BackendId::CosyVoice2), &v);
    assert!(matches!(cv2.report.entries[0].action,
                     Action::Dropped { reason: DropReason::Unsupported }));
}
