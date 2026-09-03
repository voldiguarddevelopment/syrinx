//! C3.2, to the AC rewritten in ledger A17: duration and pitch deltas are **measured** on
//! `RenderPlan::apply` over frozen fixtures — duration as the output/input frame ratio
//! (±1 frame, the warp being integral), pitch as the returned `f0_mult` (±1e-6).
//!
//! This measures the plan's effect. Whether the acoustic model follows the plan needs a
//! GPU and stays blocked; nothing here claims otherwise.

use syrinx_cue::ir::{Cue, CueDoc, CueKind, Level};
use syrinx_prosody::cues::{emphasis_deltas, plan_from_cues};
use syrinx_prosody::render_plan::semitones_to_ratio;

/// A frozen synthetic mel: 8 bands x 100 frames, deterministic content.
fn frozen_mel(frames: usize) -> Vec<Vec<f32>> {
    (0..8)
        .map(|b| (0..frames).map(|t| ((b * 31 + t * 7) % 17) as f32 * 0.1).collect())
        .collect()
}

fn doc_with(kind: CueKind, span: std::ops::Range<usize>, text: &str) -> CueDoc {
    CueDoc {
        text: text.to_string(),
        cues: vec![Cue { span, kind, raw: "cue".into(), source: 0..0 }],
        speakers: vec![],
    }
}

const TEXT: &str = "the quick brown fox jumps over the lazy dog again and again";

#[test]
fn a_global_rate_cue_changes_the_measured_duration_by_one_over_rate() {
    let frames = 100;
    let mel = frozen_mel(frames);
    for rate in [0.5f32, 0.75, 0.9, 1.0, 1.25, 2.0] {
        let doc = doc_with(
            CueKind::Prosody { rate: Some(rate), pitch_st: None, volume_db: None },
            0..TEXT.len(),
            TEXT,
        );
        let plan = plan_from_cues(&doc, frames);
        assert_eq!(plan.global_rate, rate as f64, "a full-span cue must be a global knob");
        let (out, _) = plan.apply(&mel).expect("plan must apply");
        let t_out = out[0].len() as f64;
        let expected = frames as f64 / rate as f64;
        assert!(
            (t_out - expected).abs() <= 1.0,
            "rate {rate}: measured {t_out} frames, expected {expected} (+/-1)"
        );
    }
}

#[test]
fn a_global_pitch_cue_changes_the_measured_f0_multiplier() {
    let frames = 100;
    let mel = frozen_mel(frames);
    for st in [-12.0f32, -3.0, -1.0, 0.0, 1.0, 2.0, 12.0] {
        let doc = doc_with(
            CueKind::Prosody { rate: None, pitch_st: Some(st), volume_db: None },
            0..TEXT.len(),
            TEXT,
        );
        let plan = plan_from_cues(&doc, frames);
        let (_, f0) = plan.apply(&mel).unwrap();
        let expected = semitones_to_ratio(st as f64);
        for (i, m) in f0.iter().enumerate() {
            assert!(
                (m - expected).abs() < 1e-6,
                "{st} st: f0_mult[{i}] = {m}, expected {expected}"
            );
        }
    }
}

#[test]
fn a_narrow_cue_becomes_a_region_and_only_that_region_moves() {
    let frames = 100;
    let mel = frozen_mel(frames);
    // A cue over the second half of the text.
    let half = TEXT.len() / 2;
    let doc = doc_with(
        CueKind::Prosody { rate: None, pitch_st: Some(6.0), volume_db: None },
        half..TEXT.len(),
        TEXT,
    );
    let plan = plan_from_cues(&doc, frames);
    assert_eq!(plan.regions.len(), 1, "a partial span must become a region");
    assert_eq!(plan.global_pitch_semitones, 0.0, "the global knob must stay untouched");

    let (_, f0) = plan.apply(&mel).unwrap();
    let raised = semitones_to_ratio(6.0);
    // The first frames are unshifted, the later ones are raised. Both must be present, or
    // the test would pass on a plan that did nothing.
    assert!(f0.iter().any(|m| (m - 1.0).abs() < 1e-6), "no unshifted frames: {f0:?}");
    assert!(f0.iter().any(|m| (m - raised).abs() < 1e-6), "no raised frames");
}

#[test]
fn emphasis_deltas_are_fixed_and_ordered() {
    // Pinned so the mapping cannot drift, and ordered so "strong" is never weaker than
    // "moderate" — a sign flip here would invert every emphasis in the system.
    let (rs, ps) = emphasis_deltas(Level::Strong);
    let (rm, pm) = emphasis_deltas(Level::Moderate);
    let (rr, pr) = emphasis_deltas(Level::Reduced);
    assert_eq!((rs, ps), (0.90, 2.0));
    assert_eq!((rm, pm), (0.95, 1.0));
    assert_eq!((rr, pr), (1.05, -1.0));
    assert!(rs < rm && rm < rr, "emphasis must slow and reduction must speed up");
    assert!(ps > pm && pm > pr, "emphasis must raise and reduction must lower");
}

#[test]
fn an_emphasis_cue_measurably_slows_and_lifts_its_span() {
    let frames = 100;
    let mel = frozen_mel(frames);
    let doc = doc_with(CueKind::Emphasis { level: Level::Strong }, 0..TEXT.len(), TEXT);
    let plan = plan_from_cues(&doc, frames);
    let (out, f0) = plan.apply(&mel).unwrap();
    let (rate, st) = emphasis_deltas(Level::Strong);
    // Slower => more frames.
    let expected = frames as f64 / rate;
    assert!(
        (out[0].len() as f64 - expected).abs() <= 1.0,
        "emphasis duration: {} vs {expected}",
        out[0].len()
    );
    assert!((f0[0] - semitones_to_ratio(st)).abs() < 1e-6);
}

#[test]
fn non_prosodic_cues_do_not_touch_the_plan() {
    // Emotion/style/event steer the model, not the warp. If they started producing regions
    // here they would double-apply against the backend's own control.
    let frames = 50;
    for kind in [
        CueKind::Emotion { label: "happy".into(), intensity: 0.6 },
        CueKind::Style { label: "whisper".into() },
        CueKind::Event { label: "laughs".into() },
        CueKind::Free,
        CueKind::Pause { ms: 300 },
        CueKind::SpeakerTurn { id: 1 },
    ] {
        let doc = doc_with(kind.clone(), 0..TEXT.len(), TEXT);
        let plan = plan_from_cues(&doc, frames);
        assert_eq!(plan, syrinx_prosody::render_plan::RenderPlan::identity(),
                   "{kind:?} must leave the plan at identity");
    }
}

#[test]
fn a_point_event_warps_nothing() {
    let doc = doc_with(
        CueKind::Prosody { rate: Some(0.5), pitch_st: None, volume_db: None },
        10..10,
        TEXT,
    );
    let plan = plan_from_cues(&doc, 100);
    assert!(plan.regions.is_empty(), "a zero-width span cannot warp a range");
}

#[test]
fn later_cues_win_on_overlap_matching_the_render_plan_rule() {
    let frames = 100;
    let mel = frozen_mel(frames);
    let mut doc = doc_with(
        CueKind::Prosody { rate: None, pitch_st: Some(3.0), volume_db: None },
        0..30,
        TEXT,
    );
    doc.cues.push(Cue {
        span: 0..30,
        kind: CueKind::Prosody { rate: None, pitch_st: Some(-3.0), volume_db: None },
        raw: "second".into(),
        source: 0..0,
    });
    let plan = plan_from_cues(&doc, frames);
    let (_, f0) = plan.apply(&mel).unwrap();
    assert!(
        (f0[0] - semitones_to_ratio(-3.0)).abs() < 1e-6,
        "the later cue must win, got {}",
        f0[0]
    );
}

#[test]
fn degenerate_inputs_do_not_panic() {
    let empty = CueDoc { text: String::new(), cues: vec![], speakers: vec![] };
    assert_eq!(plan_from_cues(&empty, 0), syrinx_prosody::render_plan::RenderPlan::identity());
    // zero frames, non-empty text
    let doc = doc_with(
        CueKind::Prosody { rate: Some(0.5), pitch_st: None, volume_db: None },
        0..5,
        TEXT,
    );
    let _ = plan_from_cues(&doc, 0);
}
