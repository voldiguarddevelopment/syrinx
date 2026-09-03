//! Cue → [`RenderPlan`] override path (C3.2).
//!
//! `syrinx-cue` owns cue syntax and lowering; this crate **consumes** the resulting
//! `CueDoc` and turns the prosodic cues into render-plan knobs. The dependency runs one
//! way only (ADR-0001 D6): nothing here parses cue syntax.
//!
//! ## What is measured, and what is not
//!
//! The AC (ledger A17) is measured on the **plan**: `RenderPlan::apply` warps the frame
//! axis, so a rate cue is verifiable as a frame-count ratio, and a pitch cue as the
//! returned `f0_mult`. Whether the acoustic model then *follows* the plan is a perceptual
//! question that needs the GPU, and is deliberately not claimed here.

use crate::render_plan::{Region, RenderPlan};
use syrinx_cue::ir::{CueDoc, CueKind, Level};

/// Deterministic prosodic deltas for an emphasis level.
///
/// Emphasis has no single physical definition, so the mapping is fixed here and pinned by
/// test rather than left to each caller to invent.
pub fn emphasis_deltas(level: Level) -> (f64, f64) {
    match level {
        // (rate, pitch semitones). Emphasis slows and lifts; reduced does the opposite.
        Level::Strong => (0.90, 2.0),
        Level::Moderate => (0.95, 1.0),
        Level::Reduced => (1.05, -1.0),
    }
}

/// Map a byte range in the clean text onto a frame range, proportionally.
///
/// Speech is not uniform in time, so this is an approximation — but it is a *deterministic*
/// one, and it is the only mapping available before the durations are predicted. Callers
/// that have real per-token durations should build regions from those instead.
fn frames_for(span: &std::ops::Range<usize>, text_len: usize, frames: usize) -> (usize, usize) {
    if text_len == 0 || frames == 0 {
        return (0, 0);
    }
    let scale = |b: usize| (b.min(text_len) * frames) / text_len;
    let start = scale(span.start);
    let end = scale(span.end).max(start);
    (start, end)
}

/// Build a [`RenderPlan`] from the prosodic cues in `doc`.
///
/// Only [`CueKind::Prosody`] and [`CueKind::Emphasis`] participate — they are the cues with
/// a defined effect on rate and pitch. Emotion/style/event cues steer the model itself and
/// are handled by `syrinx-cue`'s lowering, not here; pauses are timeline edits rather than
/// warps and are left to the caller.
///
/// A cue whose span covers the whole text becomes a **global** knob; a narrower cue becomes
/// a [`Region`]. Later cues win on overlap, matching `RenderPlan`'s documented rule.
pub fn plan_from_cues(doc: &CueDoc, frames: usize) -> RenderPlan {
    let text_len = doc.text.len();
    let mut plan = RenderPlan::identity();

    for cue in &doc.cues {
        let (rate, pitch) = match &cue.kind {
            CueKind::Prosody { rate, pitch_st, .. } => (
                rate.map(|r| r as f64),
                pitch_st.map(|p| p as f64),
            ),
            CueKind::Emphasis { level } => {
                let (r, p) = emphasis_deltas(*level);
                (Some(r), Some(p))
            }
            _ => continue,
        };
        if rate.is_none() && pitch.is_none() {
            continue;
        }

        let covers_all = cue.span.start == 0 && cue.span.end >= text_len && text_len > 0;
        if covers_all {
            if let Some(r) = rate {
                plan.global_rate = r;
            }
            if let Some(p) = pitch {
                plan.global_pitch_semitones = p;
            }
            continue;
        }

        let (start_frame, end_frame) = frames_for(&cue.span, text_len, frames);
        if start_frame >= end_frame {
            continue; // a point event warps nothing
        }
        plan.regions.push(Region {
            start_frame,
            end_frame,
            rate,
            pitch_semitones: pitch,
        });
    }
    plan
}
