//! Utterance-level speech-rate scaling on the typed prosody plan (T-03.04).
//!
//! A single read-only transform on an existing [`ProsodyPlan`]:
//!
//!   * [`scale_rate`](ProsodyPlan::scale_rate) — multiply every `durations_ms`
//!     entry by a positive rate factor `r`, returning a NEW plan whose total
//!     duration scales by `r` and whose per-phoneme proportions are preserved,
//!     with `pitch_hz` left element-for-element identical. `r == 1.0` is a
//!     duration identity; `r <= 0.0` yields [`PlanError::InvalidRate`] and no
//!     plan; any `r > 0.0` is accepted. Never panics.
//!
//! Scope (list.md / T-03.04): a whole-plan duration scale only — no pitch shift,
//! no per-phoneme rate, no prediction, no defaults. Whether the time-scaled audio
//! is perceptually correct and pitch-preserved on rendered output is a deferred
//! perceptual eval against the real model, not gated here.

use crate::plan::{PlanError, ProsodyPlan};

impl ProsodyPlan {
    /// Scale every phoneme duration by a positive rate factor `r`.
    ///
    /// Returns a new plan whose `durations_ms[i] == r * self.durations_ms[i]`
    /// (so the summed total and per-phoneme proportions scale uniformly) and
    /// whose `pitch_hz` is identical to the input — rate scaling never touches
    /// pitch. `r <= 0.0` returns [`PlanError::InvalidRate`] and produces no plan;
    /// this never panics on any `f32`.
    pub fn scale_rate(&self, r: impl Into<f64>) -> Result<ProsodyPlan, PlanError> {
        let r: f64 = r.into();
        if r <= 0.0 {
            return Err(PlanError::InvalidRate);
        }
        let durations_ms = self
            .durations_ms
            .iter()
            .map(|d| (*d as f64 * r) as f32)
            .collect();
        Ok(ProsodyPlan {
            schema_version: self.schema_version,
            durations_ms,
            pitch_hz: self.pitch_hz.clone(),
        })
    }
}
