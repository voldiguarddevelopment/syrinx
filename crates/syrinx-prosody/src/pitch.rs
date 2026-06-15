//! Pitch-override API on the typed prosody plan (T-03.03).
//!
//! Two edits on an existing [`ProsodyPlan`], both touching only `pitch_hz` and
//! never `durations_ms`:
//!
//!   * [`set_pitch`](ProsodyPlan::set_pitch) — a single-index write that sets
//!     exactly one pitch entry, or rejects an out-of-range index with
//!     [`PlanError::IndexOutOfRange`] without mutating anything.
//!   * [`set_word_pitch`](ProsodyPlan::set_word_pitch) — a per-word edit over the
//!     contiguous phoneme span `[start, end)`, setting every index in the span,
//!     or rejecting a span whose `end` exceeds `N` with
//!     [`PlanError::IndexOutOfRange`] without mutating anything.
//!
//! Scope (list.md / T-03.03): pitch-array editing only — no duration, volume, or
//! rate control, and no intonation presets (those build on this in T-03.07).
//! Values are caller-supplied per T-03.01. Whether the overridden F0 sounds right
//! on rendered audio is a deferred perceptual eval, not gated here.

use std::ops::Range;

use crate::plan::{PlanError, ProsodyPlan};

impl ProsodyPlan {
    /// Write a single phoneme's pitch at index `i`.
    ///
    /// On success `pitch_hz[i] == hz` and nothing else changes — every other
    /// `pitch_hz` entry and the entire `durations_ms` array are untouched. `i` at
    /// or past `N` yields [`PlanError::IndexOutOfRange`] and mutates nothing; this
    /// never panics on any `usize`.
    pub fn set_pitch(&mut self, i: usize, hz: f32) -> Result<(), PlanError> {
        if i >= self.pitch_hz.len() {
            return Err(PlanError::IndexOutOfRange);
        }
        self.pitch_hz[i] = hz;
        Ok(())
    }

    /// Write `hz` to every phoneme in the contiguous word span `[start, end)`.
    ///
    /// On success `pitch_hz[k] == hz` for every `k` in the span and nothing else
    /// changes — entries outside the span and the entire `durations_ms` array are
    /// untouched. A span whose `end` exceeds `N` yields
    /// [`PlanError::IndexOutOfRange`] and mutates nothing; a span whose `end == N`
    /// applies. This never panics on any `usize`.
    pub fn set_word_pitch(&mut self, span: Range<usize>, hz: f32) -> Result<(), PlanError> {
        if span.end > self.pitch_hz.len() {
            return Err(PlanError::IndexOutOfRange);
        }
        for k in span {
            self.pitch_hz[k] = hz;
        }
        Ok(())
    }
}
