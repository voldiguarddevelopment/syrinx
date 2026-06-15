//! Duration-override API on the typed prosody plan (T-03.02).
//!
//! Two edits on an existing [`ProsodyPlan`], both touching only `durations_ms`
//! and never `pitch_hz`:
//!
//!   * [`set_duration`](ProsodyPlan::set_duration) — a single-index write that
//!     sets exactly one duration entry, or rejects an out-of-range index with
//!     [`PlanError::IndexOutOfRange`] without mutating anything.
//!   * [`override_durations`](ProsodyPlan::override_durations) — a whole-array
//!     replacement that succeeds iff the new array has length `N`, or rejects a
//!     length disagreement with [`PlanError::LengthMismatch`] atomically.
//!
//! Scope (list.md / T-03.02): duration-array editing only — no pitch, volume, or
//! rate control, no prediction, no defaults. Values are caller-supplied per
//! T-03.01. Whether the overridden timing sounds right on rendered audio is a
//! deferred perceptual eval, not gated here.

use crate::plan::{PlanError, ProsodyPlan};

impl ProsodyPlan {
    /// Write a single phoneme's duration at index `i`.
    ///
    /// On success `durations_ms[i] == v` and nothing else changes — every other
    /// `durations_ms` entry and the entire `pitch_hz` array are untouched. `i` at
    /// or past `N` yields [`PlanError::IndexOutOfRange`] and mutates nothing; this
    /// never panics on any `usize`.
    pub fn set_duration(&mut self, i: usize, v: f32) -> Result<(), PlanError> {
        if i >= self.durations_ms.len() {
            return Err(PlanError::IndexOutOfRange);
        }
        self.durations_ms[i] = v;
        Ok(())
    }

    /// Replace the whole `durations_ms` array.
    ///
    /// Succeeds iff `new.len() == N`, after which `durations_ms == new` and
    /// `pitch_hz` is untouched. Any other length yields
    /// [`PlanError::LengthMismatch`] and leaves the plan unchanged — atomic, no
    /// partial write, never a panic.
    pub fn override_durations(&mut self, new: Vec<f32>) -> Result<(), PlanError> {
        if new.len() != self.durations_ms.len() {
            return Err(PlanError::LengthMismatch);
        }
        self.durations_ms = new;
        Ok(())
    }
}
