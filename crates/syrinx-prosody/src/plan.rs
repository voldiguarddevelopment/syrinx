//! The editable prosody-plan data model (T-03.01).
//!
//! A [`ProsodyPlan`] is caller-supplied data: a phoneme count `N` plus two
//! equal-length arrays, [`durations_ms`](ProsodyPlan::durations_ms) and
//! [`pitch_hz`](ProsodyPlan::pitch_hz), and an explicit
//! [`schema_version`](ProsodyPlan::schema_version). There is no prediction and
//! there are no defaults — every value comes from the caller. The invariant
//! `durations_ms.len() == pitch_hz.len() == N` always holds, the wire format is
//! JSON, and index access is total (a [`Result`], never a panic).

use serde::{Deserialize, Serialize};

/// The current prosody-plan schema version, stamped on every constructed plan.
pub const PLAN_SCHEMA_VERSION: u32 = 1;

/// One phoneme's plan entry: its duration in milliseconds and its pitch in hertz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Phoneme {
    /// The phoneme's duration, in milliseconds.
    pub duration_ms: f32,
    /// The phoneme's pitch, in hertz.
    pub pitch_hz: f32,
}

/// The typed errors a prosody-plan operation can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    /// The supplied arrays do not both have length `N`.
    LengthMismatch,
    /// The requested phoneme index is at or past `N`.
    IndexOutOfRange,
}

/// The typed editable prosody plan.
///
/// `schema_version` is a required serde field — JSON that omits it fails to
/// deserialize rather than silently defaulting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyPlan {
    /// The schema version this plan was written against.
    pub schema_version: u32,
    /// Per-phoneme durations, in milliseconds. Length `N`.
    pub durations_ms: Vec<f32>,
    /// Per-phoneme pitches, in hertz. Length `N`.
    pub pitch_hz: Vec<f32>,
}

impl ProsodyPlan {
    /// Construct a plan for `n` phonemes from caller-supplied arrays.
    ///
    /// Both `durations_ms` and `pitch_hz` must have length `n`; otherwise this
    /// returns [`PlanError::LengthMismatch`]. The plan is stamped with the
    /// current [`PLAN_SCHEMA_VERSION`].
    pub fn new(
        n: usize,
        durations_ms: Vec<f32>,
        pitch_hz: Vec<f32>,
    ) -> Result<ProsodyPlan, PlanError> {
        if durations_ms.len() != n || pitch_hz.len() != n {
            return Err(PlanError::LengthMismatch);
        }
        Ok(ProsodyPlan {
            schema_version: PLAN_SCHEMA_VERSION,
            durations_ms,
            pitch_hz,
        })
    }

    /// Return the phoneme at index `i`.
    ///
    /// `i` at or past `N` yields [`PlanError::IndexOutOfRange`]; this never
    /// panics on any `usize`.
    pub fn phoneme(&self, i: usize) -> Result<Phoneme, PlanError> {
        if i >= self.durations_ms.len() {
            return Err(PlanError::IndexOutOfRange);
        }
        Ok(Phoneme {
            duration_ms: self.durations_ms[i],
            pitch_hz: self.pitch_hz[i],
        })
    }

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
