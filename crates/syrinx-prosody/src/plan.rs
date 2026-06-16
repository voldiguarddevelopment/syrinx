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
    /// The requested rate factor was not strictly positive.
    InvalidRate,
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
        if durations_ms.len() != n {
            return Err(PlanError::LengthMismatch);
        }
        if pitch_hz.len() != n {
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
        match (self.durations_ms.get(i), self.pitch_hz.get(i)) {
            (Some(&duration_ms), Some(&pitch_hz)) => Ok(Phoneme {
                duration_ms,
                pitch_hz,
            }),
            _ => Err(PlanError::IndexOutOfRange),
        }
    }
}
