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

/// A typed single-phoneme edit carrying an optional new duration and/or pitch.
///
/// A `Some` field requests writing that field at the target index; a `None`
/// field leaves the corresponding entry of the plan equal to the original. The
/// two fields are written independently (T-03.09).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PhonemeEdit {
    /// The new duration in milliseconds, or `None` to leave duration unchanged.
    pub duration_ms: Option<f32>,
    /// The new pitch in hertz, or `None` to leave pitch unchanged.
    pub pitch_hz: Option<f32>,
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

    /// Apply a single-phoneme [`PhonemeEdit`] at index `i`, returning a new plan.
    ///
    /// The returned plan carries exactly the edited values at index `i` and is
    /// bit-identical to the original everywhere else: a `Some` field is written,
    /// a `None` field leaves that entry equal to the original, and the two fields
    /// are written independently. `i` at or past `N` yields
    /// [`PlanError::IndexOutOfRange`] and mutates nothing — `&self` is never
    /// modified and this never panics on any `usize`. The schema version is
    /// carried over.
    pub fn edit_phoneme(&self, i: usize, edit: PhonemeEdit) -> Result<ProsodyPlan, PlanError> {
        if i >= self.durations_ms.len() {
            return Err(PlanError::IndexOutOfRange);
        }
        let mut plan = self.clone();
        if let Some(duration_ms) = edit.duration_ms {
            plan.durations_ms[i] = duration_ms;
        }
        if let Some(pitch_hz) = edit.pitch_hz {
            plan.pitch_hz[i] = pitch_hz;
        }
        Ok(plan)
    }
}
