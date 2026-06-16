//! Intonation-contour application on the typed prosody plan (T-03.07).
//!
//! Two edits on an existing [`ProsodyPlan`], both touching only `pitch_hz` and
//! never `durations_ms`:
//!
//!   * [`apply_contour`](ProsodyPlan::apply_contour) — apply a named [`Contour`]
//!     preset to the pitch array as a deterministic additive linear ramp over the
//!     `N` phonemes. The preset's specified delta is [`CONTOUR_DELTA_HZ`]:
//!     [`Contour::Rising`] adds a `0 → +delta` ramp (first unchanged, last `+delta`
//!     above it), [`Contour::Falling`] the mirror `0 → -delta` ramp, and
//!     [`Contour::Flat`] an all-zero ramp (identity). Interior phonemes are
//!     linearly interpolated.
//!   * [`apply_curve`](ProsodyPlan::apply_curve) — apply a manual per-phoneme F0
//!     curve. A curve of length `N` sets `pitch_hz` element-for-element equal to
//!     the supplied curve; a curve whose length `!= N` yields
//!     [`PlanError::LengthMismatch`] and mutates nothing.
//!
//! An empty plan (`N == 0`) is a no-op `Ok` for every preset and for an empty
//! manual curve; nothing panics.
//!
//! Scope (list.md / T-03.07): contour shape over the pitch array only — no emotion
//! semantics and no per-phoneme/per-word pitch API (that is T-03.03). Whether the
//! applied contour is perceptually the intended intonation on rendered audio is a
//! deferred perceptual eval against the real model, not gated here.

use crate::plan::{PlanError, ProsodyPlan};

/// The contour preset's specified delta, in Hz: [`Contour::Rising`] raises the last
/// pitch entry this far above the first (on a flat input), [`Contour::Falling`]
/// lowers it this far below.
pub const CONTOUR_DELTA_HZ: f32 = 30.0;

/// A named intonation-contour preset applied to a plan's pitch array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contour {
    /// A `0 → +delta` ramp: pitch rises across the array.
    Rising,
    /// A `0 → -delta` ramp: pitch falls across the array.
    Falling,
    /// An all-zero ramp: pitch is left unchanged (identity).
    Flat,
}

impl ProsodyPlan {
    /// Apply a named contour preset to `pitch_hz` as an additive linear ramp.
    ///
    /// The ramp runs `0` at the first phoneme to the preset's signed total at the
    /// last (`+`[`CONTOUR_DELTA_HZ`] for [`Contour::Rising`], its negation for
    /// [`Contour::Falling`], `0` for [`Contour::Flat`]), with interior phonemes
    /// linearly interpolated; the ramp is added to the existing pitch. The first
    /// entry is always left unchanged. An empty plan (`N == 0`) is a no-op `Ok`.
    /// `durations_ms` is never touched; this never panics.
    pub fn apply_contour(&mut self, contour: Contour) -> Result<(), PlanError> {
        let total = match contour {
            Contour::Rising => CONTOUR_DELTA_HZ,
            Contour::Falling => -CONTOUR_DELTA_HZ,
            Contour::Flat => 0.0,
        };
        let n = self.pitch_hz.len();
        if n == 0 {
            return Ok(());
        }
        let denom = (n - 1).max(1) as f32;
        for i in 0..n {
            self.pitch_hz[i] += total * (i as f32) / denom;
        }
        Ok(())
    }

    /// Set `pitch_hz` element-for-element from a manual per-phoneme F0 curve.
    ///
    /// On success `pitch_hz[i] == curve[i]` for every `i` and `durations_ms` is
    /// untouched. A curve whose length `!= N` yields
    /// [`PlanError::LengthMismatch`] and mutates nothing; this never panics.
    pub fn apply_curve(&mut self, curve: &[f32]) -> Result<(), PlanError> {
        if curve.len() != self.pitch_hz.len() {
            return Err(PlanError::LengthMismatch);
        }
        self.pitch_hz.copy_from_slice(curve);
        Ok(())
    }
}
