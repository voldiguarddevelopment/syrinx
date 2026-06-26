//! The **render-level** editable prosody plan — the audio-affecting pitch +
//! duration control surface (DESIGN §6 Phase 3 "editable prosody + control").
//!
//! This complements the per-phoneme [`crate::plan::ProsodyPlan`] (an abstract
//! `durations_ms`/`pitch_hz` model used by the plan *editor*): a [`RenderPlan`]
//! is expressed in the units the CosyVoice2 acoustic stack actually exposes —
//! **generated-mel frames** — and maps directly onto the synth's generated mel +
//! HiFT F0 source *before the vocoder*, so it genuinely changes the rendered
//! waveform. See `docs/PITCH-DURATION.md` for the feasibility survey (what is
//! faithful without retraining vs what needs alignment/training).
//!
//! ## The two faithful, training-free levers (and one opt-in lower-fidelity one)
//!
//!   * **Duration** — time-scale the mel along its frame axis. Pitch-preserving
//!     and exact: a global `rate`, plus per-[`Region`] `rate` overrides on frame
//!     ranges (variable-rate time-warp).
//!   * **Pitch (default)** — multiply the per-frame F0 fed to the HiFT source by
//!     `2^(semitones/12)`. The HiFT vocoder is a neural source-filter: the source
//!     carries the harmonic pitch and the mel carries the formant envelope, so
//!     scaling the source F0 is a **formant-preserving** pitch shift. Global
//!     `pitch_semitones`, plus per-[`Region`] overrides.
//!   * **Pitch (opt-in, lower fidelity)** — [`mel_envelope_shift`](RenderPlan::mel_envelope_shift)
//!     additionally warps the mel along the bin axis by the global ratio. This
//!     shifts the *formants too* (the "chipmunk" artifact) and is off by default;
//!     it exists for A/B and deliberate timbre change, not faithful pitch.
//!
//! Everything here is pure Rust / Candle-free: it operates on a `[n_mels][T]`
//! grid and returns a new grid + a per-output-frame F0 multiplier. The synth
//! layer (`syrinx-serve`) converts to/from the Candle mel tensor and runs the
//! F0 predictor + vocoder.

use serde::{Deserialize, Serialize};

/// The current render-plan schema version, stamped on every constructed plan.
pub const RENDER_PLAN_SCHEMA_VERSION: u32 = 1;

/// A contiguous **generated-mel frame range** `[start_frame, end_frame)` carrying
/// optional rate and/or pitch overrides for that span.
///
/// A `None` field falls back to the plan's global value for those frames; a `Some`
/// field **replaces** the global value over the range. Ranges are in original
/// (pre-time-scale) mel frames; when ranges overlap, the **last** region in
/// [`RenderPlan::regions`] wins for the overlapping frames.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Region {
    /// First frame of the range, inclusive (original mel frame index).
    pub start_frame: usize,
    /// One past the last frame of the range, exclusive.
    pub end_frame: usize,
    /// Per-region speech rate (`>1` faster/shorter, `<1` slower/longer), replacing
    /// the global rate over this range, or `None` to keep the global rate.
    #[serde(default)]
    pub rate: Option<f64>,
    /// Per-region pitch shift in semitones, replacing the global pitch shift over
    /// this range, or `None` to keep the global pitch shift.
    #[serde(default)]
    pub pitch_semitones: Option<f64>,
}

/// A typed, JSON-round-tripping render-level prosody plan.
///
/// `schema_version` is a required serde field — JSON that omits it fails to
/// deserialize rather than silently defaulting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderPlan {
    /// The schema version this plan was written against.
    pub schema_version: u32,
    /// Global speech rate applied to all frames not covered by a region's `rate`
    /// override: `>1` faster/shorter, `<1` slower/longer, `1.0` unchanged.
    pub global_rate: f64,
    /// Global pitch shift in semitones applied to all frames not covered by a
    /// region's `pitch_semitones` override: `+` raises, `-` lowers, `0.0` unchanged.
    pub global_pitch_semitones: f64,
    /// Opt-in: also warp the mel along its bin axis by the global pitch ratio
    /// (a *formant-shifting*, lower-fidelity pitch lever — see the module docs).
    /// Off (`false`) by default; the F0 source is the faithful pitch lever.
    #[serde(default)]
    pub mel_envelope_shift: bool,
    /// Per-frame-range overrides (last wins on overlap).
    #[serde(default)]
    pub regions: Vec<Region>,
}

/// The typed errors a render-plan operation can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPlanError {
    /// A rate factor (global or region) was not finite and strictly positive.
    InvalidRate,
    /// A pitch shift (global or region) was not finite.
    InvalidPitch,
    /// A region range was empty (`start >= end`) or extended past the mel length.
    InvalidRegion,
    /// The mel grid was empty or ragged (rows of differing length).
    BadMelShape,
}

/// Convert a pitch shift in semitones to a linear frequency ratio: `2^(st/12)`.
///
/// `0` semitones is the identity `1.0`; `+12` doubles the pitch, `-12` halves it.
pub fn semitones_to_ratio(semitones: f64) -> f64 {
    2.0f64.powf(semitones / 12.0)
}

impl RenderPlan {
    /// The identity plan: rate `1.0`, pitch `0` semitones, no regions, no mel
    /// shift. `apply` on this is a frame-for-frame passthrough (up to rounding)
    /// with an all-ones F0 multiplier.
    pub fn identity() -> RenderPlan {
        RenderPlan {
            schema_version: RENDER_PLAN_SCHEMA_VERSION,
            global_rate: 1.0,
            global_pitch_semitones: 0.0,
            mel_envelope_shift: false,
            regions: Vec::new(),
        }
    }

    /// Builder: set the global speech rate.
    pub fn with_global_rate(mut self, rate: f64) -> RenderPlan {
        self.global_rate = rate;
        self
    }

    /// Builder: set the global pitch shift in semitones.
    pub fn with_global_pitch_semitones(mut self, semitones: f64) -> RenderPlan {
        self.global_pitch_semitones = semitones;
        self
    }

    /// Builder: enable/disable the opt-in formant-shifting mel-bin pitch lever.
    pub fn with_mel_envelope_shift(mut self, on: bool) -> RenderPlan {
        self.mel_envelope_shift = on;
        self
    }

    /// Builder: append a per-frame-range [`Region`] override.
    pub fn add_region(mut self, region: Region) -> RenderPlan {
        self.regions.push(region);
        self
    }

    /// Validate the plan against a mel of `t_frames` frames.
    ///
    /// Checks the global rate is finite `> 0`, the global pitch is finite, and
    /// every region is a non-empty range within `[0, t_frames]` with a finite
    /// `> 0` rate override and a finite pitch override where present. Returns the
    /// first failure; never panics.
    pub fn validate(&self, t_frames: usize) -> Result<(), RenderPlanError> {
        if !(self.global_rate.is_finite() && self.global_rate > 0.0) {
            return Err(RenderPlanError::InvalidRate);
        }
        if !self.global_pitch_semitones.is_finite() {
            return Err(RenderPlanError::InvalidPitch);
        }
        for r in &self.regions {
            if r.start_frame >= r.end_frame || r.end_frame > t_frames {
                return Err(RenderPlanError::InvalidRegion);
            }
            if let Some(rate) = r.rate {
                if !(rate.is_finite() && rate > 0.0) {
                    return Err(RenderPlanError::InvalidRate);
                }
            }
            if let Some(p) = r.pitch_semitones {
                if !p.is_finite() {
                    return Err(RenderPlanError::InvalidPitch);
                }
            }
        }
        Ok(())
    }

    /// The per-original-frame effective rate (global, overridden per region).
    ///
    /// Length `t_frames`; entry `i` is the region rate if frame `i` falls in a
    /// region with a `Some(rate)` (last region wins), else `global_rate`.
    pub fn rate_profile(&self, t_frames: usize) -> Vec<f64> {
        let mut prof = vec![self.global_rate; t_frames];
        for r in &self.regions {
            if let Some(rate) = r.rate {
                let end = r.end_frame.min(t_frames);
                for slot in prof.iter_mut().take(end).skip(r.start_frame.min(end)) {
                    *slot = rate;
                }
            }
        }
        prof
    }

    /// The per-original-frame F0 multiplier (global, overridden per region).
    ///
    /// Length `t_frames`; entry `i` is `2^(semitones/12)` for the effective
    /// semitone shift at frame `i` (region override if present, last region wins,
    /// else `global_pitch_semitones`).
    pub fn pitch_ratio_profile(&self, t_frames: usize) -> Vec<f64> {
        let g = semitones_to_ratio(self.global_pitch_semitones);
        let mut prof = vec![g; t_frames];
        for r in &self.regions {
            if let Some(semi) = r.pitch_semitones {
                let ratio = semitones_to_ratio(semi);
                let end = r.end_frame.min(t_frames);
                for slot in prof.iter_mut().take(end).skip(r.start_frame.min(end)) {
                    *slot = ratio;
                }
            }
        }
        prof
    }

    /// Apply the plan to a generated mel grid, returning the transformed mel and a
    /// per-output-frame F0 multiplier.
    ///
    /// `mel` is `[n_mels][T]` (one row per mel band, `T` frames). The result is
    /// `(mel_out [n_mels][T_out], f0_mult [T_out])`:
    ///
    ///   * the frame axis is time-warped by the per-frame rate profile (global +
    ///     per-region), so `T_out ≈ Σ 1/rate[i]` — slower spans add frames, faster
    ///     spans drop them — with each column's spectral shape (pitch) preserved;
    ///   * `f0_mult[j]` is the F0 multiplier for output frame `j`, the pitch-ratio
    ///     profile resampled along the *same* time-warp, so the synth can scale the
    ///     predicted F0 frame-for-frame;
    ///   * if [`mel_envelope_shift`](RenderPlan::mel_envelope_shift) is set, the
    ///     warped mel is additionally warped along the bin axis by the global pitch
    ///     ratio (the opt-in formant-shifting lever).
    ///
    /// Returns [`RenderPlanError::BadMelShape`] for an empty/ragged grid and the
    /// validation errors for a bad global/region knob. Never panics.
    pub fn apply(&self, mel: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<f64>), RenderPlanError> {
        let n_mels = mel.len();
        if n_mels == 0 {
            return Err(RenderPlanError::BadMelShape);
        }
        let t_in = mel[0].len();
        if t_in == 0 || mel.iter().any(|row| row.len() != t_in) {
            return Err(RenderPlanError::BadMelShape);
        }
        self.validate(t_in)?;

        let rate = self.rate_profile(t_in);
        let pitch = self.pitch_ratio_profile(t_in);

        // Variable-rate time-warp. Each input frame i contributes 1/rate[i] output
        // frames; cum[i] is the cumulative output coordinate at input frame i.
        let mut cum = vec![0f64; t_in + 1];
        for i in 0..t_in {
            cum[i + 1] = cum[i] + 1.0 / rate[i];
        }
        let total_out = cum[t_in];
        let t_out = (total_out.round() as usize).max(1);

        let mut out: Vec<Vec<f32>> = vec![vec![0.0f32; t_out]; n_mels];
        let mut f0_mult = vec![1.0f64; t_out];

        for j in 0..t_out {
            // Output frame j sits at cumulative coordinate c on [0, total_out].
            let c = if t_out == 1 {
                0.0
            } else {
                j as f64 * total_out / t_out as f64
            };
            // Invert the (monotone) cumulative map to a fractional source frame.
            let src = invert_cum(&cum, &rate, c, t_in);
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(t_in - 1);
            let frac = (src - i0 as f64) as f32;
            for m in 0..n_mels {
                let a = mel[m][i0];
                let b = mel[m][i1];
                out[m][j] = a + (b - a) * frac;
            }
            let pa = pitch[i0];
            let pb = pitch[i1];
            f0_mult[j] = pa + (pb - pa) * frac as f64;
        }

        if self.mel_envelope_shift {
            let ratio = semitones_to_ratio(self.global_pitch_semitones);
            out = shift_mel_bins(&out, ratio);
        }

        Ok((out, f0_mult))
    }
}

/// Invert the cumulative-output map `cum` to a fractional input-frame position for
/// cumulative coordinate `c`. `cum` is monotone non-decreasing of length `t_in+1`;
/// `rate[i] = 1/(cum[i+1]-cum[i])`. Clamped to `[0, t_in-1]`. Never panics.
fn invert_cum(cum: &[f64], rate: &[f64], c: f64, t_in: usize) -> f64 {
    if t_in <= 1 {
        return 0.0;
    }
    let c = c.clamp(0.0, cum[t_in]);
    // Binary search for the segment [cum[i], cum[i+1]) containing c.
    let mut lo = 0usize;
    let mut hi = t_in - 1; // last valid segment index
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if cum[mid] <= c {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let i = lo;
    // Within segment i: src = i + (c - cum[i]) * rate[i].
    let frac = (c - cum[i]) * rate[i];
    (i as f64 + frac).clamp(0.0, (t_in - 1) as f64)
}

/// Warp a mel grid along its **bin** (frequency) axis by frequency ratio `ratio`,
/// approximating a pitch shift in mel-bin space: output bin `b` samples input bin
/// `b/ratio` with linear interpolation, clamped to the band range.
///
/// `ratio > 1` moves energy to higher bins (raises pitch *and formants*); `ratio <
/// 1` lowers them. This is the lower-fidelity, **formant-shifting** lever (see the
/// module docs) — it cannot separate source from filter, so it changes timbre. A
/// non-finite or non-positive `ratio`, or the identity `1.0`, returns the grid
/// unchanged. Never panics.
pub fn shift_mel_bins(mel: &[Vec<f32>], ratio: f64) -> Vec<Vec<f32>> {
    let n_mels = mel.len();
    if n_mels == 0 || !(ratio.is_finite() && ratio > 0.0) || (ratio - 1.0).abs() < f64::EPSILON {
        return mel.to_vec();
    }
    let t = mel[0].len();
    let mut out = vec![vec![0.0f32; t]; n_mels];
    for b in 0..n_mels {
        let src = b as f64 / ratio; // input bin feeding output bin b
        let b0 = src.floor().clamp(0.0, (n_mels - 1) as f64) as usize;
        let b1 = (b0 + 1).min(n_mels - 1);
        let frac = (src - b0 as f64) as f32;
        for col in 0..t {
            let a = mel[b0][col];
            let c = mel[b1][col];
            out[b][col] = a + (c - a) * frac;
        }
    }
    out
}
