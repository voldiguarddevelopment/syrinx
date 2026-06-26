//! Render-time speech-rate control — the audio-affecting half of the rate knob
//! (DESIGN T3.4: "rate scales without pitch shift").
//!
//! [`crate::rate::scale_rate`] scales the *editable plan's* `durations_ms`
//! array — useful for the plan editor, but it does not by itself change a
//! rendered waveform. This module is the transform that actually alters the
//! audio: it time-scales the generated **mel-spectrogram** along its frame axis
//! before the vocoder runs.
//!
//! ## Why time-scaling the mel preserves pitch
//!
//! In the CosyVoice2 acoustic stack the mel is `[n_mels, T]`: each *column* is
//! one frame's spectral envelope (which mel bands carry energy), and the
//! *number of columns* sets the utterance's duration (the HiFT vocoder maps a
//! fixed hop per frame). Pitch lives in the per-frame band distribution, not in
//! the frame count. So resampling along the frame axis by a factor `1/rate`
//! (fewer frames ⇒ shorter/faster; more frames ⇒ longer/slower) changes
//! duration while leaving each retained/interpolated column's spectral shape —
//! and hence pitch — intact. This is the classic length-regulator move and is
//! the standard pitch-preserving time-scale for a frame-rate mel.
//!
//! The transform here is deterministic linear interpolation along time. It does
//! not touch the band (pitch) axis. The synth layer converts the Candle mel
//! tensor to/from the flat row-major grid this module operates on, keeping
//! `syrinx-prosody` free of any tensor/Candle dependency.

/// The error a render-time rate transform can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderError {
    /// The supplied rate factor was not strictly positive and finite.
    InvalidRate,
    /// The mel grid was empty or ragged (rows of differing length), so the
    /// frame count is undefined.
    BadMelShape,
}

/// Time-scale a mel-spectrogram by speech-rate factor `r`, preserving pitch.
///
/// `mel` is `[n_mels][T]` (one row per mel band, `T` frames per row). The result
/// is `[n_mels][T_out]` with `T_out == max(1, round(T / r))`: `r > 1.0` speeds
/// up (fewer frames, shorter audio), `r < 1.0` slows down (more frames, longer
/// audio), and `r == 1.0` is the identity up to rounding (`T_out == T`). Each
/// output column `j` is sampled from the input time axis at position
/// `j * (T - 1) / (T_out - 1)` with linear interpolation between the two nearest
/// input frames; the band axis is never touched, so the per-frame spectral shape
/// — and thus the pitch — is preserved.
///
/// `r <= 0.0` or a non-finite `r` yields [`RenderError::InvalidRate`]; an empty
/// or ragged grid yields [`RenderError::BadMelShape`]. Never panics.
pub fn time_scale_mel(mel: &[Vec<f32>], r: f64) -> Result<Vec<Vec<f32>>, RenderError> {
    if !(r.is_finite() && r > 0.0) {
        return Err(RenderError::InvalidRate);
    }
    let n_mels = mel.len();
    if n_mels == 0 {
        return Err(RenderError::BadMelShape);
    }
    let t_in = mel[0].len();
    if t_in == 0 || mel.iter().any(|row| row.len() != t_in) {
        return Err(RenderError::BadMelShape);
    }

    // Output length: round the time axis by 1/r, at least one frame.
    let t_out = (((t_in as f64) / r).round() as usize).max(1);

    let mut out: Vec<Vec<f32>> = vec![vec![0.0f32; t_out]; n_mels];

    // Degenerate single-frame input or output: hold the first input frame across
    // every output column — the constant limit of the interpolation (the
    // `(t_out - 1)` denominator below would otherwise divide by zero).
    if t_in == 1 || t_out == 1 {
        for (m, row) in mel.iter().enumerate() {
            for slot in out[m].iter_mut() {
                *slot = row[0];
            }
        }
        return Ok(out);
    }

    let scale = (t_in as f64 - 1.0) / (t_out as f64 - 1.0);
    for j in 0..t_out {
        let src = j as f64 * scale; // position on the input time axis
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(t_in - 1);
        let frac = (src - i0 as f64) as f32;
        for m in 0..n_mels {
            let a = mel[m][i0];
            let b = mel[m][i1];
            out[m][j] = a + (b - a) * frac;
        }
    }
    Ok(out)
}
