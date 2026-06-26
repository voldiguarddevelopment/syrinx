//! syrinx-vocoder — HiFi-GAN/Vocos waveform synthesis (scaffold; T-00.01).
//!
//! T-07.04 adds the deterministic anti-alias [`band_limit`] used by
//! `syrinx-stream`'s 48kHz→8kHz downsampler: a length-`taps` moving-average
//! (boxcar) low-pass. A boxcar of length equal to the decimation factor is the
//! classic decimation-by-averaging anti-alias pre-filter — its first spectral
//! null sits at `src_rate / taps` (the post-decimation sample rate), so energy
//! near and above the narrowband Nyquist is suppressed while the DC/low-frequency
//! passband is preserved with unity gain. Boundaries are edge-replicated
//! (clamped), so a constant (DC) signal stays exactly constant, including at the
//! buffer edges.

// The real CosyVoice2 HiFT vocoder forward via Candle (mel -> 24 kHz waveform).
// Gated behind the `real` feature; built + parity-verified on the model box.
#[cfg(feature = "real")]
pub mod real;

// The real CosyVoice3 CausalHiFTGenerator forward via Candle (causal convs + f64
// f0_predictor). Additive to the CV2 `real` module; same `real` feature gate.
#[cfg(feature = "real")]
pub mod real_cv3;

/// Band-limit `input` with a length-`taps` moving-average (boxcar) low-pass.
///
/// Each output sample is the mean of the `taps` input samples starting at that
/// index; indices past the end are clamped to the last sample (edge replication),
/// so a DC input maps to itself with unity gain everywhere. The unit-gain mean
/// preserves the low-frequency passband while attenuating energy near the boxcar's
/// first null at `rate / taps`. Returns an output the same length as `input`
/// (empty in, empty out); never panics.
pub fn band_limit(input: &[f32], taps: usize) -> Vec<f32> {
    let n = input.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = 0.0f32;
        for k in 0..taps {
            let idx = (i + k).min(n - 1);
            acc += input[idx];
        }
        out.push(acc / taps as f32);
    }
    out
}
