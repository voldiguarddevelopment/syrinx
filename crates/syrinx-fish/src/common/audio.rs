//! 44.1 kHz wav write + band-limited resample helpers for the Fish pipeline.
//!
//! A small self-contained copy (not a re-export of `syrinx-serve::wavio`) to avoid a
//! cross-crate dependency cycle; the resampler is the same Lanczos-windowed-sinc design.
//! The codec runs at **44.1 kHz** (vs CosyVoice's 24 kHz), so these are Fish-specific.

use std::path::Path;

/// The Fish codec's output sample rate.
pub const SAMPLE_RATE_44K: u32 = 44_100;

/// An audio I/O error (wrapping `hound`).
#[derive(Debug)]
pub enum AudioError {
    /// Underlying `hound` (WAV) error, with context.
    Wav(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Wav(m) => write!(f, "wav: {m}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// A decoded mono waveform plus its original sample rate.
pub struct MonoWav {
    /// Samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Original sample rate (Hz).
    pub sample_rate: u32,
}

/// Read a WAV file, down-mix to mono `f32` in `[-1, 1]`, preserving the original sample
/// rate. Integer (8/16/24/32-bit) and 32-bit-float PCM are decoded; channels are averaged.
pub fn read_wav_mono(path: impl AsRef<Path>) -> Result<MonoWav, AudioError> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| AudioError::Wav(format!("open {}: {e}", path.display())))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| AudioError::Wav(format!("decode float wav: {e}")))?,
        hound::SampleFormat::Int => {
            let full_scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / full_scale))
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|e| AudioError::Wav(format!("decode int wav: {e}")))?
        }
    };

    // Average channel-interleaved samples to mono.
    let samples = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok(MonoWav {
        samples,
        sample_rate: spec.sample_rate,
    })
}

/// Read a reference WAV and resample it to 44.1 kHz mono `f32` (the rate the codec
/// `encode` expects for cloning).
pub fn read_ref_wav_44k(path: impl AsRef<Path>) -> Result<Vec<f32>, AudioError> {
    let mono = read_wav_mono(path)?;
    Ok(resample(&mono.samples, mono.sample_rate, SAMPLE_RATE_44K))
}

/// Write a mono `f32` waveform to a 32-bit-float WAV at `sample_rate` (lossless — the
/// codec output is full-range f32). Samples are written verbatim, not clamped.
pub fn write_wav(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), AudioError> {
    let path = path.as_ref();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| AudioError::Wav(format!("create {}: {e}", path.display())))?;
    for &s in samples {
        writer
            .write_sample(s)
            .map_err(|e| AudioError::Wav(format!("write sample: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| AudioError::Wav(format!("finalize {}: {e}", path.display())))?;
    Ok(())
}

/// Write a mono `f32` waveform to a 44.1 kHz 32-bit-float WAV (the codec's native rate).
pub fn write_wav_44k(path: impl AsRef<Path>, samples: &[f32]) -> Result<(), AudioError> {
    write_wav(path, samples, SAMPLE_RATE_44K)
}

/// Band-limited resample `input` from `in_sr` to `out_sr` (Lanczos-windowed sinc, 16
/// lobes). Identity when the rates match; lowers the cutoff on down-sampling for
/// anti-aliasing. Same design as `syrinx-serve::wavio::resample`.
pub fn resample(input: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    if in_sr == out_sr || input.is_empty() {
        return input.to_vec();
    }
    let ratio = out_sr as f64 / in_sr as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    // Anti-alias cutoff (normalised to input rate): 1.0 up-sampling, < 1 down-sampling.
    let cutoff = ratio.min(1.0);
    const LOBES: f64 = 16.0;
    let radius = (LOBES / cutoff).ceil() as i64;

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        // Center in input-sample coordinates.
        let center = i as f64 / ratio;
        let i0 = center.floor() as i64;
        let mut acc = 0f64;
        let mut wsum = 0f64;
        for k in (i0 - radius)..=(i0 + radius) {
            if k < 0 || k as usize >= input.len() {
                continue;
            }
            let x = (center - k as f64) * cutoff;
            let w = lanczos(x, LOBES) * cutoff;
            acc += input[k as usize] as f64 * w;
            wsum += w;
        }
        let v = if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 };
        out.push(v as f32);
    }
    out
}

/// `sin(pi x) / (pi x)`, with the removable singularity at 0.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Lanczos kernel `sinc(x) * sinc(x / a)` inside `|x| < a`, else 0.
fn lanczos(x: f64, a: f64) -> f64 {
    if x.abs() >= a {
        0.0
    } else {
        sinc(x) * sinc(x / a)
    }
}
