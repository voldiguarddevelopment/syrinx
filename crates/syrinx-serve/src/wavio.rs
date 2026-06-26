//! WAV I/O + resampling for the `real` synthesizer surfaces (CLI + server).
//!
//! The [`Synthesizer`](crate::synth::Synthesizer) deliberately takes its reference
//! voice as two already-resampled mono waveforms (`ref_wav_16k`, `ref_wav_24k`) —
//! resampling is the caller's job so the parity tests can feed the *exact*
//! reference-resampled buffers (see `synth::Synthesizer::prompt_cond`). These
//! helpers are that caller-side glue, shared by the CLI and the server so there is
//! one WAV reader, one resampler, and one 24 kHz WAV encoder.
//!
//! Compiled only under the crate's `real` feature (alongside [`crate::synth`]);
//! the default Axum scaffold build stays Candle-free and `hound`-free.

use std::io::Cursor;
use std::path::Path;

use crate::synth::SynthError;

/// Reference-voice sample rates the synthesizer consumes.
const SR_16K: u32 = 16_000;
const SR_24K: u32 = 24_000;

/// Read a WAV file, down-mix to mono `f32` in `[-1, 1]`, and return the resampled
/// `(ref_wav_16k, ref_wav_24k)` pair the synthesizer expects.
///
/// Accepts any sample rate and channel count; integer (8/16/24/32-bit) and 32-bit
/// float PCM are decoded and normalized to `f32`. Channels are averaged to mono
/// (CosyVoice2's `torchaudio.load -> mono` step). Resampling is a band-limited
/// windowed-sinc (see [`resample`]).
pub fn read_ref_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, Vec<f32>), SynthError> {
    let mono = read_wav_mono(path.as_ref())?;
    let w16 = resample(&mono.samples, mono.sample_rate, SR_16K);
    let w24 = resample(&mono.samples, mono.sample_rate, SR_24K);
    Ok((w16, w24))
}

/// A decoded mono waveform plus its original sample rate.
struct MonoWav {
    samples: Vec<f32>,
    sample_rate: u32,
}

/// Decode a WAV file to a mono `f32` waveform in `[-1, 1]`.
fn read_wav_mono(path: &Path) -> Result<MonoWav, SynthError> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| SynthError::Candle(format!("open wav {}: {e}", path.display())))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    // Decode every channel-interleaved sample to f32 in [-1, 1].
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| SynthError::Candle(format!("decode float wav: {e}")))?,
        hound::SampleFormat::Int => {
            // Normalize by the full-scale magnitude for the declared bit depth.
            let full_scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / full_scale))
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|e| SynthError::Candle(format!("decode int wav: {e}")))?
        }
    };

    // Down-mix interleaved channels to mono by averaging.
    let samples: Vec<f32> = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    };

    Ok(MonoWav {
        samples,
        sample_rate: spec.sample_rate,
    })
}

/// Band-limited resample of a mono signal from `in_sr` to `out_sr`.
///
/// A Lanczos-windowed sinc kernel (radius `a = 16` lobes). When down-sampling, the
/// kernel cutoff is lowered to the output Nyquist so the sinc doubles as the
/// anti-alias filter (the analog of `torchaudio.resample`'s windowed-sinc); when
/// up-sampling the cutoff is the input Nyquist. Each output sample normalizes by the
/// sum of its kernel weights, which holds passband/DC gain at 1 and tames the
/// edge-of-signal truncation. Identity (`in_sr == out_sr`) is a copy.
///
/// This is a faithful, dependency-free resampler — good enough for prompt
/// conditioning (the prompt waveforms only drive the speaker x-vector, the prompt
/// speech-tokens, and the prompt mel). It is not claimed bit-identical to torch's
/// Kaiser-windowed resampler; the e2e parity path feeds the reference-resampled
/// buffers directly instead.
pub fn resample(input: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    if input.is_empty() || in_sr == 0 || out_sr == 0 {
        return Vec::new();
    }
    if in_sr == out_sr {
        return input.to_vec();
    }

    let ratio = out_sr as f64 / in_sr as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;

    // Window radius (in lobes) and the normalized cutoff (cycles per input sample).
    let a = 16.0_f64;
    let cutoff = if out_sr < in_sr {
        out_sr as f64 / in_sr as f64
    } else {
        1.0
    };
    // Kernel half-width in input samples grows as the cutoff shrinks.
    let half = (a / cutoff).ceil() as isize;
    let n = input.len() as isize;

    let mut out = Vec::with_capacity(out_len);
    for o in 0..out_len {
        // Continuous source position for this output sample.
        let center = o as f64 / ratio;
        let i0 = center.floor() as isize;
        let mut acc = 0.0_f64;
        let mut wsum = 0.0_f64;
        for k in (i0 - half + 1)..=(i0 + half) {
            if k < 0 || k >= n {
                continue;
            }
            let x = (center - k as f64) * cutoff;
            let w = lanczos(x, a) * cutoff;
            acc += input[k as usize] as f64 * w;
            wsum += w;
        }
        let v = if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 };
        out.push(v as f32);
    }
    out
}

/// `sinc(x) = sin(pi x) / (pi x)`, with the removable singularity at 0 filled.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let p = std::f64::consts::PI * x;
        p.sin() / p
    }
}

/// Lanczos kernel of radius `a`: `sinc(x) * sinc(x / a)` inside `|x| < a`, else 0.
fn lanczos(x: f64, a: f64) -> f64 {
    if x.abs() < a {
        sinc(x) * sinc(x / a)
    } else {
        0.0
    }
}

/// Encode a 24 kHz mono `f32` waveform to in-memory 16-bit PCM WAV bytes.
///
/// 16-bit signed PCM mono at 24 kHz is the universally-playable choice and the
/// rate the synthesizer emits. Samples are clamped to `[-1, 1]` before scaling.
pub fn encode_wav_24k(samples: &[f32]) -> Result<Vec<u8>, SynthError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR_24K,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| SynthError::Candle(format!("init wav encoder: {e}")))?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            writer
                .write_sample(v)
                .map_err(|e| SynthError::Candle(format!("write wav sample: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| SynthError::Candle(format!("finalize wav: {e}")))?;
    }
    Ok(cursor.into_inner())
}

/// Encode a 24 kHz mono waveform and write it to `path` as a 16-bit PCM WAV.
pub fn write_wav_24k(path: impl AsRef<Path>, samples: &[f32]) -> Result<(), SynthError> {
    let bytes = encode_wav_24k(samples)?;
    std::fs::write(path.as_ref(), bytes)
        .map_err(|e| SynthError::Candle(format!("write wav {}: {e}", path.as_ref().display())))
}
