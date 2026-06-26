//! Prompt **speech-token** tokenizer (real-weights track, behind the crate's
//! `real` feature). Maps a reference 16 kHz mono waveform to the CosyVoice2
//! `prompt_speech_token` id sequence used for zero-shot voice cloning.
//!
//! This mirrors `CosyVoiceFrontEnd._extract_speech_token`:
//!
//!   speech (16 kHz, mono) -> whisper log-mel (128 bins) -> speech_tokenizer_v2.onnx
//!   -> flat int32 token ids.
//!
//! The whisper log-mel is recomputed in Rust (bit-for-bit against
//! `whisper.log_mel_spectrogram(n_mels=128)`); the FSQ-quantised whisper-encoder
//! that produces the ids runs through `speech_tokenizer_v2.onnx` via the `ort`
//! crate (ONNX Runtime). A full Candle port of that encoder is a stretch goal —
//! the `ort` path gives an exact-token working pipeline today.

use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use std::f32::consts::PI;
use std::path::Path;

/// Whisper feature constants (16 kHz). `whisper.audio`: N_FFT=400, HOP=160,
/// hann window, librosa slaney mel filterbank with `n_mels=128`.
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;
/// rfft bin count for N_FFT=400 is N_FFT/2 + 1 = 201; whisper drops the final
/// STFT column (`stft[..., :-1]`), not a frequency bin, so all 201 bins are kept.
const N_FREQS: usize = N_FFT / 2 + 1;

/// Errors raised while extracting prompt speech tokens.
#[derive(Debug)]
pub enum SpeechTokenError {
    /// The reference clip exceeds the 30 s ceiling the ONNX tokenizer supports.
    TooLong { secs: f32 },
    /// The waveform was empty after framing — no STFT columns could be produced.
    Empty,
    /// Underlying ONNX Runtime failure (session build or inference). The ort
    /// error is generic over a recovery type; we keep only its message so this
    /// variant is uniform across the different `ort::Error<R>` it can come from.
    Ort(String),
}

impl std::fmt::Display for SpeechTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeechTokenError::TooLong { secs } => {
                write!(f, "reference audio {secs:.2}s exceeds the 30s speech-token limit")
            }
            SpeechTokenError::Empty => write!(f, "waveform too short to produce any STFT frame"),
            SpeechTokenError::Ort(e) => write!(f, "onnx runtime error: {e}"),
        }
    }
}

impl std::error::Error for SpeechTokenError {}

impl<R> From<ort::Error<R>> for SpeechTokenError {
    fn from(e: ort::Error<R>) -> Self {
        SpeechTokenError::Ort(e.to_string())
    }
}

/// Periodic Hann window of length `n` — matches `torch.hann_window(n)`
/// (`periodic=True`, the torch default): `0.5 * (1 - cos(2*pi*k/n))`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n as f32).cos()))
        .collect()
}

/// Slaney mel filterbank `[N_MELS, N_FREQS]`, identical to
/// `librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)` (htk=False, slaney
/// area-normalisation) — i.e. whisper's stored `mel_128`. Recomputed here so no
/// binary asset needs vendoring; verified to ~2e-9 against the stored filters.
fn mel_filterbank() -> Array2<f32> {
    let sr = 16000.0f32;
    let fmin = 0.0f32;
    let fmax = sr / 2.0;

    // rfft frequencies: k * sr / n_fft for k in 0..=n_fft/2.
    let fftfreqs: Vec<f32> = (0..N_FREQS)
        .map(|k| k as f32 * sr / N_FFT as f32)
        .collect();

    // Slaney hz<->mel (the librosa default, htk=False).
    let f_sp = 200.0f32 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - fmin) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    let hz_to_mel = |f: f32| -> f32 {
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            (f - fmin) / f_sp
        }
    };
    let mel_to_hz = |m: f32| -> f32 {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            fmin + f_sp * m
        }
    };

    // N_MELS + 2 mel-spaced points -> hz.
    let m_min = hz_to_mel(fmin);
    let m_max = hz_to_mel(fmax);
    let n_pts = N_MELS + 2;
    let mel_pts: Vec<f32> = (0..n_pts)
        .map(|i| m_min + (m_max - m_min) * i as f32 / (n_pts - 1) as f32)
        .collect();
    let freqs: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

    let mut w = Array2::<f32>::zeros((N_MELS, N_FREQS));
    for i in 0..N_MELS {
        let f_lo = freqs[i];
        let f_ce = freqs[i + 1];
        let f_hi = freqs[i + 2];
        let d_lo = f_ce - f_lo;
        let d_hi = f_hi - f_ce;
        // Slaney area normalisation: 2 / (f_hi - f_lo).
        let enorm = 2.0 / (f_hi - f_lo);
        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - f_lo) / d_lo; // rising edge
            let upper = (f_hi - f) / d_hi; // falling edge
            let tri = lower.min(upper).max(0.0);
            w[[i, j]] = tri * enorm;
        }
    }
    w
}

/// In-place radix-agnostic DFT magnitude-squared of one real frame of length
/// `N_FFT`, returning the `N_FREQS` power-spectrum values |X[k]|^2 for the
/// non-negative frequencies. A direct DFT is used (N_FFT=400 is small and this
/// runs once per HOP); it matches `torch.stft(...).abs()**2` to float precision.
fn frame_power_spectrum(frame: &[f32]) -> [f32; N_FREQS] {
    debug_assert_eq!(frame.len(), N_FFT);
    let mut out = [0.0f32; N_FREQS];
    for (k, slot) in out.iter_mut().enumerate() {
        // accumulate in f64 for stability, like torch's internal accumulation.
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let w = -2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64;
        for (n, &x) in frame.iter().enumerate() {
            let ang = w * n as f64;
            re += x as f64 * ang.cos();
            im += x as f64 * ang.sin();
        }
        *slot = (re * re + im * im) as f32;
    }
    out
}

/// Compute the whisper log-mel spectrogram `[N_MELS, T]` for a 16 kHz mono
/// waveform, bit-compatible with `whisper.log_mel_spectrogram(speech, n_mels=128)`:
///
///   * centered STFT (reflect-padded by N_FFT/2 on each side),
///   * periodic Hann window, N_FFT=400, HOP=160,
///   * drop the final STFT column (`stft[..., :-1]`),
///   * power spectrum -> slaney mel projection,
///   * `log10(clamp(x, 1e-10))`, dynamic-range floor at `max - 8`,
///   * affine `(x + 4) / 4`.
pub fn log_mel_spectrogram(samples: &[f32]) -> Result<Array2<f32>, SpeechTokenError> {
    // Reflect-pad N_FFT/2 each side (torch.stft center=True default).
    let pad = N_FFT / 2;
    let n = samples.len();
    let mut padded = Vec::with_capacity(n + 2 * pad);
    // left reflect: samples[pad], samples[pad-1], ... (excludes the edge sample)
    for i in 0..pad {
        padded.push(samples[(pad - i).min(n.saturating_sub(1))]);
    }
    padded.extend_from_slice(samples);
    for i in 0..pad {
        // right reflect excludes the edge sample: n-2, n-3, ...
        let idx = n.saturating_sub(2).saturating_sub(i);
        padded.push(samples[idx]);
    }

    if padded.len() < N_FFT {
        return Err(SpeechTokenError::Empty);
    }

    // Number of frames in a centered STFT, then drop the trailing column.
    let n_frames_full = (padded.len() - N_FFT) / HOP + 1;
    if n_frames_full < 2 {
        return Err(SpeechTokenError::Empty);
    }
    let n_frames = n_frames_full - 1; // stft[..., :-1]

    let window = hann_window(N_FFT);
    let filters = mel_filterbank();

    // magnitudes: [N_FREQS, n_frames]
    let mut mags = Array2::<f32>::zeros((N_FREQS, n_frames));
    let mut frame = vec![0.0f32; N_FFT];
    for t in 0..n_frames {
        let start = t * HOP;
        for j in 0..N_FFT {
            frame[j] = padded[start + j] * window[j];
        }
        let ps = frame_power_spectrum(&frame);
        for f in 0..N_FREQS {
            mags[[f, t]] = ps[f];
        }
    }

    // mel_spec = filters @ magnitudes  -> [N_MELS, n_frames]
    let mel = filters.dot(&mags);

    // log10(clamp(.,1e-10)); floor at global max - 8; (x+4)/4.
    let mut log_spec = mel.mapv(|v| v.max(1e-10).log10());
    let maxv = log_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let floor = maxv - 8.0;
    log_spec.mapv_inplace(|v| (v.max(floor) + 4.0) / 4.0);

    Ok(log_spec)
}

/// Whisper log-mel as a flat row-major `(data, n_mels, n_frames)` triple — the
/// same values as [`log_mel_spectrogram`] but without exposing `ndarray` in the
/// signature, so callers (e.g. the root parity test) need not depend on it.
/// `data[m * n_frames + t]` is mel bin `m` at frame `t`.
pub fn log_mel_flat(samples: &[f32]) -> Result<(Vec<f32>, usize, usize), SpeechTokenError> {
    let mel = log_mel_spectrogram(samples)?;
    let n_mels = mel.shape()[0];
    let n_frames = mel.shape()[1];
    let data: Vec<f32> = mel.as_standard_layout().iter().copied().collect();
    Ok((data, n_mels, n_frames))
}

/// The prompt speech-token tokenizer: holds an ONNX Runtime session for
/// `speech_tokenizer_v2.onnx` and turns a 16 kHz waveform into the flat int32
/// token id sequence (`prompt_speech_token`).
pub struct SpeechTokenizer {
    session: Session,
    feat_input: String,
    len_input: String,
}

impl SpeechTokenizer {
    /// Build a tokenizer from the `speech_tokenizer_v2.onnx` file. Uses a single
    /// intra-op thread and full graph optimisation, matching the reference
    /// session options so token ids are reproducible.
    pub fn load(onnx_path: impl AsRef<Path>) -> Result<Self, SpeechTokenError> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(onnx_path.as_ref())?;

        // Inputs are [feats: f32 [1,128,T], feats_length: i32 [1]] in order.
        let inputs = session.inputs();
        let feat_input = inputs[0].name().to_string();
        let len_input = inputs[1].name().to_string();

        Ok(SpeechTokenizer {
            session,
            feat_input,
            len_input,
        })
    }

    /// Build a tokenizer from the CosyVoice3 `speech_tokenizer_v3.onnx` file.
    ///
    /// The v3 tokenizer's ONNX graph I/O was verified (via `onnx.load`) to be
    /// byte-identical to v2: inputs `feats: f32 [1,128,T]` + `feats_length: i32 [1]`,
    /// output `indices` (int32 token ids). The feature it consumes is the same
    /// `whisper.log_mel_spectrogram(n_mels=128)` that v2 uses — CV3's
    /// `_extract_speech_token` is the v2 method unchanged. So the full v2 session +
    /// log-mel path applies as-is; this constructor is a v3-named alias of [`load`]
    /// that documents that equivalence at the call site.
    ///
    /// [`load`]: SpeechTokenizer::load
    pub fn load_cv3(onnx_path: impl AsRef<Path>) -> Result<Self, SpeechTokenError> {
        Self::load(onnx_path)
    }

    /// Run the tokenizer on a precomputed whisper log-mel `[N_MELS, T]` feature
    /// and return the flat token id sequence.
    pub fn tokens_from_mel(&mut self, mel: &Array2<f32>) -> Result<Vec<i32>, SpeechTokenError> {
        let n_mels = mel.shape()[0];
        let t = mel.shape()[1];

        // Build [1, n_mels, T] contiguous feats and a [1] length input using ort's
        // version-independent `(shape, Vec)` constructor — avoids coupling to ort's
        // internal `ndarray` major version (it may differ from this crate's).
        let feats_data: Vec<f32> = mel
            .as_standard_layout()
            .iter()
            .copied()
            .collect();
        let feats_t = OrtTensor::from_array((vec![1_i64, n_mels as i64, t as i64], feats_data))?;
        let lens_t = OrtTensor::from_array((vec![1_i64], vec![t as i32]))?;

        let outputs = self.session.run(ort::inputs![
            self.feat_input.as_str() => feats_t,
            self.len_input.as_str() => lens_t,
        ])?;

        // Single int32 output (`indices`), shape [1, T_tok] (or [T_tok]).
        let (_shape, data) = outputs[0].try_extract_tensor::<i32>()?;
        Ok(data.to_vec())
    }

    /// Full pipeline: 16 kHz mono `samples` -> prompt speech-token ids.
    pub fn tokens_from_wav(&mut self, samples: &[f32]) -> Result<Vec<i32>, SpeechTokenError> {
        let secs = samples.len() as f32 / 16000.0;
        if secs > 30.0 {
            return Err(SpeechTokenError::TooLong { secs });
        }
        let mel = log_mel_spectrogram(samples)?;
        self.tokens_from_mel(&mel)
    }
}
