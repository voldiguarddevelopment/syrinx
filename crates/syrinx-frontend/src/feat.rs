//! Audio feature extraction for the CosyVoice2 frontend (feature `real`).
//!
//! Two extractors, byte-for-algorithm-identical to the Python reference in
//! `cosyvoice/cli/frontend.py`:
//!
//!   * [`kaldi_fbank`] — `torchaudio.compliance.kaldi.fbank(num_mel_bins=80,
//!     dither=0, sample_frequency=16000)` at all other kaldi defaults. This is
//!     the input to the CAM++ speaker encoder (`_extract_spk_embedding`). Output
//!     is `[T, 80]`, frame-major, **before** the per-utterance mean subtraction
//!     that `_extract_spk_embedding` applies separately.
//!
//!   * [`prompt_mel`] — `matcha.utils.audio.mel_spectrogram(n_fft=1920,
//!     num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0,
//!     fmax=8000, center=False)`, the `feat_extractor` from `cosyvoice2.yaml`.
//!     This is the `prompt_feat` input to the flow decoder. Output is `[80, T']`,
//!     mel-major (already transposed to match the Python `[80, T']` layout).
//!
//! Both consume an already-loaded mono `f32` waveform in `[-1, 1]` (matching
//! `load_wav`, which is `torchaudio.load` -> channel-mean -> resample). Resampling
//! and decode are out of scope here — the parity fixtures carry the exact resampled
//! waveforms so this math is what is under test.
//!
//! ## Numerical strategy
//!
//! The window / mel-bank construction is done in `f32` to match how torch builds
//! them, but the per-frame FFT and the band accumulation are carried in `f64`. The
//! reference is itself an `f32` computation, so the residual we bound is the
//! reference's own rounding rather than ours — this keeps a comfortable margin
//! under the 1e-3 parity bar even on the ~20 % of fbank bins whose energy sits near
//! the log floor, where `f32` rounding is most amplified by the log.
//!
//! ## Kaldi conventions reproduced exactly
//!
//! frame_length 25 ms, frame_shift 10 ms, `snip_edges=true` (frames that fully
//! fit), `remove_dc_offset=true` (per-frame mean removal), `preemphasis=0.97`
//! with the first sample replicated, povey window (`hann^0.85`, non-periodic),
//! `round_to_power_of_two=true` (400 -> 512), power spectrum, kaldi mel bank
//! (`mel = 1127·ln(1+f/700)`, low_freq=20, high_freq=Nyquist) right-padded with a
//! zero Nyquist column, `log(max(x, f32::EPSILON))`. No 32768 scaling — torchaudio
//! kaldi uses the float waveform as-is.

use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

/// `numeric_limits<float>::epsilon()` — torchaudio's `EPSILON` and the log floor.
const KALDI_EPSILON: f32 = f32::EPSILON; // 1.1920929e-7

/// Smallest power of two strictly covering `x` (kaldi `_next_power_of_2`).
fn next_power_of_2(x: usize) -> usize {
    if x == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < x {
        p <<= 1;
    }
    p
}

/// Non-periodic Hann window of length `n`: `0.5 - 0.5·cos(2π·i/(n-1))`.
/// This matches `torch.hann_window(n, periodic=False)`.
fn hann_window(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

/// Povey window = non-periodic Hann raised to the 0.85 power.
fn povey_window(n: usize) -> Vec<f32> {
    hann_window(n).into_iter().map(|w| w.powf(0.85)).collect()
}

/// kaldi mel scale and its inverse.
fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Build the kaldi triangular mel filterbank, shape `[num_bins, num_fft_bins]`
/// where `num_fft_bins = padded_window_size / 2` (the Nyquist bin is appended as a
/// zero column by the caller). Matches `get_mel_banks` with `vtln_warp = 1.0`.
fn mel_banks(
    num_bins: usize,
    padded_window_size: usize,
    sample_freq: f32,
    low_freq: f32,
    high_freq_opt: f32,
) -> Vec<Vec<f32>> {
    let num_fft_bins = padded_window_size / 2;
    let nyquist = 0.5 * sample_freq;
    let high_freq = if high_freq_opt <= 0.0 {
        high_freq_opt + nyquist
    } else {
        high_freq_opt
    };

    let fft_bin_width = sample_freq / padded_window_size as f32;
    let mel_low = mel_scale(low_freq);
    let mel_high = mel_scale(high_freq);
    let mel_delta = (mel_high - mel_low) / (num_bins as f32 + 1.0);

    let mut banks = vec![vec![0.0f32; num_fft_bins]; num_bins];
    for (b, row) in banks.iter_mut().enumerate() {
        let left_mel = mel_low + b as f32 * mel_delta;
        let center_mel = mel_low + (b as f32 + 1.0) * mel_delta;
        let right_mel = mel_low + (b as f32 + 2.0) * mel_delta;
        for (i, cell) in row.iter_mut().enumerate() {
            let mel = mel_scale(fft_bin_width * i as f32);
            let up = (mel - left_mel) / (center_mel - left_mel);
            let down = (right_mel - mel) / (right_mel - center_mel);
            *cell = up.min(down).max(0.0);
        }
    }
    banks
}

/// Slice the waveform into overlapping frames (`snip_edges=true`): only frames
/// fully inside the signal, `m = 1 + (n - window_size) / window_shift`.
fn strided_frames(waveform: &[f32], window_size: usize, window_shift: usize) -> Vec<Vec<f32>> {
    let n = waveform.len();
    if n < window_size {
        return Vec::new();
    }
    let m = 1 + (n - window_size) / window_shift;
    (0..m)
        .map(|i| {
            let start = i * window_shift;
            waveform[start..start + window_size].to_vec()
        })
        .collect()
}

/// Apply kaldi's per-frame preprocessing in the exact order torchaudio uses:
/// dc-offset removal -> preemphasis (first sample replicated) -> window. Returns
/// the windowed frame of length `window_size` (zero-padding to the padded FFT size
/// is done by the FFT caller).
fn process_frame(frame: &[f32], window: &[f32], preemphasis: f32) -> Vec<f32> {
    let n = frame.len();
    // remove_dc_offset: subtract the frame mean.
    let mean: f32 = frame.iter().sum::<f32>() / n as f32;
    let mut x: Vec<f32> = frame.iter().map(|&v| v - mean).collect();

    // preemphasis with `replicate` padding: prev[0] = x[0].
    if preemphasis != 0.0 {
        let mut prev = x[0];
        for v in x.iter_mut() {
            let cur = *v;
            *v = cur - preemphasis * prev;
            prev = cur;
        }
    }

    // window
    for (v, &w) in x.iter_mut().zip(window.iter()) {
        *v *= w;
    }
    x
}

/// `f64` real FFT power spectrum of a (possibly zero-padded) frame, returning the
/// `padded_window_size/2 + 1` one-sided power bins (`|X|^2`). The `f64` carry keeps
/// our rounding below the reference's own `f32` error.
fn power_spectrum(frame: &[f32], padded_window_size: usize, fft: &Arc<dyn Fft<f64>>) -> Vec<f64> {
    let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); padded_window_size];
    for (i, &v) in frame.iter().enumerate() {
        buf[i].re = v as f64;
    }
    fft.process(&mut buf);
    let half = padded_window_size / 2;
    (0..=half).map(|i| buf[i].re * buf[i].re + buf[i].im * buf[i].im).collect()
}

/// Kaldi fbank: `[-1,1]` mono waveform at `sample_frequency` -> `[T, 80]` log-mel
/// energies (frame-major). Mirrors `kaldi.fbank(num_mel_bins=80, dither=0,
/// sample_frequency=...)` at all other defaults. Does **not** subtract the
/// per-utterance mean (the caller / speaker encoder does that).
pub fn kaldi_fbank(waveform: &[f32], sample_frequency: f32, num_mel_bins: usize) -> Vec<Vec<f32>> {
    // kaldi defaults relevant here.
    const FRAME_LENGTH_MS: f32 = 25.0;
    const FRAME_SHIFT_MS: f32 = 10.0;
    const PREEMPHASIS: f32 = 0.97;
    const LOW_FREQ: f32 = 20.0;
    const HIGH_FREQ: f32 = 0.0; // <= 0 means Nyquist

    let window_shift = (sample_frequency * FRAME_SHIFT_MS * 0.001) as usize;
    let window_size = (sample_frequency * FRAME_LENGTH_MS * 0.001) as usize;
    let padded_window_size = next_power_of_2(window_size);

    let window = povey_window(window_size);
    let banks = mel_banks(num_mel_bins, padded_window_size, sample_frequency, LOW_FREQ, HIGH_FREQ);
    // num_fft_bins = padded/2; the power spectrum has padded/2 + 1 bins. kaldi
    // right-pads the bank with one zero column (the Nyquist bin) — i.e. the
    // Nyquist power bin contributes zero. So we simply ignore the last power bin.
    let num_fft_bins = padded_window_size / 2;

    let frames = strided_frames(waveform, window_size, window_shift);
    if frames.is_empty() {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(padded_window_size);

    frames
        .iter()
        .map(|frame| {
            let processed = process_frame(frame, &window, PREEMPHASIS);
            let power = power_spectrum(&processed, padded_window_size, &fft);
            (0..num_mel_bins)
                .map(|b| {
                    // dot the bank row with the first num_fft_bins power bins
                    // (Nyquist column is zero), accumulating in f64.
                    let mut acc = 0.0f64;
                    for i in 0..num_fft_bins {
                        acc += power[i] * banks[b][i] as f64;
                    }
                    (acc as f32).max(KALDI_EPSILON).ln()
                })
                .collect::<Vec<f32>>()
        })
        .collect()
}

/// librosa-style HTK-mel filterbank with Slaney area normalization, shape
/// `[n_mels, n_fft/2 + 1]`. Matches `librosa.filters.mel(sr, n_fft, n_mels, fmin,
/// fmax)` (htk=False default, norm='slaney'). librosa's "htk=False" still uses the
/// **Slaney** mel scale, not the kaldi/HTK log scale — implemented below.
fn librosa_mel_basis(
    sr: f32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let n_fft_bins = n_fft / 2 + 1;

    // Slaney mel scale (librosa default, htk=False).
    let hz_to_mel = |f: f32| -> f32 {
        let f_min = 0.0f32;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0f32;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            (f - f_min) / f_sp
        }
    };
    let mel_to_hz = |m: f32| -> f32 {
        let f_min = 0.0f32;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0f32;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_min + f_sp * m
        }
    };

    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    // n_mels + 2 mel points -> band edges.
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_mels as f32 + 1.0))
        .collect();
    let freq_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // fft bin center frequencies.
    let fft_freqs: Vec<f32> =
        (0..n_fft_bins).map(|i| i as f32 * sr / n_fft as f32).collect();

    let mut weights = vec![vec![0.0f32; n_fft_bins]; n_mels];
    for m in 0..n_mels {
        let lower = freq_points[m];
        let center = freq_points[m + 1];
        let upper = freq_points[m + 2];
        for (j, &f) in fft_freqs.iter().enumerate() {
            let down = (f - lower) / (center - lower);
            let up = (upper - f) / (upper - center);
            weights[m][j] = down.min(up).max(0.0);
        }
        // Slaney normalization: scale by 2 / (upper - lower).
        let enorm = 2.0 / (freq_points[m + 2] - freq_points[m]);
        for w in weights[m].iter_mut() {
            *w *= enorm;
        }
    }
    weights
}

/// Prompt mel-spectrogram: `[-1,1]` mono 24 kHz waveform -> `[80, T']` log mel
/// (mel-major, matching the Python `[80, T']`). Mirrors `matcha`'s
/// `mel_spectrogram(n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480,
/// win_size=1920, fmin=0, fmax=8000, center=False)`: reflect-pad by
/// `(n_fft-hop)/2`, periodic Hann window, STFT magnitude `sqrt(power + 1e-9)`,
/// librosa Slaney mel basis, `log(clamp(x, 1e-5))`.
pub fn prompt_mel(
    waveform: &[f32],
    n_fft: usize,
    num_mels: usize,
    sampling_rate: f32,
    hop_size: usize,
    win_size: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    // matcha pads the *signal* by (n_fft - hop)/2 each side, reflect mode.
    let pad = (n_fft - hop_size) / 2;
    let padded = reflect_pad(waveform, pad);

    // torch.stft with center=False, win_length=win_size==n_fft here. Periodic
    // Hann window (torch.hann_window default periodic=True).
    let window = hann_window_periodic(win_size);

    // center=False: first frame starts at 0, frames fully inside the padded signal.
    let n = padded.len();
    let n_fft_bins = n_fft / 2 + 1;
    let num_frames = if n >= n_fft {
        1 + (n - n_fft) / hop_size
    } else {
        0
    };

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // magnitude spectrogram [n_fft_bins, num_frames]
    let basis = librosa_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax);

    // Build mel-major output [num_mels, num_frames].
    let mut out = vec![vec![0.0f32; num_frames]; num_mels];
    for t in 0..num_frames {
        let start = t * hop_size;
        let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n_fft];
        for i in 0..win_size {
            buf[i].re = (padded[start + i] * window[i]) as f64;
        }
        fft.process(&mut buf);
        // magnitude = sqrt(re^2 + im^2 + 1e-9) per matcha.
        let mut mag = vec![0.0f64; n_fft_bins];
        for (i, m) in mag.iter_mut().enumerate() {
            *m = (buf[i].re * buf[i].re + buf[i].im * buf[i].im + 1e-9).sqrt();
        }
        for mbin in 0..num_mels {
            let mut acc = 0.0f64;
            for (j, &m) in mag.iter().enumerate() {
                acc += basis[mbin][j] as f64 * m;
            }
            // spectral_normalize_torch: log(clamp(x, min=1e-5)).
            out[mbin][t] = (acc as f32).max(1e-5).ln();
        }
    }
    out
}

/// Periodic Hann window of length `n`: `0.5 - 0.5·cos(2π·i/n)`.
/// Matches `torch.hann_window(n)` (periodic=True default).
fn hann_window_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad a 1-D signal by `pad` samples each side, matching
/// `torch.nn.functional.pad(mode="reflect")` (the boundary sample is not repeated).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // left: x[pad], x[pad-1], ..., x[1]
    for k in 0..pad {
        out.push(x[pad - k]);
    }
    out.extend_from_slice(x);
    // right: x[n-2], x[n-3], ..., x[n-1-pad]
    for k in 0..pad {
        out.push(x[n - 2 - k]);
    }
    out
}
