//! The Qwen3-TTS **speaker encoder** — the x-vector that carries a reference clip's
//! timbre into the talker on the `-Base` (voice-clone) checkpoints.
//!
//! It is present as `speaker_encoder.*` (76 tensors) in `Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`
//! and absent from `CustomVoice` / `VoiceDesign`, which is exactly
//! [`QwenVariant::supports_voice_clone`](crate::config::QwenVariant::supports_voice_clone).
//!
//! ## Architecture — ECAPA-TDNN, and note what is *missing*
//!
//! Recovered from `qwen_tts.core.models.modeling_qwen3_tts`
//! (`Qwen3TTSSpeakerEncoder`, `SqueezeExcitationRes2NetBlock`, `Res2NetBlock`,
//! `SqueezeExcitationBlock`, `AttentiveStatisticsPooling`, `TimeDelayNetBlock`) and
//! cross-checked against the tensor list of `0.6B-Base/model.safetensors`:
//!
//! ```text
//!   mel [B, T, 128]  --transpose-->  [B, 128, T]
//!   blocks.0    TimeDelayNetBlock(128 -> 512, k=5, d=1)
//!   blocks.1    SE-Res2Net(512 -> 512, k=3, d=2)   \
//!   blocks.2    SE-Res2Net(512 -> 512, k=3, d=3)    >  outputs concatenated -> 1536
//!   blocks.3    SE-Res2Net(512 -> 512, k=3, d=4)   /
//!   mfa         TimeDelayNetBlock(1536 -> 1536, k=1, d=1)
//!   asp         AttentiveStatisticsPooling(1536, attention_channels=128) -> [B, 3072, 1]
//!   fc          Conv1d(3072 -> 1024, k=1)          -> [B, 1024]
//! ```
//!
//! **There is no normalisation layer anywhere in this stack.** SpeechBrain's ECAPA-TDNN
//! — which this is otherwise a transcription of — puts a `BatchNorm1d` inside every
//! `TDNNBlock`; Qwen's `TimeDelayNetBlock` is `Conv1d` + `ReLU` and nothing else. The
//! checkpoint confirms it from the other side: all 76 tensors are `conv.weight` /
//! `conv.bias` pairs, with no `running_mean` / `running_var` / `gamma` / `beta` in
//! sight. Porting the SpeechBrain block from memory would have silently inserted a
//! layer these weights were never trained with.
//!
//! Two more details that are easy to get wrong and are pinned by the fixture test below:
//!
//! * **Every convolution is `padding="same"` with `padding_mode="reflect"`** — not
//!   zero padding. With dilations up to 4 that is 4 frames of reflected context at each
//!   end of every block; zeros would inject a spurious silence-like edge into a
//!   3-second clip's statistics.
//! * **The multi-layer feature aggregation skips `blocks.0`.** The reference
//!   concatenates `hidden_states_list[1:]`, i.e. only the three SE-Res2Net outputs
//!   (3 x 512 = 1536), which is why `mfa.conv.weight` is `[1536, 1536, 1]` and not
//!   `[1536, 2048, 1]`.
//!
//! ## Attentive statistics pooling has no real mask
//!
//! `AttentiveStatisticsPooling.forward` builds its mask from
//! `lengths = torch.ones(batch) * seq_length` — unconditionally the full length, for
//! every element, on every call. The mask is therefore always all-ones, the
//! `masked_fill(mask == 0, -inf)` is always a no-op, and `mask / total` is the uniform
//! weight `1/T`. This port implements that path directly rather than carrying a
//! length argument the reference never varies; a batch of clips of *different* lengths
//! would be mispooled by the reference too, so callers must encode one clip at a time
//! (which is what [`SpeakerEncoder::embed`] does).
//!
//! The statistics are the **population** (biased, `1/T`) variance, clamped at `1e-12`
//! before the square root — note the contrast with the CAM++ encoder in
//! `syrinx-speaker`, whose ONNX graph uses the *unbiased* `T/(T-1)` estimator.
//!
//! ## The two `-Base` sizes differ in `enc_dim`
//!
//! `0.6B-Base` emits a **1024**-wide x-vector and `1.7B-Base` a **2048**-wide one —
//! `fc.weight` is `[1024, 3072, 1]` in one and `[2048, 3072, 1]` in the other, and
//! their `config.json` `speaker_encoder_config.enc_dim` says so. Every other tensor in
//! the stack is identical between them. That is the reason
//! [`SpeakerEncoderConfig::from_model_config`] exists: hardcoding 1024 loads the 0.6B
//! and fails on the 1.7B.
//!
//! ## Input: a log-mel spectrogram, not a waveform
//!
//! `Qwen3TTSForConditionalGeneration.extract_speaker_embedding` asserts `sr == 24000`
//! and feeds `mel_spectrogram(...).transpose(1, 2)`, so the encoder consumes
//! `[B, T, 128]` frames. [`MelConfig::QWEN3_TTS`] carries that call's exact arguments;
//! [`log_mel_spectrogram`] reproduces it, librosa slaney filterbank included.
//!
//! ## Numerics
//!
//! Everything here runs in **f32** regardless of the talker's compute dtype (the
//! checkpoint stores these tensors as bf16). This is a one-shot pass over a few hundred
//! frames, so the cost is irrelevant, and the pooling is a sum over the whole utterance
//! — precisely the reduction bf16 handles worst.
//!
//! ## Verification status
//!
//! **Anchored to the reference on a real clip** (2026-09-03, CPU/f32): the repo-root
//! `tests/real_qwen_speaker_parity.rs` compares this module against tensors dumped from
//! the reference's OWN modules by `scripts/gen-qwen-ref-speaker.py`, on the real call path
//! (`create_voice_clone_prompt` -> `extract_speaker_embedding` -> `Qwen3TTSSpeakerEncoder`)
//! over `voice_en_10s.wav` and the published `1.7B-Base` weights. Three anchors, so a
//! mismatch localizes to the front end or the encoder:
//!
//! | anchor | max abs diff |
//! |---|---|
//! | mel `[937, 128]` from the reference's own resampled clip | **0.00025** |
//! | x-vector from the reference's own mel | **0.0000010** |
//! | x-vector end to end (`embed`) | **0.0000010** |
//!
//! The port reproduces a 2048-wide reference x-vector to **1e-6 per component** (L2 norm
//! 17.028715 on both sides). Nothing in this module was found wrong — unlike
//! `text_projection`, whose dropped `silu` is why the anchor exists at all.
//!
//! `tests::real_checkpoint_parity` below is the older, weaker check: hand-copied numbers
//! for a synthetic waveform on both `-Base` sizes. It stays because it is the only thing
//! covering the **0.6B** (1024-wide) checkpoint — the fixture is 1.7B.
//!
//! **Not verified**: nothing here has been run on a CUDA device, and nothing has been
//! run in bf16 — the reference executes the encoder in the talker's dtype, so a bf16 GPU
//! run will differ from this f32 path by more than 1e-4. That is a dtype difference, not
//! a porting question, and the f32 result is the more accurate of the two.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, D};

use crate::nn::Weights;

/// `AttentiveStatisticsPooling.eps` — the variance clamp before `sqrt`.
const ASP_EPS: f64 = 1e-12;
/// `dynamic_range_compression_torch(clip_val=...)` — the log floor of the mel.
const LOG_MEL_CLIP: f64 = 1e-5;
/// The `+ 1e-9` inside `sqrt` when `mel_spectrogram` takes the STFT magnitude.
const STFT_MAG_EPS: f64 = 1e-9;
/// Half-width of [`resample`]'s Lanczos kernel, in periods of its cutoff. See that
/// function's docs — the value is a measurement, not a convention.
const LOBES: f64 = 64.0;
/// Fraction of Nyquist [`resample`] passes flat, leaving the kernel's skirt room to reach
/// the stopband before the image starts. Also a measurement.
const PASSBAND: f64 = 0.96;

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

/// `Qwen3TTSSpeakerEncoderConfig`. The published `config.json` carries only `enc_dim`
/// and `sample_rate`; every other field below is the reference class's default, which
/// is what the shipped checkpoints were built with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerEncoderConfig {
    /// Mel bins the encoder consumes (`blocks.0.conv.weight`'s input channels).
    pub mel_dim: usize,
    /// Width of the emitted x-vector (`fc.weight`'s output channels).
    pub enc_dim: usize,
    /// Per-stage channel widths; the last entry is the MFA / pooling width.
    pub enc_channels: Vec<usize>,
    pub enc_kernel_sizes: Vec<usize>,
    pub enc_dilations: Vec<usize>,
    pub enc_attention_channels: usize,
    pub enc_res2net_scale: usize,
    pub enc_se_channels: usize,
    pub sample_rate: u32,
}

impl Default for SpeakerEncoderConfig {
    /// The defaults of `Qwen3TTSSpeakerEncoderConfig`, with `enc_dim` at the 1024 the
    /// published `speaker_encoder_config` overrides it to (the class default is 192).
    fn default() -> Self {
        Self {
            mel_dim: 128,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
            sample_rate: 24_000,
        }
    }
}

impl SpeakerEncoderConfig {
    /// Build from the `speaker_encoder_config` block of a parsed [`crate::Qwen3TtsConfig`],
    /// i.e. the two fields the checkpoint actually specifies over the class defaults.
    pub fn from_model_config(cfg: &crate::Qwen3TtsConfig) -> Self {
        Self {
            enc_dim: cfg.speaker_enc_dim,
            sample_rate: cfg.sample_rate,
            ..Self::default()
        }
    }

    /// The MFA / pooling width — `enc_channels.last()`.
    fn pool_channels(&self) -> usize {
        *self.enc_channels.last().unwrap_or(&0)
    }

    /// Index range of the SE-Res2Net stages: `1 .. enc_channels.len() - 1`.
    fn se_res2net_range(&self) -> std::ops::Range<usize> {
        1..self.enc_channels.len().saturating_sub(1)
    }

    /// Every tensor this encoder needs, with the shape it must have — the manifest to
    /// check a checkpoint against. `blocks.0` plus each SE-Res2Net stage plus
    /// `mfa` / `asp` / `fc`; 76 entries at the published geometry.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut v = Vec::new();
        let mut conv = |name: &str, out: usize, inp: usize, k: usize| {
            v.push((format!("{name}.weight"), vec![out, inp, k]));
            v.push((format!("{name}.bias"), vec![out]));
        };
        conv("blocks.0.conv", self.enc_channels[0], self.mel_dim, self.enc_kernel_sizes[0]);
        for i in self.se_res2net_range() {
            let (c_in, c) = (self.enc_channels[i - 1], self.enc_channels[i]);
            let part = c / self.enc_res2net_scale;
            let p = format!("blocks.{i}");
            conv(&format!("{p}.tdnn1.conv"), c, c_in, 1);
            for j in 0..self.enc_res2net_scale - 1 {
                conv(&format!("{p}.res2net_block.blocks.{j}.conv"), part, part, self.enc_kernel_sizes[i]);
            }
            conv(&format!("{p}.tdnn2.conv"), c, c, 1);
            conv(&format!("{p}.se_block.conv1"), self.enc_se_channels, c, 1);
            conv(&format!("{p}.se_block.conv2"), c, self.enc_se_channels, 1);
        }
        let pc = self.pool_channels();
        conv("mfa.conv", pc, pc, *self.enc_kernel_sizes.last().unwrap());
        conv("asp.tdnn.conv", self.enc_attention_channels, pc * 3, 1);
        conv("asp.conv", pc, self.enc_attention_channels, 1);
        conv("fc", self.enc_dim, pc * 2, 1);
        v
    }
}

// ---------------------------------------------------------------------------
// the mel front end
// ---------------------------------------------------------------------------

/// Arguments of the reference `mel_spectrogram` call, verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MelConfig {
    pub n_fft: usize,
    pub num_mels: usize,
    pub sample_rate: u32,
    pub hop_size: usize,
    pub win_size: usize,
    pub fmin: u32,
    pub fmax: u32,
}

impl MelConfig {
    /// Exactly what `extract_speaker_embedding` passes: `n_fft=1024, num_mels=128,
    /// sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000`, with the
    /// function's `center=False` default and its manual `(n_fft - hop) // 2` reflect pad.
    pub const QWEN3_TTS: Self = Self {
        n_fft: 1024,
        num_mels: 128,
        sample_rate: 24_000,
        hop_size: 256,
        win_size: 1024,
        fmin: 0,
        fmax: 12_000,
    };
}

/// librosa's slaney `hz_to_mel`: linear at `200/3` Hz per mel below 1 kHz, log above.
fn hz_to_mel(f: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= MIN_LOG_HZ {
        min_log_mel + (f / MIN_LOG_HZ).ln() / logstep
    } else {
        f / F_SP
    }
}

/// Inverse of [`hz_to_mel`].
fn mel_to_hz(m: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        MIN_LOG_HZ * (logstep * (m - min_log_mel)).exp()
    } else {
        F_SP * m
    }
}

/// `librosa.filters.mel(..., htk=False, norm="slaney")` -> `[num_mels, n_fft/2 + 1]`
/// row-major: triangular filters on the slaney mel scale, each scaled by
/// `2 / (mel_f[i+2] - mel_f[i])` so it integrates to a constant rather than peaking at 1.
fn mel_filterbank(mc: &MelConfig) -> Vec<f32> {
    let n_bins = mc.n_fft / 2 + 1;
    // `librosa.fft_frequencies`: linspace(0, sr/2, 1 + n_fft/2).
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| 0.5 * mc.sample_rate as f64 * i as f64 / (n_bins - 1) as f64)
        .collect();
    // `librosa.mel_frequencies(num_mels + 2, fmin, fmax)`.
    let (m_lo, m_hi) = (hz_to_mel(mc.fmin as f64), hz_to_mel(mc.fmax as f64));
    let n_pts = mc.num_mels + 2;
    let mel_f: Vec<f64> = (0..n_pts)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_pts - 1) as f64))
        .collect();

    let mut w = vec![0f32; mc.num_mels * n_bins];
    for i in 0..mc.num_mels {
        let (lo, ctr, hi) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
        let enorm = 2.0 / (hi - lo);
        for (b, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - lo) / (ctr - lo);
            let upper = (hi - f) / (hi - ctr);
            w[i * n_bins + b] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

/// The reference `mel_spectrogram(...).transpose(1, 2)`: `[1, frames, num_mels]`.
///
/// Reproduces, in order: a manual `(n_fft - hop) / 2` reflect pad; `torch.stft` with
/// `center=False` and a periodic Hann window; `sqrt(re² + im² + 1e-9)`; the librosa
/// slaney mel filterbank; and `log(clamp(x, min=1e-5))`.
///
/// The DFT is evaluated as a dense `[n_bins, n_fft] @ [n_fft, frames]` matmul rather
/// than an FFT — a reference clip is a one-shot cost, and this keeps the transform a
/// few lines that can be read against the formula.
pub fn log_mel_spectrogram(samples: &[f32], mc: &MelConfig, dev: &Device) -> Result<Tensor> {
    let pad = (mc.n_fft - mc.hop_size) / 2;
    if samples.len() <= pad {
        return Err(candle_core::Error::Msg(format!(
            "log_mel_spectrogram: {} samples is too short for the {pad}-sample reflect pad \
             (need > {pad}, i.e. > {:.0} ms at {} Hz)",
            samples.len(),
            1000.0 * pad as f64 / mc.sample_rate as f64,
            mc.sample_rate
        )));
    }
    // Manual reflect pad, matching `F.pad(y, (pad, pad), mode="reflect")`.
    let n = samples.len();
    let mut y = Vec::with_capacity(n + 2 * pad);
    y.extend((0..pad).rev().map(|i| samples[i + 1] as f64));
    y.extend(samples.iter().map(|&s| s as f64));
    y.extend((0..pad).map(|i| samples[n - 2 - i] as f64));

    if y.len() < mc.n_fft {
        return Err(candle_core::Error::Msg(format!(
            "log_mel_spectrogram: {} padded samples yields no complete {}-sample frame",
            y.len(),
            mc.n_fft
        )));
    }
    let frames = (y.len() - mc.n_fft) / mc.hop_size + 1;

    // `torch.hann_window` is periodic: 0.5 - 0.5 cos(2 pi n / N).
    let win: Vec<f64> = (0..mc.win_size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / mc.win_size as f64).cos())
        .collect();

    // Windowed frames as [n_fft, frames] so the DFT is a left-multiply.
    let mut fm = vec![0f32; mc.n_fft * frames];
    for t in 0..frames {
        let start = t * mc.hop_size;
        for i in 0..mc.n_fft {
            fm[i * frames + t] = (win[i] * y[start + i]) as f32;
        }
    }
    let fm = Tensor::from_vec(fm, (mc.n_fft, frames), dev)?;

    let n_bins = mc.n_fft / 2 + 1;
    let mut cos_b = vec![0f32; n_bins * mc.n_fft];
    let mut sin_b = vec![0f32; n_bins * mc.n_fft];
    for k in 0..n_bins {
        for i in 0..mc.n_fft {
            let th = 2.0 * std::f64::consts::PI * (k * i) as f64 / mc.n_fft as f64;
            cos_b[k * mc.n_fft + i] = th.cos() as f32;
            sin_b[k * mc.n_fft + i] = th.sin() as f32;
        }
    }
    let re = Tensor::from_vec(cos_b, (n_bins, mc.n_fft), dev)?.matmul(&fm)?;
    let im = Tensor::from_vec(sin_b, (n_bins, mc.n_fft), dev)?.matmul(&fm)?;
    let mag = ((re.sqr()? + im.sqr()?)? + STFT_MAG_EPS)?.sqrt()?; // [n_bins, frames]

    let basis = Tensor::from_vec(mel_filterbank(mc), (mc.num_mels, n_bins), dev)?;
    let mel = basis.matmul(&mag)?.clamp(LOG_MEL_CLIP, f64::INFINITY)?.log()?;
    mel.t()?.contiguous()?.unsqueeze(0) // [1, frames, num_mels]
}

// ---------------------------------------------------------------------------
// rate conversion (the clone driver's front step)
// ---------------------------------------------------------------------------

/// Band-limited mono resample, `in_sr -> out_sr` (Lanczos-windowed sinc). Identity when
/// the rates match.
///
/// [`SpeakerEncoder::embed`] refuses anything but 24 kHz and every reference clip in
/// practice arrives at some other rate, so the clone driver needs this step; the
/// reference spends `librosa.resample` (soxr `HQ`) here.
///
/// **This is not bit-comparable to soxr, and it is deliberately outside the numeric
/// parity gate.** `scripts/gen-qwen-ref-speaker.py` captures the *resampled* 24 kHz clip
/// the reference fed its encoder, and `tests/real_qwen_speaker_parity.rs` anchors the mel
/// and the x-vector against that — so a resampler difference can never be mistaken for a
/// porting fault. What this function *is* gated on is the driver path: that test
/// resamples the fixture's original-rate clip with this code and requires the resulting
/// x-vector to stay within a measured cosine bound of the reference's.
///
/// ## Why `LOBES` is 64 and `PASSBAND` is 0.96 — both are measurements
///
/// This started as a copy of `syrinx_serve::wavio::resample` (16 lobes, cutoff exactly
/// the output Nyquist, i.e. `PASSBAND = 1.0`), and on the 16 kHz -> 24 kHz reference clip
/// that produced an x-vector at **cosine 0.996886** of the reference's — visibly worse
/// than the rest of this port, and for a real reason rather than a tolerance one. An FFT
/// of the difference against the reference's own soxr output localizes all of it to the
/// transition band: below 7 kHz the two agree to a relative energy of 3.5e-6, while
/// **above 8 kHz — the input Nyquist, where the reference has 1.5e-8 total energy —
/// Lanczos-16 at cutoff 1.0 leaks 2.8e+2**. That leak is our own imaging artefact, and
/// the mel front end feeds it straight to the encoder: `fmax` is 12 kHz, so the top ~30
/// of the 128 bands see near-silence for the reference (pinned at the `1e-5` log floor)
/// and a fabricated signal for us, which `log` then magnifies.
///
/// Suppressing the image is what a lower passband and more lobes buy. Measured on the
/// same clip, end to end through the real encoder (cosine against the reference
/// x-vector; the full sweep is in the test's log):
///
/// | lobes | passband | cosine | relative L2 |
/// |---|---|---|---|
/// | 16 | 1.00 | 0.996886 | 0.0841 |
/// | 32 | 0.96 | 0.999771 | 0.0229 |
/// | 64 | 0.95 | 0.999802 | 0.0199 |
/// | **64** | **0.96** | **0.999977** | **0.0070** |
/// | 128 | 0.96 | 0.999982 | 0.0062 |
/// | 128 | 0.94 | 0.997751 | 0.0673 |
///
/// 64/0.96 is the knee: doubling the kernel again buys 6e-6 of cosine for twice the work,
/// and moving the passband either way is worse — 0.98 leaves imaging in, 0.94 starts
/// eating real 7.5 kHz speech. (soxr `HQ`'s own passband is 0.913 of Nyquist with a much
/// steeper skirt than a windowed sinc can manage, which is why matching its *number*
/// rather than its *effect* scores badly here.) 135 taps over a 10 s clip is ~32 M
/// multiply-adds — irrelevant next to one encoder pass.
///
/// The two sibling copies in `syrinx-serve` and `syrinx-stt` still carry the 16/1.0
/// settings. Nothing here changes them: they are other crates' files, feeding a vocoder
/// and a 16 kHz Whisper front end rather than this encoder, and the measurement above is
/// only evidence about *this* path.
pub fn resample(input: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    if input.is_empty() || in_sr == 0 || out_sr == 0 {
        return Vec::new();
    }
    if in_sr == out_sr {
        return input.to_vec();
    }
    let ratio = out_sr as f64 / in_sr as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    // Cutoff in input-sample coordinates: the output Nyquist when down-sampling (so the
    // kernel doubles as the anti-alias filter), the input Nyquist when up-sampling,
    // shaded by PASSBAND either way so the kernel's skirt lands below it.
    let cutoff = ratio.min(1.0) * PASSBAND;
    let radius = (LOBES / cutoff).ceil() as i64;
    let n = input.len() as i64;

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i as f64 / ratio;
        let i0 = center.floor() as i64;
        let (mut acc, mut wsum) = (0f64, 0f64);
        for k in (i0 - radius)..=(i0 + radius) {
            if k < 0 || k >= n {
                continue;
            }
            let x = (center - k as f64) * cutoff;
            let w = lanczos(x, LOBES);
            acc += input[k as usize] as f64 * w;
            wsum += w;
        }
        out.push(if wsum.abs() > 1e-12 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}

/// `sinc(x) * sinc(x / a)` inside `|x| < a`, zero outside — the Lanczos kernel.
fn lanczos(x: f64, a: f64) -> f64 {
    if x.abs() >= a {
        return 0.0;
    }
    sinc(x) * sinc(x / a)
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

// ---------------------------------------------------------------------------
// convolution primitives
// ---------------------------------------------------------------------------

/// Where a reflected index lands: `-i` off the left edge, `2(T-1) - i` off the right,
/// repeated until inside `0..t`. This is `F.pad(mode="reflect")` — the edge sample is
/// not duplicated.
fn reflect_index(i: isize, t: isize) -> isize {
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= t {
            i = 2 * (t - 1) - i;
        } else {
            return i;
        }
    }
}

/// Reflect-pad the last (time) axis by `left` / `right` frames.
fn reflect_pad(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(D::Minus1)?;
    if left >= t || right >= t {
        return Err(candle_core::Error::Msg(format!(
            "reflect_pad: padding ({left}, {right}) needs a length > it, got {t} frames"
        )));
    }
    let idx: Vec<u32> = (0..t + left + right)
        .map(|j| reflect_index(j as isize - left as isize, t as isize) as u32)
        .collect();
    let index = Tensor::from_vec(idx, t + left + right, x.device())?;
    x.contiguous()?.index_select(&index, D::Minus1)
}

// ---------------------------------------------------------------------------
// the encoder
// ---------------------------------------------------------------------------

/// The ECAPA-TDNN speaker encoder, weights resolved and cast to f32 at load.
pub struct SpeakerEncoder {
    cfg: SpeakerEncoderConfig,
    /// Manifest-relative names (no `speaker_encoder.` prefix), f32.
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl SpeakerEncoder {
    /// Load every tensor of [`SpeakerEncoderConfig::tensor_manifest`] from `w` under
    /// `prefix` (`"speaker_encoder"` in the published checkpoints), verifying each
    /// shape against the manifest as it goes — the same load-time check `load.rs`
    /// applies to the talker, which is where a geometry slip is cheap to diagnose
    /// rather than a shape error twelve layers into a forward pass.
    pub fn load(w: &Weights, prefix: &str, cfg: SpeakerEncoderConfig) -> Result<Self> {
        let mut map = HashMap::new();
        for (name, shape) in cfg.tensor_manifest() {
            let full = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
            let t = w.g(&full)?;
            if t.dims() != shape.as_slice() {
                return Err(candle_core::Error::Msg(format!(
                    "{full}: expected shape {shape:?}, checkpoint has {:?}",
                    t.dims()
                )));
            }
            map.insert(name, t.to_dtype(DType::F32)?);
        }
        Ok(Self { cfg, w: map, dev: w.dev.clone() })
    }

    pub fn config(&self) -> &SpeakerEncoderConfig {
        &self.cfg
    }

    fn t(&self, name: &str) -> Result<&Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("speaker encoder: missing {name}")))
    }

    /// `nn.Conv1d(..., padding="same", padding_mode="reflect")` + bias, stride 1.
    ///
    /// PyTorch splits `same` padding as `left = total / 2`, `right = total - left` with
    /// `total = dilation * (kernel - 1)`, reflect-pads, then convolves with no padding.
    fn conv(&self, x: &Tensor, name: &str, dilation: usize) -> Result<Tensor> {
        let w = self.t(&format!("{name}.weight"))?;
        let (out, _, k) = w.dims3()?;
        let total = dilation * (k - 1);
        let left = total / 2;
        let xp = reflect_pad(x, left, total - left)?;
        let y = xp.conv1d(w, 0, 1, dilation, 1)?;
        let b = self.t(&format!("{name}.bias"))?.reshape((1, out, 1))?;
        y.broadcast_add(&b)
    }

    /// `TimeDelayNetBlock`: reflect-same `Conv1d` then `ReLU`. No normalisation — see
    /// the module docs.
    fn tdnn(&self, x: &Tensor, name: &str, dilation: usize) -> Result<Tensor> {
        self.conv(x, name, dilation)?.relu()
    }

    /// `Res2NetBlock`: split the channels into `scale` parts; part 0 passes through
    /// untouched, part 1 goes through `blocks[0]`, and every later part is *summed with
    /// the previous part's output* before its own block. That running sum is the whole
    /// point of Res2Net — treating the parts independently gives the right shapes and
    /// the wrong receptive field.
    fn res2net(&self, x: &Tensor, name: &str, dilation: usize) -> Result<Tensor> {
        let (_b, c, _t) = x.dims3()?;
        let scale = self.cfg.enc_res2net_scale;
        let part = c / scale;
        let mut outs: Vec<Tensor> = Vec::with_capacity(scale);
        for i in 0..scale {
            let chunk = x.narrow(1, i * part, part)?;
            let out = if i == 0 {
                chunk
            } else {
                let inp = if i == 1 { chunk } else { (chunk + &outs[i - 1])? };
                self.tdnn(&inp, &format!("{name}.blocks.{}.conv", i - 1), dilation)?
            };
            outs.push(out);
        }
        Tensor::cat(&outs, 1)
    }

    /// `SqueezeExcitationBlock`: pool over time, `conv1 -> ReLU -> conv2 -> sigmoid`,
    /// then rescale each channel of the *unpooled* input by that gate.
    fn se_block(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let s = x.mean_keepdim(D::Minus1)?; // [B, C, 1]
        let s = self.conv(&s, &format!("{name}.conv1"), 1)?.relu()?;
        let s = candle_nn::ops::sigmoid(&self.conv(&s, &format!("{name}.conv2"), 1)?)?;
        x.broadcast_mul(&s)
    }

    /// `SqueezeExcitationRes2NetBlock`: `tdnn1 -> res2net -> tdnn2 -> se`, plus the
    /// block input as a residual.
    fn se_res2net(&self, x: &Tensor, name: &str, dilation: usize) -> Result<Tensor> {
        let h = self.tdnn(x, &format!("{name}.tdnn1.conv"), 1)?;
        let h = self.res2net(&h, &format!("{name}.res2net_block"), dilation)?;
        let h = self.tdnn(&h, &format!("{name}.tdnn2.conv"), 1)?;
        let h = self.se_block(&h, &format!("{name}.se_block"))?;
        h + x
    }

    /// Uniform-weight time statistics: `(mean, std)` over `[B, C, T]`, population
    /// variance clamped at `ASP_EPS` before the root.
    fn uniform_stats(x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = x.mean_keepdim(D::Minus1)?; // [B, C, 1]
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        Ok((mean, var.clamp(ASP_EPS, f64::INFINITY)?.sqrt()?))
    }

    /// `AttentiveStatisticsPooling`: build a per-frame attention from the frame itself
    /// concatenated with the utterance mean and std, then take *attention-weighted*
    /// statistics. `[B, C, T] -> [B, 2C, 1]`.
    fn asp(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(D::Minus1)?;
        let (mean, std) = Self::uniform_stats(x)?;
        let ctx = Tensor::cat(
            &[x, &mean.expand(x.shape())?, &std.expand(x.shape())?],
            1,
        )?; // [B, 3C, T]
        let a = self.tdnn(&ctx, "asp.tdnn.conv", 1)?.tanh()?;
        let a = self.conv(&a, "asp.conv", 1)?; // [B, C, T]
        let a = candle_nn::ops::softmax_last_dim(&a)?;

        // The attention weights sum to 1 over time, so these are the same formulas as
        // `uniform_stats` with `1/T` replaced by `a`.
        let mean = x.mul(&a)?.sum_keepdim(D::Minus1)?; // [B, C, 1]
        let var = x
            .broadcast_sub(&mean)?
            .sqr()?
            .mul(&a)?
            .sum_keepdim(D::Minus1)?;
        let std = var.clamp(ASP_EPS, f64::INFINITY)?.sqrt()?;
        debug_assert_eq!(a.dim(D::Minus1)?, t);
        Tensor::cat(&[&mean, &std], 1) // [B, 2C, 1]
    }

    /// `[B, T, mel_dim]` log-mel -> `[B, enc_dim]` x-vector.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?; // [B, mel_dim, T]
        let mut h = self.tdnn(&x, "blocks.0.conv", self.cfg.enc_dilations[0])?;
        // The MFA concatenates `hidden_states_list[1:]` — `blocks.0`'s output feeds the
        // next stage but is NOT aggregated.
        let mut aggregated: Vec<Tensor> = Vec::new();
        for i in self.cfg.se_res2net_range() {
            h = self.se_res2net(&h, &format!("blocks.{i}"), self.cfg.enc_dilations[i])?;
            aggregated.push(h.clone());
        }
        let h = Tensor::cat(&aggregated, 1)?;
        let h = self.tdnn(&h, "mfa.conv", *self.cfg.enc_dilations.last().unwrap())?;
        let h = self.asp(&h)?;
        self.conv(&h, "fc", 1)?.squeeze(D::Minus1)
    }

    /// Full clone path: 24 kHz mono `f32` samples in `[-1, 1]` -> `[1, enc_dim]`.
    ///
    /// Mirrors `extract_speaker_embedding`, including its `sr == 24000` assertion —
    /// the mel front end's `fmax` is exactly Nyquist for 24 kHz, so a clip at another
    /// rate would be analysed against the wrong filterbank rather than merely resampled.
    pub fn embed(&self, samples: &[f32], sample_rate: u32) -> Result<Tensor> {
        if sample_rate != self.cfg.sample_rate {
            return Err(candle_core::Error::Msg(format!(
                "speaker encoder: reference clip is {sample_rate} Hz, the encoder needs {} Hz \
                 (resample before calling)",
                self.cfg.sample_rate
            )));
        }
        let mel = log_mel_spectrogram(samples, &MelConfig::QWEN3_TTS, &self.dev)?;
        self.forward(&mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- deterministic fixture weights ------------------------------------
    //
    // Both the reference dump (`~/.venvs/qwen` + torch, CPU) and these tests fill every
    // parameter from `val(name, i)`: FNV-1a over the tensor name, mixed with the element
    // index through the same integer avalanche. Pure u32 arithmetic, so Python and Rust
    // agree bit for bit and the fixture needs no data file.

    fn fnv1a(s: &str) -> u32 {
        let mut h: u32 = 2166136261;
        for b in s.bytes() {
            h = (h ^ b as u32).wrapping_mul(16777619);
        }
        h
    }

    fn val(name: &str, i: usize) -> f32 {
        let mut x = fnv1a(name) ^ (i as u32).wrapping_mul(2654435761);
        x ^= x >> 15;
        x = x.wrapping_mul(2246822519);
        x ^= x >> 13;
        x = x.wrapping_mul(3266489917);
        x ^= x >> 16;
        (x % 2000) as f32 / 1000.0 - 1.0
    }

    fn filled(name: &str, shape: &[usize], scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| val(name, i) * scale).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    /// The reduced geometry the torch fixture was dumped at: real kernel sizes
    /// `[5,3,3,3,1]`, real dilations `[1,2,3,4,1]`, real res2net scale 8 — only the
    /// channel widths are shrunk. `enc_channels.last()` must stay `3 x` the SE-Res2Net
    /// width, because the MFA concatenates three stage outputs.
    fn small_cfg() -> SpeakerEncoderConfig {
        SpeakerEncoderConfig {
            mel_dim: 8,
            enc_dim: 12,
            enc_channels: vec![16, 16, 16, 16, 48],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 4,
            enc_res2net_scale: 8,
            enc_se_channels: 4,
            sample_rate: 24_000,
        }
    }

    fn small_encoder() -> SpeakerEncoder {
        let cfg = small_cfg();
        let mut map = HashMap::new();
        for (name, shape) in cfg.tensor_manifest() {
            map.insert(name.clone(), filled(&name, &shape, 0.1));
        }
        let w = Weights { map, dev: Device::Cpu, dt: DType::F32 };
        SpeakerEncoder::load(&w, "", cfg).unwrap()
    }

    // ---- manifest ---------------------------------------------------------

    /// The manifest must reproduce the published checkpoint exactly: 76 tensors, with
    /// the names and shapes read out of `0.6B-Base/model.safetensors`.
    #[test]
    fn manifest_matches_the_published_checkpoint() {
        let m = SpeakerEncoderConfig::default().tensor_manifest();
        assert_eq!(m.len(), 76, "the -Base checkpoints carry exactly 76 speaker_encoder tensors");
        let by_name: HashMap<&str, &Vec<usize>> =
            m.iter().map(|(n, s)| (n.as_str(), s)).collect();
        // Every shape below is a line of the safetensors header.
        let expect: &[(&str, &[usize])] = &[
            ("blocks.0.conv.weight", &[512, 128, 5]),
            ("blocks.0.conv.bias", &[512]),
            ("blocks.1.tdnn1.conv.weight", &[512, 512, 1]),
            ("blocks.1.res2net_block.blocks.0.conv.weight", &[64, 64, 3]),
            ("blocks.1.res2net_block.blocks.6.conv.weight", &[64, 64, 3]),
            ("blocks.1.se_block.conv1.weight", &[128, 512, 1]),
            ("blocks.1.se_block.conv2.weight", &[512, 128, 1]),
            ("blocks.3.tdnn2.conv.weight", &[512, 512, 1]),
            // MFA aggregates only blocks 1..3 -> 3 x 512 = 1536 in, not 4 x 512.
            ("mfa.conv.weight", &[1536, 1536, 1]),
            // ASP sees [x, mean, std] -> 3 x 1536 = 4608.
            ("asp.tdnn.conv.weight", &[128, 4608, 1]),
            ("asp.conv.weight", &[1536, 128, 1]),
            // fc sees the pooled mean ++ std -> 2 x 1536 = 3072.
            ("fc.weight", &[1024, 3072, 1]),
            ("fc.bias", &[1024]),
        ];
        for (n, s) in expect {
            assert_eq!(by_name.get(n).map(|v| v.as_slice()), Some(*s), "{n}");
        }
        // The 1.7B-Base differs from the 0.6B-Base in `enc_dim` and NOTHING else: its
        // `fc` is [2048, 3072, 1] / [2048] and all 74 other tensors are identical. Read
        // off both checkpoints' safetensors headers.
        let big = SpeakerEncoderConfig { enc_dim: 2048, ..SpeakerEncoderConfig::default() };
        let bm = big.tensor_manifest();
        assert_eq!(bm.len(), m.len());
        let differ: Vec<&str> = bm
            .iter()
            .zip(&m)
            .filter(|((n, s), (n2, s2))| n != n2 || s != s2)
            .map(|((n, _), _)| n.as_str())
            .collect();
        assert_eq!(differ, vec!["fc.weight", "fc.bias"]);
        // There is no res2net block number 7 (scale 8 gives scale-1 = 7 blocks, 0..=6).
        assert!(!by_name.contains_key("blocks.1.res2net_block.blocks.7.conv.weight"));
        // And no normalisation anywhere: every tensor is a conv weight or bias.
        for (n, _) in &m {
            assert!(n.ends_with(".weight") || n.ends_with(".bias"), "{n}");
            assert!(!n.contains("norm") && !n.contains("running"), "{n} looks like a norm layer");
        }
    }

    /// A checkpoint whose shape disagrees with the config must fail at load, naming the
    /// tensor — not twelve layers into a forward pass.
    #[test]
    fn load_rejects_a_shape_mismatch() {
        let cfg = small_cfg();
        let mut map = HashMap::new();
        for (name, shape) in cfg.tensor_manifest() {
            map.insert(name.clone(), filled(&name, &shape, 0.1));
        }
        // blocks.0 expects mel_dim input channels; hand it one too many.
        map.insert("blocks.0.conv.weight".into(), filled("x", &[16, 9, 5], 0.1));
        let w = Weights { map, dev: Device::Cpu, dt: DType::F32 };
        let err = SpeakerEncoder::load(&w, "", cfg).err().expect("mismatch must fail").to_string();
        assert!(err.contains("blocks.0.conv.weight"), "{err}");
        assert!(err.contains("[16, 9, 5]"), "{err}");
    }

    // ---- primitives -------------------------------------------------------

    /// `mode="reflect"` mirrors *about* the edge sample without repeating it, which is
    /// what separates it from `replicate`.
    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 4), &Device::Cpu).unwrap();
        let got: Vec<f32> = reflect_pad(&x, 2, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(got, vec![3., 2., 1., 2., 3., 4., 3., 2.]);
        // replicate would have given [1,1,1,2,3,4,4,4]; zero padding [0,0,1,2,3,4,0,0].
        assert_ne!(got[0], 1.0);
        assert_ne!(got[0], 0.0);
        // asymmetric padding is left-biased the way torch's `same` split is
        let got: Vec<f32> = reflect_pad(&x, 1, 2).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![2., 1., 2., 3., 4., 3., 2.]);
        // and a pad the signal cannot support is an error, not a panic
        assert!(reflect_pad(&x, 4, 0).is_err());
        assert!(reflect_pad(&x, 3, 0).is_ok());
    }

    /// `same` padding must leave the time axis unchanged at every kernel/dilation pair
    /// the real config uses — the property that lets the MFA concatenate stage outputs.
    #[test]
    fn same_padding_preserves_the_time_axis() {
        let enc = small_encoder();
        let t = 20usize;
        let x = filled("probe", &[1, 8, t], 1.0);
        let y = enc.tdnn(&x, "blocks.0.conv", 1).unwrap();
        assert_eq!(y.dims(), &[1, 16, t], "k=5 d=1");
        for (i, d) in [(1usize, 2usize), (2, 3), (3, 4)] {
            let z = enc
                .res2net(&y, &format!("blocks.{i}.res2net_block"), d)
                .unwrap();
            assert_eq!(z.dims(), &[1, 16, t], "k=3 d={d}");
        }
    }

    // ---- mel front end ----------------------------------------------------

    /// Slaney mel is linear below 1 kHz and logarithmic above, and the two branches meet
    /// continuously at the 1 kHz knee. Also the round trip must be an identity.
    #[test]
    fn slaney_mel_scale_is_continuous_at_the_knee() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9, "1 kHz is mel 15 on the slaney scale");
        assert!((hz_to_mel(999.999) - 15.0).abs() < 1e-4, "linear side meets the knee");
        assert!((hz_to_mel(500.0) - 7.5).abs() < 1e-9, "linear below the knee");
        assert!(hz_to_mel(2000.0) > 15.0 && hz_to_mel(2000.0) < 30.0, "compressive above");
        for f in [0.0, 250.0, 999.0, 1000.0, 1001.0, 4000.0, 12000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6, "round trip at {f} Hz");
        }
    }

    /// The filterbank must be triangular, non-negative, band-limited to `fmin..fmax`,
    /// and slaney-normalised (constant area, *not* unit peak).
    #[test]
    fn mel_filterbank_is_slaney_normalised_triangles() {
        let mc = MelConfig::QWEN3_TTS;
        let n_bins = mc.n_fft / 2 + 1;
        let w = mel_filterbank(&mc);
        assert_eq!(w.len(), mc.num_mels * n_bins);
        assert!(w.iter().all(|v| *v >= 0.0), "no negative weights");
        // Slaney norm equalises filter *area*, not peak height: a `norm=None` bank would
        // peak at exactly 1.0 everywhere, whereas here the narrow low filters peak an
        // order of magnitude above the wide high ones while both integrate to ~0.042.
        // The values are `librosa.filters.mel(sr=24000, n_fft=1024, n_mels=128, fmin=0,
        // fmax=12000)` read straight off the reference bank.
        let peak = |i: usize| w[i * n_bins..(i + 1) * n_bins].iter().cloned().fold(0f32, f32::max);
        let area = |i: usize| w[i * n_bins..(i + 1) * n_bins].iter().sum::<f32>();
        assert!(peak(0) > peak(127), "narrow low filters peak higher under slaney norm");
        assert!(peak(0) < 1.0, "slaney norm does not normalise peaks to 1");
        for (i, want) in [(0usize, 0.033550426f32), (40, 0.020846143), (127, 0.0030868005)] {
            assert!((peak(i) - want).abs() < 1e-6, "peak of filter {i}: {} vs {want}", peak(i));
        }
        for (i, want) in [(0usize, 0.04211951f32), (127, 0.042641446)] {
            assert!((area(i) - want).abs() < 1e-5, "area of filter {i}: {} vs {want}", area(i));
        }
        // Each filter is unimodal: weights rise to a peak then fall.
        for i in [0usize, 40, 127] {
            let row = &w[i * n_bins..(i + 1) * n_bins];
            let argmax = (0..n_bins).max_by(|a, b| row[*a].total_cmp(&row[*b])).unwrap();
            assert!(row[..argmax].windows(2).all(|p| p[0] <= p[1]), "filter {i} rises");
            assert!(row[argmax..].windows(2).all(|p| p[0] >= p[1]), "filter {i} falls");
        }
    }

    /// Golden values from the reference `mel_spectrogram(n_fft=1024, num_mels=128,
    /// sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000)` run on
    /// CPU torch + librosa over the deterministic 2048-sample `val("wave", i) * 0.5`
    /// waveform. This pins the whole front end at once: the reflect pre-pad, the
    /// periodic Hann window, `center=False`, the `+1e-9` magnitude, the slaney
    /// filterbank and the `1e-5` log floor.
    #[test]
    fn log_mel_matches_the_reference_front_end() {
        let n = 2048usize;
        let wave: Vec<f32> = (0..n).map(|i| val("wave", i) * 0.5).collect();
        let mel = log_mel_spectrogram(&wave, &MelConfig::QWEN3_TTS, &Device::Cpu).unwrap();
        // (2048 + 2*384 - 1024) / 256 + 1 = 8 frames, 128 bins, channels-last.
        assert_eq!(mel.dims(), &[1, 8, 128]);
        let v = mel.squeeze(0).unwrap().t().unwrap().contiguous().unwrap(); // [128, 8]
        let got: Vec<Vec<f32>> = v.to_vec2().unwrap();
        let expect: &[(usize, [f32; 8])] = &[
            (0, [-2.1161153, -1.9788699, -1.5735432, -1.6354765, -1.9664986, -1.8051517, -1.1693287, -1.5941738]),
            (1, [-2.553666, -3.0988536, -1.4942969, -1.4819148, -1.7277899, -1.4652451, -1.1079483, -2.0952187]),
            (5, [-0.9807829, -1.1593872, -0.9446059, -0.8908335, -2.9697702, -2.5987668, -1.9972831, -1.2938567]),
            (40, [-2.1166794, -1.7436324, -1.2012424, -1.9712827, -1.9513607, -1.1317865, -0.9581384, -1.0456305]),
            (127, [-1.4655644, -1.3606862, -1.3406309, -1.6259067, -1.7826723, -1.7778022, -1.5169666, -1.3726014]),
        ];
        for (bin, want) in expect {
            for (t, w) in want.iter().enumerate() {
                let g = got[*bin][t];
                assert!((g - w).abs() < 2e-4, "mel bin {bin} frame {t}: {g} vs {w}");
            }
        }
    }

    /// A clip shorter than the reflect pre-pad is an error with a usable message, not a
    /// slice panic.
    #[test]
    fn log_mel_rejects_a_clip_shorter_than_its_pad() {
        let short = vec![0.1f32; 384];
        let err = log_mel_spectrogram(&short, &MelConfig::QWEN3_TTS, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(err.contains("too short"), "{err}");
        // One sample past the pad reflects fine, and at this config it also already
        // fills a frame (pad 384 > n_fft - hop*... ), so the frame guard needs a
        // geometry where it is reachable: n_fft 16 / hop 8 pads by 4, so 5..7 samples
        // pad cleanly yet cannot fill one 16-sample frame.
        assert!(log_mel_spectrogram(&vec![0.1f32; 385], &MelConfig::QWEN3_TTS, &Device::Cpu).is_ok());
        let tiny = MelConfig { n_fft: 16, num_mels: 4, sample_rate: 24_000, hop_size: 8, win_size: 16, fmin: 0, fmax: 12_000 };
        let err = log_mel_spectrogram(&[0.1f32; 7], &tiny, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no complete"), "{err}");
        // and one more sample is enough for exactly one frame
        assert!(log_mel_spectrogram(&[0.1f32; 8], &tiny, &Device::Cpu).is_ok());
    }

    // ---- the encoder ------------------------------------------------------

    /// End-to-end parity against the reference `Qwen3TTSSpeakerEncoder`, run on CPU
    /// torch at the reduced geometry of [`small_cfg`] with the shared deterministic
    /// weight fill. Real kernel sizes, real dilations, real res2net scale, real block
    /// wiring — only the channel counts are small enough to write the answer down.
    ///
    /// This is the test that would catch a zero-padded convolution, a missing residual,
    /// an independent (rather than cascaded) Res2Net, an aggregated `blocks.0`, or a
    /// mean-only pooling.
    #[test]
    fn forward_matches_the_reference_encoder() {
        let enc = small_encoder();
        let (t, mel_dim) = (20usize, 8usize);
        let mel = filled("mel", &[1, t, mel_dim], 1.0);
        let out = enc.forward(&mel).unwrap();
        assert_eq!(out.dims(), &[1, 12]);
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let want = [
            -0.0957093f32, -0.0199489, -0.1214721, 0.0228288, -0.0400302, -0.0188473,
            0.0779235, -0.0329143, 0.0642183, 0.0110553, -0.0695315, 0.0053106,
        ];
        for (i, w) in want.iter().enumerate() {
            assert!((got[i] - w).abs() < 1e-5, "component {i}: {} vs {w}", got[i]);
        }
    }

    /// The x-vector is `enc_dim` wide and the time axis is gone: pooling must collapse
    /// any number of frames to the same fixed-size embedding.
    #[test]
    fn pooling_collapses_time_to_a_fixed_enc_dim_embedding() {
        let enc = small_encoder();
        assert_eq!(enc.config().enc_dim, 12);
        let mut prev: Option<Vec<f32>> = None;
        for t in [20usize, 37, 64] {
            let mel = filled("mel", &[1, t, 8], 1.0);
            let out = enc.forward(&mel).unwrap();
            assert_eq!(out.dims(), &[1, 12], "T={t} must still give [1, enc_dim]");
            let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            // ...and it must actually depend on the content, not be a constant.
            if let Some(p) = &prev {
                assert!(p.iter().zip(&v).any(|(a, b)| (a - b).abs() > 1e-6), "T={t} embedding is constant");
            }
            prev = Some(v);
        }
    }

    /// The published geometry: `blocks.0` consumes 128 mel bins and `fc` emits the
    /// configured 1024-wide x-vector, sourced from `config.json`'s `enc_dim`.
    #[test]
    fn published_geometry_is_128_mel_in_1024_enc_out() {
        let cfg = SpeakerEncoderConfig::default();
        assert_eq!(cfg.mel_dim, MelConfig::QWEN3_TTS.num_mels, "the front end feeds blocks.0");
        assert_eq!(cfg.enc_dim, 1024);
        assert_eq!(cfg.sample_rate, MelConfig::QWEN3_TTS.sample_rate);
        // fmax is exactly Nyquist at 24 kHz — the front end analyses the whole band.
        assert_eq!(MelConfig::QWEN3_TTS.fmax * 2, MelConfig::QWEN3_TTS.sample_rate);
        // and it tracks the parsed model config rather than being hardcoded twice
        let parsed = crate::Qwen3TtsConfig::from_json(
            r#"{"model_type":"qwen3_tts","tts_model_type":"base",
                "speaker_encoder_config":{"enc_dim":192,"sample_rate":16000},
                "talker_config":{"hidden_size":1024,"num_hidden_layers":1,
                  "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
                  "intermediate_size":3072,"vocab_size":3072,"num_code_groups":16,
                  "code_predictor_config":{"hidden_size":1024,"num_hidden_layers":1,
                    "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
                    "intermediate_size":3072,"vocab_size":2048}}}"#,
        )
        .unwrap();
        let derived = SpeakerEncoderConfig::from_model_config(&parsed);
        assert_eq!(derived.enc_dim, 192);
        assert_eq!(derived.sample_rate, 16_000);
    }

    /// **On-box parity against the real published weights.** Opt-in: set
    /// `SYRINX_QWEN_BASE_DIR` to a `Qwen3-TTS-12Hz-*-Base` directory (the one holding
    /// `model.safetensors`) and this loads all 76 `speaker_encoder.*` tensors and runs a
    /// 1-second reference clip end to end. Skips loudly when the variable is unset, so
    /// it never reports a pass it did not earn.
    ///
    /// The expected vectors are the reference `Qwen3TTSSpeakerEncoder` (weights cast to
    /// f32, `strict=True` state-dict load, all 76 keys matched) over the same
    /// deterministic `val("clip", i) * 0.5` waveform, for **both** published `-Base`
    /// checkpoints — 0.6B (1024-wide) and 1.7B (2048-wide).
    #[test]
    fn real_checkpoint_parity() {
        let Ok(dir) = std::env::var("SYRINX_QWEN_BASE_DIR") else {
            eprintln!("SKIP real_checkpoint_parity: set SYRINX_QWEN_BASE_DIR to a -Base checkpoint dir");
            return;
        };
        let map = candle_core::safetensors::load(
            std::path::Path::new(&dir).join("model.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let w = Weights { map, dev: Device::Cpu, dt: DType::F32 };
        // enc_dim comes from the checkpoint's own config, not a constant: 0.6B-Base is
        // 1024 and 1.7B-Base is 2048, so this test runs unchanged against either.
        let json = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json")).unwrap();
        let cfg = SpeakerEncoderConfig::from_model_config(
            &crate::Qwen3TtsConfig::from_json(&json).unwrap(),
        );
        let enc_dim = cfg.enc_dim;
        let enc = SpeakerEncoder::load(&w, "speaker_encoder", cfg).unwrap();

        let wave: Vec<f32> = (0..24_000).map(|i| val("clip", i) * 0.5).collect();
        let out = enc.embed(&wave, 24_000).unwrap();
        assert_eq!(out.dims(), &[1, enc_dim]);
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // Components 0..3, 100, 500, 900 and the last three, plus the L2 norm, of both
        // published `-Base` checkpoints' reference embeddings for this clip.
        let (head, tail, want_norm) = match enc_dim {
            1024 => (
                [0.171781f32, 0.083519, 0.148050, -0.071909, -0.897749, 0.060546, -0.046900],
                [-0.000609f32, -0.026585, 0.026628],
                8.571402f32,
            ),
            2048 => (
                [0.083462f32, 0.049187, 0.151898, 0.084708, 0.107057, -0.130549, 0.000186],
                [0.073778f32, -0.133008, -0.138704],
                15.729869f32,
            ),
            other => panic!("no reference vector for a {other}-wide speaker encoder"),
        };
        for (k, i) in [0usize, 1, 2, 3, 100, 500, 900].iter().enumerate() {
            assert!((v[*i] - head[k]).abs() < 1e-4, "component {i}: {} vs {}", v[*i], head[k]);
        }
        for (k, i) in [enc_dim - 3, enc_dim - 2, enc_dim - 1].iter().enumerate() {
            assert!((v[*i] - tail[k]).abs() < 1e-4, "component {i}: {} vs {}", v[*i], tail[k]);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - want_norm).abs() < 1e-3, "embedding norm {norm} vs {want_norm}");
    }

    /// `embed` refuses a clip at the wrong rate rather than analysing it against a
    /// filterbank built for 24 kHz.
    #[test]
    fn embed_refuses_a_wrong_sample_rate() {
        let enc = small_encoder();
        let err = enc.embed(&vec![0.0f32; 4096], 16_000).unwrap_err().to_string();
        assert!(err.contains("16000") && err.contains("24000"), "{err}");
    }
}
