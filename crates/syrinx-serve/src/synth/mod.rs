//! End-to-end CosyVoice2 zero-shot **synthesizer** (the `real`-feature capstone).
//!
//! This is the one module that wires the five already-parity-verified Syrinx
//! components into a single `text + reference voice -> 24 kHz audio` pipeline,
//! reproducing the CosyVoice2 zero-shot data flow exactly:
//!
//! ```text
//!   text_token   = tokenizer(prompt_text) ++ tokenizer(tts_text)        (syrinx-frontend)
//!   fbank        = kaldi_fbank(ref_16k); fbank -= mean_over_time         (syrinx-frontend)
//!   spk          = camplus(fbank[1,T,80])  -> [1,192]                    (syrinx-speaker)
//!   prompt_token = speech_tokenizer(ref_16k)                            (syrinx-frontend ort)
//!   prompt_feat  = prompt_mel(ref_24k)^T   -> [1,Mp,80]                  (syrinx-frontend)
//!   (CosyVoice2 %2 alignment: token_len = min(Mp/2, |prompt_token|);
//!    prompt_feat = prompt_feat[:, :2*token_len]; prompt_token[:token_len])
//!   speech_token = lm.generate(text_token, prompt_token, min_len, max_len, seed)  (syrinx-lm)
//!   audio        = token2wav(flow, hift, prompt_token, speech_token,
//!                            prompt_feat, spk, z, s_stft)  -> [1,L]      (syrinx-acoustic+vocoder)
//! ```
//!
//! ## Determinism and the two stochastic inputs
//!
//! Two inputs of the real model are *not* bit-portable across torch/Candle RNGs:
//! the CFM noise `z` (the flow's fixed `rand_noise` design buffer) and the HiFT
//! source STFT `s_stft` (SineGen has a random initial phase + Gaussian noise). The
//! per-piece parity tracks pin these from the reference. The synthesizer therefore
//! lets the caller **inject** both (plus a pinned generated-token list), via
//! [`SynthInputs`], for a deterministic full-chain check. When they are not
//! injected, [`Synthesizer::synthesize`] falls back to:
//!   * `z` = zeros (the documented [`Flow::cfm_solve`] fallback — a valid ODE init,
//!     just not the reference's noise), and
//!   * a **deterministic, zero-phase, noise-free** source built from the HiFT F0
//!     predictor (see [`Synthesizer::deterministic_source_stft`]). This is a
//!     faithful smoke source — finite, non-silent, the right length — that exercises
//!     the entire vocoder source branch, but it does NOT reproduce the model's
//!     random-phase SineGen and so is not a parity source.
//!
//! ## Module layout
//!
//! The pipeline is split across focused submodules (all `impl Synthesizer` blocks on
//! the one struct defined here): [`cond`] (frontend conditioning), [`generate`] (LM
//! speech-token generation), [`token2wav`] (flow + vocoder glue), [`source`] (HiFT
//! source builders + the quality path), [`streaming`] (incremental synthesis),
//! [`instruct`] (instruct/emotion), [`watermark`] (watermark glue) and [`prosody`]
//! (rate + render-plan control). The struct, the input/output/error types, the shared
//! constants and the cross-module helpers live in this `mod.rs`.

mod cond;
mod generate;
mod instruct;
mod prosody;
mod source;
mod streaming;
mod token2wav;
mod watermark;

use candle_core::{DType, Device, Tensor};

use syrinx_acoustic::real::Flow;
use syrinx_frontend::speech_token::{SpeechTokenError, SpeechTokenizer};
use syrinx_frontend::tokenizer::{TextTokenizer, TokenizerError};
use syrinx_speaker::real::CamPlus;
use syrinx_vocoder::real::HiftVocoder;

/// Select the compute device for the synthesizer.
///
/// With the `cuda` feature built, `ordinal = Some(i)` opens `cuda:i`; `None`
/// asks for `cuda:0`. Without the `cuda` feature (or if any CUDA open fails),
/// this falls back to [`Device::Cpu`] — the parity device — so the CPU build
/// keeps working unchanged.
///
/// CUDA is for **speed only**; its output is not bit-equal to the CPU reference
/// (see [`Synthesizer::load_on_device`]).
pub fn pick_device(ordinal: Option<usize>) -> Device {
    #[cfg(feature = "cuda")]
    {
        let i = ordinal.unwrap_or(0);
        match Device::new_cuda(i) {
            Ok(d) => return d,
            Err(e) => {
                eprintln!("syrinx: cuda:{i} unavailable ({e}); falling back to CPU");
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
    }
    Device::Cpu
}

/// kaldi fbank params (CAM++ input): 80 mel bins, 16 kHz.
const FBANK_MELS: usize = 80;
const SR_16K: f32 = 16_000.0;

/// matcha prompt-mel params (flow `prompt_feat`): 24 kHz.
const MEL_N_FFT: usize = 1920;
const MEL_NUM_MELS: usize = 80;
const MEL_SR: f32 = 24_000.0;
const MEL_HOP: usize = 480;
const MEL_WIN: usize = 1920;
const MEL_FMIN: f32 = 0.0;
const MEL_FMAX: f32 = 8000.0;

/// CFM Euler step count (CosyVoice2 `n_timesteps`).
const N_TIMESTEPS: usize = 10;

/// CosyVoice2 LM length ratios: `min_len = (text-prompt_text)*2`, `max_len = *20`.
const MIN_TOKEN_TEXT_RATIO: usize = 2;
const MAX_TOKEN_TEXT_RATIO: usize = 20;

/// HiFT iSTFT params (must match the vocoder): n_fft=16, hop=4.
const HIFT_N_FFT: usize = 16;
const HIFT_HOP: usize = 4;
const HIFT_BINS: usize = HIFT_N_FFT / 2 + 1; // 9
/// Upsample product * istft hop = 8*5*3 * 4 = 480: the f0 -> source upsample factor.
const F0_UPSAMPLE: usize = 480;
/// SineGen sine amplitude (`sine_amp` / `nsf_alpha`, CosyVoice2 default).
const SINE_AMP: f64 = 0.1;
/// SineGen additive-noise std for *voiced* frames (`noise_std` / `nsf_sigma`).
const NSF_NOISE_STD: f64 = 0.003;
/// Number of harmonic overtones above the fundamental (`nb_harmonics`); the source
/// has `NB_HARMONICS + 1 = 9` sine components (fundamental + 8 overtones).
const NB_HARMONICS: usize = 8;
/// F0 threshold (Hz) for the voiced/unvoiced mask (`nsf_voiced_threshold`).
const NSF_VOICED_THRESHOLD: f64 = 10.0;

/// Paths to every sub-model's on-disk weights / assets. All are required for a
/// real synthesizer; parameterized so tests + callers point at the model box.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    /// `llm_fp32.safetensors` — the Qwen2-0.5B LM + `llm_decoder`.
    pub lm_weights: String,
    /// `campplus_weights.safetensors` — the CAM++ speaker encoder.
    pub spk_weights: String,
    /// `flow_fp32.safetensors` — the flow-matching mel decoder.
    pub flow_weights: String,
    /// `hift_fp32.safetensors` — the HiFT vocoder.
    pub hift_weights: String,
    /// `tokenizer.json` — the Qwen2 BPE text tokenizer.
    pub tokenizer_json: String,
    /// `speech_tokenizer_v2.onnx` — the prompt speech-token tokenizer.
    pub speech_tokenizer_onnx: String,
}

/// Optional injected inputs for a *deterministic* run. Any field left `None`
/// is derived live (see the module docs). Pinning all three makes the full chain
/// bit-reproducible against the e2e reference.
#[derive(Default)]
pub struct SynthInputs {
    /// Pinned generated speech-token ids (i64), bypassing live LM sampling.
    pub pinned_speech_token: Option<Vec<i64>>,
    /// Pinned CFM noise `z` `[1, 80, total]` (the flow's `rand_noise` slice).
    pub z: Option<Tensor>,
    /// Pinned HiFT source STFT `s_stft` `[1, 18, T_src]`.
    pub s_stft: Option<Tensor>,
    /// LM sampling seed (live path only). Defaults to 0.
    pub lm_seed: u64,
    /// Optional hard cap on live LM generation steps. The real `max_len` is
    /// `(text-prompt_text)*20`; the LM forward has no KV cache (O(n²) per step), so
    /// a cap keeps the functional smoke path tractable on CPU. `None` uses the real
    /// ratio. `min_len` is always honoured (the cap is raised to at least `min_len`).
    pub max_gen_steps: Option<usize>,
}

/// Errors raised while building or running the synthesizer.
#[derive(Debug)]
pub enum SynthError {
    /// A sub-model failed to load or a Candle op failed.
    Candle(String),
    /// The text tokenizer failed.
    Tokenizer(String),
    /// The speech-token tokenizer (ONNX) failed.
    SpeechToken(String),
}

impl std::fmt::Display for SynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthError::Candle(m) => write!(f, "synth candle error: {m}"),
            SynthError::Tokenizer(m) => write!(f, "synth tokenizer error: {m}"),
            SynthError::SpeechToken(m) => write!(f, "synth speech-token error: {m}"),
        }
    }
}

impl std::error::Error for SynthError {}

impl From<candle_core::Error> for SynthError {
    fn from(e: candle_core::Error) -> Self {
        SynthError::Candle(e.to_string())
    }
}
impl From<TokenizerError> for SynthError {
    fn from(e: TokenizerError) -> Self {
        SynthError::Tokenizer(e.to_string())
    }
}
impl From<SpeechTokenError> for SynthError {
    fn from(e: SpeechTokenError) -> Self {
        SynthError::SpeechToken(e.to_string())
    }
}

/// The prompt-side conditioning values derived from a reference voice clip — the
/// frontend half of the pipeline, exposed so the deterministic e2e parity test can
/// assert each value against the reference *before* the flow/vocoder run.
pub struct PromptCond {
    /// Text-token ids for `prompt_text ++ tts_text` (u32, as the LM consumes them).
    pub text_token: Vec<u32>,
    /// Number of leading ids belonging to `prompt_text` (for the LM length ratios).
    pub prompt_text_len: usize,
    /// Speaker x-vector `[1, 192]`.
    pub spk_embedding: Tensor,
    /// Prompt speech-token ids `[1, token_len]` (i64), after the %2 alignment.
    pub prompt_token: Tensor,
    /// Prompt mel `[1, 2*token_len, 80]` (frame-major), after the %2 alignment.
    pub prompt_feat: Tensor,
}

/// The end-to-end synthesizer: holds every loaded sub-model.
pub struct Synthesizer {
    tokenizer: TextTokenizer,
    speech_tokenizer: SpeechTokenizer,
    speaker: CamPlus,
    lm: syrinx_lm::real::Qwen2Lm,
    flow: Flow,
    vocoder: HiftVocoder,
    dev: Device,
}

impl Synthesizer {
    /// Load every sub-model from `cfg` onto the CPU (the parity device).
    ///
    /// This is the default, numerically-verified path. For the GPU speed path
    /// (built with the `cuda` feature) use [`Synthesizer::load_on_device`] with
    /// a CUDA device from [`pick_device`].
    pub fn load(cfg: &SynthConfig) -> Result<Self, SynthError> {
        Self::load_on_device(cfg, Device::Cpu)
    }

    /// Load every sub-model from `cfg` onto an explicit `dev`.
    ///
    /// All Candle sub-models already take a `Device` at load time and build
    /// every constant from `self.dev`, so a single device threaded here drives
    /// the whole pipeline on that backend. The speech-token ONNX prompt step
    /// still runs on its own CPU runtime (`ort`); only the Candle stages move.
    ///
    /// GPU output will not bit-match the CPU reference (CPU-vs-GPU gemm
    /// accumulation diverges through deep nets) — the CUDA path is for **speed**
    /// and its verification is functional (finite, non-silent, sane-length
    /// audio + a measured speedup), never 1e-3 parity. CPU stays the parity
    /// device.
    pub fn load_on_device(cfg: &SynthConfig, dev: Device) -> Result<Self, SynthError> {
        Self::load_on_device_inner(cfg, dev, false)
    }

    /// Load every sub-model in its **quantized** variant for the README 4-bit footprint
    /// track (realized ≈388 MB CV2; the early ~270 MB budget under-counted the Qwen2-0.5B
    /// body): the LM via [`syrinx_lm::real::Qwen2Lm::load_quantized`] (int4 big linears +
    /// int4 dequant-on-gather embedding tables + dropped `lm_head`), the flow via
    /// [`Flow::load_quantized`] (Q4_0 `linear()` weights), the HiFT vocoder via
    /// `HiftVocoder::load_quantized` and the CAM++ speaker via `CamPlus::load_quantized`
    /// (Q4_0 dequant-on-fetch conv/linear kernels). Tokenizers load exactly as in [`load`].
    ///
    /// int4 trades quality for size; CPU stays the parity device for the **fp32** path
    /// ([`load`]), and the quantized quality is measured on the box (SIM-o), not asserted
    /// to fp32 parity here.
    pub fn load_quantized(cfg: &SynthConfig) -> Result<Self, SynthError> {
        Self::load_on_device_inner(cfg, Device::Cpu, true)
    }

    /// [`load_quantized`](Self::load_quantized) on an explicit device (e.g. a CUDA
    /// device from [`pick_device`] for the speed path).
    pub fn load_quantized_on_device(cfg: &SynthConfig, dev: Device) -> Result<Self, SynthError> {
        Self::load_on_device_inner(cfg, dev, true)
    }

    /// Shared loader for the fp32 and quantized paths. `quantized` selects the LM **and**
    /// flow precision (int4 linears + int8 embeds for the LM, Q4_0 linears for the flow);
    /// all other sub-models are identical.
    fn load_on_device_inner(
        cfg: &SynthConfig,
        dev: Device,
        quantized: bool,
    ) -> Result<Self, SynthError> {
        let tokenizer = TextTokenizer::from_file(&cfg.tokenizer_json)?;
        let speech_tokenizer = SpeechTokenizer::load(&cfg.speech_tokenizer_onnx)?;
        let speaker = if quantized {
            CamPlus::load_quantized(&cfg.spk_weights, dev.clone())?
        } else {
            CamPlus::load(&cfg.spk_weights, dev.clone())?
        };
        let lm = if quantized {
            syrinx_lm::real::Qwen2Lm::load_quantized(&cfg.lm_weights, dev.clone())?
        } else {
            syrinx_lm::real::Qwen2Lm::load(&cfg.lm_weights, dev.clone())?
        };
        let flow = if quantized {
            Flow::load_quantized(&cfg.flow_weights, dev.clone())?
        } else {
            Flow::load(&cfg.flow_weights, dev.clone())?
        };
        let vocoder = if quantized {
            HiftVocoder::load_quantized(&cfg.hift_weights, dev.clone())?
        } else {
            HiftVocoder::load(&cfg.hift_weights, dev.clone())?
        };
        Ok(Self {
            tokenizer,
            speech_tokenizer,
            speaker,
            lm,
            flow,
            vocoder,
            dev,
        })
    }

    /// The device every Candle sub-model was loaded onto.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Full synthesis: `tts_text` spoken in the reference voice, returning the 24 kHz
    /// waveform as a flat `Vec<f32>`. `inputs` may pin the generated tokens, `z`, and
    /// the HiFT source for a deterministic run (see [`SynthInputs`]).
    pub fn synthesize(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;

        // speech tokens: pinned or live.
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        let audio = self.token2wav(&cond, &speech_token, inputs)?; // [1, L]
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }
}

// ---- text-normalization hook (additive) --------------------------------------
//
// Mirrors CosyVoice2's `frontend.text_normalize` (wetext zh+en). Gated on the
// crate `tn` feature so it is fully opt-in: with `tn` off (e.g. the raw-text
// parity tests, which run `--features real` only) `tn_normalize` is the identity,
// leaving the un-normalized text path byte-for-byte unchanged.

/// Normalize text before tokenizing when the `tn` feature is enabled.
#[cfg(feature = "tn")]
pub(crate) fn tn_normalize(s: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Owned(syrinx_frontend::textnorm::normalize_text(s))
}

/// Identity passthrough when `tn` is disabled (raw text, as before).
#[cfg(not(feature = "tn"))]
pub(crate) fn tn_normalize(s: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Borrowed(s)
}

// ---- free helpers (shared across the pipeline submodules) ---------------------

/// i64 ids -> `[1, n]` i64 tensor.
fn ids_i64_to_tensor(ids: &[i64], dev: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(ids.to_vec(), (1, ids.len()), dev)
}

/// `[1, n]` (or `[n]`) i64 token tensor -> `Vec<u32>`.
fn tensor_ids_u32(t: &Tensor) -> candle_core::Result<Vec<u32>> {
    let flat = t.flatten_all()?.to_dtype(DType::U32)?;
    flat.to_vec1::<u32>()
}
