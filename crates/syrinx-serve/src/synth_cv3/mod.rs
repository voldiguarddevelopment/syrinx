//! End-to-end **CosyVoice3** zero-shot synthesizer (the CV3 `real`-feature capstone).
//!
//! This ties the four already-parity-verified CV3 component ports into one
//! `text + reference voice -> 24 kHz audio` pipeline, mirroring the structure of the
//! CV2 [`crate::synth::Synthesizer`] (`prompt_cond` -> LM -> flow -> source -> HiFT)
//! with the CV3 parts swapped in:
//!
//! ```text
//!   text_token   = tokenizer(prompt_text ++ "<|endofprompt|>") ++ tokenizer(tts_text)
//!   fbank        = kaldi_fbank(ref_16k); fbank -= mean_over_time                (frontend)
//!   embedding    = camplus(fbank[1,T,80])  -> [1,192]                           (speaker)
//!   prompt_token = speech_tokenizer_v3(ref_16k)                                 (frontend ort)
//!   prompt_feat  = prompt_mel(ref_24k)^T   -> [1,Mp,80]                          (frontend)
//!   (CosyVoice %2 alignment: token_len = min(Mp/2, |prompt_token|);
//!    prompt_feat = prompt_feat[:, :2*token_len]; prompt_token[:token_len])
//!   speech_token = Cv3Lm.generate(text_token, prompt_token, min,max, seed)      (syrinx-lm)
//!   mel          = Cv3Flow.forward(prompt_token, speech_token, prompt_feat,
//!                                  embedding, z)[:, :, prompt_len:]              (syrinx-acoustic)
//!   source       = f0(mel) -> upsample 480 -> sine                              (HiFT excitation)
//!   audio        = Cv3Hift.decode(mel, source) -> [1, L]                        (syrinx-vocoder)
//! ```
//!
//! ## What CV3 reuses vs adapts (relative to the CV2 frontend half)
//! The whole frontend half is the **same math** as CV2 — CosyVoice3's
//! `frontend.py::_extract_speech_feat` / `_extract_spk_embedding` are unchanged from
//! CV2 (verified against the e2e fixture's `prompt_feat`/`embedding`), so the Qwen BPE
//! text tokenizer, the kaldi-fbank + per-time-mean CAM++ speaker embedding, and the
//! 24 kHz matcha prompt-mel (`prompt_feat`) are reused verbatim. The two CV3 deltas:
//!   * the **v3 speech tokenizer** (`SpeechTokenizer::load_cv3`, `speech_tokenizer_v3.onnx`)
//!     — identical I/O to v2, different ONNX file; and
//!   * the LM text prompt carries a trailing **`<|endofprompt|>`** (id 151646) on the
//!     prompt-text segment (CV3's `Qwen2LM` boundary marker; CV2 had none in zero-shot).
//!
//! ## The two stochastic inputs (determinism)
//! As in CV2, two inputs are not bit-portable across torch/Candle RNGs: the CFM noise
//! `z` (the flow's fixed `rand_noise` design buffer) and the HiFT SineGen `source`. For
//! a deterministic chain check the caller injects `z` (see [`Cv3SynthInputs`] +
//! [`Cv3Synthesizer::flow_from_reference_tokens`]). When not injected,
//! [`Cv3Synthesizer::synthesize`] now fills `z` with a **seeded standard-normal** init
//! (the flow's `rand_noise` *distribution*, reproducible from `lm_seed`) instead of the
//! degenerate `z = zeros` (which collapses the CFM ODE onto its mean trajectory — a
//! smeared mel, the measured ~0.3-MOS live-quality loss CV2 hit), and a
//! **deterministic, zero-phase, single-harmonic** HiFT source built from the CV3 HiFT
//! F0 predictor (see [`Cv3Synthesizer::deterministic_source`]). CV3's `Cv3Hift::decode`
//! consumes the source **waveform** `[1,1,L]` (it computes the STFT internally), unlike
//! CV2's `token2wav`, which takes the pre-STFT'd `s_stft`. The smoke source is a
//! faithful excitation — finite, non-silent, the right length — but is NOT a parity
//! source (it omits the model's random per-harmonic phase + Gaussian noise + learned
//! `m_source.l_linear` merge, none of which is bit-portable anyway).
//!
//! Gated behind the `real` feature + on-disk CV3 weights; the parity test
//! (`tests/real_cv3_e2e_parity.rs`) skips cleanly when the fixtures are absent. The CV2
//! [`crate::synth`] module is left byte-unchanged — this is purely additive.
//!
//! ## Module layout
//! The pipeline is split across focused submodules (all `impl Cv3Synthesizer` blocks on
//! the one struct here): [`cond`] (frontend conditioning), [`generate`] (LM speech-token
//! generation), [`token2wav`] (flow + vocoder glue + the seeded `z` builder), [`source`]
//! (HiFT source builders + the quality path + the model-free `*_from_f0` seams),
//! [`streaming`] (chunked-causal streaming) and [`instruct`] (instruct/emotion). The
//! struct, the input/config types, the shared constants/PRNG/helpers and the
//! safetensors pin-ref loaders live in this `mod.rs`.

mod cond;
mod generate;
mod instruct;
mod source;
mod streaming;
mod token2wav;

pub use source::{det_source_from_f0, quality_source_from_f0};
pub use streaming::Cv3StreamStats;

use candle_core::{safetensors, DType, Device, Tensor};

use syrinx_acoustic::real_cv3::Cv3Flow;
use syrinx_frontend::speech_token::SpeechTokenizer;
use syrinx_frontend::tokenizer::TextTokenizer;
use syrinx_lm::real_cv3::Cv3Lm;
use syrinx_speaker::real::CamPlus;
use syrinx_vocoder::real_cv3::Cv3Hift;

// Reuse the CV2 synth's error + prompt-conditioning types so the two synthesizers
// share one contract (additive; `crate::synth` is untouched).
use crate::synth::{tn_normalize, PromptCond, SynthError};

/// kaldi fbank params (CAM++ input): 80 mel bins, 16 kHz.
const FBANK_MELS: usize = 80;
const SR_16K: f32 = 16_000.0;

/// matcha prompt-mel params (flow `prompt_feat`): 24 kHz. Same n_fft/hop/win/mels as CV2,
/// but `fmax` differs — see [`MEL_FMAX`].
const MEL_N_FFT: usize = 1920;
const MEL_NUM_MELS: usize = 80;
const MEL_SR: f32 = 24_000.0;
const MEL_HOP: usize = 480;
const MEL_WIN: usize = 1920;
const MEL_FMIN: f32 = 0.0;
/// CV3 mel `fmax`. CosyVoice3's `mel_spec_transform1` sets `fmax: null`, which matcha's
/// `mel_spectrogram` forwards to `librosa_mel_fn` as `None` -> the Nyquist `sr/2 = 12000`.
/// This DIFFERS from CosyVoice2 (which pins `fmax: 8000`, see [`crate::synth`]). Copying
/// CV2's 8000 here reshapes all 80 triangular mel filters across a narrower band and
/// systematically corrupts `prompt_feat` (a ~5.8 max-abs-diff vs the CV3 reference, near
/// input-independent because it is a fixed filterbank error). CV3 must use 12000.
const MEL_FMAX: f32 = 12_000.0;

/// CFM Euler step count (CosyVoice `n_timesteps`).
const N_TIMESTEPS: usize = 10;

/// LM length ratios: `min_len = |tok(tts_text)|*2`, `max_len = *20`.
const MIN_TOKEN_TEXT_RATIO: usize = 2;
const MAX_TOKEN_TEXT_RATIO: usize = 20;

/// f0 -> source upsample factor: HiFT upsample product (8*5*3=120) * istft hop (4) = 480.
const F0_UPSAMPLE: usize = 480;
/// SineGen sine amplitude (`sine_amp` / `nsf_alpha`).
const SINE_AMP: f64 = 0.1;
/// SineGen additive-noise std for *voiced* frames (`noise_std` / `nsf_sigma`).
const NSF_NOISE_STD: f64 = 0.003;
/// Number of harmonic overtones above the fundamental (`nb_harmonics`); the source has
/// `NB_HARMONICS + 1 = 9` sine components (fundamental + 8 overtones).
const NB_HARMONICS: usize = 8;
/// F0 threshold (Hz) for the voiced/unvoiced mask (`nsf_voiced_threshold`).
const NSF_VOICED_THRESHOLD: f64 = 10.0;

/// CV3's `<|endofprompt|>` boundary marker appended to the prompt-text segment.
const ENDOFPROMPT: &str = "<|endofprompt|>";

/// Paths to every CV3 sub-model's on-disk weights / assets.
#[derive(Debug, Clone)]
pub struct Cv3SynthConfig {
    /// CV3 `llm_fp32.safetensors` — the Qwen2-0.5B LM + bias-free `llm_decoder`.
    pub lm_weights: String,
    /// `campplus_weights.safetensors` — the CAM++ speaker encoder (shared with CV2).
    pub spk_weights: String,
    /// CV3 `flow_fp32.safetensors` — the `CausalMaskedDiffWithDiT` mel decoder.
    pub flow_weights: String,
    /// CV3 `hift_fp32.safetensors` — the `CausalHiFTGenerator` vocoder.
    pub hift_weights: String,
    /// `tokenizer.json` — the Qwen2 BPE text tokenizer (shared with CV2).
    pub tokenizer_json: String,
    /// `speech_tokenizer_v3.onnx` — the v3 prompt speech-token tokenizer.
    pub speech_tokenizer_onnx: String,
}

/// Optional injected inputs for a deterministic CV3 run. Any field left `None` is
/// derived live (see the module docs).
#[derive(Default)]
pub struct Cv3SynthInputs {
    /// Pinned generated speech-token ids (i64), bypassing live LM sampling.
    pub pinned_speech_token: Option<Vec<i64>>,
    /// Pinned CFM noise `z` `[1, 80, total]` (the flow's `rand_noise` slice). When
    /// `None`, [`Cv3Synthesizer::synthesize`] fills a **seeded standard-normal** `z`
    /// (the flow's `rand_noise` distribution, seeded from `lm_seed`) — NOT a zeros init.
    /// A zeros init is the degenerate mean-ODE start and measurably degrades live quality;
    /// pin `z` explicitly only for the deterministic parity chain (the reference's noise).
    pub z: Option<Tensor>,
    /// LM sampling seed (live path only). Defaults to 0.
    pub lm_seed: u64,
    /// Optional hard cap on live LM generation steps (the real `max_len` is
    /// `|tok(tts_text)|*20`). `None` uses the real ratio; `min_len` is always honoured.
    pub max_gen_steps: Option<usize>,
}

/// The end-to-end CV3 synthesizer: holds every loaded CV3 sub-model.
pub struct Cv3Synthesizer {
    tokenizer: TextTokenizer,
    speech_tokenizer: SpeechTokenizer,
    speaker: CamPlus,
    lm: Cv3Lm,
    flow: Cv3Flow,
    vocoder: Cv3Hift,
    dev: Device,
}

impl Cv3Synthesizer {
    /// Load every CV3 sub-model from `cfg` onto the CPU (the parity device).
    pub fn load(cfg: &Cv3SynthConfig) -> Result<Self, SynthError> {
        Self::load_on_device(cfg, Device::Cpu)
    }

    /// Load every CV3 sub-model from `cfg` onto an explicit `dev`.
    ///
    /// Every Candle sub-model builds its constants from `dev`, so a single device
    /// threaded here drives the whole pipeline on that backend. The v3 speech-token
    /// ONNX prompt step still runs on its own CPU `ort` runtime. GPU output will not
    /// bit-match the CPU reference (CPU stays the parity device).
    pub fn load_on_device(cfg: &Cv3SynthConfig, dev: Device) -> Result<Self, SynthError> {
        let tokenizer = TextTokenizer::from_file(&cfg.tokenizer_json)?;
        let speech_tokenizer = SpeechTokenizer::load_cv3(&cfg.speech_tokenizer_onnx)?;
        let speaker = CamPlus::load(&cfg.spk_weights, dev.clone())?;
        let lm = Cv3Lm::load(&cfg.lm_weights, dev.clone())?;
        let flow = Cv3Flow::load(&cfg.flow_weights, dev.clone())?;
        let vocoder = Cv3Hift::load(&cfg.hift_weights, dev.clone())?;
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

    /// Load every CV3 sub-model from `cfg` in its **int4-quantized** variant onto the CPU —
    /// the README 4-bit footprint capstone for the end-to-end CV3 pipeline (realized
    /// ≈488 MB; the early ~270 MB budget under-counted the Qwen2-0.5B body).
    pub fn load_quantized(cfg: &Cv3SynthConfig) -> Result<Self, SynthError> {
        Self::load_quantized_on_device(cfg, Device::Cpu)
    }

    /// Load every CV3 sub-model from `cfg` int4-quantized onto an explicit `dev`.
    ///
    /// The quantized analogue of [`Self::load_on_device`]: each Candle sub-model is loaded
    /// through its `load_quantized` (LM int4 `Q4_0` linears + int4 embeds via
    /// [`Cv3Lm::load_quantized`]; flow `Q4_0` DiT linears via [`Cv3Flow::load_quantized`];
    /// HiFT `Q4_0` decode conv kernels via [`Cv3Hift::load_quantized`]; CAM++ `Q4_0` kernels
    /// via [`CamPlus::load_quantized`], shared with CV2). The Qwen BPE text tokenizer and
    /// the v3 speech-token ONNX runtime are unchanged (no weights to quantize). Forward math
    /// is the identical code path on the dequantized weights; int4 trades accuracy for size
    /// (an opt-in **size**, not speed, tradeoff — dequant-on-fetch stalls inference), so this
    /// output tracks but does not equal the fp32 [`Self::load_on_device`] output. The fp32
    /// loaders + the quality/instruct paths are byte-unchanged.
    pub fn load_quantized_on_device(cfg: &Cv3SynthConfig, dev: Device) -> Result<Self, SynthError> {
        let tokenizer = TextTokenizer::from_file(&cfg.tokenizer_json)?;
        let speech_tokenizer = SpeechTokenizer::load_cv3(&cfg.speech_tokenizer_onnx)?;
        let speaker = CamPlus::load_quantized(&cfg.spk_weights, dev.clone())?;
        let lm = Cv3Lm::load_quantized(&cfg.lm_weights, dev.clone())?;
        let flow = Cv3Flow::load_quantized(&cfg.flow_weights, dev.clone())?;
        let vocoder = Cv3Hift::load_quantized(&cfg.hift_weights, dev.clone())?;
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

    /// Load every CV3 sub-model, **overriding only the LM weight path** — the entry point
    /// for the **RL post-trained LM** (`llm.rl_fp32.safetensors`).
    ///
    /// The RL checkpoint is a drop-in replacement for `llm_fp32.safetensors`: identical
    /// architecture and key layout (the Qwen2-0.5B body + the bias-free CV3 `llm_decoder`,
    /// `llm.model.model.layers.N.*` / `speech_embedding.weight` / `llm_decoder.weight`), so
    /// [`Cv3Lm::load`] consumes it with no change. This is exactly equivalent to setting
    /// `cfg.lm_weights = rl_lm_weights` before [`Self::load_on_device`]; it is provided as a
    /// named helper so the RL-vs-base A/B is explicit at the call site. Every other
    /// sub-model (speaker / flow / HiFT / tokenizers) is loaded from `cfg` unchanged.
    pub fn load_with_lm(
        cfg: &Cv3SynthConfig,
        rl_lm_weights: &str,
        dev: Device,
    ) -> Result<Self, SynthError> {
        let mut cfg = cfg.clone();
        cfg.lm_weights = rl_lm_weights.to_string();
        Self::load_on_device(&cfg, dev)
    }

    /// The device every Candle sub-model was loaded onto.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Full CV3 synthesis: `tts_text` spoken in the reference voice, returning the 24 kHz
    /// waveform as a flat `Vec<f32>`. `inputs` may pin the generated tokens and `z`.
    pub fn synthesize(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &Cv3SynthInputs,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;

        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: a pinned z (parity chain), else a SEEDED standard-normal init. The
        // old zeros fallback collapsed the CFM ODE onto its mean trajectory (a smeared,
        // low-detail mel — the live-quality defect); a standard normal is the flow's
        // `rand_noise` *distribution*, reproducible from `lm_seed`. token2wav's own `None`
        // branch keeps an explicit zeros init available for callers that need it.
        let z = match inputs.z.as_ref() {
            Some(z) => z.clone(),
            None => {
                let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
                self.seeded_normal_z(total, inputs.lm_seed)?
            }
        };
        let audio = self.token2wav(&cond, &speech_token, Some(&z))?; // [1, L]
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }
}

/// Minimal seeded **SplitMix64** PRNG — the quality source / CFM-noise randomness, so the
/// random-phase source and seeded `z` are reproducible from a seed (never system RNG). A
/// local copy (the `lm` crate's PRNG is private + has no Gaussian); keeps `crate::synth`
/// byte-unchanged. Drives both the per-harmonic initial phases (uniform) and the additive
/// Gaussian noise (Box–Muller).
struct SplitMix64 {
    state: u64,
    /// Cached second Box–Muller normal (generated in pairs).
    spare: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    /// Next raw 64-bit value (the canonical SplitMix64 mix).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)` (top 53 bits, the standard construction).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal `N(0, 1)` via Box–Muller, generating two at a time and caching the
    /// spare (matches `torch.randn`'s distribution, not its byte stream).
    fn next_gauss(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        // Guard u1 away from 0 so ln is finite.
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

// ---- free helpers (local copies; `crate::synth` stays byte-unchanged) -------------
//
// The shared token helpers live here; the fbank/grid helpers used only by the frontend
// conditioning live in [`cond`].

/// i64 ids -> `[1, n]` i64 tensor.
fn ids_i64_to_tensor(ids: &[i64], dev: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(ids.to_vec(), (1, ids.len()), dev)
}

/// `[1, n]` (or `[n]`) i64 token tensor -> `Vec<u32>`.
fn tensor_ids_u32(t: &Tensor) -> candle_core::Result<Vec<u32>> {
    let flat = t.flatten_all()?.to_dtype(DType::U32)?;
    flat.to_vec1::<u32>()
}

/// Load a flat `Vec<i64>` from a safetensors `key` (any int dtype is coerced to i64) —
/// the loader behind the `evaluate_cv3` pin-ref diagnostic (e.g. the e2e reference
/// `speech_token`). Kept here (not in `syrinx-eval`) so that crate stays Candle-free.
pub fn load_ref_i64(path: &str, key: &str) -> Result<Vec<i64>, SynthError> {
    let map = safetensors::load(path, &Device::Cpu)?;
    let t = map
        .get(key)
        .ok_or_else(|| SynthError::Candle(format!("safetensors `{path}` has no tensor `{key}`")))?;
    let flat = t.flatten_all()?.to_dtype(DType::I64)?;
    Ok(flat.to_vec1::<i64>()?)
}

/// Load an f32 `Tensor` (its on-disk shape preserved) from a safetensors `key` onto `dev`
/// — the loader behind the optional pin-`z` arm of the `evaluate_cv3` pin-ref diagnostic
/// (e.g. the flow reference `z` `[1, 80, total]`).
pub fn load_ref_tensor(path: &str, key: &str, dev: &Device) -> Result<Tensor, SynthError> {
    let map = safetensors::load(path, dev)?;
    let t = map
        .get(key)
        .ok_or_else(|| SynthError::Candle(format!("safetensors `{path}` has no tensor `{key}`")))?;
    Ok(t.to_dtype(DType::F32)?)
}
