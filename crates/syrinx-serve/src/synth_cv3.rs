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
//! [`Cv3Synthesizer::synthesize`] falls back to `z = zeros` (a valid ODE init) and a
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

use candle_core::{DType, Device, Tensor};

use syrinx_acoustic::real_cv3::Cv3Flow;
use syrinx_frontend::feat::{kaldi_fbank, prompt_mel};
use syrinx_frontend::speech_token::SpeechTokenizer;
use syrinx_frontend::tokenizer::TextTokenizer;
use syrinx_lm::real_cv3::Cv3Lm;
use syrinx_speaker::real::CamPlus;
use syrinx_vocoder::real_cv3::Cv3Hift;

// Reuse the CV2 synth's error + prompt-conditioning types so the two synthesizers
// share one contract (additive; `crate::synth` is untouched).
use crate::synth::{PromptCond, SynthError};

/// kaldi fbank params (CAM++ input): 80 mel bins, 16 kHz.
const FBANK_MELS: usize = 80;
const SR_16K: f32 = 16_000.0;

/// matcha prompt-mel params (flow `prompt_feat`): 24 kHz (identical to CV2).
const MEL_N_FFT: usize = 1920;
const MEL_NUM_MELS: usize = 80;
const MEL_SR: f32 = 24_000.0;
const MEL_HOP: usize = 480;
const MEL_WIN: usize = 1920;
const MEL_FMIN: f32 = 0.0;
const MEL_FMAX: f32 = 8000.0;

/// CFM Euler step count (CosyVoice `n_timesteps`).
const N_TIMESTEPS: usize = 10;

/// LM length ratios: `min_len = |tok(tts_text)|*2`, `max_len = *20`.
const MIN_TOKEN_TEXT_RATIO: usize = 2;
const MAX_TOKEN_TEXT_RATIO: usize = 20;

/// f0 -> source upsample factor: HiFT upsample product (8*5*3=120) * istft hop (4) = 480.
const F0_UPSAMPLE: usize = 480;
/// SineGen sine amplitude (`sine_amp` / `nsf_alpha`).
const SINE_AMP: f64 = 0.1;

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
    /// `None`, a zeros init is used (a valid ODE start, not the reference's noise).
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

    /// The device every Candle sub-model was loaded onto.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Run the CV3 frontend half: tokenize text (with the `<|endofprompt|>` boundary on
    /// the prompt-text segment) and derive the prompt-side conditioning (`embedding`,
    /// `prompt_token`, `prompt_feat`) from the reference waveforms, applying the CosyVoice
    /// `%2` token/feat alignment.
    ///
    /// `ref_wav_16k` is the 16 kHz mono reference (fbank + speech-token input);
    /// `ref_wav_24k` is the same clip at 24 kHz (prompt-mel input). Resampling is the
    /// caller's job (only the `feat` math is under parity test), so a deterministic test
    /// can feed the exact reference-resampled waveforms.
    pub fn prompt_cond(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<PromptCond, SynthError> {
        // --- text tokens: prompt_text(+<|endofprompt|>) ++ tts_text. ---
        // CV3 appends the endofprompt boundary marker to the prompt-text segment; the
        // tokenizer recognises it as one atomic special id. Idempotent if already present.
        let prompt_text = if prompt_text.contains(ENDOFPROMPT) {
            std::borrow::Cow::Borrowed(prompt_text)
        } else {
            std::borrow::Cow::Owned(format!("{prompt_text}{ENDOFPROMPT}"))
        };
        let prompt_text_ids = self.tokenizer.encode(prompt_text.as_ref())?;
        let tts_text_ids = self.tokenizer.encode(tts_text)?;
        let prompt_text_len = prompt_text_ids.len();
        let mut text_token = prompt_text_ids;
        text_token.extend_from_slice(&tts_text_ids);

        // --- speaker x-vector: kaldi fbank -> per-time mean subtraction -> CAM++. ---
        let fbank_grid = kaldi_fbank(ref_wav_16k, SR_16K, FBANK_MELS); // [T][80]
        let fbank = grid_to_tensor(&fbank_grid, &self.dev)?; // [T, 80]
        let fbank = subtract_time_mean(&fbank)?; // feat - feat.mean(dim=0)
        let fbank = fbank.unsqueeze(0)?; // [1, T, 80]
        let spk_embedding = self.speaker.forward(&fbank)?; // [1, 192]

        // --- prompt speech tokens via the v3 ONNX tokenizer (16 kHz). ---
        let prompt_token_i32 = self.speech_tokenizer.tokens_from_wav(ref_wav_16k)?;

        // --- prompt mel (24 kHz): feat returns [80, T'] mel-major; flow wants
        //     [1, T', 80] frame-major. ---
        let mel_grid = prompt_mel(
            ref_wav_24k,
            MEL_N_FFT,
            MEL_NUM_MELS,
            MEL_SR,
            MEL_HOP,
            MEL_WIN,
            MEL_FMIN,
            MEL_FMAX,
        ); // [80][T']
        let prompt_feat = mel_major_to_frame_major(&mel_grid, &self.dev)?; // [1, T', 80]

        // --- %2 alignment: token_len = min(T'/2, |prompt_token|); truncate. ---
        let n_feat_frames = prompt_feat.dim(1)?;
        let token_len = (n_feat_frames / 2).min(prompt_token_i32.len());
        let prompt_feat = prompt_feat.narrow(1, 0, 2 * token_len)?.contiguous()?;
        let prompt_token = i32_ids_to_tensor(&prompt_token_i32[..token_len], &self.dev)?; // [1, token_len]

        Ok(PromptCond {
            text_token,
            prompt_text_len,
            spk_embedding,
            prompt_token,
            prompt_feat,
        })
    }

    /// CAM++ speaker x-vector `[1, 192]` for a 16 kHz mono waveform (kaldi-fbank ->
    /// per-time mean-subtraction -> CAM++). Exposed for evaluation (e.g. SIM-o).
    pub fn speaker_embedding(&self, audio_16k: &[f32]) -> Result<Tensor, SynthError> {
        let fbank_grid = kaldi_fbank(audio_16k, SR_16K, FBANK_MELS);
        let fbank = grid_to_tensor(&fbank_grid, &self.dev)?;
        let fbank = subtract_time_mean(&fbank)?;
        let fbank = fbank.unsqueeze(0)?;
        Ok(self.speaker.forward(&fbank)?)
    }

    /// Generate the CV3 speech-token sequence with the **live** LM (pinned-PRNG
    /// sampling). Returns the ids as an i64 `[1, N]` tensor ready for the flow.
    pub fn generate_speech_token(
        &self,
        cond: &PromptCond,
        seed: u64,
        max_gen_steps: Option<usize>,
    ) -> Result<Tensor, SynthError> {
        let text_len = cond.text_token.len();
        let net = text_len.saturating_sub(cond.prompt_text_len);
        let min_len = net * MIN_TOKEN_TEXT_RATIO;
        let real_max = net * MAX_TOKEN_TEXT_RATIO;
        let max_len = match max_gen_steps {
            Some(cap) => cap.max(min_len + 1),
            None => real_max,
        };
        let prompt_speech_u32 = tensor_ids_u32(&cond.prompt_token)?;
        let gen = self
            .lm
            .generate(&cond.text_token, &prompt_speech_u32, min_len, max_len, seed)?;
        let ids: Vec<i64> = gen.iter().map(|&t| t as i64).collect();
        ids_i64_to_tensor(&ids, &self.dev).map_err(Into::into)
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

        let audio = self.token2wav(&cond, &speech_token, inputs.z.as_ref())?; // [1, L]
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }

    /// The flow + vocoder half: speech tokens -> mel -> audio `[1, L]`. `z` is the CFM
    /// noise (pinned, else a zeros init); the HiFT source is the deterministic F0 source.
    pub fn token2wav(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        z: Option<&Tensor>,
    ) -> Result<Tensor, SynthError> {
        // total flow length = 2 * (|prompt_token| + |speech_token|).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };
        // Cv3Flow.forward returns the generated mel (prompt-mel prefix already dropped).
        let mel = self.flow.forward(
            &cond.prompt_token,
            speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            N_TIMESTEPS,
        )?; // [1, 80, 2*Tg]
        let source = self.deterministic_source(&mel)?; // [1, 1, L_src]
        let audio = self.vocoder.decode(&mel, &source)?; // [1, L]
        Ok(audio)
    }

    /// Run only the CV3 flow CFM mel decoder over **explicit reference** prompt/speech
    /// tokens + conditioning — the parity-test entry point.
    ///
    /// Feeds `prompt_token`, `token`, `prompt_feat`, `embedding`, `z` straight to
    /// [`Cv3Flow::forward`] and returns the generated mel `[1, 80, 2*Tg]` (the prompt-mel
    /// prefix is dropped inside `forward`). Used by the deterministic chain anchor: feed
    /// the fixture's reference `speech_token`/`prompt_feat`/`embedding` and compare the
    /// mel to the reference `mel`, verifying frontend->flow end to end without invoking
    /// the non-bit-reproducible LM sampler.
    pub fn flow_from_reference_tokens(
        &self,
        prompt_token: &Tensor,
        token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor, SynthError> {
        let mel = self
            .flow
            .forward(prompt_token, token, prompt_feat, embedding, z, n_timesteps)?;
        Ok(mel)
    }

    /// Build a **deterministic, zero-phase, single-harmonic** HiFT source **waveform**
    /// `[1, 1, L]` from a generated mel, for the functional (non-parity) path.
    ///
    /// Mirrors the deterministic core of CV3's `m_source` excitation: the CV3 HiFT F0
    /// predictor -> nearest-upsample by 480 -> single sine `sine_amp * sin(2π·cumsum(f0/sr))`.
    /// `Cv3Hift::decode` STFTs this waveform internally. The model's `SourceModuleHnNSF`
    /// SineGen additionally draws a random initial phase + Gaussian noise and applies a
    /// learned `l_linear(9->1)+tanh` harmonic merge (none bit-portable across RNGs), so
    /// this is a faithful smoke source, not a parity source.
    pub fn deterministic_source(&self, mel: &Tensor) -> Result<Tensor, SynthError> {
        let f0 = self.vocoder.f0_predict(mel)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        // Nearest-upsample f0 by 480, then a zero-phase instantaneous-phase ramp.
        let n = f0v.len() * F0_UPSAMPLE;
        let mut source: Vec<f32> = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        let two_pi = 2.0 * std::f64::consts::PI;
        for &v in &f0v {
            let fhz = v as f64;
            for _ in 0..F0_UPSAMPLE {
                acc += fhz / MEL_SR as f64;
                source.push((SINE_AMP * (two_pi * acc).sin()) as f32);
            }
        }
        Ok(Tensor::from_vec(source, (1, 1, n), &self.dev)?)
    }
}

// ---- free helpers (local copies; `crate::synth` stays byte-unchanged) -------------

/// `[T][D]` row-major grid -> `[T, D]` f32 tensor.
fn grid_to_tensor(grid: &[Vec<f32>], dev: &Device) -> candle_core::Result<Tensor> {
    let t = grid.len();
    let d = if t == 0 { 0 } else { grid[0].len() };
    let mut flat = Vec::with_capacity(t * d);
    for row in grid {
        flat.extend_from_slice(row);
    }
    Tensor::from_vec(flat, (t, d), dev)
}

/// Subtract the per-column (over time/rows) mean: `x - x.mean(dim=0, keepdim=True)`.
fn subtract_time_mean(x: &Tensor) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(0)?; // [1, D]
    x.broadcast_sub(&mean)
}

/// `[80][T']` mel-major grid -> `[1, T', 80]` frame-major f32 tensor.
fn mel_major_to_frame_major(grid: &[Vec<f32>], dev: &Device) -> candle_core::Result<Tensor> {
    let n_mels = grid.len();
    let t = if n_mels == 0 { 0 } else { grid[0].len() };
    let mut flat = vec![0f32; t * n_mels];
    for (m, row) in grid.iter().enumerate() {
        for (frame, &v) in row.iter().enumerate() {
            flat[frame * n_mels + m] = v;
        }
    }
    let tensor = Tensor::from_vec(flat, (t, n_mels), dev)?;
    tensor.unsqueeze(0)
}

/// i32 prompt-token ids -> `[1, n]` i64 tensor.
fn i32_ids_to_tensor(ids: &[i32], dev: &Device) -> candle_core::Result<Tensor> {
    let v: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
    ids_i64_to_tensor(&v, dev)
}

/// i64 ids -> `[1, n]` i64 tensor.
fn ids_i64_to_tensor(ids: &[i64], dev: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(ids.to_vec(), (1, ids.len()), dev)
}

/// `[1, n]` (or `[n]`) i64 token tensor -> `Vec<u32>`.
fn tensor_ids_u32(t: &Tensor) -> candle_core::Result<Vec<u32>> {
    let flat = t.flatten_all()?.to_dtype(DType::U32)?;
    flat.to_vec1::<u32>()
}
