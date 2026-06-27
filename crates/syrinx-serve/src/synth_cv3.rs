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

use candle_core::{safetensors, DType, Device, Tensor};

use syrinx_acoustic::real_cv3::{Cv3Flow, Cv3StreamCfg};
use syrinx_frontend::feat::{kaldi_fbank, prompt_mel};
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
        // Text-normalization (`tn` feature, matching CV2 — identity when off). The tts
        // text is always normalized; a plain prompt transcript is normalized then gets the
        // endofprompt boundary marker; a pre-formatted instruct prompt (already carrying the
        // marker) is left byte-untouched so tn can't mangle the marker/instruction.
        let tts_text = tn_normalize(tts_text);
        let tts_text = tts_text.as_ref();
        let prompt_text = if prompt_text.contains(ENDOFPROMPT) {
            std::borrow::Cow::Borrowed(prompt_text)
        } else {
            std::borrow::Cow::Owned(format!("{}{ENDOFPROMPT}", tn_normalize(prompt_text)))
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

    /// The flow + vocoder half: speech tokens -> mel -> audio `[1, L]`. `z` is the CFM
    /// noise; the HiFT source is the deterministic F0 source. A `Some(z)` is used as-is;
    /// a `None` falls back to an **explicit zeros** init — the degenerate mean-ODE start,
    /// retained only for callers that deliberately want it (parity). The live
    /// [`Cv3Synthesizer::synthesize`] no longer passes `None`: it fills a seeded
    /// standard-normal `z` first (see [`Cv3Synthesizer::seeded_normal_z`]).
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
        let mel = self.flow_forward(cond, speech_token, &z, N_TIMESTEPS)?; // [1,80,2*Tg]
        let source = self.deterministic_source(&mel)?; // [1, 1, L_src]
        self.vocode(&mel, &source) // [1, L]
    }

    /// Build the default CFM noise `z` `[1, 80, total]`: a **seeded standard-normal** init
    /// (the flow's `rand_noise` *distribution*, matched in distribution to `torch.randn`,
    /// not its byte stream). This is the live default `synthesize` / `synthesize_quality` /
    /// `synthesize_instruct` now use in place of the degenerate zeros init: a zeros `z`
    /// starts the CFM ODE at the conditional mean and integrates to a low-detail, smeared
    /// mel (the live-quality defect), whereas a true Gaussian draw is the distribution the
    /// flow was trained to denoise. Seeded (never system RNG), so a given `(tokens, seed)`
    /// is bit-reproducible. The byte-exact reference `z` is still injectable via
    /// [`Cv3SynthInputs::z`] for the deterministic parity chain.
    fn seeded_normal_z(&self, total: usize, seed: u64) -> Result<Tensor, SynthError> {
        let nz = MEL_NUM_MELS * total;
        let mut rng = SplitMix64::new(seed ^ 0x5DEE_CE66_C0DE_F10D);
        let zv: Vec<f32> = (0..nz).map(|_| rng.next_gauss() as f32).collect();
        Ok(Tensor::from_vec(zv, (1, MEL_NUM_MELS, total), &self.dev)?)
    }

    /// Run only the CV3 flow CFM mel decoder for live conditioning: feeds `cond`'s
    /// prompt token/feat/embedding plus the generated `speech_token` + noise `z` to
    /// [`Cv3Flow::forward`], returning the generated mel `[1, 80, 2*Tg]` (prompt-mel
    /// prefix dropped). Exposed so callers can inspect the intermediate mel (e.g. the
    /// smoke test prints its shape before vocoding).
    pub fn flow_forward(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor, SynthError> {
        let mel = self.flow.forward(
            &cond.prompt_token,
            speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            z,
            n_timesteps,
        )?;
        Ok(mel)
    }

    /// Vocode a generated mel `[1, 80, T]` with a source waveform `[1, 1, L]` into 24 kHz
    /// audio `[1, L]` via [`Cv3Hift::decode`]. The decode STFTs the source internally.
    pub fn vocode(&self, mel: &Tensor, source: &Tensor) -> Result<Tensor, SynthError> {
        Ok(self.vocoder.decode(mel, source)?)
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
        let source = det_source_from_f0(&f0v);
        let n = source.len();
        Ok(Tensor::from_vec(source, (1, 1, n), &self.dev)?)
    }

    /// Build the **random-phase NSF** HiFT source **waveform** `[1, 1, L]` from a generated
    /// mel — the faithful CV3 `SourceModuleHnNSF` excitation for the perceptual-quality
    /// path, the exact analogue of CV2's `Synthesizer::random_phase_source_stft` but
    /// emitting the *waveform* (CV3's `Cv3Hift::decode` STFTs it internally), not a
    /// pre-STFT'd `s_stft`.
    ///
    /// CV3's HiFT (`CausalHiFTGenerator`, 24 kHz) instantiates `SourceModuleHnNSF` with
    /// `SineGen2(causal=True)` (`sinegen_type='2'` when `sampling_rate != 22050`). Its
    /// steady-state output equals the per-sample cumulative-phase ramp used here; every
    /// perceptually load-bearing element is reproduced (and seeded for reproducibility):
    ///   * **`NB_HARMONICS + 1 = 9` sines** — fundamental + 8 overtones, the `i`-th at
    ///     instantaneous phase `(i+1)·θ(t)` with `θ(t) = 2π·cumsum(f0_up/sr)`;
    ///   * **random initial phase per harmonic** — `φ_0 = 0` (fundamental), `φ_i ∈ (-π, π]`
    ///     (CV3 `SineGen2`'s `rand_ini` with `rand_ini[:,0]=0`);
    ///   * **voiced/unvoiced mask** `uv = f0 > 10 Hz` (gates the sines to 0 when unvoiced);
    ///   * **additive Gaussian noise** `noise_amp = uv·noise_std + (1-uv)·sine_amp/3`
    ///     (`σ=0.003` breath when voiced, `sine_amp/3 ≈ 0.033` broadband when unvoiced);
    ///   * **learned merge** — the 9 components fused by the checkpoint's
    ///     `m_source.l_linear` `Linear(9→1)` + `tanh` ([`Cv3Hift::source_merge_linear`]).
    ///
    /// All randomness comes from a **seeded** SplitMix64 (never system RNG), so a given
    /// `(mel, seed)` is bit-reproducible. This is a perceptual-quality source, not a
    /// bit-parity source — the model's torch RNG draw order is not portable. The smoke
    /// [`Cv3Synthesizer::deterministic_source`] is left byte-unchanged.
    pub fn quality_source(&self, mel: &Tensor, seed: u64) -> Result<Tensor, SynthError> {
        let f0 = self.vocoder.f0_predict(mel)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        let (merge_w, merge_b) = self.vocoder.source_merge_linear()?; // ([9], b)
        if merge_w.len() != NB_HARMONICS + 1 {
            return Err(SynthError::Candle(format!(
                "m_source.l_linear expects {} weights, got {}",
                NB_HARMONICS + 1,
                merge_w.len()
            )));
        }
        let source = quality_source_from_f0(&f0v, &merge_w, merge_b, seed);
        let n = source.len();
        Ok(Tensor::from_vec(source, (1, 1, n), &self.dev)?)
    }

    /// Full CV3 synthesis with the **random-phase NSF source** (perceptual-quality path),
    /// returning the 24 kHz waveform. Same pipeline as [`Cv3Synthesizer::synthesize`] but
    /// the HiFT excitation is [`Cv3Synthesizer::quality_source`] (seeded by `source_seed`)
    /// instead of the buzzy single-harmonic [`Cv3Synthesizer::deterministic_source`], and
    /// the CFM noise `z` is a **seeded standard-normal** init (the flow's `rand_noise`
    /// *distribution*) rather than the degenerate zeros init — both reproducible from
    /// `source_seed`, which also seeds live LM sampling so a `(text, ref, source_seed)`
    /// triple is fully reproducible.
    pub fn synthesize_quality(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        source_seed: u64,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = self.generate_speech_token(&cond, source_seed, None)?;

        // CFM noise z: a SEEDED standard-normal init (torch.randn's distribution, not its
        // byte stream). Now the same default `synthesize` uses (see `seeded_normal_z`).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = self.seeded_normal_z(total, source_seed)?;

        let mel = self.flow_forward(&cond, &speech_token, &z, N_TIMESTEPS)?; // [1,80,2*Tg]
        let source = self.quality_source(&mel, source_seed)?; // [1,1,L]
        let audio = self.vocode(&mel, &source)?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }

    /// **Instruct / emotion** synthesis: speak `tts_text` in the reference voice while
    /// following the natural-language `instruct_text` (e.g. "用开心的语气说", "speak in a
    /// sad tone"), returning the 24 kHz waveform. This is CosyVoice3's `inference_instruct2`
    /// on the **same** CV3 weights — not a separate model.
    ///
    /// It differs from [`Cv3Synthesizer::synthesize`] exactly as CV3's `frontend_instruct2`
    /// does (`cosyvoice/cli/frontend.py:209`):
    ///   1. the instruct text is fed in the **`prompt_text` role**, so the LM text prompt is
    ///      `tok(instruct_text ++ <|endofprompt|>) ++ tok(tts_text)`; and
    ///   2. the LM is driven with an **empty prompt speech-token prefix** (`frontend_instruct2`
    ///      deletes `llm_prompt_speech_token`) — content + requested style come from text
    ///      alone — assembled by [`Cv3Lm::build_lm_input_instruct`] (the CV3 structural
    ///      instruct slot `[sos, instruct, text, task_id]`).
    ///
    /// The flow + vocoder run **identically** to `synthesize` (reference `prompt_token`,
    /// `prompt_feat`, speaker embedding kept), so the cloned voice is preserved and only the
    /// prosody/emotion follows the instruction. CV3 requires a trailing `<|endofprompt|>` on
    /// the instruct text (asserted by `Qwen2LM.inference`); it is appended if absent (the
    /// append is idempotent). The LM sampling seed is fixed (0) for reproducibility.
    pub fn synthesize_instruct(
        &mut self,
        tts_text: &str,
        instruct_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<Vec<f32>, SynthError> {
        // (1) instruct text takes the prompt_text role; append CV3's <|endofprompt|> marker
        //     if the caller did not. `prompt_cond` then builds the ref prompt conditioning
        //     and text_token = tok(instruct ++ <|endofprompt|>) ++ tok(tts).
        let instruct = if instruct_text.contains(ENDOFPROMPT) {
            std::borrow::Cow::Borrowed(instruct_text)
        } else {
            std::borrow::Cow::Owned(format!("{instruct_text}{ENDOFPROMPT}"))
        };
        let cond = self.prompt_cond(tts_text, instruct.as_ref(), ref_wav_16k, ref_wav_24k)?;

        // (2) LM with an EMPTY prompt speech-token prefix via the structural instruct input.
        //     text_token = instruct(prefix) ++ tts(suffix); split at prompt_text_len.
        let instruct_ids = &cond.text_token[..cond.prompt_text_len];
        let tts_ids = &cond.text_token[cond.prompt_text_len..];
        let net = tts_ids.len();
        let min_len = net * MIN_TOKEN_TEXT_RATIO;
        let max_len = net * MAX_TOKEN_TEXT_RATIO;
        let gen = self
            .lm
            .generate_instruct(instruct_ids, tts_ids, min_len, max_len, 0)?;
        let ids: Vec<i64> = gen.iter().map(|&t| t as i64).collect();
        let speech_token = ids_i64_to_tensor(&ids, &self.dev)?;

        // flow + vocoder exactly as `synthesize` (ref prompt_token/feat/spk kept; det source).
        // Use the same seeded standard-normal CFM init as `synthesize` (seed 0), not zeros.
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = self.seeded_normal_z(total, 0)?;
        let audio = self.token2wav(&cond, &speech_token, Some(&z))?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }
}

// ============================================================================
// Chunked-causal STREAMING synthesis (additive — low time-to-first-byte path).
//
// Mirrors CV2's `crate::synth::Synthesizer::synthesize_streaming`, swapping the CV3
// parts in: the causal DiT flow chunk-mask (`Cv3Flow::forward_zero_shot_streaming`)
// replaces the non-causal batch flow, and the CV3 causal HiFT (`Cv3Hift::decode`, which
// STFTs the source waveform internally) replaces CV2's `s_stft`-taking HiFT. The HiFT
// assembly (mel overlap cache + phase-continuous source + hamming boundary cross-fade)
// is the CV2 `token2wav_streaming` recipe. The batch `synthesize` path is untouched.
//
// ★ Correctness note: the actual fix that makes streaming faithful is the causal DiT
//   mask — a finalized mel frame never attends to the future, so it is bit-stable as the
//   token prefix grows (proven by `tests/real_cv3_flow_stream_consistency.rs`). The
//   cross-faded *audio* is, by CV2/CV3 design, NOT sample-identical to the batch render
//   (the boundary fade changes a chunk of samples); intelligibility (Whisper CER on the
//   streamed audio) is the human/on-box quality signal.
// ============================================================================

/// Speech-token right-lookahead the CV3 flow front-end needs (the `pre_lookahead_len` of
/// `CausalMaskedDiffWithDiT`): a chunk finalizing tokens `[off .. off+hop)` needs tokens up
/// to `off+hop+PRE_LOOKAHEAD` present (mirrors CV3 `model.py`'s
/// `token_offset + this_token_hop_len + self.flow.pre_lookahead_len`).
const STREAM_PRE_LOOKAHEAD: usize = 3;
/// token -> mel ratio (CV3 `token_mel_ratio`).
const STREAM_TOKEN_MEL_RATIO: usize = 2;
/// Mel-frame overlap carried across chunks for the HiFT boundary (CV2's `mel_cache_len`).
const STREAM_MEL_CACHE: usize = 20;
/// Trailing source/speech samples held back + cross-faded at each boundary
/// (`mel_cache * f0_upsample`; CV2's `source_cache_len`).
const STREAM_SOURCE_CACHE: usize = STREAM_MEL_CACHE * F0_UPSAMPLE; // 9600

/// Timing + shape summary of a streaming run, so callers can report **time-to-first-byte**
/// (the headline latency win) and total throughput.
#[derive(Debug, Clone, Copy)]
pub struct Cv3StreamStats {
    /// Wall-clock from the call to the first emitted chunk — the TTFB.
    pub ttfb: std::time::Duration,
    /// Wall-clock for the whole streamed utterance.
    pub total: std::time::Duration,
    /// Number of audio chunks emitted.
    pub n_chunks: usize,
    /// Total 24 kHz samples emitted across all chunks.
    pub n_samples: usize,
}

impl Cv3Synthesizer {
    /// **Streaming** CV3 synthesis: same `tts_text`-in-reference-voice flow + vocoder as
    /// [`Cv3Synthesizer::synthesize`], but audio is emitted **incrementally** chunk by
    /// chunk (low time-to-first-byte) via `on_chunk`, instead of one final `Vec`.
    ///
    /// `chunk_size` is the number of *finalized* speech tokens released per chunk; a chunk
    /// for tokens `[off .. off+chunk_size)` only needs tokens up to
    /// `off+chunk_size+PRE_LOOKAHEAD` present (the flow's right lookahead), so the first
    /// chunk lands after `chunk_size + PRE_LOOKAHEAD` tokens instead of the whole
    /// utterance. Each emitted chunk is a flat `Vec<f32>` of 24 kHz samples in order;
    /// concatenating them yields the full streamed waveform.
    ///
    /// The flow runs under the chunk-causal DiT mask
    /// ([`Cv3Flow::forward_zero_shot_streaming`], [`Cv3StreamCfg::cosyvoice3`]) so finalized
    /// frames are stable as the prefix grows. The HiFT runs per chunk over a mel-overlap
    /// cache with a **phase-continuous** source (the F0 excitation phase is carried across
    /// chunks, and each chunk's overlap source is overwritten by the previous chunk's
    /// trailing source) plus a hamming cross-fade at every boundary — the CV2 streaming
    /// recipe. `inputs.z`, the LM seed/cap, and `inputs.pinned_speech_token` are honoured
    /// exactly as in `synthesize`; a fresh seeded standard-normal `z` is used when `z` is
    /// not pinned. Returns [`Cv3StreamStats`] (TTFB + totals).
    pub fn synthesize_streaming(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &Cv3SynthInputs,
        chunk_size: usize,
        mut on_chunk: impl FnMut(Vec<f32>) -> Result<(), SynthError>,
    ) -> Result<Cv3StreamStats, SynthError> {
        assert!(chunk_size >= 1, "chunk_size must be >= 1");
        let start = std::time::Instant::now();
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;

        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // Full-length CFM noise z (covers the final flow length 2*(|prompt|+|token|)):
        // pinned (parity) else a seeded standard-normal (same default as `synthesize`).
        let total_flow = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z_full = match inputs.z.as_ref() {
            Some(z) => z.clone(),
            None => self.seeded_normal_z(total_flow, inputs.lm_seed)?,
        };

        let n_tokens = speech_token.dim(1)?;
        let prompt_len = cond.prompt_token.dim(1)?;
        let cfg = Cv3StreamCfg::cosyvoice3();

        // HiFT carry-over caches (mirrors CosyVoice2Model.hift_cache_dict).
        let mut cache_mel: Option<Tensor> = None; // last STREAM_MEL_CACHE mel frames
        let mut cache_speech_tail: Option<Tensor> = None; // held-back fade-out leg [1, S]
        let mut cache_source_tail: Option<Tensor> = None; // trailing source samples [1, S]
        let mut phase = 0f64; // global F0 excitation phase (cycles), carried across chunks

        let mut ttfb: Option<std::time::Duration> = None;
        let mut n_chunks = 0usize;
        let mut n_samples = 0usize;

        let mut offset = 0usize;
        while offset < n_tokens {
            let want_end = (offset + chunk_size).min(n_tokens);
            let finalize = want_end == n_tokens;
            // Tokens that must be present to finalize `want_end`: finalized region plus
            // PRE_LOOKAHEAD of right context (clamped to the utterance end).
            let avail_end = (want_end + STREAM_PRE_LOOKAHEAD).min(n_tokens);
            let tok_slice = speech_token.narrow(1, 0, avail_end)?.contiguous()?;

            // z slice for this (prompt + avail) flow length.
            let flow_len = STREAM_TOKEN_MEL_RATIO * (prompt_len + avail_end);
            let z = z_full.narrow(2, 0, flow_len)?.contiguous()?;

            // Chunked-causal streaming flow over the grown prefix: the DiT mask keeps the
            // [2*offset .. 2*want_end] region bit-stable as the prefix grows.
            let mel_full = self.flow.forward_zero_shot_streaming(
                &cond.prompt_token,
                &tok_slice,
                &cond.prompt_feat,
                &cond.spk_embedding,
                &z,
                N_TIMESTEPS,
                cfg,
            )?; // [1, 80, 2*avail_end]

            // Newly-finalized mel region: [2*offset .. 2*want_end].
            let mel_start = STREAM_TOKEN_MEL_RATIO * offset;
            let mel_count = STREAM_TOKEN_MEL_RATIO * (want_end - offset);
            let mel_new = mel_full.narrow(2, mel_start, mel_count)?.contiguous()?;

            // --- HiFT chunk: mel overlap + phase-continuous source + cross-fade. ---
            let overlap_frames = match &cache_mel {
                Some(prev) => prev.dim(2)?,
                None => 0,
            };
            let mel_in = match &cache_mel {
                Some(prev) => Tensor::cat(&[prev, &mel_new], 2)?,
                None => mel_new.clone(),
            };
            let overlap_samples = overlap_frames * F0_UPSAMPLE;
            let (src_3d, source_flat, phase_out) =
                self.streaming_source(&mel_in, phase, overlap_samples, cache_source_tail.as_ref())?;
            phase = phase_out;

            let speech = self.vocode(&mel_in, &src_3d)?; // [1, L]
            let total_len = speech.dim(1)?;

            // Hold back the trailing SOURCE_CACHE samples (unless final) and trim them now.
            let (mut emit, new_tail) = if !finalize && total_len > STREAM_SOURCE_CACHE {
                let keep = total_len - STREAM_SOURCE_CACHE;
                let head = speech.narrow(1, 0, keep)?.contiguous()?;
                let tail = speech.narrow(1, keep, STREAM_SOURCE_CACHE)?.contiguous()?;
                (head, Some(tail))
            } else {
                (speech, None)
            };

            // Cross-fade the leading samples of this emit with the previous chunk's tail.
            if let Some(prev_tail) = &cache_speech_tail {
                emit = hamming_crossfade(&emit, prev_tail, &self.dev)?;
            }

            let wav: Vec<f32> = emit.flatten_all()?.to_vec1::<f32>()?;
            n_samples += wav.len();
            n_chunks += 1;
            if ttfb.is_none() {
                ttfb = Some(start.elapsed());
            }
            on_chunk(wav)?;

            // Update caches for the next chunk.
            let mlen = mel_in.dim(2)?;
            let keep_mel = STREAM_MEL_CACHE.min(mlen);
            cache_mel = Some(mel_in.narrow(2, mlen - keep_mel, keep_mel)?.contiguous()?);
            cache_speech_tail = new_tail;
            let swlen = source_flat.dim(1)?;
            let keep_src = STREAM_SOURCE_CACHE.min(swlen);
            cache_source_tail =
                Some(source_flat.narrow(1, swlen - keep_src, keep_src)?.contiguous()?);

            offset = want_end;
        }

        Ok(Cv3StreamStats {
            ttfb: ttfb.unwrap_or_default(),
            total: start.elapsed(),
            n_chunks,
            n_samples,
        })
    }

    /// Build the **streaming** CV3 HiFT source for one mel chunk with *global F0-phase
    /// continuity* (the CV3 analogue of CV2's `Synthesizer::streaming_source_phase`, but
    /// emitting the source **waveform** `[1,1,L]` that `Cv3Hift::decode` STFTs internally,
    /// not a pre-STFT'd `s_stft`).
    ///
    /// Same deterministic F0 -> single-harmonic sine core as
    /// [`Cv3Synthesizer::deterministic_source`], but: (1) the instantaneous phase continues
    /// from `phase_in` (one continuous sinusoid across chunks, not reset per chunk), and
    /// (2) the leading `overlap_samples` are overwritten by `prev_tail` (the previous
    /// chunk's trailing source), so the boundary cross-fade stays phase-coherent
    /// (CosyVoice2 `cache_source`). Returns `(source [1,1,n], source_flat [1,n], phase_out)`.
    fn streaming_source(
        &self,
        mel_in: &Tensor,
        phase_in: f64,
        overlap_samples: usize,
        prev_tail: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, f64), SynthError> {
        let f0 = self.vocoder.f0_predict(mel_in)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        let n = f0v.len() * F0_UPSAMPLE;

        // New-region excitation: continue the global cumulative phase from phase_in. The
        // overlap region [0, overlap_samples) is overwritten below (so it does not advance
        // the global phase — phase_in already integrated those frames in the prev chunk).
        let mut source = vec![0f32; n];
        let mut acc = phase_in;
        let two_pi = 2.0 * std::f64::consts::PI;
        for s in overlap_samples.min(n)..n {
            let fhz = f0v[s / F0_UPSAMPLE] as f64;
            acc += fhz / MEL_SR as f64;
            source[s] = (SINE_AMP * (two_pi * acc).sin()) as f32;
        }
        let phase_out = acc;

        // Overwrite the overlap with the previous chunk's trailing source (phase-coherent).
        if let Some(t) = prev_tail {
            let prev: Vec<f32> = t.flatten_all()?.to_vec1::<f32>()?;
            let m = overlap_samples.min(prev.len()).min(n);
            let off = prev.len() - m; // align on the tail if lengths differ
            source[..m].copy_from_slice(&prev[off..off + m]);
        }

        let src_3d = Tensor::from_vec(source.clone(), (1, 1, n), &self.dev)?;
        let src_flat = Tensor::from_vec(source, (1, n), &self.dev)?;
        Ok((src_3d, src_flat, phase_out))
    }
}

/// Cross-fade `emit`'s leading `STREAM_SOURCE_CACHE` samples with `prev_tail` using a
/// hamming window of length `2*STREAM_SOURCE_CACHE` — CosyVoice2's `fade_in_out`:
/// `emit[:n] = emit[:n]*w_in + prev_tail*w_out`, `w_in` the rising (first) half and
/// `w_out` the falling (second) half. (Local copy of CV2's `hamming_crossfade`.)
fn hamming_crossfade(emit: &Tensor, prev_tail: &Tensor, dev: &Device) -> Result<Tensor, SynthError> {
    let n = STREAM_SOURCE_CACHE;
    let emit_len = emit.dim(1)?;
    let tail_len = prev_tail.dim(1)?;
    let overlap = n.min(emit_len).min(tail_len);
    if overlap == 0 {
        return Ok(emit.clone());
    }
    let win_len = 2 * n;
    let mut w_in = vec![0f32; overlap];
    let mut w_out = vec![0f32; overlap];
    let hm = |idx: usize| -> f32 {
        0.54 - 0.46 * (2.0 * std::f32::consts::PI * idx as f32 / (win_len as f32 - 1.0)).cos()
    };
    for m in 0..overlap {
        w_in[m] = hm(m);
        w_out[m] = hm(win_len - overlap + m);
    }
    let w_in = Tensor::from_vec(w_in, (1, overlap), dev)?;
    let w_out = Tensor::from_vec(w_out, (1, overlap), dev)?;

    let head = emit.narrow(1, 0, overlap)?;
    let tail_seg = prev_tail.narrow(1, tail_len - overlap, overlap)?;
    let blended = (head.broadcast_mul(&w_in)? + tail_seg.broadcast_mul(&w_out)?)?;
    if emit_len > overlap {
        let rest = emit.narrow(1, overlap, emit_len - overlap)?;
        Ok(Tensor::cat(&[&blended, &rest], 1)?)
    } else {
        Ok(blended)
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

// ---- HiFT source-excitation math (pure; the `*_source` methods call these) --------
//
// Extracted verbatim from `Cv3Synthesizer::deterministic_source` / `quality_source` so
// the f0 → source math is testable **model-free** (the methods only add the `f0_predict`
// + merge-weight fetch + tensor wrap around these). The computation is byte-unchanged.
// `#[doc(hidden)] pub` is the test seam — not part of the documented surface.

/// Deterministic, zero-phase, single-harmonic HiFT source from an f0 frame vector:
/// nearest-upsample f0 by [`F0_UPSAMPLE`], then `sine_amp·sin(2π·cumsum(f0_up/sr))`.
#[doc(hidden)]
pub fn det_source_from_f0(f0v: &[f32]) -> Vec<f32> {
    let n = f0v.len() * F0_UPSAMPLE;
    let mut source: Vec<f32> = Vec::with_capacity(n);
    let mut acc = 0.0f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    for &v in f0v {
        let fhz = v as f64;
        for _ in 0..F0_UPSAMPLE {
            acc += fhz / MEL_SR as f64;
            source.push((SINE_AMP * (two_pi * acc).sin()) as f32);
        }
    }
    source
}

/// Random-phase NSF (`SourceModuleHnNSF`) source from an f0 frame vector + the learned
/// `m_source.l_linear` merge `(merge_w, merge_b)`: `merge_w.len()` harmonics with seeded
/// random initial phase (φ_0 = 0), a voiced/unvoiced mask, additive Gaussian breath, and
/// the learned `linear + tanh` merge. All randomness is the **seeded** [`SplitMix64`], so
/// a `(f0v, merge, seed)` triple is bit-reproducible.
#[doc(hidden)]
pub fn quality_source_from_f0(f0v: &[f32], merge_w: &[f32], merge_b: f32, seed: u64) -> Vec<f32> {
    let n = f0v.len() * F0_UPSAMPLE; // source samples = mel frames * 480
    let mut rng = SplitMix64::new(seed);

    // Random initial phase per harmonic: φ_0 = 0 (fundamental), φ_i ∈ (-π, π].
    let mut phi = vec![0f64; merge_w.len()];
    for p in phi.iter_mut().skip(1) {
        *p = (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI;
    }

    // Single-channel merged excitation, sample by sample.
    let mut source: Vec<f32> = Vec::with_capacity(n);
    let mut base_phase = 0.0f64; // θ(t) = 2π·cumsum(f0_up/sr), built incrementally
    let two_pi = 2.0 * std::f64::consts::PI;
    for s in 0..n {
        let fhz = f0v[s / F0_UPSAMPLE] as f64; // nearest-upsample of f0 by 480
        base_phase += two_pi * (fhz / MEL_SR as f64);
        let uv = if fhz > NSF_VOICED_THRESHOLD { 1.0 } else { 0.0 };
        let noise_amp = uv * NSF_NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
        // harmonic components -> learned linear(9->1) + tanh merge.
        let mut acc = merge_b as f64;
        for (i, &w) in merge_w.iter().enumerate() {
            let h = (i + 1) as f64; // harmonic multiplier (fundamental = 1)
            let sine = SINE_AMP * (h * base_phase + phi[i]).sin();
            let noise = noise_amp * rng.next_gauss();
            let comp = sine * uv + noise; // SineGen: sine_waves * uv + noise
            acc += w as f64 * comp;
        }
        source.push(acc.tanh() as f32);
    }
    source
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
