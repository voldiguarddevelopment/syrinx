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

use candle_core::{DType, Device, Tensor};

use syrinx_acoustic::real::{token2wav, token2wav_streaming, AudioChunk, Flow};
use syrinx_frontend::feat::{kaldi_fbank, prompt_mel};
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

    /// Load every sub-model in its **quantized** variant for the README ~270 MB size
    /// goal: the LM via [`syrinx_lm::real::Qwen2Lm::load_quantized`] (int4 big linears +
    /// int8 dequant-on-gather embedding tables) and the flow via [`Flow::load_quantized`]
    /// (Q4_0 `linear()` weights). The CAM++ speaker, HiFT vocoder and tokenizers load
    /// exactly as in [`load`] (small / not the footprint bulk).
    ///
    /// int4/int8 trade quality for size; CPU stays the parity device for the **fp32**
    /// path ([`load`]), and the quantized quality is measured on the box (SIM-o), not
    /// asserted to fp32 parity here.
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
        let speaker = CamPlus::load(&cfg.spk_weights, dev.clone())?;
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
        let vocoder = HiftVocoder::load(&cfg.hift_weights, dev.clone())?;
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

    /// Run the frontend half: tokenize text and derive the prompt-side conditioning
    /// (`spk`, `prompt_token`, `prompt_feat`) from the reference waveforms, applying
    /// the CosyVoice2 `%2` token/feat alignment.
    ///
    /// `ref_wav_16k` is the 16 kHz mono reference (fbank + speech-token input);
    /// `ref_wav_24k` is the same clip resampled to 24 kHz (prompt-mel input). Both
    /// are derived from the prompt wav by `torchaudio.load -> mono -> resample`
    /// upstream — resampling is intentionally the caller's job (the frontend `feat`
    /// math is the only thing under parity test), so the deterministic e2e test can
    /// feed the *exact* reference-resampled waveforms.
    pub fn prompt_cond(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<PromptCond, SynthError> {
        // --- text tokens: prompt_text ++ tts_text (CosyVoice2 concatenates them). ---
        // Text-normalization hook (additive, `tn` feature): match CosyVoice2's
        // `frontend.text_normalize` (wetext zh+en) before tokenizing. Off by default
        // so the raw-text parity tests (run with `--features real` only) are unchanged.
        let prompt_text = tn_normalize(prompt_text);
        let prompt_text = prompt_text.as_ref();
        let tts_text = tn_normalize(tts_text);
        let tts_text = tts_text.as_ref();
        let prompt_text_ids = self.tokenizer.encode(prompt_text)?;
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

        // --- prompt speech tokens via the ONNX tokenizer (16 kHz). ---
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

        // --- CosyVoice2 %2 alignment: token_len = min(T'/2, |prompt_token|);
        //     truncate prompt_feat to 2*token_len frames, prompt_token to token_len. ---
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

    /// CAM++ speaker x-vector `[1, 192]` for a 16 kHz mono waveform, via the same
    /// kaldi-fbank -> per-time mean-subtraction -> CAM++ path used for zero-shot
    /// conditioning. Exposed for evaluation — e.g. SIM-o (speaker cosine) between a
    /// reference clip and a synthesized clip.
    pub fn speaker_embedding(&self, audio_16k: &[f32]) -> Result<Tensor, SynthError> {
        let fbank_grid = kaldi_fbank(audio_16k, SR_16K, FBANK_MELS);
        let fbank = grid_to_tensor(&fbank_grid, &self.dev)?;
        let fbank = subtract_time_mean(&fbank)?;
        let fbank = fbank.unsqueeze(0)?;
        Ok(self.speaker.forward(&fbank)?)
    }

    /// Generate the speech-token sequence with the **live** LM (pinned-PRNG sampling).
    /// Returns the ids as i64 `[1, N]` ready for the flow.
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
        // Honour any caller cap (never below min_len so the EOS-mask window is intact).
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

    /// The flow + vocoder half: speech tokens -> mel -> audio `[1, L]`. `z` and
    /// `s_stft` come from `inputs` when pinned, else are derived (zeros `z`,
    /// deterministic F0 source).
    pub fn token2wav(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        inputs: &SynthInputs,
    ) -> Result<Tensor, SynthError> {
        // total flow length = 2 * (|prompt_token| + |speech_token|).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);

        // CFM noise z: pinned or zeros fallback.
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // HiFT source STFT: pinned, or a deterministic zero-phase F0 source.
        let s_stft = match &inputs.s_stft {
            Some(s) => s.clone(),
            None => {
                // Need the generated mel to drive the F0 predictor, so compute it
                // here (forward_zero_shot), build the source, then decode with both.
                let mel = self.flow.forward_zero_shot(
                    &cond.prompt_token,
                    speech_token,
                    &cond.prompt_feat,
                    &cond.spk_embedding,
                    &z,
                    N_TIMESTEPS,
                )?; // [1, 80, 2*Tg]
                let s = self.deterministic_source_stft(&mel)?;
                let audio = self.vocoder.decode(&mel, &s)?;
                return Ok(audio);
            }
        };

        // Pinned-source path: the single-call token2wav glue.
        let audio = token2wav(
            &self.flow,
            &self.vocoder,
            &cond.prompt_token,
            speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            &s_stft,
            N_TIMESTEPS,
        )?;
        Ok(audio)
    }

    /// **Streaming** synthesis: same `tts_text`-in-reference-voice flow + vocoder as
    /// [`Synthesizer::synthesize`], but audio is emitted **incrementally** chunk by
    /// chunk (low first-byte latency) via `on_chunk`, instead of one final `Vec`.
    ///
    /// This drives [`token2wav_streaming`], which replicates CosyVoice2's
    /// `token2wav` streaming path: a HiFT mel/source/speech overlap cache across
    /// chunks plus a hamming cross-fade at every chunk boundary. `token_hop` is the
    /// number of finalized speech tokens per chunk (a chunk needs only
    /// `token_hop + pre_lookahead` tokens present, so the first chunk lands long
    /// before the utterance finishes).
    ///
    /// Each emitted chunk is delivered to `on_chunk` as a flat `Vec<f32>` of 24 kHz
    /// samples in order; concatenating them yields the full streamed waveform.
    ///
    /// The per-chunk HiFT source is built by [`Synthesizer::streaming_source_phase`],
    /// which carries the deterministic F0 excitation phase **continuously** across
    /// chunks (one global sinusoid, matching the non-streaming source) and overwrites
    /// each chunk's overlap region with the previous chunk's trailing source samples
    /// (CosyVoice2 `cache_source`). This is what makes the streamed waveform
    /// sample-faithful to the non-streaming path rather than only length/energy-faithful.
    /// A pinned `inputs.s_stft` is **not** used here (the streaming source is rebuilt
    /// per chunk); `inputs.z`, the LM seed/cap, and `inputs.pinned_speech_token` are
    /// honoured exactly as in `synthesize`.
    pub fn synthesize_streaming(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        token_hop: usize,
        mut on_chunk: impl FnMut(Vec<f32>) -> Result<(), SynthError>,
    ) -> Result<(), SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;

        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // Per-chunk **phase-continuous** source builder: continues the global F0
        // excitation phase across chunks and overwrites each chunk's overlap with the
        // previous chunk's trailing source (CosyVoice2 `cache_source`), so the streamed
        // waveform is sample-faithful to the non-streaming source. `&self` capture
        // keeps the vocoder + device in scope.
        let source_fn = |mel_in: &Tensor,
                         phase_in: f64,
                         overlap: usize,
                         prev_tail: Option<&Tensor>|
         -> candle_core::Result<(Tensor, Tensor, f64)> {
            self.streaming_source_phase(mel_in, phase_in, overlap, prev_tail)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        };

        let mut cb_err: Option<SynthError> = None;
        let res = token2wav_streaming(
            &self.flow,
            &self.vocoder,
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            &source_fn,
            token_hop,
            N_TIMESTEPS,
            |chunk: AudioChunk| {
                let wav: Vec<f32> = chunk
                    .wav
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())?;
                if let Err(e) = on_chunk(wav) {
                    cb_err = Some(e);
                    return Err(candle_core::Error::Msg("streaming callback aborted".into()));
                }
                Ok(())
            },
        );
        if let Some(e) = cb_err {
            return Err(e);
        }
        res.map_err(SynthError::from)
    }

    /// Run only the flow-matching CFM mel decoder (`forward_zero_shot`) — the
    /// heaviest acoustic stage — returning the generated mel `[1, 80, 2*Tg]`.
    ///
    /// Exposed so callers (e.g. the GPU speed benchmark) can time the flow ODE in
    /// isolation. The full pipeline uses this internally inside [`token2wav`].
    pub fn flow_forward(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor, SynthError> {
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            z,
            n_timesteps,
        )?;
        Ok(mel)
    }

    /// Build a **deterministic, zero-phase, noise-free** HiFT source STFT `[1, 18, T]`
    /// from a generated mel, for the functional (non-parity) path. Mirrors the real
    /// source branch's deterministic core: F0 predictor -> nearest-upsample by 480 ->
    /// single-harmonic sine `sine_amp * sin(2π·cumsum(f0/sr))` -> STFT (n_fft=16,
    /// hop=4, hann window) -> `cat([real, imag])`. The model's SineGen adds a random
    /// initial phase + Gaussian noise (intentionally not ported), so this is a
    /// faithful *smoke* source — finite, non-silent, correctly shaped — not a parity
    /// source.
    pub fn deterministic_source_stft(&self, mel: &Tensor) -> Result<Tensor, SynthError> {
        let f0 = self.vocoder.f0_predict(mel)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        self.f0_to_source_stft(&f0v)
    }

    /// Build the deterministic zero-phase HiFT source STFT `[1, 18, T]` from an
    /// explicit per-mel-frame F0 vector `f0v` (Hz). This is the F0→source half of
    /// [`Synthesizer::deterministic_source_stft`], factored out so a prosody plan
    /// can retune the F0 (pitch control) before the source is built: nearest
    /// upsample by 480 → single-harmonic `sine_amp·sin(2π·cumsum(f0/sr))` → STFT.
    fn f0_to_source_stft(&self, f0v: &[f32]) -> Result<Tensor, SynthError> {
        // Upsample F0 by nearest (Upsample(scale_factor=480)) to the source rate.
        let mut f0_up: Vec<f64> = Vec::with_capacity(f0v.len() * F0_UPSAMPLE);
        for &v in f0v {
            for _ in 0..F0_UPSAMPLE {
                f0_up.push(v as f64);
            }
        }
        // Zero-phase instantaneous phase: phase[t] = 2π · cumsum(f0/sr).
        let mut source: Vec<f32> = Vec::with_capacity(f0_up.len());
        let mut acc = 0.0f64;
        for &fhz in &f0_up {
            acc += fhz / MEL_SR as f64;
            let s = SINE_AMP * (2.0 * std::f64::consts::PI * acc).sin();
            source.push(s as f32);
        }
        self.source_stft(&source)
    }

    /// Build the **random-phase NSF** HiFT source STFT `[1, 18, T]` from a generated
    /// mel — a faithful port of CosyVoice2's `SourceModuleHnNSF` (`SineGen`) source,
    /// for the perceptual-quality path. Unlike [`deterministic_source_stft`] (a
    /// single zero-phase harmonic, no noise), this reproduces every stochastic and
    /// multi-harmonic element of the real excitation:
    ///
    ///   * **`NB_HARMONICS + 1 = 9` sines** — fundamental + 8 overtones, the `i`-th
    ///     at instantaneous phase `(i+1)·θ(t)` where `θ(t) = 2π·cumsum(f0_up/sr)` is
    ///     the shared zero-phase ramp (the same per-sample cumulative phase the
    ///     deterministic source uses);
    ///   * **random initial phase per harmonic** — a constant offset `φ_i ∈ (-π, π]`
    ///     added to each overtone (the fundamental keeps `φ_0 = 0`, exactly as
    ///     `SineGen`'s `phase_vec[:,0,:] = 0`), decorrelating the overtones so the
    ///     source is no longer a single rigidly phase-locked buzz;
    ///   * **voiced/unvoiced mask** `uv = f0 > NSF_VOICED_THRESHOLD (10 Hz)` — the
    ///     harmonic sines are gated to zero in unvoiced frames;
    ///   * **additive Gaussian noise** at `noise_amp = uv·noise_std + (1-uv)·sine_amp/3`
    ///     — quiet (`σ=0.003`) breath in voiced frames, and `sine_amp/3 ≈ 0.033`
    ///     broadband noise replacing the (masked-out) sines in unvoiced frames;
    ///   * **learned harmonic merge** — the 9 components are fused by the checkpoint's
    ///     `m_source.l_linear` `Linear(9→1)` + `tanh`, the real `SourceModuleHnNSF`
    ///     merge, yielding the single-channel excitation that is then STFT'd.
    ///
    /// All randomness comes from a **seeded** SplitMix64 stream keyed by `seed`
    /// (never system RNG), so a given `(mel, seed)` is bit-reproducible across runs —
    /// the A/B harness can pin a seed and the result is stable.
    ///
    /// Honesty note: the 24 kHz config technically instantiates `SineGen2`, whose
    /// `_f02sine` accumulates phase at *frame* rate and linearly interpolates back up
    /// (an anti-aliasing detail); its steady-state output equals the per-sample
    /// `cumsum` ramp used here, and the perceptually load-bearing elements (the
    /// random per-harmonic phase, the 9 harmonics, the noise, the uv mask, and the
    /// learned merge) are identical between the two. This is a perceptual-quality
    /// source, not a bit-parity source — the model's RNG draw order is not portable.
    pub fn random_phase_source_stft(
        &self,
        mel: &Tensor,
        seed: u64,
    ) -> Result<Tensor, SynthError> {
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

        let n = f0v.len() * F0_UPSAMPLE; // source samples = mel frames * 480
        let mut rng = SplitMix64::new(seed);

        // Random initial phase per harmonic: φ_0 = 0 (fundamental), φ_i ∈ (-π, π].
        let n_comp = NB_HARMONICS + 1;
        let mut phi = vec![0f64; n_comp];
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
            // 9 harmonic components -> learned linear(9->1) + tanh merge.
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
        self.source_stft(&source)
    }

    /// Build the **streaming** HiFT source for one mel chunk with *global F0-phase
    /// continuity* (the faithfulness fix). Same deterministic F0 -> single-harmonic
    /// sine -> STFT core as [`Synthesizer::deterministic_source_stft`], but:
    ///
    ///   * the instantaneous phase is **not** reset to 0 for the chunk — it continues
    ///     from `phase_in` (the global phase, in cycles, at the start of this chunk's
    ///     *new* region), so concatenated chunks form one continuous sinusoid exactly
    ///     like the non-streaming source. (Per-chunk phase reset is what drove the
    ///     streaming-vs-non-streaming sample correlation to ~0.)
    ///   * the leading `overlap_samples` (the carried mel-overlap region) are
    ///     overwritten by `prev_tail`, the previous chunk's trailing source samples,
    ///     so the overlap is sample-identical to what the vocoder already emitted
    ///     (CosyVoice2 `cache_source`), keeping the boundary cross-fade coherent.
    ///
    /// Returns `(s_stft [1,18,T], source_wave [1, T_src], phase_out)` where `phase_out`
    /// is the global phase at this chunk's end (the next chunk's `phase_in`).
    pub fn streaming_source_phase(
        &self,
        mel_in: &Tensor,
        phase_in: f64,
        overlap_samples: usize,
        prev_tail: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, f64), SynthError> {
        let f0 = self.vocoder.f0_predict(mel_in)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        let n = f0v.len() * F0_UPSAMPLE; // source samples = mel frames * 480

        // New-region excitation: continue the global cumulative phase from phase_in.
        // The overlap region [0, overlap_samples) is overwritten below, so it does not
        // advance the global phase (phase_in already accounts for it via the previous
        // chunk's integral over the same frames).
        let mut source: Vec<f32> = vec![0f32; n];
        let mut acc = phase_in;
        for s in overlap_samples.min(n)..n {
            let fhz = f0v[s / F0_UPSAMPLE] as f64;
            acc += fhz / MEL_SR as f64;
            source[s] = (SINE_AMP * (2.0 * std::f64::consts::PI * acc).sin()) as f32;
        }
        let phase_out = acc;

        // Overwrite the overlap with the previous chunk's trailing source (phase-coherent).
        if let Some(t) = prev_tail {
            let prev: Vec<f32> = t.flatten_all()?.to_vec1::<f32>()?;
            let m = overlap_samples.min(prev.len()).min(n);
            let off = prev.len() - m; // align on the tail if lengths differ
            source[..m].copy_from_slice(&prev[off..off + m]);
        }

        let s_stft = self.source_stft(&source)?;
        let src_wave = Tensor::from_vec(source, (1, n), &self.dev)?;
        Ok((s_stft, src_wave, phase_out))
    }

    /// STFT a 1-D source waveform into `[1, 18, T]` (n_fft=16, hop=4, periodic hann,
    /// `center=True` reflect padding) matching the HiFT `_stft`, real ++ imag.
    fn source_stft(&self, source: &[f32]) -> Result<Tensor, SynthError> {
        let n_fft = HIFT_N_FFT;
        let hop = HIFT_HOP;
        let bins = HIFT_BINS;
        // periodic hann window (get_window("hann", n_fft, fftbins=True)).
        let window: Vec<f64> = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
            .collect();
        // center=True: reflect-pad n_fft/2 each side.
        let pad = n_fft / 2;
        let padded = reflect_pad(source, pad);
        let n = padded.len();
        if n < n_fft {
            return Err(SynthError::Candle("source too short for STFT".into()));
        }
        let n_frames = (n - n_fft) / hop + 1;
        // real and imag channels: [n_fft+2] = [bins (re) ++ bins (im)] over T frames.
        let mut data = vec![0f32; (2 * bins) * n_frames];
        for t in 0..n_frames {
            let start = t * hop;
            for k in 0..bins {
                let mut re = 0f64;
                let mut im = 0f64;
                let w = -2.0 * std::f64::consts::PI * k as f64 / n_fft as f64;
                for j in 0..n_fft {
                    let x = padded[start + j] as f64 * window[j];
                    let ang = w * j as f64;
                    re += x * ang.cos();
                    im += x * ang.sin();
                }
                data[k * n_frames + t] = re as f32;
                data[(bins + k) * n_frames + t] = im as f32;
            }
        }
        let t = Tensor::from_vec(data, (1, 2 * bins, n_frames), &self.dev)?;
        Ok(t)
    }
}

// ============================================================================
// Prosody control hook (additive — speech-rate, DESIGN T3.4).
//
// Kept in its own `impl` block + free helpers so it does not overlap the core
// pipeline methods above. The rate knob time-scales the *generated mel* along
// its frame axis (via `syrinx_prosody::render::time_scale_mel`) between the flow
// and the vocoder: more frames => longer/slower audio, fewer => shorter/faster,
// with each frame's spectral shape (and thus pitch) preserved. This is the
// length-regulator move and the one place a `rate` actually changes the rendered
// duration. The non-`real`-deterministic (live z=zeros / F0 source) path is used,
// matching the functional smoke; a pinned-source parity run keeps the original
// `synthesize` untouched.
// ============================================================================
impl Synthesizer {
    /// Synthesize `tts_text` in the reference voice at speech-rate `rate`,
    /// returning the 24 kHz waveform.
    ///
    /// `rate == 1.0` reproduces the default-speed [`Synthesizer::synthesize`]
    /// audio (up to mel-frame rounding); `rate > 1.0` is faster/shorter and
    /// `rate < 1.0` slower/longer, with pitch preserved. The rate is applied by
    /// time-scaling the generated mel before the vocoder; the speech-token
    /// sequence and prompt conditioning are unchanged, so the *content* is the
    /// same utterance, only its duration differs. A non-positive or non-finite
    /// `rate` returns [`SynthError::Candle`] describing the bad knob.
    ///
    /// This always uses the deterministic (live `z = zeros`, deterministic F0)
    /// source — the same functional path as the default smoke — so any pinned
    /// `s_stft` in `inputs` is ignored here (a pinned source is length-locked to
    /// the unscaled mel and cannot describe a rate-scaled one). `inputs` is still
    /// honoured for `pinned_speech_token`, `lm_seed`, and `max_gen_steps`.
    pub fn synthesize_with_rate(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        rate: f64,
    ) -> Result<Vec<f32>, SynthError> {
        if !(rate.is_finite() && rate > 0.0) {
            return Err(SynthError::Candle(format!(
                "speech rate must be finite and > 0, got {rate}"
            )));
        }
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: pinned or zeros (the deterministic functional fallback).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // Flow -> generated mel [1, 80, 2*Tg].
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            N_TIMESTEPS,
        )?;

        // The rate knob: time-scale the mel along its frame axis (pitch-preserving).
        let mel = time_scale_mel_tensor(&mel, rate, &self.dev)?;

        // Deterministic source from the (now rate-scaled) mel, then vocode.
        let s = self.deterministic_source_stft(&mel)?;
        let audio = self.vocoder.decode(&mel, &s)?;
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }
}

// ============================================================================
// Editable prosody-plan control (additive — pitch + duration, DESIGN §6 Phase 3).
//
// `synthesize_with_plan` applies a `syrinx_prosody::render_plan::RenderPlan` to the
// generated mel (duration = frame-axis time-warp; pitch = F0-source retune, plus an
// opt-in mel-bin formant shift) BEFORE the HiFT vocoder. Kept in its own `impl`
// block + helpers, separate from the core pipeline and the rate hook above. Like
// the rate hook it always uses the deterministic (live z=zeros, deterministic F0)
// source — a pinned `inputs.s_stft` is length/pitch-locked to the unscaled mel and
// cannot describe a retuned/rescaled one, so it is ignored here; `inputs` is still
// honoured for `pinned_speech_token`, `lm_seed`, and `max_gen_steps`. The existing
// `synthesize` / `synthesize_with_rate` are untouched.
// ============================================================================
impl Synthesizer {
    /// Synthesize `tts_text` in the reference voice under an editable
    /// [`RenderPlan`](syrinx_prosody::render_plan::RenderPlan), returning the 24 kHz
    /// waveform.
    ///
    /// The plan is applied to the flow's generated mel before the vocoder:
    ///   * **duration** — the mel frame axis is time-warped by the plan's global +
    ///     per-region rate (pitch-preserving length regulation);
    ///   * **pitch** — the HiFT F0 predicted from the (warped) mel is multiplied by
    ///     the plan's per-frame F0 ratio (global + per-region semitone shift) before
    ///     the sine source is built — a formant-preserving pitch shift via the NSF
    ///     source; with `mel_envelope_shift` the mel is also bin-warped (formant
    ///     shifting, opt-in).
    ///
    /// [`RenderPlan::identity`](syrinx_prosody::render_plan::RenderPlan::identity)
    /// reproduces the default-speed [`Synthesizer::synthesize`] audio (up to mel
    /// rounding). A plan that fails validation against the generated mel length
    /// returns [`SynthError::Candle`] describing the bad knob.
    pub fn synthesize_with_plan(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        plan: &syrinx_prosody::render_plan::RenderPlan,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: pinned or zeros (the deterministic functional fallback).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // Flow -> generated mel [1, 80, 2*Tg].
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            N_TIMESTEPS,
        )?;

        // Apply the plan to the mel grid: time-warp (duration) + per-output-frame
        // F0 multiplier (pitch) + opt-in mel-bin formant shift.
        let grid = mel_tensor_to_grid(&mel)?;
        let (scaled_grid, f0_mult) = plan
            .apply(&grid)
            .map_err(|e| SynthError::Candle(format!("prosody plan apply failed: {e:?}")))?;
        let scaled_mel = grid_to_mel_tensor(&scaled_grid, &self.dev)?; // [1, 80, T_out]

        // Predict F0 from the warped mel, retune it by the per-frame ratio, build
        // the source, then vocode.
        let f0 = self.vocoder.f0_predict(&scaled_mel)?; // [1, T_out]
        let mut f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        if f0v.len() != f0_mult.len() {
            return Err(SynthError::Candle(format!(
                "prosody plan F0 length mismatch: f0={} mult={}",
                f0v.len(),
                f0_mult.len()
            )));
        }
        for (v, m) in f0v.iter_mut().zip(f0_mult.iter()) {
            *v = (*v as f64 * *m) as f32;
        }
        let s = self.f0_to_source_stft(&f0v)?;
        let audio = self.vocoder.decode(&scaled_mel, &s)?;
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }
}

// ============================================================================
// Output watermarking (additive — the README's "post-edit-detectable watermark
// on every synthesized output").
//
// Kept in its own `impl` block so it does not overlap the core pipeline. The
// watermark itself lives in the pure-Rust, model-free `crate::watermark` module
// (embed/detect on a 24 kHz mono `f32` buffer); this is only the synth-side glue
// that runs the existing `synthesize` and stamps its output. The plain
// `synthesize` is untouched, so the parity path is unaffected.
// ============================================================================
impl Synthesizer {
    /// Synthesize `tts_text` in the reference voice and **stamp the output with a
    /// spread-spectrum watermark** (see [`crate::watermark`]), returning the 24 kHz
    /// waveform.
    ///
    /// Identical to [`Synthesizer::synthesize`] followed by
    /// [`crate::watermark::embed_watermark`] at the default amplitude
    /// ([`crate::watermark::DEFAULT_AMP`], ≈ −48 dBFS). The `key` is the shared
    /// secret the detector needs; `payload` is a 16-bit tag (e.g. a model/version
    /// or batch id). The perturbation is `±DEFAULT_AMP` per sample — well below the
    /// perceptual threshold for speech — and [`crate::watermark::detect_watermark`]
    /// recovers `(present, confidence, payload)` from the unmodified output (and
    /// after light post-editing; see the module's honest robustness boundary).
    pub fn synthesize_watermarked(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        key: u64,
        payload: u16,
    ) -> Result<Vec<f32>, SynthError> {
        let mut wav = self.synthesize(tts_text, prompt_text, ref_wav_16k, ref_wav_24k, inputs)?;
        crate::watermark::embed_watermark(&mut wav, key, payload);
        Ok(wav)
    }
}

// ============================================================================
// Perceptual-quality synthesis (additive — random-phase NSF source).
//
// `synthesize_quality` is byte-for-byte `synthesize`'s flow + vocoder, except the
// HiFT excitation is built by `random_phase_source_stft` (CosyVoice2's real
// `SourceModuleHnNSF`: 9 harmonics + random per-harmonic phase + Gaussian noise +
// uv mask + learned merge) instead of the deterministic zero-phase smoke source.
// The deterministic source is buzzy (a single rigidly phase-locked harmonic, no
// noise) and is the main reason the functional UTMOS came out low (~2.03); the
// random-phase source restores the natural harmonic decorrelation + breath the
// HiFT filter expects, which should raise MOS. Kept in its own `impl` block so the
// parity-default `synthesize` and the existing smoke tests are untouched. A pinned
// `inputs.s_stft` is ignored here (this path *is* the source choice); `inputs` is
// still honoured for `pinned_speech_token`, `z`, `lm_seed`, and `max_gen_steps`.
// ============================================================================
impl Synthesizer {
    /// Full synthesis with the **random-phase NSF source** (perceptual-quality
    /// path), returning the 24 kHz waveform. Same pipeline as
    /// [`Synthesizer::synthesize`] but the HiFT source is
    /// [`Synthesizer::random_phase_source_stft`] (seeded by `source_seed`) instead
    /// of the deterministic zero-phase source.
    ///
    /// `source_seed` makes the (otherwise stochastic) source reproducible: the same
    /// `source_seed` + same tokens yields the same waveform. `inputs.z` is honoured
    /// (pinned or zeros fallback) exactly as in `synthesize`; any `inputs.s_stft` is
    /// ignored (this method builds the source itself).
    pub fn synthesize_quality(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        source_seed: u64,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: pinned or zeros (the deterministic functional fallback).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // Flow -> generated mel, random-phase NSF source from it, then vocode.
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            N_TIMESTEPS,
        )?;
        let s = self.random_phase_source_stft(&mel, source_seed)?;
        let audio = self.vocoder.decode(&mel, &s)?;
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }
}

/// Minimal seeded **SplitMix64** PRNG — the source builder's *only* randomness, so
/// the random-phase source is reproducible from its `seed` (never system RNG).
///
/// SplitMix64 is the standard fast seeding generator (the one Rust's `StdRng` and
/// xoshiro use to expand a seed); a single stream here drives both the per-harmonic
/// initial phases (uniform) and the additive Gaussian noise (Box–Muller).
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

    /// Standard normal `N(0, 1)` via Box–Muller, generating two at a time and
    /// caching the spare (matches `torch.randn`'s distribution, not its byte stream).
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

/// Convert a `[1, n_mels, T]` Candle mel tensor to a `[n_mels][T]` row-major grid
/// for the pure-Rust `syrinx-prosody` transforms (keeping that crate Candle-free).
fn mel_tensor_to_grid(mel: &Tensor) -> Result<Vec<Vec<f32>>, SynthError> {
    let (b, n_mels, t_in) = mel.dims3()?;
    if b != 1 {
        return Err(SynthError::Candle(format!(
            "mel grid conversion expects batch 1, got {b}"
        )));
    }
    let m2 = mel.squeeze(0)?; // [n_mels, T]
    let flat: Vec<f32> = m2.flatten_all()?.to_vec1::<f32>()?;
    let mut grid: Vec<Vec<f32>> = Vec::with_capacity(n_mels);
    for row in 0..n_mels {
        grid.push(flat[row * t_in..(row + 1) * t_in].to_vec());
    }
    Ok(grid)
}

/// Rebuild a `[1, n_mels, T]` Candle mel tensor from a `[n_mels][T]` grid on `dev`.
fn grid_to_mel_tensor(grid: &[Vec<f32>], dev: &Device) -> Result<Tensor, SynthError> {
    let n_mels = grid.len();
    let t = if n_mels == 0 { 0 } else { grid[0].len() };
    let mut flat: Vec<f32> = Vec::with_capacity(n_mels * t);
    for row in grid {
        flat.extend_from_slice(row);
    }
    Ok(Tensor::from_vec(flat, (1, n_mels, t), dev)?)
}

/// Time-scale a `[1, n_mels, T]` Candle mel tensor by speech-rate `rate` using
/// [`syrinx_prosody::render::time_scale_mel`], returning `[1, n_mels, T_out]`.
///
/// The tensor is moved to a flat `[n_mels][T]` grid for the pure-Rust transform
/// (keeping `syrinx-prosody` Candle-free) and rebuilt on `dev`. `rate > 1` yields
/// fewer frames (shorter), `rate < 1` more (longer); the band axis is untouched.
fn time_scale_mel_tensor(
    mel: &Tensor,
    rate: f64,
    dev: &Device,
) -> Result<Tensor, SynthError> {
    let grid = mel_tensor_to_grid(mel)?; // [n_mels][T]
    let scaled = syrinx_prosody::render::time_scale_mel(&grid, rate)
        .map_err(|e| SynthError::Candle(format!("mel time-scale failed: {e:?}")))?;
    grid_to_mel_tensor(&scaled, dev)
}

// ---- text-normalization hook (additive) --------------------------------------
//
// Mirrors CosyVoice2's `frontend.text_normalize` (wetext zh+en). Gated on the
// crate `tn` feature so it is fully opt-in: with `tn` off (e.g. the raw-text
// parity tests, which run `--features real` only) `tn_normalize` is the identity,
// leaving the un-normalized text path byte-for-byte unchanged.

/// Normalize text before tokenizing when the `tn` feature is enabled.
#[cfg(feature = "tn")]
fn tn_normalize(s: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Owned(syrinx_frontend::textnorm::normalize_text(s))
}

/// Identity passthrough when `tn` is disabled (raw text, as before).
#[cfg(not(feature = "tn"))]
fn tn_normalize(s: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Borrowed(s)
}

// ---- free helpers ------------------------------------------------------------

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

/// Reflect-pad a 1-D signal by `pad` each side (torch reflect: boundary not repeated).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![0.0; 2 * pad];
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    for k in 0..pad {
        out.push(x[(pad - k).min(n - 1)]);
    }
    out.extend_from_slice(x);
    for k in 0..pad {
        let idx = n.saturating_sub(2).saturating_sub(k);
        out.push(x[idx]);
    }
    out
}
