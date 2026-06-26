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
/// SineGen sine amplitude (`sine_amp`, CosyVoice2 default).
const SINE_AMP: f64 = 0.1;

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
        let tokenizer = TextTokenizer::from_file(&cfg.tokenizer_json)?;
        let speech_tokenizer = SpeechTokenizer::load(&cfg.speech_tokenizer_onnx)?;
        let speaker = CamPlus::load(&cfg.spk_weights, dev.clone())?;
        let lm = syrinx_lm::real::Qwen2Lm::load(&cfg.lm_weights, dev.clone())?;
        let flow = Flow::load(&cfg.flow_weights, dev.clone())?;
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
    /// The per-chunk HiFT source is built with the same deterministic, zero-phase F0
    /// source as the non-streaming smoke path ([`Synthesizer::deterministic_source_stft`]),
    /// applied to each (overlap-extended) mel chunk. A pinned `inputs.s_stft` is **not**
    /// used here (the streaming source must be rebuilt per chunk); `inputs.z`, the LM
    /// seed/cap, and `inputs.pinned_speech_token` are honoured exactly as in `synthesize`.
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

        // Per-chunk source builder: the deterministic F0 source STFT over the chunk's
        // (overlap-extended) mel. `&self` capture keeps the vocoder + device in scope.
        let source_fn = |mel: &Tensor| -> candle_core::Result<Tensor> {
            self.deterministic_source_stft(mel)
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
        // Upsample F0 by nearest (Upsample(scale_factor=480)) to the source rate.
        let mut f0_up: Vec<f64> = Vec::with_capacity(f0v.len() * F0_UPSAMPLE);
        for &v in &f0v {
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
    let (b, n_mels, t_in) = mel.dims3()?;
    if b != 1 {
        return Err(SynthError::Candle(format!(
            "rate mel time-scale expects batch 1, got {b}"
        )));
    }
    // [1, n_mels, T] -> [n_mels, T] -> Vec<Vec<f32>> rows.
    let m2 = mel.squeeze(0)?; // [n_mels, T]
    let flat: Vec<f32> = m2.flatten_all()?.to_vec1::<f32>()?;
    let mut grid: Vec<Vec<f32>> = Vec::with_capacity(n_mels);
    for row in 0..n_mels {
        grid.push(flat[row * t_in..(row + 1) * t_in].to_vec());
    }

    let scaled = syrinx_prosody::render::time_scale_mel(&grid, rate)
        .map_err(|e| SynthError::Candle(format!("mel time-scale failed: {e:?}")))?;

    let t_out = scaled[0].len();
    let mut out_flat: Vec<f32> = Vec::with_capacity(n_mels * t_out);
    for row in &scaled {
        out_flat.extend_from_slice(row);
    }
    let out = Tensor::from_vec(out_flat, (1, n_mels, t_out), dev)?;
    Ok(out)
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
