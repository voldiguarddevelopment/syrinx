//! The **real** Qwen3-TTS engine: the Candle half of [`crate::qwen`].
//!
//! [`QwenModelEngine`] is a [`QwenEngine`] that holds a loaded talker + code predictor
//! (`syrinx_qwen::model::Qwen3Tts`), the text tokenizer, the prompt-side ids, and the
//! 12 Hz tokenizer's RVQ + decoder stack, and turns one [`QwenRequest`] into 24 kHz mono
//! samples. Everything that decides *what* to ask the model lives in [`crate::qwen`] and
//! is model-free; this module only drives the port.
//!
//! The chain per request is exactly the one `syrinx-qwen`'s own `examples/synth.rs` and
//! `examples/clone.rs` drive — no reimplementation, only the crate's public API:
//!
//! ```text
//!   prompt::build_{custom_voice,voice_design,voice_clone}
//!     -> Qwen3Tts::realize_plan -> Qwen3Tts::generate
//!     -> Rvq::decode (rvq_first + rvq_rest, PARALLEL stacks)
//!     -> Decoder::decode -> f32 @ 24 kHz
//! ```
//!
//! Requests are serialized through one [`Mutex`]: `generate` takes `&mut self`, and one
//! GPU/CPU box runs one synthesis at a time anyway (the same choice [`crate::RealSynth`]
//! makes). There is no chunk-streaming path in the port, so [`crate::qwen::QwenSynth`]
//! does not override [`crate::Synth::synthesize_stream`] and a `response_format:
//! "stream"` request gets the handler's buffered fallback.
//!
//! To serve one: [`QwenModelEngine::load`] → [`QwenModelEngine::into_synth`] →
//! [`crate::qwen::router_with_qwen_synth`], or [`crate::serve_blocking_dyn`] with the
//! synth in an `Arc`.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use syrinx_cue::BackendId;
use syrinx_qwen::codec::decoder::{Decoder, DecoderConfig};
use syrinx_qwen::codec::encoder::{MimiEncoder, MimiEncoderConfig};
use syrinx_qwen::codec::rvq::Rvq;
use syrinx_qwen::model::{DriveParams, Qwen3Tts};
use syrinx_qwen::nn::Weights;
use syrinx_qwen::prompt::{
    build_custom_voice, build_voice_clone, build_voice_design, CloneRef, PromptConfig,
    CUSTOM_VOICE_NON_STREAMING, VOICE_CLONE_NON_STREAMING, VOICE_DESIGN_NON_STREAMING,
};
use syrinx_qwen::speaker::{resample, SpeakerEncoder, SpeakerEncoderConfig};
use syrinx_qwen::tokenizer::QwenTokenizer;
use syrinx_qwen::{Qwen3TtsConfig, QwenVariant};

use crate::qwen::{QwenEngine, QwenMode, QwenPlanError, QwenRequest, QwenSynth};

/// Anything that can go wrong loading or driving the port, flattened to a message.
///
/// The upstream errors are five unrelated types (Candle, `PromptError`, `TokenizerError`,
/// `io::Error`, and the codec configs' plain `String`); [`QwenEngine::render`] returns a
/// `String` anyway, so carrying them separately would buy nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenEngineError(pub String);

impl std::fmt::Display for QwenEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for QwenEngineError {}

impl From<candle_core::Error> for QwenEngineError {
    fn from(e: candle_core::Error) -> Self {
        Self(format!("candle: {e}"))
    }
}

impl From<std::io::Error> for QwenEngineError {
    fn from(e: std::io::Error) -> Self {
        Self(format!("io: {e}"))
    }
}

impl From<syrinx_qwen::prompt::PromptError> for QwenEngineError {
    fn from(e: syrinx_qwen::prompt::PromptError) -> Self {
        Self(format!("prompt: {e}"))
    }
}

impl From<syrinx_qwen::tokenizer::TokenizerError> for QwenEngineError {
    fn from(e: syrinx_qwen::tokenizer::TokenizerError) -> Self {
        Self(format!("tokenizer: {e}"))
    }
}

impl From<QwenPlanError> for QwenEngineError {
    fn from(e: QwenPlanError) -> Self {
        Self(e.to_string())
    }
}

/// How this engine is conditioned — one per generation mode, and the reason a checkpoint
/// and a voice cannot be mixed and matched.
pub enum QwenVoice {
    /// `-CustomVoice`: one of the nine preset timbres (`prompt::PRESET_SPEAKERS`).
    Preset(String),
    /// `-VoiceDesign`: nothing is configured here — the instruction *is* the voice, so it
    /// arrives per request (from a cue, or from
    /// [`QwenSynth::with_instruct`](crate::qwen::QwenSynth::with_instruct)).
    Design,
    /// `-Base`: the reference clip, already reduced to what the prompt consumes. Build it
    /// with [`QwenVoice::clone_from_wav`].
    Clone {
        /// `[1, 1, speaker_enc_dim]` x-vector from the reference clip.
        xvector: Tensor,
        /// The reference's `[T][num_code_groups]` RVQ frames — empty in x-vector-only
        /// mode, populated for an in-context (ICL) clone.
        ref_frames: Vec<Vec<u32>>,
        /// The reference transcript; `Some` exactly when `ref_frames` is non-empty.
        ref_text: Option<String>,
    },
}

impl QwenVoice {
    /// The generation mode this voice belongs to.
    pub fn mode(&self) -> QwenMode {
        match self {
            Self::Preset(_) => QwenMode::CustomVoice,
            Self::Design => QwenMode::VoiceDesign,
            Self::Clone { .. } => QwenMode::VoiceClone,
        }
    }

    /// Reduce a reference clip to a [`QwenVoice::Clone`], mirroring
    /// `Qwen3TTSModel.create_voice_clone_prompt`: resample to the model rate, extract the
    /// x-vector with the checkpoint's own `speaker_encoder.*` tensors, and — when
    /// `ref_text` is given — additionally Mimi-encode the clip into reference RVQ frames
    /// (the reference's default `icl_mode`).
    ///
    /// `samples` is mono in `[-1, 1]` at `sample_rate`; `base_dir` must be a `-Base`
    /// checkpoint (the only variant carrying a speaker encoder).
    pub fn clone_from_wav(
        base_dir: impl AsRef<Path>,
        tokenizer_dir: impl AsRef<Path>,
        dev: &Device,
        samples: &[f32],
        sample_rate: u32,
        ref_text: Option<&str>,
    ) -> Result<Self, QwenEngineError> {
        let base_dir = base_dir.as_ref();
        let cfg = read_model_config(base_dir)?;
        if !cfg.variant.supports_voice_clone() {
            return Err(QwenEngineError(format!(
                "{} is a {:?} checkpoint — cloning needs a -Base one (the only variant \
                 carrying speaker_encoder.*)",
                base_dir.display(),
                cfg.variant
            )));
        }
        let dt = compute_dtype(dev);
        let wav24 = resample(samples, sample_rate, cfg.sample_rate);

        // Only the 76 `speaker_encoder.*` tensors, mmapped: the talker is loaded
        // separately, so materialising the whole multi-GB bag twice is pure waste.
        let spk_cfg = SpeakerEncoderConfig::from_model_config(&cfg);
        if spk_cfg.enc_dim != cfg.talker.hidden_size {
            return Err(QwenEngineError(format!(
                "speaker_enc_dim {} != talker hidden size {} — the x-vector cannot occupy \
                 a codec-stream position on this checkpoint",
                spk_cfg.enc_dim, cfg.talker.hidden_size
            )));
        }
        let st = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(base_dir.join("model.safetensors"))?
        };
        let mut map = std::collections::HashMap::new();
        for (name, _) in st.tensors() {
            if name.starts_with("speaker_encoder.") {
                let t = st.load(&name, dev)?;
                map.insert(name, t);
            }
        }
        // The x-vector is always f32: it is one pass over a few hundred frames and the
        // pooling is a whole-utterance sum, the reduction bf16 handles worst.
        let spk_w = Weights { map, dev: dev.clone(), dt: DType::F32 };
        let xvector = SpeakerEncoder::load(&spk_w, "speaker_encoder", spk_cfg)?
            .embed(&wav24, cfg.sample_rate)?;
        drop(spk_w);
        drop(st);

        let Some(ref_text) = ref_text else {
            return Ok(Self::Clone { xvector, ref_frames: Vec::new(), ref_text: None });
        };

        let tokenizer_dir = tokenizer_dir.as_ref();
        let ecfg = MimiEncoderConfig::from_json(&read_config_json(tokenizer_dir)?)
            .map_err(QwenEngineError)?;
        let w = load_codec_weights(tokenizer_dir, dev, dt)?;
        let enc = MimiEncoder::new(w, ecfg)?;
        let wav_t = Tensor::from_vec(wav24.clone(), wav24.len(), dev)?;
        // `[n_q][frames]` out of the encoder; the prompt wants one row per FRAME.
        let rows = enc.encode(&wav_t)?;
        if rows.len() != cfg.num_code_groups {
            return Err(QwenEngineError(format!(
                "encoder produced {} quantizer rows, the talker consumes {}",
                rows.len(),
                cfg.num_code_groups
            )));
        }
        let frames = rows[0].len();
        let ref_frames = (0..frames)
            .map(|t| rows.iter().map(|r| r[t]).collect())
            .collect();
        Ok(Self::Clone { xvector, ref_frames, ref_text: Some(ref_text.to_string()) })
    }
}

/// The 12 Hz tokenizer's decode side: the shared weight bag plus the two **parallel** RVQ
/// stacks and the waveform decoder built over it.
struct CodecStack {
    weights: Weights,
    semantic: Rvq,
    acoustic: Rvq,
    decoder: Decoder,
    dtype: DType,
}

impl CodecStack {
    /// `rows[g][t]` (one row per RVQ layer) → 24 kHz mono samples.
    fn decode(&self, rows: &[Vec<u32>]) -> Result<Vec<f32>, QwenEngineError> {
        // Split RVQ: group 0 through the semantic stack, groups 1.. through the acoustic
        // one. Decode is `rvq_first(...) + rvq_rest(...)` — the two stacks are PARALLEL,
        // not serial (the acoustic stack does not start from the semantic residual).
        let z_sem = self.semantic.decode(&self.weights, &rows[..1], self.dtype)?;
        let z_ac = self.acoustic.decode(&self.weights, &rows[1..], self.dtype)?;
        let z = (z_sem + z_ac)?;
        let wav = self.decoder.decode(&self.weights, &z)?;
        Ok(wav.flatten_all()?.to_vec1()?)
    }
}

/// A loaded Qwen3-TTS checkpoint, ready to answer `POST /v1/audio/speech`.
pub struct QwenModelEngine {
    backend: BackendId,
    model: Mutex<Qwen3Tts>,
    tok: QwenTokenizer,
    pcfg: PromptConfig,
    codec: CodecStack,
    voice: QwenVoice,
    num_code_groups: usize,
    language: String,
    params: DriveParams,
}

impl QwenModelEngine {
    /// Load a checkpoint and bind it to a voice.
    ///
    /// `talker_dir` is a published `Qwen3-TTS-12Hz-*` directory and `tokenizer_dir` the
    /// separate `Qwen3-TTS-Tokenizer-12Hz` checkpoint. The compute dtype follows the
    /// device (f32 on CPU — the parity dtype; bf16 on CUDA, to fit), exactly as
    /// `Qwen3Tts::load` chooses it.
    ///
    /// Three facts must agree or the load fails, because a mismatch would otherwise
    /// surface as a shape error deep inside `realize_plan`: the `backend` row's mode, the
    /// checkpoint's own `tts_model_type`, and the `voice`.
    pub fn load(
        backend: BackendId,
        talker_dir: impl AsRef<Path>,
        tokenizer_dir: impl AsRef<Path>,
        dev: Device,
        voice: QwenVoice,
    ) -> Result<Self, QwenEngineError> {
        let talker_dir = talker_dir.as_ref();
        let mode = QwenMode::of(backend).ok_or(QwenPlanError::NotQwen(backend))?;
        if voice.mode() != mode {
            return Err(QwenEngineError(format!(
                "backend `{}` is {mode:?} but the configured voice is {:?}",
                backend.as_str(),
                voice.mode()
            )));
        }
        let cfg_json = read_config_json(talker_dir)?;
        let cfg = Qwen3TtsConfig::from_json(&cfg_json).map_err(|e| QwenEngineError(e.to_string()))?;
        if variant_mode(cfg.variant) != mode {
            return Err(QwenEngineError(format!(
                "backend `{}` is {mode:?} but {} is a {:?} checkpoint",
                backend.as_str(),
                talker_dir.display(),
                cfg.variant
            )));
        }
        let pcfg = PromptConfig::from_json(&cfg_json)?;
        let tok = QwenTokenizer::from_dir(talker_dir)?;

        let dt = compute_dtype(&dev);
        let model = Qwen3Tts::load_with_dtype(talker_dir, dev.clone(), dt)?;

        let tokenizer_dir = tokenizer_dir.as_ref();
        let dcfg = DecoderConfig::from_json(&read_config_json(tokenizer_dir)?)
            .map_err(QwenEngineError)?;
        let weights = load_codec_weights(tokenizer_dir, &dev, dt)?;
        let semantic = Rvq::load(&weights, "decoder.quantizer.rvq_first", 1)?;
        let acoustic =
            Rvq::load(&weights, "decoder.quantizer.rvq_rest", cfg.num_code_groups - 1)?;
        let codec = CodecStack {
            weights,
            semantic,
            acoustic,
            decoder: Decoder::new("decoder", dcfg),
            dtype: dt,
        };

        Ok(Self {
            backend,
            model: Mutex::new(model),
            tok,
            pcfg,
            codec,
            voice,
            num_code_groups: cfg.num_code_groups,
            language: "english".to_string(),
            params: DriveParams::default(),
        })
    }

    /// Set the synthesis language. Must be one of the checkpoint's
    /// `PromptConfig::supported_languages()` — **there is no Polish**. Default
    /// `"english"`.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Set the **configured** sampling seed (default `0`) — the draw
    /// [`render`](QwenEngine::render) takes. `(seed, weights, prompt)` reproduces
    /// bit-for-bit.
    ///
    /// This is engine state, and therefore clobberable: see
    /// [`with_drive_params`](Self::with_drive_params). To choose a draw for **one** render
    /// without any of that hazard, pass it to
    /// [`render_seeded`](QwenEngine::render_seeded) instead, which takes the seed as an
    /// argument and leaves this value untouched.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.params.seed = seed;
        self
    }

    /// Replace the whole [`DriveParams`] (frame caps, warpers, greedy decode).
    ///
    /// **Whole** means the seed too: this discards anything
    /// [`with_seed`](Self::with_seed) set before it, so call it *first* when you use both.
    /// The hazard is confined to the *configured* seed — a seed handed to
    /// [`render_seeded`](QwenEngine::render_seeded) is an argument to that call and no
    /// builder, in any order, can overwrite it.
    pub fn with_drive_params(mut self, params: DriveParams) -> Self {
        self.params = params;
        self
    }

    /// The checkpoint this engine was loaded for.
    pub fn backend(&self) -> BackendId {
        self.backend
    }

    /// The generation mode it drives.
    pub fn mode(&self) -> QwenMode {
        self.voice.mode()
    }

    /// Wrap the engine in the [`QwenSynth`] for the backend it was loaded for, so the
    /// checkpoint id is stated once and cannot disagree with itself.
    pub fn into_synth(self) -> Result<QwenSynth<Self>, QwenPlanError> {
        let backend = self.backend;
        QwenSynth::new(backend, self)
    }

    /// Build the talker prompt for one request. The mode was already checked against the
    /// checkpoint at load time; the per-mode builders are the port's own.
    fn build_plan(
        &self,
        req: &QwenRequest<'_>,
    ) -> Result<syrinx_qwen::prompt::PromptPlan, QwenEngineError> {
        match &self.voice {
            QwenVoice::Preset(speaker) => Ok(build_custom_voice(
                &self.tok,
                &self.pcfg,
                req.text,
                speaker,
                req.instruct,
                &self.language,
                CUSTOM_VOICE_NON_STREAMING,
            )?),
            QwenVoice::Design => {
                // VoiceDesign has no other conditioning: with no instruction there is
                // nothing to synthesise a voice from, and upstream would design one from
                // an empty string.
                let instruct = req.instruct.ok_or_else(|| {
                    QwenEngineError(
                        "voice-design needs an instruction describing the voice: send a cue \
                         or configure QwenSynth::with_instruct"
                            .to_string(),
                    )
                })?;
                Ok(build_voice_design(
                    &self.tok,
                    &self.pcfg,
                    req.text,
                    instruct,
                    &self.language,
                    VOICE_DESIGN_NON_STREAMING,
                )?)
            }
            QwenVoice::Clone { ref_frames, ref_text, .. } => {
                let reference = match ref_text.as_deref() {
                    Some(rt) => CloneRef::InContext { ref_text: rt, frames: ref_frames.len() },
                    None => CloneRef::XVectorOnly,
                };
                Ok(build_voice_clone(
                    &self.tok,
                    &self.pcfg,
                    req.text,
                    reference,
                    &self.language,
                    VOICE_CLONE_NON_STREAMING,
                )?)
            }
        }
    }

    /// Render `req` under exactly `params`.
    ///
    /// `params` is passed in rather than read from `self` so that the per-render seed of
    /// [`QwenEngine::render_seeded`] has a way through that does not touch the configured
    /// [`DriveParams`] — no interior mutability, no lock ordering, nothing another
    /// concurrent request could observe half-applied.
    fn render_inner(
        &self,
        req: &QwenRequest<'_>,
        params: &DriveParams,
    ) -> Result<Vec<f32>, QwenEngineError> {
        if req.mode != self.mode() {
            return Err(QwenEngineError(format!(
                "request is {:?} but this engine drives {:?}",
                req.mode,
                self.mode()
            )));
        }
        let plan = self.build_plan(req)?;
        let (xvector, ref_frames) = match &self.voice {
            QwenVoice::Clone { xvector, ref_frames, .. } => (Some(xvector), ref_frames.as_slice()),
            _ => (None, [].as_slice()),
        };

        let mut model = match self.model.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let prompt = model.realize_plan(&plan, xvector, ref_frames)?;
        let generated = model.generate(&prompt, params)?;
        drop(model);
        if generated.frames.is_empty() {
            return Err(QwenEngineError("no frames generated".to_string()));
        }

        // In-context mode decodes `cat(ref_code, generated)` and cuts the reference's
        // share of the waveform back off, exactly as `generate_voice_clone` does — the
        // decoder is convolutional, so decoding the two halves separately is not the same
        // thing.
        let mut frames: Vec<Vec<u32>> = ref_frames.to_vec();
        frames.extend(generated.frames.iter().cloned());
        let total = frames.len();
        let rows: Vec<Vec<u32>> = (0..self.num_code_groups)
            .map(|g| frames.iter().map(|f| f[g]).collect())
            .collect();
        let mut wav = self.codec.decode(&rows)?;
        if !ref_frames.is_empty() {
            let cut = (ref_frames.len() as f64 / total as f64 * wav.len() as f64) as usize;
            wav.drain(..cut.min(wav.len()));
        }
        Ok(wav)
    }
}

impl QwenEngine for QwenModelEngine {
    fn render(&self, req: &QwenRequest<'_>) -> Result<Vec<f32>, String> {
        self.render_inner(req, &self.params).map_err(|e| e.0)
    }

    /// Render at `seed`, leaving every other knob as configured.
    ///
    /// The seed arrives as an argument and is applied to a throwaway copy of the engine's
    /// [`DriveParams`], so this render **cannot** be affected by the ordering trap on
    /// [`with_drive_params`](Self::with_drive_params): there is no builder that can reach
    /// a value that only exists for the duration of the call. `render_seeded(req, s)` is
    /// the same audio for the same `s` whatever `with_seed` / `with_drive_params` were
    /// called before it, and equals [`render`](QwenEngine::render) exactly when `s` is the
    /// configured seed. `tests/real_qwen_seed.rs` holds both properties on real weights.
    fn render_seeded(&self, req: &QwenRequest<'_>, seed: u64) -> Result<Vec<f32>, String> {
        let params = DriveParams { seed, ..self.params.clone() };
        self.render_inner(req, &params).map_err(|e| e.0)
    }

    /// True unless this engine was configured for greedy decode.
    ///
    /// `DriveParams::greedy` turns every draw into an argmax and, in the port's own words,
    /// "the run stops depending on `seed`". Reporting `true` there would promise a caller
    /// variance that a greedy engine physically cannot produce.
    fn honors_seed(&self) -> bool {
        !self.params.greedy
    }
}

/// The mode a checkpoint's own `tts_model_type` implies.
fn variant_mode(variant: QwenVariant) -> QwenMode {
    match variant {
        QwenVariant::Base => QwenMode::VoiceClone,
        QwenVariant::CustomVoice => QwenMode::CustomVoice,
        QwenVariant::VoiceDesign => QwenMode::VoiceDesign,
    }
}

/// CPU stays f32 (the parity dtype); CUDA takes bf16, to fit. Same rule as
/// `Qwen3Tts::load`.
fn compute_dtype(dev: &Device) -> DType {
    if dev.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

fn read_config_json(dir: &Path) -> Result<String, QwenEngineError> {
    std::fs::read_to_string(dir.join("config.json")).map_err(|e| {
        QwenEngineError(format!("read {}: {e}", dir.join("config.json").display()))
    })
}

fn read_model_config(dir: &Path) -> Result<Qwen3TtsConfig, QwenEngineError> {
    Qwen3TtsConfig::from_json(&read_config_json(dir)?).map_err(|e| QwenEngineError(e.to_string()))
}

fn load_codec_weights(dir: &Path, dev: &Device, dt: DType) -> Result<Weights, QwenEngineError> {
    let map = syrinx_qwen::load::load_tensors(dir, dev, dt)?;
    Ok(Weights { map, dev: dev.clone(), dt })
}
