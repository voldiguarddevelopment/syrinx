//! Qwen3-TTS on the OpenAI-compatible surface — the **model-free** half.
//!
//! `syrinx-qwen` is a verified pure-Rust port, but until now nothing could reach it over
//! HTTP: `syrinx-serve` did not even depend on the crate, while `syrinx-cue`'s `caps.toml`
//! already carried all five Qwen `[[backend]]` rows. This module closes that gap on the
//! side that needs no weights — turning a [`SpeechRequest`] into the **per-checkpoint
//! correct** sequence of synthesis requests — and [`crate::synth_qwen`] (feature `real`)
//! supplies the engine that renders them.
//!
//! # Why the split
//!
//! Everything that decides *what the model is asked to say* is deterministic and testable
//! without a GPU or a 3.9 GB checkpoint: the cue lowering, the utterance-scoped
//! instruction, the per-checkpoint capability truth, and the hard invariant that **no cue
//! markup ever reaches a backend as literal text**. Putting that behind the `real` feature
//! would have made all of it unverifiable off-box. So the planning lives here, the engine
//! is a [`QwenEngine`] trait, and the Candle implementation is the only `real`-gated part.
//!
//! # Cue handling is `syrinx-cue`'s job
//!
//! This module parses **nothing**. It calls `parse_any` → `lower_full` → `pass_hoist`, the
//! same sequence `syrinx cue` uses, and consumes the [`UtteranceSegment`]s that come back.
//! Qwen has no inline cue syntax at all (`inline = "none"` in every row), so the *only*
//! expressive channel a cue can reach is the per-utterance `instruct` string that
//! `pass_hoist` builds — one pass **after** `lower_full`. Calling `lower_full` alone, as
//! the `?explain=1` route does, would hide that channel entirely.
//!
//! # The five checkpoints are not interchangeable
//!
//! | backend id | mode | instruct |
//! |------------|------|----------|
//! | `qwen3-0.6b-base`, `qwen3-1.7b-base` | [`QwenMode::VoiceClone`] | none — clone-only, no expressive channel |
//! | `qwen3-0.6b-customvoice` | [`QwenMode::CustomVoice`] | **accepted and silently discarded** |
//! | `qwen3-1.7b-customvoice` | [`QwenMode::CustomVoice`] | honoured — describes the delivery |
//! | `qwen3-1.7b-voicedesign` | [`QwenMode::VoiceDesign`] | honoured — describes the **voice** |
//!
//! Three consequences are enforced here rather than left to the caller:
//!
//! * On a `-Base` checkpoint the instruction is dropped before the engine sees it
//!   ([`InstructEffect::Unsupported`]), so a configured default cannot leak into a
//!   clone-only path that has no slot for it.
//! * On the 0.6B CustomVoice the instruction is still **passed through** — that is what
//!   the upstream API does, and `build_custom_voice` drops it exactly as upstream does —
//!   but the plan reports [`InstructEffect::AcceptedAndDiscarded`], so "nothing happened"
//!   is visible instead of being mistaken for a null result.
//! * VoiceDesign is **never split**. On a delivery-instruct backend a second, conflicting
//!   cue means "say the rest differently", and `pass_hoist` splits the utterance into two
//!   requests. On VoiceDesign the instruction designs the *timbre*, so splitting would
//!   change **who is speaking** mid-line. `SplitOptions { allow_split: false }` keeps one
//!   voice for the whole utterance and reports the cues it could not honour.

use std::sync::Arc;

use syrinx_cue::{
    lower_full, parse_any, pass_hoist, BackendId, LoweringReport, ParseOptions, SplitOptions,
    Support, UtteranceSegment, Vocab,
};

use crate::{SpeechRequest, Synth};

/// Output sample rate of the Qwen 12 Hz tokenizer's decoder (`syrinx_qwen::SAMPLE_RATE_24K`,
/// restated here because that crate is `real`-gated and this half must build without it).
pub const QWEN_SAMPLE_RATE: u32 = 24_000;

/// Which of the three Qwen generation entry points a checkpoint uses.
///
/// Mirrors `syrinx_qwen::QwenVariant` (`Base` / `CustomVoice` / `VoiceDesign`), which
/// lives behind that crate's `real` feature; [`crate::synth_qwen::QwenModelEngine`] checks
/// the two agree against the loaded `config.json` at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenMode {
    /// `-Base`: `generate_voice_clone` — a reference clip, no instruction.
    VoiceClone,
    /// `-CustomVoice`: `generate_custom_voice` — one of nine preset timbres.
    CustomVoice,
    /// `-VoiceDesign`: `generate_voice_design` — the instruction *is* the voice.
    VoiceDesign,
}

impl QwenMode {
    /// The mode of a Qwen checkpoint, or `None` for a backend that is not Qwen.
    pub fn of(backend: BackendId) -> Option<Self> {
        match backend {
            BackendId::Qwen06bBase | BackendId::Qwen17bBase => Some(Self::VoiceClone),
            BackendId::Qwen06bCustomVoice | BackendId::Qwen17bCustomVoice => {
                Some(Self::CustomVoice)
            }
            BackendId::Qwen17bVoiceDesign => Some(Self::VoiceDesign),
            BackendId::FishS1Mini
            | BackendId::FishS2Pro
            | BackendId::CosyVoice2
            | BackendId::CosyVoice3 => None,
        }
    }
}

/// What actually happens to an instruction on this checkpoint — the distinction
/// `caps.toml` exists to carry, restated where the request is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructEffect {
    /// No instruction channel at all (`-Base`). The instruction is dropped here.
    Unsupported,
    /// Taken without error and silently discarded (0.6B CustomVoice). Passed to the
    /// engine, because that is the real API's behaviour, and reported.
    AcceptedAndDiscarded,
    /// Verified to affect the audio (1.7B CustomVoice, VoiceDesign).
    Honored,
}

impl InstructEffect {
    /// Read the effect off a checkpoint's declared `instruct` support.
    pub fn of(support: Support) -> Self {
        match support {
            Support::Unsupported => Self::Unsupported,
            Support::Accepted => Self::AcceptedAndDiscarded,
            Support::Honored => Self::Honored,
        }
    }

    /// Will the instruction reach the model at all?
    pub fn reaches_model(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// Why a request could not be planned.
#[derive(Debug, Clone, PartialEq)]
pub enum QwenPlanError {
    /// The backend id is real but is not one of the five Qwen checkpoints.
    NotQwen(BackendId),
    /// The embedded capability/vocabulary tables could not be read.
    Tables(String),
    /// The input is not valid cue syntax (e.g. bracket cues and SSML in one document).
    Cue(String),
}

impl std::fmt::Display for QwenPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotQwen(b) => write!(f, "`{}` is not a Qwen3-TTS backend", b.as_str()),
            Self::Tables(e) => write!(f, "cue tables unavailable: {e}"),
            Self::Cue(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for QwenPlanError {}

/// One utterance's worth of synthesis requests for one Qwen checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenPlan {
    /// The checkpoint this plan is for.
    pub backend: BackendId,
    /// Its generation mode.
    pub mode: QwenMode,
    /// What the checkpoint does with an instruction.
    pub instruct_effect: InstructEffect,
    /// The whole utterance's clean text — cue markup already stripped.
    pub text: String,
    /// The per-utterance requests `pass_hoist` produced. Exactly one unless a conflicting
    /// cue forced a split (never more than one on VoiceDesign; see the module docs).
    pub segments: Vec<UtteranceSegment>,
    /// Everything lowering **and** hoisting did, in source order.
    pub report: LoweringReport,
}

/// Plan `input` for `backend`: lower its cues and hoist them into per-utterance
/// instructions, exactly as `syrinx cue --backend <id>` reports.
pub fn plan(backend: BackendId, input: &str) -> Result<QwenPlan, QwenPlanError> {
    let mode = QwenMode::of(backend).ok_or(QwenPlanError::NotQwen(backend))?;
    let caps = backend.caps().map_err(|e| QwenPlanError::Tables(e.to_string()))?;
    let vocab = Vocab::embedded().map_err(|e| QwenPlanError::Tables(e.to_string()))?;

    // One entry point for both authoring syntaxes; a mixed document is a hard error, not
    // a guess. `syrinx-serve` parses nothing itself (CLAUDE.md / ADR-0001 D6).
    let doc = parse_any(input, &vocab, &ParseOptions::default())
        .map_err(|e| QwenPlanError::Cue(e.to_string()))?;
    let lowered = lower_full(&doc, &caps, &vocab);

    let mut report = lowered.report.clone();
    let opts = SplitOptions {
        // VoiceDesign's instruction describes the voice, not the delivery: splitting would
        // re-design the timbre mid-utterance.
        allow_split: mode != QwenMode::VoiceDesign,
        ..SplitOptions::default()
    };
    let segments = pass_hoist(&lowered, &caps, &opts, &mut report);

    Ok(QwenPlan {
        backend,
        mode,
        instruct_effect: InstructEffect::of(caps.instruct),
        text: lowered.text,
        segments,
        report,
    })
}

/// One synthesis call the engine must render: clean text plus the single instruction in
/// effect for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenRequest<'a> {
    /// Which generation entry point to use.
    pub mode: QwenMode,
    /// The text to speak. **Never** carries cue markup.
    pub text: &'a str,
    /// The utterance-scoped instruction, already filtered by [`InstructEffect`].
    pub instruct: Option<&'a str>,
    /// What this checkpoint will do with `instruct`.
    pub instruct_effect: InstructEffect,
}

/// The thing that actually turns a [`QwenRequest`] into audio.
///
/// Split out of [`QwenSynth`] so the request-building half above is testable without
/// weights. The production implementation is
/// [`crate::synth_qwen::QwenModelEngine`] (feature `real`).
pub trait QwenEngine: Send + Sync {
    /// Render one request to 24 kHz mono `f32` samples, or describe why it failed.
    fn render(&self, req: &QwenRequest<'_>) -> Result<Vec<f32>, String>;
}

/// The Qwen [`Synth`]: plans a [`SpeechRequest`] for one checkpoint and renders every
/// resulting segment through a [`QwenEngine`], concatenated into one 24 kHz WAV body.
///
/// Bound to **one** checkpoint, like [`crate::RealSynth`] is bound to one reference voice:
/// the OpenAI request carries no checkpoint selector, and the five Qwen checkpoints are
/// separate multi-gigabyte models. `req.model` / `req.voice` are advisory.
pub struct QwenSynth<E> {
    backend: BackendId,
    engine: E,
    default_instruct: Option<String>,
}

impl<E: QwenEngine> QwenSynth<E> {
    /// Bind an engine to a Qwen checkpoint. Fails for a non-Qwen backend id.
    pub fn new(backend: BackendId, engine: E) -> Result<Self, QwenPlanError> {
        QwenMode::of(backend).ok_or(QwenPlanError::NotQwen(backend))?;
        Ok(Self { backend, engine, default_instruct: None })
    }

    /// The instruction to use for a segment that carries no cue-derived one.
    ///
    /// On VoiceDesign this is the **voice description** — the checkpoint's only way to
    /// know what it is designing — and a cue-derived instruction replaces it rather than
    /// being appended to it, because on that checkpoint both strings describe the same
    /// thing. On a `-Base` checkpoint it is dropped like any other instruction.
    pub fn with_instruct(mut self, instruct: impl Into<String>) -> Self {
        self.default_instruct = Some(instruct.into());
        self
    }

    /// The checkpoint this synth is bound to.
    pub fn backend(&self) -> BackendId {
        self.backend
    }

    /// Plan `input` for this synth's checkpoint (what `synthesize` will render).
    pub fn plan(&self, input: &str) -> Result<QwenPlan, QwenPlanError> {
        plan(self.backend, input)
    }

    /// Plan and render, returning the concatenated 24 kHz mono samples.
    fn render_all(&self, input: &str) -> Result<Vec<f32>, String> {
        let plan = self.plan(input).map_err(|e| e.to_string())?;
        let mut out: Vec<f32> = Vec::new();
        for seg in &plan.segments {
            // A checkpoint with no instruction channel never sees one — not the
            // cue-derived string, not the configured default.
            let instruct = match plan.instruct_effect {
                InstructEffect::Unsupported => None,
                _ => seg.instruct.as_deref().or(self.default_instruct.as_deref()),
            };
            let req = QwenRequest {
                mode: plan.mode,
                text: &seg.text,
                instruct,
                instruct_effect: plan.instruct_effect,
            };
            out.extend(self.engine.render(&req)?);
        }
        Ok(out)
    }
}

impl<E: QwenEngine> Synth for QwenSynth<E> {
    fn synthesize(&self, req: &SpeechRequest) -> Vec<u8> {
        // The `Synth` trait has no error channel: an empty body is the agreed signal that
        // the handler should answer a typed 500 rather than a 200 with no audio.
        let samples = match self.render_all(&req.input) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("syrinx-serve: qwen synthesis failed: {e}");
                return Vec::new();
            }
        };
        if samples.is_empty() {
            eprintln!("syrinx-serve: qwen synthesis produced no audio");
            return Vec::new();
        }
        encode_wav24(&samples)
    }
}

/// Build the audio router wired to a [`QwenSynth`], so `POST /v1/audio/speech` returns
/// Qwen3-TTS audio. Thin convenience over [`crate::router_with_synth`], mirroring
/// [`crate::router_with_cv3_synth`].
pub fn router_with_qwen_synth<E: QwenEngine + 'static>(synth: QwenSynth<E>) -> axum::Router {
    crate::router_with_synth(Arc::new(synth))
}

/// Encode 24 kHz mono `f32` samples (clamped to `[-1, 1]`) as a complete 16-bit PCM WAV.
///
/// [`crate::wavio::encode_wav_24k`] does the same job with `hound`, but `hound` is pulled
/// in only by the `real` feature and this path must work in the model-free build.
pub fn encode_wav24(samples: &[f32]) -> Vec<u8> {
    const CHANNELS: u16 = 1;
    const BITS: u16 = 16;
    let block_align = CHANNELS * (BITS / 8);
    let byte_rate = QWEN_SAMPLE_RATE * block_align as u32;
    let data_len = (samples.len() * 2) as u32;

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes()); // RIFF size = 44 - 8 + data
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&CHANNELS.to_le_bytes());
    out.extend_from_slice(&QWEN_SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
