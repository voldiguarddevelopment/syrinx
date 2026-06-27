//! Reusable **voice bundles**, a directory-backed **voice library**, and embedding-space
//! **voice manipulation** (blend / interpolate / arithmetic / attributes) on top of the
//! CV2 / CV3 zero-shot synthesizers.
//!
//! ## What a "voice" actually is
//! A CosyVoice zero-shot voice is fully captured by what
//! [`Cv3Synthesizer::prompt_cond`](crate::synth_cv3::Cv3Synthesizer::prompt_cond) /
//! [`Synthesizer::prompt_cond`](crate::synth::Synthesizer::prompt_cond) extract from a
//! reference clip:
//!   * the CAM++ **speaker embedding** `[1, 192]` — the timbre identity,
//!   * the **prompt mel** `prompt_feat` `[1, 2·token_len, 80]`,
//!   * the **prompt speech tokens** `[token_len]` (i64), and
//!   * the **prompt transcript** (`prompt_text`).
//!
//! [`Voice`] bundles exactly those four, so one frontend extraction (fbank → CAM++,
//! the v2/v3 speech tokenizer, the 24 kHz prompt-mel) feeds **many** syntheses with no
//! re-run of the frontend per call — see [`Voice::from_reference`] and the synthesizers'
//! `synthesize_with_voice` methods.
//!
//! ## The manipulation surface, and what is honest about it
//! [`Voice::blend`] / [`Voice::interpolate`] / [`VoiceArithmetic`] / [`Voice::with_attribute`]
//! all operate **only on the speaker embedding** (the timbre identity), because that is
//! the one piece of conditioning that is a single fixed-length vector comparable across
//! clips. The CAM++ embedding is compared by **cosine** (magnitude-invariant), so every
//! op L2-normalizes its operands and re-normalizes its result.
//!
//! The other three pieces (`prompt_feat`, `prompt_token`, `prompt_text`) are
//! **time-aligned sequences tied to one specific clip** and CANNOT be averaged across
//! clips — there is no meaningful "half of clip A's mel plus half of clip B's mel". So
//! every manipulation **carries a chosen base voice's** `prompt_feat` / `prompt_token` /
//! `prompt_text` unchanged and only mixes the embedding. The synthesized result therefore
//! speaks with the blended *timbre* but is prosodically anchored to the base clip. This is
//! a deliberate, documented limitation, not a bug.

mod library;
mod manip;

pub use library::{VoiceLibrary, VoiceLibraryError};
pub use manip::VoiceArithmetic;

use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::synth::{PromptCond, SynthError};

/// A reusable, serializable zero-shot voice: the prompt-side conditioning extracted from
/// one reference clip, reusable across many syntheses.
///
/// Construct one with [`Voice::from_reference`] (extracts via the synthesizer's
/// `prompt_cond` once), derive new ones with the manipulation methods
/// ([`Voice::blend`] / [`Voice::interpolate`] / [`VoiceArithmetic`] /
/// [`Voice::with_attribute`]), persist them with a [`VoiceLibrary`], and synthesize with
/// `Cv3Synthesizer::synthesize_with_voice` / `Synthesizer::synthesize_with_voice`.
#[derive(Debug, Clone)]
pub struct Voice {
    /// Human-readable identifier (also the [`VoiceLibrary`] file stem).
    pub name: String,
    /// CAM++ speaker x-vector `[1, 192]` — the timbre identity (compared by cosine).
    pub speaker_embedding: Tensor,
    /// Prompt mel `[1, 2·token_len, 80]` (frame-major), tied to the source clip.
    pub prompt_feat: Tensor,
    /// Prompt speech-token ids `[token_len]` (i64), tied to the source clip.
    pub prompt_token: Vec<i64>,
    /// Prompt transcript — the text spoken in the reference clip.
    pub prompt_text: String,
    /// Optional provenance string (e.g. the source clip path); `None` for a derived voice.
    pub source: Option<String>,
}

/// The serde sidecar for a persisted [`Voice`]: everything except the two tensors (which
/// go to a companion safetensors file). Round-trips byte-exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VoiceMeta {
    pub name: String,
    pub prompt_text: String,
    pub prompt_token: Vec<i64>,
    pub source: Option<String>,
}

/// Synthesizers from which a [`Voice`] can be extracted. Implemented for both the CV2
/// [`Synthesizer`](crate::synth::Synthesizer) and the CV3
/// [`Cv3Synthesizer`](crate::synth_cv3::Cv3Synthesizer); both expose the same
/// [`PromptCond`] / [`SynthError`] contract, so [`Voice::from_reference`] is generic over
/// them.
pub trait VoiceExtract {
    /// Run the frontend half and return the prompt-side conditioning for a reference clip.
    ///
    /// `prompt_text` is the reference transcript; `ref_wav_16k` / `ref_wav_24k` are the
    /// already-resampled mono reference waveforms (resampling is the caller's job, exactly
    /// as for `prompt_cond`). The returned [`PromptCond`]'s `text_token` field is unused by
    /// a [`Voice`] (it is `tts_text`-specific); only the prompt-side fields are kept.
    fn extract_prompt_cond(
        &mut self,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<PromptCond, SynthError>;
}

impl VoiceExtract for crate::synth::Synthesizer {
    fn extract_prompt_cond(
        &mut self,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<PromptCond, SynthError> {
        // tts_text is empty: only the prompt-side fields are read out (the text tokens are
        // re-derived per synthesis from the real tts text), and the prompt-side frontend
        // (CAM++ embedding / prompt_token / prompt_feat) does not depend on tts_text.
        self.prompt_cond("", prompt_text, ref_wav_16k, ref_wav_24k)
    }
}

impl VoiceExtract for crate::synth_cv3::Cv3Synthesizer {
    fn extract_prompt_cond(
        &mut self,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<PromptCond, SynthError> {
        self.prompt_cond("", prompt_text, ref_wav_16k, ref_wav_24k)
    }
}

impl Voice {
    /// Extract a reusable [`Voice`] from a reference clip via the synthesizer's
    /// `prompt_cond`, **once** — works for both CV2 and CV3 (the [`VoiceExtract`] bound).
    ///
    /// The clip's `source` is recorded as `None` (use [`Voice::with_source`] to attach a
    /// provenance string). The 16 kHz / 24 kHz reference waveforms must already be
    /// resampled mono (the caller's job, as for `prompt_cond`).
    pub fn from_reference<S: VoiceExtract>(
        synth: &mut S,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        prompt_text: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Voice, SynthError> {
        let prompt_text = prompt_text.into();
        let cond = synth.extract_prompt_cond(&prompt_text, ref_wav_16k, ref_wav_24k)?;
        let prompt_token = cond.prompt_token.flatten_all()?.to_vec1::<i64>()?;
        Ok(Voice {
            name: name.into(),
            speaker_embedding: cond.spk_embedding,
            prompt_feat: cond.prompt_feat,
            prompt_token,
            prompt_text,
            source: None,
        })
    }

    /// Builder-style: attach a provenance string (e.g. the source clip path).
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Builder-style: rename the voice (e.g. when deriving a blend).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The speaker embedding as a flat `Vec<f32>` (length 192). A convenience for callers
    /// computing cosine similarity / inspecting the timbre vector.
    pub fn embedding_vec(&self) -> Result<Vec<f32>, SynthError> {
        Ok(self.speaker_embedding.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Build the serde sidecar (everything but the tensors).
    pub(crate) fn meta(&self) -> VoiceMeta {
        VoiceMeta {
            name: self.name.clone(),
            prompt_text: self.prompt_text.clone(),
            prompt_token: self.prompt_token.clone(),
            source: self.source.clone(),
        }
    }
}
