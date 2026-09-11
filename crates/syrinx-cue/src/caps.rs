//! Backend control capabilities and the [`ExpressiveBackend`] trait (C2.1).
//!
//! Lowering is capability-driven: a pass never asks "is this Fish?", it asks what the
//! target can express. That keeps backend knowledge in one declarative table
//! (`caps.toml`) instead of scattered across `match` arms that drift.
//!
//! ## Accepted is not honored
//!
//! The distinction [`Support::Accepted`] vs [`Support::Honored`] is the reason this table
//! exists rather than a set of booleans. Qwen3-TTS 0.6B-CustomVoice and 1.7B-CustomVoice
//! expose an identical API; only the 1.7B obeys `instruct` (upstream gates it on a
//! substring test for the model size). A boolean "supports instruct" would have to lie
//! about one of them, and the lie would surface as a user whose emotion cue does nothing
//! with no diagnostic. `Accepted` lowers exactly like `Unsupported` and is always
//! reported.

use crate::ir::CueKind;
use crate::vocab::VocabError;
use serde::{Deserialize, Serialize};

/// How well a backend supports one control axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Support {
    /// No channel for this control. Lowering falls back and reports.
    Unsupported,
    /// Taken without error, but does not change the output. Treated as
    /// [`Support::Unsupported`] by lowering, and always reported.
    Accepted,
    /// Verified to affect the audio.
    Honored,
}

impl Support {
    /// The only question lowering should ask. `Accepted` is deliberately false: a control
    /// that is swallowed is not a control.
    pub fn is_effective(self) -> bool {
        matches!(self, Self::Honored)
    }
}

/// What inline markup the backend understands in its text stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Inline {
    /// No inline markup. Text must arrive with every cue stripped.
    None,
    /// A fixed vocabulary; anything outside it must be mapped or dropped.
    Closed,
    /// Free text is accepted, so `Free` cues pass through byte-identical.
    Open,
}

/// The finest scope at which control can be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Granularity {
    /// No control at any scope.
    None,
    /// One setting for the whole request; conflicting cues force a split (C2.3).
    Utterance,
    /// Control can change mid-utterance.
    Word,
}

/// One checkpoint's declared capabilities.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ControlCaps {
    pub id: String,
    /// Owning crate. `krate` because `crate` is a keyword.
    pub krate: String,
    /// Checkpoint directory / model name this row is true for.
    pub model: String,
    pub inline: Inline,
    pub granularity: Granularity,
    pub instruct: Support,
    pub emotion: Support,
    pub style: Support,
    pub event: Support,
    pub prosody: Support,
    pub emphasis: Support,
    pub pause: Support,
    pub speaker_turn: Support,
    /// Where the claim was verified. Required — an uncited capability is a guess.
    pub source: String,
    pub notes: String,
}

impl ControlCaps {
    /// Support for the axis a cue lives on.
    pub fn support_for(&self, kind: &CueKind) -> Support {
        match kind {
            CueKind::Emotion { .. } => self.emotion,
            CueKind::Style { .. } => self.style,
            CueKind::Event { .. } => self.event,
            CueKind::Prosody { .. } => self.prosody,
            CueKind::Emphasis { .. } => self.emphasis,
            CueKind::Pause { .. } => self.pause,
            CueKind::SpeakerTurn { .. } => self.speaker_turn,
            // Free text reaches a backend two ways: an open inline vocabulary (Fish S2),
            // or a natural-language instruction channel (Qwen CustomVoice). Missing the
            // second is what made an SSML prosody cue report as dropped on a backend that
            // could in fact have taken it as an instruction.
            CueKind::Free => match self.inline {
                Inline::Open => Support::Honored,
                _ if self.instruct.is_effective() => Support::Honored,
                _ if self.instruct == Support::Accepted => Support::Accepted,
                _ => Support::Unsupported,
            },
        }
    }

    /// Can this cue be expressed at all, as written, on this backend?
    pub fn can_express(&self, kind: &CueKind) -> bool {
        self.support_for(kind).is_effective()
    }
}

/// A backend that can be driven by lowered cues.
///
/// Deliberately narrow: C2.1 establishes only the capability handshake. The lowering
/// passes (C2.2–C2.4) consume [`ControlCaps`] and do not need anything else from a
/// backend, so nothing else belongs here yet.
pub trait ExpressiveBackend {
    fn caps(&self) -> &ControlCaps;

    fn backend_id(&self) -> &str {
        &self.caps().id
    }
}

#[derive(Deserialize)]
struct CapsFile {
    backend: Vec<ControlCaps>,
}

/// Every backend's capabilities, loaded from the embedded table.
#[derive(Debug, Clone, PartialEq)]
pub struct CapsTable {
    entries: Vec<ControlCaps>,
}

/// The capability table compiled into the binary.
pub const CAPS_TOML: &str = include_str!("../caps.toml");

impl CapsTable {
    pub fn parse(src: &str) -> Result<Self, VocabError> {
        let f: CapsFile =
            toml::from_str(src).map_err(|e| VocabError::Parse(e.to_string()))?;
        Ok(Self { entries: f.backend })
    }

    /// The embedded table.
    pub fn embedded() -> Result<Self, VocabError> {
        Self::parse(CAPS_TOML)
    }

    pub fn get(&self, id: &str) -> Option<&ControlCaps> {
        self.entries.iter().find(|c| c.id == id)
    }

    pub fn all(&self) -> &[ControlCaps] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Every checkpoint variant Syrinx can drive.
///
/// This enum is the registry the C2.1 gate checks `caps.toml` against: adding a variant
/// here without adding its row makes `tests/caps_gate.rs` fail, which is the AC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendId {
    FishS1Mini,
    FishS2Pro,
    Qwen06bBase,
    Qwen06bCustomVoice,
    Qwen17bBase,
    Qwen17bCustomVoice,
    Qwen17bVoiceDesign,
    CosyVoice2,
    CosyVoice3,
    /// **Unadopted candidate.** See the `chatterbox-turbo` row in `caps.toml`: it is here
    /// so the cue layer can describe a backend with a real event channel, not because the
    /// project has adopted it. No weights exist on this box.
    ChatterboxTurbo,
}

impl BackendId {
    /// Every variant. Kept honest by `ALL_COVERS_EVERY_VARIANT` in the gate test, which
    /// counts the variants in this source file.
    pub const ALL: &'static [BackendId] = &[
        Self::FishS1Mini,
        Self::FishS2Pro,
        Self::Qwen06bBase,
        Self::Qwen06bCustomVoice,
        Self::Qwen17bBase,
        Self::Qwen17bCustomVoice,
        Self::Qwen17bVoiceDesign,
        Self::CosyVoice2,
        Self::CosyVoice3,
        Self::ChatterboxTurbo,
    ];

    /// The `caps.toml` id. The exhaustive match means a new variant cannot compile until
    /// it is given an id here.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FishS1Mini => "fish-s1-mini",
            Self::FishS2Pro => "fish-s2-pro",
            Self::Qwen06bBase => "qwen3-0.6b-base",
            Self::Qwen06bCustomVoice => "qwen3-0.6b-customvoice",
            Self::Qwen17bBase => "qwen3-1.7b-base",
            Self::Qwen17bCustomVoice => "qwen3-1.7b-customvoice",
            Self::Qwen17bVoiceDesign => "qwen3-1.7b-voicedesign",
            Self::CosyVoice2 => "cosyvoice2",
            Self::CosyVoice3 => "cosyvoice3",
            Self::ChatterboxTurbo => "chatterbox-turbo",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|b| b.as_str() == s)
    }

    /// This backend's capabilities, from the embedded table.
    pub fn caps(self) -> Result<ControlCaps, VocabError> {
        CapsTable::embedded()?
            .get(self.as_str())
            .cloned()
            .ok_or_else(|| VocabError::Schema(format!("no caps entry for `{}`", self.as_str())))
    }
}
