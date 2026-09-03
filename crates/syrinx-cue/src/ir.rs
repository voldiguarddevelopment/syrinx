//! The canonical cue IR (`CueDoc`), per ADR-0001 §2.
//!
//! Spans are **byte offsets into the emitted clean text**, not into the author's source.
//! That is the offset stability C1.2 requires: the clean text is what a backend receives
//! and what `syrinx-frontend` normalizes, so a cue's span stays meaningful downstream even
//! though cue markup has been removed. `Cue::source` keeps the original byte range for
//! diagnostics.

use std::ops::Range;

/// A cue's extent over the clean text. An **empty** range is a point event, which does not
/// scope the text that follows (spec §2.1).
pub type Span = Range<usize>;

/// Emphasis strength for `[emphasis]`-family cues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Reduced,
    Moderate,
    Strong,
}

/// What a cue asks for.
#[derive(Debug, Clone, PartialEq)]
pub enum CueKind {
    Emotion { label: String, intensity: f32 },
    Style { label: String },
    Event { label: String },
    Prosody { rate: Option<f32>, pitch_st: Option<f32>, volume_db: Option<f32> },
    Emphasis { level: Level },
    Pause { ms: u32 },
    SpeakerTurn { id: u32 },
    /// Unrecognised. Carried verbatim via [`Cue::raw`] so open-vocabulary backends
    /// (Fish S2) lose nothing — ADR-0001 §2.2.
    Free,
}

/// One cue, with the author's exact text always retained.
#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    pub span: Span,
    pub kind: CueKind,
    /// Exactly what the author wrote inside the delimiters, untrimmed of meaning:
    /// canonicalisation never destroys it.
    pub raw: String,
    /// Byte range of the original markup in the source, for diagnostics and `--explain`.
    pub source: Span,
}

impl Cue {
    /// A point event occupies no text.
    pub fn is_point(&self) -> bool {
        self.span.start == self.span.end
    }
}

/// A speaker turn resolved to a position in the clean text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerRef {
    pub id: u32,
    /// Byte offset in the clean text where this speaker's text begins.
    pub at: usize,
}

/// The parsed document: clean text plus the cues that apply to it.
#[derive(Debug, Clone, PartialEq)]
pub struct CueDoc {
    /// The text a backend would speak, with **all** cue markup removed.
    pub text: String,
    pub cues: Vec<Cue>,
    pub speakers: Vec<SpeakerRef>,
}

impl CueDoc {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.cues.is_empty()
    }

    /// Cues that scope text (non-empty span), in source order.
    pub fn spans(&self) -> impl Iterator<Item = &Cue> {
        self.cues.iter().filter(|c| !c.is_point())
    }

    /// Zero-width point events, in source order.
    pub fn points(&self) -> impl Iterator<Item = &Cue> {
        self.cues.iter().filter(|c| c.is_point())
    }
}
