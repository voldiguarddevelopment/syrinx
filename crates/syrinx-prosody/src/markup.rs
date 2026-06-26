//! Paralinguistic markup — the caller-facing control surface that inserts
//! CosyVoice2's paralinguistic markers ([breath], <strong>…</strong>,
//! [laughter], [noise], …) into the text that is handed to the synthesizer.
//!
//! This is the **first concrete differentiator** (DESIGN P5 / "paralinguistic
//! artifacts"). It needs no model change: the text tokenizer already recognises
//! these markers as atomic special-token ids (see
//! `syrinx-frontend::tokenizer` — the added-token table carries `[breath]`,
//! `<strong>`, … `[mn]`), so a marker placed in the text flows
//! `text -> text_embed -> LM.generate` and the LM can emit the corresponding
//! speech tokens. The job here is purely to give callers a clean, typed way to
//! position those markers in a string rather than hand-splicing literals.
//!
//! Scope: string assembly only. Whether a given marker audibly changes the
//! rendered speech is a property of the trained base model, exercised by the
//! on-box smoke (`examples`/root test), not asserted here.

/// A CosyVoice2 paralinguistic marker.
///
/// Each variant maps to exactly the literal the model's tokenizer was extended
/// with. [`Marker::point`] markers are single atomic tokens dropped at one point
/// in the text; [`Marker::Strong`] is a *span* emphasis that wraps a run of text
/// in `<strong>…</strong>`.
///
/// The set mirrors the markers the base tokenizer actually carries. Anything not
/// in this enum is, by construction, not a marker the base understands — so the
/// API cannot emit an unknown literal that the tokenizer would silently BPE-split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marker {
    /// An audible in-breath: `[breath]`.
    Breath,
    /// Laughter: `[laughter]`.
    Laughter,
    /// Background/onomatopoeic noise: `[noise]`.
    Noise,
    /// A cough: `[cough]`.
    Cough,
    /// A lip-smack/click: `[lipsmack]`.
    Lipsmack,
    /// Mouth noise (`[mn]` in the reference table).
    MouthNoise,
    /// Span emphasis: wraps text in `<strong>…</strong>`. Used by
    /// [`Markup::emphasize`], not as a point marker.
    Strong,
}

impl Marker {
    /// The opening literal for this marker (the whole literal for point markers,
    /// the `<strong>` open tag for [`Marker::Strong`]).
    pub fn open_literal(self) -> &'static str {
        match self {
            Marker::Breath => "[breath]",
            Marker::Laughter => "[laughter]",
            Marker::Noise => "[noise]",
            Marker::Cough => "[cough]",
            Marker::Lipsmack => "[lipsmack]",
            Marker::MouthNoise => "[mn]",
            Marker::Strong => "<strong>",
        }
    }

    /// The closing literal — non-empty only for the span marker
    /// [`Marker::Strong`] (`</strong>`); empty for every point marker.
    pub fn close_literal(self) -> &'static str {
        match self {
            Marker::Strong => "</strong>",
            _ => "",
        }
    }

    /// Whether this marker is a point marker (a single atomic insertion) rather
    /// than a span that wraps text.
    pub fn is_point(self) -> bool {
        !matches!(self, Marker::Strong)
    }
}

/// A typed builder for a marked-up utterance.
///
/// The builder accumulates plain text and paralinguistic markers in order, then
/// [`render`](Markup::render)s them to the single string the synthesizer's text
/// frontend consumes. The render is a pure concatenation: markers land exactly
/// where they were pushed.
#[derive(Debug, Clone, Default)]
pub struct Markup {
    parts: Vec<String>,
}

impl Markup {
    /// A new, empty markup builder.
    pub fn new() -> Self {
        Markup { parts: Vec::new() }
    }

    /// Append a run of plain text.
    pub fn text(mut self, s: impl Into<String>) -> Self {
        self.parts.push(s.into());
        self
    }

    /// Insert a point paralinguistic marker (e.g. [`Marker::Breath`]).
    ///
    /// A span marker ([`Marker::Strong`]) inserted here drops only its opening
    /// tag, which is rarely what a caller wants — prefer [`emphasize`](Markup::emphasize)
    /// for emphasis. The method still does the literal-correct thing (emits the
    /// `<strong>` open tag) so it never silently no-ops.
    pub fn marker(mut self, m: Marker) -> Self {
        self.parts.push(m.open_literal().to_string());
        self
    }

    /// Wrap `s` in `<strong>…</strong>` span emphasis.
    pub fn emphasize(mut self, s: impl Into<String>) -> Self {
        self.parts.push(Marker::Strong.open_literal().to_string());
        self.parts.push(s.into());
        self.parts.push(Marker::Strong.close_literal().to_string());
        self
    }

    /// Render the accumulated parts to the single text string the frontend
    /// tokenizer consumes. Markers appear exactly where they were inserted.
    pub fn render(&self) -> String {
        self.parts.concat()
    }

    /// Whether any part has been pushed.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}
