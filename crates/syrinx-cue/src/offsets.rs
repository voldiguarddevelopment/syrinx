//! Cue → token alignment (C3.1).
//!
//! `syrinx-frontend` turns clean text into tokens; `syrinx-lm` emits those tokens in an
//! interleaved stream. A cue has to land at a **specific position** in that stream — a
//! `[whisper]` that fires one token late whispers the wrong word.
//!
//! This module is deliberately **tokenizer-agnostic**: it takes the byte spans a tokenizer
//! produced and does the arithmetic. That keeps the alignment rule — *a cue's token index
//! is the first token of its span* — gateable by golden fixtures without loading any model
//! weights, while the same code serves the real tokenizer in `syrinx-frontend`.

use crate::ir::{Cue, CueDoc};

/// The byte range one token covers in the clean text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

impl TokenSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// A cue pinned to a position in the token stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueAnchor {
    /// Index into [`CueDoc::cues`].
    pub cue: usize,
    /// The token this cue takes effect at — **the first token of its span**.
    pub token: usize,
    /// One past the last token the cue scopes; equals `token` for a point event.
    pub token_end: usize,
}

/// Byte spans of the tokens covering one text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OffsetMap {
    tokens: Vec<TokenSpan>,
}

impl OffsetMap {
    pub fn new(tokens: Vec<TokenSpan>) -> Self {
        Self { tokens }
    }

    /// Build from a tokenizer that reports each token's byte range.
    pub fn from_spans(spans: impl IntoIterator<Item = (usize, usize)>) -> Self {
        Self::new(spans.into_iter().map(|(s, e)| TokenSpan::new(s, e)).collect())
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[TokenSpan] {
        &self.tokens
    }

    /// The index of the first token at or after `byte`.
    ///
    /// A cue sitting inside a token belongs to **that** token (it cannot take effect
    /// mid-token), which is why a containing token wins over the next one.
    pub fn token_at(&self, byte: usize) -> usize {
        for (i, t) in self.tokens.iter().enumerate() {
            if byte < t.end && byte >= t.start {
                return i; // inside this token
            }
            if byte <= t.start {
                return i; // before this token starts
            }
        }
        self.tokens.len()
    }

    /// Anchor every cue in `doc` to the token stream.
    pub fn anchor(&self, doc: &CueDoc) -> Vec<CueAnchor> {
        doc.cues
            .iter()
            .enumerate()
            .map(|(i, c)| CueAnchor {
                cue: i,
                token: self.token_at(c.span.start),
                token_end: if c.is_point() {
                    self.token_at(c.span.start)
                } else {
                    self.token_at(c.span.end)
                },
            })
            .collect()
    }
}

/// One item in the interleaved stream `syrinx-lm` consumes.
///
/// Not `Eq`: a cue carries `f32` intensity, so only `PartialEq` is available.
#[derive(Debug, Clone, PartialEq)]
pub enum Emission<'a> {
    /// A cue takes effect here, before the token that follows.
    Cue(&'a Cue),
    /// A text token, by index into the token stream.
    Token(usize),
}

/// Interleave cues into the token stream at their anchored positions.
///
/// Cues are emitted **before** the token they attach to, so the LM has the control in hand
/// when it produces that token. Several cues at the same position keep their source order.
pub fn interleave<'a>(doc: &'a CueDoc, map: &OffsetMap) -> Vec<Emission<'a>> {
    let anchors = map.anchor(doc);
    let mut out = Vec::with_capacity(map.len() + anchors.len());
    let mut next = 0usize;
    for token in 0..map.len() {
        while next < anchors.len() && anchors[next].token <= token {
            out.push(Emission::Cue(&doc.cues[anchors[next].cue]));
            next += 1;
        }
        out.push(Emission::Token(token));
    }
    // Cues anchored past the last token (a trailing cue) still belong in the stream.
    while next < anchors.len() {
        out.push(Emission::Cue(&doc.cues[anchors[next].cue]));
        next += 1;
    }
    out
}

/// A deterministic whitespace tokenizer, for fixtures and for callers that have no model.
///
/// Not a substitute for the real tokenizer — it exists so the alignment rule can be
/// golden-tested without weights, and so a caller can sanity-check spans.
pub fn whitespace_spans(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                out.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
    out
}
