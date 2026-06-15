//! Cross-sentence context windowing (T-01.09).
//!
//! Assembles a bounded conditioning window around a target sentence. The input
//! is a pre-split slice of sentences; this module performs no tokenization,
//! sentence splitting, or semantic relevance weighting — the window is
//! positional only. It is clamped at both ends so it never indexes out of
//! range, and `radius == 0` yields only the current sentence.

/// A bounded conditioning window around a target sentence.
///
/// `before` and `after` are in source order and each hold at most `radius`
/// sentences (fewer when clamped at an edge).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextWindow<'a> {
    /// Up to `radius` sentences immediately preceding `current`, in source order.
    pub before: Vec<&'a str>,
    /// The target sentence at the requested index.
    pub current: &'a str,
    /// Up to `radius` sentences immediately following `current`, in source order.
    pub after: Vec<&'a str>,
}

/// Assemble the positional context window of the given `radius` around the
/// sentence at `current`. Both sides are clamped to the available range.
pub fn window<'a>(sentences: &[&'a str], current: usize, radius: usize) -> ContextWindow<'a> {
    let start = current.saturating_sub(radius);
    let before = sentences[start..current].to_vec();

    let end = (current + radius + 1).min(sentences.len());
    let after = sentences[current + 1..end].to_vec();

    ContextWindow {
        before,
        current: sentences[current],
        after,
    }
}
