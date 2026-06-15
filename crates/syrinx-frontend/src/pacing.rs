//! Paragraph pacing — deterministic breath-marker placement (T-01.10).
//!
//! `breath_markers` scans `text` and returns the ordered list of breath-marker
//! positions. A position is the count of words that precede the marker (a
//! global, 1-based word index): a marker at position `N` sits immediately after
//! the N-th word. Insertion is positional only — no prosodic duration is
//! assigned and no language-specific breathing model is applied (a uniform
//! word-interval policy). Two rules generate the positions:
//!
//!   1. Interior interval: within a paragraph, a breath marker falls strictly
//!      AFTER each completed interval of `interval_words` words, but only when
//!      the paragraph's word count EXCEEDS that multiple of the interval (the
//!      boundary is `>`, not `>=`). A paragraph whose word count exactly equals
//!      the interval reaches the boundary without exceeding it → no interior
//!      marker.
//!   2. Paragraph break: every boundary BETWEEN two paragraphs forces a breath
//!      marker, regardless of either paragraph's length. The end of the whole
//!      text is not a paragraph break and forces nothing.

/// Compute the ordered breath-marker positions for `text` at the given
/// words-per-breath `interval_words`. Each position is a global 1-based word
/// index: the number of words preceding the marker. Identical input yields
/// identical positions on every call.
pub fn breath_markers(text: &str, interval_words: usize) -> Vec<usize> {
    let mut markers = Vec::new();
    let mut global_words = 0usize;

    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    for (index, paragraph) in paragraphs.iter().enumerate() {
        let count = paragraph.split_whitespace().count();

        // Interior interval: a marker after each multiple of the interval that
        // the paragraph's word count strictly exceeds (`multiple < count`).
        let mut multiple = interval_words;
        while multiple < count {
            markers.push(global_words + multiple);
            multiple += interval_words;
        }

        global_words += count;

        // Paragraph break: every boundary between two paragraphs forces a
        // marker; the final paragraph has no following break.
        if index + 1 < paragraphs.len() {
            markers.push(global_words);
        }
    }

    markers
}
