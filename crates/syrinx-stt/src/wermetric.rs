//! Normalized word error rate (WER) — pure Rust, no model.
//!
//! Used by the TTS intelligibility tests: transcribe a synthesized clip, then
//! `wer(expected, hypothesis)` scores how faithfully the audio carried the text.

/// Normalized word error rate between a reference and a hypothesis transcript.
///
/// Both strings are normalized (lowercased, punctuation stripped, whitespace
/// collapsed), split into words, and compared with word-level Levenshtein edit
/// distance. The result is `edits / reference_word_count` — `0.0` is a perfect
/// match, `1.0` is "every reference word wrong" (it can exceed `1.0` when the
/// hypothesis has many insertions).
///
/// Edge cases: an empty reference returns `0.0` if the hypothesis is also empty,
/// else `1.0` (all insertions, normalized by the hypothesis length).
pub fn wer(reference: &str, hypothesis: &str) -> f32 {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);

    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }

    let dist = word_levenshtein(&r, &h);
    dist as f32 / r.len() as f32
}

/// Lowercase, drop punctuation/symbols (keep alphanumerics + whitespace), and
/// split on whitespace into a word vector.
fn normalize_words(s: &str) -> Vec<String> {
    s.chars()
        .map(|c| {
            let c = c.to_ascii_lowercase();
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(|w| w.to_string())
        .collect()
}

/// Word-level Levenshtein distance (substitution / insertion / deletion all cost
/// 1) computed with a rolling two-row DP — `O(r*h)` time, `O(h)` space.
fn word_levenshtein(r: &[String], h: &[String]) -> usize {
    let n = h.len();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur = vec![0usize; n + 1];

    for (i, rw) in r.iter().enumerate() {
        cur[0] = i + 1;
        for (j, hw) in h.iter().enumerate() {
            let cost = if rw == hw { 0 } else { 1 };
            cur[j + 1] = (prev[j] + cost)
                .min(prev[j + 1] + 1) // deletion
                .min(cur[j] + 1); // insertion
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::wer;

    #[test]
    fn identical_is_zero() {
        assert_eq!(wer("the quick brown fox", "the quick brown fox"), 0.0);
    }

    #[test]
    fn punctuation_and_case_normalized() {
        assert_eq!(wer("Hello, world!", "hello world"), 0.0);
    }

    #[test]
    fn one_substitution_in_four() {
        // 1 edit over 4 reference words.
        assert!((wer("the quick brown fox", "the quick brown dog") - 0.25).abs() < 1e-6);
    }

    #[test]
    fn empty_reference() {
        assert_eq!(wer("", ""), 0.0);
        assert_eq!(wer("", "stuff"), 1.0);
    }
}
