//! Emotion tagging for the CV3 instruct path — now a **delegating adapter**.
//!
//! Per ADR-0001 **D1**, the registry, the tag parser and the segmentation moved to
//! `syrinx-cue` (`syrinx_cue::legacy_emotion`), so that crate is the single owner of
//! expressive-cue syntax (`CLAUDE.md` crate table, D6). Only the **audio** half stays
//! here: [`equal_power_crossfade`] and [`concat_crossfade`] join per-segment waveforms and
//! are not cue logic.
//!
//! # Deprecation
//!
//! The re-exported parser has legacy semantics that the strict cue rule (D5) forbids — an
//! unclosed `[` stays literal, and `TagSyntax::Parens` lets `[happy]` reach the backend
//! verbatim. They are preserved because `tests/emotion_tags.rs` is frozen and pins them;
//! see ADR-0001 §11. **New code must use `syrinx_cue::parse`**, which produces the `CueDoc`
//! IR and satisfies the hard invariant. This path exists for CV2/CV3, which are deprecated.

pub use syrinx_cue::legacy_emotion::{
    parse_tagged, EmotionInstruct, EmotionRegistry, InstructLang, Segment, TagSyntax,
};

/// Default cross-fade length (samples) between adjacent emotion segments — ~10 ms at
/// 24 kHz. Short enough not to blur the content, long enough to kill the boundary click.
pub const DEFAULT_XFADE_SAMPLES: usize = 240;

/// Equal-power cross-fade joining waveform `a` into waveform `b`.
///
/// Over the overlap region of length `L = min(fade, a.len(), b.len())`, the tail of `a` is
/// faded out by `cos(θ)` and the head of `b` faded in by `sin(θ)` (θ sweeping `0..π/2`), so
/// the summed power `cos²+sin² = 1` stays constant (no dip/bump at the seam). The result is
/// `a[..a.len()-L] ++ blend(L) ++ b[L..]`, i.e. length `a.len() + b.len() - L`. With
/// `fade == 0` (or either side empty) this is a plain concatenation.
pub fn equal_power_crossfade(a: &[f32], b: &[f32], fade: usize) -> Vec<f32> {
    let l = fade.min(a.len()).min(b.len());
    if l == 0 {
        let mut out = Vec::with_capacity(a.len() + b.len());
        out.extend_from_slice(a);
        out.extend_from_slice(b);
        return out;
    }
    let mut out = Vec::with_capacity(a.len() + b.len() - l);
    out.extend_from_slice(&a[..a.len() - l]);
    let a_tail = &a[a.len() - l..];
    for i in 0..l {
        // Centered phase so the fade is symmetric across the seam.
        let t = (i as f32 + 0.5) / l as f32;
        let theta = t * std::f32::consts::FRAC_PI_2;
        let g_out = theta.cos();
        let g_in = theta.sin();
        out.push(a_tail[i] * g_out + b[i] * g_in);
    }
    out.extend_from_slice(&b[l..]);
    out
}

/// Concatenate per-segment waveforms left-to-right with an [`equal_power_crossfade`] of
/// `fade` samples at every boundary. Empty segments are skipped; the total length is
/// `sum(len) - fade*(boundaries crossed)` (each boundary's overlap clamped to the shorter
/// side). Returns an empty `Vec` for no (non-empty) segments.
pub fn concat_crossfade(segments: &[Vec<f32>], fade: usize) -> Vec<f32> {
    let mut iter = segments.iter().filter(|s| !s.is_empty());
    let mut acc = match iter.next() {
        Some(first) => first.clone(),
        None => return Vec::new(),
    };
    for seg in iter {
        acc = equal_power_crossfade(&acc, seg, fade);
    }
    acc
}
