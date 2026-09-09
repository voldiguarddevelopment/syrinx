//! **Pass 4 — hoist + sub-utterance splitting** (C2.3).
//!
//! A backend with [`Granularity::Utterance`] has exactly one control setting per request.
//! When the author asks for two different deliveries in one sentence, the only faithful
//! answer is to synthesise it as several requests and join them — so this pass turns one
//! `CueDoc` into N [`UtteranceSegment`]s, each with the single instruction in effect.
//!
//! The rules are pinned in the upgrade ledger (A14) so "correct prefix" is a fact rather
//! than a judgement call. In particular:
//!
//! * a segment's prefix comes from [`instruct_for`], a **total** function;
//! * **no split ever lands inside a word** — offsets snap left to a token boundary;
//! * point events do not split; they attach to the segment containing them;
//! * with splitting opted out, one segment is emitted and the losing cues are reported.

use crate::caps::{ControlCaps, Granularity, Inline};
use crate::ir::{Cue, CueKind};
use crate::instruct::InstructTable;
use crate::legacy_emotion::InstructLang;
use crate::lower::{Action, DropReason, Lowered, LoweringReport};

/// Splitting policy.
#[derive(Debug, Clone)]
pub struct SplitOptions {
    /// Allow one request to become several. Off means the author gets one delivery.
    pub allow_split: bool,
    /// Which instruct language the prefix is rendered in.
    pub lang: InstructLang,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self { allow_split: true, lang: InstructLang::En }
    }
}

/// One synthesis request: text plus the single instruction in effect for it.
#[derive(Debug, Clone, PartialEq)]
pub struct UtteranceSegment {
    pub text: String,
    /// The prefix. `None` = speak plainly.
    pub instruct: Option<String>,
    /// Point events falling inside this segment, kept for backends that can place them.
    pub cues: Vec<Cue>,
}

/// The label a cue carries.
fn label_of(kind: &CueKind) -> Option<&str> {
    match kind {
        CueKind::Emotion { label, .. } | CueKind::Style { label } | CueKind::Event { label } => {
            Some(label)
        }
        _ => None,
    }
}

/// The prefix for one cue — the total function pinned in ledger A14.
pub fn instruct_for(cue: &Cue, lang: InstructLang) -> Option<String> {
    // 1. Free text is already a natural-language instruction; pass it through untouched.
    if cue.kind == CueKind::Free {
        let raw = cue.raw.trim();
        return (!raw.is_empty()).then(|| raw.to_string());
    }
    let label = label_of(&cue.kind)?;
    // 2. A curated phrase, keyed on the CANONICAL vocab id. Synonyms were resolved to that
    //    id during parse, so a lookup here either hits the authored prose or the label has
    //    none. The table used to live in `legacy_emotion` keyed on legacy CosyVoice tag
    //    names, which missed 38 of 51 ids and sent them all to the fallback below.
    if let Some(phrase) = InstructTable::shared().phrase(label, lang) {
        return Some(phrase.to_string());
    }
    // 3. Otherwise a deterministic phrasing, so the function is total. Reached only by the
    //    `event` labels, which `pass_hoist` filters out before this on every backend that
    //    cannot express them — "Speak in a cough tone" is not an instruction anyone wants,
    //    and the article at least agrees.
    Some(match lang {
        InstructLang::En => {
            format!("Speak in {} {label} tone", crate::instruct::indefinite_article(label))
        }
        InstructLang::Zh => format!("用{label}的语气说"),
    })
}

/// Does this backend need splitting at all?
fn needs_split(caps: &ControlCaps) -> bool {
    caps.granularity == Granularity::Utterance && caps.inline == Inline::None
}

/// Snap a split offset left to a word boundary, so no token is cut in half.
///
/// Returns the offset of the whitespace run that precedes the token containing `at`. If
/// `at` is already at a boundary it is returned unchanged.
fn snap_to_word_boundary(text: &str, at: usize) -> usize {
    if at == 0 || at >= text.len() {
        return at.min(text.len());
    }
    // Already on a boundary: the byte before is whitespace.
    let before_is_space = text[..at].chars().next_back().is_some_and(char::is_whitespace);
    let here_is_space = text[at..].chars().next().is_some_and(char::is_whitespace);
    if before_is_space || here_is_space {
        return at;
    }
    // Walk left to the start of the token we are standing in.
    text[..at].rfind(char::is_whitespace).map_or(0, |i| i + 1)
}

/// Split a lowered document into per-utterance requests.
pub fn pass_hoist(
    lowered: &Lowered,
    caps: &ControlCaps,
    opts: &SplitOptions,
    report: &mut LoweringReport,
) -> Vec<UtteranceSegment> {
    let text = &lowered.text;

    // Cues that can actually steer this backend, in source order.
    let mut effective: Vec<&Cue> = lowered
        .cues
        .iter()
        .filter(|c| caps.can_express(&c.kind) || c.kind == CueKind::Free)
        .collect();
    effective.sort_by_key(|c| c.span.start);

    // A zero-span cue is a POINT — a sound that happens at an instant, which is exactly
    // what `[laughs]` or `[cough]` is. An emotion or a style is a MANNER of speaking and
    // cannot occur at an instant, so a zero-span one is not an event: it is a cue written
    // after the text it describes ("... all week. [angry]"), which the parser gives an
    // empty span because a spanning cue scopes what FOLLOWS and nothing follows.
    //
    // Classifying it as a point made it inert on every backend and recorded no drop, so a
    // trailing `[angry]` silently did nothing — found by the C4.2' runner on 2026-09-09,
    // and only because that runner asserts the difference between "caps cannot express
    // this" and "the cue vanished".
    let is_manner = |c: &Cue| matches!(c.kind, CueKind::Emotion { .. } | CueKind::Style { .. });
    let trailing_manner: Vec<&Cue> =
        lowered.cues.iter().filter(|c| c.is_point() && is_manner(c)).collect();
    let points: Vec<Cue> = lowered
        .cues
        .iter()
        .filter(|c| c.is_point() && !is_manner(c))
        .cloned()
        .collect();
    let spanning: Vec<&&Cue> = effective.iter().filter(|c| !c.is_point()).collect();

    // Word-granular or inline backends carry their cues in the text stream; one request.
    if !needs_split(caps) {
        let instruct = spanning.first().and_then(|c| instruct_for(c, opts.lang));
        return vec![UtteranceSegment {
            text: text.clone(),
            instruct: if caps.granularity == Granularity::Utterance { instruct } else { None },
            cues: lowered.cues.clone(),
        }];
    }

    // Opt-out: one delivery. The first cue wins; the rest are reported, not ignored.
    if !opts.allow_split {
        let first = spanning.first().copied();
        for c in spanning.iter().skip(1) {
            report.push(
                c.source.clone(),
                c.raw.clone(),
                Action::Dropped { reason: DropReason::Unsupported },
            );
        }
        return vec![UtteranceSegment {
            text: text.clone(),
            instruct: first.and_then(|c| instruct_for(c, opts.lang)),
            cues: points,
        }];
    }

    // Split at each conflicting cue: one whose instruction differs from the one in effect.
    let mut boundaries: Vec<(usize, Option<String>)> = vec![(0, None)];
    for c in &spanning {
        let instruct = instruct_for(c, opts.lang);
        let current = &boundaries.last().unwrap().1;
        if &instruct == current {
            continue; // not a conflict — same delivery, no reason to split
        }
        let at = snap_to_word_boundary(text, c.span.start);
        if at == boundaries.last().unwrap().0 {
            // Same starting point: this cue replaces the pending one rather than making
            // an empty segment.
            boundaries.last_mut().unwrap().1 = instruct;
        } else {
            boundaries.push((at, instruct));
        }
    }

    let mut out: Vec<UtteranceSegment> = Vec::with_capacity(boundaries.len());
    // A boundary can carve off a whitespace-only sliver (a cue sitting just inside a
    // token snaps left past the space). Dropping it would silently delete that whitespace
    // from the utterance, so it is carried forward onto the next segment instead — the
    // concatenation property in the tests is what caught this.
    let mut pending = String::new();
    let mut pending_start: Option<usize> = None;
    for (i, (start, instruct)) in boundaries.iter().enumerate() {
        let end = boundaries.get(i + 1).map_or(text.len(), |(s, _)| *s);
        let slice = &text[*start..end];
        if slice.trim().is_empty() {
            pending_start.get_or_insert(*start);
            pending.push_str(slice);
            continue;
        }
        let seg_start = pending_start.take().unwrap_or(*start);
        let mut t = String::with_capacity(pending.len() + slice.len());
        t.push_str(&pending);
        pending.clear();
        t.push_str(slice);
        out.push(UtteranceSegment {
            text: t,
            instruct: instruct.clone(),
            cues: points
                .iter()
                .filter(|p| p.span.start >= seg_start && p.span.start < end)
                .cloned()
                .collect(),
        });
    }
    // A trailing manner cue describes the delivery of the text BEFORE it, so it applies to
    // the last segment — but only if that segment has no instruction of its own. Two
    // different deliveries for one span is a conflict, and the loser is reported rather
    // than silently discarded.
    for c in &trailing_manner {
        match out.last_mut() {
            Some(seg) if seg.instruct.is_none() => {
                seg.instruct = instruct_for(c, opts.lang);
            }
            _ => report.push(
                c.source.clone(),
                c.raw.clone(),
                Action::Dropped { reason: DropReason::Unsupported },
            ),
        }
    }

    // Trailing whitespace belongs to the last segment.
    if !pending.is_empty() {
        match out.last_mut() {
            Some(last) => last.text.push_str(&pending),
            None => out.push(UtteranceSegment {
                text: std::mem::take(&mut pending),
                instruct: None,
                cues: points.clone(),
            }),
        }
    }
    if out.is_empty() && !text.trim().is_empty() {
        out.push(UtteranceSegment { text: text.clone(), instruct: None, cues: points });
    }
    out
}
