//! A cue written **after** the text it describes.
//!
//! `"That was the strangest thing I have seen all week. [angry]"` did nothing. The parser
//! gives a trailing cue an **empty span** — a spanning cue scopes what FOLLOWS, and nothing
//! follows — so `is_point()` returned true and it was filed as a point event. That is right
//! for `[laughs]`, which is a sound occurring at an instant. It is a category error for an
//! emotion: "angry" is a manner of speaking and cannot happen at an instant. So it never
//! reached `instruct_for`, produced no instruction, and recorded no drop.
//!
//! It was found by the C4.2' runner, and only because that runner distinguishes "the
//! backend cannot express this kind" from "the cue vanished" — `placement: trailing` is one
//! of the three placements in the frozen cue set, so the set has been silently measuring a
//! no-op on those cases.
//!
//! The rule pinned here: a zero-span **emotion or style** cue applies to the segment before
//! it. A zero-span **event** stays a point, because that is what an event is.

use syrinx_cue::caps::BackendId;
use syrinx_cue::hoist::{pass_hoist, SplitOptions};
use syrinx_cue::ir::CueKind;
use syrinx_cue::lower::{lower_full, Action};
use syrinx_cue::parse::ParseOptions;
use syrinx_cue::ssml::parse_any;
use syrinx_cue::vocab::Vocab;

fn segs(text: &str) -> (Vec<(Option<String>, String)>, usize) {
    let v = Vocab::embedded().unwrap();
    let caps = BackendId::Qwen17bCustomVoice.caps().unwrap();
    let doc = parse_any(text, &v, &ParseOptions::default()).unwrap();
    let lowered = lower_full(&doc, &caps, &v);
    let mut report = lowered.report.clone();
    let out = pass_hoist(&lowered, &caps, &SplitOptions::default(), &mut report);
    let dropped = report
        .entries
        .iter()
        .filter(|e| matches!(e.action, Action::Dropped { .. }))
        .count();
    (out.iter().map(|s| (s.instruct.clone(), s.text.clone())).collect(), dropped)
}

/// The defect itself: a trailing emotion must produce an instruction.
#[test]
fn a_trailing_emotion_applies_to_the_text_before_it() {
    let (s, _) = segs("That was the strangest thing I have seen all week. [angry]");
    assert_eq!(s.len(), 1, "one segment: {s:?}");
    assert_eq!(s[0].0.as_deref(), Some("Speak in an angry tone"), "{s:?}");
    assert!(s[0].1.starts_with("That was the strangest"), "{s:?}");
}

#[test]
fn a_trailing_style_applies_to_the_text_before_it() {
    let (s, _) = segs("Keep your voice down in here [whisper]");
    assert_eq!(s.last().unwrap().0.as_deref(), Some("Say this in a soft whisper"), "{s:?}");
}

/// Leading and mid placements are unchanged — the fix must not disturb what worked.
#[test]
fn leading_and_mid_placements_are_unchanged() {
    let (lead, _) = segs("[angry] That was the strangest thing I have seen all week.");
    assert_eq!(lead.len(), 1);
    assert_eq!(lead[0].0.as_deref(), Some("Speak in an angry tone"));

    let (mid, _) = segs("We should probably leave [angry] before it gets any later.");
    assert_eq!(mid.len(), 2, "a mid cue still splits: {mid:?}");
    assert_eq!(mid[0].0, None, "the text before the cue is unstyled");
    assert_eq!(mid[1].0.as_deref(), Some("Speak in an angry tone"));
}

/// An **event** with an empty span stays a point. This is the distinction the fix turns on,
/// and collapsing it would make `[laughs]` set an instruction, which is exactly the
/// nonsense (`"Speak in a laugh tone"`) that `instruct.toml` deliberately has no row for.
#[test]
fn a_trailing_event_is_still_a_point_and_sets_no_instruction() {
    let (s, _) = segs("He finally stopped talking [laughs]");
    assert_eq!(s.len(), 1, "{s:?}");
    assert_eq!(s[0].0, None, "an event must not become an instruction: {s:?}");
}

/// Two deliveries for one span is a conflict. The trailing cue loses to an instruction
/// already in effect, and the loss is REPORTED rather than silently discarded — which was
/// the original sin here.
#[test]
fn a_trailing_cue_that_conflicts_is_reported_not_silently_dropped() {
    let (s, dropped) = segs("[happy] What a lovely surprise [angry]");
    assert_eq!(
        s.last().unwrap().0.as_deref(),
        Some("Speak in a happy, cheerful tone"),
        "the cue already in effect wins: {s:?}"
    );
    assert!(dropped >= 1, "the losing trailing cue must be reported as dropped");
}

/// The hard invariant, restated at this seam: whatever the placement, no markup survives
/// into the spoken text.
#[test]
fn no_placement_lets_markup_reach_the_text() {
    for t in [
        "[angry] leading text",
        "mid [angry] text here",
        "trailing text [angry]",
        "trailing event [laughs]",
        "[happy] both ends [angry]",
    ] {
        let (s, _) = segs(t);
        for (_, text) in &s {
            assert!(!text.contains('['), "{t:?} leaked markup: {text:?}");
            assert!(!text.contains(']'), "{t:?} leaked markup: {text:?}");
        }
    }
}

/// Concatenating the segments must still reproduce the spoken text exactly — the property
/// that caught the whitespace-sliver bug, re-asserted because this change adds a branch
/// that touches the last segment.
#[test]
fn segments_still_concatenate_to_the_whole_text() {
    for t in [
        "trailing text [angry]",
        "We should probably leave [angry] before it gets any later.",
        "[happy] both ends [angry]",
    ] {
        let v = Vocab::embedded().unwrap();
        let doc = parse_any(t, &v, &ParseOptions::default()).unwrap();
        let (s, _) = segs(t);
        let joined: String = s.iter().map(|(_, x)| x.as_str()).collect();
        assert_eq!(joined, doc.text, "{t:?}");
    }
}

/// **An equivalent mutant, guarded at its precondition rather than forced.**
///
/// The emotion/style-vs-event partition is unobservable today: making events "manner" too
/// changes no behaviour, because every backend with `event = honored` also has inline
/// markup and word granularity, so `pass_hoist` returns before the partition is reached.
/// Mutation confirms this — that flip survives, and no test through the public API can
/// kill it.
///
/// Rather than contort a test into killing an equivalent mutant, this pins the
/// **precondition**. If a backend ever declares events honoured *and* utterance
/// granularity with no inline channel, the partition becomes live, the mutant becomes
/// killable, and this fails pointing at the reason.
#[test]
fn no_backend_needs_both_the_event_channel_and_utterance_splitting() {
    use syrinx_cue::caps::{Granularity, Inline, Support};
    let mut live = Vec::new();
    for b in BackendId::ALL {
        let Ok(c) = b.caps() else { continue };
        if c.event == Support::Honored
            && c.inline == Inline::None
            && c.granularity == Granularity::Utterance
        {
            live.push(format!("{b:?}"));
        }
    }
    assert!(
        live.is_empty(),
        "{live:?} declares honoured events with utterance splitting and no inline channel. \
         The zero-span emotion/style-vs-event partition in pass_hoist is now observable on \
         that backend and needs a behavioural test — see this test's doc comment."
    );
}
