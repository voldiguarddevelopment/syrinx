//! Regression gate: a self-closing SSML tag must close only the element **it** opened.
//!
//! Found 2026-09-12 while gating the SSML dialect against a new backend. `parse_ssml`
//! ended every self-closing tag with an unconditional `stack.pop()`, but `<break/>` pushes
//! nothing — it emits its `Pause` cue directly. So a `<break/>` inside any enclosing
//! element popped that element instead, and the document's own close tag was then rejected:
//!
//!     <speak>wait<break time="400ms"/>then go</speak>
//!       -> Err(MismatchedClose { expected: None, found: "speak" })
//!
//! Every existing fixture exercised `<break/>` at the document root, where the stack is
//! empty and the bug is invisible, so the shape that failed was the ordinary one — an SSML
//! document with a `<speak>` root and a pause in it. `<speak>` is documented as the
//! optional root (`ssml.rs` §subset), which makes this a rejection of valid input, not a
//! strictness choice.
//!
//! The fix guards the pop on `stack.last().source_start == i` — "the top of the stack is
//! the element this very tag pushed". Both sides of that guard are pinned below: a
//! self-closing tag that DID push must still close (`<prosody .../>`), and one that did
//! not must leave the stack alone (`<break/>`).

use syrinx_cue::ir::CueKind;
use syrinx_cue::{parse_ssml, SsmlError};

fn text_of(input: &str) -> String {
    parse_ssml(input)
        .unwrap_or_else(|e| panic!("{input:?} must parse, got {e:?}"))
        .text
}

fn pauses(input: &str) -> Vec<u32> {
    parse_ssml(input)
        .unwrap()
        .cues
        .iter()
        .filter_map(|c| match c.kind {
            CueKind::Pause { ms } => Some(ms),
            _ => None,
        })
        .collect()
}

/// The reported failure, exactly as written.
#[test]
fn a_break_inside_speak_does_not_close_speak() {
    let input = "<speak>wait<break time=\"400ms\"/>then go</speak>";
    assert_eq!(text_of(input), "waitthen go");
    assert_eq!(pauses(input), vec![400]);
}

/// Two breaks in a row: the second must not close `<speak>` either, and each keeps its own
/// duration. Pins that the guard is not "pop at most once".
#[test]
fn several_breaks_inside_one_element_all_survive() {
    let input = "<speak>a<break/>b<break strength=\"weak\"/>c</speak>";
    assert_eq!(text_of(input), "abc");
    assert_eq!(pauses(input), vec![400, 200]);
}

/// One level deeper: the break must not close the `<prosody>` that encloses it, so the
/// prosody cue still spans the whole of `ab`.
#[test]
fn a_break_does_not_close_the_element_that_encloses_it() {
    let input = "<speak><prosody rate=\"slow\">a<break/>b</prosody></speak>";
    let doc = parse_ssml(input).expect("must parse");
    assert_eq!(doc.text, "ab");
    let prosody: Vec<_> = doc
        .cues
        .iter()
        .filter(|c| matches!(c.kind, CueKind::Prosody { .. }))
        .collect();
    assert_eq!(prosody.len(), 1);
    assert_eq!(prosody[0].span, 0..2, "the break truncated the prosody scope");
    assert_eq!(pauses(input), vec![400]);
}

/// The OTHER side of the guard: a self-closing tag that really did open an element must
/// still be closed by it. Without this, the fix would trade one bug for an `UnclosedTag`.
#[test]
fn a_self_closing_element_that_opened_is_still_closed() {
    let input = "<speak><prosody rate=\"slow\"/>x</speak>";
    let doc = parse_ssml(input).expect("must parse");
    assert_eq!(doc.text, "x");
    let prosody: Vec<_> = doc
        .cues
        .iter()
        .filter(|c| matches!(c.kind, CueKind::Prosody { .. }))
        .collect();
    assert_eq!(prosody.len(), 1, "a self-closing prosody must still emit its cue");
    assert_eq!(prosody[0].span, 0..0, "it takes no content, so it scopes nothing");
}

/// A break at the document root — the shape every previous fixture used — must keep
/// working. This is the empty-stack side of `stack.last()`.
#[test]
fn a_break_at_the_document_root_still_works() {
    assert_eq!(text_of("a<break time=\"250ms\"/>b"), "ab");
    assert_eq!(pauses("a<break time=\"250ms\"/>b"), vec![250]);
    assert_eq!(text_of("<break/>"), "");
}

/// A NON-self-closing open tag must not be popped by this branch at all: if it were, its
/// own close tag would mismatch. Pins the `tag.self_closing &&` half of the guard.
#[test]
fn an_ordinary_open_tag_is_not_closed_by_the_self_closing_branch() {
    assert_eq!(text_of("<speak><prosody rate=\"slow\">x</prosody></speak>"), "x");
    assert_eq!(text_of("<speak><emphasis level=\"strong\">now</emphasis></speak>"), "now");
}

/// The genuine mismatch must still be an error — the fix must not turn the stack check
/// into a no-op.
#[test]
fn a_real_mismatched_close_is_still_rejected() {
    assert!(matches!(
        parse_ssml("<speak><prosody rate=\"slow\">x</emphasis></speak>"),
        Err(SsmlError::MismatchedClose { .. })
    ));
    assert!(matches!(
        parse_ssml("<speak><prosody rate=\"slow\">x</speak>"),
        Err(SsmlError::MismatchedClose { .. }) | Err(SsmlError::UnclosedTag { .. })
    ));
    assert!(matches!(parse_ssml("</speak>"), Err(SsmlError::MismatchedClose { .. })));
}

/// And the hard invariant, on every shape above: no SSML tag may survive into the text a
/// backend would speak.
#[test]
fn no_ssml_markup_survives_any_of_these_shapes() {
    for input in [
        "<speak>wait<break time=\"400ms\"/>then go</speak>",
        "<speak>a<break/>b<break strength=\"weak\"/>c</speak>",
        "<speak><prosody rate=\"slow\">a<break/>b</prosody></speak>",
        "<speak><prosody rate=\"slow\"/>x</speak>",
        "a<break time=\"250ms\"/>b",
    ] {
        let t = text_of(input);
        for tag in ["<speak", "</speak", "<prosody", "</prosody", "<emphasis", "</emphasis", "<break"]
        {
            assert!(!t.contains(tag), "LEAK: {tag:?} in {t:?} from {input:?}");
        }
        assert!(!t.contains('<') && !t.contains('>'), "LEAK: angle bracket in {t:?}");
    }
}
