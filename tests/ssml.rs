//! Frozen RED tests for T-01.07 — parse the SSML subset.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `syrinx_frontend::ssml::parse(input: &str) -> Result<Vec<ControlEvent>, SsmlError>`
//!     — a total, recursive-descent parser over the documented SSML subset. Every
//!     input yields either `Ok(events)` (typed control events in source order) or
//!     `Err(SsmlError)`; it never panics.
//!
//! Contract (DESIGN / plan.md / spec.md): the *fixed* subset is exactly the tags
//! `break`, `emphasis`, `prosody`, `say-as`, `phoneme`, and `sub`. `break` is a
//! void element that must self-close (`<break .../>`); the other five are
//! container tags that open (`<tag ...>`), wrap inner content, and close
//! (`</tag>`). Each maps to its typed `ControlEvent` variant:
//!
//!   * `<break time="200ms"/>`              -> `[Break { ms: 200 }]`
//!   * `<emphasis level="strong">hi</...>`  -> `[Emphasis { level: Strong }, Text("hi"), EmphasisEnd]`
//!   * `<prosody rate="slow">x</prosody>`   -> `[Prosody { rate: "slow" }, Text("x"), ProsodyEnd]`
//!   * `<say-as interpret-as="...">A</...>` -> `[SayAs { interpret_as: ... }, Text("A"), SayAsEnd]`
//!   * `<phoneme ph="...">t</phoneme>`      -> `[Phoneme { ph: ... }, Text("t"), PhonemeEnd]`
//!   * `<sub alias="...">WWW</sub>`         -> `[Sub { alias: ... }, Text("WWW"), SubEnd]`
//!
//! Malformed input (an unclosed void tag, `<break time="200ms">`) and any
//! out-of-subset tag (`<blink>x</blink>`) return `Err(SsmlError)` — never a panic,
//! never a silent ignore. Plain text with no markup becomes a single `Text` event.
//!
//! Out of scope: no DTD/namespace validation or external entity resolution; tags
//! outside the fixed subset are errors, not silently dropped.
//!
//! RED: `syrinx-frontend` exposes no `ssml` module yet, so `parse`, `ControlEvent`,
//! `EmphasisLevel`, and `SsmlError` do not exist and this test target fails to
//! build — every criterion is unmet. GREEN adds the module so each assertion holds.

use syrinx_frontend::ssml::{parse, ControlEvent, EmphasisLevel, SsmlError};

/// Parse `input`, asserting it is `Ok`, and return the event sequence. A failure
/// surfaces the error context loudly rather than unwrapping blindly.
fn ok_events(input: &str) -> Vec<ControlEvent> {
    match parse(input) {
        Ok(events) => events,
        Err(_) => panic!("expected Ok for {input:?}, got Err(SsmlError)"),
    }
}

/// Parse `input`, asserting it is `Err(SsmlError)`, and return the typed error.
/// Binding the `SsmlError` here also pins that the error type exists and is the
/// declared error variant of `parse`'s `Result`.
fn err_of(input: &str) -> SsmlError {
    match parse(input) {
        Ok(events) => panic!("expected Err(SsmlError) for {input:?}, got Ok({events:?})"),
        Err(e) => e,
    }
}

// ----------------------------------------------------------------------------
// C1 — `<break time="200ms"/>` parses to exactly one typed `Break { ms: 200 }`,
// pinning the void-element event and its parsed millisecond duration. A second
// distinct value (`375ms`) guards the duration parse against off-by-one and
// arithmetic mutants that "200" (with its zeros) alone would leave alive.
// ----------------------------------------------------------------------------

/// `<break time="200ms"/>` yields a single `Break { ms: 200 }` event (criterion C1).
#[test]
fn break_void_tag_parses_to_single_break_event_200ms() {
    let events = ok_events("<break time=\"200ms\"/>");
    assert_eq!(events.len(), 1, "exactly one event expected: {events:?}");
    match &events[0] {
        ControlEvent::Break { ms } => assert_eq!(*ms, 200, "parsed break duration"),
        other => panic!("expected Break {{ ms: 200 }}, got {other:?}"),
    }
}

/// A different, all-nonzero-digit duration round-trips to its exact value,
/// pinning the numeric parse beyond the zero-laden `200` case (criterion C1).
#[test]
fn break_void_tag_parses_distinct_duration_375ms() {
    let events = ok_events("<break time=\"375ms\"/>");
    assert_eq!(events.len(), 1, "exactly one event expected: {events:?}");
    match &events[0] {
        ControlEvent::Break { ms } => assert_eq!(*ms, 375, "parsed break duration"),
        other => panic!("expected Break {{ ms: 375 }}, got {other:?}"),
    }
}

// ----------------------------------------------------------------------------
// C2 — container tags map to their typed open/Text/End variants in source order.
// Emphasis is fully pinned (level + ordering); prosody/say-as/phoneme/sub each
// map to their own typed variant on the fixed subset.
// ----------------------------------------------------------------------------

/// `<emphasis level="strong">hi</emphasis>` yields exactly
/// `[Emphasis { level: Strong }, Text("hi"), EmphasisEnd]`, in order (criterion C2).
#[test]
fn emphasis_strong_yields_open_text_end_in_order() {
    let events = ok_events("<emphasis level=\"strong\">hi</emphasis>");
    assert_eq!(events.len(), 3, "open + text + end expected: {events:?}");
    match &events[0] {
        ControlEvent::Emphasis { level } => {
            assert_eq!(*level, EmphasisLevel::Strong, "emphasis level")
        }
        other => panic!("expected Emphasis {{ level: Strong }} first, got {other:?}"),
    }
    assert_eq!(events[1], ControlEvent::Text("hi".to_string()), "inner text");
    assert_eq!(events[2], ControlEvent::EmphasisEnd, "closing event");
}

/// `<prosody rate="slow">x</prosody>` maps to the typed `Prosody` variant,
/// capturing the `rate` attribute, wrapping its text, and closing (criterion C2).
#[test]
fn prosody_rate_slow_maps_to_prosody_variant() {
    let events = ok_events("<prosody rate=\"slow\">x</prosody>");
    assert_eq!(events.len(), 3, "open + text + end expected: {events:?}");
    assert_eq!(
        events[0],
        ControlEvent::Prosody { rate: "slow".to_string() },
        "prosody open event with rate attribute",
    );
    assert_eq!(events[1], ControlEvent::Text("x".to_string()), "inner text");
    assert_eq!(events[2], ControlEvent::ProsodyEnd, "closing event");
}

/// `<say-as interpret-as="characters">A</say-as>` maps to the typed `SayAs`
/// variant, capturing its `interpret-as` attribute (criterion C2).
#[test]
fn say_as_maps_to_say_as_variant() {
    let events = ok_events("<say-as interpret-as=\"characters\">A</say-as>");
    assert_eq!(events.len(), 3, "open + text + end expected: {events:?}");
    assert_eq!(
        events[0],
        ControlEvent::SayAs { interpret_as: "characters".to_string() },
        "say-as open event with interpret-as attribute",
    );
    assert_eq!(events[1], ControlEvent::Text("A".to_string()), "inner text");
    assert_eq!(events[2], ControlEvent::SayAsEnd, "closing event");
}

/// `<phoneme ph="t">tomato</phoneme>` maps to the typed `Phoneme` variant,
/// capturing its `ph` attribute (criterion C2).
#[test]
fn phoneme_maps_to_phoneme_variant() {
    let events = ok_events("<phoneme ph=\"t\">tomato</phoneme>");
    assert_eq!(events.len(), 3, "open + text + end expected: {events:?}");
    assert_eq!(
        events[0],
        ControlEvent::Phoneme { ph: "t".to_string() },
        "phoneme open event with ph attribute",
    );
    assert_eq!(events[1], ControlEvent::Text("tomato".to_string()), "inner text");
    assert_eq!(events[2], ControlEvent::PhonemeEnd, "closing event");
}

/// `<sub alias="World Wide Web">WWW</sub>` maps to the typed `Sub` variant,
/// capturing its multi-word `alias` attribute (criterion C2).
#[test]
fn sub_maps_to_sub_variant() {
    let events = ok_events("<sub alias=\"World Wide Web\">WWW</sub>");
    assert_eq!(events.len(), 3, "open + text + end expected: {events:?}");
    assert_eq!(
        events[0],
        ControlEvent::Sub { alias: "World Wide Web".to_string() },
        "sub open event with alias attribute",
    );
    assert_eq!(events[1], ControlEvent::Text("WWW".to_string()), "inner text");
    assert_eq!(events[2], ControlEvent::SubEnd, "closing event");
}

// ----------------------------------------------------------------------------
// C3 — malformed and out-of-subset input return `Err(SsmlError)`, never a panic.
// The unclosed void tag pins the void self-close boundary against C1's valid
// `/>`; the unknown tag pins the known-subset boundary against C2's valid tags.
// ----------------------------------------------------------------------------

/// An unclosed void `break` tag (`>` instead of `/>`) returns `Err(SsmlError)`
/// without panicking — the malformed side of C1's valid self-close (criterion C3).
#[test]
fn unclosed_void_break_tag_is_error() {
    let _ = err_of("<break time=\"200ms\">");
}

/// An out-of-subset tag (`<blink>`) returns `Err(SsmlError)` — not silently
/// ignored — the unknown side of C2's known-subset tags (criterion C3).
#[test]
fn unknown_tag_is_error() {
    let _ = err_of("<blink>x</blink>");
}

// ----------------------------------------------------------------------------
// C4 — plain text with no markup becomes exactly one `Text` event, no error,
// with the full string (including interior whitespace) preserved.
// ----------------------------------------------------------------------------

/// `parse("hello world")` returns `Ok(vec![Text("hello world")])` — a single
/// text event carrying the whole string, no error (criterion C4).
#[test]
fn plain_text_becomes_single_text_event() {
    let events = ok_events("hello world");
    assert_eq!(
        events,
        vec![ControlEvent::Text("hello world".to_string())],
        "plain text is one Text event preserving the full string",
    );
}
