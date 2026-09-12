//! What the C4.2′ runner scores against what — the segment/text pairing, frozen.
//!
//! ## Why this file exists
//!
//! `adr/0005-trailing-manner-cues.md` records a defect found on 2026-09-09: the C4.2′
//! runner read `segments.first()` where it needed *the segment carrying the instruct*,
//! and so mislabelled four of six sentinels. That is a whole class of bug — **the runner
//! holding the wrong end of a split utterance** — and the fix pinned only one instance of
//! it, in the arm that decides `not_applicable`.
//!
//! The same class has a second instance with no gate at all: the **WER reference**. The
//! runner renders `clean` (the carrier segment's text) and then scores the transcript of
//! that render against `clean`. If either half ever drifted to the *whole* utterance the
//! two would no longer describe the same audio, and the C4.2′ report would carry a WER
//! that measures a text/audio mismatch while looking exactly like an oracle error.
//!
//! The 2026-09-09 certification run made that concrete: `en-emotion-calm-mid` came back
//! `wer 0.200` where every other case was `0.000`. This file freezes the arithmetic that
//! tells the two explanations apart, so nobody has to re-derive it:
//!
//! | what was compared                        | WER   |
//! |------------------------------------------|-------|
//! | carrier text vs carrier audio (correct)   | 1 edit in 5 words = **0.200** |
//! | whole-utterance text vs carrier audio     | 4 deletions in 9 = **0.444** |
//! | carrier text vs whole-utterance audio     | 4 insertions in 5 = **0.800** |
//!
//! 0.200 is not reachable by either mismatch, so the observed value is a real one-word
//! transcription error and not a wrong-end-of-the-split bug. Freezing the three numbers
//! means a future 0.444 or 0.800 is recognisable on sight.
//!
//! Everything here is model-free: `plan()` is the deterministic half of the Qwen path and
//! `wer()` is pure Rust. No weights, no GPU, no oracle.

use syrinx_cue::BackendId;
use syrinx_serve::qwen::plan;
use syrinx_stt::wer;

/// The runner's own whitespace normalization, restated so this test scores the exact
/// string the runner hands to the engine and to `wer`.
fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The backend the C4.2′ run certifies.
const BACKEND: BackendId = BackendId::Qwen17bCustomVoice;

/// The six pre-registered sentinels of `tests/real_cue_activation_qwen.rs`, with the
/// span each cue actually scopes. Written out rather than derived: a derivation would
/// reproduce whatever the splitter does, including a wrong one.
const SENTINELS: [(&str, &str, &str); 6] = [
    (
        "en-emotion-happy-leading",
        "[happy] I really cannot believe what you just told me.",
        "I really cannot believe what you just told me.",
    ),
    (
        "en-emotion-sad-mid",
        "We should probably leave [sad] before it gets any later.",
        "before it gets any later.",
    ),
    (
        "en-emotion-angry-trailing",
        "That was the strangest thing I have seen all week. [angry]",
        "That was the strangest thing I have seen all week.",
    ),
    (
        "en-emotion-calm-mid",
        "We should probably leave [calm] before it gets any later.",
        "before it gets any later.",
    ),
    (
        "en-style-whisper-leading",
        "[whisper] I really cannot believe what you just told me.",
        "I really cannot believe what you just told me.",
    ),
    (
        "en-style-shout-mid",
        "We should probably leave [shout] before it gets any later.",
        "before it gets any later.",
    ),
];

/// The carrier text the runner renders and scores is exactly the span the cue scopes.
///
/// A leading cue scopes the whole line, a trailing manner cue scopes the line it follows
/// (ADR-0005), and a mid cue scopes only what comes **after** it. The third is the one
/// that bites: for the three `mid` sentinels the carrier is a five-word fragment, not the
/// nine-word sentence, and a runner that grabbed the sentence would be scoring a
/// transcript of the fragment against text the audio never said.
#[test]
fn the_carrier_span_is_what_the_runner_renders() {
    for (id, input, expect) in SENTINELS {
        let p = plan(BACKEND, input).unwrap_or_else(|e| panic!("{id}: plan failed: {e:?}"));
        let carriers: Vec<_> = p.segments.iter().filter(|s| s.instruct.is_some()).collect();
        assert_eq!(carriers.len(), 1, "{id}: exactly one segment must carry the instruct");
        assert_eq!(squeeze(&carriers[0].text), expect, "{id}: wrong carrier span");
    }
}

/// A mid cue really does split, and a leading/trailing one really does not.
///
/// Pinned in both directions so a splitter that stopped splitting — or started splitting
/// where it should not — is caught rather than quietly changing what C4.2′ measures.
#[test]
fn only_a_mid_cue_splits_the_utterance() {
    let n_segments = |input: &str| plan(BACKEND, input).expect("plan").segments.len();
    assert_eq!(n_segments("We should probably leave [calm] before it gets any later."), 2);
    assert_eq!(n_segments("[happy] I really cannot believe what you just told me."), 1);
    assert_eq!(n_segments("That was the strangest thing I have seen all week. [angry]"), 1);
}

/// The split loses and duplicates nothing: the segments concatenate to the lowered text.
///
/// This is the invariant a wrong-end-of-the-split bug violates first, and it holds for
/// every sentinel regardless of placement.
#[test]
fn segments_concatenate_to_the_whole_utterance() {
    for (id, input, _) in SENTINELS {
        let p = plan(BACKEND, input).expect("plan");
        let joined: String = p.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, p.text, "{id}: segments do not reconstruct the lowered text");
        // And nothing of the cue markup survives into either.
        assert!(!p.text.contains('['), "{id}: cue markup leaked into the spoken text");
    }
}

/// The three WER values that tell an oracle error apart from a text/audio mismatch.
///
/// `wer` is a pure function, so the mismatch cases can be computed exactly without ever
/// rendering audio: substitute the transcript with the text the audio *would* have
/// carried. A correctly paired reference and transcript give 0.0; a one-word slip gives
/// 0.200; the two ways of holding the wrong end of the split give 0.444 and 0.800.
#[test]
fn a_wrong_end_of_the_split_cannot_produce_the_observed_wer() {
    let whole = "We should probably leave before it gets any later.";
    let carrier = "before it gets any later.";
    assert_eq!(carrier.split_whitespace().count(), 5, "the carrier is five words");
    assert_eq!(whole.split_whitespace().count(), 9, "the utterance is nine words");

    // Correct pairing, perfect transcript.
    assert_eq!(wer(carrier, "Before it gets any later."), 0.0);
    // Correct pairing, one word misheard — the observed 0.200, from a single edit.
    assert!((wer(carrier, "Before it gets any late.") - 0.2).abs() < 1e-6);
    assert!((wer(carrier, "Before it gets any later, yeah.") - 0.2).abs() < 1e-6);
    // Whole-utterance reference against carrier audio: four deletions in nine.
    assert!((wer(whole, "Before it gets any later.") - 4.0 / 9.0).abs() < 1e-6);
    // Carrier reference against whole-utterance audio: four insertions in five.
    assert!((wer(carrier, "We should probably leave before it gets any later.") - 0.8).abs() < 1e-6);

    // The point of the table: neither mismatch can land on the value that was observed.
    assert!((4.0f32 / 9.0 - 0.2).abs() > 0.2, "0.444 is nowhere near 0.200");
    assert!((0.8f32 - 0.2).abs() > 0.2, "0.800 is nowhere near 0.200");
}

/// The plain and cued arms of a case are scored against the *same* reference string.
///
/// The runner derives one `clean` and uses it for the render, for the cued WER and for
/// the plain WER. That is what makes `wer_cued` and `wer_plain` comparable at all, and it
/// is pinned here because the planner is where it could drift: an instruct that changed
/// the spoken text would silently make the two arms score different sentences.
#[test]
fn the_instruct_never_changes_the_spoken_text() {
    let cued = plan(BACKEND, "We should probably leave [calm] before it gets any later.")
        .expect("plan");
    let plain = plan(BACKEND, "We should probably leave before it gets any later.")
        .expect("plan");
    let cued_carrier =
        squeeze(&cued.segments.iter().find(|s| s.instruct.is_some()).expect("carrier").text);
    // The plain sentence is one segment and its tail is the cued case's carrier span.
    assert_eq!(plain.segments.len(), 1, "no cue, no split");
    assert!(
        squeeze(&plain.text).ends_with(&cued_carrier),
        "the carrier span must be a literal tail of the un-cued sentence: {:?} vs {:?}",
        squeeze(&plain.text),
        cued_carrier
    );
    assert_eq!(wer(&cued_carrier, &cued_carrier), 0.0, "a reference scores zero against itself");
}
