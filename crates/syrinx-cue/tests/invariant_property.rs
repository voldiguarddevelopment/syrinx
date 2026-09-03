//! The **hard invariant**, property-tested (session rule; ADR-0001 §5):
//!
//! > No bracket cue text and no speaker token may ever reach a backend as literal text.
//!
//! A failure here is a release blocker, not a bug to trim. Also carries the C1.2
//! round-trip property for the Fish path.
//!
//! No external proptest dependency: a deterministic xorshift generator composes inputs
//! from cue-ish and prose-ish fragments, which gives reproducible counterexamples and
//! keeps the dependency budget at zero for this file.

use syrinx_cue::ir::CueKind;
use syrinx_cue::{parse, ParseOptions, Vocab};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next() % xs.len() as u64) as usize]
    }
}

/// Fragments deliberately include every shape that has ever leaked in a TTS pipeline:
/// real cues, unknown cues, stage directions, escapes, speaker tokens, stray brackets.
const FRAGMENTS: &[&str] = &[
    "[happy]", "[sad]", "[whisper]", "[laughs]", "[sigh]", "[pause 300ms]", "[emphasis]",
    "[/emphasis]", "[very excited]", "[wibble]", "[professional broadcast tone]",
    "<|speaker:0|>", "<|speaker:3|>", "[speaker 2]",
    "hello", " world", " and then. ", "! ", "? ", "\n",
    "[He turns, slowly]", "[]", "[   ]", "unclosed [tag", "a ] stray", "[a [b] c]",
    r"\[literal\]", r"\]", "grüße", "日本語", "🎉",
];

fn compose(seed: u64, n: usize) -> String {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| *r.pick(FRAGMENTS)).collect::<Vec<_>>().concat()
}

/// The invariant. Applies to the clean text for EVERY backend, because the clean text is
/// what a backend with `Inline::None` receives verbatim.
fn assert_no_leak(input: &str, text: &str) {
    // Any bracket surviving into clean text must have come from an escape in the source.
    let escaped_open = input.matches(r"\[").count();
    let escaped_close = input.matches(r"\]").count();
    let open = text.matches('[').count();
    let close = text.matches(']').count();
    assert!(
        open <= escaped_open,
        "LEAK: {open} '[' in clean text but only {escaped_open} escaped in source\n  in:  {input:?}\n  out: {text:?}"
    );
    assert!(
        close <= escaped_close,
        "LEAK: {close} ']' in clean text but only {escaped_close} escaped in source\n  in:  {input:?}\n  out: {text:?}"
    );
    assert!(
        !text.contains("<|speaker"),
        "LEAK: speaker token reached the clean text\n  in:  {input:?}\n  out: {text:?}"
    );
}

#[test]
fn hard_invariant_no_cue_markup_ever_reaches_the_clean_text() {
    let v = Vocab::embedded().unwrap();
    let opts = ParseOptions::default();
    for seed in 1..=2000u64 {
        let n = 1 + (seed as usize % 12);
        let input = compose(seed, n);
        let doc = parse(&input, &v, &opts);
        assert_no_leak(&input, &doc.text);
    }
}

#[test]
fn hard_invariant_holds_for_the_fixture_corpus_too() {
    let v = Vocab::embedded().unwrap();
    let opts = ParseOptions::default();
    // Prose that merely looks cue-ish must not leak either.
    for input in [
        "[He turns to the window, slowly] he said.",
        "array[0] and array[1] and array[2].",
        "a ] b [ c",
        "[[nested]]",
        "<|speaker:notanumber|> hello",
        "<|speaker:1|",
    ] {
        let doc = parse(input, &v, &opts);
        assert_no_leak(input, &doc.text);
    }
}

/// C1.2 AC: `serialize(parse(x))` round-trips for the Fish path.
///
/// Fish S2 is `Inline::Open`, so a cue re-serialises as `[raw]` at its span start. The
/// round-trip property is that parsing that serialisation yields the same clean text and
/// the same cue sequence — i.e. the IR is a faithful, lossless carrier for the open path.
/// Literal brackets that survived into the clean text MUST be re-escaped, or the
/// serialisation reparses them as cues. This is the serialiser's half of the escape
/// contract.
fn esc(s: &str) -> String {
    s.replace('[', "\\[").replace(']', "\\]")
}

fn serialize_fish(doc: &syrinx_cue::CueDoc) -> String {
    let mut out = String::new();
    let mut cursor = 0usize;
    let mut cues: Vec<_> = doc.cues.iter().collect();
    cues.sort_by_key(|c| c.span.start);
    for c in cues {
        let at = c.span.start.min(doc.text.len());
        if at > cursor {
            out.push_str(&esc(&doc.text[cursor..at]));
            cursor = at;
        }
        match &c.kind {
            CueKind::SpeakerTurn { id } => out.push_str(&format!("<|speaker:{id}|>")),
            _ => out.push_str(&format!("[{}]", c.raw)),
        }
    }
    out.push_str(&esc(&doc.text[cursor..]));
    out
}

#[test]
fn fish_path_round_trips() {
    let v = Vocab::embedded().unwrap();
    let opts = ParseOptions::default();
    for seed in 1..=800u64 {
        let input = compose(seed, 1 + (seed as usize % 8));
        let a = parse(&input, &v, &opts);
        let round = serialize_fish(&a);
        let b = parse(&round, &v, &opts);
        assert_eq!(
            a.text, b.text,
            "round-trip changed the clean text\n  src:   {input:?}\n  ser:   {round:?}"
        );
        let ka: Vec<_> = a.cues.iter().map(|c| (&c.kind, &c.raw)).collect();
        let kb: Vec<_> = b.cues.iter().map(|c| (&c.kind, &c.raw)).collect();
        assert_eq!(ka, kb, "round-trip changed the cue sequence\n  src: {input:?}\n  ser: {round:?}");
    }
}

// ---------------------------------------------------------------- C1.3: the SSML path

/// SSML fragments, including every rejection shape, so the generator explores the error
/// paths as hard as the happy ones.
const SSML_FRAGMENTS: &[&str] = &[
    "<speak>", "</speak>", "<prosody rate=\"slow\">", "<prosody pitch=\"+2st\">",
    "<prosody volume=\"loud\">", "</prosody>", "<emphasis>", "<emphasis level=\"strong\">",
    "</emphasis>", "<break time=\"400ms\"/>", "<break strength=\"weak\"/>", "<break/>",
    "hello", " world", " and then. ", "! ", "\n", "5 &lt; 6", "&amp;", "x < y",
    "<voice name=\"bob\">", "</voice>", "<prosody rate=\"quickly\">", "<unclosed",
];

fn compose_ssml(seed: u64, n: usize) -> String {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| *r.pick(SSML_FRAGMENTS)).collect::<Vec<_>>().concat()
}

/// No SSML tag may ever be spoken either. The invariant is about *markup*, and the
/// dialect it was written in does not change what a backend must not receive.
fn assert_no_ssml_leak(input: &str, text: &str) {
    for tag in ["<speak", "</speak", "<prosody", "</prosody", "<emphasis", "</emphasis", "<break"] {
        assert!(
            !text.contains(tag),
            "LEAK: {tag:?} reached the clean text\n  in:  {input:?}\n  out: {text:?}"
        );
    }
}

#[test]
fn hard_invariant_holds_on_the_ssml_path_including_every_rejection() {
    let mut parsed = 0usize;
    let mut rejected = 0usize;
    for seed in 1..=1000u64 {
        for n in [1usize, 3, 6, 12] {
            let input = compose_ssml(seed.wrapping_mul(0x9E37_79B9), n);
            match syrinx_cue::parse_ssml(&input) {
                // A document that parses must be clean...
                Ok(doc) => {
                    assert_no_ssml_leak(&input, &doc.text);
                    parsed += 1;
                }
                // ...and one that does not must produce NO text at all, rather than a
                // partial render. Erroring is the only safe answer: half-applied prosody
                // is worse than a rejection the author can see.
                Err(_) => rejected += 1,
            }
        }
    }
    // Both arms must actually be exercised, or this test proves nothing.
    assert!(parsed > 100, "generator produced too few valid documents: {parsed}");
    assert!(rejected > 100, "generator produced too few rejections: {rejected}");
}

#[test]
fn a_mixed_document_is_always_rejected_never_half_parsed() {
    // Cross the two generators: any input carrying both dialects must be a hard error.
    for seed in 1..=400u64 {
        let a = compose(seed, 3);
        let b = compose_ssml(seed.wrapping_mul(31), 3);
        for input in [format!("{a}{b}"), format!("{b}{a}")] {
            let has_bracket = {
                // unescaped bracket markup present in the source?
                let mut found = false;
                let bytes = input.as_bytes();
                let mut i = 0;
                while i < bytes.len() {
                    match bytes[i] {
                        b'\\' => i += 2,
                        b'[' | b']' => { found = true; break }
                        b'<' if input[i..].starts_with("<|speaker") => { found = true; break }
                        _ => i += 1,
                    }
                }
                found
            };
            let has_ssml = ["<speak", "</speak", "<prosody", "</prosody", "<emphasis",
                            "</emphasis", "<break", "<voice", "</voice"]
                .iter().any(|t| input.contains(t));
            if has_bracket && has_ssml {
                assert!(
                    matches!(syrinx_cue::parse_ssml(&input), Err(syrinx_cue::SsmlError::MixedSyntax { .. })),
                    "mixed document was not rejected: {input:?}"
                );
            }
        }
    }
}
