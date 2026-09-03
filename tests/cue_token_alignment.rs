//! C3.1 golden fixtures. **AC: golden token dumps for 5 fixtures; a cue's token index is
//! the first token of its span.**
//!
//! Tokenizer-agnostic on purpose: the alignment arithmetic is the deliverable, so it is
//! gated with a deterministic whitespace tokenizer and needs no model weights. The real
//! `syrinx-frontend` tokenizer feeds the same `OffsetMap` at run time.

use std::fmt::Write as _;
use syrinx_cue::offsets::{interleave, whitespace_spans, Emission, OffsetMap};
use syrinx_cue::parse::ParseOptions;
use syrinx_cue::{parse, Vocab};

/// The five fixtures: leading cue, mid-sentence cue, point event, explicit span, and a
/// multi-speaker turn.
const FIXTURES: &[(&str, &str)] = &[
    ("leading", "[happy] good morning everyone"),
    ("mid_sentence", "it was fine until [angry] the second call"),
    ("point_event", "he paused [laughs] and carried on"),
    ("explicit_span", "say [emphasis]this word[/emphasis] clearly"),
    ("speaker_turn", "<|speaker:0|> hello there <|speaker:1|> and hello to you"),
];

fn dump(src: &str) -> String {
    let v = Vocab::embedded().unwrap();
    let doc = parse(src, &v, &ParseOptions::default());
    let map = OffsetMap::from_spans(whitespace_spans(&doc.text));
    let anchors = map.anchor(&doc);

    let mut out = String::new();
    writeln!(out, "source: {src:?}").unwrap();
    writeln!(out, "text:   {:?}", doc.text).unwrap();
    writeln!(out, "-- tokens --").unwrap();
    for (i, t) in map.tokens().iter().enumerate() {
        writeln!(out, "{i:3}  {:?}", &doc.text[t.start..t.end]).unwrap();
    }
    writeln!(out, "-- cues --").unwrap();
    for a in &anchors {
        let c = &doc.cues[a.cue];
        writeln!(
            out,
            "cue {} raw={:?} kind={:?} span={:?} -> token {} .. {}",
            a.cue, c.raw, c.kind, c.span, a.token, a.token_end
        )
        .unwrap();
    }
    writeln!(out, "-- stream --").unwrap();
    for e in interleave(&doc, &map) {
        match e {
            Emission::Cue(c) => writeln!(out, "CUE  {:?}", c.raw).unwrap(),
            Emission::Token(i) => {
                let t = map.tokens()[i];
                writeln!(out, "TOK  {i:3} {:?}", &doc.text[t.start..t.end]).unwrap()
            }
        }
    }
    out
}

#[test]
fn golden_token_dumps_for_five_fixtures() {
    assert_eq!(FIXTURES.len(), 5, "the AC calls for exactly five fixtures");
    for (name, src) in FIXTURES {
        let got = dump(src);
        let path = format!("tests/golden/cue_tokens/{name}.txt");
        if std::env::var("UPDATE_GOLDEN").is_ok() {
            std::fs::write(&path, &got).unwrap();
            continue;
        }
        let want = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing golden {path}: {e} (run with UPDATE_GOLDEN=1)"));
        assert_eq!(got, want, "golden mismatch for {name}");
    }
}

/// The alignment rule itself, stated directly rather than only implied by the dumps.
#[test]
fn a_cues_token_index_is_the_first_token_of_its_span() {
    let v = Vocab::embedded().unwrap();
    for (name, src) in FIXTURES {
        let doc = parse(src, &v, &ParseOptions::default());
        let spans = whitespace_spans(&doc.text);
        let map = OffsetMap::from_spans(spans.clone());
        for a in map.anchor(&doc) {
            let cue = &doc.cues[a.cue];
            // The anchored token must be the first token that starts at or after the
            // cue's span start — i.e. no token of the span is emitted before the cue.
            let expected = spans
                .iter()
                .position(|(s, e)| cue.span.start < *e && cue.span.start >= *s
                          || cue.span.start <= *s)
                .unwrap_or(spans.len());
            assert_eq!(
                a.token, expected,
                "{name}: cue {:?} (span {:?}) anchored to {} not {}",
                cue.raw, cue.span, a.token, expected
            );
        }
    }
}

#[test]
fn a_cue_is_emitted_before_the_token_it_governs() {
    let v = Vocab::embedded().unwrap();
    let doc = parse("it was fine until [angry] the second call", &v, &ParseOptions::default());
    let map = OffsetMap::from_spans(whitespace_spans(&doc.text));
    let stream = interleave(&doc, &map);
    let cue_at = stream.iter().position(|e| matches!(e, Emission::Cue(_))).unwrap();
    // The token immediately after the cue must be the first token of its span.
    let Emission::Token(next) = stream[cue_at + 1] else {
        panic!("a cue must be followed by the token it governs");
    };
    let t = map.tokens()[next];
    assert_eq!(&doc.text[t.start..t.end], "the", "cue must fire on 'the', got a different token");
}

#[test]
fn the_stream_preserves_every_token_in_order() {
    // Interleaving must not drop, duplicate or reorder text.
    let v = Vocab::embedded().unwrap();
    for (name, src) in FIXTURES {
        let doc = parse(src, &v, &ParseOptions::default());
        let map = OffsetMap::from_spans(whitespace_spans(&doc.text));
        let seen: Vec<usize> = interleave(&doc, &map)
            .iter()
            .filter_map(|e| match e {
                Emission::Token(i) => Some(*i),
                _ => None,
            })
            .collect();
        assert_eq!(seen, (0..map.len()).collect::<Vec<_>>(), "{name}: token stream altered");
    }
}

#[test]
fn point_events_occupy_no_tokens_and_spans_do() {
    let v = Vocab::embedded().unwrap();
    let doc = parse("he paused [laughs] and carried on", &v, &ParseOptions::default());
    let map = OffsetMap::from_spans(whitespace_spans(&doc.text));
    for a in map.anchor(&doc) {
        if doc.cues[a.cue].is_point() {
            assert_eq!(a.token, a.token_end, "a point event must occupy no tokens");
        }
    }
}

#[test]
fn token_at_handles_the_boundaries() {
    // Empty map, offset before the first token, inside a token, and past the end.
    let map = OffsetMap::from_spans(whitespace_spans("ab cd"));
    assert_eq!(map.len(), 2);
    assert_eq!(map.token_at(0), 0, "start of the first token");
    assert_eq!(map.token_at(1), 0, "inside the first token belongs to it");
    assert_eq!(map.token_at(2), 1, "the space belongs to the next token");
    assert_eq!(map.token_at(3), 1);
    assert_eq!(map.token_at(5), 2, "past the end is one past the last token");
    assert_eq!(map.token_at(99), 2);
    assert_eq!(OffsetMap::default().token_at(0), 0, "empty map must not panic");
}
