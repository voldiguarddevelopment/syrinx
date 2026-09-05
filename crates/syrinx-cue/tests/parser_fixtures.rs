//! C1.2 gate: >=50 parser fixtures including adversarial cases.
//!
//! Every fixture asserts on the CLEAN TEXT (what a backend would receive) as well as the
//! cue shape. The clean-text assertions are the ones that matter: they are the first line
//! of defence for the hard invariant that no cue markup reaches a backend.

use syrinx_cue::ir::{CueKind, Level};
use syrinx_cue::{parse, ParseOptions, Syntax, Vocab};

fn v() -> Vocab { Vocab::embedded().unwrap() }
fn p(s: &str) -> syrinx_cue::CueDoc { parse(s, &v(), &ParseOptions::default()) }
fn p_syntax(s: &str, syntax: Syntax) -> syrinx_cue::CueDoc {
    parse(s, &v(), &ParseOptions { syntax, ..Default::default() })
}

/// (input, expected clean text, expected cue count)
const CASES: &[(&str, &str, usize)] = &[
    ("[happy] hello there.", " hello there.", 1),
    ("[sad] goodbye.", " goodbye.", 1),
    ("[whisper] quiet now.", " quiet now.", 1),
    ("[excited] we won!", " we won!", 1),
    ("[angry] stop that.", " stop that.", 1),
    ("I said [angry] stop.", "I said  stop.", 1),
    ("well [sad] that is that.", "well  that is that.", 1),
    ("[happy] hi [sad] bye.", " hi  bye.", 2),
    ("[whisper] a [shout] b.", " a  b.", 2),
    ("She [laughs] said it.", "She  said it.", 1),
    ("Oh [sigh] fine.", "Oh  fine.", 1),
    ("A [cough] B.", "A  B.", 1),
    ("[breath] okay.", " okay.", 1),
    ("[gasp] really?", " really?", 1),
    ("wait [pause 400ms] then go.", "wait  then go.", 1),
    ("wait [pause 0.4s] then go.", "wait  then go.", 1),
    ("wait [pause] then go.", "wait  then go.", 1),
    ("wait [break 250] then go.", "wait  then go.", 1),
    ("[emphasis] now.", " now.", 1),
    ("say [emphasis]this[/emphasis] loudly.", "say this loudly.", 1),
    ("[strong emphasis] listen.", " listen.", 1),
    ("<|speaker:0|>hi.", "hi.", 1),
    ("<|speaker:12|>hello.", "hello.", 1),
    ("[speaker 1] hey.", " hey.", 1),
    ("<|speaker:0|>a.<|speaker:1|>b.", "a.b.", 2),
    (r"a \[b\] c.", r"a \[b\] c.", 0),
    (r"\[happy\] not a cue.", r"\[happy\] not a cue.", 0),
    (r"array\[0\] index.", r"array\[0\] index.", 0),
    // The documented cost of the strict rule: unescaped brackets in technical text are
    // cue syntax. Loud (text visibly missing + a cue in the report), not silent.
    ("array[0] index.", "array index.", 1),
    // STRICT (ADR-0001 §9a): an unescaped bracket is cue syntax or it is stripped.
    // A stage direction is a legitimate free-text style instruction, so it becomes a
    // Free cue rather than being spoken aloud.
    ("[He turns to the window, slowly and deliberately] ok.", " ok.", 1),
    ("[this tag is far too long to be a real cue label indeed] x.", " x.", 1),
    // Shapes that cannot be a cue are dropped, never emitted as literal text.
    ("[] empty.", " empty.", 0),
    // Delimiters are dropped, inner content preserved: whitespace-only brackets leave
    // their whitespace. Consistent with the newline case below.
    ("[   ] spaces.", "    spaces.", 0),
    ("unclosed [happy and then nothing", "unclosed happy and then nothing", 0),
    ("line\nbreak [in\nside] x.", "line\nbreak in\nside x.", 0),
    ("all done [happy]", "all done ", 1),
    ("done. [laughs]", "done. ", 1),
    ("[happy][sad] which?", " which?", 2),
    ("[happy] [happy] same.", "  same.", 2),
    ("[happy] one. two.", " one. two.", 1),
    ("[sad] a! b? c.", " a! b? c.", 1),
    ("[professional broadcast tone] news.", " news.", 1),
    ("[in a hurry] quick.", " quick.", 1),
    ("[wibble] odd.", " odd.", 1),
    ("[very happy] yes.", " yes.", 1),
    ("[slightly sad] hm.", " hm.", 1),
    ("[super excited] wow!", " wow!", 1),
    ("[happy] grüße über brücken.", " grüße über brücken.", 1),
    ("[sad] 日本語のテキスト。", " 日本語のテキスト。", 1),
    ("[excited] emoji 🎉 here.", " emoji 🎉 here.", 1),
    ("", "", 0),
    ("   ", "   ", 0),
    ("no cues at all.", "no cues at all.", 0),
];

#[test]
fn fixture_count_meets_the_ac() {
    let named = 6usize;
    assert!(CASES.len() + named >= 50, "C1.2 requires >=50 fixtures, have {}", CASES.len() + named);
}

#[test]
fn all_fixtures_produce_the_expected_clean_text_and_cue_count() {
    let mut bad = Vec::new();
    for (input, want_text, want_cues) in CASES {
        let doc = p(input);
        if doc.text != *want_text {
            bad.push(format!("  {input:?}\n    text want {want_text:?}\n         got  {:?}", doc.text));
        }
        if doc.cues.len() != *want_cues {
            bad.push(format!("  {input:?}\n    cues want {want_cues}, got {}", doc.cues.len()));
        }
    }
    assert!(bad.is_empty(), "fixture failures:\n{}", bad.join("\n"));
}

#[test]
fn point_events_are_zero_width_and_spans_are_not() {
    let doc = p("She [laughs] said [happy] it.");
    let ev = doc.cues.iter().find(|c| matches!(c.kind, CueKind::Event { .. })).unwrap();
    assert!(ev.is_point());
    let em = doc.cues.iter().find(|c| matches!(c.kind, CueKind::Emotion { .. })).unwrap();
    assert!(!em.is_point());
}

#[test]
fn spans_are_valid_byte_ranges_into_the_clean_text() {
    for (input, _, _) in CASES {
        let doc = p(input);
        for c in &doc.cues {
            assert!(c.span.end <= doc.text.len(), "{input:?}: span past end");
            assert!(c.span.start <= c.span.end, "{input:?}: inverted span");
            assert!(doc.text.is_char_boundary(c.span.start), "{input:?}: start mid-char");
            assert!(doc.text.is_char_boundary(c.span.end), "{input:?}: end mid-char");
        }
    }
}

#[test]
fn explicit_close_bounds_the_span_exactly() {
    let doc = p("say [emphasis]this[/emphasis] loudly.");
    let c = doc.cues.iter()
        .find(|c| matches!(c.kind, CueKind::Emphasis { level: Level::Moderate })).unwrap();
    assert_eq!(&doc.text[c.span.clone()], "this");
}

#[test]
fn free_cues_retain_the_author_text_verbatim() {
    // Deliberately not in vocab.toml, and not resolvable by any synonym.
    let doc = p("[wibble frobnicate] news.");
    let c = &doc.cues[0];
    assert!(matches!(c.kind, CueKind::Free), "unknown label must become Free, not be dropped");
    assert_eq!(c.raw, "wibble frobnicate", "Free must carry the author text verbatim");
    assert_eq!(doc.text, " news.", "and the markup must still be stripped");
}

#[test]
fn legacy_paren_mode_is_off_by_default_and_opt_in() {
    let off = p("(happy) hello.");
    assert_eq!(off.text, "(happy) hello.");
    assert_eq!(off.cues.len(), 0);
    let on = p_syntax("(happy) hello.", Syntax::Parens);
    assert_eq!(on.text, " hello.");
    assert_eq!(on.cues.len(), 1);
}
