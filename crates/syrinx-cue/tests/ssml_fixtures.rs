//! C1.3 fixtures: the SSML subset lands in the SAME IR as bracket syntax, and every
//! rejection path is exercised. AC: fixtures + error path tested.

use syrinx_cue::ir::{CueKind, Level};
use syrinx_cue::parse::ParseOptions;
use syrinx_cue::ssml::{detect, parse_any, parse_ssml, Dialect, SsmlError};
use syrinx_cue::vocab::Vocab;

fn ssml(input: &str) -> syrinx_cue::ir::CueDoc {
    parse_ssml(input).unwrap_or_else(|e| panic!("{input:?} should parse, got {e}"))
}

fn err(input: &str) -> SsmlError {
    parse_ssml(input).expect_err(&format!("{input:?} should have been rejected"))
}

/// (source, clean text, cue count)
const CLEAN: &[(&str, &str, usize)] = &[
    // the root carries no cue
    ("<speak>hello there</speak>", "hello there", 0),
    ("no markup at all", "no markup at all", 0),
    // prosody
    ("<prosody rate=\"slow\">slow bit</prosody> rest", "slow bit rest", 1),
    ("<prosody pitch=\"+2st\">up</prosody>", "up", 1),
    ("<prosody volume=\"-6dB\">quiet</prosody>", "quiet", 1),
    ("<prosody rate=\"150%\" pitch=\"high\" volume=\"loud\">all three</prosody>", "all three", 1),
    // emphasis
    ("say <emphasis>this</emphasis> now", "say this now", 1),
    ("say <emphasis level=\"strong\">this</emphasis>", "say this", 1),
    // break is a point event
    ("a<break time=\"400ms\"/>b", "ab", 1),
    ("a<break strength=\"weak\"/>b", "ab", 1),
    ("a<break/>b", "ab", 1),
    // nesting
    ("<speak><prosody rate=\"fast\">a <emphasis>b</emphasis> c</prosody></speak>", "a b c", 2),
    // entities decode, and never re-open a tag
    ("5 &lt; 6 &amp; 7", "5 < 6 & 7", 0),
    ("&quot;quoted&quot;", "\"quoted\"", 0),
    // a bare `<` that is not a tag stays literal
    ("x < y", "x < y", 0),
    // self-closing prosody has no content: an empty span, not an error
    ("a<prosody rate=\"slow\"/>b", "ab", 1),
];

#[test]
fn ssml_fixtures_produce_the_expected_clean_text_and_cue_count() {
    for (src, want_text, want_cues) in CLEAN {
        let doc = ssml(src);
        assert_eq!(&doc.text, want_text, "text for {src:?}");
        assert_eq!(doc.cues.len(), *want_cues, "cue count for {src:?}");
    }
}

#[test]
fn no_ssml_markup_ever_survives_into_the_clean_text() {
    // The hard invariant, on the SSML path: a tag must never be spoken. `<` may survive
    // only when it came from an entity or a non-tag `<`, so check tag shapes specifically.
    for (src, _, _) in CLEAN {
        let t = ssml(src).text;
        for tag in ["<speak", "<prosody", "<emphasis", "<break", "</"] {
            assert!(!t.contains(tag), "LEAK {tag:?} from {src:?} -> {t:?}");
        }
    }
}

#[test]
fn spans_index_the_clean_text_and_slice_the_intended_words() {
    let doc = ssml("say <emphasis level=\"strong\">this word</emphasis> now");
    assert_eq!(doc.text, "say this word now");
    let c = &doc.cues[0];
    assert_eq!(&doc.text[c.span.clone()], "this word");
    assert_eq!(c.kind, CueKind::Emphasis { level: Level::Strong });
    // source range points back at the original markup, for --explain
    assert!(doc.text[..c.span.start].ends_with("say "));
}

#[test]
fn a_break_is_a_zero_width_point_event_at_the_right_offset() {
    let doc = ssml("before<break time=\"0.4s\"/>after");
    assert_eq!(doc.text, "beforeafter");
    let c = &doc.cues[0];
    assert!(c.is_point(), "break must not scope text");
    assert_eq!(c.span.start, "before".len());
    assert_eq!(c.kind, CueKind::Pause { ms: 400 });
}

#[test]
fn attribute_values_map_onto_the_scalar_axes() {
    let by = |s: &str| ssml(s).cues[0].kind.clone();
    // rate: keyword, percent and bare multiplier all normalise to a multiplier
    assert_eq!(by("<prosody rate=\"slow\">x</prosody>"),
               CueKind::Prosody { rate: Some(0.75), pitch_st: None, volume_db: None });
    assert_eq!(by("<prosody rate=\"150%\">x</prosody>"),
               CueKind::Prosody { rate: Some(1.5), pitch_st: None, volume_db: None });
    assert_eq!(by("<prosody rate=\"1.5\">x</prosody>"),
               CueKind::Prosody { rate: Some(1.5), pitch_st: None, volume_db: None });
    // pitch: semitones stay semitones; keywords and percent convert
    assert_eq!(by("<prosody pitch=\"-3st\">x</prosody>"),
               CueKind::Prosody { rate: None, pitch_st: Some(-3.0), volume_db: None });
    assert_eq!(by("<prosody pitch=\"x-high\">x</prosody>"),
               CueKind::Prosody { rate: None, pitch_st: Some(6.0), volume_db: None });
    // volume
    assert_eq!(by("<prosody volume=\"x-loud\">x</prosody>"),
               CueKind::Prosody { rate: None, pitch_st: None, volume_db: Some(12.0) });
    // break strength ladder, both ends
    assert_eq!(by("<break strength=\"none\"/>"), CueKind::Pause { ms: 0 });
    assert_eq!(by("<break strength=\"x-strong\"/>"), CueKind::Pause { ms: 1000 });
    // emphasis default vs explicit, both directions off moderate
    assert_eq!(by("<emphasis>x</emphasis>"), CueKind::Emphasis { level: Level::Moderate });
    assert_eq!(by("<emphasis level=\"reduced\">x</emphasis>"),
               CueKind::Emphasis { level: Level::Reduced });
}

#[test]
fn percent_pitch_converts_through_the_semitone_relation() {
    // +100% is one octave = 12 st; -50% is one octave down.
    let st = |s: &str| match ssml(s).cues[0].kind {
        CueKind::Prosody { pitch_st, .. } => pitch_st.unwrap(),
        ref k => panic!("{k:?}"),
    };
    assert!((st("<prosody pitch=\"+100%\">x</prosody>") - 12.0).abs() < 1e-4);
    assert!((st("<prosody pitch=\"-50%\">x</prosody>") + 12.0).abs() < 1e-4);
}

// ------------------------------------------------------------------ error paths

#[test]
fn mixed_syntax_is_a_hard_error_in_both_orders() {
    assert!(matches!(err("<emphasis>a</emphasis> [happy] b"), SsmlError::MixedSyntax { .. }));
    assert!(matches!(err("[happy] a <emphasis>b</emphasis>"), SsmlError::MixedSyntax { .. }));
    // a speaker token counts as bracket-dialect markup
    assert!(matches!(err("<|speaker:1|> a <emphasis>b</emphasis>"), SsmlError::MixedSyntax { .. }));
    // ...but an ESCAPED bracket is literal text, so it is not a mix
    assert!(parse_ssml("<emphasis>a</emphasis> \\[not a cue\\]").is_ok());
}

#[test]
fn unsupported_tags_are_rejected_not_silently_dropped() {
    match err("<speak><voice name=\"bob\">hi</voice></speak>") {
        SsmlError::UnsupportedTag { name, .. } => assert_eq!(name, "voice"),
        e => panic!("wrong error: {e}"),
    }
    assert!(matches!(err("<phoneme ph=\"x\">a</phoneme>"), SsmlError::UnsupportedTag { .. }));
}

#[test]
fn structural_errors_are_reported_with_a_position() {
    assert!(matches!(err("<emphasis>never closed"), SsmlError::UnclosedTag { .. }));
    assert!(matches!(err("<emphasis>a</prosody>"), SsmlError::MismatchedClose { .. }));
    assert!(matches!(err("a</emphasis>"), SsmlError::MismatchedClose { expected: None, .. }));
    assert!(matches!(err("<emphasis level=\"strong\">a"), SsmlError::UnclosedTag { .. }));
    // a tag that never terminates
    assert!(matches!(err("<prosody rate=\"slow\" hello"), SsmlError::Malformed { .. }));
}

#[test]
fn bad_attribute_values_are_rejected_on_every_axis() {
    for (src, attr) in [
        ("<prosody rate=\"quickly\">a</prosody>", "rate"),
        ("<prosody pitch=\"shrill\">a</prosody>", "pitch"),
        ("<prosody volume=\"eleven\">a</prosody>", "volume"),
        ("<break time=\"soon\"/>", "time"),
        ("<break strength=\"enormous\"/>", "strength"),
        ("<emphasis level=\"colossal\">a</emphasis>", "level"),
    ] {
        match parse_ssml(src) {
            Err(SsmlError::BadAttribute { attr: a, .. }) => assert_eq!(a, attr, "for {src:?}"),
            other => panic!("{src:?} should be a BadAttribute, got {other:?}"),
        }
    }
    // boundary: a zero or negative rate is not a rate
    assert!(matches!(err("<prosody rate=\"0\">a</prosody>"), SsmlError::BadAttribute { .. }));
    assert!(matches!(err("<prosody rate=\"-1\">a</prosody>"), SsmlError::BadAttribute { .. }));
}

#[test]
fn detect_routes_each_dialect_and_parse_any_agrees() {
    let v = Vocab::embedded().unwrap();
    let o = ParseOptions::default();
    assert_eq!(detect("[happy] hi").unwrap(), Dialect::Brackets);
    assert_eq!(detect("<emphasis>hi</emphasis>").unwrap(), Dialect::Ssml);
    assert_eq!(detect("plain text").unwrap(), Dialect::Brackets);
    // both syntaxes reach the one IR through the one entry point
    assert_eq!(parse_any("<emphasis>hi</emphasis>", &v, &o).unwrap().text, "hi");
    assert_eq!(parse_any("[happy] hi", &v, &o).unwrap().text, " hi");
    assert!(matches!(parse_any("[happy] <emphasis>hi</emphasis>", &v, &o),
                     Err(SsmlError::MixedSyntax { .. })));
}
