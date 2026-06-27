//! Inline emotion-tagging tests — pure-Rust, **no model required**.
//!
//! These exercise `syrinx_serve::emotion` (the model-free tag parser, the `tag -> instruct`
//! registry, and the equal-power cross-fade) on plain strings + synthetic `f32` buffers, so
//! they neither load weights nor touch Candle/`Tensor` — the same model-free shape as
//! `tests/watermark.rs`, per the frozen "tests live at the repo root" rule.
//!
//! NOTE: at the repo root every test binary links Candle (via the non-optional default-`real`
//! `syrinx-lm`), so on a CPU whose Candle kernels SIGILL these are compile-verified on the
//! dev box and *run* on the model box (where Candle is healthy) — exactly like the existing
//! `watermark` model-free test. None of the assertions below execute any tensor op.

use syrinx_serve::emotion::{
    concat_crossfade, equal_power_crossfade, parse_tagged, EmotionRegistry, InstructLang,
    TagSyntax, DEFAULT_XFADE_SAMPLES,
};

// ----------------------------------------------------------------------------
// Parser
// ----------------------------------------------------------------------------

#[test]
fn leading_tag_sets_first_segment_emotion() {
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("[happy] hello there", &reg);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion.as_deref(), Some("happy"));
    assert_eq!(segs[0].text, "hello there");
}

#[test]
fn mid_text_tag_starts_a_new_segment() {
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("[happy] hi [sad] bye", &reg);
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].emotion.as_deref(), Some("happy"));
    assert_eq!(segs[0].text, "hi");
    assert_eq!(segs[1].emotion.as_deref(), Some("sad"));
    assert_eq!(segs[1].text, "bye");
}

#[test]
fn text_before_any_tag_is_neutral() {
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("hello [angry] you", &reg);
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].emotion, None);
    assert_eq!(segs[0].text, "hello");
    assert_eq!(segs[1].emotion.as_deref(), Some("angry"));
    assert_eq!(segs[1].text, "you");
}

#[test]
fn unknown_tag_is_neutral() {
    // Unknown tags resolve to neutral (None); a warning is logged to stderr (not asserted).
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("[definitelynotanemotion] hi there", &reg);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion, None);
    assert_eq!(segs[0].text, "hi there");
}

#[test]
fn unclosed_bracket_is_literal_text_not_a_panic() {
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("[happy hello there", &reg);
    // No closing `]` -> the whole thing is literal neutral text.
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion, None);
    assert_eq!(segs[0].text, "[happy hello there");
}

#[test]
fn unclosed_bracket_after_a_valid_tag_stays_literal() {
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("[happy] hi [sad bye", &reg);
    // First tag is well-formed; the trailing unclosed `[sad bye` is literal within the span.
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion.as_deref(), Some("happy"));
    assert_eq!(segs[0].text, "hi [sad bye");
}

#[test]
fn paren_syntax_is_accepted_by_default() {
    // Default syntax is Both -> Fish-Speech S1 `(tag)` works alongside `[tag]`.
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("(excited) wow [calm] ok", &reg);
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].emotion.as_deref(), Some("excited"));
    assert_eq!(segs[0].text, "wow");
    assert_eq!(segs[1].emotion.as_deref(), Some("calm"));
    assert_eq!(segs[1].text, "ok");
}

#[test]
fn brackets_only_syntax_treats_parens_as_literal() {
    let reg = EmotionRegistry::default().with_syntax(TagSyntax::Brackets);
    let segs = parse_tagged("(excited) wow", &reg);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion, None);
    assert_eq!(segs[0].text, "(excited) wow");
}

#[test]
fn parens_only_syntax_treats_brackets_as_literal() {
    let reg = EmotionRegistry::default().with_syntax(TagSyntax::Parens);
    let segs = parse_tagged("[happy] hi", &reg);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion, None);
    assert_eq!(segs[0].text, "[happy] hi");
}

#[test]
fn tag_lookup_is_case_insensitive() {
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("[HAPPY] hi", &reg);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion.as_deref(), Some("happy"));
}

#[test]
fn empty_and_whitespace_input_yield_no_segments() {
    let reg = EmotionRegistry::default();
    assert!(parse_tagged("", &reg).is_empty());
    assert!(parse_tagged("    \n\t ", &reg).is_empty());
}

#[test]
fn chinese_text_with_no_tag_is_one_neutral_segment() {
    // A non-tag-shaped bracket content must not swallow real (CJK) text.
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("收到好友从远方寄来的生日礼物。", &reg);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion, None);
    assert_eq!(segs[0].text, "收到好友从远方寄来的生日礼物。");
}

#[test]
fn hyphenated_tone_marker_parses() {
    let reg = EmotionRegistry::default();
    let segs = parse_tagged("[in-a-hurry] quick", &reg);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].emotion.as_deref(), Some("in-a-hurry"));
}

#[test]
fn has_emotion_tags_detects_only_known_tags() {
    let reg = EmotionRegistry::default();
    assert!(reg.has_emotion_tags("[happy] hi"));
    assert!(reg.has_emotion_tags("hello (sad) world"));
    assert!(!reg.has_emotion_tags("just plain text"));
    assert!(!reg.has_emotion_tags("[unknowntag] text"));
}

// ----------------------------------------------------------------------------
// Registry
// ----------------------------------------------------------------------------

const REQUIRED_TAGS: &[&str] = &[
    "happy", "sad", "angry", "excited", "calm", "gentle", "serious", "fearful", "surprised",
    "disgusted", "whisper", "shout", "in-a-hurry", "soft", "slow",
];

#[test]
fn default_registry_has_the_required_vocabulary() {
    let reg = EmotionRegistry::default();
    for &tag in REQUIRED_TAGS {
        assert!(reg.contains(tag), "default registry is missing tag `{tag}`");
        assert!(
            reg.instruct(tag).is_some(),
            "tag `{tag}` has no instruct string"
        );
    }
    // Aliases resolve too.
    assert!(reg.contains("afraid"));
    assert!(reg.contains("disdainful"));
}

#[test]
fn instruct_language_default_is_chinese_and_en_is_selectable() {
    let zh = EmotionRegistry::default();
    assert_eq!(zh.lang(), InstructLang::Zh);
    // The zh form is non-empty and not the English phrasing.
    let zh_happy = zh.instruct("happy").unwrap();
    assert!(!zh_happy.is_empty());
    assert!(!zh_happy.is_ascii(), "zh instruct should contain CJK text");

    let en = EmotionRegistry::default().with_lang(InstructLang::En);
    assert_eq!(en.lang(), InstructLang::En);
    let en_happy = en.instruct("happy").unwrap();
    assert!(en_happy.starts_with("Speak"));

    // Both variants are available regardless of active language.
    let pair = zh.instruct_pair("sad").unwrap();
    assert!(!pair.zh.is_empty() && !pair.en.is_empty());
}

#[test]
fn instruct_strings_do_not_carry_the_endofprompt_marker() {
    // The registry strings stay clean; the synthesizer appends `<|endofprompt|>`.
    let reg = EmotionRegistry::default();
    for tag in reg.tags() {
        let pair = reg.instruct_pair(tag).unwrap();
        assert!(!pair.zh.contains("<|endofprompt|>"));
        assert!(!pair.en.contains("<|endofprompt|>"));
    }
}

#[test]
fn unknown_tag_has_no_instruct() {
    let reg = EmotionRegistry::default();
    assert!(reg.instruct("nope-not-real").is_none());
}

#[test]
fn register_adds_and_overrides() {
    let mut reg = EmotionRegistry::empty();
    assert!(reg.is_empty());
    reg.register("sleepy", "用困倦的语气说", "Speak in a sleepy tone");
    assert!(reg.contains("sleepy"));
    assert_eq!(reg.instruct("sleepy"), Some("用困倦的语气说"));

    // Override an existing tag.
    reg.register("sleepy", "ZZZ", "zzz");
    assert_eq!(reg.instruct("sleepy"), Some("ZZZ"));
    assert_eq!(reg.len(), 1);
}

// ----------------------------------------------------------------------------
// Equal-power cross-fade
// ----------------------------------------------------------------------------

#[test]
fn crossfade_zero_fade_is_plain_concat() {
    let a = vec![1.0f32; 50];
    let b = vec![2.0f32; 30];
    let out = equal_power_crossfade(&a, &b, 0);
    assert_eq!(out.len(), 80);
    assert_eq!(&out[..50], &a[..]);
    assert_eq!(&out[50..], &b[..]);
}

#[test]
fn crossfade_length_is_sum_minus_overlap() {
    let a = vec![1.0f32; 100];
    let b = vec![1.0f32; 100];
    let out = equal_power_crossfade(&a, &b, 20);
    assert_eq!(out.len(), 100 + 100 - 20);
    // The non-overlapped heads/tails are untouched.
    assert_eq!(&out[..80], &a[..80]);
    assert_eq!(&out[out.len() - 80..], &b[..80]);
}

#[test]
fn crossfade_overlap_clamps_to_shorter_side() {
    let a = vec![1.0f32; 5];
    let b = vec![1.0f32; 50];
    // fade (20) > a.len() (5) -> overlap clamps to 5.
    let out = equal_power_crossfade(&a, &b, 20);
    assert_eq!(out.len(), 5 + 50 - 5);
}

#[test]
fn crossfade_is_equal_power_across_the_seam() {
    // Fade a constant-1 buffer into a constant-0 buffer (gives the fade-OUT gain curve),
    // and a constant-0 into a constant-1 (the fade-IN gain curve). Equal-power means
    // g_out(i)^2 + g_in(i)^2 == 1 at every sample of the overlap.
    let n = 64usize;
    let ones = vec![1.0f32; n];
    let zeros = vec![0.0f32; n];

    let fade_out = equal_power_crossfade(&ones, &zeros, n); // length n; overlap = whole
    let fade_in = equal_power_crossfade(&zeros, &ones, n);
    assert_eq!(fade_out.len(), n);
    assert_eq!(fade_in.len(), n);

    for i in 0..n {
        let power = fade_out[i] * fade_out[i] + fade_in[i] * fade_in[i];
        assert!(
            (power - 1.0).abs() < 1e-5,
            "sample {i}: power {power} != 1 (not equal-power)"
        );
    }
    // The fade-out curve falls and the fade-in curve rises monotonically.
    assert!(fade_out[0] > fade_out[n - 1]);
    assert!(fade_in[0] < fade_in[n - 1]);
}

#[test]
fn concat_crossfade_chains_segments_with_one_fade_per_boundary() {
    let fade = 10usize;
    let segs = vec![vec![1.0f32; 100], vec![1.0f32; 100], vec![1.0f32; 100]];
    let out = concat_crossfade(&segs, fade);
    // 3 segments -> 2 boundaries -> total = 300 - 2*fade.
    assert_eq!(out.len(), 300 - 2 * fade);
}

#[test]
fn concat_crossfade_skips_empty_segments_and_handles_none() {
    assert!(concat_crossfade(&[], DEFAULT_XFADE_SAMPLES).is_empty());
    let segs = vec![Vec::<f32>::new(), vec![1.0f32; 40], Vec::<f32>::new()];
    let out = concat_crossfade(&segs, 5);
    assert_eq!(out.len(), 40); // only the one non-empty segment survives
}
