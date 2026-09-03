//! C2.4. **AC (property test): output text for any backend with `Inline::None` contains no
//! `[` / `]` / `<|speaker` sequences unless escaped in source.**

use syrinx_cue::caps::BackendId;
use syrinx_cue::ir::{Cue, CueDoc, CueKind, Level};
use syrinx_cue::lower::{lower, pass_project_fallbacks, pass_strip, project_emphasis,
                        project_prosody, Action, LoweringReport};
use syrinx_cue::parse::ParseOptions;
use syrinx_cue::{parse, Inline, Vocab};

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

const FRAGMENTS: &[&str] = &[
    "[happy]", "[sad]", "[whisper]", "[laughs]", "[pause 300ms]", "[emphasis]", "[/emphasis]",
    "[wibble]", "[a very long free text instruction indeed]", "<|speaker:0|>", "<|speaker:7|>",
    "<|speaker:bad|>", "[speaker 1]", "hello", " world", ". ", "! ", "\n", "[", "]",
    "unclosed [tag", "a ] stray", r"\[literal\]", r"\]", "grüße", "日本語", "🎉", "[]", "[  ]",
];

fn compose(seed: u64, n: usize) -> String {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| *r.pick(FRAGMENTS)).collect::<Vec<_>>().concat()
}

/// The headline AC, as a property over generated input and every no-inline backend.
#[test]
fn no_inline_backends_never_receive_bracket_or_speaker_markup() {
    let v = Vocab::embedded().unwrap();
    let targets: Vec<_> = BackendId::ALL
        .iter()
        .map(|id| id.caps().unwrap())
        .filter(|c| c.inline == Inline::None)
        .collect();
    assert!(!targets.is_empty(), "no Inline::None backend to test — AC would be vacuous");

    let mut checked = 0usize;
    for seed in 1..=800u64 {
        for n in [1usize, 4, 9, 16] {
            let src = compose(seed.wrapping_mul(0x9E37_79B9), n);
            let doc = parse(&src, &v, &ParseOptions::default());
            for caps in &targets {
                let out = lower(&doc, caps, &v);
                let text = pass_strip(&out.text);
                // Escapes in the SOURCE are the only way a bracket may appear.
                let escaped = src.matches(r"\[").count() + src.matches(r"\]").count();
                let brackets = text.matches('[').count() + text.matches(']').count();
                assert!(
                    brackets <= escaped,
                    "LEAK on {}: {brackets} brackets from {escaped} escapes\n  in:  {src:?}\n  out: {text:?}",
                    caps.id
                );
                assert!(
                    !text.contains("<|speaker"),
                    "speaker token leaked on {}\n  in:  {src:?}\n  out: {text:?}",
                    caps.id
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 5000, "property ran only {checked} times");
}

// ------------------------------------------------------------------ strip pass

#[test]
fn strip_restores_escapes_and_is_deliberately_run_exactly_once() {
    assert_eq!(pass_strip(r"a \[b\] c"), "a [b] c");
    // NOT idempotent, by design: the second pass would strip the now-literal brackets it
    // just produced. Pinned so the "run it once, at the very end" ordering constraint
    // stays visible to whoever adds the next pass.
    assert_eq!(pass_strip(&pass_strip(r"a \[b\] c")), "a b c");
    assert_eq!(pass_strip("plain text"), "plain text");
    assert_eq!(pass_strip("a [ b ] c"), "a  b  c");
    assert_eq!(pass_strip("x <|speaker:3|> y"), "x  y");
    // A speaker marker with no closing delimiter loses just the marker.
    assert_eq!(pass_strip("x <|speaker y"), "x  y");
}

#[test]
fn strip_handles_malformed_speaker_tokens_and_multibyte_text() {
    assert!(!pass_strip("<|speaker:notanumber|> hi").contains("<|speaker"));
    assert!(!pass_strip("<|speaker no close").contains("<|speaker"));
    // Multibyte text must survive untouched — an off-by-one here would corrupt it.
    assert_eq!(pass_strip("grüße 日本語 🎉"), "grüße 日本語 🎉");
}

// ------------------------------------------------------------------ scalar projection

#[test]
fn prosody_projects_on_both_sides_of_every_threshold() {
    // Rate: 0.9 and 1.1 are the boundaries, and they are inclusive.
    assert_eq!(project_prosody(Some(0.9), None, None).as_deref(), Some("Speak slowly"));
    assert_eq!(project_prosody(Some(0.91), None, None), None, "just inside the dead zone");
    assert_eq!(project_prosody(Some(1.1), None, None).as_deref(), Some("Speak quickly"));
    assert_eq!(project_prosody(Some(1.09), None, None), None);
    assert_eq!(project_prosody(Some(1.0), None, None), None, "unchanged rate says nothing");

    // Pitch: +/- 1 semitone.
    assert_eq!(project_prosody(None, Some(-1.0), None).as_deref(), Some("Speak in a lower pitch"));
    assert_eq!(project_prosody(None, Some(-0.99), None), None);
    assert_eq!(project_prosody(None, Some(1.0), None).as_deref(), Some("Speak in a higher pitch"));
    assert_eq!(project_prosody(None, Some(0.99), None), None);

    // Volume: +/- 3 dB.
    assert_eq!(project_prosody(None, None, Some(-3.0)).as_deref(), Some("Speak quietly"));
    assert_eq!(project_prosody(None, None, Some(-2.99)), None);
    assert_eq!(project_prosody(None, None, Some(3.0)).as_deref(), Some("Speak loudly"));
    assert_eq!(project_prosody(None, None, Some(2.99)), None);

    // Combined, in a stable order.
    assert_eq!(
        project_prosody(Some(0.5), Some(4.0), Some(6.0)).as_deref(),
        Some("Speak slowly, in a higher pitch, loudly")
    );
    assert_eq!(project_prosody(None, None, None), None);
}

#[test]
fn emphasis_projects_at_every_level() {
    assert_eq!(project_emphasis(Level::Reduced), "Speak this part with less emphasis");
    assert_eq!(project_emphasis(Level::Moderate), "Emphasise this part");
    assert_eq!(project_emphasis(Level::Strong), "Emphasise this part strongly");
}

fn prosody_doc(rate: Option<f32>) -> CueDoc {
    CueDoc {
        text: "hello".into(),
        cues: vec![Cue {
            span: 0..5,
            kind: CueKind::Prosody { rate, pitch_st: None, volume_db: None },
            raw: "<prosody rate=\"slow\">".into(),
            source: 0..21,
        }],
        speakers: vec![],
    }
}

#[test]
fn an_instructable_backend_gets_prosody_as_a_phrase_instead_of_nothing() {
    let mut d = prosody_doc(Some(0.5));
    let caps = BackendId::Qwen17bCustomVoice.caps().unwrap();
    let mut r = LoweringReport::default();
    pass_project_fallbacks(&mut d, &caps, &mut r);
    assert_eq!(d.cues.len(), 1, "prosody must survive as an instruction");
    assert_eq!(d.cues[0].kind, CueKind::Free);
    assert_eq!(d.cues[0].raw, "Speak slowly");
    assert!(matches!(&r.entries[0].action, Action::Projected { to } if to == "Speak slowly"));
    assert!(r.explain().contains("projected to"));
}

#[test]
fn a_backend_with_no_instruction_channel_drops_prosody_with_a_reason() {
    let mut d = prosody_doc(Some(0.5));
    let caps = BackendId::Qwen06bBase.caps().unwrap();
    let mut r = LoweringReport::default();
    pass_project_fallbacks(&mut d, &caps, &mut r);
    assert!(d.cues.is_empty(), "clone-only backend cannot take an instruction");
    assert!(matches!(r.entries[0].action, Action::Dropped { .. }));
    assert!(r.explain().contains("DROPPED"));
}

#[test]
fn the_06b_checkpoint_is_not_handed_a_projection_it_would_ignore() {
    // instruct is `accepted`, not `honored` — projecting into it would be theatre.
    let mut d = prosody_doc(Some(0.5));
    let caps = BackendId::Qwen06bCustomVoice.caps().unwrap();
    let mut r = LoweringReport::default();
    pass_project_fallbacks(&mut d, &caps, &mut r);
    assert!(d.cues.is_empty());
    assert!(matches!(r.entries[0].action, Action::Dropped { .. }));
}

#[test]
fn a_projection_that_says_nothing_is_a_drop_not_an_empty_instruction() {
    // rate 1.0 projects to nothing; the cue must not become an empty instruction.
    let mut d = prosody_doc(Some(1.0));
    let caps = BackendId::Qwen17bCustomVoice.caps().unwrap();
    let mut r = LoweringReport::default();
    pass_project_fallbacks(&mut d, &caps, &mut r);
    assert!(d.cues.is_empty());
    assert!(matches!(r.entries[0].action, Action::Dropped { .. }));
}

#[test]
fn natively_supported_cues_are_left_alone_by_the_fallback_pass() {
    let v = Vocab::embedded().unwrap();
    let d = parse("[happy] hi", &v, &ParseOptions::default());
    let caps = BackendId::FishS2Pro.caps().unwrap();
    let mut d2 = lower(&d, &caps, &v);
    let before = d2.cues.clone();
    let mut doc = CueDoc { text: d2.text.clone(), cues: std::mem::take(&mut d2.cues), speakers: vec![] };
    let mut r = LoweringReport::default();
    pass_project_fallbacks(&mut doc, &caps, &mut r);
    assert_eq!(doc.cues, before, "a natively supported cue must pass through untouched");
    assert!(r.entries.is_empty());
}
