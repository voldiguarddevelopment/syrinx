//! C2.3. **AC:** conflicting cues on an utterance-scoped backend produce N sequential
//! segments with correct prefixes; no split inside a word; opt-out honored.
//! "Correct prefix" is the total function pinned in ledger A14 — not a judgement call.

use syrinx_cue::caps::BackendId;
use syrinx_cue::hoist::{instruct_for, pass_hoist, SplitOptions, UtteranceSegment};
use syrinx_cue::legacy_emotion::InstructLang;
use syrinx_cue::lower::{lower, LoweringReport};
use syrinx_cue::parse::ParseOptions;
use syrinx_cue::{parse, Vocab};

fn split(src: &str, id: BackendId, opts: &SplitOptions) -> (Vec<UtteranceSegment>, LoweringReport) {
    let v = Vocab::embedded().unwrap();
    let d = parse(src, &v, &ParseOptions::default());
    let caps = id.caps().unwrap();
    let low = lower(&d, &caps, &v);
    let mut rep = low.report.clone();
    let segs = pass_hoist(&low, &caps, opts, &mut rep);
    (segs, rep)
}

/// The headline AC.
#[test]
fn conflicting_cues_on_an_utterance_backend_produce_n_segments_with_correct_prefixes() {
    let (segs, _) = split(
        "[happy] good morning [sad] but not for long",
        BackendId::Qwen17bCustomVoice,
        &SplitOptions::default(),
    );
    assert_eq!(segs.len(), 2, "two conflicting cues must yield two requests: {segs:#?}");
    assert_eq!(segs[0].instruct.as_deref(), Some("Speak in a happy, cheerful tone"));
    assert_eq!(segs[1].instruct.as_deref(), Some("Speak in a sad, sorrowful tone"));
    assert!(segs[0].text.contains("good morning"));
    assert!(segs[1].text.contains("but not for long"));
    // The segments are sequential and reconstruct the text.
    let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
    let v = Vocab::embedded().unwrap();
    assert_eq!(joined, parse("[happy] good morning [sad] but not for long", &v,
                             &ParseOptions::default()).text);
}

#[test]
fn three_way_conflict_yields_three_segments_in_order() {
    let (segs, _) = split(
        "[happy] one [sad] two [angry] three",
        BackendId::Qwen17bCustomVoice,
        &SplitOptions::default(),
    );
    assert_eq!(segs.len(), 3);
    let got: Vec<_> = segs.iter().map(|s| s.instruct.clone().unwrap()).collect();
    assert_eq!(
        got,
        vec![
            "Speak in a happy, cheerful tone".to_string(),
            "Speak in a sad, sorrowful tone".to_string(),
            "Speak in an angry tone".to_string(),
        ]
    );
    assert!(segs[0].text.contains("one"));
    assert!(segs[1].text.contains("two"));
    assert!(segs[2].text.contains("three"));
}

#[test]
fn identical_consecutive_cues_do_not_split() {
    // Not a conflict: same delivery twice. Splitting here would cost a whole extra
    // synthesis pass and a join artefact for no expressive gain.
    let (segs, _) = split(
        "[happy] one [happy] two",
        BackendId::Qwen17bCustomVoice,
        &SplitOptions::default(),
    );
    assert_eq!(segs.len(), 1, "same instruction must not split: {segs:#?}");
}

#[test]
fn no_split_ever_lands_inside_a_word() {
    // A cue placed mid-token: the split must move left to the token boundary rather than
    // cutting "some|thing" in half.
    let v = Vocab::embedded().unwrap();
    for src in [
        "[happy] some[sad]thing happens",
        "[happy] a[sad]b",
        "[happy] hello[sad]world again",
    ] {
        let (segs, _) = split(src, BackendId::Qwen17bCustomVoice, &SplitOptions::default());
        let clean = parse(src, &v, &ParseOptions::default()).text;
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, clean, "splitting altered the text for {src:?}");
        for s in &segs {
            let t = s.text.trim();
            if t.is_empty() {
                continue;
            }
            // Each segment must begin and end on a token boundary within the clean text.
            let at = clean.find(t).expect("segment must be a slice of the clean text");
            if at > 0 {
                let prev = clean[..at].chars().next_back().unwrap();
                assert!(
                    prev.is_whitespace(),
                    "segment {t:?} starts mid-word (after {prev:?}) for {src:?}"
                );
            }
        }
    }
}

#[test]
fn opt_out_is_honored_and_the_losing_cues_are_reported() {
    let opts = SplitOptions { allow_split: false, ..Default::default() };
    let (segs, rep) =
        split("[happy] one [sad] two [angry] three", BackendId::Qwen17bCustomVoice, &opts);
    assert_eq!(segs.len(), 1, "opt-out must produce exactly one request");
    assert_eq!(segs[0].instruct.as_deref(), Some("Speak in a happy, cheerful tone"),
               "the first cue wins");
    // The two that lost are reported, not silently ignored.
    let dropped: Vec<_> = rep.dropped().map(|e| e.raw.clone()).collect();
    assert!(dropped.contains(&"sad".to_string()), "dropped: {dropped:?}");
    assert!(dropped.contains(&"angry".to_string()), "dropped: {dropped:?}");
}

#[test]
fn a_word_granular_backend_is_never_split() {
    // Fish carries cues inline, so splitting it would be pure loss.
    for id in [BackendId::FishS2Pro, BackendId::FishS1Mini] {
        let (segs, _) = split("[happy] one [sad] two", id, &SplitOptions::default());
        assert_eq!(segs.len(), 1, "{} must not be split", id.as_str());
        assert_eq!(segs[0].instruct, None, "{} steers inline, not by prefix", id.as_str());
    }
}

#[test]
fn a_backend_with_no_control_gets_one_plain_segment() {
    let (segs, _) = split("[happy] one [sad] two", BackendId::Qwen06bBase, &SplitOptions::default());
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].instruct, None, "clone-only backend must receive no instruction");
}

#[test]
fn the_06b_checkpoint_is_not_given_an_instruction_it_would_ignore() {
    // The accepted-vs-honored distinction, reaching all the way to the request.
    let (segs, _) =
        split("[happy] one [sad] two", BackendId::Qwen06bCustomVoice, &SplitOptions::default());
    assert_eq!(segs.len(), 1, "no point splitting for a checkpoint that ignores instruct");
    assert_eq!(segs[0].instruct, None);
}

// ------------------------------------------------------------------ prefix function

#[test]
fn instruct_for_is_total_and_follows_the_pinned_order() {
    let v = Vocab::embedded().unwrap();
    let cue = |src: &str| parse(src, &v, &ParseOptions::default()).cues[0].clone();

    // 1. Free text passes through verbatim — it is already an instruction.
    let free = cue("[like a distant foghorn, slowly] x");
    assert_eq!(
        instruct_for(&free, InstructLang::En).as_deref(),
        Some("like a distant foghorn, slowly")
    );

    // 2. A curated phrase, in both languages.
    let happy = cue("[happy] x");
    assert_eq!(
        instruct_for(&happy, InstructLang::En).as_deref(),
        Some("Speak in a happy, cheerful tone")
    );
    assert_eq!(instruct_for(&happy, InstructLang::Zh).as_deref(), Some("用开心愉悦的语气说"));

    // 3. A vocabulary label with no curated phrase still gets a deterministic prefix,
    //    which is what makes the function total.
    let uncurated = v
        .iter()
        .find(|(_, e)| EmotionRegistryHas::no(&e.id))
        .map(|(_, e)| e.id.clone());
    if let Some(id) = uncurated {
        let c = cue(&format!("[{id}] x"));
        let got = instruct_for(&c, InstructLang::En).unwrap();
        assert!(got.starts_with("Speak in a"), "uncurated {id:?} -> {got:?}");
    }
}

/// Helper: is a label absent from the legacy curated set?
struct EmotionRegistryHas;
impl EmotionRegistryHas {
    fn no(id: &str) -> bool {
        !syrinx_cue::legacy_emotion::EmotionRegistry::default().contains(id)
    }
}

#[test]
fn segments_never_invent_reorder_or_lose_words() {
    // The property that makes splitting safe: concatenation reproduces the clean text.
    let v = Vocab::embedded().unwrap();
    for src in [
        "[happy] a b c [sad] d e f",
        "[happy] one [sad] two [happy] three",
        "no cues at all",
        "[laughs] point event only",
        "[happy]immediately after the tag",
        "trailing cue [sad]",
    ] {
        let clean = parse(src, &v, &ParseOptions::default()).text;
        let (segs, _) = split(src, BackendId::Qwen17bCustomVoice, &SplitOptions::default());
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, clean, "round trip failed for {src:?}: {segs:#?}");
    }
}

#[test]
fn point_events_do_not_split_and_are_accounted_for() {
    let (segs, rep) = split(
        "[happy] hello [laughs] there [sad] goodbye",
        BackendId::Qwen17bCustomVoice,
        &SplitOptions::default(),
    );
    // [laughs] is zero-width: it must not create a third segment.
    assert_eq!(segs.len(), 2, "a point event must not split: {segs:#?}");
    // Qwen has no event channel at all, so the honest outcome is that it is DROPPED —
    // and, per the C2.2 rule, never silently: it must appear in the report.
    assert!(
        rep.dropped().any(|e| e.raw == "laughs"),
        "the point event vanished without a report line: {:?}",
        rep.entries
    );
}

#[test]
fn a_point_event_survives_on_a_backend_that_can_place_it() {
    // The other half: CosyVoice does have an inline event token, so [laughs] is kept.
    let v = Vocab::embedded().unwrap();
    let d = parse("[happy] hello [laughs] there", &v, &ParseOptions::default());
    let caps = BackendId::CosyVoice2.caps().unwrap();
    let low = lower(&d, &caps, &v);
    assert!(
        low.cues.iter().any(|c| c.is_point()),
        "CosyVoice supports events; the point cue must survive lowering: {:?}",
        low.report.entries
    );
}
