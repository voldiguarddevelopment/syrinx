//! Chatterbox Turbo — the cue-layer data gate.
//!
//! Chatterbox Turbo is an **UNADOPTED candidate** second TTS family (CLAUDE.md: Qwen3-TTS
//! is the TTS path). It is interesting for exactly one reason: it has a native
//! paralinguistic *event* channel, which the shipping path provably lacks
//! (`renders/2026-09-08-event-induction/`). This file gates the cue-layer data that lets
//! the question be asked — the `chatterbox_turbo` spelling column, the `caps.toml` row,
//! and the lowering behaviour that follows from them. It certifies **no audio**: no
//! Chatterbox weights are on this box and nothing here has been heard.
//!
//! Three things are gated, in the order they can go wrong:
//!
//! 1. **No invented spellings.** The native tag set is CLOSED — nineteen tokens at
//!    contiguous ids 50257-50275 in `added_tokens.json`, enumerated in
//!    `docs/backends/CHATTERBOX_PORT_SCOPE.md`. Every `chatterbox_turbo` value in
//!    `vocab.toml` must be one of those nineteen and nothing else. A spelling that is
//!    *nearly* right (`[whisper]` for `[whispering]`, `[laughs]` for `[laugh]`) would
//!    BPE into ordinary words and be **spoken**, which is the failure this closes.
//! 2. **Absence stays meaningful.** The mapped set and the deliberately-unmapped set are
//!    both pinned, in both directions, so a later editor cannot quietly bend `sad` onto
//!    `[crying]` or `excited` onto `[happy]` to make a column look complete.
//! 3. **The hard invariant holds here too.** No cue markup of any dialect may reach this
//!    backend as literal text; the ONLY brackets it may receive are the native spellings
//!    the lowering pass deliberately emits, and the one literal a `\[` escape buys.

use syrinx_cue::caps::{BackendId, CapsTable, ControlCaps};
use syrinx_cue::ir::CueKind;
use syrinx_cue::lower::{lower_full, Action, DropReason, LoweringReport};
use syrinx_cue::{
    pass_hoist, Granularity, Inline, ParseOptions, SplitOptions, Support, Vocab,
};

// ===================================================================== the verified set

/// The complete native tag set, transcribed from the checkpoint's own
/// `added_tokens.json` as enumerated in `docs/backends/CHATTERBOX_PORT_SCOPE.md`
/// (2026-09-12). Ids are carried alongside the spellings so the transcription is
/// self-checking: they must be the contiguous run 50257..=50275, which is what
/// `t3_turbo_v1.yaml`'s `text_tokens_dict_size: 50276` (= 50257 base + 19) corroborates.
///
/// This is the ONLY place the nineteen are written down in code. If upstream ever adds a
/// twentieth, this constant is the deliberate edit that admits it.
const TAGS: &[(&str, u32)] = &[
    ("[angry]", 50257),
    ("[fear]", 50258),
    ("[surprised]", 50259),
    ("[whispering]", 50260),
    ("[advertisement]", 50261),
    ("[dramatic]", 50262),
    ("[narration]", 50263),
    ("[crying]", 50264),
    ("[happy]", 50265),
    ("[sarcastic]", 50266),
    ("[clear throat]", 50267),
    ("[sigh]", 50268),
    ("[shush]", 50269),
    ("[cough]", 50270),
    ("[groan]", 50271),
    ("[sniff]", 50272),
    ("[gasp]", 50273),
    ("[chuckle]", 50274),
    ("[laugh]", 50275),
];

/// The GPT-2 base vocabulary the tags are appended to.
const BASE_VOCAB: u32 = 50257;
/// `t3_turbo_v1.yaml: text_tokens_dict_size`.
const TEXT_TOKENS_DICT_SIZE: u32 = 50276;

/// The mapping this crate commits to: (an author spelling, our canonical id, their tag).
///
/// The author spelling is included because it is the only end-to-end proof that a person
/// typing the obvious thing lands on the right native token. Four of these are the
/// spelling traps the scope doc records, and a fifth (`crying`) behaves the same way.
const MAPPED: &[(&str, &str, &str)] = &[
    ("happy", "happy", "[happy]"),
    ("angry", "angry", "[angry]"),
    ("afraid", "afraid", "[fear]"),           // trap: their [fear] is our `afraid`
    ("surprised", "surprised", "[surprised]"),
    ("sarcastic", "sarcastic", "[sarcastic]"),
    ("whispering", "whisper", "[whispering]"), // trap: [whispering] is not [whisper]
    ("narration", "narrator", "[narration]"),  // trap: [narration] is our `narrator`
    ("laughs", "laugh", "[laugh]"),
    ("chuckle", "chuckle", "[chuckle]"),
    ("sigh", "sigh", "[sigh]"),
    ("gasp", "quick_breath", "[gasp]"),        // trap: their [gasp] is our `quick_breath`
    ("cough", "cough", "[cough]"),
    ("crying", "sob", "[crying]"),             // their emotion, our event — see vocab.toml
    ("groan", "groan", "[groan]"),
];

/// The five native tags with no canonical counterpart, and therefore no mapping.
///
/// Pinned as an exact set because **absence is load-bearing** in this schema: it is what
/// makes lowering report `Unmapped` instead of silently substituting something adjacent.
/// Adding a mapping here is a deliberate change to what the vocabulary claims, and it
/// should cost an edit to a frozen test.
const UNMAPPED_TAGS: &[&str] = &[
    "[advertisement]", // no canonical style; our `broadcast` is news/anchor, not advertising
    "[clear throat]",  // our vocabulary merges throat-clearing into `cough`
    "[dramatic]",      // no canonical style for it
    "[shush]",         // no canonical event
    "[sniff]",         // no canonical event
];

/// Canonical ids that must NOT gain a spelling by being bent onto an adjacent tag.
const MUST_STAY_UNMAPPED: &[&str] = &[
    "sad",      // they have no sad; [crying] is not sadness
    "excited",  // collapsing onto [happy] would discard the arousal difference
    "shout", "scream", "soft", "broadcast", "authoritative", "warm", "singing",
    "breath", "pant", "tsk", "uhm", "clucking", "audience_laughter",
    "serious", "amused", "calm", "curious", "confused",
];

fn vocab() -> Vocab {
    Vocab::embedded().expect("embedded vocab must load and validate")
}

fn caps() -> ControlCaps {
    BackendId::ChatterboxTurbo
        .caps()
        .expect("chatterbox-turbo must have a caps.toml row")
}

fn spelling(v: &Vocab, id: &str) -> Option<String> {
    v.by_id(id)
        .unwrap_or_else(|| panic!("no canonical entry `{id}`"))
        .1
        .chatterbox_turbo
        .clone()
}

// ============================================================ 1. the transcription itself

#[test]
fn the_tag_table_is_the_nineteen_contiguous_ids_upstream_declares() {
    assert_eq!(TAGS.len(), 19, "the verified set has exactly nineteen tags");

    let mut ids: Vec<u32> = TAGS.iter().map(|(_, i)| *i).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), TAGS.len(), "duplicate token id in the transcription");
    assert_eq!(*ids.first().unwrap(), BASE_VOCAB, "the run must start at the base vocab size");
    assert_eq!(*ids.last().unwrap(), BASE_VOCAB + 18, "the run must end at 50275");
    for (n, id) in ids.iter().enumerate() {
        assert_eq!(*id, BASE_VOCAB + n as u32, "ids are contiguous, no gaps");
    }
    // The independent corroboration: 50257 base + 19 tags is the declared text vocab.
    assert_eq!(BASE_VOCAB + TAGS.len() as u32, TEXT_TOKENS_DICT_SIZE);

    let mut spellings: Vec<&str> = TAGS.iter().map(|(s, _)| *s).collect();
    spellings.sort_unstable();
    spellings.dedup();
    assert_eq!(spellings.len(), TAGS.len(), "duplicate spelling in the transcription");
    for (s, _) in TAGS {
        assert!(s.starts_with('[') && s.ends_with(']'), "native form is `[tag]`: {s:?}");
        assert!(s.len() > 2, "empty tag body: {s:?}");
    }
}

/// The other side of the boundary. Every one of these is a plausible near-miss — the
/// shape a mistake actually takes — and none of them is a real token.
#[test]
fn near_miss_spellings_are_not_in_the_verified_set() {
    for bad in [
        "[whisper]",          // our id, not their tag
        "[laughs]",           // an author synonym, not their tag
        "[laughter]",         // CosyVoice's spelling
        "[afraid]",           // our id; theirs is [fear]
        "[quick_breath]",     // our id; theirs is [gasp]
        "[narrator]",         // our id; theirs is [narration]
        "[sad]", "[excited]", // tags they simply do not have
        "[clearing throat]",  // our synonym; theirs is [clear throat]
        "[clear_throat]",     // underscore, not a space
        "happy",              // unbracketed
        "[HAPPY]",            // wrong case
    ] {
        assert!(
            !TAGS.iter().any(|(s, _)| *s == bad),
            "{bad:?} must not be treated as a native tag"
        );
    }
    // ...and the positive side of the same predicate, so the check above is not vacuous.
    for good in ["[happy]", "[fear]", "[clear throat]", "[laugh]"] {
        assert!(TAGS.iter().any(|(s, _)| *s == good), "{good:?} IS a native tag");
    }
}

// ================================================ 2. no invented spellings in vocab.toml

/// **The assertion that stops an invented spelling shipping.**
#[test]
fn every_chatterbox_spelling_in_the_vocabulary_is_one_of_the_nineteen() {
    let v = vocab();
    let claimed: Vec<(String, String)> = v
        .iter()
        .filter_map(|(_, e)| e.chatterbox_turbo.clone().map(|s| (e.id.clone(), s)))
        .collect();

    // Not vacuous: the column must actually exist.
    assert_eq!(
        claimed.len(),
        MAPPED.len(),
        "expected {} chatterbox_turbo spellings, found {}: {claimed:?}",
        MAPPED.len(),
        claimed.len()
    );

    for (id, s) in &claimed {
        assert!(
            TAGS.iter().any(|(t, _)| t == s),
            "`{id}` claims chatterbox_turbo = {s:?}, which is NOT one of the nineteen \
             verified tags. A spelling outside the set is not a token: it BPEs into \
             ordinary words and gets SPOKEN."
        );
        assert!(s.starts_with('[') && s.ends_with(']'), "`{id}` spelling must keep its brackets");
    }

    // Two canonical ids collapsing onto one token would report `Mapped` for both while
    // the backend cannot tell them apart. Not a law of the schema, but a true property of
    // the mapping chosen here, and a cheap guard against a careless copy-paste.
    let mut seen: Vec<&str> = claimed.iter().map(|(_, s)| s.as_str()).collect();
    seen.sort_unstable();
    let before = seen.len();
    seen.dedup();
    assert_eq!(before, seen.len(), "two canonical ids share one native token: {claimed:?}");
}

#[test]
fn the_mapping_is_exactly_the_one_this_gate_commits_to() {
    let v = vocab();
    for (_, id, tag) in MAPPED {
        assert_eq!(
            spelling(&v, id).as_deref(),
            Some(*tag),
            "`{id}` must map to {tag:?}"
        );
    }
    // ...and the negative side: nothing else may have gained a spelling.
    let mapped_ids: Vec<&str> = MAPPED.iter().map(|(_, id, _)| *id).collect();
    for (_, e) in v.iter() {
        if !mapped_ids.contains(&e.id.as_str()) {
            assert_eq!(
                e.chatterbox_turbo, None,
                "`{}` gained a chatterbox_turbo spelling that this gate does not sanction",
                e.id
            );
        }
    }
    for id in MUST_STAY_UNMAPPED {
        assert_eq!(
            spelling(&v, id),
            None,
            "`{id}` must stay unmapped — absence is the honest answer, not a gap to fill"
        );
    }
}

/// The complement, pinned from the tag side rather than the vocabulary side, so the two
/// sets are checked to partition the nineteen exactly.
#[test]
fn the_five_unaddressable_tags_are_exactly_the_ones_recorded() {
    let v = vocab();
    let used: Vec<String> = v.iter().filter_map(|(_, e)| e.chatterbox_turbo.clone()).collect();

    let mut unused: Vec<&str> = TAGS
        .iter()
        .map(|(s, _)| *s)
        .filter(|s| !used.iter().any(|u| u == s))
        .collect();
    unused.sort_unstable();
    let mut want: Vec<&str> = UNMAPPED_TAGS.to_vec();
    want.sort_unstable();
    assert_eq!(unused, want, "the set of native tags we cannot address has changed");

    // They partition: used + unused == all nineteen, with no overlap.
    assert_eq!(used.len() + unused.len(), TAGS.len());
    for u in &unused {
        assert!(!used.iter().any(|x| x == u), "{u:?} is both used and unused");
    }
}

/// The traps, stated as the round trip an author actually performs. Each is asserted
/// against BOTH the wrong answer and the right one.
#[test]
fn the_spelling_traps_map_by_meaning_not_by_spelling() {
    let v = vocab();

    // [whispering] is not [whisper]: our id is `whisper`, their token is `[whispering]`.
    assert_eq!(v.resolve("whispering").unwrap().1.id, "whisper");
    assert_eq!(spelling(&v, "whisper").as_deref(), Some("[whispering]"));
    assert!(v.by_id("whispering").is_none(), "`whispering` is a synonym, never an id");

    // [fear] is our `afraid`; there is no canonical `fear`.
    assert_eq!(spelling(&v, "afraid").as_deref(), Some("[fear]"));
    assert!(v.by_id("fear").is_none(), "`fear` is their spelling, not a canonical id");

    // [gasp] is our `quick_breath`; `gasp` resolves there, and `breath` is a DIFFERENT
    // entry that must not pick the tag up.
    assert_eq!(v.resolve("gasp").unwrap().1.id, "quick_breath");
    assert_eq!(spelling(&v, "quick_breath").as_deref(), Some("[gasp]"));
    assert_eq!(spelling(&v, "breath"), None, "`breath` is not `quick_breath`");

    // [narration] is our `narrator`.
    assert_eq!(v.resolve("narration").unwrap().1.id, "narrator");
    assert_eq!(spelling(&v, "narrator").as_deref(), Some("[narration]"));

    // [crying] is our `sob` (their emotion, our event). `sad` must NOT pick it up.
    assert_eq!(v.resolve("crying").unwrap().1.id, "sob");
    assert_eq!(spelling(&v, "sob").as_deref(), Some("[crying]"));
    assert_eq!(spelling(&v, "sad"), None, "crying is not sadness");

    // They have [cough] and [clear throat] as separate tokens; we have one merged entry,
    // so throat-clearing lands on [cough]. Pinned because it is a real loss, not a bug to
    // discover later in a render.
    assert_eq!(v.resolve("clearing throat").unwrap().1.id, "cough");
    assert_eq!(spelling(&v, "cough").as_deref(), Some("[cough]"));
}

// ============================================================== 3. the caps row

#[test]
fn the_caps_row_declares_a_closed_inline_word_granular_event_channel() {
    let c = caps();
    assert_eq!(c.id, "chatterbox-turbo");
    assert_eq!(c.krate, "syrinx-chatterbox");
    assert!(!c.model.trim().is_empty());

    // The two fields that decide whether pass_hoist splits or passes through.
    assert_eq!(c.inline, Inline::Closed, "nineteen fixed tokens is a CLOSED vocabulary");
    assert_ne!(c.inline, Inline::Open, "free text would be BPE'd into words and spoken");
    assert_ne!(c.inline, Inline::None, "the native surface form is inline `[tag]` text");
    assert_eq!(c.granularity, Granularity::Word, "tags interleave with the text");
    assert_ne!(c.granularity, Granularity::Utterance);
    assert_ne!(c.granularity, Granularity::None);

    // The channel that makes this backend interesting at all.
    assert_eq!(c.emotion, Support::Honored);
    assert_eq!(c.style, Support::Honored);
    assert_eq!(c.event, Support::Honored, "the event channel is the whole reason for this row");
    assert!(c.event.is_effective());

    // Everything it does not have.
    assert_eq!(c.instruct, Support::Unsupported, "no natural-language instruction channel");
    assert_eq!(c.prosody, Support::Unsupported);
    assert_eq!(c.emphasis, Support::Unsupported);
    assert_eq!(c.pause, Support::Unsupported);
    assert_eq!(c.speaker_turn, Support::Unsupported);
    for s in [c.instruct, c.prosody, c.emphasis, c.pause, c.speaker_turn] {
        assert!(!s.is_effective());
    }
}

#[test]
fn can_express_answers_both_ways_for_this_backend() {
    let c = caps();
    for k in [
        CueKind::Emotion { label: "happy".into(), intensity: 0.6 },
        CueKind::Style { label: "whisper".into() },
        CueKind::Event { label: "laugh".into() },
    ] {
        assert!(c.can_express(&k), "expected expressible: {k:?}");
    }
    for k in [
        CueKind::Prosody { rate: Some(0.5), pitch_st: None, volume_db: None },
        CueKind::Emphasis { level: syrinx_cue::Level::Strong },
        CueKind::Pause { ms: 300 },
        CueKind::SpeakerTurn { id: 1 },
        // A closed set with no instruct channel cannot carry free text at all.
        CueKind::Free,
    ] {
        assert!(!c.can_express(&k), "expected inexpressible: {k:?}");
    }
    assert_eq!(c.support_for(&CueKind::Free), Support::Unsupported);
}

#[test]
fn the_row_is_cited_and_declares_itself_unadopted() {
    let c = caps();
    // Same rule as every other row: an uncited capability claim is a guess.
    for needle in ["added_tokens.json", "50257", "50275", "CHATTERBOX_PORT_SCOPE.md"] {
        assert!(c.source.contains(needle), "source must cite {needle:?}: {:?}", c.source);
    }
    // And the thing that makes this row different from every other one in the table.
    assert!(
        c.notes.contains("UNADOPTED"),
        "the row must say in its notes that this backend is not adopted: {:?}",
        c.notes
    );
    for needle in ["no weights", "MIT"] {
        assert!(c.notes.contains(needle), "notes must record {needle:?}");
    }
    // No OTHER row may claim to be unadopted without saying so deliberately: this is the
    // only caps row whose `krate` is allowed not to exist on disk yet.
    let t = CapsTable::embedded().unwrap();
    let unadopted: Vec<&str> = t
        .all()
        .iter()
        .filter(|r| r.notes.contains("UNADOPTED"))
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(unadopted, vec!["chatterbox-turbo"]);
}

#[test]
fn the_backend_id_variant_resolves_and_near_misses_do_not() {
    assert_eq!(BackendId::ChatterboxTurbo.as_str(), "chatterbox-turbo");
    assert_eq!(BackendId::from_str("chatterbox-turbo"), Some(BackendId::ChatterboxTurbo));
    assert!(BackendId::ALL.contains(&BackendId::ChatterboxTurbo));
    for bad in ["chatterbox", "chatterbox_turbo", "Chatterbox-Turbo", "chatterbox-turbo-v1"] {
        assert_eq!(BackendId::from_str(bad), None, "{bad:?} must not resolve");
    }
    assert!(CapsTable::embedded().unwrap().get("chatterbox-turbo").is_some());
}

// ============================================================== 4. lowering

fn lower_one(input: &str) -> syrinx_cue::Lowered {
    let v = vocab();
    let c = caps();
    let doc = syrinx_cue::parse(input, &v, &ParseOptions::default());
    lower_full(&doc, &c, &v)
}

/// **The lowering path produces native spellings and never our canonical ids.**
#[test]
fn every_mapped_cue_lowers_to_its_native_spelling() {
    for (author, id, tag) in MAPPED {
        let lowered = lower_one(&format!("[{author}] hello there"));

        let mapped: Vec<&String> = lowered
            .report
            .entries
            .iter()
            .filter_map(|e| match &e.action {
                Action::Mapped { to } => Some(to),
                _ => None,
            })
            .collect();
        assert_eq!(mapped, vec![tag], "[{author}] must map to {tag:?}");

        // The cue that survives carries the NATIVE spelling, not our id.
        assert_eq!(lowered.cues.len(), 1, "[{author}] should leave exactly one cue");
        let label = match &lowered.cues[0].kind {
            CueKind::Emotion { label, .. }
            | CueKind::Style { label }
            | CueKind::Event { label } => label.clone(),
            other => panic!("[{author}] became {other:?}"),
        };
        assert_eq!(&label, tag);
        assert_ne!(&label, id, "the canonical id must never survive as the label");
        assert_ne!(&label, author, "the author spelling must never survive as the label");

        // Nothing was passed through verbatim — that is the OPEN-vocabulary path and
        // taking it here would ship an unknown token.
        assert!(
            !lowered
                .report
                .entries
                .iter()
                .any(|e| matches!(e.action, Action::PassedThrough { .. })),
            "[{author}] took the open-vocabulary path on a closed backend"
        );
        assert_eq!(lowered.report.dropped().count(), 0, "[{author}] must not be dropped");
        // The markup is gone; the whitespace that surrounded it is not (spans stay stable).
        assert_eq!(lowered.text, " hello there", "[{author}] left markup in the clean text");
    }
}

/// The other side: a label we recognise but cannot spell here is reported `Unmapped` —
/// never `Mapped` to something adjacent, and never silently gone.
#[test]
fn a_recognised_label_with_no_native_spelling_reports_unmapped() {
    for author in ["sad", "excited", "shouting", "panting", "tsk"] {
        let lowered = lower_one(&format!("[{author}] hello there"));
        let actions: Vec<&Action> = lowered.report.entries.iter().map(|e| &e.action).collect();
        assert!(
            actions.iter().any(|a| matches!(a, Action::Unmapped)),
            "[{author}] should report Unmapped, got {actions:?}"
        );
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Mapped { .. })),
            "[{author}] was mapped to something it has no token for: {actions:?}"
        );
        assert_eq!(lowered.text, " hello there");
    }
}

/// And a label we do not recognise at all is DROPPED, because a closed token set has
/// nowhere to put free text. This is the case that would leak if `inline` were `open`.
#[test]
fn an_unrecognised_cue_is_dropped_not_passed_through() {
    let lowered = lower_one("[wibble wobble] hello there");
    assert_eq!(lowered.cues.len(), 0, "a Free cue must not survive onto a closed backend");
    let dropped: Vec<DropReason> = lowered
        .report
        .entries
        .iter()
        .filter_map(|e| match e.action {
            Action::Dropped { reason } => Some(reason),
            _ => None,
        })
        .collect();
    assert_eq!(dropped, vec![DropReason::Unsupported]);
    assert_eq!(lowered.text, " hello there");
}

/// `inline = closed` + `granularity = word` must make `pass_hoist` PASS THROUGH.
/// Asserted against a backend that does split, so both sides of the decision are pinned.
#[test]
fn a_word_granular_inline_backend_is_never_split_into_utterances() {
    let v = vocab();
    let input = "[happy] one two [sad] three four";
    let doc = syrinx_cue::parse(input, &v, &ParseOptions::default());
    let opts = SplitOptions::default();
    assert!(opts.allow_split, "the comparison is only meaningful with splitting ENABLED");

    let cb = caps();
    let lowered = lower_full(&doc, &cb, &v);
    let mut report = LoweringReport::default();
    let segs = pass_hoist(&lowered, &cb, &opts, &mut report);
    assert_eq!(segs.len(), 1, "chatterbox carries its cues inline; it must not be split");
    assert_eq!(segs[0].text, " one two  three four");
    assert_eq!(segs[0].instruct, None, "there is no instruct channel to put a prefix in");

    // The contrast: an utterance-granular backend with no inline channel DOES split.
    let qwen = BackendId::Qwen17bCustomVoice.caps().unwrap();
    assert_eq!(qwen.inline, Inline::None);
    assert_eq!(qwen.granularity, Granularity::Utterance);
    let q_lowered = lower_full(&doc, &qwen, &v);
    let mut q_report = LoweringReport::default();
    let q_segs = pass_hoist(&q_lowered, &qwen, &opts, &mut q_report);
    assert!(
        q_segs.len() > 1,
        "the contrast case must actually split, or this test proves nothing"
    );
}

// ============================================================== 5. the hard invariant

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

/// Deliberately includes the native spellings themselves, our ids, author synonyms,
/// malformed markup and speaker tokens — every shape that could put a bracket in front of
/// this backend.
const FRAGMENTS: &[&str] = &[
    "[happy]", "[whispering]", "[laugh]", "[clear throat]", "[gasp]", "[crying]",
    "[whisper]", "[laughs]", "[quick_breath]", "[sad]", "[dramatic]", "[sniff]",
    "[wibble]", "[very angry]", "[pause 300ms]", "[emphasis]", "[/emphasis]",
    "<|speaker:0|>", "<|speaker:2|>", "[speaker 1]",
    "hello", " world", " and then. ", "! ", "\n",
    "[He turns, slowly]", "[]", "[   ]", "unclosed [tag", "a ] stray", "[a [b] c]",
    r"\[literal\]", r"\]", "grüße", "🎉",
];

fn compose(seed: u64, n: usize) -> String {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| *r.pick(FRAGMENTS)).collect::<Vec<_>>().concat()
}

#[test]
fn no_cue_markup_of_any_dialect_reaches_chatterbox_as_literal_text() {
    let v = vocab();
    let c = caps();
    let opts = ParseOptions::default();
    let mut saw_escape = 0usize;
    for seed in 1..=2000u64 {
        let input = compose(seed, 1 + (seed as usize % 12));
        let doc = syrinx_cue::parse(&input, &v, &opts);
        let lowered = lower_full(&doc, &c, &v);
        let text = &lowered.text;

        // `lower_full`'s strip pass resolves `\[` to a literal `[`, so an escape is the
        // ONLY way a bracket may appear. Anything above that count is a leak.
        let escaped_open = input.matches(r"\[").count();
        let escaped_close = input.matches(r"\]").count();
        saw_escape += escaped_open + escaped_close;
        assert!(
            text.matches('[').count() <= escaped_open,
            "LEAK: unescaped '[' reached the backend\n  in:  {input:?}\n  out: {text:?}"
        );
        assert!(
            text.matches(']').count() <= escaped_close,
            "LEAK: unescaped ']' reached the backend\n  in:  {input:?}\n  out: {text:?}"
        );
        assert!(
            !text.contains("<|speaker"),
            "LEAK: speaker token reached the backend\n  in:  {input:?}\n  out: {text:?}"
        );
    }
    assert!(saw_escape > 0, "the generator never exercised the escape path");
}

/// The same invariant stated as the thing that would actually be heard: a native tag must
/// never appear in the spoken text. It reaches the backend as a *cue*, or not at all.
#[test]
fn a_native_tag_never_appears_in_the_spoken_text() {
    for (author, _, tag) in MAPPED {
        let lowered = lower_one(&format!("say [{author}] this"));
        for (t, _) in TAGS {
            assert!(
                !lowered.text.contains(t),
                "LEAK: native tag {t:?} is in the spoken text {:?}",
                lowered.text
            );
        }
        // It did reach the backend — through the cue channel, which is the whole point.
        assert!(lowered.cues.iter().any(|c| match &c.kind {
            CueKind::Emotion { label, .. } | CueKind::Style { label } | CueKind::Event { label } =>
                label == tag,
            _ => false,
        }));
    }
    // Writing a native tag directly is still cue syntax, and still never spoken.
    for (t, _) in TAGS {
        let lowered = lower_one(&format!("before {t} after"));
        assert!(!lowered.text.contains('['), "LEAK: {t:?} survived as text");
        assert!(!lowered.text.contains(']'), "LEAK: {t:?} survived as text");
    }
    // The single sanctioned escape hatch, both halves: escaped brackets ARE spoken,
    // unescaped ones are not.
    assert_eq!(lower_one(r"say \[happy\] now").text, "say [happy] now");
    assert_eq!(lower_one("say [happy] now").text, "say  now");
}

#[test]
fn the_ssml_dialect_cannot_reach_chatterbox_either() {
    let v = vocab();
    let c = caps();
    for input in [
        "<speak>hello <prosody rate=\"slow\">slowly</prosody> there</speak>",
        "<speak><emphasis level=\"strong\">now</emphasis></speak>",
        "<speak>wait<break time=\"400ms\"/>then go</speak>",
    ] {
        let doc = syrinx_cue::parse_ssml(input).expect("valid SSML must parse");
        let lowered = lower_full(&doc, &c, &v);
        for tag in ["<speak", "</speak", "<prosody", "</prosody", "<emphasis", "</emphasis", "<break"]
        {
            assert!(
                !lowered.text.contains(tag),
                "LEAK: {tag:?} reached the backend\n  in:  {input:?}\n  out: {:?}",
                lowered.text
            );
        }
        assert!(!lowered.text.contains('<'), "no angle-bracket markup may survive");
        // A prosody/emphasis/break cue has no channel here and no instruct to project
        // onto, so it must be reported dropped rather than vanishing.
        assert!(
            lowered.report.dropped().count() > 0,
            "{input:?} lost its cues without reporting a drop"
        );
    }
}
