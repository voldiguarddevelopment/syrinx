//! Per-sentence tuning decisions, frozen.
//!
//! The pooled decision (`tests/cue_tune_decision.rs`, still frozen and still green) turned
//! out to be unusable in practice: the tuner pooled 3 sentences x 4 seeds into one
//! permutation test and **the incumbent itself could not clear the acoustic bar** —
//! p=0.0166 on the tune split, 0.3580 on the holdout, against the same phrase scoring
//! 0.0003–0.0057 when measured per sentence at n=8.
//!
//! Twelve pooled samples doing worse than eight per-sentence samples is inflated variance,
//! and the sentence-dependence sweeps say where it comes from. A gate the incumbent cannot
//! pass accepts nothing, which is the failure ADR-0003 named for the 0.85 floor reached
//! from the other side.
//!
//! These assertions pin the aggregation: pass on `min_sentences`, never pool.

use syrinx_eval::tune::{
    decide_per_sentence, Reject, SentenceMeasurement, Split, TuneArm, TuneMeasurement,
    TuneThresholds, Void,
};

fn m(arm: TuneArm, split: Split, phrase: Option<&str>, delta: f64) -> TuneMeasurement {
    TuneMeasurement {
        label: "sad".into(),
        phrase: phrase.map(str::to_string),
        arm,
        split,
        p_vs_plain: 0.001,
        p_vs_sham: 0.001,
        cued_delta: delta,
        cued_noise: 1.0,
        largest_gain_label: "sad".into(),
        wer: 0.0,
        speaker_similarity: 0.99,
    }
}
fn s(sentence: &str, arm: TuneArm, split: Split, phrase: Option<&str>, d: f64) -> SentenceMeasurement {
    SentenceMeasurement { sentence: sentence.into(), m: m(arm, split, phrase, d) }
}
fn th() -> TuneThresholds {
    TuneThresholds { alpha: 0.05, candidates: 10, min_margin: 1.0, ..Default::default() }
}

/// Three tune sentences and three **different** holdout sentences.
///
/// The ids are disjoint per split because that is what the driver produces — tune and
/// holdout use different sentences, which is what a holdout is. A first version of this
/// helper reused one id set across both splits, and the aggregation bug it therefore could
/// not see (a union of ids, then a demand for a Tune incumbent on every one) voided every
/// real round on its holdout ids. A synthetic fixture that does not model the real shape is
/// worse than no fixture: it converts a missing test into a passing one.
fn round(deltas: [f64; 3]) -> Vec<SentenceMeasurement> {
    let mut v = Vec::new();
    for (i, d) in deltas.iter().enumerate() {
        for (split, id) in [
            (Split::Tune, format!("tune-s{i}")),
            (Split::Holdout, format!("holdout-s{i}")),
        ] {
            if split == Split::Tune {
                let mut sham = s(&id, TuneArm::Sham, Split::Tune, None, 0.0);
                sham.m.p_vs_plain = 0.9;
                v.push(sham);
            }
            v.push(s(&id, TuneArm::Incumbent, split, Some("old"), 2.0));
            v.push(s(&id, TuneArm::Candidate, split, Some("new"), *d));
        }
    }
    v
}

/// **Regression, 2026-09-08.** Disjoint sentence ids across splits must not void the round.
///
/// The first implementation took the union of all sentence ids and then required a
/// `Split::Tune` incumbent for each — so every holdout id tripped `IncumbentNotRemeasured`
/// and every real round voided before a single candidate was scored. It reached a live
/// 324-render GPU run because the fixture reused one id set across both splits.
#[test]
fn disjoint_sentence_ids_across_splits_do_not_void_the_round() {
    let v = round([5.0, 5.0, 5.0]);
    let tune: Vec<&str> = v.iter().filter(|x| x.m.split == Split::Tune).map(|x| x.sentence.as_str()).collect();
    let hold: Vec<&str> = v.iter().filter(|x| x.m.split == Split::Holdout).map(|x| x.sentence.as_str()).collect();
    assert!(
        tune.iter().all(|t| !hold.contains(t)),
        "the fixture must model disjoint splits or it cannot catch this"
    );
    let r = decide_per_sentence(&v, 0, &th(), 2);
    assert!(r.void.is_none(), "disjoint ids must not void: {:?}", r.void);
    assert!(r.has_proposal());
}

#[test]
fn a_candidate_clearing_every_sentence_is_proposed() {
    let r = decide_per_sentence(&round([5.0, 5.0, 5.0]), 0, &th(), 2);
    assert!(r.void.is_none(), "{:?}", r.void);
    assert_eq!(r.proposed.map(|c| c.phrase), Some("new".to_string()));
}

/// The whole point: a candidate that works on SOME sentences can still win, which pooling
/// made impossible. Both sides of the threshold.
#[test]
fn passing_on_min_sentences_is_enough_and_one_fewer_is_not() {
    // Two of three clear (the third has no margin over the incumbent's 2.0).
    let two = round([5.0, 5.0, 2.0]);
    assert!(decide_per_sentence(&two, 0, &th(), 2).has_proposal(), "2 of 3 with min 2");
    assert!(!decide_per_sentence(&two, 0, &th(), 3).has_proposal(), "2 of 3 with min 3");

    // One of three.
    let one = round([5.0, 2.0, 2.0]);
    assert!(!decide_per_sentence(&one, 0, &th(), 2).has_proposal(), "1 of 3 with min 2");
    assert!(decide_per_sentence(&one, 0, &th(), 1).has_proposal(), "1 of 3 with min 1");
}

/// Sentences are never pooled: a per-sentence failure is attributed to its own sentence.
#[test]
fn failures_are_attributed_to_the_sentence_they_occurred_on() {
    let r = decide_per_sentence(&round([5.0, 5.0, 2.0]), 0, &th(), 2);
    let (_, cleared, per) = r.detail.iter().find(|(c, _, _)| c.phrase == "new").expect("detail");
    assert_eq!(*cleared, 2);
    let failed: Vec<&str> =
        per.iter().filter(|p| !p.passed).map(|p| p.sentence.as_str()).collect();
    assert_eq!(failed, vec!["tune-s2"], "the failing sentence must be named");
    assert!(matches!(
        per.iter().find(|p| p.sentence == "tune-s2").unwrap().reject,
        Some(Reject::NoMarginOverIncumbent { .. })
    ));
}

/// Clearing the tune split on enough sentences is not enough — the holdout must clear too,
/// on its own sentences. This is the overfitting the real `[sad]` round exhibited: a
/// candidate beat the incumbent on tune and lost on holdout.
#[test]
fn tune_breadth_does_not_substitute_for_holdout_breadth() {
    let mut v = round([9.0, 9.0, 9.0]);
    for x in v.iter_mut().filter(|x| {
        x.m.arm == TuneArm::Candidate && x.m.split == Split::Holdout
    }) {
        x.m.cued_delta = 2.0; // no margin on any holdout sentence
    }
    assert!(!decide_per_sentence(&v, 0, &th(), 2).has_proposal());
}

/// A sham that activates on ANY sentence voids the round — multiplicity is already paid
/// for in the corrected alpha, so one hit is one too many.
#[test]
fn a_sham_activating_on_any_single_sentence_voids_the_round() {
    let mut v = round([5.0, 5.0, 5.0]);
    let i = v.iter().position(|x| x.sentence == "tune-s1" && x.m.arm == TuneArm::Sham).unwrap();
    v[i].m.p_vs_plain = 0.0001;
    let r = decide_per_sentence(&v, 0, &th(), 2);
    assert!(matches!(r.void, Some(Void::ShamActivated { .. })));
    assert!(r.proposed.is_none());
}

/// The sham void is INCLUSIVE at alpha: p == alpha voids. Both sides, because `<=` and `<`
/// are one mutation apart and only an exact-boundary case separates them.
#[test]
fn a_sham_exactly_at_alpha_voids_the_round() {
    let t = th();
    assert_eq!(t.corrected_alpha(), 0.005);

    let mut at = round([5.0, 5.0, 5.0]);
    let i = at.iter().position(|x| x.sentence == "tune-s1" && x.m.arm == TuneArm::Sham).unwrap();
    at[i].m.p_vs_plain = 0.005;
    assert!(
        matches!(decide_per_sentence(&at, 0, &t, 2).void, Some(Void::ShamActivated { .. })),
        "a sham at exactly alpha must void"
    );

    let mut past = round([5.0, 5.0, 5.0]);
    let j = past.iter().position(|x| x.sentence == "tune-s1" && x.m.arm == TuneArm::Sham).unwrap();
    past[j].m.p_vs_plain = 0.00625;
    assert!(decide_per_sentence(&past, 0, &t, 2).void.is_none(), "just past alpha must not");
}

/// The counter-cue must lose on a MAJORITY, not everywhere — one noisy sentence should not
/// discard an otherwise sound round, and it should not be able to rescue a bad one either.
#[test]
fn the_counter_cue_voids_on_a_majority_not_on_a_single_sentence() {
    let mut one = round([5.0, 5.0, 5.0]);
    one.push(s("tune-s0", TuneArm::CounterCue, Split::Tune, Some("wrong"), 9.0));
    assert!(decide_per_sentence(&one, 0, &th(), 2).void.is_none(), "1 of 3 must not void");

    let mut two = round([5.0, 5.0, 5.0]);
    two.push(s("tune-s0", TuneArm::CounterCue, Split::Tune, Some("wrong"), 9.0));
    two.push(s("tune-s1", TuneArm::CounterCue, Split::Tune, Some("wrong"), 9.0));
    assert!(
        matches!(decide_per_sentence(&two, 0, &th(), 2).void, Some(Void::CounterCueWon { .. })),
        "2 of 3 must void"
    );
}

/// A missing incumbent on any sentence voids: without it there is nothing to have a margin
/// over, and silently skipping that sentence would quietly lower the bar.
#[test]
fn a_missing_incumbent_on_any_sentence_voids_the_round() {
    let v: Vec<_> = round([5.0, 5.0, 5.0])
        .into_iter()
        .filter(|x| !(x.sentence == "tune-s1" && x.m.arm == TuneArm::Incumbent && x.m.split == Split::Tune))
        .collect();
    assert!(matches!(
        decide_per_sentence(&v, 0, &th(), 2).void,
        Some(Void::IncumbentNotRemeasured)
    ));
}

#[test]
fn holdout_expiry_still_applies() {
    let t = TuneThresholds { max_holdout_uses: 5, ..th() };
    assert!(decide_per_sentence(&round([5.0; 3]), 4, &t, 2).void.is_none());
    assert!(matches!(
        decide_per_sentence(&round([5.0; 3]), 5, &t, 2).void,
        Some(Void::HoldoutExpired { .. })
    ));
}

#[test]
fn an_unsafe_phrase_is_rejected_before_any_sentence_is_scored() {
    let mut v = round([5.0, 5.0, 5.0]);
    for x in v.iter_mut().filter(|x| x.m.arm == TuneArm::Candidate) {
        x.m.phrase = Some("speak [sadly]".into());
    }
    let r = decide_per_sentence(&v, 0, &th(), 2);
    assert!(!r.has_proposal());
    assert!(matches!(r.detail[0].2[0].reject, Some(Reject::UnsafePhrase { .. })));
}

/// The winner is the one with the broadest HOLDOUT coverage, not the best tune score.
#[test]
fn the_winner_is_ranked_by_holdout_breadth() {
    let mut v = round([9.0, 9.0, 9.0]); // "new": 3 tune, 3 holdout
    for i in 0..3 {
        // "wide" looks worse on tune but clears every holdout sentence.
        v.push(s(&format!("tune-s{i}"), TuneArm::Candidate, Split::Tune, Some("wide"), 4.0));
        v.push(s(&format!("holdout-s{i}"), TuneArm::Candidate, Split::Holdout, Some("wide"), 9.0));
    }
    // Break "new" on two holdout sentences so "wide" has strictly broader coverage.
    for x in v.iter_mut().filter(|x| {
        x.m.phrase.as_deref() == Some("new")
            && x.m.split == Split::Holdout
            && (x.sentence == "holdout-s1" || x.sentence == "holdout-s2")
    }) {
        x.m.cued_delta = 2.0;
    }
    let r = decide_per_sentence(&v, 0, &th(), 1);
    assert_eq!(r.proposed.map(|c| c.phrase), Some("wide".to_string()));
}

/// An empty round is void for want of an incumbent, and proposes nothing.
#[test]
fn an_empty_round_is_void_and_proposes_nothing() {
    let r = decide_per_sentence(&[], 0, &th(), 2);
    assert!(!r.has_proposal());
}
