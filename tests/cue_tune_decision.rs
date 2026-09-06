//! The cue-tuning accept criteria (ADR-0004), frozen.
//!
//! These assertions ARE the Goodhart guards. The loop that uses them needs a GPU and
//! several minutes per candidate, so without this file the guards would live only in code
//! nobody runs and could be relaxed by anyone in a hurry. Every criterion is pinned on both
//! sides of its boundary so the mutation gate cannot flip a comparison and survive.

use syrinx_eval::tune::{
    decide, phrase_is_safe, Reject, Split, TuneArm, TuneMeasurement, TuneThresholds, Void,
};

/// A measurement that passes everything. Individual tests break one thing at a time, so
/// each failure is attributable to the thing it broke.
fn good(arm: TuneArm, split: Split, phrase: Option<&str>, delta: f64) -> TuneMeasurement {
    TuneMeasurement {
        label: "angry".into(),
        phrase: phrase.map(str::to_string),
        arm,
        split,
        p_vs_plain: 0.001,
        p_vs_sham: 0.001,
        cued_delta: delta,
        cued_noise: 1.0,
        largest_gain_label: "angry".into(),
        wer: 0.0,
        speaker_similarity: 0.99,
    }
}

fn th() -> TuneThresholds {
    TuneThresholds { alpha: 0.05, candidates: 10, min_margin: 1.0, ..Default::default() }
}

/// Incumbent at 2.0 on both splits, one candidate at 5.0 on both. Passes.
fn round(cand_delta: f64) -> Vec<TuneMeasurement> {
    vec![
        TuneMeasurement { p_vs_plain: 0.9, ..good(TuneArm::Sham, Split::Tune, None, 0.0) },
        good(TuneArm::Incumbent, Split::Tune, Some("old"), 2.0),
        good(TuneArm::Incumbent, Split::Holdout, Some("old"), 2.0),
        good(TuneArm::Candidate, Split::Tune, Some("new"), cand_delta),
        good(TuneArm::Candidate, Split::Holdout, Some("new"), cand_delta),
    ]
}

// ---------------------------------------------------------------- baseline

#[test]
fn a_candidate_that_clears_everything_is_proposed() {
    let r = decide(&round(5.0), 0, &th());
    assert!(r.void.is_none(), "{:?}", r.void);
    assert_eq!(r.proposed.map(|c| c.phrase), Some("new".to_string()));
}

// ---------------------------------------------------------------- round voids

#[test]
fn a_sham_that_beats_plain_voids_the_whole_round() {
    let mut m = round(5.0);
    m[0].p_vs_plain = 0.0001;
    let r = decide(&m, 0, &th());
    assert!(matches!(r.void, Some(Void::ShamActivated { .. })));
    assert!(r.proposed.is_none(), "a void round must propose nothing");
}

#[test]
fn the_wrong_labels_phrase_beating_the_incumbent_voids_the_round() {
    let mut m = round(5.0);
    // Counter-cue scores 9.0 on `angry` -- above the incumbent's 2.0. The judge is not
    // tracking the target, so nothing measured this round can be trusted.
    m.push(good(TuneArm::CounterCue, Split::Tune, Some("happy phrase"), 9.0));
    let r = decide(&m, 0, &th());
    assert!(matches!(r.void, Some(Void::CounterCueWon { .. })), "{:?}", r.void);
    assert!(r.proposed.is_none());
}

#[test]
fn a_counter_cue_that_loses_does_not_void_the_round() {
    let mut m = round(5.0);
    m.push(good(TuneArm::CounterCue, Split::Tune, Some("happy phrase"), 1.0));
    let r = decide(&m, 0, &th());
    assert!(r.void.is_none());
    assert!(r.proposed.is_some());
}

#[test]
fn the_holdout_expires_at_its_budget_and_is_usable_one_below_it() {
    let t = TuneThresholds { max_holdout_uses: 5, ..th() };
    assert!(decide(&round(5.0), 4, &t).void.is_none(), "4 of 5 uses must still work");
    let r = decide(&round(5.0), 5, &t);
    assert!(
        matches!(r.void, Some(Void::HoldoutExpired { uses: 5, budget: 5 })),
        "a holdout used its full budget must expire: {:?}",
        r.void
    );
}

#[test]
fn a_round_without_a_remeasured_incumbent_is_void() {
    let m: Vec<_> = round(5.0)
        .into_iter()
        .filter(|m| !(m.arm == TuneArm::Incumbent && m.split == Split::Tune))
        .collect();
    let r = decide(&m, 0, &th());
    assert!(matches!(r.void, Some(Void::IncumbentNotRemeasured)));
}

// ---------------------------------------------------------------- criteria

#[test]
fn the_margin_is_required_and_a_tie_keeps_the_incumbent() {
    // Incumbent 2.0, margin 1.0 -> 3.0 is exactly the bar and passes.
    assert!(decide(&round(3.0), 0, &th()).has_proposal(), "exactly at the margin passes");
    // Just under it does not.
    let r = decide(&round(2.9999), 0, &th());
    assert!(!r.has_proposal());
    assert!(matches!(r.rejected[0].1, Reject::NoMarginOverIncumbent { .. }));
    // And an exact tie with the incumbent keeps the incumbent.
    let tie = decide(&round(2.0), 0, &th());
    assert!(!tie.has_proposal(), "a tie must keep the incumbent");
}

#[test]
fn beating_plain_but_not_the_sham_is_perturbation_and_is_rejected() {
    let mut m = round(5.0);
    for x in m.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.p_vs_sham = 0.9;
    }
    let r = decide(&m, 0, &th());
    assert!(matches!(r.rejected[0].1, Reject::PerturbationOnly { .. }));
    assert!(!r.has_proposal());
}

#[test]
fn a_judge_delta_inside_its_noise_is_rejected_and_just_outside_is_not() {
    let mut inside = round(5.0);
    for x in inside.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.cued_noise = 5.0; // delta == noise, the boundary
    }
    let r = decide(&inside, 0, &th());
    assert!(
        matches!(r.rejected[0].1, Reject::InsideJudgeNoise { .. }),
        "delta == noise must be rejected: {:?}",
        r.rejected
    );

    let mut outside = round(5.0);
    for x in outside.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.cued_noise = 4.9999;
    }
    assert!(decide(&outside, 0, &th()).has_proposal(), "delta just above noise must pass");
}

#[test]
fn a_phrase_that_moves_the_wrong_class_most_is_rejected() {
    let mut m = round(5.0);
    for x in m.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.largest_gain_label = "surprised".into();
    }
    let r = decide(&m, 0, &th());
    assert!(matches!(r.rejected[0].1, Reject::WrongClassGainedMost { .. }));
}

#[test]
fn the_wer_veto_is_inclusive_at_the_limit_and_fires_just_past_it() {
    let t = TuneThresholds { max_wer_regression: 0.25, ..th() };
    let mut at = round(5.0);
    for x in at.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.wer = 0.25;
    }
    assert!(decide(&at, 0, &t).has_proposal(), "wer == limit passes");

    let mut past = round(5.0);
    for x in past.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.wer = 0.5;
    }
    assert!(matches!(decide(&past, 0, &t).rejected[0].1, Reject::WerRegressed { .. }));
}

/// The guard that stops "Speak like a frightened old man" from winning: it would change
/// delivery, move `fearful`, keep the words — and destroy the requested voice.
#[test]
fn a_phrase_that_changes_the_speaker_is_rejected_at_and_below_the_floor() {
    let t = TuneThresholds { min_speaker_similarity: 0.95, ..th() };
    let mut at = round(5.0);
    for x in at.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.speaker_similarity = 0.95;
    }
    assert!(decide(&at, 0, &t).has_proposal(), "similarity == floor passes");

    let mut below = round(5.0);
    for x in below.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.speaker_similarity = 0.9375;
    }
    assert!(matches!(decide(&below, 0, &t).rejected[0].1, Reject::SpeakerDrift { .. }));
}

/// The acoustic bars are INCLUSIVE at alpha — p == alpha is significant. Pinned on both
/// sides for both contrasts, because `>` and `>=` are one mutation apart and the two
/// contrasts are separate branches.
#[test]
fn the_acoustic_bars_are_inclusive_at_alpha() {
    let t = th();
    assert_eq!(t.corrected_alpha(), 0.005);

    for (field, name) in [(0usize, "p_vs_plain"), (1usize, "p_vs_sham")] {
        // Exactly at alpha: passes.
        let mut at = round(5.0);
        for x in at.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
            if field == 0 { x.p_vs_plain = 0.005 } else { x.p_vs_sham = 0.005 }
        }
        assert!(decide(&at, 0, &t).has_proposal(), "{name} == alpha must pass");

        // Just past it: rejected, and with the matching reason.
        let mut past = round(5.0);
        for x in past.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
            if field == 0 { x.p_vs_plain = 0.00625 } else { x.p_vs_sham = 0.00625 }
        }
        let r = decide(&past, 0, &t);
        let ok = if field == 0 {
            matches!(r.rejected[0].1, Reject::NoAcousticChange { .. })
        } else {
            matches!(r.rejected[0].1, Reject::PerturbationOnly { .. })
        };
        assert!(ok, "{name} just past alpha must reject: {:?}", r.rejected);
    }
}

/// The sham void is inclusive too: a sham significant exactly AT alpha voids the round.
#[test]
fn a_sham_exactly_at_alpha_voids_the_round() {
    let mut at = round(5.0);
    at[0].p_vs_plain = 0.005;
    assert!(
        matches!(decide(&at, 0, &th()).void, Some(Void::ShamActivated { .. })),
        "a sham at exactly alpha must void"
    );

    let mut past = round(5.0);
    past[0].p_vs_plain = 0.00625;
    assert!(decide(&past, 0, &th()).void.is_none(), "a sham just past alpha must not void");
}

// ---------------------------------------------------------------- holdout

#[test]
fn a_candidate_with_no_holdout_measurement_fails_closed() {
    let m: Vec<_> = round(5.0)
        .into_iter()
        .filter(|m| !(m.arm == TuneArm::Candidate && m.split == Split::Holdout))
        .collect();
    let r = decide(&m, 0, &th());
    assert!(matches!(r.rejected[0].1, Reject::NotConfirmedOnHoldout));
    assert!(!r.has_proposal(), "an absent confirmation must never read as a pass");
}

/// Winning on the split you optimised on is not winning. This is the overfitting case the
/// holdout exists for.
#[test]
fn a_candidate_that_wins_on_tune_but_fails_on_holdout_is_rejected() {
    let mut m = round(5.0);
    for x in m.iter_mut().filter(|x| x.arm == TuneArm::Candidate && x.split == Split::Holdout) {
        x.cued_delta = 2.0; // no margin over the incumbent on the unseen split
    }
    let r = decide(&m, 0, &th());
    assert!(
        matches!(r.rejected[0].1, Reject::FailedOnHoldout { .. }),
        "{:?}",
        r.rejected
    );
    assert!(!r.has_proposal());
}

/// The winner is chosen by HOLDOUT score, not by the score it was selected on.
#[test]
fn the_winner_is_ranked_by_the_split_it_was_not_optimised_on() {
    // Exactly two candidates and no others, so the ordering is unambiguous: "a" looks far
    // better on the split it was selected on, "b" is better on the split it was not.
    let m = vec![
        TuneMeasurement { p_vs_plain: 0.9, ..good(TuneArm::Sham, Split::Tune, None, 0.0) },
        good(TuneArm::Incumbent, Split::Tune, Some("old"), 2.0),
        good(TuneArm::Incumbent, Split::Holdout, Some("old"), 2.0),
        good(TuneArm::Candidate, Split::Tune, Some("a"), 20.0),
        good(TuneArm::Candidate, Split::Holdout, Some("a"), 4.0),
        good(TuneArm::Candidate, Split::Tune, Some("b"), 6.0),
        good(TuneArm::Candidate, Split::Holdout, Some("b"), 15.0),
    ];
    let r = decide(&m, 0, &th());
    assert_eq!(
        r.proposed.map(|c| c.phrase),
        Some("b".to_string()),
        "ranking by the tune split would pick \"a\" (20 vs 6) — it must pick \"b\" (15 vs 4)"
    );
}

// ---------------------------------------------------------------- phrase safety

#[test]
fn an_unsafe_phrase_is_rejected_before_any_measurement_is_believed() {
    for bad in ["say [happy] now", "a<|endofprompt|>b", "two\nlines", " padded", ""] {
        assert!(phrase_is_safe(bad).is_err(), "{bad:?} must be rejected");
    }
    for ok in ["Speak in an angry tone", "Shout this loudly", "用愤怒生气的语气说"] {
        assert!(phrase_is_safe(ok).is_ok(), "{ok:?} must be accepted: {:?}", phrase_is_safe(ok));
    }
    // Both sides of the length bound.
    assert!(phrase_is_safe(&"a".repeat(120)).is_ok());
    assert!(phrase_is_safe(&"a".repeat(121)).is_err());
}

#[test]
fn an_unsafe_phrase_is_rejected_even_when_its_numbers_are_perfect() {
    let mut m = round(5.0);
    for x in m.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.phrase = Some("speak [angrily]".into());
    }
    let r = decide(&m, 0, &th());
    assert!(matches!(r.rejected[0].1, Reject::UnsafePhrase { .. }));
    assert!(!r.has_proposal(), "no measurement may buy a cue-markup leak");
}

// ---------------------------------------------------------------- correction

#[test]
fn alpha_is_corrected_by_the_candidate_count() {
    assert_eq!(th().corrected_alpha(), 0.005);
    let one = TuneThresholds { candidates: 1, ..th() };
    assert_eq!(one.corrected_alpha(), 0.05);
    let zero = TuneThresholds { candidates: 0, ..th() };
    assert_eq!(zero.corrected_alpha(), 0.05, "must not divide by zero");
}

/// With more candidates the bar rises. Same p-value, different verdict — this is what
/// stops a large search from buying an accept with volume alone.
#[test]
fn more_candidates_make_the_acoustic_bar_stricter() {
    let mut m = round(5.0);
    for x in m.iter_mut().filter(|x| x.arm == TuneArm::Candidate) {
        x.p_vs_plain = 0.01;
        x.p_vs_sham = 0.01;
    }
    let few = TuneThresholds { candidates: 1, ..th() }; // alpha 0.05
    assert!(decide(&m, 0, &few).has_proposal(), "p=0.01 clears alpha=0.05");
    let many = TuneThresholds { candidates: 10, ..th() }; // alpha 0.005
    let r = decide(&m, 0, &many);
    assert!(matches!(r.rejected[0].1, Reject::NoAcousticChange { .. }));
}

// ---------------------------------------------------------------- shape

#[test]
fn an_empty_round_proposes_nothing_and_is_void_for_want_of_an_incumbent() {
    let r = decide(&[], 0, &th());
    assert!(matches!(r.void, Some(Void::IncumbentNotRemeasured)));
    assert!(!r.has_proposal());
}
