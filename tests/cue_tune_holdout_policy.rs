//! What the holdout split is allowed to require — frozen.
//!
//! # The measurement that raised the question
//!
//! `renders/2026-09-09-tune-sad/FINDINGS.md`, `[sad]` at n=8 on `1.7B-CustomVoice`. The
//! **incumbent** — the phrase that actually ships, `"Speak in a sad, sorrowful tone"` —
//! cleared 2 of 3 tune sentences and **0 of 3 holdout sentences** (p = 0.0284, 0.0519,
//! 0.1206 against a corrected alpha, with `other`/`neutral` the top-gaining class on two
//! of the three). Under the aggregation frozen in `tests/cue_tune_per_sentence.rs` a
//! challenger must therefore clear 2 of 3 sentences on which the shipping phrase clears
//! none.
//!
//! That is not confirmation of a selection. It is a second, harder qualifying exam on
//! sentences that happen to be hard — and the sentence-dependence sweeps
//! (`renders/2026-09-06-sad-sentences/`, `renders/2026-09-06-angry-sentences/`) show
//! between-sentence variance large enough to flip a cue's verdict, so a three-sentence
//! partition where the channel barely works is an ordinary draw, not a freak one.
//!
//! # What is pinned here
//!
//! `syrinx_eval::tune::HoldoutPolicy`, the four ways that requirement can be written, and
//! in particular that `decide_per_sentence` is **exactly** the `Absolute` arm — so the
//! 2026-09-11 work is additive and the frozen per-sentence test keeps meaning what it
//! meant. Every boundary is asserted on both sides, because the difference between these
//! policies is entirely a matter of which comparison operator runs.
//!
//! ADR-0004 §PROPOSED (2026-09-11) argues the options. Nothing here accepts anything: the
//! recommended default is a *recommendation*, and choosing a policy for a round is a
//! pre-registration a human makes before the round, never after seeing the numbers.

use syrinx_eval::tune::{
    decide_per_sentence, decide_per_sentence_with_policy, HoldoutPolicy, PerSentenceReport,
    SentenceMeasurement, Split, TuneArm, TuneMeasurement, TuneThresholds, Void,
};

/// The corrected alpha these fixtures run at: 0.05 / 10 candidates.
const ALPHA: f64 = 0.005;

/// The incumbent's real acoustic p-value on the first 2026-09-09 holdout sentence. Above
/// `ALPHA`, which is precisely why the partition could not confirm anything.
const MEASURED_INCUMBENT_MISS: f64 = 0.0284;

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

fn s(
    sentence: &str,
    arm: TuneArm,
    split: Split,
    phrase: Option<&str>,
    d: f64,
) -> SentenceMeasurement {
    SentenceMeasurement { sentence: sentence.into(), m: m(arm, split, phrase, d) }
}

fn th() -> TuneThresholds {
    TuneThresholds { alpha: 0.05, candidates: 10, min_margin: 1.0, ..Default::default() }
}

/// Three tune sentences and three **different** holdout sentences, as the driver produces.
///
/// * `cand_tune[i]` / `cand_hold[i]` — the candidate's judge delta on tune/holdout
///   sentence `i`. The incumbent sits at 2.0 everywhere and `min_margin` is 1.0, so a
///   delta of 5.0 clears and 2.0 does not.
/// * `inc_fit[i]` — whether the incumbent itself clears holdout sentence `i`. It is made
///   to fail the way it actually failed: its own acoustic p-value above the corrected
///   alpha. Its `cued_delta` stays 2.0 either way, so the bar a candidate must clear on
///   that sentence is unchanged — only the partition's *fitness* moves.
fn round(cand_tune: [f64; 3], cand_hold: [f64; 3], inc_fit: [bool; 3]) -> Vec<SentenceMeasurement> {
    let mut v = Vec::new();
    for i in 0..3 {
        let (t, h) = (format!("tune-s{i}"), format!("holdout-s{i}"));

        let mut sham = s(&t, TuneArm::Sham, Split::Tune, None, 0.0);
        sham.m.p_vs_plain = 0.9;
        v.push(sham);
        v.push(s(&t, TuneArm::Incumbent, Split::Tune, Some("old"), 2.0));
        v.push(s(&t, TuneArm::Candidate, Split::Tune, Some("new"), cand_tune[i]));

        let mut inc = s(&h, TuneArm::Incumbent, Split::Holdout, Some("old"), 2.0);
        if !inc_fit[i] {
            inc.m.p_vs_plain = MEASURED_INCUMBENT_MISS;
        }
        v.push(inc);
        v.push(s(&h, TuneArm::Candidate, Split::Holdout, Some("new"), cand_hold[i]));
    }
    v
}

fn decide(v: &[SentenceMeasurement], policy: HoldoutPolicy) -> PerSentenceReport {
    decide_per_sentence_with_policy(v, 0, &th(), 2, policy)
}

// ---------------------------------------------------------------- the fixture is honest

/// The fixture must actually model the 2026-09-09 failure, or none of this tests anything.
/// The incumbent's miss has to sit ABOVE the corrected alpha and its hit BELOW it.
#[test]
fn the_fixture_models_the_measured_holdout_failure() {
    assert_eq!(th().corrected_alpha(), ALPHA);
    assert!(MEASURED_INCUMBENT_MISS > ALPHA, "a 'miss' that clears alpha models nothing");
    assert!(0.001 < ALPHA, "a 'hit' that misses alpha models nothing");

    let v = round([5.0; 3], [5.0; 3], [true, true, false]);
    let tune: Vec<&str> =
        v.iter().filter(|x| x.m.split == Split::Tune).map(|x| x.sentence.as_str()).collect();
    let hold: Vec<&str> =
        v.iter().filter(|x| x.m.split == Split::Holdout).map(|x| x.sentence.as_str()).collect();
    assert!(tune.iter().all(|t| !hold.contains(t)), "the splits must use disjoint ids");
}

// ---------------------------------------------------------------- additivity

/// **The additivity guarantee.** `decide_per_sentence` is defined to be the `Absolute`
/// arm, so `tests/cue_tune_per_sentence.rs` keeps meaning exactly what it meant. Asserted
/// over the whole cross-product of partition fitness and candidate strength, including the
/// rounds where the policies disagree with each other — a single happy-path comparison
/// would pass even if the delegation had quietly gained a condition.
#[test]
fn decide_per_sentence_is_exactly_the_absolute_policy() {
    for fit in [[true, true, true], [true, true, false], [true, false, false], [false; 3]] {
        for ct in [[5.0, 5.0, 5.0], [5.0, 5.0, 2.0], [5.0, 2.0, 2.0], [2.0; 3]] {
            for ch in [[5.0, 5.0, 5.0], [5.0, 5.0, 2.0], [5.0, 2.0, 2.0], [2.0; 3]] {
                let v = round(ct, ch, fit);
                assert_eq!(
                    decide_per_sentence(&v, 0, &th(), 2),
                    decide(&v, HoldoutPolicy::Absolute),
                    "fit {fit:?} tune {ct:?} hold {ch:?}: the wrapper is not the Absolute arm"
                );
            }
        }
    }
}

/// And the delegation carries the round-level voids too, not merely the accept path.
#[test]
fn the_wrapper_and_the_absolute_arm_agree_on_voided_rounds() {
    let v = round([5.0; 3], [5.0; 3], [true; 3]);
    let t = th();
    assert_eq!(
        decide_per_sentence(&v, 5, &t, 2),
        decide_per_sentence_with_policy(&v, 5, &t, 2, HoldoutPolicy::Absolute),
    );
    assert!(matches!(
        decide_per_sentence_with_policy(&v, 5, &t, 2, HoldoutPolicy::Absolute).void,
        Some(Void::HoldoutExpired { .. })
    ));
}

// ---------------------------------------------------------------- RequireFitPartition

/// **The 2026-09-09 round, replayed.** The incumbent clears no holdout sentence.
///
/// `Absolute` proceeds and scores candidates against that partition — which is the
/// behaviour in question, not a bug. `RequireFitPartition` says the partition cannot
/// answer the question and voids, naming the count. Both arms asserted on the same input,
/// so neither the policy test nor its negation can be mutated away.
#[test]
fn a_partition_the_incumbent_cannot_pass_voids_only_under_require_fit() {
    let v = round([5.0; 3], [5.0; 3], [false; 3]);

    let absolute = decide(&v, HoldoutPolicy::Absolute);
    assert!(absolute.void.is_none(), "Absolute must not void: {:?}", absolute.void);
    assert!(absolute.has_proposal(), "Absolute scores against the hard partition");

    let fit = decide(&v, HoldoutPolicy::RequireFitPartition);
    assert_eq!(
        fit.void,
        Some(Void::HoldoutPartitionUnfit { incumbent_cleared: 0, of: 3, required: 2 }),
        "the void must name what the incumbent managed and what was required"
    );
    assert!(!fit.has_proposal(), "a void round proposes nothing");
    assert!(fit.detail.is_empty(), "a void round scores nothing");
    assert_eq!(fit.required, 2);
}

/// Both sides of the fitness boundary. Exactly `min_sentences` is FIT; one fewer is not.
/// `<` and `<=` are one mutation apart and only the exact-boundary case separates them.
#[test]
fn the_fitness_boundary_is_inclusive_at_min_sentences() {
    let at = round([5.0; 3], [5.0, 5.0, 2.0], [true, true, false]); // incumbent clears 2 of 3
    let r = decide(&at, HoldoutPolicy::RequireFitPartition);
    assert!(r.void.is_none(), "2 of 3 with min 2 must NOT void: {:?}", r.void);
    assert!(r.has_proposal());

    let below = round([5.0; 3], [5.0, 5.0, 2.0], [true, false, false]); // clears 1 of 3
    assert_eq!(
        decide(&below, HoldoutPolicy::RequireFitPartition).void,
        Some(Void::HoldoutPartitionUnfit { incumbent_cleared: 1, of: 3, required: 2 }),
        "1 of 3 with min 2 must void"
    );
}

/// On a partition the incumbent can pass, `RequireFitPartition` is `Absolute` — it moves
/// no bar and adds no lever. That is the whole argument for preferring it, so it is pinned
/// rather than asserted in prose: every candidate strength, same verdict.
#[test]
fn on_a_fit_partition_require_fit_is_indistinguishable_from_absolute() {
    for ch in [[5.0, 5.0, 5.0], [5.0, 5.0, 2.0], [5.0, 2.0, 2.0], [2.0; 3]] {
        let v = round([5.0; 3], ch, [true; 3]);
        assert_eq!(
            decide(&v, HoldoutPolicy::Absolute),
            decide(&v, HoldoutPolicy::RequireFitPartition),
            "hold {ch:?}: RequireFitPartition changed a verdict on a fit partition"
        );
    }
}

/// The fitness void is checked LAST, after the voids about round integrity. A round that
/// is both expired and unfit reports the expiry: the budget is the graver fact, and a
/// stable order is what makes a void report reproducible.
#[test]
fn an_expired_holdout_outranks_an_unfit_one() {
    let v = round([5.0; 3], [5.0; 3], [false; 3]);
    let t = th();
    assert!(matches!(
        decide_per_sentence_with_policy(&v, 5, &t, 2, HoldoutPolicy::RequireFitPartition).void,
        Some(Void::HoldoutExpired { .. })
    ));
    assert!(matches!(
        decide_per_sentence_with_policy(&v, 4, &t, 2, HoldoutPolicy::RequireFitPartition).void,
        Some(Void::HoldoutPartitionUnfit { .. })
    ));
}

// ---------------------------------------------------------------- StrictlyBroaderThanIncumbent

/// The relative arm: strictly more holdout sentences than the incumbent. All three sides
/// of that comparison — greater, equal, fewer — because `>` mutates to `>=` and to `<`.
#[test]
fn strictly_broader_requires_more_than_the_incumbent_not_as_many() {
    // Incumbent clears 1 of 3 (holdout-s0).
    let more = round([5.0; 3], [5.0, 5.0, 2.0], [true, false, false]); // candidate clears 2
    assert!(
        decide(&more, HoldoutPolicy::StrictlyBroaderThanIncumbent).has_proposal(),
        "2 > 1 must propose"
    );

    let equal = round([5.0; 3], [5.0, 2.0, 2.0], [true, false, false]); // candidate clears 1
    assert!(
        !decide(&equal, HoldoutPolicy::StrictlyBroaderThanIncumbent).has_proposal(),
        "1 > 1 is false — matching the incumbent's breadth is not beating it"
    );

    let fewer = round([5.0; 3], [2.0; 3], [true, true, false]); // candidate clears 0, incumbent 2
    assert!(
        !decide(&fewer, HoldoutPolicy::StrictlyBroaderThanIncumbent).has_proposal(),
        "0 > 2 is false"
    );
}

/// **Why this arm was not chosen**, pinned as behaviour rather than left as an opinion:
/// when the incumbent clears nothing the bar collapses to "clear one sentence", and one
/// sentence is exactly what the per-sentence design exists to stop being decisive.
#[test]
fn strictly_broader_collapses_to_a_single_sentence_when_the_incumbent_clears_none() {
    let v = round([5.0; 3], [5.0, 2.0, 2.0], [false; 3]); // candidate clears 1, incumbent 0
    assert!(
        !decide(&v, HoldoutPolicy::Absolute).has_proposal(),
        "Absolute wants 2 of 3 and gets 1"
    );
    assert!(
        decide(&v, HoldoutPolicy::StrictlyBroaderThanIncumbent).has_proposal(),
        "1 > 0 proposes — the bar is now one sentence"
    );
    // It does not void either: this arm answers the question by lowering the bar, which is
    // precisely the difference from RequireFitPartition.
    assert!(decide(&v, HoldoutPolicy::StrictlyBroaderThanIncumbent).void.is_none());
}

// ---------------------------------------------------------------- OnlyWhereIncumbentClears

/// The paired arm counts only the holdout sentences the incumbent also clears. A candidate
/// that wins on an INELIGIBLE sentence gains nothing by it — the discriminating case,
/// since dropping the eligibility filter makes this round pass.
#[test]
fn the_paired_arm_counts_only_sentences_the_incumbent_also_clears() {
    // Incumbent clears s0 and s1. The candidate clears s0 and s2 — two sentences, but only
    // one of them eligible.
    let v = round([5.0; 3], [5.0, 2.0, 5.0], [true, true, false]);
    assert!(
        decide(&v, HoldoutPolicy::Absolute).has_proposal(),
        "Absolute counts both, so 2 of 3 clears"
    );
    assert!(
        !decide(&v, HoldoutPolicy::OnlyWhereIncumbentClears).has_proposal(),
        "only 1 of the 2 eligible sentences cleared"
    );

    // The other direction, so an inverted filter dies too: the candidate clears every
    // sentence, and the eligible pair is enough.
    let both = round([5.0; 3], [5.0; 3], [true, true, false]);
    assert!(decide(&both, HoldoutPolicy::OnlyWhereIncumbentClears).has_proposal());
}

/// **Why this arm was not chosen.** On the round that raised the whole question the
/// eligible set is empty, so it fails closed and says nothing — an unanswerable partition
/// must never read as a pass, but neither does it answer anything.
#[test]
fn the_paired_arm_fails_closed_on_an_empty_eligible_set() {
    let v = round([5.0; 3], [5.0; 3], [false; 3]);
    assert!(decide(&v, HoldoutPolicy::Absolute).has_proposal());
    assert!(
        !decide(&v, HoldoutPolicy::OnlyWhereIncumbentClears).has_proposal(),
        "an empty conjunction must not be vacuously true"
    );
    assert!(
        decide(&v, HoldoutPolicy::OnlyWhereIncumbentClears).void.is_none(),
        "and it fails quietly rather than voiding, which is the objection to it"
    );
}

// ---------------------------------------------------------------- the tune side is untouched

/// The policy governs the HOLDOUT requirement only. Tune breadth is still required under
/// every arm — the conjunction must not become a disjunction, which is the single most
/// damaging mutation available here.
#[test]
fn no_policy_lets_holdout_breadth_substitute_for_tune_breadth() {
    for (policy, fit) in [
        (HoldoutPolicy::Absolute, [true; 3]),
        (HoldoutPolicy::RequireFitPartition, [true; 3]),
        (HoldoutPolicy::StrictlyBroaderThanIncumbent, [true, false, false]),
        (HoldoutPolicy::OnlyWhereIncumbentClears, [true; 3]),
    ] {
        // Nothing clears on tune; everything clears on holdout.
        let v = round([2.0; 3], [5.0; 3], fit);
        let r = decide(&v, policy);
        assert!(r.void.is_none(), "{policy:?}: {:?}", r.void);
        assert!(!r.has_proposal(), "{policy:?}: tune breadth was not required");
    }
}

/// And the tune-side threshold is still inclusive at `min_sentences` under every policy —
/// both sides, so `>=` cannot become `>`.
///
/// The fitness pattern differs per arm only so that each arm's HOLDOUT requirement is
/// satisfied and the tune side is the one thing under test.
#[test]
fn tune_breadth_is_inclusive_at_min_sentences_under_every_policy() {
    for (policy, fit) in [
        (HoldoutPolicy::Absolute, [true; 3]),
        (HoldoutPolicy::RequireFitPartition, [true; 3]),
        (HoldoutPolicy::StrictlyBroaderThanIncumbent, [true, false, false]),
        (HoldoutPolicy::OnlyWhereIncumbentClears, [true; 3]),
    ] {
        let at = round([5.0, 5.0, 2.0], [5.0; 3], fit);
        assert!(decide(&at, policy).has_proposal(), "{policy:?}: 2 of 3 tune with min 2");

        let below = round([5.0, 2.0, 2.0], [5.0; 3], fit);
        assert!(!decide(&below, policy).has_proposal(), "{policy:?}: 1 of 3 tune with min 2");
    }
}
