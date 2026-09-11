//! **PROPOSED, unaccepted** — the sham-only activation rule, frozen against the shipped one.
//!
//! `evaluate_contrast` calls a case content-activated only when the cue clears BOTH
//! `cue vs plain` and `cue vs sham` (ADR-0003 §3). The first certification run
//! (`renders/2026-09-09-c42-certification/`) put that conjunction in question: two cases
//! separated from their sham at the resolution floor of the test (p = 4.1e-5 and 8.2e-5,
//! literally the 1st and 2nd most extreme of 24310 labelings) while separating from plain
//! at only 0.0182 and 0.0038, so the shipped rule reported nothing.
//!
//! This file pins a **second** classification computed from the SAME measurements, so the
//! two criteria can be compared on real numbers rather than argued about. It does not
//! change the shipped verdict: `cue_contrast_gate.rs` still owns `evaluate_contrast`, and
//! `agreement_with_the_shipped_verdict` asserts the two never drift apart on the shipped
//! rule. Which rule C4.2' should use is a maintainer decision under ADR-0001 §7; this test
//! exists to make that decision cheap to check, not to make it.
//!
//! Every boundary is pinned on both sides, and each fixture is ordered so that a
//! `&&` -> `||` or `==` -> `!=` flip inside the contrast lookup selects a DIFFERENT row and
//! changes the answer. Mutation results are recorded in the ADR amendment.

use syrinx_eval::contrast::{
    activated_under, compare_rules, evaluate_contrast, Arm, ArmContrast, ContrastRule,
    ContrastThresholds,
};

fn c(case: &str, arm: Arm, against: Arm, p: f64) -> ArmContrast {
    ArmContrast {
        case_id: case.into(),
        backend: "qwen3-1.7b-customvoice".into(),
        arm,
        against,
        p_value: p,
        effect: 2.0,
        wer: 0.0,
        baseline_wer: 0.0,
    }
}

/// alpha 0.05 over 10 comparisons = 0.005 corrected, so the boundaries below are exact.
fn th() -> ContrastThresholds {
    ContrastThresholds { alpha: 0.05, comparisons: 10, ..Default::default() }
}

/// Three cases that separate the two rules, plus one sham-vs-plain row that a lookup flip
/// would wrongly select.
///
/// The ordering is load-bearing: the first row belongs to `b-plain-only`, so a lookup whose
/// `&&` became `||` matches it while resolving `a-sham-only` and reads the wrong p-value.
fn mixed() -> Vec<ArmContrast> {
    vec![
        c("b-plain-only", Arm::Cue, Arm::Plain, 0.001),
        c("b-plain-only", Arm::Cue, Arm::Sham, 0.900),
        c("a-sham-only", Arm::Cue, Arm::Plain, 0.900),
        c("a-sham-only", Arm::Cue, Arm::Sham, 0.001),
        c("c-both", Arm::Cue, Arm::Plain, 0.001),
        c("c-both", Arm::Cue, Arm::Sham, 0.002),
        c("a-sham-only", Arm::Sham, Arm::Plain, 0.001),
    ]
}

// ------------------------------------------------------------------ the two rules differ

#[test]
fn the_two_rules_disagree_exactly_on_the_cases_that_beat_sham_but_not_plain() {
    let r = compare_rules(&mixed(), &th());
    assert_eq!(r.corrected_alpha, 0.005);
    assert_eq!(r.plain_and_sham, vec!["c-both".to_string()]);
    assert_eq!(r.sham_only, vec!["a-sham-only".to_string(), "c-both".to_string()]);
    assert_eq!(r.divergent, vec!["a-sham-only".to_string()]);
}

/// A case that beats plain but not its sham is admitted by NEITHER rule. Dropping the
/// plain leg does not make the conjunction weaker in that direction — it is the reason
/// `PlainAndSham` is a strict sub-rule of `ShamOnly` rather than a different one.
#[test]
fn beating_plain_alone_is_never_enough_under_either_rule() {
    let r = compare_rules(&mixed(), &th());
    assert!(!r.plain_and_sham.contains(&"b-plain-only".to_string()));
    assert!(!r.sham_only.contains(&"b-plain-only".to_string()));
}

/// `PlainAndSham` adds a conjunct to `ShamOnly`, so its verdict is always a subset. A rule
/// comparison that ever violated this would mean the two lists were computed from
/// different measurements.
#[test]
fn the_shipped_rule_is_a_strict_subrule_of_the_proposed_one() {
    for m in [mixed(), the_2026_09_09_certification_run(), vec![]] {
        let r = compare_rules(&m, &th());
        for id in &r.plain_and_sham {
            assert!(r.sham_only.contains(id), "{id} admitted by the conjunction but not by sham-only");
        }
    }
}

// ------------------------------------------------------------------------- the boundaries

/// Inclusive at the corrected alpha, exclusive just past it, under BOTH rules.
#[test]
fn each_rule_is_inclusive_at_alpha_and_excludes_just_past_it() {
    let at = [c("x", Arm::Cue, Arm::Plain, 0.005), c("x", Arm::Cue, Arm::Sham, 0.005)];
    let past = [c("x", Arm::Cue, Arm::Plain, 0.005), c("x", Arm::Cue, Arm::Sham, 0.0050001)];

    let r_at = compare_rules(&at, &th());
    assert_eq!(r_at.plain_and_sham, vec!["x".to_string()], "p == alpha must activate");
    assert_eq!(r_at.sham_only, vec!["x".to_string()]);

    let r_past = compare_rules(&past, &th());
    assert!(r_past.sham_only.is_empty(), "a sham p just past alpha must not activate");
    assert!(r_past.plain_and_sham.is_empty());

    // The other leg, on the same boundary: sham exactly at alpha, plain just past it.
    let plain_past = [c("x", Arm::Cue, Arm::Plain, 0.0050001), c("x", Arm::Cue, Arm::Sham, 0.005)];
    let r = compare_rules(&plain_past, &th());
    assert!(r.plain_and_sham.is_empty(), "the conjunction must fail when the plain leg fails");
    assert_eq!(r.sham_only, vec!["x".to_string()], "sham-only must not care about the plain leg");
    assert_eq!(r.divergent, vec!["x".to_string()]);
}

// ------------------------------------------------------------------------- failing closed

/// A rule may never be satisfied by a measurement the run did not make.
#[test]
fn a_missing_contrast_is_never_significant_under_either_rule() {
    assert!(!ContrastRule::PlainAndSham.admits(None, None, 0.005));
    assert!(!ContrastRule::ShamOnly.admits(None, None, 0.005));

    let sig = c("x", Arm::Cue, Arm::Plain, 0.001);
    // Plain measured, sham absent: both rules must refuse.
    assert!(!ContrastRule::PlainAndSham.admits(Some(&sig), None, 0.005));
    assert!(!ContrastRule::ShamOnly.admits(Some(&sig), None, 0.005));

    // Sham measured and significant, plain absent: only the proposed rule admits.
    let sham_sig = c("x", Arm::Cue, Arm::Sham, 0.001);
    assert!(!ContrastRule::PlainAndSham.admits(None, Some(&sham_sig), 0.005));
    assert!(ContrastRule::ShamOnly.admits(None, Some(&sham_sig), 0.005));
}

#[test]
fn a_run_with_no_contrasts_admits_nothing_under_either_rule() {
    let r = compare_rules(&[], &th());
    assert!(r.plain_and_sham.is_empty());
    assert!(r.sham_only.is_empty());
    assert!(r.divergent.is_empty());
}

// ------------------------------------------------------- agreement with the shipped gate

/// The proposed rule is additive: on `PlainAndSham` the sibling must reproduce
/// `evaluate_contrast`'s own verdict exactly, from the same measurements. If this ever
/// fails, the two have drifted and the comparison is meaningless.
#[test]
fn agreement_with_the_shipped_verdict() {
    for m in [mixed(), the_2026_09_09_certification_run()] {
        let mut shipped = evaluate_contrast(&m, &[], &[], &th()).content_activated;
        shipped.sort();
        assert_eq!(
            activated_under(&m, th().corrected_alpha(), ContrastRule::PlainAndSham),
            shipped,
            "the sibling must compute the shipped rule, not a lookalike"
        );
    }
}

// --------------------------------------------------------------------- the real numbers

/// The measured C4.2' certification run, 2026-09-09, n=9, alpha 0.05/24 = 0.00208.
/// `renders/2026-09-09-c42-certification/report.json`, verbatim.
fn the_2026_09_09_certification_run() -> Vec<ArmContrast> {
    let mut v = Vec::new();
    for (id, cue_plain, cue_sham) in [
        ("en-emotion-happy-leading", 0.068_325_791_855_203_61, 0.059_810_777_457_836_28),
        ("en-emotion-sad-mid", 0.003_825_586_178_527_355, 8.227_067_050_596_462e-5),
        ("en-emotion-angry-trailing", 0.867_749_897_161_661_8, 0.830_028_794_734_677_1),
        ("en-emotion-calm-mid", 0.159_440_559_440_559_43, 0.082_476_347_182_229_53),
        ("en-style-whisper-leading", 0.004_154_668_860_551_214, 0.206_787_330_316_742_07),
        ("en-style-shout-mid", 0.018_222_953_517_071_164, 4.113_533_525_298_231e-5),
    ] {
        v.push(c(id, Arm::Cue, Arm::Plain, cue_plain));
        v.push(c(id, Arm::Cue, Arm::Sham, cue_sham));
    }
    // The three distinct sham arms — one per carrier text, NOT one per case. The five
    // `mid`/`leading` sentinels share two sham render sets between them, which is why the
    // run has three independent sham measurements and not six.
    v.push(c("en-emotion-happy-leading", Arm::Sham, Arm::Plain, 0.479_185_520_361_990_97));
    v.push(c("en-emotion-sad-mid", Arm::Sham, Arm::Plain, 0.023_282_599_753_187_99));
    v.push(c("en-emotion-angry-trailing", Arm::Sham, Arm::Plain, 0.806_705_059_646_236_1));
    v
}

/// The whole question, on the measured data: at the run's own corrected alpha the shipped
/// rule admits **nothing**, and the proposed rule admits **exactly the two cases whose cue
/// separated from its sham at the permutation test's resolution floor**. Both cases fall
/// out of the shipped verdict on the `cue vs plain` leg alone.
#[test]
fn on_the_measured_run_the_rules_disagree_on_sad_mid_and_shout_mid() {
    let m = the_2026_09_09_certification_run();
    let t = ContrastThresholds { alpha: 0.05, comparisons: 24, ..Default::default() };
    let r = compare_rules(&m, &t);

    assert!(r.corrected_alpha > 0.002_083_333_3 && r.corrected_alpha < 0.002_083_333_4);
    assert!(r.plain_and_sham.is_empty(), "the shipped rule found nothing: {:?}", r.plain_and_sham);
    assert_eq!(
        r.sham_only,
        vec!["en-emotion-sad-mid".to_string(), "en-style-shout-mid".to_string()]
    );
    assert_eq!(r.divergent, r.sham_only, "every sham-only hit is a divergence on this run");

    // And the shipped gate agrees it found nothing — not even a perturbation-only case,
    // because nothing cleared `cue vs plain` either.
    let shipped = evaluate_contrast(&m, &[], &[], &t);
    assert!(shipped.content_activated.is_empty());
    assert!(shipped.perturbation_only.is_empty());
}

/// The two divergent cases are the two whose sham arm is shared with a third case that
/// does NOT diverge. Recorded because it is the reason the run is one observation and not
/// three: `sad-mid`, `calm-mid` and `shout-mid` are three cue arms against ONE sham arm.
#[test]
fn the_divergent_cases_share_their_sham_arm_with_a_non_divergent_one() {
    let m = the_2026_09_09_certification_run();
    let t = ContrastThresholds { alpha: 0.05, comparisons: 24, ..Default::default() };
    let r = compare_rules(&m, &t);
    assert!(r.divergent.contains(&"en-emotion-sad-mid".to_string()));
    assert!(r.divergent.contains(&"en-style-shout-mid".to_string()));
    assert!(
        !r.divergent.contains(&"en-emotion-calm-mid".to_string()),
        "calm-mid shares the same sham renders and does NOT clear — the divergence is not \
         a property of the sham arm alone"
    );
}
