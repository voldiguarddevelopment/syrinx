//! C4.2′ decision logic (ADR-0003), frozen.
//!
//! The numbers this gate consumes cost 26 minutes of GPU. The *decision* costs nothing, so
//! it lives here on the model-free board: a future pass that wants to relax a clause has
//! to get past a test to do it, rather than editing a threshold in a file nobody runs.
//!
//! Every clause is pinned on **both sides** of its boundary — at the threshold and just
//! past it — because the mutation gate flips the comparison operators and a happy-path
//! assertion leaves the mutant alive.

use syrinx_eval::contrast::{
    evaluate_contrast, Arm, ArmContrast, ContrastThresholds, ContrastViolation, PipelineControl,
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

/// alpha 0.05 over 10 comparisons = 0.005 corrected. Round numbers so the boundary tests
/// below are exact rather than approximately exact.
fn th() -> ContrastThresholds {
    ContrastThresholds { alpha: 0.05, comparisons: 10, ..Default::default() }
}

// ---------------------------------------------------------------- correction

#[test]
fn alpha_is_bonferroni_corrected_by_the_comparison_count() {
    assert_eq!(th().corrected_alpha(), 0.005);
    let one = ContrastThresholds { alpha: 0.05, comparisons: 1, ..Default::default() };
    assert_eq!(one.corrected_alpha(), 0.05);
    // A zero count must not divide by zero.
    let zero = ContrastThresholds { alpha: 0.05, comparisons: 0, ..Default::default() };
    assert_eq!(zero.corrected_alpha(), 0.05);
}

#[test]
fn significance_is_inclusive_at_the_threshold_and_excludes_just_past_it() {
    let at = c("x", Arm::Cue, Arm::Plain, 0.005);
    let past = c("x", Arm::Cue, Arm::Plain, 0.0050001);
    assert!(at.significant_at(0.005), "p == alpha must count as significant");
    assert!(!past.significant_at(0.005), "p just above alpha must not");
}

// ---------------------------------------------------------------- clause A1

#[test]
fn a1_pipeline_control_must_render_identically() {
    let ok = [PipelineControl { backend: "qwen3-0.6b-customvoice".into(), renders_identical: true }];
    let bad =
        [PipelineControl { backend: "qwen3-0.6b-customvoice".into(), renders_identical: false }];

    assert!(evaluate_contrast(&[], &ok, &[], &th()).passed());

    let r = evaluate_contrast(&[], &bad, &[], &th());
    assert!(!r.passed());
    assert!(matches!(
        r.violations[0],
        ContrastViolation::PipelineControlBroken { .. }
    ));
}

// ---------------------------------------------------------------- clause A2

/// The derived budget: ceil(alpha * n_aa) from the NOMINAL alpha. Pinned at values where
/// ceil and floor disagree, because an earlier formula collapsed to a constant 1 and only
/// a surviving `ceil -> floor` mutant revealed it.
#[test]
fn a2_the_derived_budget_is_ceil_alpha_times_the_aa_count() {
    let t = ContrastThresholds { alpha: 0.05, comparisons: 10, ..Default::default() };
    assert_eq!(t.aa_budget(0), 0, "no A/A contrasts, no budget");
    assert_eq!(t.aa_budget(8), 1, "ceil(0.40) = 1, floor would be 0");
    assert_eq!(t.aa_budget(30), 2, "ceil(1.50) = 2, floor would be 1");
    assert_eq!(t.aa_budget(20), 1, "ceil(1.00) = 1 exactly on the integer");
    // An explicit budget overrides the derivation entirely.
    let fixed = ContrastThresholds { max_aa_activations: Some(7), ..t.clone() };
    assert_eq!(fixed.aa_budget(30), 7);
}

/// The derived budget is what actually gates a run when none is set: 8 A/A contrasts give
/// a budget of 1, so one activation passes and two fail.
#[test]
fn a2_the_derived_budget_gates_a_run_with_no_explicit_budget() {
    let t = ContrastThresholds { alpha: 0.05, comparisons: 10, ..Default::default() };
    assert!(t.max_aa_activations.is_none());
    let mut m: Vec<ArmContrast> = (0..8)
        .map(|i| c(&format!("aa{i}"), Arm::ControlA, Arm::ControlB, 0.9))
        .collect();
    m[0].p_value = 0.001;
    assert!(evaluate_contrast(&m, &[], &[], &t).passed(), "1 of 8 is within ceil(0.4)=1");
    m[1].p_value = 0.001;
    let r = evaluate_contrast(&m, &[], &[], &t);
    assert!(matches!(
        r.violations[0],
        ContrastViolation::CalibrationFailed { activations: 2, budget: 1 }
    ));
}

#[test]
fn a2_calibration_tolerates_the_budget_and_fails_one_past_it() {
    let mut t = th();
    t.max_aa_activations = Some(1);

    // Exactly at budget: one A/A activation is tolerated.
    let at = [c("a", Arm::ControlA, Arm::ControlB, 0.001)];
    assert!(evaluate_contrast(&at, &[], &[], &t).passed(), "1 activation with budget 1");

    // One past: fails.
    let past = [
        c("a", Arm::ControlA, Arm::ControlB, 0.001),
        c("b", Arm::ControlA, Arm::ControlB, 0.001),
    ];
    let r = evaluate_contrast(&past, &[], &[], &t);
    assert!(!r.passed());
    assert!(matches!(
        r.violations[0],
        ContrastViolation::CalibrationFailed { activations: 2, budget: 1 }
    ));
}

/// A2 is checked at the NOMINAL alpha, and this is the case that says so: p=0.02 is
/// significant at alpha=0.05 but not at the corrected 0.005. If calibration used the
/// corrected alpha it would almost never fire, which is a calibration nobody can trip.
#[test]
fn a2_calibration_uses_the_nominal_alpha_not_the_corrected_one() {
    let t = ContrastThresholds {
        alpha: 0.05,
        comparisons: 10,
        max_aa_activations: Some(0),
        ..Default::default()
    };
    assert_eq!(t.corrected_alpha(), 0.005);
    // Between the two: fires at nominal, silent at corrected.
    let between = [c("aa", Arm::ControlA, Arm::ControlB, 0.02)];
    let r = evaluate_contrast(&between, &[], &[], &t);
    assert!(
        matches!(r.violations[0], ContrastViolation::CalibrationFailed { activations: 1, .. }),
        "p=0.02 must count against a nominal alpha of 0.05: {:?}",
        r.violations
    );

    // Above the nominal alpha: silent either way.
    let above = [c("aa", Arm::ControlA, Arm::ControlB, 0.20)];
    assert!(evaluate_contrast(&above, &[], &[], &t).passed());
}

#[test]
fn a2_a_nonsignificant_aa_contrast_is_not_counted() {
    let mut t = th();
    t.max_aa_activations = Some(0);
    let quiet = [c("a", Arm::ControlA, Arm::ControlB, 0.9)];
    assert!(evaluate_contrast(&quiet, &[], &[], &t).passed());
}

// ---------------------------------------------------------------- clause A3

#[test]
fn a3_a_sham_that_activates_voids_the_run() {
    let quiet = [c("s", Arm::Sham, Arm::Plain, 0.40)];
    assert!(evaluate_contrast(&quiet, &[], &[], &th()).passed());

    let loud = [c("s", Arm::Sham, Arm::Plain, 0.001)];
    let r = evaluate_contrast(&loud, &[], &[], &th());
    assert!(!r.passed());
    assert!(matches!(r.violations[0], ContrastViolation::ShamActivated { .. }));
}

/// The real 2026-09-06 numbers. A regression that made the sham arm fire on this data
/// would be a change in the decision logic, and this pins it.
#[test]
fn a3_the_measured_sham_values_pass() {
    let measured = [
        c("happy", Arm::Sham, Arm::Plain, 0.3164),
        c("sad", Arm::Sham, Arm::Plain, 0.7453),
        c("angry", Arm::Sham, Arm::Plain, 0.0325),
        c("happy-zh", Arm::Sham, Arm::Plain, 0.4059),
        c("sad-zh", Arm::Sham, Arm::Plain, 0.9091),
        c("angry-zh", Arm::Sham, Arm::Plain, 0.2862),
    ];
    let t = ContrastThresholds { alpha: 0.05, comparisons: 21, ..Default::default() };
    let r = evaluate_contrast(&measured, &[], &[], &t);
    assert!(r.passed(), "the measured shams must pass: {:?}", r.violations);
    // And the angry cell, the closest of the six, is genuinely inside — not passing by
    // accident of ordering.
    assert!(0.0325 > t.corrected_alpha());
}

// ------------------------------------------------- the sham/plain distinction

/// The whole reason the sham arm exists: beating plain is not beating sham.
#[test]
fn a_cue_that_beats_plain_but_not_its_sham_is_perturbation_not_content() {
    let m = [
        c("x", Arm::Cue, Arm::Plain, 0.001),
        c("x", Arm::Cue, Arm::Sham, 0.40),
    ];
    let r = evaluate_contrast(&m, &[], &[], &th());
    assert_eq!(r.perturbation_only, vec!["x".to_string()]);
    assert!(r.content_activated.is_empty(), "must NOT count as content activation");
}

#[test]
fn a_cue_that_beats_both_is_content_activation() {
    let m = [
        c("x", Arm::Cue, Arm::Plain, 0.001),
        c("x", Arm::Cue, Arm::Sham, 0.002),
    ];
    let r = evaluate_contrast(&m, &[], &[], &th());
    assert_eq!(r.content_activated, vec!["x".to_string()]);
    assert!(r.perturbation_only.is_empty());
}

/// A cue that fails against plain is in neither list, whatever it did against its sham.
#[test]
fn a_cue_that_does_not_beat_plain_is_in_neither_list() {
    let m = [
        c("x", Arm::Cue, Arm::Plain, 0.40),
        c("x", Arm::Cue, Arm::Sham, 0.001),
    ];
    let r = evaluate_contrast(&m, &[], &[], &th());
    assert!(r.content_activated.is_empty());
    assert!(r.perturbation_only.is_empty());
}

/// A missing sham contrast must fail closed — counted as perturbation, never promoted to
/// content on the strength of an absent measurement.
#[test]
fn a_missing_sham_contrast_fails_closed() {
    let m = [c("x", Arm::Cue, Arm::Plain, 0.001)];
    let r = evaluate_contrast(&m, &[], &[], &th());
    assert_eq!(r.perturbation_only, vec!["x".to_string()]);
    assert!(r.content_activated.is_empty());
}

// ---------------------------------------------------------------- clause B

#[test]
fn b_a_silent_sentinel_fails_and_a_firing_one_passes() {
    let mut t = th();
    t.sentinel = Some("sent".into());

    let fires = [c("sent", Arm::Cue, Arm::Plain, 0.001)];
    assert!(evaluate_contrast(&fires, &[], &[], &t).passed());

    let silent = [c("sent", Arm::Cue, Arm::Plain, 0.20)];
    let r = evaluate_contrast(&silent, &[], &[], &t);
    assert!(matches!(r.violations[0], ContrastViolation::SentinelSilent { .. }));
}

#[test]
fn b_a_sentinel_with_no_measurement_is_a_failure_not_a_pass() {
    let mut t = th();
    t.sentinel = Some("missing".into());
    let other = [c("elsewhere", Arm::Cue, Arm::Plain, 0.001)];
    let r = evaluate_contrast(&other, &[], &[], &t);
    assert!(matches!(r.violations[0], ContrastViolation::SentinelNotMeasured { .. }));
}

/// ADR-0003 records "clause B not enabled" as an acceptable pre-registered outcome.
#[test]
fn b_no_sentinel_means_the_clause_is_not_enforced() {
    let t = th();
    assert!(t.sentinel.is_none());
    assert!(evaluate_contrast(&[], &[], &[], &t).passed());
}

// ---------------------------------------------------------------- clause C

#[test]
fn c_the_wer_veto_is_inclusive_at_the_limit_and_fires_just_past_it() {
    // Exactly representable, deliberately: 0.60 - 0.10 is 0.49999999999999994 in f64, so
    // the obvious spelling of this test never reaches the boundary and lets a `>` -> `>=`
    // mutant survive. It did, and this is the fix.
    let mut at = c("x", Arm::Cue, Arm::Plain, 0.5);
    at.baseline_wer = 0.0;
    at.wer = 0.5;
    assert_eq!(at.wer_delta(), 0.5, "the boundary case must actually sit on the boundary");
    assert!(evaluate_contrast(&[at], &[], &[], &th()).passed(), "delta == limit passes");

    let mut past = c("x", Arm::Cue, Arm::Plain, 0.5);
    past.baseline_wer = 0.0;
    past.wer = 0.5625; // exactly representable, delta 0.5625
    let r = evaluate_contrast(&[past], &[], &[], &th());
    assert!(matches!(r.violations[0], ContrastViolation::WerRegressed { .. }));
}

/// An improvement in WER is never a violation.
#[test]
fn c_a_wer_improvement_is_not_a_violation() {
    let mut better = c("x", Arm::Cue, Arm::Plain, 0.5);
    better.baseline_wer = 0.60;
    better.wer = 0.10;
    assert!(evaluate_contrast(&[better], &[], &[], &th()).passed());
}

// ---------------------------------------------------------------- clause 5

/// An unexpressable kind is reported as not-applicable and never as a zero rate — a
/// fabricated measurement of a channel that does not exist.
#[test]
fn unexpressable_cases_are_not_applicable_and_do_not_fail_the_run() {
    let na = vec!["en-event-laughs".to_string(), "en-event-cough".to_string()];
    let r = evaluate_contrast(&[], &[], &na, &th());
    assert!(r.passed());
    assert_eq!(r.not_applicable.len(), 2);
    assert!(r.content_activated.is_empty());
}

// ---------------------------------------------------------------- composition

/// Violations accumulate — one clause failing must not mask another.
#[test]
fn every_failing_clause_is_reported_not_just_the_first() {
    let mut t = th();
    t.sentinel = Some("sent".into());
    t.max_aa_activations = Some(0);

    let mut wer_bad = c("w", Arm::Cue, Arm::Plain, 0.9);
    wer_bad.baseline_wer = 0.0;
    wer_bad.wer = 0.9;

    let m = [
        c("a", Arm::ControlA, Arm::ControlB, 0.001), // A2
        c("s", Arm::Sham, Arm::Plain, 0.001),        // A3
        c("sent", Arm::Cue, Arm::Plain, 0.9),        // B
        wer_bad,                                     // C
    ];
    let r = evaluate_contrast(&m, &[], &[], &t);
    assert_eq!(r.violations.len(), 4, "all four clauses must report: {:?}", r.violations);
}

/// An empty run passes only because there is nothing to fail — recorded so nobody reads a
/// green empty report as evidence. The runner, not this function, must assert it measured
/// the cases it expected.
#[test]
fn an_empty_run_passes_and_reports_nothing() {
    let r = evaluate_contrast(&[], &[], &[], &th());
    assert!(r.passed());
    assert!(r.content_activated.is_empty());
    assert!(r.perturbation_only.is_empty());
    assert!(r.not_applicable.is_empty());
}
