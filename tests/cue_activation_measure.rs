//! Pins the acoustic feature extractor and the activation test that feed C4.1.
//!
//! `tests/cue_activation_gate.rs` certifies that the harness *aggregates and gates*
//! correctly given measurements. This file certifies the layer underneath it: that a
//! [`Measurement::activated`] flag means what it claims. Two properties matter, and a
//! failure of either would make every activation number meaningless in a way that no
//! amount of GPU time would reveal:
//!
//!   * **Calibration** — two groups drawn from the *same* condition must almost never be
//!     called activated. Without this the harness reports ~100 % activation for a backend
//!     that ignores cues entirely, because sampled renders always differ.
//!   * **Power** — two groups that genuinely differ must be called activated, at the
//!     smallest p-value the design can produce.
//!
//! Everything here is synthetic and deterministic: signals come from a fixed LCG, so
//! these are reproducible facts about the estimator, not a flaky statistical smoke test.

use syrinx_eval::acoustic::{activation_test, features, min_n_for_alpha, Features};

const SR: u32 = 16_000;
const DUR: f64 = 0.6;

/// Deterministic LCG — the numeric-recipes constants. Tests must not depend on the
/// platform's RNG.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Uniform in `[-1, 1]`.
    fn next_sym(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

/// A voice-like harmonic stack at `f0` with a little noise, `dur` seconds long.
fn voiced(f0: f64, amp: f64, dur: f64, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let n = (dur * f64::from(SR)) as usize;
    let n_harm = ((f64::from(SR) / 2.0) / f0).floor().min(20.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(SR);
            let mut s = 0.0;
            for h in 1..=n_harm.max(1) {
                // 1/h rolloff: a sawtooth-ish spectrum, which is what a glottal source
                // looks like and what makes the centroid meaningful.
                s += (2.0 * std::f64::consts::PI * f0 * h as f64 * t).sin() / h as f64;
            }
            (amp * (s + 0.01 * rng.next_sym())) as f32
        })
        .collect()
}

fn noise(amp: f64, dur: f64, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let n = (dur * f64::from(SR)) as usize;
    (0..n).map(|_| (amp * rng.next_sym()) as f32).collect()
}

fn silence(dur: f64) -> Vec<f32> {
    vec![0.0; (dur * f64::from(SR)) as usize]
}

fn feat(samples: &[f32]) -> Features {
    features(samples, SR)
}

fn get(f: &Features, name: &str) -> f64 {
    f.get(name).unwrap_or_else(|| panic!("no feature named {name}"))
}

// ---- feature extraction ------------------------------------------------------------

#[test]
fn duration_is_the_exact_sample_count() {
    assert!((get(&feat(&voiced(150.0, 0.3, 1.0, 1)), "duration_s") - 1.0).abs() < 1e-9);
    assert!((get(&feat(&voiced(150.0, 0.3, 2.5, 1)), "duration_s") - 2.5).abs() < 1e-9);
}

#[test]
fn empty_input_yields_zeros_not_nan() {
    let f = feat(&[]);
    for (i, v) in f.v.iter().enumerate() {
        assert!(v.is_finite(), "dimension {i} is not finite");
        assert_eq!(*v, 0.0, "dimension {i} should be zero for empty input");
    }
}

#[test]
fn input_shorter_than_one_frame_still_reports_duration_and_level() {
    // 10 ms is well under the ~46 ms frame, so the framed dimensions cannot be computed.
    let short = voiced(150.0, 0.4, 0.010, 7);
    let f = feat(&short);
    assert!((get(&f, "duration_s") - 0.010).abs() < 1e-6);
    assert!(get(&f, "rms_mean") > 0.0, "a non-silent signal must have non-zero RMS");
    assert_eq!(get(&f, "f0_mean"), 0.0, "no frame fits, so f0 must stay unset");
}

#[test]
fn f0_tracks_the_generated_pitch_across_the_range() {
    // Both ends of the tracker's search range, so a mutant that clamps or offsets the
    // lag conversion cannot pass by being right in the middle only.
    for (target, tol) in [(90.0, 4.0), (150.0, 4.0), (300.0, 8.0)] {
        let f = feat(&voiced(target, 0.3, DUR, 3));
        let got = get(&f, "f0_mean");
        assert!(
            (got - target).abs() < tol,
            "f0 for a {target} Hz stack came out {got} (tolerance {tol})"
        );
    }
}

#[test]
fn voicing_separates_a_harmonic_stack_from_noise() {
    let tone = get(&feat(&voiced(150.0, 0.3, DUR, 11)), "voiced_ratio");
    let hiss = get(&feat(&noise(0.3, DUR, 11)), "voiced_ratio");
    assert!(tone > 0.8, "a harmonic stack should read as voiced, got {tone}");
    assert!(hiss < 0.3, "white noise should not read as voiced, got {hiss}");
    assert!(tone > hiss);
}

#[test]
fn rms_scales_with_amplitude() {
    let quiet = get(&feat(&voiced(150.0, 0.1, DUR, 5)), "rms_mean");
    let loud = get(&feat(&voiced(150.0, 0.5, DUR, 5)), "rms_mean");
    assert!(quiet > 0.0);
    let ratio = loud / quiet;
    assert!((ratio - 5.0).abs() < 0.5, "5x amplitude should give ~5x RMS, got {ratio}");
}

#[test]
fn pause_structure_counts_runs_not_just_silent_frames() {
    let speech = |seed| voiced(150.0, 0.4, 0.5, seed);
    let mut one_gap = speech(1);
    one_gap.extend(silence(0.4));
    one_gap.extend(speech(2));

    let mut two_gaps = speech(1);
    two_gaps.extend(silence(0.4));
    two_gaps.extend(speech(2));
    two_gaps.extend(silence(0.4));
    two_gaps.extend(speech(3));

    let runs = |s: &[f32]| {
        let f = feat(s);
        (get(&f, "pause_rate_hz") * get(&f, "duration_s")).round()
    };
    assert_eq!(runs(&one_gap), 1.0, "one silent stretch is one pause run");
    assert_eq!(runs(&two_gaps), 2.0, "two silent stretches are two pause runs");

    // The ratio must rise too, but it cannot distinguish the two cases on its own —
    // which is exactly why the rate dimension exists alongside it.
    assert!(get(&feat(&two_gaps), "pause_ratio") > 0.0);
    assert_eq!(get(&feat(&speech(1)), "pause_ratio"), 0.0, "continuous speech has no pause");
}

#[test]
fn brightness_moves_centroid_and_tilt_together() {
    let dark = feat(&voiced(120.0, 0.3, DUR, 9));
    let bright = feat(&voiced(1200.0, 0.3, DUR, 9));
    assert!(
        get(&bright, "centroid_mean") > get(&dark, "centroid_mean"),
        "a higher stack must have a higher spectral centroid"
    );
    assert!(
        get(&bright, "spectral_tilt") > get(&dark, "spectral_tilt"),
        "a higher stack must put more energy above the tilt split"
    );
}

// ---- the activation test -----------------------------------------------------------

#[test]
fn minimum_group_size_follows_the_permutation_count() {
    // C(2n,n)/2 labelings => smallest p is its reciprocal: 10 at n=3, 35 at n=4, 126 at n=5.
    assert_eq!(min_n_for_alpha(0.10), 3, "1/10 == 0.10 is exactly attainable at n=3");
    assert_eq!(min_n_for_alpha(0.05), 4, "1/10 > 0.05, so n=3 cannot reject");
    assert_eq!(min_n_for_alpha(0.02), 5, "1/35 > 0.02, so n=4 cannot reject");
}

fn group(f0: f64, amp: f64, seeds: std::ops::Range<u64>) -> Vec<Features> {
    seeds
        .map(|s| {
            let mut rng = Lcg::new(s);
            // Per-render jitter: this is the stand-in for the model's sampling noise.
            let f = f0 * (1.0 + 0.03 * rng.next_sym());
            let a = amp * (1.0 + 0.05 * rng.next_sym());
            feat(&voiced(f, a, DUR, s))
        })
        .collect()
}

#[test]
fn a_group_too_small_to_reject_is_refused_not_reported_false() {
    let a = group(150.0, 0.3, 0..3);
    let b = group(260.0, 0.3, 100..103);
    assert!(
        activation_test(&a, &b, 0.05).is_none(),
        "n=3 cannot reach p<=0.05 and must refuse rather than answer 'not activated'"
    );
    // One more render per side and the same comparison becomes answerable.
    let a4 = group(150.0, 0.3, 0..4);
    let b4 = group(260.0, 0.3, 100..104);
    assert!(activation_test(&a4, &b4, 0.05).is_some(), "n=4 is answerable at alpha=0.05");
}

#[test]
fn malformed_groups_are_refused() {
    let a = group(150.0, 0.3, 0..4);
    let b = group(150.0, 0.3, 100..105);
    assert!(activation_test(&a, &b, 0.05).is_none(), "unequal group sizes");
    assert!(activation_test(&[], &[], 0.05).is_none(), "empty groups");
}

#[test]
fn permutation_count_is_the_exact_c2nn_over_two() {
    let a = group(150.0, 0.3, 0..4);
    let b = group(260.0, 0.3, 100..104);
    let out = activation_test(&a, &b, 0.05).expect("answerable");
    assert_eq!(out.n_permutations, 35, "C(8,4)/2 == 35 distinct labelings");
    let a5 = group(150.0, 0.3, 0..5);
    let b5 = group(260.0, 0.3, 100..105);
    let out5 = activation_test(&a5, &b5, 0.05).expect("answerable");
    assert_eq!(out5.n_permutations, 126, "C(10,5)/2 == 126 distinct labelings");
}

#[test]
fn a_real_difference_activates_at_the_smallest_attainable_p() {
    let cued = group(260.0, 0.3, 0..4);
    let uncued = group(150.0, 0.3, 100..104);
    let out = activation_test(&cued, &uncued, 0.05).expect("answerable");
    assert!(out.activated, "a 110 Hz pitch difference must register as activation");
    assert!(
        (out.p_value - 1.0 / 35.0).abs() < 1e-9,
        "a separation this large should be the most extreme of all 35 labelings, got p={}",
        out.p_value
    );
    assert!(out.effect > 0.0);
}

#[test]
fn identical_conditions_do_not_activate() {
    // The single most important assertion in this file. Both groups come from the same
    // generator and differ only by seed, exactly like two sets of renders of the same
    // text. Calling this activated would make the whole metric a rubber stamp.
    let a = group(150.0, 0.3, 0..4);
    let b = group(150.0, 0.3, 100..104);
    let out = activation_test(&a, &b, 0.05).expect("answerable");
    assert!(
        !out.activated,
        "same-condition groups must not activate (p={}, effect={})",
        out.p_value, out.effect
    );
}

#[test]
fn false_positive_rate_stays_near_alpha_over_many_replications() {
    // Calibration, not a single lucky seed: 20 independent same-condition comparisons.
    // An exact permutation test at alpha=0.05 rejects at most 5 % of the time under the
    // null, so more than 3 hits out of 20 means the statistic is not calibrated.
    let mut hits = 0;
    for rep in 0..20u64 {
        let a = group(150.0, 0.3, rep * 100..rep * 100 + 4);
        let b = group(150.0, 0.3, rep * 100 + 50..rep * 100 + 54);
        if activation_test(&a, &b, 0.05).expect("answerable").activated {
            hits += 1;
        }
    }
    assert!(hits <= 3, "{hits}/20 same-condition comparisons activated; the test is not calibrated");
}

#[test]
fn the_test_is_deterministic() {
    let a = group(150.0, 0.3, 0..4);
    let b = group(260.0, 0.3, 100..104);
    let first = activation_test(&a, &b, 0.05).expect("answerable");
    let second = activation_test(&a, &b, 0.05).expect("answerable");
    assert_eq!(first, second);
}

#[test]
fn alpha_is_honoured_in_both_directions() {
    let a = group(260.0, 0.3, 0..4);
    let b = group(150.0, 0.3, 100..104);
    // p is 1/35 ~= 0.0286: activated at alpha=0.05, not at alpha=0.01.
    assert!(activation_test(&a, &b, 0.05).expect("answerable").activated);
    assert!(
        activation_test(&a, &b, 0.01).is_none(),
        "alpha=0.01 needs n>=5; n=4 must be refused rather than silently answered"
    );
}
