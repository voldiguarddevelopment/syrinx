//! Pure (model-free) tests for the render-level editable prosody plan
//! (`syrinx_prosody::render_plan`): JSON round-trip, validation, the semitone→ratio
//! math, and the mel time-warp + per-output-frame F0 multiplier produced by
//! `RenderPlan::apply`. These are deterministic and fast — the audio-affecting
//! end-to-end behaviour is the `real`-gated `prosody_plan_render_smoke`.
//!
//! Lives at the repo root per the frozen scaffold rule (member crates host no
//! tests); `syrinx-prosody` is a root dev-dependency.

use syrinx_prosody::render_plan::{
    semitones_to_ratio, shift_mel_bins, Region, RenderPlan, RenderPlanError,
    RENDER_PLAN_SCHEMA_VERSION,
};

/// A flat `[n_mels][T]` mel where each band is a distinct constant — easy to
/// reason about under interpolation (band `m` holds value `m` everywhere).
fn const_band_mel(n_mels: usize, t: usize) -> Vec<Vec<f32>> {
    (0..n_mels)
        .map(|m| vec![m as f32; t])
        .collect()
}

/// A mel whose single band ramps `0..T` linearly in time — exercises the time-warp.
fn ramp_mel(t: usize) -> Vec<Vec<f32>> {
    vec![(0..t).map(|i| i as f32).collect()]
}

#[test]
fn semitone_ratio_is_octave_exact() {
    assert!((semitones_to_ratio(0.0) - 1.0).abs() < 1e-12);
    assert!((semitones_to_ratio(12.0) - 2.0).abs() < 1e-9);
    assert!((semitones_to_ratio(-12.0) - 0.5).abs() < 1e-9);
    // +4 semitones ≈ 1.2599, -4 ≈ 0.7937.
    assert!((semitones_to_ratio(4.0) - 1.259_921).abs() < 1e-4);
    assert!((semitones_to_ratio(-4.0) - 0.793_700).abs() < 1e-4);
}

#[test]
fn json_round_trips_and_requires_schema_version() {
    let plan = RenderPlan::identity()
        .with_global_rate(0.8)
        .with_global_pitch_semitones(4.0)
        .with_mel_envelope_shift(true)
        .add_region(Region {
            start_frame: 2,
            end_frame: 5,
            rate: Some(1.5),
            pitch_semitones: Some(-3.0),
        });
    let js = serde_json::to_string(&plan).expect("serialize");
    let back: RenderPlan = serde_json::from_str(&js).expect("deserialize");
    assert_eq!(plan, back);
    assert_eq!(back.schema_version, RENDER_PLAN_SCHEMA_VERSION);

    // Omitting the required schema_version must fail (no silent default).
    let no_ver = r#"{"global_rate":1.0,"global_pitch_semitones":0.0}"#;
    assert!(serde_json::from_str::<RenderPlan>(no_ver).is_err());
}

#[test]
fn validate_rejects_bad_knobs_and_regions() {
    let t = 10;
    assert_eq!(RenderPlan::identity().validate(t), Ok(()));
    assert_eq!(
        RenderPlan::identity().with_global_rate(0.0).validate(t),
        Err(RenderPlanError::InvalidRate)
    );
    assert_eq!(
        RenderPlan::identity()
            .with_global_pitch_semitones(f64::NAN)
            .validate(t),
        Err(RenderPlanError::InvalidPitch)
    );
    // Region past the mel end.
    assert_eq!(
        RenderPlan::identity()
            .add_region(Region {
                start_frame: 8,
                end_frame: 11,
                rate: None,
                pitch_semitones: None,
            })
            .validate(t),
        Err(RenderPlanError::InvalidRegion)
    );
    // Empty region.
    assert_eq!(
        RenderPlan::identity()
            .add_region(Region {
                start_frame: 4,
                end_frame: 4,
                rate: None,
                pitch_semitones: None,
            })
            .validate(t),
        Err(RenderPlanError::InvalidRegion)
    );
    // Region with a bad rate override.
    assert_eq!(
        RenderPlan::identity()
            .add_region(Region {
                start_frame: 0,
                end_frame: 2,
                rate: Some(-1.0),
                pitch_semitones: None,
            })
            .validate(t),
        Err(RenderPlanError::InvalidRate)
    );
}

#[test]
fn identity_is_a_passthrough() {
    let mel = ramp_mel(20);
    let (out, f0) = RenderPlan::identity().apply(&mel).expect("apply");
    assert_eq!(out[0].len(), 20, "identity preserves frame count");
    assert_eq!(out, mel, "identity is value-for-value passthrough");
    assert!(f0.iter().all(|&m| (m - 1.0).abs() < 1e-12), "identity F0 mult is 1");
}

#[test]
fn global_rate_scales_frame_count() {
    let mel = ramp_mel(40);
    // rate 0.5 -> twice as many frames; rate 2.0 -> half.
    let (slow, _) = RenderPlan::identity()
        .with_global_rate(0.5)
        .apply(&mel)
        .unwrap();
    let (fast, _) = RenderPlan::identity()
        .with_global_rate(2.0)
        .apply(&mel)
        .unwrap();
    assert_eq!(slow[0].len(), 80, "rate 0.5 doubles frames");
    assert_eq!(fast[0].len(), 20, "rate 2.0 halves frames");
    // Monotone ramp is preserved (still non-decreasing across the warp).
    assert!(slow[0].windows(2).all(|w| w[1] >= w[0] - 1e-4));
}

#[test]
fn per_region_rate_adds_frames_only_in_the_region() {
    let t = 30;
    let mel = ramp_mel(t);
    // Slow down frames [10,20) by 0.5 (each contributes 2 output frames); the rest
    // stay at rate 1.0. Expected output ≈ 20 (outside) + 20 (inside doubled) = 40.
    let plan = RenderPlan::identity().add_region(Region {
        start_frame: 10,
        end_frame: 20,
        rate: Some(0.5),
        pitch_semitones: None,
    });
    let (out, _) = plan.apply(&mel).unwrap();
    assert!(
        (out[0].len() as i64 - 40).abs() <= 1,
        "expected ~40 frames, got {}",
        out[0].len()
    );
}

#[test]
fn global_pitch_sets_a_uniform_f0_multiplier() {
    let mel = ramp_mel(16);
    let (_, f0) = RenderPlan::identity()
        .with_global_pitch_semitones(12.0)
        .apply(&mel)
        .unwrap();
    assert!(
        f0.iter().all(|&m| (m - 2.0).abs() < 1e-6),
        "+12 semitones must double F0 everywhere"
    );
}

#[test]
fn per_region_pitch_overrides_only_its_frames() {
    let t = 30;
    let mel = ramp_mel(t);
    // Global +0, but frames [10,20) get +12 (ratio 2.0). No rate change, so output
    // frame count == input and the region maps frame-for-frame.
    let plan = RenderPlan::identity().add_region(Region {
        start_frame: 10,
        end_frame: 20,
        rate: None,
        pitch_semitones: Some(12.0),
    });
    let (out, f0) = plan.apply(&mel).unwrap();
    assert_eq!(out[0].len(), t, "no rate change keeps frame count");
    // Endpoints outside the region are ~1.0, the region interior is ~2.0.
    assert!((f0[0] - 1.0).abs() < 1e-6, "pre-region F0 mult ~1.0");
    assert!((f0[15] - 2.0).abs() < 1e-6, "in-region F0 mult ~2.0, got {}", f0[15]);
    assert!((f0[29] - 1.0).abs() < 1e-6, "post-region F0 mult ~1.0");
}

#[test]
fn mel_bin_shift_moves_energy_up_and_is_identity_at_ratio_one() {
    // const_band_mel: band m holds value m. Shifting up by ratio 2 maps output bin
    // b <- input bin b/2, so output bin b holds value ~b/2 (energy pulled from lower
    // bins => the envelope moves up the axis).
    let mel = const_band_mel(8, 4);
    let up = shift_mel_bins(&mel, 2.0);
    assert_eq!(up.len(), 8);
    assert!((up[4][0] - 2.0).abs() < 1e-5, "bin 4 <- input bin 2 (=2.0), got {}", up[4][0]);
    // ratio 1.0 is the identity.
    assert_eq!(shift_mel_bins(&mel, 1.0), mel);
}

#[test]
fn apply_rejects_empty_and_ragged_mel() {
    assert_eq!(RenderPlan::identity().apply(&[]), Err(RenderPlanError::BadMelShape));
    let ragged = vec![vec![0.0f32; 4], vec![0.0f32; 3]];
    assert_eq!(
        RenderPlan::identity().apply(&ragged),
        Err(RenderPlanError::BadMelShape)
    );
}
