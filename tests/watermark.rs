//! Spread-spectrum output-watermark tests — pure-Rust, **no model required**.
//!
//! These exercise `syrinx_serve::watermark` (the model-free embed/detect) on
//! synthetic 24 kHz audio, so they run in the default (Candle-free, no `real`)
//! build at the repo root, per the frozen rule that member crates host no tests.
//!
//! They pin the watermark's characterized behaviour:
//!   * embed -> detect round-trips with high confidence and the exact payload;
//!   * clean (un-watermarked) audio detects as absent;
//!   * the wrong key detects as absent;
//!   * the mark survives added Gaussian noise at a modest SNR and a gain change;
//!   * the per-sample perturbation stays at/below the stated amplitude
//!     (imperceptibility budget).

use syrinx_serve::watermark::{
    detect_watermark, embed_watermark, DEFAULT_AMP, PRESENT_Z_THRESHOLD,
};

const SR: usize = 24_000;

/// A deterministic, mildly-realistic "speech-like" host signal: a couple of
/// summed sinusoids with a slow amplitude envelope, at a modest level
/// (RMS ≈ 0.05). Deterministic so the tests are reproducible.
fn host_signal(n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
        let tone = (2.0 * std::f32::consts::PI * 180.0 * t).sin()
            + 0.5 * (2.0 * std::f32::consts::PI * 320.0 * t).sin();
        v.push(0.045 * env * tone);
    }
    v
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

/// A tiny deterministic Gaussian-ish noise (Box-Muller on a SplitMix64), so the
/// noise-robustness test does not depend on `rand`.
fn gaussian_noise(n: usize, std: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64 // (0,1]
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let g1 = r * (2.0 * std::f64::consts::PI * u2).cos();
        let g2 = r * (2.0 * std::f64::consts::PI * u2).sin();
        out.push((g1 as f32) * std);
        if out.len() < n {
            out.push((g2 as f32) * std);
        }
    }
    out
}

const KEY: u64 = 0x5179_1A2B_3C4D_5E6F;
const PAYLOAD: u16 = 0xB17E;

/// Round-trip: embed then detect on the unmodified output recovers the payload
/// with high confidence (well above the present-threshold).
#[test]
fn roundtrip_recovers_payload_with_high_confidence() {
    let mut audio = host_signal(2 * SR); // 2 s
    embed_watermark(&mut audio, KEY, PAYLOAD);

    let r = detect_watermark(&audio, KEY).expect("detect returns a result");
    assert!(r.present, "watermark must be detected (confidence {})", r.confidence);
    assert_eq!(r.payload, PAYLOAD, "payload must round-trip exactly");
    assert!(
        r.confidence > 2.0 * PRESENT_Z_THRESHOLD,
        "clean round-trip confidence should be comfortably high, got {}",
        r.confidence
    );
}

/// Clean (un-watermarked) audio detects as absent with low confidence.
#[test]
fn clean_audio_is_absent() {
    let audio = host_signal(2 * SR);
    let r = detect_watermark(&audio, KEY).expect("detect returns a result");
    assert!(
        !r.present,
        "clean audio must not detect a watermark (confidence {})",
        r.confidence
    );
    assert!(
        r.confidence < PRESENT_Z_THRESHOLD,
        "clean confidence must be below threshold, got {}",
        r.confidence
    );
}

/// A wrong key detects as absent: the chip sequence is uncorrelated, so the
/// correlation collapses to the noise floor.
#[test]
fn wrong_key_is_absent() {
    let mut audio = host_signal(2 * SR);
    embed_watermark(&mut audio, KEY, PAYLOAD);

    let wrong = KEY ^ 0xDEAD_BEEF_0000_0001;
    let r = detect_watermark(&audio, wrong).expect("detect returns a result");
    assert!(
        !r.present,
        "wrong key must not detect the watermark (confidence {})",
        r.confidence
    );
    assert!(r.confidence < PRESENT_Z_THRESHOLD, "wrong-key confidence {}", r.confidence);
}

/// Survives light additive Gaussian noise at a modest SNR.
#[test]
fn survives_added_noise() {
    let mut audio = host_signal(2 * SR);
    let host_rms = rms(&audio);
    embed_watermark(&mut audio, KEY, PAYLOAD);

    // Add noise at ~ host level / 2.5  (≈ 8 dB signal-to-noise) — "light noise".
    let noise = gaussian_noise(audio.len(), (host_rms as f32) / 2.5, 0xC0FFEE);
    for (a, nz) in audio.iter_mut().zip(noise.iter()) {
        *a += *nz;
    }

    let r = detect_watermark(&audio, KEY).expect("detect returns a result");
    assert!(
        r.present,
        "watermark must survive light noise (confidence {})",
        r.confidence
    );
    assert_eq!(r.payload, PAYLOAD, "payload must still round-trip under noise");
}

/// Survives a linear gain change: the z-score is scale-invariant.
#[test]
fn survives_gain_change() {
    let mut audio = host_signal(2 * SR);
    embed_watermark(&mut audio, KEY, PAYLOAD);

    for g in [0.5f32, 1.7] {
        let scaled: Vec<f32> = audio.iter().map(|&s| s * g).collect();
        let r = detect_watermark(&scaled, KEY).expect("detect returns a result");
        assert!(
            r.present,
            "watermark must survive gain {g} (confidence {})",
            r.confidence
        );
        assert_eq!(r.payload, PAYLOAD, "payload must round-trip under gain {g}");
    }
}

/// Survives a small integer-sample crop (the detector's chip-phase sync search).
#[test]
fn survives_small_crop() {
    let mut audio = host_signal(2 * SR);
    embed_watermark(&mut audio, KEY, PAYLOAD);

    // Drop 137 samples from the front (a phase desync the sync search must undo).
    let cropped = &audio[137..];
    let r = detect_watermark(cropped, KEY).expect("detect returns a result");
    assert!(
        r.present,
        "watermark must survive a small crop (confidence {})",
        r.confidence
    );
    assert_eq!(r.payload, PAYLOAD, "payload must round-trip after crop");
}

/// Imperceptibility: the per-sample perturbation stays at/below the stated embed
/// amplitude, i.e. `max |Δ| <= DEFAULT_AMP` and `rms(Δ) <= DEFAULT_AMP`.
#[test]
fn perturbation_stays_below_threshold() {
    let clean = host_signal(2 * SR);
    let mut marked = clean.clone();
    embed_watermark(&mut marked, KEY, PAYLOAD);

    let mut max_abs = 0.0f32;
    let mut sumsq = 0.0f64;
    for (c, m) in clean.iter().zip(marked.iter()) {
        let d = (m - c).abs();
        max_abs = max_abs.max(d);
        sumsq += (d as f64) * (d as f64);
    }
    let rms_delta = (sumsq / clean.len() as f64).sqrt();

    // Allow a hair of float slack over the exact amplitude.
    let budget = DEFAULT_AMP as f64 * 1.0001;
    assert!(
        (max_abs as f64) <= budget,
        "max perturbation {max_abs} exceeds amplitude budget {budget}"
    );
    assert!(
        rms_delta <= budget,
        "rms perturbation {rms_delta} exceeds amplitude budget {budget}"
    );
    // Sanity: the watermark sits well below the host. This synthetic host is
    // deliberately quiet (RMS ≈ 0.022), giving ~14-15 dB headroom here; for a
    // typical-level speech output (RMS ~0.1-0.3) the same 4e-3 amplitude is
    // ~28-37 dB below the signal.
    let host = rms(&clean);
    let headroom_db = 20.0 * (host / rms_delta).log10();
    assert!(
        headroom_db > 12.0,
        "watermark should sit comfortably below the host signal, got {headroom_db} dB"
    );
}

