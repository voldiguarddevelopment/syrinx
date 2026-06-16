//! Frozen RED tests for T-07.04 — resample audio to 8 kHz telephony.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must build in `syrinx-stream` (which, per the criteria, performs the
//! decimation while delegating the anti-alias band-limit to `syrinx-vocoder`):
//!
//!   * `downsample_48k_to_8k(input: &[f32]) -> Vec<f32>` — a deterministic
//!     48kHz→8kHz downsampler over an `f32` sample buffer. It band-limits the
//!     input to the 8kHz narrowband passband (4kHz Nyquist) and decimates by the
//!     6:1 rate ratio. The output length equals `L * 8000 / 48000` within ±1
//!     sample. Total — never panics on any buffer.
//!
//! Contract (list.md / DESIGN §T7.04): a deterministic DSP transform over a
//! synthetic `f32` buffer — no codec, no full-band path, no model. The output is
//! length-correct for the rate ratio across multiple `L` (so the 6:1 ratio is
//! pinned, not a single constant), a DC (flat) input stays flat through resample
//! plus band-limit, and the anti-alias band-limit attenuates energy above the
//! 4kHz narrowband Nyquist while leaving an in-band tone's energy intact. The
//! perceptual narrowband-intelligibility eval is deferred to a later eval task.
//!
//! RED: `syrinx-stream` exposes no `resample` module yet, so the symbol does not
//! resolve and the test target fails to build — every criterion is unmet. GREEN
//! adds the module (wiring in `syrinx-vocoder`'s band-limit) so each assertion
//! below holds.

use std::f64::consts::PI;

use syrinx_stream::resample::downsample_48k_to_8k;

/// Source sample rate of the inputs under test.
const SRC_RATE: f64 = 48_000.0;

/// Declared anti-alias bound, as a fraction of reference energy. Out-of-band
/// energy must fall BELOW this fraction of the in-band reference; an in-band tone
/// must retain energy ABOVE this fraction. The same bound pins both sides of the
/// anti-alias filter. A real low-pass beats this lenient bound comfortably, so a
/// correct filter passes while a no-filter pure decimator (which aliases the
/// above-Nyquist tone back in-band) fails.
const ANTI_ALIAS_BOUND: f32 = 0.25;

/// A pure tone of unit amplitude: `sin(2π f n / rate)`.
fn tone(rate: f64, freq: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * PI * freq * (i as f64) / rate).sin() as f32)
        .collect()
}

/// Mean square (energy per sample) of a buffer; `0.0` for an empty buffer.
fn mean_square(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32
}

/// The ideal real-valued output length for an input of `l` 48kHz samples.
fn expected_len(l: usize) -> f64 {
    (l as f64) * 8000.0 / SRC_RATE
}

// ----------------------------------------------------------------------------
// C1 — output length equals `L * 8000 / 48000` (±1 sample), pinned across at
//      least two distinct L so the 6:1 ratio is fixed rather than a constant.
// ----------------------------------------------------------------------------

/// Two distinct input lengths each yield an output length within ±1 of the rate
/// ratio, AND the two output lengths differ — so a constant-length stub cannot
/// satisfy both. Both `L` here are exact multiples of 6 (4800→800, 9000→1500).
#[test]
fn test_length_ratio_pins_across_two_L() {
    let lengths = [4800usize, 9000usize];
    let mut out_lens = Vec::new();

    for &l in &lengths {
        let out = downsample_48k_to_8k(&vec![0.0f32; l]);
        let expected = expected_len(l);
        let diff = (out.len() as f64 - expected).abs();
        assert!(
            diff <= 1.0,
            "L={l}: output length {} not within ±1 of expected {expected}",
            out.len()
        );
        out_lens.push(out.len());
    }

    // Distinct inputs → distinct output lengths: the ratio is pinned, not a
    // single hard-coded constant.
    assert_ne!(
        out_lens[0], out_lens[1],
        "distinct input lengths must produce distinct output lengths"
    );
}

/// A non-multiple-of-6 length still lands within ±1 of the real ratio value,
/// pinning the ±1 rounding rule on a length whose ideal value is fractional
/// (10000 → 1666.67).
#[test]
fn test_length_within_one_sample_nondivisible() {
    let l = 10_000usize;
    let out = downsample_48k_to_8k(&vec![0.0f32; l]);
    let expected = expected_len(l); // 1666.666…
    let diff = (out.len() as f64 - expected).abs();
    assert!(
        diff <= 1.0,
        "L={l}: output length {} not within ±1 of expected {expected}",
        out.len()
    );
}

// ----------------------------------------------------------------------------
// C2 — a DC (constant) input downsamples to an output whose every sample equals
//      that constant within tolerance: a flat signal stays flat through resample
//      plus band-limit (DC is in-band and the band-limit must preserve it,
//      including at the buffer edges).
// ----------------------------------------------------------------------------

/// A flat buffer of value `c` resamples to a non-empty buffer in which every
/// single sample equals `c` within tolerance.
#[test]
fn test_dc_input_stays_flat() {
    let c = 0.42f32;
    let l = 6000usize;
    let out = downsample_48k_to_8k(&vec![c; l]);

    assert!(
        !out.is_empty(),
        "a non-trivial DC input must yield a non-empty output"
    );
    for (i, &s) in out.iter().enumerate() {
        assert!(
            (s - c).abs() <= 1e-3,
            "DC output sample {i} = {s} drifted from the constant {c}"
        );
    }
}

// ----------------------------------------------------------------------------
// C3 — a tone above the 4kHz narrowband Nyquist is attenuated below the declared
//      anti-alias bound (its post-resample energy is a fraction of a
//      same-amplitude in-band tone's energy), while an in-band tone is NOT
//      attenuated past that bound. Both sides of the anti-alias filter are
//      pinned; nothing panics.
// ----------------------------------------------------------------------------

/// An above-Nyquist tone (7kHz, above the 4kHz narrowband Nyquist) has its
/// post-resample energy fall below `ANTI_ALIAS_BOUND` times both the in-band
/// tone's output energy AND its own input energy. A pure decimator with no
/// anti-alias filter would alias 7kHz back into the passband and fail this.
#[test]
fn test_above_nyquist_tone_attenuated() {
    let n = 4800usize;
    let in_band_in = tone(SRC_RATE, 1000.0, n); // below 4kHz Nyquist
    let out_band_in = tone(SRC_RATE, 7000.0, n); // above 4kHz Nyquist, same amplitude

    let in_band_out = downsample_48k_to_8k(&in_band_in);
    let out_band_out = downsample_48k_to_8k(&out_band_in);

    let e_in = mean_square(&in_band_out);
    let e_out = mean_square(&out_band_out);
    let e_out_input = mean_square(&out_band_in);

    // Out-of-band energy is a small fraction of the in-band tone's energy.
    assert!(
        e_out < ANTI_ALIAS_BOUND * e_in,
        "above-Nyquist output energy {e_out} not below {ANTI_ALIAS_BOUND} × in-band energy {e_in}"
    );
    // And it is attenuated relative to its own pre-resample energy.
    assert!(
        e_out < ANTI_ALIAS_BOUND * e_out_input,
        "above-Nyquist output energy {e_out} not below {ANTI_ALIAS_BOUND} × its input energy {e_out_input}"
    );
}

/// An in-band tone (1kHz, below the 4kHz narrowband Nyquist) is NOT attenuated
/// past the bound: its post-resample energy stays ABOVE `ANTI_ALIAS_BOUND` times
/// its own input energy — the passband is preserved.
#[test]
fn test_in_band_tone_passes() {
    let n = 4800usize;
    let in_band_in = tone(SRC_RATE, 1000.0, n);
    let in_band_out = downsample_48k_to_8k(&in_band_in);

    let e_in_input = mean_square(&in_band_in);
    let e_in_out = mean_square(&in_band_out);

    assert!(
        e_in_out > ANTI_ALIAS_BOUND * e_in_input,
        "in-band output energy {e_in_out} dropped below {ANTI_ALIAS_BOUND} × its input energy {e_in_input}"
    );
}
