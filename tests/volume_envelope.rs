//! Frozen RED tests for T-03.05 — apply volume automation curves.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build in `syrinx-prosody::volume`:
//!
//!   * `apply_gain_envelope(samples: &[f32], envelope: &[f32])
//!        -> Result<Vec<f32>, EnvelopeError>` — a deterministic per-sample gain
//!     transform. The envelope is a per-sample gain curve that must have exactly
//!     the same length as the sample buffer; output sample `i` is
//!     `samples[i] * envelope[i]`, the output length equals the input length, and
//!     nothing about pitch or duration is touched (amplitude/gain only).
//!   * `EnvelopeError::LengthMismatch` — returned (never a panic) when the
//!     envelope length differs from the buffer length in either direction.
//!
//! Contract (list.md / DESIGN T3.5): a pure deterministic DSP transform over a
//! caller-supplied buffer — no model inference. A flat-1.0 envelope is the
//! identity (bit-for-bit), a flat-0.5 envelope halves every sample exactly, a
//! linearly-ramped envelope applies its interpolated gain sample-exact (first
//! sample uses A, last uses B, midpoint uses (A+B)/2), and an envelope whose
//! length disagrees with the buffer is a typed `LengthMismatch` error while an
//! exactly-equal length applies.
//!
//! RED: `syrinx-prosody` exposes no `volume` module yet, so neither symbol
//! resolves and the test target fails to build — every criterion is unmet. GREEN
//! implements the module so each assertion below holds.

use syrinx_prosody::volume::{apply_gain_envelope, EnvelopeError};

/// Assert two `f32`s are bit-for-bit identical (stricter than `==`: it also pins
/// the sign of zero), so "bit-identical to the input" means exactly that.
fn assert_bits_eq(actual: f32, expected: f32) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "expected bit-identical {expected} (bits {:#010x}), got {actual} (bits {:#010x})",
        expected.to_bits(),
        actual.to_bits()
    );
}

/// Assert `actual` is within `tol` of `expected` (for the interpolation midpoint,
/// which the spec pins "within tolerance").
fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected} within {tol}, got {actual} (delta {})",
        (actual - expected).abs()
    );
}

// ----------------------------------------------------------------------------
// C1 — a flat-1.0 envelope is the bit-identical identity.
// ----------------------------------------------------------------------------

/// Applying an all-1.0 envelope returns each sample bit-identical to the input
/// (including a signed zero and negative values), and preserves the buffer length.
#[test]
fn test_flat_unity_is_bit_identical() {
    let samples: Vec<f32> = vec![0.0, -0.0, 1.0, -0.4, 123.456, -987.5];
    let envelope: Vec<f32> = vec![1.0; samples.len()];

    let out = apply_gain_envelope(&samples, &envelope).expect("equal length applies");

    // Output length equals input length (the invariant).
    assert_eq!(out.len(), samples.len());
    // Every sample is bit-for-bit identical to the input.
    for (got, want) in out.iter().zip(samples.iter()) {
        assert_bits_eq(*got, *want);
    }
}

// ----------------------------------------------------------------------------
// C2 — a flat-0.5 envelope halves exactly; a non-0.5 gain scales exactly too.
// ----------------------------------------------------------------------------

/// An all-0.5 envelope yields each output equal to exactly 0.5× the input
/// (1.0 → 0.5, -0.4 → -0.2, 0.0 → 0.0), and a separate non-0.5 gain (0.25) scales
/// exactly (4.0 → 1.0, -8.0 → -2.0) — together killing scale/operator mutants.
#[test]
fn test_half_and_nonhalf_gain_scale_exactly() {
    // Flat 0.5 — exact halving at the spec's example values.
    let samples: Vec<f32> = vec![1.0, -0.4, 0.0];
    let half: Vec<f32> = vec![0.5; samples.len()];
    let out = apply_gain_envelope(&samples, &half).expect("equal length applies");
    assert_eq!(out.len(), samples.len());
    assert_bits_eq(out[0], 0.5); // 1.0  -> 0.5
    assert_bits_eq(out[1], -0.2); // -0.4 -> -0.2 (exact: scaling by 0.5)
    assert_bits_eq(out[2], 0.0); // 0.0  -> 0.0

    // Non-0.5 gain (0.25) — a different scale, also exact. If the implementation
    // hardcoded or mutated the factor, these would diverge.
    let samples2: Vec<f32> = vec![4.0, -8.0];
    let quarter: Vec<f32> = vec![0.25; samples2.len()];
    let out2 = apply_gain_envelope(&samples2, &quarter).expect("equal length applies");
    assert_bits_eq(out2[0], 1.0); // 4.0  -> 1.0
    assert_bits_eq(out2[1], -2.0); // -8.0 -> -2.0
}

// ----------------------------------------------------------------------------
// C3 — a ramp across a segment boundary interpolates A -> B sample-exact.
// ----------------------------------------------------------------------------

/// A linear gain ramp from A to B over the buffer applies per spec: the first
/// sample uses A, the last uses B, and the midpoint uses (A+B)/2 (within
/// tolerance). A unit buffer isolates the applied gain so the output IS the
/// per-sample gain at each index.
#[test]
fn test_segment_boundary_interpolates() {
    const A: f32 = 1.0;
    const B: f32 = 2.0;
    const N: usize = 5; // odd length => an exact middle index at N/2 == 2.

    // Build the interpolated envelope: env[i] = A + (B-A) * i/(N-1).
    let mut envelope = Vec::with_capacity(N);
    for i in 0..N {
        envelope.push(A + (B - A) * (i as f32) / ((N - 1) as f32));
    }
    let samples: Vec<f32> = vec![1.0; N];

    let out = apply_gain_envelope(&samples, &envelope).expect("equal length applies");
    assert_eq!(out.len(), N);

    // First sample uses gain A, last sample uses gain B.
    assert_close(out[0], A, 1e-6);
    assert_close(out[N - 1], B, 1e-6);
    // The midpoint sample uses (A+B)/2.
    assert_close(out[N / 2], (A + B) / 2.0, 1e-6);
    // The intermediate samples track the ramp (pins the interpolation shape on
    // both sides of the midpoint, not just the three named points).
    assert_close(out[1], 1.25, 1e-6);
    assert_close(out[3], 1.75, 1e-6);
}

// ----------------------------------------------------------------------------
// C4 — length boundary: differing length errors (both ways), equal length applies.
// ----------------------------------------------------------------------------

/// An envelope longer than the buffer and one shorter than the buffer both return
/// `Err(EnvelopeError::LengthMismatch)`; an exactly-equal length applies (Ok).
/// Both sides of the length check are pinned.
#[test]
fn test_length_mismatch_both_directions_and_equal_applies() {
    let samples: Vec<f32> = vec![1.0, 2.0, 3.0]; // length 3

    // Envelope longer than the buffer (length 4) -> LengthMismatch.
    let longer: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let e_long = apply_gain_envelope(&samples, &longer)
        .expect_err("a longer envelope must be rejected");
    assert!(matches!(e_long, EnvelopeError::LengthMismatch));

    // Envelope shorter than the buffer (length 2) -> LengthMismatch.
    let shorter: Vec<f32> = vec![1.0, 1.0];
    let e_short = apply_gain_envelope(&samples, &shorter)
        .expect_err("a shorter envelope must be rejected");
    assert!(matches!(e_short, EnvelopeError::LengthMismatch));

    // Exactly-equal length (length 3) -> Ok, and applies the gains.
    let equal: Vec<f32> = vec![1.0, 0.5, 2.0];
    let out = apply_gain_envelope(&samples, &equal)
        .expect("an exactly-equal-length envelope must apply");
    assert_eq!(out.len(), samples.len());
    assert_bits_eq(out[0], 1.0); // 1.0 * 1.0
    assert_bits_eq(out[1], 1.0); // 2.0 * 0.5
    assert_bits_eq(out[2], 6.0); // 3.0 * 2.0
}
