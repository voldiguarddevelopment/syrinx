//! Volume automation — apply a per-sample gain envelope to an f32 buffer.
//!
//! A pure deterministic DSP transform (T-03.05): output sample `i` is
//! `samples[i] * envelope[i]`. The envelope is a per-sample gain curve (any
//! interpolation across segment boundaries is baked into the curve by the
//! caller); this transform only multiplies. Amplitude/gain only — pitch and
//! duration are untouched.

/// Error returned by [`apply_gain_envelope`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    /// The envelope length does not equal the sample-buffer length.
    LengthMismatch,
}

/// Apply a per-sample gain `envelope` to `samples`, returning a new buffer where
/// output `i` is `samples[i] * envelope[i]`.
///
/// The envelope must have exactly the same length as the sample buffer;
/// otherwise [`EnvelopeError::LengthMismatch`] is returned (never a panic). The
/// output length equals the input length.
pub fn apply_gain_envelope(
    samples: &[f32],
    envelope: &[f32],
) -> Result<Vec<f32>, EnvelopeError> {
    if samples.len() != envelope.len() {
        return Err(EnvelopeError::LengthMismatch);
    }
    Ok(samples
        .iter()
        .zip(envelope.iter())
        .map(|(sample, gain)| sample * gain)
        .collect())
}
