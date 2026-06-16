//! T-07.04 — deterministic 48kHz→8kHz telephony downsampler.
//!
//! [`downsample_48k_to_8k`] band-limits a 48kHz `f32` buffer to the 8kHz
//! narrowband passband (4kHz Nyquist) via `syrinx-vocoder`'s anti-alias boxcar,
//! then decimates by the 6:1 rate ratio (48000 / 8000). The output length is
//! `L * 8000 / 48000` within ±1 sample. No codec, no full-band path, no model —
//! just the deterministic DSP on a synthetic buffer.

use syrinx_vocoder::band_limit;

/// Integer decimation factor: 48000 / 8000.
const FACTOR: usize = 6;

/// Downsample a 48kHz `f32` buffer to 8kHz.
///
/// Anti-alias band-limits with a length-`FACTOR` boxcar (first null at the 8kHz
/// post-decimation rate, so it suppresses energy near/above the 4kHz narrowband
/// Nyquist while passing DC and the low passband at unity gain), then keeps every
/// `FACTOR`-th sample. The output length is `input.len() * 8000 / 48000` within ±1
/// sample. Never panics on any buffer.
pub fn downsample_48k_to_8k(input: &[f32]) -> Vec<f32> {
    let filtered = band_limit(input, FACTOR);
    let mut out = Vec::new();
    let mut i = 0;
    while i < filtered.len() {
        out.push(filtered[i]);
        i += FACTOR;
    }
    out
}
