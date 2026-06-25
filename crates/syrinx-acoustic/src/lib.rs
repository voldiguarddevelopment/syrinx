//! syrinx-acoustic — flow-matching decoder + chunk-aware streaming (scaffold; T-00.01).

// The real CosyVoice2 flow-matching mel decoder on Candle, gated behind the
// `real` feature + on-disk fp32 weights (too large to vendor). Default builds
// stay Candle-free.
#[cfg(feature = "real")]
pub mod real;
