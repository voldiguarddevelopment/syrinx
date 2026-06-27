//! syrinx-acoustic — flow-matching decoder + chunk-aware streaming (scaffold; T-00.01).

// The real CosyVoice2 flow-matching mel decoder on Candle, gated behind the
// `real` feature + on-disk fp32 weights (too large to vendor). Default builds
// stay Candle-free.
#[cfg(feature = "real")]
pub mod cv2;

// The real CosyVoice3 flow-matching mel decoder (`CausalMaskedDiffWithDiT`) on
// Candle: a 22-layer DiT transformer CFM estimator (replacing CV2's U-Net) plus
// the CV3 token->mu front-end. Additive next to `real` (CV2); shares the same
// `real` feature gate. Default builds stay Candle-free.
#[cfg(feature = "real")]
pub mod cv3;
