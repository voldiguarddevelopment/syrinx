//! syrinx-speaker — speaker encoder, embedding store, blend/morph (scaffold; T-00.01).

// The real CosyVoice2 CAM++ speaker encoder via Candle (80-d fbank -> 192-d x-vector).
// Gated behind the `real` feature; built + parity-verified on the model box.
#[cfg(feature = "real")]
pub mod real;
