//! syrinx-vocoder — the real CosyVoice2/3 HiFT vocoder (mel -> 24 kHz waveform).

// The real CosyVoice2 HiFT vocoder forward via Candle (mel -> 24 kHz waveform).
// The `real` feature is on by default; built + parity-verified on the model box.
#[cfg(feature = "real")]
pub mod real;

// The real CosyVoice3 CausalHiFTGenerator forward via Candle (causal convs + f64
// f0_predictor). Additive to the CV2 `real` module; same `real` feature gate.
#[cfg(feature = "real")]
pub mod real_cv3;
