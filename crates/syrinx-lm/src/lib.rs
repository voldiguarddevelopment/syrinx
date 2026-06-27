//! syrinx-lm — the real CosyVoice2 / CosyVoice3 LM forward (Candle).
//!
//! `real` is the real CosyVoice2 LM forward (DESIGN T2.1): the Qwen2-0.5B body
//! (24 layers, 14/2 GQA heads, head_dim 64, RoPE θ=1e6, RMSNorm eps 1e-6) plus
//! KV-cache decode and fp32/int4 weight loading on Candle.
//!
//! `real_cv3` is the real CosyVoice3 LM forward — the first CV3 component port. It
//! REUSES `real`'s Qwen2-0.5B body (architecturally identical) and adds only the
//! CV3-specific embedding assembly + bias-free `llm_decoder` head. CV2 code paths
//! are unchanged.

// The real CosyVoice2 LM forward via Candle. The `real` feature is on by default
// (see `Cargo.toml`'s `default = ["real"]`); it pulls Candle in as a normal
// dependency. Building `--no-default-features` yields an empty, Candle-free crate.
#[cfg(feature = "real")]
pub mod real;

// The real CosyVoice3 LM forward via Candle, layered on `real`'s Qwen2 body.
#[cfg(feature = "real")]
pub mod real_cv3;
