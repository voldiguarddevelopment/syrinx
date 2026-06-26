//! Crate root for `syrinx-lm`.
//!
//! `lib.rs` holds the T-02.02a causal multi-head `attention` and is kept
//! byte-identical to its freeze base so the mutation gate never re-targets it;
//! this root only wires the modules together. `block.rs` (T-02.02b) is isolated
//! in its own file so the gate scopes its mutants to the frozen
//! `block_prop.rs` tests.
#[path = "lib.rs"]
mod attn;
mod block;
mod forward;
mod stage;

pub use attn::*;
pub use block::{block, swiglu_ffn};
pub use forward::forward;
pub use stage::{embed_tokens, layer_attention, transformer_block};

// The real CosyVoice2 LM forward via Candle (DESIGN T2.1). Gated behind the
// `real` feature so the default/CI build stays Candle-free and the toy parity
// tests are unaffected; built + verified against real weights on the GPU box.
#[cfg(feature = "real")]
pub mod real;

// The real CosyVoice3 LM forward via Candle — the first CV3 component port. It
// REUSES `real`'s Qwen2-0.5B body (architecturally identical: 24 layers, 14/2
// GQA heads, head_dim 64, RoPE θ=1e6, RMSNorm eps 1e-6, sliding-window disabled)
// and adds only the CV3-specific embedding assembly + bias-free `llm_decoder`
// head. Same `real` feature + on-box weights/fixtures; CV2 code paths unchanged.
#[cfg(feature = "real")]
pub mod real_cv3;
