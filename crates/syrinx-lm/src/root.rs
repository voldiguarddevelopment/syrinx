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
