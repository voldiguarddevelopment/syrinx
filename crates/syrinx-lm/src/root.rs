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

pub use attn::*;
pub use block::{block, swiglu_ffn};
