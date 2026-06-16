//! syrinx-lm — the full LM forward logits (T-02.02c-b).
//!
//! `forward(token_ids) -> logits[T, vocab]`: the exact `reference.py` §5
//! (= PARITY.md §5) end-to-end forward, composed over the T-02.02c-a-verified
//! stage (`embed_tokens` + `transformer_block`):
//!
//!   h = embed(token_ids, tok_embeddings)        # [T, dim]
//!   for L in 0..n_layers: h = block(h, L, positions)
//!   h = rmsnorm(h, norm.weight, eps)            # final pre-head norm
//!   logits = linear(h, output.weight)           # [T, vocab], untied, no bias
//!
//! with `positions = [0, 1, ..., T-1]`, every weight drawn by its literal §3
//! name. Lives in its own source file so the task-scoped mutation gate targets
//! only this composition and leaves the byte-identical T-02.02c-a `stage.rs`
//! (and the earlier `lib.rs`/`block.rs`) untouched.

use crate::{embed_tokens, transformer_block};
use syrinx_core::{linear, rmsnorm, weights, Tensor};

/// LM config (PARITY.md §3 / REFERENCE.md): vocab=512, dim=128, n_layers=4.
const VOCAB: usize = 512;
const DIM: usize = 128;
const N_LAYERS: usize = 4;
/// RMSNorm epsilon (PARITY.md "Global": `eps = 1e-5`).
const EPS: f32 = 1e-5;

/// The `[rows, cols]` weight matrix named `name` from the T-02.01c name-seeded
/// stream (`[out,in]` orientation, as `linear` expects).
fn mat(name: &str, rows: usize, cols: usize) -> Tensor {
    Tensor::new(weights(name, rows * cols), vec![rows, cols])
}

/// The `[d]` weight vector named `name` (an RMSNorm weight).
fn vecw(name: &str, d: usize) -> Tensor {
    Tensor::new(weights(name, d), vec![d])
}

/// `forward(token_ids[T]) -> logits[T, vocab]` (`reference.py` §5).
///
/// Embeds the tokens, runs the `n_layers` pre-RMSNorm residual blocks over
/// `positions = [0..T-1]`, applies the final `rmsnorm(norm.weight)`, then the
/// untied `output.weight` lm_head — a single deterministic forward producing
/// logits only (no sampling/decoding/KV-cache).
pub fn forward(token_ids: &[usize]) -> Tensor {
    let positions: Vec<usize> = (0..token_ids.len()).collect();

    // embed -> n_layers blocks, over the verified T-02.02c-a stage.
    let mut h = embed_tokens(token_ids);
    for layer in 0..N_LAYERS {
        h = transformer_block(&h, layer, &positions);
    }

    // Final pre-head RMSNorm, THEN the untied lm_head (norm before head).
    let normed = rmsnorm(&h, &vecw("norm.weight", DIM), EPS);
    linear(&normed, &mat("output.weight", VOCAB, DIM), None)
}
