//! syrinx-lm — the layer-0 forward stage (T-02.02c-a).
//!
//! The minimal §3-named-weight assembly that drives the already-verified
//! `embed`/`attention`/`block` on the `reference.py` §3 named weights:
//!
//!   * `embed_tokens(ids) -> [T,dim]` — gather of the `tok_embeddings`
//!     `[vocab,dim]` table rows by id (`syrinx_core::embed`).
//!   * `layer_attention(x[T,dim], layer, positions) -> [T,dim]` — fetch the
//!     `layers.{L}.attention.w{q,k,v,o}.weight` `[dim,dim]` projections by name
//!     and run the verified `attention` (no pre-norm here — the caller norms).
//!   * `transformer_block(h[T,dim], layer, positions) -> [T,dim]` — fetch every
//!     `layers.{L}.` weight by name (`attention_norm.weight`, the four attention
//!     projections, `ffn_norm.weight`, `feed_forward.w{1,3,2}.weight`) and run
//!     the verified pre-RMSNorm residual `block`.
//!
//! Lives in its own source file so the task-scoped mutation gate targets only
//! this assembly and leaves the byte-identical `lib.rs`/`block.rs` untouched.

use crate::{attention, block};
use syrinx_core::{embed, weights, Tensor};

/// LM config (PARITY.md §3 / REFERENCE.md): vocab=512, dim=128, ffn_hidden=256.
const VOCAB: usize = 512;
const DIM: usize = 128;
const FFN_HIDDEN: usize = 256;

/// The `[rows, cols]` weight matrix named `name` from the T-02.01c name-seeded
/// stream (`[out,in]` orientation, as `linear` expects).
fn mat(name: &str, rows: usize, cols: usize) -> Tensor {
    Tensor::new(weights(name, rows * cols), vec![rows, cols])
}

/// The `[d]` weight vector named `name` (an RMSNorm weight).
fn vecw(name: &str, d: usize) -> Tensor {
    Tensor::new(weights(name, d), vec![d])
}

/// `embed_tokens(ids) -> [ids.len(), dim]`: gather row `id` of the
/// `tok_embeddings` `[vocab, dim]` table for each token id.
pub fn embed_tokens(token_ids: &[usize]) -> Tensor {
    let table = mat("tok_embeddings", VOCAB, DIM);
    embed(token_ids, &table)
}

/// `layer_attention(x[T,dim], layer, positions) -> [T,dim]`: the verified
/// `attention` over the `layers.{layer}.attention.w{q,k,v,o}.weight` projections.
pub fn layer_attention(x: &Tensor, layer: usize, positions: &[usize]) -> Tensor {
    let wq = mat(&format!("layers.{layer}.attention.wq.weight"), DIM, DIM);
    let wk = mat(&format!("layers.{layer}.attention.wk.weight"), DIM, DIM);
    let wv = mat(&format!("layers.{layer}.attention.wv.weight"), DIM, DIM);
    let wo = mat(&format!("layers.{layer}.attention.wo.weight"), DIM, DIM);
    attention(x, &wq, &wk, &wv, &wo, positions)
}

/// `transformer_block(h[T,dim], layer, positions) -> [T,dim]`: the verified
/// pre-RMSNorm residual `block` over every `layers.{layer}.` named weight.
pub fn transformer_block(h: &Tensor, layer: usize, positions: &[usize]) -> Tensor {
    let attn_norm = vecw(&format!("layers.{layer}.attention_norm.weight"), DIM);
    let ffn_norm = vecw(&format!("layers.{layer}.ffn_norm.weight"), DIM);
    let wq = mat(&format!("layers.{layer}.attention.wq.weight"), DIM, DIM);
    let wk = mat(&format!("layers.{layer}.attention.wk.weight"), DIM, DIM);
    let wv = mat(&format!("layers.{layer}.attention.wv.weight"), DIM, DIM);
    let wo = mat(&format!("layers.{layer}.attention.wo.weight"), DIM, DIM);
    let w1 = mat(&format!("layers.{layer}.feed_forward.w1.weight"), FFN_HIDDEN, DIM);
    let w3 = mat(&format!("layers.{layer}.feed_forward.w3.weight"), FFN_HIDDEN, DIM);
    let w2 = mat(&format!("layers.{layer}.feed_forward.w2.weight"), DIM, FFN_HIDDEN);
    block(
        h,
        &attn_norm,
        &ffn_norm,
        (&wq, &wk, &wv, &wo),
        (&w1, &w3, &w2),
        positions,
    )
}
