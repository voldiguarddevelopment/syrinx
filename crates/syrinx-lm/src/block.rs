//! syrinx-lm — the SwiGLU FFN and the pre-RMSNorm residual transformer block.
//!
//! T-02.02b: an exact transcription of `reference.py` §5.3 (SwiGLU FFN) and
//! §5.1 (pre-RMSNorm residual block) = PARITY.md §5.3/§5.1. Both fns COMPOSE the
//! already-verified `syrinx-core` reference primitives (`silu`/`linear`/`mul`/
//! `add`/`rmsnorm`) and the T-02.02a `attention`, so this file introduces no
//! bare arithmetic of its own. It lives apart from `lib.rs` (which holds
//! `attention`) so the mutation gate scopes its mutants to the frozen
//! `tests/block_prop.rs` and leaves the byte-identical `attention` source
//! untouched.

use crate::attention;
use syrinx_core::{add, linear, mul, rmsnorm, silu, Tensor};

/// RMSNorm epsilon (PARITY.md "Global": `eps = 1e-5`).
const EPS: f32 = 1e-5;

/// `swiglu_ffn(x[T,dim], w1, w3, w2) -> [T,dim]` (`reference.py` §5.3).
///
/// `gate = silu(linear(x, w1))`, `up = linear(x, w3)`,
/// `out = linear(gate ⊙ up, w2)` — `w1` is the gate `[ffn_hidden,dim]`, `w3` the
/// up `[ffn_hidden,dim]`, `w2` the down `[dim,ffn_hidden]`, and `⊙` is the
/// elementwise Hadamard product on the hidden dim.
pub fn swiglu_ffn(x: &Tensor, w1: &Tensor, w3: &Tensor, w2: &Tensor) -> Tensor {
    let gate = silu(&linear(x, w1, None));
    let up = linear(x, w3, None);
    let hidden = mul(&gate, &up).unwrap();
    linear(&hidden, w2, None)
}

/// `block(h[T,dim], attn_norm_w, ffn_norm_w, attn_weights, ffn_weights,
/// positions) -> [T,dim]` (`reference.py` §5.1).
///
/// Pre-RMSNorm residual order: `h = h + attention(rmsnorm(h, attn_norm_w))` then
/// `h = h + swiglu_ffn(rmsnorm(h, ffn_norm_w))`. Each residual adds the sub-block
/// output to the PRE-norm `h` (the value before that sub-block's norm), never to
/// the normed tensor. `attn_weights` is the `(wq,wk,wv,wo)` tuple §5.2 consumes;
/// `ffn_weights` is the `(w1,w3,w2)` tuple §5.3 consumes.
pub fn block(
    h: &Tensor,
    attn_norm_w: &Tensor,
    ffn_norm_w: &Tensor,
    attn_weights: (&Tensor, &Tensor, &Tensor, &Tensor),
    ffn_weights: (&Tensor, &Tensor, &Tensor),
    positions: &[usize],
) -> Tensor {
    let (wq, wk, wv, wo) = attn_weights;
    let (w1, w3, w2) = ffn_weights;

    // Attention sub-block: residual on the PRE-norm `h`.
    let n1 = rmsnorm(h, attn_norm_w, EPS);
    let attn_out = attention(&n1, wq, wk, wv, wo, positions);
    let h = add(h, &attn_out).unwrap();

    // FFN sub-block: residual on the PRE-norm `h` (the post-attention value).
    let n2 = rmsnorm(&h, ffn_norm_w, EPS);
    let ffn_out = swiglu_ffn(&n2, w1, w3, w2);
    add(&h, &ffn_out).unwrap()
}
