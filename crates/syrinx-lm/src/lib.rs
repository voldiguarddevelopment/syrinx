//! syrinx-lm — AR semantic LM forward pass + paralinguistic tokens.
//!
//! T-02.02a: the causal multi-head self-attention sub-block, an exact
//! transcription of `reference.py` §5.2 (= PARITY.md §5.2). It composes the
//! `syrinx-core` reference primitives (`linear` for the bias-free Q/K/V/O
//! projections, `rope` for the rotary embedding) over the `Tensor` contract and
//! does the per-head scaled-dot-product attention by hand:
//!
//!   * project `x[T,dim]` to Q/K/V with `wq`/`wk`/`wv` (bias-free),
//!   * split each `[T,dim]` into `n_heads = 4` contiguous head slices of
//!     `head_dim = dim / n_heads` (head `hi` owns columns
//!     `hi*head_dim .. (hi+1)*head_dim`),
//!   * apply RoPE to Q and K only (not V),
//!   * score `(q·k) * (1/sqrt(head_dim))` for keys `j <= i`, `-inf` otherwise
//!     (the additive causal mask, applied *after* the scale),
//!   * softmax over keys, take the value-weighted context, scatter it back to
//!     the head's columns,
//!   * project the assembled context with `wo` (bias-free).

use syrinx_core::{linear, rope, Tensor};

/// LM config (PARITY.md §3): four attention heads.
const N_HEADS: usize = 4;
/// RoPE base (PARITY.md §4.7 / §5.2).
const THETA: f32 = 10000.0;

/// Copy head `hi`'s contiguous `head_dim` columns out of a `[T,dim]` flat buffer
/// into a packed `[T,head_dim]` buffer. Head `hi` owns columns
/// `base .. base+head_dim` where `base = hi*head_dim` (`reference.py` §5.2).
fn gather_head(flat: &[f32], base: usize, dim: usize, head_dim: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * head_dim];
    for i in 0..t {
        for d in 0..head_dim {
            out[i * head_dim + d] = flat[i * dim + base + d];
        }
    }
    out
}

/// Causal multi-head self-attention (`reference.py` §5.2).
///
/// `attention(x[T,dim], wq, wk, wv, wo, positions) -> [T,dim]`: bias-free Q/K/V
/// projections, a contiguous `n_heads`-way split, RoPE on Q/K only, scaled
/// (`1/sqrt(head_dim)`) dot-product scores with an additive causal mask applied
/// after the scale, softmax over keys, the value-weighted context, and the `wo`
/// output projection.
pub fn attention(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    positions: &[usize],
) -> Tensor {
    let t = x.shape()[0];
    let dim = x.shape()[1];
    let head_dim = dim / N_HEADS;
    // Score scale: exactly `1/sqrt(head_dim)`, applied to the q·k dot product
    // *before* the additive causal mask.
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Bias-free Q/K/V projections.
    let q = linear(x, wq, None);
    let k = linear(x, wk, None);
    let v = linear(x, wv, None);

    let mut ctx = vec![0.0f32; t * dim];
    for hi in 0..N_HEADS {
        // Head `hi` owns the contiguous column block starting at `hi*head_dim`.
        let base = hi * head_dim;
        let qh = gather_head(q.data(), base, dim, head_dim, t);
        let kh = gather_head(k.data(), base, dim, head_dim, t);
        let vh = gather_head(v.data(), base, dim, head_dim, t);
        // RoPE applies to Q and K only — V is gathered raw.
        let qh = rope(&Tensor::new(qh, vec![t, 1, head_dim]), positions, THETA);
        let kh = rope(&Tensor::new(kh, vec![t, 1, head_dim]), positions, THETA);
        let qh = qh.data();
        let kh = kh.data();

        for i in 0..t {
            // scores over keys j; future keys (j > i) keep the -inf mask.
            let mut scores = vec![f64::NEG_INFINITY; t];
            for j in 0..=i {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += qh[i * head_dim + d] as f64 * kh[j * head_dim + d] as f64;
                }
                scores[j] = dot * scale; // scale BEFORE the (implicit) mask add
            }
            // softmax over keys (future keys carry -inf -> exp = 0).
            let mut s = 0.0f64;
            let mut e = vec![0.0f64; t];
            for j in 0..t {
                e[j] = scores[j].exp();
                s += e[j];
            }
            // value-weighted context, scattered back to head `hi`'s columns.
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for j in 0..t {
                    acc += (e[j] / s) * vh[j * head_dim + d] as f64;
                }
                ctx[i * dim + base + d] = acc as f32;
            }
        }
    }

    // Output projection (bias-free).
    linear(&Tensor::new(ctx, vec![t, dim]), wo, None)
}
