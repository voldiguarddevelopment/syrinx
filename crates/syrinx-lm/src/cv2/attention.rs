//! Qwen2 self-attention (full + KV-cached) and its RoPE / GQA / causal-mask helpers.
//!
//! Split out verbatim from the original single-file `real` port — the `attn`/`attn_cached`
//! methods plus the rotary-embedding (`rope`, `rope_cos_sin{,_at}`), GQA expansion
//! (`repeat_kv`) and additive causal-mask (`causal_mask{,_at}`) free functions. The
//! cross-module entry points (`attn`/`attn_cached`, called by the forward loops in
//! `mod.rs`, and the cos/sin/mask builders, also called there) are `pub(super)`; `rope`
//! and `repeat_kv` are only used here and stay private.

use super::{KvCache, Qwen2Lm, HEAD_DIM, HIDDEN, N_HEADS, N_KV, ROPE_THETA};
use candle_core::{Device, Result, Tensor, D};

impl Qwen2Lm {
    pub(super) fn attn(&self, x: &Tensor, layer: usize, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let p = format!("llm.model.model.layers.{layer}.self_attn");
        let (b, t, _) = x.dims3()?;
        let q = self.linear(x, &format!("{p}.q_proj.weight"), Some(&format!("{p}.q_proj.bias")))?;
        let k = self.linear(x, &format!("{p}.k_proj.weight"), Some(&format!("{p}.k_proj.bias")))?;
        let v = self.linear(x, &format!("{p}.v_proj.weight"), Some(&format!("{p}.v_proj.bias")))?;
        let q = q.reshape((b, t, N_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,14,t,64]
        let k = k.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t,64]
        let v = v.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t,64]
        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let k = repeat_kv(&k, N_HEADS / N_KV)?; // [b,14,t,64]
        let v = repeat_kv(&v, N_HEADS / N_KV)?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [b,14,t,t]
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?; // [b,14,t,64]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, HIDDEN))?;
        self.linear(&ctx, &format!("{p}.o_proj.weight"), None)
    }

    /// Cached attention for `t_new` query tokens at absolute positions
    /// `offset..offset+t_new`. Computes their Q/K/V, applies RoPE at the **absolute**
    /// positions (via `cos`/`sin` built for that range), appends the new K/V to
    /// `cache[layer]`, then attends the new queries over the **full** cached K/V
    /// (`offset+t_new` keys) under `mask` `[t_new, offset+t_new]`.
    ///
    /// This is numerically the same computation as `attn` restricted to the last
    /// `t_new` query rows of a full-sequence forward: identical Q/K/V projections,
    /// identical RoPE phases (absolute positions), identical causal visibility, so its
    /// logits match the full recompute to within fp rounding.
    pub(super) fn attn_cached(
        &self,
        x: &Tensor,
        layer: usize,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let p = format!("llm.model.model.layers.{layer}.self_attn");
        let (b, t, _) = x.dims3()?;
        let q = self.linear(x, &format!("{p}.q_proj.weight"), Some(&format!("{p}.q_proj.bias")))?;
        let k = self.linear(x, &format!("{p}.k_proj.weight"), Some(&format!("{p}.k_proj.bias")))?;
        let v = self.linear(x, &format!("{p}.v_proj.weight"), Some(&format!("{p}.v_proj.bias")))?;
        let q = q.reshape((b, t, N_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,14,t_new,64]
        let k = k.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t_new,64]
        let v = v.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t_new,64]
        // RoPE the new Q and new K at their absolute positions, then commit the new
        // (post-RoPE) K and raw V to the cache, getting back the full cached K/V.
        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let (k_full, v_full) = cache.append(layer, &k, &v)?; // [b,2,offset+t_new,64]
        // GQA: repeat the *cached* KV heads up to the query-head count, then attend.
        let k_full = repeat_kv(&k_full, N_HEADS / N_KV)?; // [b,14,offset+t_new,64]
        let v_full = repeat_kv(&v_full, N_HEADS / N_KV)?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k_full.transpose(2, 3)?.contiguous()?)? * scale)?; // [b,14,t_new,offset+t_new]
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v_full)?; // [b,14,t_new,64]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, HIDDEN))?;
        self.linear(&ctx, &format!("{p}.o_proj.weight"), None)
    }
}

/// Apply rotary position embedding (HF `rotate_half` convention) to `[b, h, t, d]`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)?.add(&rot.broadcast_mul(sin)?)
}

/// GQA: expand `[b, kv, t, d]` KV heads so each serves `n` query heads -> `[b, kv*n, t, d]`.
fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, kv, n, t, d))?.reshape((b, kv * n, t, d))
}

/// Build RoPE cos/sin tables `[t, head_dim]` for positions `0..t`.
pub(super) fn rope_cos_sin(t: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    rope_cos_sin_at(0, t, dev)
}

/// Build RoPE cos/sin tables `[t, head_dim]` for the **absolute** positions
/// `offset..offset+t`. Caching feeds only the new token(s), so their rotary phase
/// must use their true absolute position (= `offset`, the current cache length),
/// not a reset-to-zero position — this is the load-bearing correctness detail.
pub(super) fn rope_cos_sin_at(offset: usize, t: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), dev)?;
    let pos: Vec<f32> = (0..t).map(|i| (offset + i) as f32).collect();
    let pos = Tensor::from_vec(pos, (t,), dev)?;
    let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [t, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [t, head_dim]
    Ok((emb.cos()?, emb.sin()?))
}

/// Additive causal mask `[t, t]`: 0 on/below the diagonal, -inf above.
pub(super) fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
    causal_mask_at(0, t, dev)
}

/// Additive causal mask `[t_new, offset + t_new]` for `t_new` queries at absolute
/// positions `offset..offset+t_new` attending over all `offset+t_new` keys.
/// Query row `i` (absolute position `offset+i`) may attend key column `j` iff
/// `j <= offset + i` (causal); entries above that are `-inf`. With `offset = 0`
/// this is the square mask; with one new query over a full cache it is all-zeros.
pub(super) fn causal_mask_at(offset: usize, t_new: usize, dev: &Device) -> Result<Tensor> {
    let total = offset + t_new;
    let mut data = vec![0f32; t_new * total];
    for i in 0..t_new {
        let q_abs = offset + i;
        for j in (q_abs + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (t_new, total), dev)
}
