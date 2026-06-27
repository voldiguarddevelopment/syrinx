//! The **fast** AR head: the s1 `DualARTransformer` fast transformer that expands one
//! frame's slow hidden into all `num_codebooks` residual RVQ codes.
//!
//! Ports `forward_generate_fast` + the `decode_one_token_ar` fast loop from the
//! reference:
//!
//! * a single shared `fast_embeddings` table (s1 indexes it by *code value*, not by
//!   codebook id — the codebook identity is carried by the **codebook-axis RoPE**
//!   position, `fast_freqs_cis[input_pos]`);
//! * `n_fast_layer` `TransformerBlock`s with a **local** KV cache whose length is the
//!   codebook count (rebuilt per frame — see [`crate::common::dualar`]'s contract that
//!   `fast_expand` is `&self`);
//! * `fast_norm` → `fast_output` → the residual-codebook logits.
//!
//! The frame is seeded with codebook-0 (`first_code`, the clamped semantic-derived
//! code); positions `1..num_codebooks` are sampled via the **driver-owned**
//! [`Sampler`], so the shared PRNG stream stays deterministic.

use candle_core::{Result, Tensor};

use super::nn::{attention, precompute_rope, swiglu, AttnShape, KvCache, Weights};
use crate::common::config::FishConfig;
use crate::common::sampling::Sampler;

/// The fast head: precomputed codebook-axis RoPE tables over the (immutable) shared
/// checkpoint weights held by the slow backbone.
pub struct FastAr {
    /// RoPE cos/sin over the codebook axis `[num_codebooks, fast_head_dim/2]`.
    cos: Tensor,
    sin: Tensor,
    cfg: FishConfig,
}

impl FastAr {
    /// Build from the resolved config (RoPE tables sized to the codebook count).
    pub fn new(cfg: FishConfig, dev: &candle_core::Device) -> Result<Self> {
        let fast = &cfg.fast.transformer;
        let (cos, sin) = precompute_rope(
            cfg.codec.num_codebooks,
            fast.head_dim,
            fast.rope_base,
            dev,
        )?;
        Ok(Self { cos, sin, cfg })
    }

    fn attn_shape(&self) -> AttnShape {
        let f = &self.cfg.fast.transformer;
        AttnShape {
            n_head: f.n_head,
            n_local_heads: f.n_local_heads,
            head_dim: f.head_dim,
            qkv_bias: f.attention_qkv_bias,
            o_bias: f.attention_o_bias,
            qk_norm: f.attention_qk_norm,
            eps: f.norm_eps,
        }
    }

    /// One fast step: run the fast layers over the single token `x` `[1, 1, fast_dim]`
    /// at codebook position `pos`, advancing `cache`, then `fast_norm` → `fast_output`.
    /// Returns the residual-codebook logits `[1, 1, residual_size]`.
    fn step(&self, w: &Weights, x: &Tensor, pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let cos = self.cos.narrow(0, pos, 1)?;
        let sin = self.sin.narrow(0, pos, 1)?;
        let eps = self.cfg.fast.transformer.norm_eps;
        let shape = self.attn_shape();
        let mut h = x.clone();
        // Single new token attending over the full local cache ⇒ no mask needed.
        for l in 0..self.cfg.fast.transformer.n_layer {
            let pre = format!("fast_layers.{l}");
            let r = h.clone();
            let hn = w.rms_norm(&h, &format!("{pre}.attention_norm.weight"), eps)?;
            let a = attention(
                w,
                &format!("{pre}.attention"),
                &hn,
                &cos,
                &sin,
                None,
                shape,
                cache,
                l,
            )?;
            h = (r + a)?;
            let r = h.clone();
            let hn = w.rms_norm(&h, &format!("{pre}.ffn_norm.weight"), eps)?;
            h = (r + swiglu(w, &format!("{pre}.feed_forward"), &hn)?)?;
        }
        cache.advance(1);
        let fast_out = w.rms_norm(&h, "fast_norm.weight", eps)?;
        w.linear(&fast_out, "fast_output.weight", None)
    }

    /// Map a sampled semantic token id to codebook-0: `clamp(tok - semantic_begin, 0,
    /// residual_size - 1)`. Deterministic (the reference `a = clamp(codebooks[0] -
    /// semantic_begin_id, 0, codebook_size - 1)`).
    pub fn first_code(&self, semantic_token: u32) -> u32 {
        let begin = self.cfg.semantic_begin_id;
        let raw = semantic_token.saturating_sub(begin) as i64;
        let max = (self.cfg.codec.residual_size as i64) - 1;
        raw.clamp(0, max) as u32
    }

    /// Run the fast AR for one frame. `hidden` is the slow hidden `[1, 1, fast_dim]`
    /// (already `fast_project_in`-projected by the slow backbone); `first_code` seeds
    /// codebook-0; each residual `1..num_codebooks` is sampled via `sampler`. Returns
    /// the full frame `[num_codebooks]` (index 0 == `first_code`).
    pub fn expand(
        &self,
        w: &Weights,
        hidden: &Tensor,
        first_code: u32,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        let n_cb = self.cfg.codec.num_codebooks;
        let fast_dim = self.cfg.fast.transformer.dim;
        let mut cache = KvCache::new(self.cfg.fast.transformer.n_layer);

        // Position 0: prime the cache with the slow hidden (its logits are discarded —
        // codebook-0 is the deterministic `first_code`, not a sample).
        let h0 = hidden.reshape((1, 1, fast_dim))?;
        let _ = self.step(w, &h0, 0, &mut cache)?;

        let mut codes: Vec<u32> = Vec::with_capacity(n_cb);
        codes.push(first_code);

        // Seed the next input with the embedding of codebook-0.
        let mut cur = w
            .embedding("fast_embeddings.weight", &[first_code])?
            .reshape((1, 1, fast_dim))?;

        // Positions 1..num_codebooks: the reference `range(1, num_codebooks)` loop. The
        // fast codebooks are drawn freely (no semantic constraint, no repetition
        // penalty — hence the empty `recent`), via the driver-owned sampler.
        for cb in 1..n_cb {
            let logits = self.step(w, &cur, cb, &mut cache)?;
            let logits = logits.reshape((self.cfg.codec.residual_size,))?;
            let v: Vec<f32> = logits.to_vec1()?;
            let code = sampler.sample_codebook(&v, &[]);
            codes.push(code);
            cur = w
                .embedding("fast_embeddings.weight", &[code])?
                .reshape((1, 1, fast_dim))?;
        }
        Ok(codes)
    }
}
