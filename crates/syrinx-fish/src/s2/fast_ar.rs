//! The **fast** AR (depth-wise acoustic decoder): the `fish_qwen3_audio_decoder` head
//! that expands one frame's slow hidden into all `num_codebooks` residual RVQ codes.
//!
//! Per the S2 tech report: "a lightweight Fast AR network — 4 Transformer layers with
//! independent weights and embedding tables — to reconstruct the remaining fine-grained
//! acoustic details. … All N codebook layers share a single embedding table within the
//! Fast AR; the codebook layer identity is encoded through RoPE positional embeddings.
//! … h_t^slow is first linearly projected to the Fast AR's dimension and placed at
//! position 0 as a conditioning prefix."
//!
//! So the head is structurally the Fish dual-AR fast transformer:
//! * a single shared `fast_embeddings` table (indexed by *code value*; codebook
//!   identity is carried by the **codebook-axis RoPE** `fast_freqs_cis[input_pos]`);
//! * `n_layer` (=4) `TransformerBlock`s — `fish_qwen3_audio_decoder` runs **no**
//!   QK-norm and **no** qkv/o bias (config flags all false) — with a **local** KV cache
//!   of length `num_codebooks`, rebuilt per frame (`fast_expand` is `&self`);
//! * `fast_norm` → `fast_output` → the 4096-way residual-codebook logits.
//!
//! Codebook-0 is the deterministic semantic-derived code (`first_code`); positions
//! `1..num_codebooks` are sampled via the **driver-owned** [`Sampler`].

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
        let (cos, sin) = precompute_rope(cfg.codec.num_codebooks, fast.head_dim, fast.rope_base, dev)?;
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
    /// residual_size - 1)`. Deterministic. For s2 the semantic range is exactly
    /// `[151678, 155773]` (4096 wide), so this is `tok - 151678` clamped to `[0, 4095]`.
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

        // Position 0: prime the cache with the slow hidden (the conditioning prefix; its
        // logits are discarded — codebook-0 is the deterministic `first_code`).
        let h0 = hidden.reshape((1, 1, fast_dim))?;
        let _ = self.step(w, &h0, 0, &mut cache)?;

        let mut codes: Vec<u32> = Vec::with_capacity(n_cb);
        codes.push(first_code);

        // Seed the next input with the embedding of codebook-0 (shared fast table).
        let mut cur = w
            .embedding("fast_embeddings.weight", &[first_code])?
            .reshape((1, 1, fast_dim))?;

        // Positions 1..num_codebooks: the reference `range(1, num_codebooks)` loop. Fast
        // codebooks are drawn freely (no semantic constraint, no repetition penalty —
        // hence empty `recent`), via the driver-owned sampler.
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
