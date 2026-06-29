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
    /// Build from the resolved config (RoPE tables sized to the codebook count). `dt` is
    /// the compute dtype — the codebook-axis RoPE tables are materialised in it so they
    /// match the `dt`-typed fast activations (f32 on CPU, bf16 on GPU).
    pub fn new(cfg: FishConfig, dev: &candle_core::Device, dt: candle_core::DType) -> Result<Self> {
        let fast = &cfg.fast.transformer;
        let (cos, sin) =
            precompute_rope(cfg.codec.num_codebooks, fast.head_dim, fast.rope_base, dev, dt)?;
        Ok(Self { cos, sin, cfg })
    }

    fn attn_shape(&self, w: &Weights) -> AttnShape {
        let f = &self.cfg.fast.transformer;
        // The REAL s2-pro fast AR (`audio_decoder.*`) has the SAME block geometry as the
        // slow backbone (head_dim 128, GQA 32:8, fused wqkv 6144 = q4096+k1024+v1024) but
        // carries NO QK-norm and NO qkv/o bias — derive those from the actual tensors.
        AttnShape {
            n_head: f.n_head,
            n_local_heads: f.n_local_heads,
            head_dim: f.head_dim,
            qkv_bias: w.has("fast_layers.0.attention.wqkv.bias"),
            o_bias: w.has("fast_layers.0.attention.wo.bias"),
            qk_norm: w.has("fast_layers.0.attention.q_norm.weight"),
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
        let shape = self.attn_shape(w);
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
        // PARITY: residual-codebook logits go to the f32 sampler — cast up here so the
        // bf16 GPU path matches the f32 sampling semantics. Identity on the CPU path.
        w.linear(&fast_out, "fast_output.weight", None)?
            .to_dtype(candle_core::DType::F32)
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

    /// Batched fast AR: expand N frames' slow hiddens together. The fast head's per-frame KV
    /// cache becomes `[N, ...]`; the codebook-axis RoPE is per-POSITION (the codebook index),
    /// so it is the SAME for all N — the existing single-sample [`attention`](super::nn::attention)
    /// (whose `cos`/`sin` broadcast over the batch dim and which carries no mask) drives the
    /// batched step verbatim. Sampling is **per-sample**: residual `cb` of sample `i` is drawn
    /// from `samplers[i]` so each sample's PRNG stream matches a batch=1 run seeded the same.
    ///
    /// `hidden` is `[N, 1, fast_dim]`; `first_codes[i]` seeds sample `i`'s codebook-0; `active`
    /// marks samples still generating (a finished sample draws no codes — its RNG must not
    /// advance, and its row is discarded by the driver). Returns `N` frames `[num_codebooks]`,
    /// index 0 == `first_codes[i]`.
    pub fn expand_batch(
        &self,
        w: &Weights,
        hidden: &Tensor,
        first_codes: &[u32],
        active: &[bool],
        samplers: &mut [Sampler],
    ) -> Result<Vec<Vec<u32>>> {
        let n = first_codes.len();
        let n_cb = self.cfg.codec.num_codebooks;
        let residual = self.cfg.codec.residual_size;
        let fast_dim = self.cfg.fast.transformer.dim;
        debug_assert_eq!(active.len(), n);
        debug_assert_eq!(samplers.len(), n);

        let mut cache = KvCache::new(self.cfg.fast.transformer.n_layer); // batch dim = N

        // Position 0: prime the cache with each sample's slow hidden (logits discarded —
        // codebook-0 is the deterministic `first_codes[i]`).
        let h0 = hidden.reshape((n, 1, fast_dim))?;
        let _ = self.step(w, &h0, 0, &mut cache)?;

        let mut codes: Vec<Vec<u32>> = first_codes
            .iter()
            .map(|&c| {
                let mut v = Vec::with_capacity(n_cb);
                v.push(c);
                v
            })
            .collect();

        // Seed the next input with the embedding of each sample's codebook-0.
        let mut cur = w
            .embedding("fast_embeddings.weight", first_codes)?
            .reshape((n, 1, fast_dim))?;

        for cb in 1..n_cb {
            let logits = self.step(w, &cur, cb, &mut cache)?; // [N, 1, residual]
            let rows: Vec<Vec<f32>> = logits.reshape((n, residual))?.to_vec2()?;
            let mut next_codes = vec![0u32; n];
            for i in 0..n {
                // Inactive (finished) samples draw nothing — their RNG must not advance and
                // their codes are discarded by the driver. Use 0 as an inert placeholder.
                let code = if active[i] {
                    samplers[i].sample_codebook(&rows[i], &[])
                } else {
                    0
                };
                codes[i].push(code);
                next_codes[i] = code;
            }
            cur = w
                .embedding("fast_embeddings.weight", &next_codes)?
                .reshape((n, 1, fast_dim))?;
        }
        Ok(codes)
    }
}
