//! The **slow** AR: the Qwen3-4B `fish_qwen3` backbone of the s2 dual-AR stack.
//!
//! Ports the `fish_qwen3_omni` Slow AR (`Qwen3-4B`, 36 layers, dim 2560, GQA 32:8,
//! head_dim 128, QK-RMSNorm, **no** qkv/o bias, tied output head, RoPE base 1e6):
//!
//! * [`SlowAr::embed`] — the **Multi-Codebook Fusion (MCF)** input embedding. The S2
//!   tech report eq. (1) is `x_{t+1} = e_t^LM + Σ_{k=0}^{N-1} E^(k)[q_t^(k)]`: the LM
//!   token embedding of the semantic token id, plus the summed per-codebook embeddings
//!   over all N codebooks. This is exactly the Fish dual-AR `embed`: `embeddings(inp[0])
//!   + Σ_i codebook_embeddings(inp[i+1] + i*codebook_size)`, zeroed on non-semantic
//!   positions (the reference `vq_embeds_sum[~is_semantic] = 0`).
//! * 36 `TransformerBlock`s (RMSNorm → GQA + QK-RMSNorm + interleaved RoPE → RMSNorm →
//!   SwiGLU), KV-cached.
//! * final RMSNorm → **tied** LM-head logits (`F.linear(norm(x), embeddings.weight)`;
//!   `text_config.tie_word_embeddings == true`).
//!
//! [`SlowAr::prefill`] / [`SlowAr::slow_step`] return raw, **unmasked** semantic logits
//! over the full 155 776 slow vocab (the driver owns the semantic constraint + RAS) and
//! the per-frame `hidden` the fast head expands — already passed through
//! `fast_project_in` (the slow→fast dim projection) when the checkpoint carries it.

use candle_core::{Result, Tensor};

use super::nn::{attention, precompute_rope, swiglu, AttnShape, KvCache, Weights};
use crate::common::config::FishConfig;

/// The loaded slow backbone + its KV cache + precomputed RoPE tables.
pub struct SlowAr {
    w: Weights,
    cfg: FishConfig,
    /// Full RoPE cos/sin tables `[max_seq_len_cap, head_dim/2]`.
    cos: Tensor,
    sin: Tensor,
    cache: KvCache,
    /// Whether the checkpoint carries a `fast_project_in` Linear (s2: the slow hidden is
    /// "linearly projected to the Fast AR's dimension"; identity when fast_dim == dim).
    has_project_in: bool,
}

impl SlowAr {
    /// Build from a loaded weight bag + the resolved config.
    ///
    /// The RoPE table is sized to `min(max_seq_len, rope_cap)` so a 32 768-position
    /// table isn't materialised eagerly (the driver bounds generation far below that);
    /// the cap is a generous default and is the only sizing knob here.
    pub fn new(w: Weights, cfg: FishConfig) -> Result<Self> {
        // PARITY: 32 768 RoPE positions × head_dim/2 is large; cap the precompute to a
        // practical synthesis length. Confirm the production max frame budget on-box.
        let rope_cap = cfg.slow.max_seq_len.min(8192);
        let (cos, sin) = precompute_rope(rope_cap, cfg.slow.head_dim, cfg.slow.rope_base, &w.dev)?;
        let has_project_in = w.has("fast_project_in.weight");
        let cache = KvCache::new(cfg.slow.n_layer);
        Ok(Self {
            w,
            cfg,
            cos,
            sin,
            cache,
            has_project_in,
        })
    }

    /// Borrow the weight bag (the fast head shares the same checkpoint bag).
    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// Clear the slow KV cache for a fresh utterance.
    pub fn reset(&mut self) {
        self.cache = KvCache::new(self.cfg.slow.n_layer);
    }

    fn attn_shape(&self) -> AttnShape {
        // The REAL s2-pro slow backbone carries QK-RMSNorm (`attention.{q,k}_norm`) and
        // NO qkv/o bias. Derive these from the actual checkpoint tensors so the block is
        // correct regardless of what the (sometimes incomplete) config.json declares.
        AttnShape {
            n_head: self.cfg.slow.n_head,
            n_local_heads: self.cfg.slow.n_local_heads,
            head_dim: self.cfg.slow.head_dim,
            qkv_bias: self.w.has("layers.0.attention.wqkv.bias"),
            o_bias: self.w.has("layers.0.attention.wo.bias"),
            qk_norm: self.w.has("layers.0.attention.q_norm.weight"),
            eps: self.cfg.slow.norm_eps,
        }
    }

    /// Embed the prompt / frame `inp` `[1 + num_codebooks, T]` (row 0 = slow-vocab
    /// token ids; rows `1..=num_codebooks` = the RVQ codes, 0 on text positions) into
    /// `[1, T, dim]` — the MCF input embedding (S2 report eq. 1).
    fn embed(&self, inp: &Tensor) -> Result<Tensor> {
        let rows = inp.dim(0)?; // 1 + num_codebooks
        let t = inp.dim(1)?;
        let n_cb = self.cfg.codec.num_codebooks;
        // The LM's per-codebook table stride (== the residual RVQ codebook size, 4096).
        let cb_size = self.cfg.codec.residual_size;
        debug_assert_eq!(rows, 1 + n_cb, "prompt must be [1 + num_codebooks, T]");

        let host: Vec<u32> = inp
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .to_vec1()?;
        let row = |r: usize| -> Vec<u32> { (0..t).map(|c| host[r * t + c]).collect() };

        let tok_ids = row(0);
        // e_t^LM: token embeddings for row 0.
        let emb_tok = self.w.embedding("embeddings.weight", &tok_ids)?; // [t, dim]

        // Σ_k E^(k)[q^(k)]: summed, offset codebook embeddings over the n_cb code rows.
        let mut acc: Option<Tensor> = None;
        for i in 0..n_cb {
            let ids: Vec<u32> = row(i + 1).iter().map(|&c| c + (i * cb_size) as u32).collect();
            let e = self.w.embedding("codebook_embeddings.weight", &ids)?; // [t, dim]
            acc = Some(match acc {
                None => e,
                Some(a) => (a + e)?,
            });
        }
        let vq_sum = acc.expect("num_codebooks >= 1"); // [t, dim]

        // Zero the codebook sum on non-semantic positions.
        let begin = self.cfg.semantic_begin_id;
        let end = self.cfg.semantic_end_id;
        let mask: Vec<f32> = tok_ids
            .iter()
            .map(|&id| if id >= begin && id <= end { 1.0 } else { 0.0 })
            .collect();
        let mask = Tensor::from_vec(mask, (t, 1), &self.w.dev)?;
        let vq_sum = vq_sum.broadcast_mul(&mask)?;

        let x = (emb_tok + vq_sum)?; // [t, dim]
        x.reshape((1, t, self.cfg.slow.dim))
    }

    /// Run the 36-layer decoder stack over `embeds` `[1, t_new, dim]` at absolute
    /// positions `offset..offset+t_new`, KV-cached, returning the last hidden state
    /// `[1, t_new, dim]` (pre-final-RMSNorm).
    fn run_layers(&mut self, embeds: &Tensor, offset: usize, causal: bool) -> Result<Tensor> {
        let t_new = embeds.dim(1)?;
        let cos = self.cos.narrow(0, offset, t_new)?;
        let sin = self.sin.narrow(0, offset, t_new)?;
        let mask = if causal {
            Some(super::nn::causal_mask_at(offset, t_new, &self.w.dev)?)
        } else {
            None
        };
        let shape = self.attn_shape();
        let eps = self.cfg.slow.norm_eps;
        let mut h = embeds.clone();
        for l in 0..self.cfg.slow.n_layer {
            let pre = format!("layers.{l}");
            let r = h.clone();
            let hn = self
                .w
                .rms_norm(&h, &format!("{pre}.attention_norm.weight"), eps)?;
            let a = attention(
                &self.w,
                &format!("{pre}.attention"),
                &hn,
                &cos,
                &sin,
                mask.as_ref(),
                shape,
                &mut self.cache,
                l,
            )?;
            h = (r + a)?;
            let r = h.clone();
            let hn = self.w.rms_norm(&h, &format!("{pre}.ffn_norm.weight"), eps)?;
            h = (r + swiglu(&self.w, &format!("{pre}.feed_forward"), &hn)?)?;
        }
        self.cache.advance(t_new);
        Ok(h)
    }

    /// Final RMSNorm → tied LM-head logits for the **last** position of `h`
    /// `[1, t_new, dim]`, plus the (optionally `fast_project_in`-projected) hidden for
    /// that same last position. Returns `(semantic_logits [vocab], hidden [1, 1, fast_dim])`.
    fn head(&self, h: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = h.dim(1)?;
        let last = h.narrow(1, t - 1, 1)?; // [1, 1, dim] — pre-norm residual
        let eps = self.cfg.slow.norm_eps;
        let normed = self.w.rms_norm(&last, "norm.weight", eps)?; // [1, 1, dim]

        // Tied head: the REAL s2-pro checkpoint ships NO `lm_head`/`output.weight`, so the
        // slow head reuses `embeddings.weight` (`F.linear(norm, embeddings)`). Fall back to
        // the tied path whenever a dedicated head is absent, independent of the config flag.
        let logits = if self.cfg.slow.tie_word_embeddings || !self.w.has("output.weight") {
            let emb = self.w.g("embeddings.weight")?; // [vocab, dim]
            self.w.linear_w(&normed, &emb)?
        } else {
            self.w.linear(&normed, "output.weight", None)?
        };
        let logits = logits.reshape((self.cfg.slow.vocab_size,))?;

        // Hidden for the fast head: the report projects the slow hidden h_t^slow to the
        // Fast AR's dim via `fast_project_in` (identity when fast_dim == slow dim — the
        // real s2-pro case, both 2560; the checkpoint carries NO `fast_project_in`).
        // PARITY: `audio_decoder_config.norm_fastlayer_input` is true; the reference may
        // feed the FINAL-RMSNorm'd hidden (`normed`, above) as the position-0 conditioning
        // prefix rather than this pre-norm residual. Confirm the conditioning tap on-box.
        let hidden = if self.has_project_in {
            // PARITY: confirm whether fast_project_in carries a bias on-box.
            if self.w.has("fast_project_in.bias") {
                self.w
                    .linear(&last, "fast_project_in.weight", Some("fast_project_in.bias"))?
            } else {
                self.w.linear(&last, "fast_project_in.weight", None)?
            }
        } else {
            last
        };
        Ok((logits, hidden))
    }

    /// Prefill the encoded prompt `[1 + num_codebooks, T]` into the slow KV cache,
    /// returning the slow step for the **last** prompt position.
    pub fn prefill(&mut self, prompt: &Tensor) -> Result<(Tensor, Tensor)> {
        let embeds = self.embed(prompt)?;
        let offset = self.cache.len();
        let h = self.run_layers(&embeds, offset, true)?;
        self.head(&h)
    }

    /// One slow step on the previous full frame `[1 + num_codebooks]` at absolute
    /// position `pos`, advancing the cache by one. Returns `(semantic_logits, hidden)`.
    pub fn slow_step(&mut self, frame: &[u32], pos: usize) -> Result<(Tensor, Tensor)> {
        debug_assert_eq!(
            pos,
            self.cache.len(),
            "slow_step pos must equal the current cache length"
        );
        let n = frame.len();
        let inp = Tensor::from_vec(frame.to_vec(), (n, 1), &self.w.dev)?;
        let embeds = self.embed(&inp)?;
        // Single new token over the full cache: causal visibility is total ⇒ no mask.
        let h = self.run_layers(&embeds, pos, false)?;
        self.head(&h)
    }
}
