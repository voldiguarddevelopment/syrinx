//! The **slow** AR: the Llama-style `BaseTransformer` of the s1 `DualARTransformer`.
//!
//! Ports `fish_speech/models/text2semantic/llama.py` `BaseTransformer.{embed,
//! forward_generate}` for the s1 (`dual_ar`, no QK-norm, no qkv bias, tied output
//! head) variant:
//!
//! * [`SlowAr::embed`] — token-embedding + summed offset codebook-embeddings on the
//!   semantic positions (zeroed elsewhere), exactly the reference `embed`.
//! * per-layer `TransformerBlock` (RMSNorm → GQA + interleaved RoPE → RMSNorm →
//!   SwiGLU), KV-cached.
//! * final RMSNorm → **tied** LM-head logits (`F.linear(norm(x), embeddings.weight)`).
//!
//! [`SlowAr::prefill`] / [`SlowAr::slow_step`] return the raw, **unmasked** semantic
//! logits over the full slow vocab (the driver owns the semantic constraint + RAS)
//! and the per-frame `hidden` the fast head expands — already passed through
//! `fast_project_in` when the checkpoint has it (otherwise identity, as for s1 where
//! `fast_dim == dim`).

use candle_core::{Result, Tensor};

use super::nn::{attention, precompute_rope, swiglu, AttnShape, KvCache, Weights};
use crate::common::config::FishConfig;

/// The loaded slow backbone + its KV cache + precomputed RoPE tables.
pub struct SlowAr {
    w: Weights,
    cfg: FishConfig,
    /// Full RoPE cos/sin tables `[max_seq_len, head_dim/2]`.
    cos: Tensor,
    sin: Tensor,
    cache: KvCache,
    /// Whether the checkpoint carries a `fast_project_in` Linear (s1: usually not —
    /// `fast_dim == dim` ⇒ `nn.Identity`).
    has_project_in: bool,
}

impl SlowAr {
    /// Build from a loaded weight bag + the resolved config.
    pub fn new(w: Weights, cfg: FishConfig) -> Result<Self> {
        let (cos, sin) = precompute_rope(
            cfg.slow.max_seq_len,
            cfg.slow.head_dim,
            cfg.slow.rope_base,
            &w.dev,
        )?;
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

    /// Borrow the weight bag (the fast head shares the same checkpoint).
    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// Clear the slow KV cache for a fresh utterance.
    pub fn reset(&mut self) {
        self.cache = KvCache::new(self.cfg.slow.n_layer);
    }

    fn attn_shape(&self) -> AttnShape {
        AttnShape {
            n_head: self.cfg.slow.n_head,
            n_local_heads: self.cfg.slow.n_local_heads,
            head_dim: self.cfg.slow.head_dim,
            qkv_bias: self.cfg.slow.attention_qkv_bias,
            o_bias: self.cfg.slow.attention_o_bias,
            qk_norm: self.cfg.slow.attention_qk_norm,
            eps: self.cfg.slow.norm_eps,
        }
    }

    /// Embed the prompt / frame `inp` `[1 + num_codebooks, T]` (row 0 = slow-vocab
    /// token ids; rows `1..=num_codebooks` = the RVQ codes, 0 on text positions) into
    /// `[1, T, dim]`, reproducing the reference `BaseTransformer.embed`:
    /// `embeddings(inp[0]) + Σ_i codebook_embeddings(inp[i+1] + i*codebook_size)`,
    /// with the codebook sum zeroed on non-semantic positions.
    fn embed(&self, inp: &Tensor) -> Result<Tensor> {
        let rows = inp.dim(0)?; // 1 + num_codebooks
        let t = inp.dim(1)?;
        let n_cb = self.cfg.codec.num_codebooks;
        // The LM's per-codebook table stride (== the residual RVQ codebook size).
        let cb_size = self.cfg.codec.residual_size;
        debug_assert_eq!(rows, 1 + n_cb, "prompt must be [1 + num_codebooks, T]");

        let host: Vec<u32> = inp.to_dtype(candle_core::DType::U32)?.flatten_all()?.to_vec1()?;
        let row = |r: usize| -> Vec<u32> { (0..t).map(|c| host[r * t + c]).collect() };

        let tok_ids = row(0);
        // Token embeddings for row 0.
        let emb_tok = self.w.embedding("embeddings.weight", &tok_ids)?; // [t, dim]

        // Summed, offset codebook embeddings over the n_cb code rows.
        let mut acc: Option<Tensor> = None;
        for i in 0..n_cb {
            let ids: Vec<u32> = row(i + 1)
                .iter()
                .map(|&c| c + (i * cb_size) as u32)
                .collect();
            let e = self.w.embedding("codebook_embeddings.weight", &ids)?; // [t, dim]
            acc = Some(match acc {
                None => e,
                Some(a) => (a + e)?,
            });
        }
        let vq_sum = acc.expect("num_codebooks >= 1"); // [t, dim]

        // Zero the codebook sum on non-semantic positions (reference `vq_embeds_sum[~is_semantic] = 0`).
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

    /// Run the decoder stack over `embeds` `[1, t_new, dim]` at absolute positions
    /// `offset..offset+t_new`, KV-cached, returning the last hidden state `[1, t_new, dim]`
    /// (pre-final-RMSNorm — the reference `x` that becomes `hidden_states`).
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
            let hn = self.w.rms_norm(&h, &format!("{pre}.attention_norm.weight"), eps)?;
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

        // Tied head: F.linear(slow_out, embeddings.weight). (s1 `tie_word_embeddings`.)
        let logits = if self.cfg.slow.tie_word_embeddings {
            let emb = self.w.g("embeddings.weight")?; // [vocab, dim]
            self.w.linear_w(&normed, &emb)?
        } else {
            self.w.linear(&normed, "output.weight", None)?
        };
        let logits = logits.reshape((self.cfg.slow.vocab_size,))?;

        // Hidden for the fast head: the reference passes the PRE-norm `x` (since s1's
        // `norm_fastlayer_input` is false), then `fast_project_in` it.
        let hidden = if self.has_project_in {
            self.w
                .linear(&last, "fast_project_in.weight", Some("fast_project_in.bias"))?
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
