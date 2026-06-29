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

use candle_core::{DType, Result, Tensor};

use super::nn::{
    attention, attention_batched, precompute_rope, swiglu, AttnShape, KvCache, Weights,
};
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
    /// Per-sample left-pad lengths for the BATCHED generation path (set by
    /// [`SlowAr::prefill_batch`], read by [`SlowAr::slow_step_batch`]). Sample `i`'s real
    /// token at physical column `c` has RoPE position `c - pad_lens[i]`, so padding is
    /// position-invisible and the real tokens keep positions `0..T_i`. Empty on the
    /// single-sample path (which never touches it).
    pad_lens: Vec<usize>,
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
        let (cos, sin) =
            precompute_rope(rope_cap, cfg.slow.head_dim, cfg.slow.rope_base, &w.dev, w.dt)?;
        let has_project_in = w.has("fast_project_in.weight");
        let cache = KvCache::new(cfg.slow.n_layer);
        Ok(Self {
            w,
            cfg,
            cos,
            sin,
            cache,
            has_project_in,
            pad_lens: Vec::new(),
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

        // Identify the semantic positions (the only ones that carry codebook codes).
        let begin = self.cfg.semantic_begin_id;
        let end = self.cfg.semantic_end_id;
        let is_semantic: Vec<bool> = tok_ids.iter().map(|&id| id >= begin && id <= end).collect();

        // Zero the codebook sum on non-semantic positions (reference: `vq_embeds_sum[~is_semantic] = 0`).
        let mask: Vec<f32> = is_semantic
            .iter()
            .map(|&s| if s { 1.0 } else { 0.0 })
            .collect();
        // Cast the f32 mask to the compute dtype so the multiply doesn't mix dtypes
        // (Candle errors on f32 × bf16); identity for the f32 CPU path.
        let mask = Tensor::from_vec(mask, (t, 1), &self.w.dev)?.to_dtype(self.w.dt)?;
        let vq_sum = vq_sum.broadcast_mul(&mask)?;

        let x = (emb_tok + vq_sum)?; // [t, dim]

        // FIX #1 — MCF embedding scale (`scale_codebook_embeddings=True`, llama.py:416-420).
        // On SEMANTIC positions only, divide the combined embedding by sqrt(num_codebooks + 1)
        // (= 1/sqrt(11) for N=10); text positions are left unchanged. Build a per-row scale
        // (1/sqrt(N+1) on semantic rows, 1.0 on text rows) and broadcast-multiply.
        let inv_scale = 1.0f32 / ((n_cb as f32) + 1.0).sqrt();
        let scale: Vec<f32> = is_semantic
            .iter()
            .map(|&s| if s { inv_scale } else { 1.0 })
            .collect();
        let scale = Tensor::from_vec(scale, (t, 1), &self.w.dev)?.to_dtype(self.w.dt)?;
        let x = x.broadcast_mul(&scale)?; // [t, dim]

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
            Some(super::nn::causal_mask_at(offset, t_new, &self.w.dev, self.w.dt)?)
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
        // PARITY: the sampler (and the driver's RAS / semantic constraint) operate in
        // f32; cast the head logits up before returning. Identity on the f32 CPU path.
        let logits = logits
            .reshape((self.cfg.slow.vocab_size,))?
            .to_dtype(candle_core::DType::F32)?;

        // Hidden for the fast head. FIX #3 — `norm_fastlayer_input=True` (llama.py:459-461):
        // the fast head is conditioned on `self.norm(x)`, i.e. the SAME post-final-RMSNorm
        // hidden (`normed`) used for the slow logits — NOT the pre-norm residual `last`.
        // This checkpoint carries NO `fast_project_in` (fast_dim == slow dim, both 2560), so
        // no projection is needed; when a project_in IS present it operates on the normed
        // hidden too.
        let hidden = if self.has_project_in {
            // PARITY: confirm whether fast_project_in carries a bias on-box.
            if self.w.has("fast_project_in.bias") {
                self.w
                    .linear(&normed, "fast_project_in.weight", Some("fast_project_in.bias"))?
            } else {
                self.w.linear(&normed, "fast_project_in.weight", None)?
            }
        } else {
            normed
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

    // ====================================================================================
    // Batched generation path (N samples rendered together). The 5B forward is memory-
    // bandwidth-bound at batch=1 (one weight read per token), so batching N short samples
    // gives ~Nx throughput. The single-sample path above is ADDITIVE-untouched.
    //
    // Left-pad scheme: the N prompts have different lengths T_i. Pad each on the LEFT to
    // Tmax = max(T_i) (real tokens right-aligned in columns [pad_i, Tmax)). Then:
    //   * RoPE positions: sample i's physical column c has position `c - pad_i` (clamped to
    //     0 in the pad region), so the real tokens keep positions 0..T_i and the padding is
    //     position-invisible. After prefill, physical position `pos` (≥ Tmax) maps to
    //     sample i's real position `pos - pad_i` (= T_i for the first generated frame).
    //   * Attention mask: a real query attends only real keys j ≥ pad_i, causally (j ≤ qi);
    //     a pad query (discarded) attends causally in the pad region (kept non-empty so the
    //     softmax never sees an all -inf row → no NaN). Padded keys are -inf for everyone,
    //     so a sample's left-pad can never leak into any real position.
    //   * The LAST physical column (Tmax-1) is a real token for EVERY sample (right-aligned),
    //     so the prefill head reads logits/hidden from that single column for all samples.
    // ====================================================================================

    /// Run the decoder stack over a left-padded batch `embeds` `[N, t_new, dim]` with
    /// **per-sample** RoPE tables `cos`/`sin` `[N, t_new, head_dim/2]` and an optional
    /// per-sample additive mask `[N, 1, t_new, total]`, KV-cached. Returns `[N, t_new, dim]`
    /// (pre-final-RMSNorm) and advances the cache by `t_new`.
    fn run_layers_batch(
        &mut self,
        embeds: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t_new = embeds.dim(1)?;
        let shape = self.attn_shape();
        let eps = self.cfg.slow.norm_eps;
        let mut h = embeds.clone();
        for l in 0..self.cfg.slow.n_layer {
            let pre = format!("layers.{l}");
            let r = h.clone();
            let hn = self
                .w
                .rms_norm(&h, &format!("{pre}.attention_norm.weight"), eps)?;
            let a = attention_batched(
                &self.w,
                &format!("{pre}.attention"),
                &hn,
                cos,
                sin,
                mask,
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

    /// Batched embed: build `[N, Tmax, dim]` by embedding each prompt with the (unchanged)
    /// single-sample MCF [`SlowAr::embed`] and LEFT-padding the result with zero rows. The
    /// pad rows are masked out as keys for every real position, so a zero embedding there is
    /// inert; using the real `embed` per sample keeps the MCF math byte-identical to batch=1.
    fn embed_batch_padded(&self, prompts: &[Tensor], pad: &[usize], tmax: usize) -> Result<Tensor> {
        let dim = self.cfg.slow.dim;
        let mut rows: Vec<Tensor> = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            let e = self.embed(p)?; // [1, T_i, dim]
            let padded = if pad[i] == 0 {
                e
            } else {
                let zeros = Tensor::zeros((1, pad[i], dim), self.w.dt, &self.w.dev)?;
                Tensor::cat(&[&zeros, &e], 1)? // [1, Tmax, dim]
            };
            debug_assert_eq!(padded.dim(1)?, tmax, "left-padded row must reach Tmax");
            rows.push(padded);
        }
        Tensor::cat(&rows, 0) // [N, Tmax, dim]
    }

    /// Gather per-sample RoPE cos/sin for the supplied flat `position_ids` (length
    /// `N * t_new`, row-major over `[N, t_new]`) → `(cos, sin)` each `[N, t_new, half]`.
    // PARITY: positions index the precomputed RoPE table (capped at `rope_cap` in `new`),
    // exactly like the single path's `self.cos.narrow`. Generation beyond `rope_cap` would
    // overflow the table on BOTH paths; the driver bounds frames well below it.
    fn gather_rope(&self, position_ids: &[u32], n: usize, t_new: usize) -> Result<(Tensor, Tensor)> {
        let half = self.cfg.slow.head_dim / 2;
        let idx = Tensor::from_vec(position_ids.to_vec(), (position_ids.len(),), &self.w.dev)?;
        let cos = self.cos.index_select(&idx, 0)?.reshape((n, t_new, half))?;
        let sin = self.sin.index_select(&idx, 0)?.reshape((n, t_new, half))?;
        Ok((cos, sin))
    }

    /// Final RMSNorm → tied LM-head logits + the fast-head hidden for the **last** column of
    /// a batched hidden `h` `[N, t_new, dim]`. Returns `(logits [N, vocab], hidden [N, 1,
    /// fast_dim])`. The per-step math matches [`SlowAr::head`]; only the batch axis is added.
    fn head_batch(&self, h: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = h.dim(0)?;
        let t = h.dim(1)?;
        let eps = self.cfg.slow.norm_eps;
        let last = h.narrow(1, t - 1, 1)?; // [N, 1, dim]
        let normed = self.w.rms_norm(&last, "norm.weight", eps)?; // [N, 1, dim]

        let logits = if self.cfg.slow.tie_word_embeddings || !self.w.has("output.weight") {
            let emb = self.w.g("embeddings.weight")?;
            self.w.linear_w(&normed, &emb)?
        } else {
            self.w.linear(&normed, "output.weight", None)?
        };
        let logits = logits
            .reshape((n, self.cfg.slow.vocab_size))?
            .to_dtype(DType::F32)?;

        let hidden = if self.has_project_in {
            if self.w.has("fast_project_in.bias") {
                self.w
                    .linear(&normed, "fast_project_in.weight", Some("fast_project_in.bias"))?
            } else {
                self.w.linear(&normed, "fast_project_in.weight", None)?
            }
        } else {
            normed
        };
        Ok((logits, hidden))
    }

    /// Prefill N left-padded prompts into a shared `[N, ...]` slow KV cache and return the
    /// slow step for the last (real) prompt column of every sample. Records each sample's
    /// left-pad length for the subsequent [`SlowAr::slow_step_batch`] calls. Returns
    /// `(logits [N, vocab], hidden [N, 1, fast_dim])`.
    pub fn prefill_batch(&mut self, prompts: &[Tensor]) -> Result<(Tensor, Tensor)> {
        let n = prompts.len();
        let lens: Vec<usize> = prompts
            .iter()
            .map(|p| p.dim(1))
            .collect::<Result<Vec<_>>>()?;
        let tmax = lens.iter().copied().max().unwrap_or(0);
        let pad: Vec<usize> = lens.iter().map(|&l| tmax - l).collect();
        self.pad_lens = pad.clone();

        let embeds = self.embed_batch_padded(prompts, &pad, tmax)?; // [N, Tmax, dim]

        // Per-sample RoPE position ids: column c → max(c - pad_i, 0).
        let mut pos_ids = vec![0u32; n * tmax];
        for (s, &ps) in pad.iter().enumerate() {
            for c in 0..tmax {
                pos_ids[s * tmax + c] = c.saturating_sub(ps) as u32;
            }
        }
        let (cos, sin) = self.gather_rope(&pos_ids, n, tmax)?;

        // Per-sample causal + left-pad mask [N, 1, Tmax, Tmax].
        let mut data = vec![0f32; n * tmax * tmax];
        for (s, &ps) in pad.iter().enumerate() {
            for qi in 0..tmax {
                for j in 0..tmax {
                    let allow = if qi >= ps {
                        // real query: causal over the sample's real keys only
                        j >= ps && j <= qi
                    } else {
                        // pad query (output discarded): causal in the pad region; kept
                        // non-empty (j == qi always allowed) so softmax never gets all -inf
                        j <= qi
                    };
                    if !allow {
                        data[(s * tmax + qi) * tmax + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mask = Tensor::from_vec(data, (n, tmax, tmax), &self.w.dev)?
            .to_dtype(self.w.dt)?
            .reshape((n, 1, tmax, tmax))?;

        let h = self.run_layers_batch(&embeds, &cos, &sin, Some(&mask))?;
        self.head_batch(&h)
    }

    /// One batched slow step: feed every sample's previous full frame `[1 + num_codebooks]`
    /// (`frames[i]`) at physical position `pos`, advancing the shared cache by one. The new
    /// token attends all of its sample's real past (left-pad keys masked out per sample).
    /// Returns `(logits [N, vocab], hidden [N, 1, fast_dim])`.
    pub fn slow_step_batch(&mut self, frames: &[Vec<u32>], pos: usize) -> Result<(Tensor, Tensor)> {
        debug_assert_eq!(
            pos,
            self.cache.len(),
            "slow_step_batch pos must equal the current cache length"
        );
        let n = frames.len();
        debug_assert_eq!(n, self.pad_lens.len(), "batch size changed mid-utterance");

        // Embed each one-column frame with the single-sample MCF embed, then stack → [N,1,dim].
        let n_full = 1 + self.cfg.codec.num_codebooks;
        let mut rows: Vec<Tensor> = Vec::with_capacity(n);
        for f in frames {
            debug_assert_eq!(f.len(), n_full, "frame must be [1 + num_codebooks]");
            let inp = Tensor::from_vec(f.clone(), (n_full, 1), &self.w.dev)?;
            rows.push(self.embed(&inp)?); // [1, 1, dim]
        }
        let embeds = Tensor::cat(&rows, 0)?; // [N, 1, dim]

        // The new token's RoPE position for sample i is `pos - pad_i`.
        let pos_ids: Vec<u32> = self
            .pad_lens
            .iter()
            .map(|&ps| (pos - ps) as u32)
            .collect();
        let (cos, sin) = self.gather_rope(&pos_ids, n, 1)?;

        // Mask the new query against the full cache (length `total = pos + 1`): real keys
        // are j ≥ pad_i; the new token itself (j == pos) is always real. Causal is implicit
        // (the query is the newest position).
        let total = pos + 1;
        let mut data = vec![0f32; n * total];
        for (s, &ps) in self.pad_lens.iter().enumerate() {
            for j in 0..ps.min(total) {
                data[s * total + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(data, (n, total), &self.w.dev)?
            .to_dtype(self.w.dt)?
            .reshape((n, 1, 1, total))?;

        let h = self.run_layers_batch(&embeds, &cos, &sin, Some(&mask))?;
        self.head_batch(&h)
    }
}
