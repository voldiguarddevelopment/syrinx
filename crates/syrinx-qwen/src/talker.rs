//! The **talker**: Qwen3-TTS's semantic LM.
//!
//! A standard Qwen3 decoder stack (`talker.model.layers.*`) with three heads around it:
//!
//! * `talker.model.text_embedding` — `[text_vocab, text_embed_dim]`. Text enters at the
//!   *wider* embedding width and is narrowed to the model width by `text_projection`
//!   (a 2-layer MLP with biases), so the table is NOT at `hidden_size`.
//! * `talker.model.codec_embedding` — `[codec_vocab, hidden]`. Code group 0 fed back in.
//! * `talker.codec_head` — `[codec_vocab, hidden]`. Predicts code group 0 for the frame.
//!
//! Groups `1..num_code_groups` come from the code predictor, which consumes this stack's
//! hidden state; see [`crate::code_predictor`].

use candle_core::{DType, Device, Result, Tensor};

use crate::config::Qwen3TtsConfig;
use crate::nn::{attention, causal_mask_at, precompute_rope, rms_norm_w, swiglu, AttnShape, KvCache, Weights};

/// The loaded talker backbone plus its RoPE tables and KV cache.
pub struct Talker {
    pub w: Weights,
    pub cfg: Qwen3TtsConfig,
    cos: Tensor,
    sin: Tensor,
    cache: KvCache,
}

impl Talker {
    /// Wrap a loaded weight bag. RoPE is precomputed once for the whole context.
    pub fn new(w: Weights, cfg: Qwen3TtsConfig) -> Result<Self> {
        let t = &cfg.talker;
        // Cap the precompute: `max_position_embeddings` is 32768 and the table is
        // [pos, head_dim] in the compute dtype, which is cheap, but there is no reason
        // to build more rows than the model can attend over.
        let cap = t.max_position_embeddings.min(32_768);
        let (cos, sin) = precompute_rope(cap, t.head_dim, t.rope_theta, &w.dev, w.dt)?;
        let cache = KvCache::new(t.num_hidden_layers);
        Ok(Self { w, cfg, cos, sin, cache })
    }

    pub fn device(&self) -> Device {
        self.w.dev.clone()
    }

    pub fn dtype(&self) -> DType {
        self.w.dt
    }

    /// Drop the KV cache for a fresh utterance.
    pub fn reset(&mut self) {
        self.cache = KvCache::new(self.cfg.talker.num_hidden_layers);
    }

    /// Current cache length == the absolute position of the next token.
    pub fn pos(&self) -> usize {
        self.cache.len()
    }

    /// Embed text ids: gather at `text_embed_dim`, then project down to the model width.
    ///
    /// `text_projection` is a `Qwen3TTSTalkerResizeMLP`, whose forward is
    /// `linear_fc2(act_fn(linear_fc1(x)))` — both linears biased, with the talker
    /// config's `hidden_act` (**silu**) BETWEEN them.
    ///
    /// An earlier revision of this function omitted the activation and said in this very
    /// comment that the reference applied none. It does. Dropping it turns the projection
    /// into a composition of two linear maps — i.e. a plain linear map — which inflated
    /// every projected text embedding (norms ran ~1.3-1.9x the reference's) and left the
    /// talker prone to repeating the target text, most visibly whenever an `instruct`
    /// block widened the prompt. Confirmed by diffing this tensor against the reference's
    /// `inputs_embeds`, step by step, on the same token ids.
    pub fn embed_text(&self, ids: &[u32]) -> Result<Tensor> {
        let e = self.w.embedding("talker.model.text_embedding.weight", ids)?;
        let e = e.unsqueeze(0)?; // [1, t, text_embed_dim]
        let h = self.w.linear(
            &e,
            "talker.text_projection.linear_fc1.weight",
            Some("talker.text_projection.linear_fc1.bias"),
        )?;
        let h = candle_nn::ops::silu(&h)?;
        self.w.linear(
            &h,
            "talker.text_projection.linear_fc2.weight",
            Some("talker.text_projection.linear_fc2.bias"),
        )
    }

    /// Embed code-group-0 ids at the model width.
    pub fn embed_codec(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .w
            .embedding("talker.model.codec_embedding.weight", ids)?
            .unsqueeze(0)?)
    }

    /// Run the decoder stack over `x` `[1, t, hidden]`, advancing the KV cache.
    /// Returns the hidden states `[1, t, hidden]` after the final norm.
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let t = &self.cfg.talker;
        let shape = AttnShape {
            n_head: t.num_attention_heads,
            n_kv: t.num_key_value_heads,
            head_dim: t.head_dim,
            eps: t.rms_norm_eps,
        };
        let t_new = x.dim(1)?;
        let offset = self.cache.len();
        // A single new token sees the whole cache, so no mask is needed there.
        let mask = if t_new == 1 {
            None
        } else {
            Some(causal_mask_at(offset, t_new, &self.w.dev, self.w.dt)?)
        };

        let mut h = x.clone();
        for l in 0..t.num_hidden_layers {
            let p = format!("talker.model.layers.{l}");
            let hn = self.w.rms_norm(&h, &format!("{p}.input_layernorm.weight"), t.rms_norm_eps)?;
            let a = attention(
                &self.w,
                &format!("{p}.self_attn"),
                &hn,
                &self.cos,
                &self.sin,
                mask.as_ref(),
                shape,
                &mut self.cache,
                l,
            )?;
            h = (h + a)?;
            let hn = self
                .w
                .rms_norm(&h, &format!("{p}.post_attention_layernorm.weight"), t.rms_norm_eps)?;
            let f = swiglu(&self.w, &format!("{p}.mlp"), &hn)?;
            h = (h + f)?;
        }
        self.cache.advance(t_new);
        rms_norm_w(&h, &self.w.g("talker.model.norm.weight")?, t.rms_norm_eps)
    }

    /// Logits over the codec vocabulary for code group 0, from the LAST position only.
    ///
    /// Narrowing before the head matters: the head is `[codec_vocab, hidden]` and
    /// generation only ever needs the final row, so projecting the whole block would be
    /// `t x` the work for one useful row.
    pub fn codec_logits(&self, h: &Tensor) -> Result<Tensor> {
        let t = h.dim(1)?;
        let last = h.narrow(1, t - 1, 1)?;
        self.w
            .linear(&last, "talker.codec_head.weight", None)?
            .to_dtype(DType::F32)
    }
}
