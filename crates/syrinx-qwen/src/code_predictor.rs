//! The **code predictor**: Qwen3-TTS's per-frame residual head.
//!
//! For each frame the talker emits code group 0; this 5-layer stack emits the remaining
//! `num_code_groups - 1` groups autoregressively *within* the frame. It is the direct
//! analogue of Fish's fast AR, with two Qwen-specific wrinkles:
//!
//! * **Per-group tables.** Unlike Fish's single shared embedding, there is one
//!   `codec_embedding.{i}` and one `lm_head.{i}` per residual group, so group identity
//!   is carried by *which* table is used rather than by a position encoding.
//! * **The width bridge.** `codec_embedding.{i}` is `[cp_vocab, TALKER hidden]`, not
//!   `[cp_vocab, cp hidden]`. When the two stacks differ in width (the 1.7B: talker
//!   2048, predictor 1024) the checkpoint ships
//!   `talker.code_predictor.small_to_mtp_projection` to narrow the sum; the 0.6B is
//!   1024/1024 and ships no such tensor. Presence is derived from the widths, so one
//!   implementation is correct for both sizes.

use candle_core::{DType, Result, Tensor};

use crate::config::Qwen3TtsConfig;
use crate::nn::{attention, causal_mask_at, precompute_rope, rms_norm_w, swiglu, AttnShape, KvCache, Weights};

/// The loaded code-predictor stack.
pub struct CodePredictor {
    pub w: Weights,
    pub cfg: Qwen3TtsConfig,
    cos: Tensor,
    sin: Tensor,
    cache: KvCache,
    /// True when the checkpoint carries `small_to_mtp_projection`.
    has_bridge: bool,
}

const PREFIX: &str = "talker.code_predictor";

impl CodePredictor {
    pub fn new(w: Weights, cfg: Qwen3TtsConfig) -> Result<Self> {
        let cp = &cfg.code_predictor;
        // The predictor only ever attends within one frame: at most `num_code_groups`
        // positions, so the RoPE table is tiny.
        let (cos, sin) = precompute_rope(
            cfg.num_code_groups.max(2),
            cp.head_dim,
            cp.rope_theta,
            &w.dev,
            w.dt,
        )?;
        let has_bridge = w.has(&format!("{PREFIX}.small_to_mtp_projection.weight"));
        // Cross-check the checkpoint against the config rather than trusting either
        // alone: differing widths must come with a bridge, equal widths must not.
        let widths_differ = cfg.talker.hidden_size != cp.hidden_size;
        if widths_differ != has_bridge {
            return Err(candle_core::Error::Msg(format!(
                "code predictor width bridge mismatch: talker hidden {} vs predictor {} \
                 (differ = {widths_differ}) but small_to_mtp_projection present = {has_bridge}",
                cfg.talker.hidden_size, cp.hidden_size
            )));
        }
        let cache = KvCache::new(cp.num_hidden_layers);
        Ok(Self { w, cfg, cos, sin, cache, has_bridge })
    }

    /// Reset between frames — the predictor's context is one frame, not the utterance.
    pub fn reset(&mut self) {
        self.cache = KvCache::new(self.cfg.code_predictor.num_hidden_layers);
    }

    /// Number of residual groups this stack predicts (`num_code_groups - 1`).
    pub fn n_groups(&self) -> usize {
        self.cfg.code_predictor_heads()
    }

    /// Embed residual group `i`'s code at the talker width.
    pub fn embed_group(&self, i: usize, code: u32) -> Result<Tensor> {
        Ok(self
            .w
            .embedding(&format!("{PREFIX}.model.codec_embedding.{i}.weight"), &[code])?
            .unsqueeze(0)?)
    }

    /// Narrow a talker-width vector to the predictor width, if this checkpoint needs it.
    pub fn bridge(&self, x: &Tensor) -> Result<Tensor> {
        if !self.has_bridge {
            return Ok(x.clone());
        }
        self.w.linear(
            x,
            &format!("{PREFIX}.small_to_mtp_projection.weight"),
            Some(&format!("{PREFIX}.small_to_mtp_projection.bias")),
        )
    }

    /// One step of the predictor over `x` `[1, t, cp_hidden]`, advancing its cache.
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let cp = &self.cfg.code_predictor;
        let shape = AttnShape {
            n_head: cp.num_attention_heads,
            n_kv: cp.num_key_value_heads,
            head_dim: cp.head_dim,
            eps: cp.rms_norm_eps,
        };
        let t_new = x.dim(1)?;
        let offset = self.cache.len();
        let mask = if t_new == 1 {
            None
        } else {
            Some(causal_mask_at(offset, t_new, &self.w.dev, self.w.dt)?)
        };

        let mut h = x.clone();
        for l in 0..cp.num_hidden_layers {
            let p = format!("{PREFIX}.model.layers.{l}");
            let hn = self.w.rms_norm(&h, &format!("{p}.input_layernorm.weight"), cp.rms_norm_eps)?;
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
                .rms_norm(&h, &format!("{p}.post_attention_layernorm.weight"), cp.rms_norm_eps)?;
            let f = swiglu(&self.w, &format!("{p}.mlp"), &hn)?;
            h = (h + f)?;
        }
        self.cache.advance(t_new);
        rms_norm_w(&h, &self.w.g(&format!("{PREFIX}.model.norm.weight"))?, cp.rms_norm_eps)
    }

    /// Logits for residual group `i` from the last position.
    pub fn group_logits(&self, i: usize, h: &Tensor) -> Result<Tensor> {
        let t = h.dim(1)?;
        let last = h.narrow(1, t - 1, 1)?;
        self.w
            .linear(&last, &format!("{PREFIX}.lm_head.{i}.weight"), None)?
            .to_dtype(DType::F32)
    }
}
