//! The Qwen3-TTS 12 Hz tokenizer's **decode** stack: RVQ latent -> 24 kHz waveform.
//!
//! Everything here is transcribed from the shipped reference,
//! `qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2`
//! (`Qwen3TTSTokenizerV2Decoder.forward`), not inferred from shapes. The pipeline is
//!
//! ```text
//!   [1, codebook_dim=512, T]                       <- crate::codec::rvq::Rvq::decode
//!     pre_conv          causal Conv1d 512 -> 1024, k3          length preserved
//!     (transpose to [1, T, 1024])
//!     pre_transformer   input_proj 1024 -> 512
//!                       8 x { RMSNorm -> sliding-window causal attn -> LayerScale +res
//!                             RMSNorm -> SwiGLU MLP              -> LayerScale +res }
//!                       RMSNorm -> output_proj 512 -> 1024
//!     (transpose to [1, 1024, T])
//!     upsample.{0,1}    causal ConvTranspose1d k=r stride=r (r = 2, 2) + ConvNeXt block
//!     decoder.0         causal Conv1d 1024 -> 1536, k7
//!     decoder.{1..4}    SnakeBeta -> causal ConvTranspose1d k=2r stride=r
//!                       -> 3 residual units (dilations 1, 3, 9)
//!                       r = 8, 5, 4, 3; channels 1536 -> 768 -> 384 -> 192 -> 96
//!     decoder.5         SnakeBeta(96)
//!     decoder.6         causal Conv1d 96 -> 1, k7
//!     clamp(-1, 1)
//! ```
//!
//! Total upsampling is `2*2 * 8*5*4*3 = 1920` = `decode_upsample_rate`, so `T` frames
//! become exactly `T * 1920` samples — 12.5 Hz frames at 24 kHz.
//!
//! ## Semantics that are easy to get wrong, and where each was confirmed
//!
//! * **Snake carries BOTH `alpha` and `beta`, and both are exponentiated.** The
//!   reference `SnakeBeta.forward` is
//!   `x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha))^2`. That is *not* the
//!   `syrinx-fish` `Snake1d` (`x + (alpha + 1e-9)^-1 * sin(alpha*x)^2`, no `exp`, no
//!   `beta`). The checkpoint stores `alpha`/`beta` as the raw pre-`exp` parameters —
//!   both are initialised to zeros, i.e. `exp(.) == 1` — so feeding them in directly
//!   would apply a wrong, per-channel gain that still decodes to audio-shaped output.
//! * **`pre_transformer` is not the talker block.** `self_attn.q_norm` / `k_norm` are
//!   `nn.Identity` (no per-head RMSNorm before RoPE, unlike [`crate::nn::attention`]),
//!   `attention_bias` is false, and each residual branch is multiplied by a learnt
//!   per-channel `LayerScale` (`self_attn_layer_scale.scale`, `mlp_layer_scale.scale`)
//!   **after** the sublayer and **before** the add. The checkpoint carries no
//!   `q_norm`/`k_norm` tensors, which corroborates the `Identity`.
//! * **Attention is causal AND sliding-window.** `Qwen3TTSTokenizerV2DecoderConfig
//!   .layer_types` returns `["sliding_attention"] * num_hidden_layers` with
//!   `sliding_window = 72`, and transformers 4.57.3's
//!   `sliding_window_causal_mask_function` is `and_masks(kv_idx > q_idx - window,
//!   kv_idx <= q_idx)`. So a query attends the **72 positions** `q-71 ..= q` — a closed
//!   window that includes itself. Dropping the window (plain causal) or making it
//!   `q-72 ..= q` is silent corruption, and it also changes the chunking bound below.
//! * **Causality of the convs.** `Qwen3TTSTokenizerV2CausalConvNet` left-pads by
//!   `kernel_eff - stride` and right-pads by `_get_extra_padding_for_conv1d`. Every conv
//!   in the decode path has `stride == 1`, where that alignment padding is *always* zero:
//!   `n_frames = (L - ke + (ke-1))/1 + 1 = L`, so `ideal = (L-1) + (ke - (ke-1)) = L`.
//!   Hence a stride-1 causal conv is exactly "left-pad `(k-1)*dilation`, no right pad",
//!   and is length-preserving. `Qwen3TTSTokenizerV2CausalTransConvNet` runs a plain
//!   `ConvTranspose1d(k, stride)` and then trims `k - stride` from the **right only**
//!   (`left_pad = 0`), giving exactly `L * stride` outputs and no right dependency.
//! * **ConvNeXt is V1, not V2.** `Qwen3TTSTokenizerV2ConvNeXtBlock` has no GRN (unlike
//!   the `syrinx-fish` s2 downsampler): depthwise k7 -> LayerNorm(eps 1e-6) -> Linear ->
//!   exact-erf `nn.GELU()` -> Linear -> `gamma` -> residual.
//!
//! ## Chunking: two stages, two different bounds
//!
//! candle materialises an `L * C_in * k` `im2col` buffer per `conv1d`. This stack runs
//! at up to 1920x the frame rate, so the final k7 conv alone costs
//! `1920 * 96 * 7 = 1.29 M` elements **per frame** (5.2 MiB in f32). A one-shot decode of
//! a 60 s utterance would allocate ~3.8 GiB in that one op. The stack is strictly causal,
//! so it can be split over frames with a left context — but the *right* context to use is
//! not one number, because the transformer's dependency is ~60x the conv stack's:
//!
//! * **Pre stage** (`pre_conv` + `pre_transformer`, at the 12.5 Hz frame rate):
//!   `(3 - 1) + 8 * (72 - 1) = 570` frames. Cheap per frame; chunked only to bound the
//!   dense `[1, 16, T, T]` attention matrix on very long inputs.
//! * **Wave stage** (`upsample` + `decoder`, up to 1920x the frame rate): **10 frames** —
//!   see [`Decoder::wave_left_ctx_frames`] for the derivation. This is the stage whose
//!   memory actually needs bounding, and its context is tiny.
//!
//! Splitting there is what makes the chunked path both exact and cheap. Note that the
//! reference's own `chunked_decode(chunk_size=300, left_context_size=25)` chunks the
//! *whole* decoder with 25 frames of context — far below the 570 its own sliding-window
//! transformer needs — so the reference chunked path is an approximation of its own
//! one-shot path. This port's is not.
//!
//! ## What has actually been verified, and what has not
//!
//! * **Tensor names, shapes and dtypes** were read from
//!   `~/models/Qwen3-TTS-Tokenizer-12Hz/model.safetensors` (271 `decoder.*` tensors, all
//!   f32): `pre_conv.conv.weight [1024, 512, 3]`, `upsample.{i}.0.conv.weight
//!   [1024, 1024, 2]` (ConvTranspose1d, so `[c_in, c_out, k]`), `upsample.{i}.1.dwconv
//!   .conv.weight [1024, 1, 7]` (depthwise), `decoder.0.conv.weight [1536, 1024, 7]`,
//!   the four blocks' `block.1.conv.weight [1536, 768, 16] / [768, 384, 10] /
//!   [384, 192, 8] / [192, 96, 6]`, and `decoder.6.conv.weight [1, 96, 7]`. Every Snake
//!   carries both `.alpha` and `.beta`; no `q_norm`/`k_norm` exists anywhere.
//! * **Numerics** were checked against the PyTorch reference itself, on CPU: the same
//!   model — same geometry, same weights, rebuilt on both sides from a name-seeded
//!   generator — agrees stage for stage to f32 rounding (max relative error 2e-7 across
//!   `pre_conv`, `input_proj`, the transformer, both upsample stages and the full
//!   waveform, at both T = 6 and T = 20). `matches_the_pytorch_reference` freezes that
//!   as a golden and is killed by a plain-causal mask, by dividing Snake by `exp(alpha)`
//!   instead of `exp(beta)`, and by trimming the transposed conv on the wrong end.
//! * **NOT verified**, and marked as such: anything that needs the real 682 MB
//!   checkpoint on a GPU box — that the published weights actually load into these
//!   names through the crate's loader, end-to-end audio quality, and the bf16/CUDA path
//!   (only the f32 CPU path has been exercised). The memory figures quoted above are
//!   derived from candle's `im2col` sizing, not measured on a card.
//!
//! RoPE lets the pre stage be chunked at all: attention scores depend only on the
//! relative offset (`(R_i q) . (R_j k) = q^T R_{j-i} k`) and values are never rotated, so
//! feeding a chunk at positions `0..len` gives the same result as at its absolute
//! positions. Each chunk therefore rebuilds its cos/sin from zero.

use candle_core::{DType, Device, Result, Tensor, D};

use crate::nn::{apply_rope, precompute_rope, repeat_kv, rms_norm_w, swiglu, Weights};

/// `pre_conv` kernel (`Qwen3TTSTokenizerV2Decoder.__init__`: `kernel_size=3`).
const PRE_CONV_KERNEL: usize = 3;
/// `decoder.0` and `decoder.{last}` kernel, and the residual units' `conv1`.
const K7: usize = 7;
/// `Qwen3TTSTokenizerV2ConvNeXtBlock.dwconv` kernel.
const CONVNEXT_KERNEL: usize = 7;
/// Dilations of the three residual units in every `DecoderBlock` (`for dilation in
/// (1, 3, 9)`).
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];
/// `SnakeBeta.no_div_by_zero`.
const SNAKE_EPS: f64 = 1e-9;
/// `Qwen3TTSTokenizerV2ConvNeXtBlock.norm = nn.LayerNorm(dim, eps=1e-6)`.
const CONVNEXT_LN_EPS: f64 = 1e-6;

/// Default wave-stage chunk length in frames (5.12 s at 12.5 Hz).
///
/// The wave stage costs roughly 9 MiB of f32 working set per in-flight frame (the k7
/// convs at 1920x96, 640x192 and 160x384 dominate: `L * C_in * k` is 5.2 + 3.4 + 1.7
/// MiB/frame of `im2col` on top of ~1.5 MiB/frame of activations), so a 64-frame chunk
/// caps the spike near 700 MiB **regardless of utterance length**, for a
/// `(64 + 10) / 64 = 16 %` recompute overhead. Override with
/// `SYRINX_QWEN_CODEC_CHUNK_FRAMES`; `0` selects the one-shot path.
const DEFAULT_WAVE_CHUNK_FRAMES: usize = 64;

/// Default pre-stage chunk length in frames (82 s at 12.5 Hz).
///
/// The pre stage's only super-linear term is the dense attention matrix,
/// `[1, heads, T, T]`; at 16 heads in f32 that is `64 * T^2` bytes, so 1024 frames caps
/// it near 160 MiB. Recompute overhead is `(1024 + 570) / 1024 = 56 %` of a stage that is
/// a rounding error next to the wave stage. Override with
/// `SYRINX_QWEN_CODEC_PRE_CHUNK_FRAMES`; `0` selects the one-shot path.
const DEFAULT_PRE_CHUNK_FRAMES: usize = 1024;

/// Resolve the wave-stage chunk length from `SYRINX_QWEN_CODEC_CHUNK_FRAMES`.
pub fn wave_chunk_frames_env() -> usize {
    std::env::var("SYRINX_QWEN_CODEC_CHUNK_FRAMES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_WAVE_CHUNK_FRAMES)
}

/// Resolve the pre-stage chunk length from `SYRINX_QWEN_CODEC_PRE_CHUNK_FRAMES`.
pub fn pre_chunk_frames_env() -> usize {
    std::env::var("SYRINX_QWEN_CODEC_PRE_CHUNK_FRAMES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_PRE_CHUNK_FRAMES)
}

/// The decode-side geometry, read from the tokenizer checkpoint's
/// `config.json -> decoder_config`.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderConfig {
    /// RVQ output width == `pre_conv` in-channels (512).
    pub codebook_dim: usize,
    /// Width the transformer reads and writes outside its own layers (1024).
    pub latent_dim: usize,
    /// Transformer layer width (512).
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Closed attention window, in frames, including the query itself (72).
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Channels entering `decoder.1` (1536); halves per block.
    pub decoder_dim: usize,
    /// Strides of the four `DecoderBlock`s (`[8, 5, 4, 3]`).
    pub upsample_rates: Vec<usize>,
    /// Strides of the two `upsample` stages (`[2, 2]`).
    pub upsampling_ratios: Vec<usize>,
}

fn u(v: &serde_json::Value, k: &str) -> Option<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize)
}

fn uvec(v: &serde_json::Value, k: &str) -> Option<Vec<usize>> {
    v.get(k)?
        .as_array()?
        .iter()
        .map(|x| x.as_u64().map(|n| n as usize))
        .collect()
}

impl DecoderConfig {
    /// Parse the published `Qwen3-TTS-Tokenizer-12Hz/config.json`.
    pub fn from_json(json: &str) -> std::result::Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("parse config.json: {e}"))?;
        let d = v
            .get("decoder_config")
            .ok_or("config.json: no `decoder_config`")?;
        let need = |k: &str| u(d, k).ok_or_else(|| format!("decoder_config: missing `{k}`"));
        let hidden_size = need("hidden_size")?;
        let num_attention_heads = need("num_attention_heads")?;
        Ok(Self {
            codebook_dim: need("codebook_dim")?,
            latent_dim: need("latent_dim")?,
            hidden_size,
            intermediate_size: need("intermediate_size")?,
            num_hidden_layers: need("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: need("num_key_value_heads")?,
            // `Qwen3TTSTokenizerV2DecoderAttention.__init__` falls back to
            // `hidden_size // num_attention_heads`; the published file sets 64 while
            // that fallback would give 32, so the explicit field is load-bearing.
            head_dim: u(d, "head_dim").unwrap_or(hidden_size / num_attention_heads),
            sliding_window: need("sliding_window")?,
            rope_theta: d
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .ok_or("decoder_config: missing `rope_theta`")?,
            rms_norm_eps: d
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .ok_or("decoder_config: missing `rms_norm_eps`")?,
            decoder_dim: need("decoder_dim")?,
            upsample_rates: uvec(d, "upsample_rates")
                .ok_or("decoder_config: missing `upsample_rates`")?,
            upsampling_ratios: uvec(d, "upsampling_ratios")
                .ok_or("decoder_config: missing `upsampling_ratios`")?,
        })
    }

    /// Samples emitted per code frame: `prod(upsample_rates) * prod(upsampling_ratios)`
    /// — `Qwen3TTSTokenizerV2Decoder.total_upsample`, and equal to the config's
    /// `decode_upsample_rate` (1920).
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates.iter().product::<usize>()
            * self.upsampling_ratios.iter().product::<usize>()
    }
}

/// The loaded decode stack. Holds only geometry and a weight prefix; the weight bag is
/// passed in per call, matching [`crate::codec::rvq::Rvq`].
pub struct Decoder {
    prefix: String,
    cfg: DecoderConfig,
}

impl Decoder {
    /// `prefix` is the checkpoint path of `Qwen3TTSTokenizerV2Model.decoder`, i.e.
    /// `"decoder"` for the published tokenizer.
    pub fn new(prefix: &str, cfg: DecoderConfig) -> Self {
        Self { prefix: prefix.to_string(), cfg }
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    // --- receptive fields -----------------------------------------------------

    /// Left context, in frames, that [`Self::pre_stage`] needs for an exact result.
    ///
    /// `pre_conv` is a stride-1 causal k3 conv, reaching back `3 - 1 = 2` frames. Each
    /// transformer layer's sliding window admits `q-71 ..= q`, so it reaches back
    /// `sliding_window - 1` frames, and 8 stacked layers compose to `8 * 71`. Total
    /// `2 + 568 = 570` for the published geometry.
    pub fn pre_left_ctx_frames(&self) -> usize {
        (PRE_CONV_KERNEL - 1) + self.cfg.num_hidden_layers * (self.cfg.sliding_window - 1)
    }

    /// Left context, in frames, that [`Self::wave_stage`] needs for an exact result.
    ///
    /// Accumulated in units of *output samples* (1920 per frame), walking the stack
    /// forward and tracking `u`, the samples-per-frame rate at each point:
    ///
    /// * a stride-1 causal conv with effective kernel `ke` reaches back `ke - 1` inputs,
    ///   i.e. `(ke - 1) * (total / u)` output samples;
    /// * a `ConvTranspose1d(k, stride=s)` output `n` reads inputs
    ///   `ceil((n-k+1)/s) ..= floor(n/s)`, a span of at most `floor((k-1)/s)` extra
    ///   inputs to the left — and no inputs to the right of `floor(n/s)`, which is what
    ///   makes the whole stack causal.
    ///
    /// For the published geometry the terms are
    ///
    /// ```text
    ///   upsample.0  transconv k2 s2   0            u: 1 -> 2
    ///   upsample.0  dwconv    k7      6 * 960 = 5760
    ///   upsample.1  transconv k2 s2   0            u: 2 -> 4
    ///   upsample.1  dwconv    k7      6 * 480 = 2880
    ///   decoder.0   conv      k7      6 * 480 = 2880
    ///   decoder.1   transconv k16 s8  1 * 480 =  480   u: 4 -> 32
    ///               residual  k7 d{1,3,9}  6*13 * 60 = 4680
    ///   decoder.2   transconv k10 s5  1 *  60 =   60   u: 32 -> 160
    ///               residual                 78 * 12 =  936
    ///   decoder.3   transconv k8  s4  1 *  12 =   12   u: 160 -> 640
    ///               residual                 78 *  3 =  234
    ///   decoder.4   transconv k6  s3  1 *   3 =    3   u: 640 -> 1920
    ///               residual                 78 *  1 =   78
    ///   decoder.6   conv      k7      6 *   1 =    6
    ///                                        = 18009 samples = 9.379 frames
    /// ```
    ///
    /// so **10** frames, and 9 is genuinely insufficient: output sample `a * 1920` reads
    /// back to sample `a * 1920 - 18009`, which lands inside frame `a - 10`.
    /// `wave_left_ctx_matches_receptive_field` pins that boundary by sweeping it.
    pub fn wave_left_ctx_frames(&self) -> usize {
        let total = self.cfg.total_upsample();
        let mut u = 1usize;
        let mut rf = 0usize;
        for &r in &self.cfg.upsampling_ratios {
            rf += ((r - 1) / r) * (total / u); // ConvTranspose1d(k = r, stride = r)
            u *= r;
            rf += (CONVNEXT_KERNEL - 1) * (total / u);
        }
        rf += (K7 - 1) * (total / u); // decoder.0
        for &r in &self.cfg.upsample_rates {
            rf += ((2 * r - 1) / r) * (total / u); // ConvTranspose1d(k = 2r, stride = r)
            u *= r;
            for d in RESIDUAL_DILATIONS {
                rf += (K7 - 1) * d * (total / u); // conv1; conv2 is k1 and adds nothing
            }
        }
        rf += (K7 - 1) * (total / u); // decoder.{last}
        rf.div_ceil(total)
    }

    // --- causal conv primitives ----------------------------------------------

    /// `Qwen3TTSTokenizerV2CausalConvNet.forward` for `stride == 1`: left-pad
    /// `(kernel - 1) * dilation` zeros, then an unpadded conv. Length-preserving.
    ///
    /// Every conv in the decode stack has `stride == 1`, and the reference's
    /// `_get_extra_padding_for_conv1d` is provably `0` there (see the module docs), so
    /// there is no right padding to model.
    fn causal_conv1d(
        &self,
        w: &Weights,
        x: &Tensor,
        conv: &str,
        kernel: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        let weight = w.g(&format!("{conv}.weight"))?;
        let bias = w.g(&format!("{conv}.bias"))?;
        let pad = (kernel - 1) * dilation;
        let y = x
            .pad_with_zeros(D::Minus1, pad, 0)?
            .conv1d(&weight, 0, 1, dilation, groups)?;
        y.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)
    }

    /// `Qwen3TTSTokenizerV2CausalTransConvNet.forward`: `ConvTranspose1d(kernel, stride)`
    /// then drop `kernel - stride` from the right (`left_pad` is 0). Yields exactly
    /// `L * stride` samples and introduces no right dependency.
    fn causal_transpose1d(
        &self,
        w: &Weights,
        x: &Tensor,
        conv: &str,
        kernel: usize,
        stride: usize,
    ) -> Result<Tensor> {
        let weight = w.g(&format!("{conv}.weight"))?; // [c_in, c_out, k]
        let bias = w.g(&format!("{conv}.bias"))?;
        let y = x.conv_transpose1d(&weight, 0, 0, stride, 1, 1)?;
        let y = y.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)?;
        let right = kernel - stride;
        let len = y.dim(D::Minus1)?;
        y.narrow(D::Minus1, 0, len - right)
    }

    /// `SnakeBeta.forward`: `x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha))^2`,
    /// channel-wise. `prefix` owns `.alpha` and `.beta`.
    fn snake(&self, w: &Weights, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let a = w.g(&format!("{prefix}.alpha"))?;
        let b = w.g(&format!("{prefix}.beta"))?;
        let c = a.elem_count();
        let alpha = a.reshape((1, c, 1))?.exp()?;
        let beta = b.reshape((1, c, 1))?.exp()?;
        let s = x.broadcast_mul(&alpha)?.sin()?.sqr()?;
        let inv = beta.affine(1.0, SNAKE_EPS)?.recip()?;
        x.add(&s.broadcast_mul(&inv)?)
    }

    /// `Qwen3TTSTokenizerV2DecoderDecoderResidualUnit`:
    /// `Snake -> conv1(k7, dilation) -> Snake -> conv2(k1)`, added back to the input.
    fn residual_unit(
        &self,
        w: &Weights,
        x: &Tensor,
        prefix: &str,
        dilation: usize,
    ) -> Result<Tensor> {
        let y = self.snake(w, x, &format!("{prefix}.act1"))?;
        let y = self.causal_conv1d(w, &y, &format!("{prefix}.conv1.conv"), K7, dilation, 1)?;
        let y = self.snake(w, &y, &format!("{prefix}.act2"))?;
        let y = self.causal_conv1d(w, &y, &format!("{prefix}.conv2.conv"), 1, 1, 1)?;
        y.add(x)
    }

    /// `Qwen3TTSTokenizerV2ConvNeXtBlock`: depthwise causal k7 -> channels-last
    /// `LayerNorm(eps 1e-6)` -> `Linear` -> exact-erf GELU -> `Linear` -> `gamma` ->
    /// residual. There is **no** GRN here (this is ConvNeXt V1).
    fn convnext(&self, w: &Weights, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let c = x.dim(1)?;
        let h = self.causal_conv1d(w, x, &format!("{prefix}.dwconv.conv"), CONVNEXT_KERNEL, 1, c)?;
        let h = h.transpose(1, 2)?.contiguous()?; // [B, T, C]
        let h = layer_norm(
            &h,
            &w.g(&format!("{prefix}.norm.weight"))?,
            &w.g(&format!("{prefix}.norm.bias"))?,
            CONVNEXT_LN_EPS,
        )?;
        let h = w.linear(
            &h,
            &format!("{prefix}.pwconv1.weight"),
            Some(&format!("{prefix}.pwconv1.bias")),
        )?;
        let h = h.gelu_erf()?;
        let h = w.linear(
            &h,
            &format!("{prefix}.pwconv2.weight"),
            Some(&format!("{prefix}.pwconv2.bias")),
        )?;
        let h = h.broadcast_mul(&w.g(&format!("{prefix}.gamma"))?)?;
        x.add(&h.transpose(1, 2)?.contiguous()?)
    }

    // --- pre stage: pre_conv + pre_transformer --------------------------------

    /// `Qwen3TTSTokenizerV2DecoderAttention.forward`. No `q_norm`/`k_norm` (both are
    /// `nn.Identity`) and no projection biases (`attention_bias = false`).
    fn self_attn(
        &self,
        w: &Weights,
        prefix: &str,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;

        let proj = |name: &str, heads: usize| -> Result<Tensor> {
            w.linear(x, &format!("{prefix}.{name}.weight"), None)?
                .reshape((b, t, heads, hd))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = apply_rope(&proj("q_proj", nh)?, cos, sin)?;
        let k = apply_rope(&proj("k_proj", nkv)?, cos, sin)?;
        let v = proj("v_proj", nkv)?;

        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = 1f64 / (hd as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(mask)?;
        let att =
            candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        let out = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, nh * hd))?;
        w.linear(&out, &format!("{prefix}.o_proj.weight"), None)
    }

    /// `Qwen3TTSTokenizerV2DecoderTransformerModel.forward` over `[1, T, latent_dim]`.
    fn pre_transformer(&self, w: &Weights, x: &Tensor) -> Result<Tensor> {
        let p = format!("{}.pre_transformer", self.prefix);
        let eps = self.cfg.rms_norm_eps;
        let t = x.dim(1)?;

        let mut h = w.linear(
            x,
            &format!("{p}.input_proj.weight"),
            Some(&format!("{p}.input_proj.bias")),
        )?;
        let (cos, sin) = precompute_rope(t, self.cfg.head_dim, self.cfg.rope_theta, &w.dev, w.dt)?;
        let mask = sliding_causal_mask(t, self.cfg.sliding_window, &w.dev, w.dt)?;

        for l in 0..self.cfg.num_hidden_layers {
            let lp = format!("{p}.layers.{l}");
            let hn = w.rms_norm(&h, &format!("{lp}.input_layernorm.weight"), eps)?;
            let a = self.self_attn(w, &format!("{lp}.self_attn"), &hn, &cos, &sin, &mask)?;
            // LayerScale multiplies the branch, then the residual is added.
            let a = a.broadcast_mul(&w.g(&format!("{lp}.self_attn_layer_scale.scale"))?)?;
            h = (h + a)?;
            let hn = w.rms_norm(&h, &format!("{lp}.post_attention_layernorm.weight"), eps)?;
            let f = swiglu(w, &format!("{lp}.mlp"), &hn)?;
            let f = f.broadcast_mul(&w.g(&format!("{lp}.mlp_layer_scale.scale"))?)?;
            h = (h + f)?;
        }
        let h = rms_norm_w(&h, &w.g(&format!("{p}.norm.weight"))?, eps)?;
        w.linear(
            &h,
            &format!("{p}.output_proj.weight"),
            Some(&format!("{p}.output_proj.bias")),
        )
    }

    /// `pre_conv` then `pre_transformer`: `[1, codebook_dim, T] -> [1, latent_dim, T]`.
    pub fn pre_stage(&self, w: &Weights, z: &Tensor) -> Result<Tensor> {
        let h = self.causal_conv1d(
            w,
            z,
            &format!("{}.pre_conv.conv", self.prefix),
            PRE_CONV_KERNEL,
            1,
            1,
        )?;
        let h = self.pre_transformer(w, &h.transpose(1, 2)?.contiguous()?)?;
        h.transpose(1, 2)?.contiguous()
    }

    // --- wave stage: upsample + decoder ---------------------------------------

    /// `upsample` then `decoder`: `[1, latent_dim, T] -> [1, 1, T * total_upsample]`,
    /// clamped to `[-1, 1]`. Length-exact, which the chunked path depends on.
    pub fn wave_stage(&self, w: &Weights, h: &Tensor) -> Result<Tensor> {
        let p = &self.prefix;
        let mut x = h.clone();
        for (i, &r) in self.cfg.upsampling_ratios.iter().enumerate() {
            x = self.causal_transpose1d(w, &x, &format!("{p}.upsample.{i}.0.conv"), r, r)?;
            x = self.convnext(w, &x, &format!("{p}.upsample.{i}.1"))?;
        }
        x = self.causal_conv1d(w, &x, &format!("{p}.decoder.0.conv"), K7, 1, 1)?;
        for (i, &r) in self.cfg.upsample_rates.iter().enumerate() {
            let bp = format!("{p}.decoder.{}.block", i + 1);
            x = self.snake(w, &x, &format!("{bp}.0"))?;
            x = self.causal_transpose1d(w, &x, &format!("{bp}.1.conv"), 2 * r, r)?;
            for (j, d) in RESIDUAL_DILATIONS.into_iter().enumerate() {
                x = self.residual_unit(w, &x, &format!("{bp}.{}", j + 2), d)?;
            }
        }
        let last = self.cfg.upsample_rates.len() + 1;
        x = self.snake(w, &x, &format!("{p}.decoder.{last}"))?;
        x = self.causal_conv1d(w, &x, &format!("{p}.decoder.{}.conv", last + 1), K7, 1, 1)?;
        x.clamp(-1f64, 1f64)
    }

    // --- entry points ---------------------------------------------------------

    /// Decode an RVQ latent `[1, codebook_dim, T]` to a mono `[T * 1920]` f32 waveform.
    ///
    /// Uses the chunked path with [`pre_chunk_frames_env`] / [`wave_chunk_frames_env`]
    /// so peak memory is bounded independently of the utterance length. Set both env
    /// vars to `0`, or call [`Self::decode_oneshot`], for the reference path.
    pub fn decode(&self, w: &Weights, z: &Tensor) -> Result<Tensor> {
        self.decode_chunked(w, z, pre_chunk_frames_env(), wave_chunk_frames_env())
    }

    /// The **one-shot** decode — the parity reference, matching
    /// `Qwen3TTSTokenizerV2Decoder.forward` after the quantizer. Peak memory grows
    /// linearly (attention: quadratically) with `T`.
    pub fn decode_oneshot(&self, w: &Weights, z: &Tensor) -> Result<Tensor> {
        self.decode_chunked(w, z, 0, 0)
    }

    /// Decode with each stage split into chunks of its own size (`0` == one-shot for
    /// that stage), each using the derived left context for its stage.
    ///
    /// Exact in the mathematical sense: both stages are strictly causal with no right
    /// dependency, so a kept frame sees its entire receptive field and takes the value
    /// the one-shot path computes.
    ///
    /// Bit-for-bit equality is a separate question, and the two stages answer it
    /// differently. The wave stage contracts only over fixed-length conv kernels, so its
    /// chunked output is bit-identical (`wave_left_ctx_matches_receptive_field` asserts
    /// exact equality). The pre stage's attention contracts over the *key axis*, whose
    /// length is the chunk's, so the GEMM groups partial sums differently and the result
    /// agrees to rounding rather than to the last bit — ~1e-8 on the CPU f32 fixture
    /// against ~1e-5 for a context one frame short, i.e. three orders of margin between
    /// float noise and a real chunking bug.
    pub fn decode_chunked(
        &self,
        w: &Weights,
        z: &Tensor,
        pre_chunk: usize,
        wave_chunk: usize,
    ) -> Result<Tensor> {
        let z = z.to_dtype(w.dt)?;
        let h = self.pre_stage_ctx(w, &z, pre_chunk, self.pre_left_ctx_frames())?;
        let wav = self.wave_stage_ctx(w, &h, wave_chunk, self.wave_left_ctx_frames())?;
        let n = wav.dim(D::Minus1)?;
        wav.reshape((n,))?.to_dtype(DType::F32)
    }

    /// [`Self::pre_stage`] over `chunk`-frame pieces with an explicit left context.
    ///
    /// `ctx` is a parameter rather than a constant so `pre_left_ctx_matches_receptive_field`
    /// can sweep it and prove where the dependency actually ends.
    pub fn pre_stage_ctx(
        &self,
        w: &Weights,
        z: &Tensor,
        chunk: usize,
        ctx: usize,
    ) -> Result<Tensor> {
        let t = z.dim(D::Minus1)?;
        if chunk == 0 || t <= chunk {
            return self.pre_stage(w, z);
        }
        let mut parts: Vec<Tensor> = Vec::new();
        let mut a = 0usize;
        while a < t {
            let b = (a + chunk).min(t);
            let lo = a.saturating_sub(ctx);
            let piece = self.pre_stage(w, &z.narrow(D::Minus1, lo, b - lo)?.contiguous()?)?;
            parts.push(piece.narrow(D::Minus1, a - lo, b - a)?.contiguous()?);
            a = b;
        }
        Tensor::cat(&parts, D::Minus1)
    }

    /// [`Self::wave_stage`] over `chunk`-frame pieces with an explicit left context.
    /// Only the `[a, b)` frames of each piece's output are kept, so the seam carries the
    /// same samples the one-shot path produces.
    pub fn wave_stage_ctx(
        &self,
        w: &Weights,
        h: &Tensor,
        chunk: usize,
        ctx: usize,
    ) -> Result<Tensor> {
        let t = h.dim(D::Minus1)?;
        if chunk == 0 || t <= chunk {
            return self.wave_stage(w, h);
        }
        let hop = self.cfg.total_upsample();
        let mut parts: Vec<Tensor> = Vec::new();
        let mut a = 0usize;
        while a < t {
            let b = (a + chunk).min(t);
            let lo = a.saturating_sub(ctx);
            let wav = self.wave_stage(w, &h.narrow(D::Minus1, lo, b - lo)?.contiguous()?)?;
            let n = wav.dim(D::Minus1)?;
            let want = (b - lo) * hop;
            if n != want {
                return Err(candle_core::Error::Msg(format!(
                    "wave stage produced {n} samples for frames [{lo}, {b}), expected \
                     {want} — the stack is not length-exact and chunking is unsound"
                )));
            }
            parts.push(
                wav.narrow(D::Minus1, (a - lo) * hop, (b - a) * hop)?
                    .contiguous()?,
            );
            a = b;
        }
        Tensor::cat(&parts, D::Minus1)
    }
}

// --- free helpers ---------------------------------------------------------------

/// Additive mask for `and_masks(sliding_window_overlay(w), causal)`: query `i` may
/// attend key `j` iff `j <= i` **and** `j > i - w`, i.e. the closed window
/// `i - w + 1 ..= i` of exactly `w` positions.
fn sliding_causal_mask(t: usize, window: usize, dev: &Device, dt: DType) -> Result<Tensor> {
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i || j + window <= i {
                m[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(m, (t, t), dev)?.to_dtype(dt)
}

/// Channels-last `nn.LayerNorm` over the last dim.
//
// PARITY: the mean/variance reduction runs in f32 for bf16-stability; the normalised
// activation is cast back before the affine. Identity on the f32 path.
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xc.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(dt)?;
    xn.broadcast_mul(w)?.broadcast_add(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The published `Qwen3-TTS-Tokenizer-12Hz/config.json`, verbatim in the fields this
    /// parser reads.
    const TOKENIZER_CONFIG: &str = r#"{
      "model_type": "qwen3_tts_tokenizer_12hz",
      "decode_upsample_rate": 1920,
      "decoder_config": {
        "attention_bias": false, "latent_dim": 1024, "codebook_dim": 512,
        "codebook_size": 2048, "decoder_dim": 1536, "hidden_act": "silu",
        "hidden_size": 512, "intermediate_size": 1024, "layer_scale_initial_scale": 0.01,
        "max_position_embeddings": 8000, "head_dim": 64, "num_attention_heads": 16,
        "num_hidden_layers": 8, "num_key_value_heads": 16, "num_quantizers": 16,
        "rms_norm_eps": 1e-05, "rope_theta": 10000, "sliding_window": 72,
        "upsample_rates": [8, 5, 4, 3], "upsampling_ratios": [2, 2]
      }
    }"#;

    fn real_cfg() -> DecoderConfig {
        DecoderConfig::from_json(TOKENIZER_CONFIG).unwrap()
    }

    // --- the synthetic fixture -------------------------------------------------
    //
    // REAL structural geometry — every kernel size, stride, dilation and rate schedule
    // is the checkpoint's (`upsampling_ratios [2,2]`, `upsample_rates [8,5,4,3]`, k3
    // pre_conv, k7 dwconv / decoder convs / residual conv1, k1 residual conv2, k2r
    // transposed convs) — with the channel widths and the layer/window counts shrunk.
    // Widths never enter the chunking arithmetic; the window and layer counts do, and
    // shrinking them keeps the pre-stage sweep cheap while still exercising the same
    // `2 + L*(W-1)` formula. The transformer IS included: a chunking scheme that is
    // wrong only inside attention passes a conv-only fixture.

    const FIX_LATENT: usize = 8;
    const FIX_CODEBOOK: usize = 6;
    const FIX_HIDDEN: usize = 4;
    const FIX_HEADS: usize = 2;
    const FIX_HEAD_DIM: usize = 2;
    const FIX_LAYERS: usize = 2;
    const FIX_WINDOW: usize = 4;
    const FIX_DECODER_DIM: usize = 32;

    fn fixture_cfg() -> DecoderConfig {
        DecoderConfig {
            codebook_dim: FIX_CODEBOOK,
            latent_dim: FIX_LATENT,
            hidden_size: FIX_HIDDEN,
            intermediate_size: 8,
            num_hidden_layers: FIX_LAYERS,
            num_attention_heads: FIX_HEADS,
            num_key_value_heads: FIX_HEADS,
            head_dim: FIX_HEAD_DIM,
            sliding_window: FIX_WINDOW,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            decoder_dim: FIX_DECODER_DIM,
            upsample_rates: vec![8, 5, 4, 3],
            upsampling_ratios: vec![2, 2],
        }
    }

    /// FNV-1a over the tensor name: every tensor's values depend only on its own name
    /// and shape, so the same fixture can be rebuilt in any order — and, more usefully,
    /// rebuilt byte-identically by the Python reference for a cross-check.
    fn fnv1a(s: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Per-tensor scale rule, a pure function of name and shape so it can be mirrored
    /// exactly. Norm weights sit near 1, biases and layer scales near 0, and ordinary
    /// weights are damped by `1/sqrt(fan_in)` so 4 stacked decoder blocks stay well
    /// inside the final `clamp(-1, 1)` — a saturated fixture would make the context
    /// sweeps insensitive, which `fixture_output_is_inside_the_clamp` guards.
    fn fill(name: &str, dims: &[usize]) -> Vec<f32> {
        let n: usize = dims.iter().product();
        let fan_in: usize = if dims.len() > 1 { dims[1..].iter().product() } else { 1 };
        let mut s = fnv1a(name);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / 2147483648.0 * 2.0 - 1.0; // [-1, 1)
            out.push(if name.ends_with(".scale") {
                0.05 + 0.05 * u
            } else if name.ends_with(".gamma") {
                0.1 * u
            } else if name.ends_with(".alpha") || name.ends_with(".beta") {
                0.2 * u
            } else if name.ends_with(".bias") {
                0.05 * u
            } else if name.ends_with("norm.weight") {
                1.0 + 0.1 * u
            } else {
                0.5 / (fan_in as f32).sqrt() * u
            });
        }
        out
    }

    struct Bag {
        map: HashMap<String, Tensor>,
        dev: Device,
    }

    impl Bag {
        fn put(&mut self, name: &str, dims: &[usize]) {
            let v = fill(name, dims);
            let t = Tensor::from_vec(v, dims, &self.dev).unwrap();
            self.map.insert(name.to_string(), t);
        }
    }

    /// Build the full weight bag the fixture geometry implies.
    fn fixture() -> (Weights, Decoder) {
        let cfg = fixture_cfg();
        let mut bag = Bag { map: HashMap::new(), dev: Device::Cpu };
        let p = "decoder";
        let (lat, hid) = (cfg.latent_dim, cfg.hidden_size);

        bag.put(&format!("{p}.pre_conv.conv.weight"), &[lat, cfg.codebook_dim, PRE_CONV_KERNEL]);
        bag.put(&format!("{p}.pre_conv.conv.bias"), &[lat]);

        let tp = format!("{p}.pre_transformer");
        bag.put(&format!("{tp}.input_proj.weight"), &[hid, lat]);
        bag.put(&format!("{tp}.input_proj.bias"), &[hid]);
        bag.put(&format!("{tp}.output_proj.weight"), &[lat, hid]);
        bag.put(&format!("{tp}.output_proj.bias"), &[lat]);
        bag.put(&format!("{tp}.norm.weight"), &[hid]);
        let qd = cfg.num_attention_heads * cfg.head_dim;
        let kd = cfg.num_key_value_heads * cfg.head_dim;
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{tp}.layers.{l}");
            bag.put(&format!("{lp}.input_layernorm.weight"), &[hid]);
            bag.put(&format!("{lp}.post_attention_layernorm.weight"), &[hid]);
            bag.put(&format!("{lp}.self_attn.q_proj.weight"), &[qd, hid]);
            bag.put(&format!("{lp}.self_attn.k_proj.weight"), &[kd, hid]);
            bag.put(&format!("{lp}.self_attn.v_proj.weight"), &[kd, hid]);
            bag.put(&format!("{lp}.self_attn.o_proj.weight"), &[hid, qd]);
            bag.put(&format!("{lp}.self_attn_layer_scale.scale"), &[hid]);
            bag.put(&format!("{lp}.mlp_layer_scale.scale"), &[hid]);
            bag.put(&format!("{lp}.mlp.gate_proj.weight"), &[cfg.intermediate_size, hid]);
            bag.put(&format!("{lp}.mlp.up_proj.weight"), &[cfg.intermediate_size, hid]);
            bag.put(&format!("{lp}.mlp.down_proj.weight"), &[hid, cfg.intermediate_size]);
        }

        for (i, &r) in cfg.upsampling_ratios.iter().enumerate() {
            bag.put(&format!("{p}.upsample.{i}.0.conv.weight"), &[lat, lat, r]);
            bag.put(&format!("{p}.upsample.{i}.0.conv.bias"), &[lat]);
            let cn = format!("{p}.upsample.{i}.1");
            bag.put(&format!("{cn}.dwconv.conv.weight"), &[lat, 1, CONVNEXT_KERNEL]);
            bag.put(&format!("{cn}.dwconv.conv.bias"), &[lat]);
            bag.put(&format!("{cn}.norm.weight"), &[lat]);
            bag.put(&format!("{cn}.norm.bias"), &[lat]);
            bag.put(&format!("{cn}.pwconv1.weight"), &[4 * lat, lat]);
            bag.put(&format!("{cn}.pwconv1.bias"), &[4 * lat]);
            bag.put(&format!("{cn}.pwconv2.weight"), &[lat, 4 * lat]);
            bag.put(&format!("{cn}.pwconv2.bias"), &[lat]);
            bag.put(&format!("{cn}.gamma"), &[lat]);
        }

        bag.put(&format!("{p}.decoder.0.conv.weight"), &[cfg.decoder_dim, lat, K7]);
        bag.put(&format!("{p}.decoder.0.conv.bias"), &[cfg.decoder_dim]);
        for (i, &r) in cfg.upsample_rates.iter().enumerate() {
            let (ci, co) = (cfg.decoder_dim >> i, cfg.decoder_dim >> (i + 1));
            let bp = format!("{p}.decoder.{}.block", i + 1);
            bag.put(&format!("{bp}.0.alpha"), &[ci]);
            bag.put(&format!("{bp}.0.beta"), &[ci]);
            bag.put(&format!("{bp}.1.conv.weight"), &[ci, co, 2 * r]);
            bag.put(&format!("{bp}.1.conv.bias"), &[co]);
            for j in 0..RESIDUAL_DILATIONS.len() {
                let rp = format!("{bp}.{}", j + 2);
                bag.put(&format!("{rp}.act1.alpha"), &[co]);
                bag.put(&format!("{rp}.act1.beta"), &[co]);
                bag.put(&format!("{rp}.act2.alpha"), &[co]);
                bag.put(&format!("{rp}.act2.beta"), &[co]);
                bag.put(&format!("{rp}.conv1.conv.weight"), &[co, co, K7]);
                bag.put(&format!("{rp}.conv1.conv.bias"), &[co]);
                bag.put(&format!("{rp}.conv2.conv.weight"), &[co, co, 1]);
                bag.put(&format!("{rp}.conv2.conv.bias"), &[co]);
            }
        }
        let last = cfg.upsample_rates.len() + 1;
        let out_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
        bag.put(&format!("{p}.decoder.{last}.alpha"), &[out_dim]);
        bag.put(&format!("{p}.decoder.{last}.beta"), &[out_dim]);
        bag.put(&format!("{p}.decoder.{}.conv.weight", last + 1), &[1, out_dim, K7]);
        bag.put(&format!("{p}.decoder.{}.conv.bias", last + 1), &[1]);

        let w = Weights { map: bag.map, dev: Device::Cpu, dt: DType::F32 };
        (w, Decoder::new(p, cfg))
    }

    /// A deterministic `[1, codebook_dim, t]` RVQ-latent stand-in.
    fn latent(t: usize) -> Tensor {
        let v = fill("fixture.latent", &[1, FIX_CODEBOOK, t]);
        Tensor::from_vec(v, (1, FIX_CODEBOOK, t), &Device::Cpu).unwrap()
    }

    fn vec1(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    // --- config ----------------------------------------------------------------

    #[test]
    fn parses_the_published_tokenizer_config() {
        let c = real_cfg();
        assert_eq!(c.codebook_dim, 512);
        assert_eq!(c.latent_dim, 1024);
        assert_eq!(c.hidden_size, 512);
        assert_eq!(c.num_hidden_layers, 8);
        // head_dim is 64 explicitly; the HF fallback hidden/heads would give 32 and
        // silently halve every attention head.
        assert_eq!(c.head_dim, 64);
        assert_ne!(c.head_dim, c.hidden_size / c.num_attention_heads);
        assert_eq!(c.sliding_window, 72);
        assert_eq!(c.decoder_dim, 1536);
        assert_eq!(c.upsample_rates, vec![8, 5, 4, 3]);
        assert_eq!(c.upsampling_ratios, vec![2, 2]);
        // == the config's own `decode_upsample_rate`, i.e. 12.5 Hz frames at 24 kHz.
        assert_eq!(c.total_upsample(), 1920);
    }

    // --- primitives ------------------------------------------------------------

    /// Snake must exponentiate BOTH parameters and divide by `exp(beta)`, not by alpha.
    /// The checkpoint stores the pre-`exp` values, so the two forms differ by a
    /// per-channel gain — audible, but not obviously broken.
    #[test]
    fn snake_beta_exponentiates_alpha_and_beta() {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // alpha = ln(2) -> exp = 2 ; beta = ln(4) -> exp = 4
        map.insert(
            "s.alpha".into(),
            Tensor::from_vec(vec![2f32.ln()], (1,), &dev).unwrap(),
        );
        map.insert(
            "s.beta".into(),
            Tensor::from_vec(vec![4f32.ln()], (1,), &dev).unwrap(),
        );
        let w = Weights { map, dev: dev.clone(), dt: DType::F32 };
        let d = Decoder::new("decoder", fixture_cfg());
        let x = Tensor::from_vec(vec![0.3f32, -0.7], (1, 1, 2), &dev).unwrap();
        let got = vec1(&d.snake(&w, &x, "s").unwrap());
        for (i, &xv) in [0.3f32, -0.7].iter().enumerate() {
            let want = xv + (1.0 / (4.0 + 1e-9)) * (xv * 2.0).sin().powi(2);
            assert!((got[i] - want).abs() < 1e-6, "i={i}: {} vs {want}", got[i]);
            // the Fish-style `x + 1/(alpha+eps) * sin(alpha*x)^2` on the raw (un-exp'd)
            // parameters is a different number
            let fishy = xv + (1.0 / (2f32.ln() + 1e-9)) * (xv * 2f32.ln()).sin().powi(2);
            assert!((got[i] - fishy).abs() > 1e-3, "i={i}: matched the Fish Snake form");
        }
    }

    /// The causal conv must see only the past: shifting a future sample changes nothing
    /// at earlier positions, and the output length equals the input length.
    #[test]
    fn causal_conv_is_left_padded_and_length_exact() {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // 1 channel, k3, weights [1,2,4] so each output names its taps unambiguously
        map.insert(
            "c.weight".into(),
            Tensor::from_vec(vec![1f32, 2., 4.], (1, 1, 3), &dev).unwrap(),
        );
        map.insert("c.bias".into(), Tensor::from_vec(vec![0f32], (1,), &dev).unwrap());
        let w = Weights { map, dev: dev.clone(), dt: DType::F32 };
        let d = Decoder::new("decoder", fixture_cfg());
        let x = Tensor::from_vec(vec![1f32, 0., 0., 0., 0.], (1, 1, 5), &dev).unwrap();
        let y = d.causal_conv1d(&w, &x, "c", 3, 1, 1).unwrap();
        assert_eq!(y.dims(), &[1, 1, 5], "stride-1 causal conv must preserve length");
        // the impulse at t=0 reaches t=0 (tap 4, the last kernel column), t=1 (tap 2)
        // and t=2 (tap 1) — never backwards.
        assert_eq!(vec1(&y), vec![4., 2., 1., 0., 0.]);
        // dilation 2 spreads the same taps to t = 0, 2, 4
        let y = d.causal_conv1d(&w, &x, "c", 3, 2, 1).unwrap();
        assert_eq!(vec1(&y), vec![4., 0., 2., 0., 1.]);
    }

    /// The transposed conv must emit exactly `L * stride` samples, trimming the tail
    /// (never the head) — a left trim would break causality and the chunking bound.
    #[test]
    fn causal_transpose_is_length_exact_and_trims_the_right() {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // [c_in=1, c_out=1, k=4], stride 2 -> right trim of 2
        map.insert(
            "c.weight".into(),
            Tensor::from_vec(vec![1f32, 2., 4., 8.], (1, 1, 4), &dev).unwrap(),
        );
        map.insert("c.bias".into(), Tensor::from_vec(vec![0f32], (1,), &dev).unwrap());
        let w = Weights { map, dev: dev.clone(), dt: DType::F32 };
        let d = Decoder::new("decoder", fixture_cfg());
        let x = Tensor::from_vec(vec![1f32, 0., 0.], (1, 1, 3), &dev).unwrap();
        let y = d.causal_transpose1d(&w, &x, "c", 4, 2).unwrap();
        assert_eq!(y.dims(), &[1, 1, 6], "must be exactly L * stride");
        // the impulse at input 0 writes taps [1,2,4,8] at outputs 0..3; nothing is
        // dropped from the front.
        assert_eq!(vec1(&y), vec![1., 2., 4., 8., 0., 0.]);
    }

    /// The window is closed and `W` wide: `j == i - W + 1` is admitted, `j == i - W` is
    /// not, and `j == i + 1` is not. Off-by-one here silently changes the model AND the
    /// chunking bound.
    #[test]
    fn sliding_window_mask_admits_exactly_w_positions() {
        let m = sliding_causal_mask(6, 3, &Device::Cpu, DType::F32).unwrap();
        let v = vec1(&m);
        let open = |i: usize, j: usize| v[i * 6 + j] == 0.0;
        assert!(open(4, 4), "self must be visible");
        assert!(open(4, 3));
        assert!(open(4, 2), "i - W + 1 must be visible");
        assert!(!open(4, 1), "i - W must be masked");
        assert!(!open(4, 5), "the future must be masked");
        // every row admits exactly min(i + 1, W) keys
        for i in 0..6 {
            let n = (0..6).filter(|&j| open(i, j)).count();
            assert_eq!(n, (i + 1).min(3), "row {i}");
        }
    }

    // --- derived receptive fields ----------------------------------------------

    #[test]
    fn derived_left_contexts_match_the_checkpoint_geometry() {
        let d = Decoder::new("decoder", real_cfg());
        // pre_conv k3 (2 frames) + 8 layers x (72 - 1)
        assert_eq!(d.pre_left_ctx_frames(), 570);
        // 18009 output samples / 1920 -> 10 frames (9.379 rounded up)
        assert_eq!(d.wave_left_ctx_frames(), 10);
        // the reference's own chunked_decode uses 25 frames for the WHOLE decoder,
        // which is below its pre-stage requirement — that path is approximate.
        assert!(d.pre_left_ctx_frames() > 25);
    }

    // --- end-to-end shape / range ----------------------------------------------

    #[test]
    fn oneshot_length_is_frames_times_total_upsample() {
        let (w, d) = fixture();
        for t in [1usize, 5, 17] {
            let y = d.decode_oneshot(&w, &latent(t)).unwrap();
            assert_eq!(y.dims(), &[t * d.config().total_upsample()], "t={t}");
        }
    }

    /// The context sweeps only mean something if the fixture is not pinned against the
    /// output clamp, where every context would agree trivially.
    #[test]
    fn fixture_output_is_inside_the_clamp() {
        let (w, d) = fixture();
        let y = vec1(&d.decode_oneshot(&w, &latent(20)).unwrap());
        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.99, "fixture saturates the clamp (peak {peak}); sweeps are blind");
        assert!(peak > 1e-4, "fixture output is ~zero (peak {peak}); sweeps are blind");
    }

    // --- negative controls: the chunk boundaries ------------------------------

    /// Sweep the wave-stage left context and assert it flips exactly at the derived
    /// bound: every value below it must visibly disagree with the one-shot result, and
    /// the bound itself (and anything above) must agree exactly.
    #[test]
    fn wave_left_ctx_matches_receptive_field() {
        let (w, d) = fixture();
        let bound = d.wave_left_ctx_frames();
        assert_eq!(bound, 10, "the fixture uses the real rate schedule");
        let t = 20usize;
        let h = d.pre_stage(&w, &latent(t)).unwrap();
        let want = vec1(&d.wave_stage(&w, &h).unwrap());
        let at = |ctx: usize| vec1(&d.wave_stage_ctx(&w, &h, 6, ctx).unwrap());
        for ctx in 0..bound {
            assert!(
                worst(&at(ctx), &want) > 0.0,
                "ctx={ctx} is below the {bound}-frame receptive field but reproduced the \
                 one-shot output exactly — the derivation or the chunking is wrong"
            );
        }
        for ctx in [bound, bound + 1, bound + 7] {
            assert_eq!(worst(&at(ctx), &want), 0.0, "ctx={ctx} should be sufficient");
        }
    }

    /// The same sweep for the pre stage, where the dependency runs through the sliding
    /// -window attention rather than through convs. The bound is `2 + L * (W - 1)`; the
    /// fixture's 2 layers and window 4 make it 8.
    #[test]
    fn pre_left_ctx_matches_receptive_field() {
        let (w, d) = fixture();
        let bound = d.pre_left_ctx_frames();
        assert_eq!(bound, (PRE_CONV_KERNEL - 1) + FIX_LAYERS * (FIX_WINDOW - 1));
        let z = latent(20);
        let want = vec1(&d.pre_stage(&w, &z).unwrap());
        let at = |ctx: usize| vec1(&d.pre_stage_ctx(&w, &z, 6, ctx).unwrap());
        // Attention contracts over the key axis, so a sufficient context agrees to
        // GEMM rounding rather than bit-exactly (see `decode_chunked`). The two regimes
        // are three orders apart on this fixture — 9.1e-6 at ctx = bound - 1 against
        // 1.5e-8 at and above the bound — so these thresholds cannot straddle noise.
        for ctx in 0..bound {
            assert!(
                worst(&at(ctx), &want) > 1e-6,
                "ctx={ctx} is below the {bound}-frame receptive field but reproduced the \
                 one-shot output to better than rounding — the window or the layer \
                 composition is wrong"
            );
        }
        for ctx in [bound, bound + 1, bound + 5] {
            assert!(worst(&at(ctx), &want) < 1e-7, "ctx={ctx} should be sufficient");
        }
    }

    /// RoPE is applied to q and k only, so a chunk fed at positions `0..len` scores
    /// identically to the same frames at their absolute positions. If that were false,
    /// no left context would ever make the pre stage exact — which is precisely what
    /// the sweep above would have shown.
    #[test]
    fn default_and_chunked_decode_match_oneshot() {
        let (w, d) = fixture();
        let z = latent(24);
        let want = vec1(&d.decode_oneshot(&w, &z).unwrap());
        // chunk both stages, at sizes that force several seams in each. The pre stage's
        // attention makes this agree to rounding, not bit-exactly (see `decode_chunked`).
        assert!(worst(&vec1(&d.decode_chunked(&w, &z, 5, 7).unwrap()), &want) < 1e-7);
        // `0` on either stage selects that stage's one-shot path; with the pre stage
        // one-shot the whole path is bit-exact again.
        assert_eq!(vec1(&d.decode_chunked(&w, &z, 0, 7).unwrap()), want);
        assert!(worst(&vec1(&d.decode_chunked(&w, &z, 5, 0).unwrap()), &want) < 1e-7);
        // the shipped defaults are far larger than this input, so `decode` is one-shot
        // here; the point is that the public entry point agrees.
        assert_eq!(vec1(&d.decode(&w, &z).unwrap()), want);
    }

    /// A chunk shorter than the whole input must still be rejected if the wave stage
    /// ever stops being length-exact, since the seam arithmetic assumes it.
    #[test]
    fn wave_stage_is_length_exact_per_chunk() {
        let (w, d) = fixture();
        let hop = d.config().total_upsample();
        for t in [1usize, 3, 9] {
            let h = d.pre_stage(&w, &latent(t)).unwrap();
            let y = d.wave_stage(&w, &h).unwrap();
            assert_eq!(y.dims(), &[1, 1, t * hop], "t={t}");
        }
    }

    // --- parity against the PyTorch reference ---------------------------------
    //
    // The two constants below were produced by running the *reference* classes on the
    // same fixture. That is possible without shipping a weight file because `fill` is a
    // pure function of a tensor's name and shape, so the reference can rebuild this
    // exact model from the names alone. The generator, verbatim:
    //
    // ```python
    // # ~/.venvs/qwen/bin/python  — CPU only, no checkpoint needed
    // import numpy as np, torch
    // from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    //     Qwen3TTSTokenizerV2DecoderConfig)
    // from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    //     Qwen3TTSTokenizerV2Decoder)
    // f32 = np.float32
    // def fnv1a(s):
    //     h = 0xcbf29ce484222325
    //     for b in s.encode(): h ^= b; h = (h * 0x100000001b3) & (2**64 - 1)
    //     return h
    // def fill(name, dims):                       # mirrors `fill` below exactly
    //     n = int(np.prod(dims)); fan = int(np.prod(dims[1:])) if len(dims) > 1 else 1
    //     s = fnv1a(name); out = np.empty(n, dtype=np.float32)
    //     for i in range(n):
    //         s = (s * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
    //         u = f32(f32(f32(s >> 33) / f32(2147483648.0)) * f32(2.0)) - f32(1.0)
    //         if   name.endswith(".scale"):  v = f32(f32(0.05) + f32(f32(0.05) * u))
    //         elif name.endswith(".gamma"):  v = f32(f32(0.1) * u)
    //         elif name.endswith((".alpha", ".beta")): v = f32(f32(0.2) * u)
    //         elif name.endswith(".bias"):   v = f32(f32(0.05) * u)
    //         elif name.endswith("norm.weight"): v = f32(f32(1.0) + f32(f32(0.1) * u))
    //         else: v = f32(f32(f32(0.5) / f32(np.sqrt(f32(fan)))) * u)
    //         out[i] = v
    //     return out.reshape(dims)
    // cfg = Qwen3TTSTokenizerV2DecoderConfig(
    //     codebook_size=16, hidden_size=4, latent_dim=8, rope_theta=10000,
    //     num_attention_heads=2, num_key_value_heads=2, attention_bias=False,
    //     sliding_window=4, intermediate_size=8, hidden_act="silu", rms_norm_eps=1e-5,
    //     num_hidden_layers=2, num_quantizers=16, upsample_rates=(8, 5, 4, 3),
    //     upsampling_ratios=(2, 2), decoder_dim=32, codebook_dim=6, head_dim=2,
    //     attn_implementation="eager")
    // torch.set_grad_enabled(False)
    // dec = Qwen3TTSTokenizerV2Decoder(cfg).to(torch.float32).eval()
    // for name, p in dec.named_parameters():      # buffers (inv_freq) are NOT touched
    //     if not name.startswith("quantizer."):
    //         p.copy_(torch.from_numpy(fill("decoder." + name, list(p.shape))))
    // z = torch.from_numpy(fill("fixture.latent", [1, 6, 6]))
    // h = dec.pre_conv(z).transpose(1, 2)
    // pre = dec.pre_transformer(inputs_embeds=h).last_hidden_state.permute(0, 2, 1)
    // h = pre
    // for blocks in dec.upsample:
    //     for b in blocks: h = b(h)
    // for b in dec.decoder: h = b(h)
    // wav = h.clamp(min=-1, max=1)
    // print(pre.flatten().numpy(), wav.flatten().numpy()[[*range(12), *range(13, 11520, 457)]])
    // ```

    /// `pre_stage` output for `latent(6)`, from the PyTorch reference.
    // 9 significant digits: the shortest decimal that round-trips an f32.
    #[allow(clippy::excessive_precision)]
    const REF_PRE_T6: [f32; 48] = [
        4.99787442e-02, 2.95203179e-04, 5.22400476e-02, 3.96817550e-03, 3.37730832e-02,
        -6.37542158e-02, 1.88350886e-01, 2.37081051e-01, 1.86714917e-01, 2.31517375e-01,
        1.86638474e-01, 2.60324270e-01, -1.38030350e-01, -1.46742418e-01, -1.35593101e-01,
        -1.56535655e-01, -1.05591267e-01, -1.39886051e-01, -2.35614270e-01, -2.49551684e-01,
        -2.37616181e-01, -2.37535626e-01, -2.32715070e-01, -2.05165222e-01, 1.27287045e-01,
        1.50936514e-01, 1.30707413e-01, 1.28908694e-01, 1.88234091e-01, 1.81678653e-01,
        1.33675426e-01, 5.70919551e-03, 1.33441955e-01, 4.68456075e-02, 1.49870783e-01,
        5.01417443e-02, -5.10180175e-01, -5.69141269e-01, -5.12667239e-01, -5.43786526e-01,
        -5.26592016e-01, -5.41916430e-01, -4.83254075e-01, -5.32541335e-01, -4.83512461e-01,
        -5.20822823e-01, -4.87234354e-01, -5.28803587e-01,
    ];

    /// `decode_oneshot` output for `latent(6)`, from the PyTorch reference, sampled at
    /// indices `0..12` then `13, 470, 927, ...` (stride 457 — a stride coprime with the
    /// 1920-sample frame so it does not alias the stack's periodic structure).
    // 9 significant digits: the shortest decimal that round-trips an f32.
    #[allow(clippy::excessive_precision)]
    const REF_WAV_T6: [f32; 38] = [
        5.64664379e-02, 5.18466309e-02, 3.96115072e-02, 3.82112265e-02, 5.10178134e-02,
        5.18934391e-02, 5.33171631e-02, 5.88498674e-02, 5.63922450e-02, 5.13672605e-02,
        5.94830588e-02, 5.70506118e-02, 5.93978986e-02, 5.89976385e-02, 5.20804673e-02,
        6.00839183e-02, 5.76845966e-02, 5.19218706e-02, 6.05612472e-02, 5.82033098e-02,
        5.19905128e-02, 6.13957457e-02, 5.76180816e-02, 5.25969677e-02, 6.00450486e-02,
        5.92165664e-02, 5.18260226e-02, 6.02852032e-02, 5.78929670e-02, 5.17838784e-02,
        6.06448576e-02, 5.83945699e-02, 5.13712019e-02, 6.12486824e-02, 5.76713309e-02,
        5.26226982e-02, 6.02988228e-02, 5.91790676e-02,
    ];

    /// The whole stack, checked value-for-value against the reference implementation
    /// running on the identical model. This is the test that would have caught a wrong
    /// Snake form, a missing `exp`, a plain-causal instead of sliding-window mask, a
    /// LayerScale applied on the wrong side of the residual, a transposed conv trimmed
    /// on the wrong end, or ConvNeXt's LayerNorm/GELU variants — none of which the
    /// structural tests above can see.
    #[test]
    fn matches_the_pytorch_reference() {
        let (w, d) = fixture();
        let z = latent(6);

        let pre = vec1(&d.pre_stage(&w, &z).unwrap());
        assert_eq!(pre.len(), REF_PRE_T6.len());
        let e = worst(&pre, &REF_PRE_T6);
        assert!(e < 1e-6, "pre_stage differs from the reference by {e:e}");

        let wav = vec1(&d.decode_oneshot(&w, &z).unwrap());
        assert_eq!(wav.len(), 6 * d.config().total_upsample());
        let idx: Vec<usize> = (0..12).chain((13..wav.len()).step_by(457)).collect();
        assert_eq!(idx.len(), REF_WAV_T6.len());
        let got: Vec<f32> = idx.iter().map(|&i| wav[i]).collect();
        let e = worst(&got, &REF_WAV_T6);
        assert!(e < 1e-6, "decode differs from the reference by {e:e}");
    }
}
