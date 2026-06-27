//! Real CosyVoice2 LM forward via Candle (the DESIGN T2.1 "port base weights to a
//! Rust tensor format" path — the real model behind the toy reference parity).
//!
//! Loads the base model's **Qwen2-0.5B** LM backbone plus CosyVoice2's `llm_decoder`
//! head (converted to fp32 safetensors offline — too large to vendor) and reproduces
//! the reference per-position logits. This is gated behind the `real` cargo feature
//! and a model path on disk; the parity test skips cleanly when the weights are absent
//! (mirroring the device-bound task recipe) and runs for real where they exist.
//!
//! Architecture (from the checkpoint manifest): 24 decoder layers, hidden 896,
//! GQA with 14 query heads / 2 KV heads (head_dim 64, q/k/v carry bias, o_proj does
//! not), SwiGLU MLP (intermediate 4864), RoPE θ=1e6, RMSNorm eps 1e-6. The CosyVoice2
//! head is `llm_decoder: Linear(896 -> 6564)`.
//!
//! Module layout (a pure structural split — all logic is byte-preserved from the
//! original single-file port): this `mod.rs` owns the [`Qwen2Lm`] struct, the shared
//! [`QEmbed`]/[`KvCache`] types, the architecture constants and the core
//! forward/`linear`/`rms_norm`/`mlp` glue; `load` owns the fp32/int4 loaders +
//! footprint instrumentation; `attention` owns `attn`/`attn_cached` + RoPE/mask helpers;
//! `sampling` owns the PRNG + nucleus/ras/random samplers; `quant` owns the embedding
//! quantizers + [`EmbedScheme`]/[`Footprint`]; `generate` owns the embedding lookups,
//! prompt assembly and the autoregressive generation loops.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use std::collections::HashMap;

mod attention;
mod generate;
mod load;
mod quant;
mod sampling;

pub use quant::{EmbedScheme, Footprint, DEFAULT_EMBED_SCHEME};

const HIDDEN: usize = 896;
const N_LAYERS: usize = 24;
const N_HEADS: usize = 14;
const N_KV: usize = 2;
const HEAD_DIM: usize = 64;
const EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

// --- generation constants (from the CosyVoice2 Qwen2LM definition) -----------

/// `sos` row index into `llm_embedding`.
const SOS: u32 = 0;
/// `task_id` row index into `llm_embedding`.
const TASK_ID: u32 = 1;
/// `llm_decoder` output width (`speech_token_size + 3` = 6561 + 3).
const SPEECH_VOCAB: usize = 6564;
/// `speech_token_size`; `eos_token = speech_token_size`, and the stop set is the three
/// ids `[speech_token_size + i for i in range(3)]`.
const SPEECH_TOKEN_SIZE: u32 = 6561;
/// The decode-stop token ids (`stop_token_ids`).
const STOP_TOKENS: [u32; 3] = [SPEECH_TOKEN_SIZE, SPEECH_TOKEN_SIZE + 1, SPEECH_TOKEN_SIZE + 2];

/// The real Qwen2-0.5B LM + CosyVoice2 `llm_decoder`, loaded from fp32 safetensors.
///
/// Two precisions share this one struct and one forward:
///   * **fp32 (default, parity)** — every weight kept in `w` as f32; `linear` is the
///     plain `x @ Wᵀ` reference matmul. This is [`Qwen2Lm::load`], unchanged.
///   * **int4 (`load_quantized`)** — the big linear weights (`q/k/v/o_proj`,
///     `gate/up/down_proj`, and the `llm_decoder` head) are quantized to GGML `Q4_0`
///     and live in `qmm` as [`QMatMul`]; `linear` dispatches to the QMatMul (`x @ Wᵀ`,
///     int4 weight × f32 activation). Embeddings are quantized per-row to int4 (default;
///     see [`DEFAULT_EMBED_SCHEME`]) for dequant-on-gather, RMSNorm weights and the small
///     biases stay f32, and the unused Qwen2 `lm_head` (the ~520 MB dense remainder) is
///     dropped entirely. RoPE / softmax / attention math are f32 in both. The forward,
///     sampler, KV-cache and generation loop are byte-for-byte the same code path; only
///     the per-`linear` weight representation differs.
pub struct Qwen2Lm {
    /// Dense weights: norms + biases (f32). In the fp32 build this holds every weight
    /// (embeds included, as f32); in the quantized build the embeds move to `qembed`.
    w: HashMap<String, Tensor>,
    /// Quantized linear weights, keyed by the same name as the fp32 weight. Empty for
    /// the fp32 build; populated by [`Qwen2Lm::load_quantized`].
    qmm: HashMap<String, QMatMul>,
    /// int8-quantized embedding lookup tables (dequant-on-gather), keyed by table name.
    /// Empty for the fp32 build; populated by [`Qwen2Lm::load_quantized`].
    qembed: HashMap<String, QEmbed>,
    /// Sum of the `QTensor` storage sizes (bytes) realized by quantization — 0 in the
    /// fp32 build. Combined with the retained dense `w` to report the footprint.
    quant_bytes: usize,
    dev: Device,
}

/// A per-row symmetric quantized embedding table, supporting a *dequant-on-gather*
/// lookup: an embedding is a row lookup (`index_select`), not a matmul, so there is no
/// `QMatMul` here — we gather the needed rows of the quantized store + their per-row
/// scales, **then** dequantize only those few rows to f32. The full f32 table is never
/// reconstructed. Stores either int8 (one u8/weight) or int4 (two nibbles/u8) per
/// [`QEmbed::scheme`].
struct QEmbed {
    /// Which bit width this table is stored at (selects the (de)quant + pack path).
    scheme: EmbedScheme,
    /// Packed weights. **Int8**: `[V, H]` u8, each `q ∈ [-127,127]` stored as `q+128`
    /// (range `[1,255]`). **Int4**: `[V, H/2]` u8, two weights `q ∈ [-7,7]` per byte,
    /// each stored as the nibble `q+8` (range `[1,15]`); element `2i` is the low nibble,
    /// `2i+1` the high nibble. U8 is candle's only 1-byte dtype.
    q: Tensor,
    /// `[V, 1]` f32 per-row scale. Int8: `max(|row|)/127`; Int4: `max(|row|)/7`.
    scale: Tensor,
    /// Logical row width `H` (Int4 packs two per byte, so `q`'s last dim is `H/2`).
    h: usize,
    /// Realized storage bytes (`q` u8 + `scale` f32).
    bytes: usize,
}

/// Per-layer accumulated K/V for incremental (O(n)) autoregressive decoding.
///
/// Each entry holds that layer's running key/value sequence, **K stored
/// post-RoPE** at the keys' true absolute positions, shaped `[b, N_KV, seq, HEAD_DIM]`
/// (the pre-`repeat_kv` GQA layout — repetition is applied per step on read, never
/// stored). `seq` grows by the number of tokens fed each step. `len()` is the current
/// cache length and is exactly the absolute position the *next* token will occupy.
pub struct KvCache {
    /// `kv[layer] = Some((k, v))` once layer `layer` has been populated.
    kv: Vec<Option<(Tensor, Tensor)>>,
    /// Number of positions currently cached (== next token's absolute position).
    len: usize,
}

impl KvCache {
    /// An empty cache sized for the model's `N_LAYERS` decoder layers.
    pub fn new() -> Self {
        Self {
            kv: (0..N_LAYERS).map(|_| None).collect(),
            len: 0,
        }
    }

    /// Current cache length (number of cached positions). The next token fed will sit
    /// at absolute position `len()`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty (no positions cached yet).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append the new (post-RoPE) `k`/`v` slabs `[b, N_KV, t_new, HEAD_DIM]` for `layer`,
    /// returning the full cached `(k, v)` covering all positions `0..len+t_new`.
    /// Concatenation happens on the sequence axis (dim 2); existing entries are kept.
    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (nk, nv) = match self.kv[layer].take() {
            Some((ck, cv)) => (
                Tensor::cat(&[&ck, k], 2)?.contiguous()?,
                Tensor::cat(&[&cv, v], 2)?.contiguous()?,
            ),
            None => (k.contiguous()?, v.contiguous()?),
        };
        self.kv[layer] = Some((nk.clone(), nv.clone()));
        Ok((nk, nv))
    }

    /// Record that `t_new` positions were just appended (advance the shared length).
    /// Called once per forward after all layers have appended their slabs.
    fn advance(&mut self, t_new: usize) {
        self.len += t_new;
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen2Lm {
    fn g(&self, name: &str) -> Result<Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
            .cloned()
    }

    /// `x * rsqrt(mean(x^2, -1) + eps) * weight`, computed in f32 (Qwen2 RMSNorm).
    fn rms_norm(&self, x: &Tensor, wname: &str) -> Result<Tensor> {
        let w = self.g(wname)?; // [HIDDEN]
        let var = x.sqr()?.mean_keepdim(D::Minus1)?; // [.., 1]
        let xn = x.broadcast_div(&(var + EPS)?.sqrt()?)?;
        xn.broadcast_mul(&w)
    }

    /// `x @ W^T (+ b)` for a `[.., in]` input and a `[out, in]` weight.
    ///
    /// When a quantized `QMatMul` exists for `wname` (the int4 build) it computes the
    /// same `x @ Wᵀ` with an int4 weight (QMatMul requires a contiguous, f32 input);
    /// otherwise it is the dense fp32 matmul. The bias, when present, is always added in
    /// f32. The fp32 build never has a `qmm` entry, so its path is byte-for-byte the
    /// original reference matmul.
    fn linear(&self, x: &Tensor, wname: &str, bias: Option<&str>) -> Result<Tensor> {
        let y = if let Some(qm) = self.qmm.get(wname) {
            qm.forward(&x.contiguous()?)?
        } else {
            let w = self.g(wname)?;
            if w.dtype() == DType::F32 {
                x.broadcast_matmul(&w.t()?)?
            } else {
                // f16 dense fallback (non-block-aligned weight): upcast for the matmul.
                x.broadcast_matmul(&w.to_dtype(DType::F32)?.t()?)?
            }
        };
        match bias {
            Some(b) => y.broadcast_add(&self.g(b)?),
            None => Ok(y),
        }
    }

    /// Public `x @ Wᵀ (+b)` over a **named loaded weight** — an additive seam that lets a
    /// sibling speech LM sharing this exact Qwen2 body (the CosyVoice3 LM in `real_cv3`)
    /// apply its own output head (`llm_decoder`, bias-free) without re-implementing the
    /// projection. This is a thin pass-through to the private [`Qwen2Lm::linear`]; the CV2
    /// forward paths (`forward_logits`, `attn`, `mlp`, …) are unchanged and never call it.
    pub fn head_linear(&self, x: &Tensor, wname: &str, bias: Option<&str>) -> Result<Tensor> {
        self.linear(x, wname, bias)
    }

    fn mlp(&self, x: &Tensor, layer: usize) -> Result<Tensor> {
        let p = format!("llm.model.model.layers.{layer}.mlp");
        let gate = self.linear(x, &format!("{p}.gate_proj.weight"), None)?;
        let up = self.linear(x, &format!("{p}.up_proj.weight"), None)?;
        let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.linear(&act, &format!("{p}.down_proj.weight"), None)
    }

    /// Run the 24 decoder layers + final RMSNorm over an input embedding sequence
    /// `[b, t, 896]`, returning the last hidden state `[b, t, 896]`.
    pub fn forward_hidden(&self, embeds: &Tensor) -> Result<Tensor> {
        let (_b, t, _) = embeds.dims3()?;
        let (cos, sin) = attention::rope_cos_sin(t, &self.dev)?;
        let mask = attention::causal_mask(t, &self.dev)?;
        let mut h = embeds.clone();
        for l in 0..N_LAYERS {
            let pre = format!("llm.model.model.layers.{l}");
            let r = h.clone();
            let hn = self.rms_norm(&h, &format!("{pre}.input_layernorm.weight"))?;
            h = (r + self.attn(&hn, l, &cos, &sin, &mask)?)?;
            let r = h.clone();
            let hn = self.rms_norm(&h, &format!("{pre}.post_attention_layernorm.weight"))?;
            h = (r + self.mlp(&hn, l)?)?;
        }
        self.rms_norm(&h, "llm.model.model.norm.weight")
    }

    /// Full LM forward: hidden state -> CosyVoice2 `llm_decoder` -> logits `[b, t, 6564]`.
    pub fn forward_logits(&self, embeds: &Tensor) -> Result<Tensor> {
        let h = self.forward_hidden(embeds)?;
        self.linear(&h, "llm_decoder.weight", Some("llm_decoder.bias"))
    }

    /// Incremental (cached) variant of `forward_hidden`: run the 24 decoder layers
    /// over **only** the new tokens `embeds` `[b, t_new, 896]`, attending each layer's
    /// new queries over the full per-layer K/V in `cache`. The new tokens occupy
    /// absolute positions `cache.len()..cache.len()+t_new` (used for RoPE + the causal
    /// mask). On return the cache has grown by `t_new`. Output is the last hidden state
    /// for the new tokens only, `[b, t_new, 896]`.
    pub fn forward_hidden_cached(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (_b, t_new, _) = embeds.dims3()?;
        let offset = cache.len();
        let (cos, sin) = attention::rope_cos_sin_at(offset, t_new, &self.dev)?;
        let mask = attention::causal_mask_at(offset, t_new, &self.dev)?;
        let mut h = embeds.clone();
        for l in 0..N_LAYERS {
            let pre = format!("llm.model.model.layers.{l}");
            let r = h.clone();
            let hn = self.rms_norm(&h, &format!("{pre}.input_layernorm.weight"))?;
            h = (r + self.attn_cached(&hn, l, &cos, &sin, &mask, cache)?)?;
            let r = h.clone();
            let hn = self.rms_norm(&h, &format!("{pre}.post_attention_layernorm.weight"))?;
            h = (r + self.mlp(&hn, l)?)?;
        }
        // All layers have appended their slabs; advance the shared cache length once.
        cache.advance(t_new);
        self.rms_norm(&h, "llm.model.model.norm.weight")
    }

    /// Incremental (cached) variant of `forward_logits`: `forward_hidden_cached` then the
    /// `llm_decoder` head, returning logits for the new tokens only `[b, t_new, 6564]`.
    pub fn forward_logits_cached(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let h = self.forward_hidden_cached(embeds, cache)?;
        self.linear(&h, "llm_decoder.weight", Some("llm_decoder.bias"))
    }
}
