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

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{safetensors, DType, Device, Module, Result, Tensor, D};
use std::collections::HashMap;

const HIDDEN: usize = 896;
const N_LAYERS: usize = 24;
const N_HEADS: usize = 14;
const N_KV: usize = 2;
const HEAD_DIM: usize = 64;
const EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

/// The real Qwen2-0.5B LM + CosyVoice2 `llm_decoder`, loaded from fp32 safetensors.
///
/// Two precisions share this one struct and one forward:
///   * **fp32 (default, parity)** — every weight kept in `w` as f32; `linear` is the
///     plain `x @ Wᵀ` reference matmul. This is [`Qwen2Lm::load`], unchanged.
///   * **int4 (`load_quantized`)** — the big linear weights (`q/k/v/o_proj`,
///     `gate/up/down_proj`, and the `llm_decoder` head) are quantized to GGML `Q4_0`
///     and live in `qmm` as [`QMatMul`]; `linear` dispatches to the QMatMul (`x @ Wᵀ`,
///     int4 weight × f32 activation). Embeddings are kept as an f16 lookup table,
///     RMSNorm weights and the small biases stay f32. RoPE / softmax / attention math
///     are f32 in both. The forward, sampler, KV-cache and generation loop are byte-for
///     -byte the same code path; only the per-`linear` weight representation differs.
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

/// A per-row symmetric **int8**-quantized embedding table, supporting a
/// *dequant-on-gather* lookup: an embedding is a row lookup (`index_select`), not a
/// matmul, so there is no `QMatMul` here — we `index_select` the needed rows of the
/// int8 store + their per-row scales, **then** dequantize only those few rows to f32.
/// The full f32 table is never reconstructed.
struct QEmbed {
    /// `[V, H]` u8: each signed int8 weight `q ∈ [-127,127]` stored as `q + 128`
    /// (range `[1,255]`) so it fits U8 — candle's only 1-byte dtype. One byte per
    /// element ⇒ exactly half the f16 table it replaces.
    q: Tensor,
    /// `[V, 1]` f32 per-row scale `max(|row|)/127`; dequant of a row is `(q-128)*scale`.
    scale: Tensor,
    /// Realized storage bytes (`q` u8 + `scale` f32).
    bytes: usize,
}

/// Per-row symmetric int8-quantize an `[V, H]` f32 embedding table for
/// dequant-on-gather. Each row carries its own scale `max(|row|)/127`; its weights are
/// `round(row/scale)` clamped to `[-127,127]`, stored `+128` as u8. An all-zero row
/// quantizes to all-`128` (the `+1e-12` keeps the `0/0` finite) ⇒ dequantizes to zeros.
fn quantize_embed_int8(table: &Tensor) -> Result<QEmbed> {
    let amax = table.abs()?.max_keepdim(D::Minus1)?; // [V, 1]
    let scale = ((amax / 127.0)? + 1e-12)?; // [V, 1], f32; +eps guards an all-zero row
    let q = table
        .broadcast_div(&scale)?
        .round()?
        .clamp(-127f32, 127f32)?; // [V, H], integer-valued f32 in [-127,127]
    let q = (q + 128.0)?.to_dtype(DType::U8)?; // store offset by +128 (range [1,255])
    let bytes = q.elem_count() + scale.elem_count() * DType::F32.size_in_bytes();
    Ok(QEmbed { q, scale, bytes })
}

/// Realized on-disk-equivalent footprint of a loaded [`Qwen2Lm`], split into the
/// quantized (int4) and dense (f16 embed + f32 norm/bias) parts. `total_bytes` is what
/// the model actually occupies for its weights, the headline number for the README's
/// size goal.
#[derive(Debug, Clone, Copy)]
pub struct Footprint {
    /// Bytes held by the `Q4_0` quantized linear weights (0 in the fp32 build).
    pub quant_bytes: usize,
    /// Bytes held by the int8-quantized embedding tables (0 in the fp32 build, where
    /// the embeds live in `dense_bytes` as f32 instead).
    pub embed_bytes: usize,
    /// Bytes held by the retained dense weights (norms/biases f32, plus the f32 embeds
    /// in the fp32 build).
    pub dense_bytes: usize,
    /// Number of weights that were quantized to int4.
    pub n_quantized: usize,
}

impl Footprint {
    /// Total realized weight bytes (`quant + int8-embed + dense`).
    pub fn total_bytes(&self) -> usize {
        self.quant_bytes + self.embed_bytes + self.dense_bytes
    }
    /// Total realized weight footprint in mebibytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
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
    /// Load the converted fp32 checkpoint (`llm_fp32.safetensors`) onto `dev`.
    ///
    /// This is the parity build: every weight is normalised to f32 and the forward is a
    /// clean fp32 reference match. Use [`Qwen2Lm::load_quantized`] for the int4 build.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        // Normalise to f32 so the forward is a clean fp32 reference match.
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self {
            w,
            qmm: HashMap::new(),
            qembed: HashMap::new(),
            quant_bytes: 0,
            dev,
        })
    }

    /// Load the same `llm_fp32.safetensors`, but quantize the big linear weights to
    /// **int4** (GGML `Q4_0`) for a ~4× smaller LM footprint (the README size goal).
    ///
    /// Quantized (one `QMatMul` each): every layer's `q/k/v/o_proj` and
    /// `gate/up/down_proj`, plus the `llm_decoder` output head — these are all true
    /// `x @ Wᵀ` matmuls, so `QMatMul::forward` is numerically the same op as the fp32
    /// `broadcast_matmul`, just with an int4 weight. `Q4_0` blocks of 32 run along each
    /// weight's inner (`in_features`) dim; every Qwen2 inner dim here (896 and the MLP
    /// intermediate 4864) is a multiple of 32, so all quantize cleanly. Any weight whose
    /// inner dim is **not** a multiple of 32 is left dense in f16 (recorded, none occur
    /// for these dims).
    ///
    /// Quantized to **int8** (per-row symmetric, [`QEmbed`]): the embedding tables
    /// (`embed_tokens` / `llm_embedding` / `speech_embedding`). These are `index_select`
    /// gathers, not matmuls, so they are *not* `QMatMul`s — the gathered rows are
    /// dequantized on lookup (see [`Qwen2Lm::embed_rows`]). int8 halves the f16 tables
    /// (1 byte/elem + a tiny per-row scale) — the embeds were the post-int4 footprint bulk.
    ///
    /// Kept dense: the RMSNorm weights (f32, tiny) and the attention q/k/v **biases**
    /// (f32, tiny).
    ///
    /// int4 trades accuracy for size; the forward is otherwise identical, so the
    /// quantized logits track but do not equal the fp32 logits (see the root
    /// `real_lm_quant` test, which measures argmax agreement + the realized footprint).
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        let mut qmm = HashMap::new();
        let mut qembed = HashMap::new();
        let mut quant_bytes = 0usize;
        for (k, v) in raw {
            let vf = v.to_dtype(DType::F32)?;
            if is_quantizable_linear(&k) {
                let dims = vf.dims();
                let inner = *dims.last().unwrap_or(&0);
                // Q4_0 needs the inner dim to be a multiple of the 32-element block.
                if dims.len() == 2 && inner % GgmlDType::Q4_0.block_size() == 0 {
                    let qt = QTensor::quantize(&vf, GgmlDType::Q4_0)?;
                    quant_bytes += qt.storage_size_in_bytes();
                    qmm.insert(k, QMatMul::from_qtensor(qt)?);
                    continue;
                }
                // Not block-aligned: keep this one dense in f16 (none occur for Qwen2).
                w.insert(k, vf.to_dtype(DType::F16)?);
                continue;
            }
            // Embedding tables -> per-row int8 dequant-on-gather lookup; everything else
            // (norms, biases) stays f32.
            if is_embedding_table(&k) {
                qembed.insert(k, quantize_embed_int8(&vf)?);
            } else {
                w.insert(k, vf);
            }
        }
        Ok(Self {
            w,
            qmm,
            qembed,
            quant_bytes,
            dev,
        })
    }

    /// Realized weight footprint (quantized + dense bytes) of this loaded model.
    pub fn footprint(&self) -> Footprint {
        let dense_bytes: usize = self
            .w
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        let embed_bytes: usize = self.qembed.values().map(|e| e.bytes).sum();
        Footprint {
            quant_bytes: self.quant_bytes,
            embed_bytes,
            dense_bytes,
            n_quantized: self.qmm.len(),
        }
    }

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

    fn attn(&self, x: &Tensor, layer: usize, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let p = format!("llm.model.model.layers.{layer}.self_attn");
        let (b, t, _) = x.dims3()?;
        let q = self.linear(x, &format!("{p}.q_proj.weight"), Some(&format!("{p}.q_proj.bias")))?;
        let k = self.linear(x, &format!("{p}.k_proj.weight"), Some(&format!("{p}.k_proj.bias")))?;
        let v = self.linear(x, &format!("{p}.v_proj.weight"), Some(&format!("{p}.v_proj.bias")))?;
        let q = q.reshape((b, t, N_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,14,t,64]
        let k = k.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t,64]
        let v = v.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t,64]
        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let k = repeat_kv(&k, N_HEADS / N_KV)?; // [b,14,t,64]
        let v = repeat_kv(&v, N_HEADS / N_KV)?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [b,14,t,t]
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?; // [b,14,t,64]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, HIDDEN))?;
        self.linear(&ctx, &format!("{p}.o_proj.weight"), None)
    }

    /// Cached attention for `t_new` query tokens at absolute positions
    /// `offset..offset+t_new`. Computes their Q/K/V, applies RoPE at the **absolute**
    /// positions (via `cos`/`sin` built for that range), appends the new K/V to
    /// `cache[layer]`, then attends the new queries over the **full** cached K/V
    /// (`offset+t_new` keys) under `mask` `[t_new, offset+t_new]`.
    ///
    /// This is numerically the same computation as `attn` restricted to the last
    /// `t_new` query rows of a full-sequence forward: identical Q/K/V projections,
    /// identical RoPE phases (absolute positions), identical causal visibility, so its
    /// logits match the full recompute to within fp rounding.
    fn attn_cached(
        &self,
        x: &Tensor,
        layer: usize,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let p = format!("llm.model.model.layers.{layer}.self_attn");
        let (b, t, _) = x.dims3()?;
        let q = self.linear(x, &format!("{p}.q_proj.weight"), Some(&format!("{p}.q_proj.bias")))?;
        let k = self.linear(x, &format!("{p}.k_proj.weight"), Some(&format!("{p}.k_proj.bias")))?;
        let v = self.linear(x, &format!("{p}.v_proj.weight"), Some(&format!("{p}.v_proj.bias")))?;
        let q = q.reshape((b, t, N_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,14,t_new,64]
        let k = k.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t_new,64]
        let v = v.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t_new,64]
        // RoPE the new Q and new K at their absolute positions, then commit the new
        // (post-RoPE) K and raw V to the cache, getting back the full cached K/V.
        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let (k_full, v_full) = cache.append(layer, &k, &v)?; // [b,2,offset+t_new,64]
        // GQA: repeat the *cached* KV heads up to the query-head count, then attend.
        let k_full = repeat_kv(&k_full, N_HEADS / N_KV)?; // [b,14,offset+t_new,64]
        let v_full = repeat_kv(&v_full, N_HEADS / N_KV)?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k_full.transpose(2, 3)?.contiguous()?)? * scale)?; // [b,14,t_new,offset+t_new]
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v_full)?; // [b,14,t_new,64]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, HIDDEN))?;
        self.linear(&ctx, &format!("{p}.o_proj.weight"), None)
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
        let (cos, sin) = rope_cos_sin(t, &self.dev)?;
        let mask = causal_mask(t, &self.dev)?;
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
        let (cos, sin) = rope_cos_sin_at(offset, t_new, &self.dev)?;
        let mask = causal_mask_at(offset, t_new, &self.dev)?;
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

    // ---------------------------------------------------------------------
    // Autoregressive speech-token generation (CosyVoice2 `Qwen2LM.inference`)
    // ---------------------------------------------------------------------

    /// Gather rows `ids` from a `[V, HIDDEN]` embedding table, returning `[1, n, HIDDEN]`.
    ///
    /// `ids` are u32 token ids; this is a plain row lookup (the `nn.Embedding` op).
    ///
    /// In the int8 quantized build the table lives in `qembed` as per-row int8: we
    /// `index_select` only the needed rows of the u8 store + their per-row scales, then
    /// **dequantize just those rows** to f32 (`(q-128)*scale`) — the full f32 table is
    /// never reconstructed. In the fp32 build the table is a plain f32 tensor in `w`.
    fn embed_rows(&self, table: &str, ids: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.dev)?;
        if let Some(qe) = self.qembed.get(table) {
            let q = qe.q.index_select(&idx, 0)?.to_dtype(DType::F32)?; // [n, HIDDEN]
            let s = qe.scale.index_select(&idx, 0)?; // [n, 1]
            let rows = (q - 128.0)?.broadcast_mul(&s)?; // [n, HIDDEN], dequantized
            return rows.unsqueeze(0); // [1, n, HIDDEN]
        }
        let w = self.g(table)?; // [V, HIDDEN]
        let rows = w.index_select(&idx, 0)?; // [n, HIDDEN]
        let rows = rows.to_dtype(DType::F32)?;
        rows.unsqueeze(0) // [1, n, HIDDEN]
    }

    /// The Qwen2 text embedding for `text_token` ids (`embed_tokens`), `[1, n, HIDDEN]`.
    pub fn text_embed(&self, text_token: &[u32]) -> Result<Tensor> {
        self.embed_rows("llm.model.model.embed_tokens.weight", text_token)
    }

    /// One `llm_embedding` row (`sos`=0 / `task_id`=1), shaped `[1, 1, HIDDEN]`.
    fn llm_embed_row(&self, id: u32) -> Result<Tensor> {
        self.embed_rows("llm_embedding.weight", &[id])
    }

    /// `speech_embedding` rows for the given speech-token ids, `[1, n, HIDDEN]`.
    pub fn speech_embed(&self, speech_token: &[u32]) -> Result<Tensor> {
        self.embed_rows("speech_embedding.weight", speech_token)
    }

    /// Assemble the step-0 LM input exactly as `Qwen2LM.inference`:
    /// `[sos_emb, text_emb(text_token), task_id_emb, prompt_speech_emb]` -> `[1, T0, HIDDEN]`.
    ///
    /// `text_token` here is already the concatenation of `prompt_text` and `text`
    /// (the reference concatenates them before embedding). `prompt_speech_token` may be
    /// empty, in which case the prompt-speech segment is omitted.
    pub fn build_lm_input(&self, text_token: &[u32], prompt_speech_token: &[u32]) -> Result<Tensor> {
        let sos = self.llm_embed_row(SOS)?; // [1,1,H]
        let task = self.llm_embed_row(TASK_ID)?; // [1,1,H]
        let text = self.text_embed(text_token)?; // [1,Tt,H]
        let mut parts: Vec<Tensor> = vec![sos, text, task];
        if !prompt_speech_token.is_empty() {
            parts.push(self.speech_embed(prompt_speech_token)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Last-position raw `llm_decoder` logits for an input embedding sequence
    /// `[1, T, HIDDEN]`, returning `[V]` (`V = SPEECH_VOCAB = 6564`).
    ///
    /// Recomputes the full transformer each call (O(n²) but logit-identical to a KV
    /// cache — same positions, same causal mask), which is what we want for parity.
    pub fn step_logits(&self, embeds: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(embeds)?; // [1, T, V]
        let t = logits.dim(1)?;
        logits.narrow(1, t - 1, 1)?.reshape((SPEECH_VOCAB,))
    }

    /// Last-position raw `llm_decoder` logits for `embeds` `[1, t_new, HIDDEN]` fed into
    /// the **cached** path, returning `[V]`. Advances `cache` by `t_new`. With the cache
    /// at length `L`, this is the logit-identical O(t_new) analogue of `step_logits` over
    /// an `L+t_new` recompute (same positions, same causal visibility).
    pub fn step_logits_cached(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let logits = self.forward_logits_cached(embeds, cache)?; // [1, t_new, V]
        let t = logits.dim(1)?;
        logits.narrow(1, t - 1, 1)?.reshape((SPEECH_VOCAB,))
    }

    /// Autoregressively generate speech tokens, mirroring `Qwen2LM.inference`, using a
    /// **KV cache** so each step is O(n) instead of O(n²).
    ///
    /// Prefill: assemble `build_lm_input` and run it once through the cached forward,
    /// populating every layer's K/V and yielding the step-0 last-position logits. Then
    /// per step: `log_softmax` -> `ras_sampling` (with `seed`-pinned multinomial draws)
    /// -> stop if the chosen id is a stop token, else append its `speech_embedding` row
    /// and feed **only that one token** through the cached forward (cache grows by 1).
    /// EOS is masked while `step < min_len`. Returns the generated token ids (stop token
    /// excluded), matching the reference's `out_tokens`. Because the cache carries the
    /// full history, generation may run to the real `max_len` with no practical cap.
    pub fn generate(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let lm_input0 = self.build_lm_input(text_token, prompt_speech_token)?;
        let mut cache = KvCache::new();
        let mut rng = SplitMix64::new(seed);
        let mut out: Vec<u32> = Vec::new();
        // Prefill the assembled prompt once; `logits` is the step-0 last-position logit.
        let mut logits = self.step_logits_cached(&lm_input0, &mut cache)?;
        for i in 0..max_len {
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            if STOP_TOKENS.contains(&top) {
                break;
            }
            out.push(top);
            // Feed only the newly sampled token; the cache supplies all prior context.
            let row = self.speech_embed(&[top])?; // [1,1,H]
            logits = self.step_logits_cached(&row, &mut cache)?;
        }
        Ok(out)
    }

    /// Reference O(n²) full-recompute generation — the pre-cache algorithm, kept as the
    /// correctness oracle for the cached `generate`. Identical sampling, stop conditions,
    /// pinned PRNG and `min_len` EOS masking; the *only* difference from `generate` is
    /// that each step re-runs the whole sequence (`step_logits`) instead of using a cache.
    /// A fixed seed must yield the exact same token vector as `generate`.
    pub fn generate_full_recompute(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let mut embeds = self.build_lm_input(text_token, prompt_speech_token)?;
        let mut rng = SplitMix64::new(seed);
        let mut out: Vec<u32> = Vec::new();
        for i in 0..max_len {
            let logits = self.step_logits(&embeds)?;
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            if STOP_TOKENS.contains(&top) {
                break;
            }
            out.push(top);
            let row = self.speech_embed(&[top])?; // [1,1,H]
            embeds = Tensor::cat(&[&embeds, &row], 1)?;
        }
        Ok(out)
    }

    /// Teacher-forced per-step logits: given the reference's chosen token sequence,
    /// rebuild the full embedding sequence and return every step's last-position logits
    /// as `[N, V]` (step `k`'s logits at row `k`). This proves the AR forward reproduces
    /// the reference logit-for-logit independent of the (stochastic) sampler, and is the
    /// real correctness signal for the generation loop.
    pub fn teacher_forced_logits(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        gen_tokens: &[u32],
    ) -> Result<Tensor> {
        let lm_input0 = self.build_lm_input(text_token, prompt_speech_token)?;
        let t0 = lm_input0.dim(1)?;
        // Append speech embeddings for all but the last generated token: step k's
        // last-position logit lives at absolute position (t0 - 1 + k).
        let embeds = if gen_tokens.len() > 1 {
            let tail = self.speech_embed(&gen_tokens[..gen_tokens.len() - 1])?;
            Tensor::cat(&[&lm_input0, &tail], 1)?
        } else {
            lm_input0
        };
        let logits = self.forward_logits(&embeds)?; // [1, T, V]
        let n = gen_tokens.len();
        // rows [t0-1 .. t0-1+n) are the n per-step last-position logits.
        logits.narrow(1, t0 - 1, n)?.reshape((n, SPEECH_VOCAB))
    }
}

// --- weight classification for the int4 (Q4_0) quantized build ---------------

/// The per-layer linear-weight suffixes that are real `x @ Wᵀ` matmuls and so are
/// quantized to int4 (their separate `.bias` entries end in `.bias` and are excluded).
const QUANT_PROJ_SUFFIXES: [&str; 7] = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
];

/// Whether `name` is a big linear weight to quantize: any layer projection above, or
/// the `llm_decoder` output head. Biases, norms and embeddings are excluded.
fn is_quantizable_linear(name: &str) -> bool {
    name == "llm_decoder.weight" || QUANT_PROJ_SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// Whether `name` is an embedding lookup table (kept as an f16 gather, not a matmul).
fn is_embedding_table(name: &str) -> bool {
    name == "llm.model.model.embed_tokens.weight"
        || name == "llm_embedding.weight"
        || name == "speech_embedding.weight"
}

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

/// `log_softmax` over a 1-D logit vector `[V]`, returned as a host `Vec<f32>`.
fn log_softmax_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let v: Vec<f32> = logits.to_vec1()?;
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &x in &v {
        sum += ((x - m) as f64).exp();
    }
    let lse = m as f64 + sum.ln();
    Ok(v.iter().map(|&x| (x as f64 - lse) as f32).collect())
}

/// Deterministic SplitMix64 PRNG — pins the otherwise-stochastic multinomial draws so a
/// `generate` run is bit-reproducible from a seed (the reference pins torch's RNG; we
/// pin ours). `next_f64` yields a uniform in `[0, 1)`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa uniform in [0,1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Sample one index from a categorical distribution given by `probs` (need not be
/// normalised) using inverse-CDF on a single uniform draw — the deterministic analogue
/// of `torch.multinomial(probs, 1)`.
fn multinomial1(probs: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f64 = probs.iter().map(|&p| p as f64).sum();
    let u = rng.next_f64() * total;
    let mut acc = 0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p as f64;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// `nucleus_sampling`: softmax(logp) is `exp(logp)`; sort descending (stable), take the
/// leading tokens while `cum_prob < top_p` AND `count < top_k`, then sample one of those
/// by `multinomial`. Returns the chosen vocab id. `logp` is a log-probability vector.
fn nucleus_sampling(logp: &[f32], top_p: f32, top_k: usize, rng: &mut SplitMix64) -> u32 {
    // probabilities = exp(log_softmax)
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    // stable descending sort by probability (ties keep ascending index, like torch stable)
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut cum = 0f32;
    let mut cand_idx: Vec<u32> = Vec::new();
    let mut cand_prob: Vec<f32> = Vec::new();
    for &i in &order {
        if cum < top_p && cand_prob.len() < top_k {
            cum += probs[i];
            cand_prob.push(probs[i]);
            cand_idx.push(i as u32);
        } else {
            break;
        }
    }
    let pick = multinomial1(&cand_prob, rng);
    cand_idx[pick]
}

/// `random_sampling`: full-softmax multinomial over the whole vocab (used by `ras` after
/// it masks a repeated token).
fn random_sampling(logp: &[f32], rng: &mut SplitMix64) -> u32 {
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    multinomial1(&probs, rng) as u32
}

/// `ras_sampling` (Repetition-Aware Sampling): nucleus-sample a candidate; if it has
/// repeated `>= win_size * tau_r` times in the last `win_size` decoded tokens, mask it
/// and fall back to `random_sampling`. EOS (`speech_token_size`) is `-inf`-masked first
/// when `ignore_eos`. Mirrors `cosyvoice.utils.common.ras_sampling` with the pinned
/// `top_p=0.8, top_k=25, win_size=10, tau_r=0.1`.
fn ras_sampling(logp: &[f32], decoded: &[u32], ignore_eos: bool, rng: &mut SplitMix64) -> u32 {
    const TOP_P: f32 = 0.8;
    const TOP_K: usize = 25;
    const WIN: usize = 10;
    const TAU_R: f32 = 0.1;
    let mut logp = logp.to_vec();
    if ignore_eos {
        logp[SPEECH_TOKEN_SIZE as usize] = f32::NEG_INFINITY;
    }
    let top = nucleus_sampling(&logp, TOP_P, TOP_K, rng);
    let start = decoded.len().saturating_sub(WIN);
    let rep = decoded[start..].iter().filter(|&&t| t == top).count();
    if (rep as f32) >= WIN as f32 * TAU_R {
        logp[top as usize] = f32::NEG_INFINITY;
        return random_sampling(&logp, rng);
    }
    top
}

/// Apply rotary position embedding (HF `rotate_half` convention) to `[b, h, t, d]`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)?.add(&rot.broadcast_mul(sin)?)
}

/// GQA: expand `[b, kv, t, d]` KV heads so each serves `n` query heads -> `[b, kv*n, t, d]`.
fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, kv, n, t, d))?.reshape((b, kv * n, t, d))
}

/// Build RoPE cos/sin tables `[t, head_dim]` for positions `0..t`.
fn rope_cos_sin(t: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    rope_cos_sin_at(0, t, dev)
}

/// Build RoPE cos/sin tables `[t, head_dim]` for the **absolute** positions
/// `offset..offset+t`. Caching feeds only the new token(s), so their rotary phase
/// must use their true absolute position (= `offset`, the current cache length),
/// not a reset-to-zero position — this is the load-bearing correctness detail.
fn rope_cos_sin_at(offset: usize, t: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), dev)?;
    let pos: Vec<f32> = (0..t).map(|i| (offset + i) as f32).collect();
    let pos = Tensor::from_vec(pos, (t,), dev)?;
    let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [t, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [t, head_dim]
    Ok((emb.cos()?, emb.sin()?))
}

/// Additive causal mask `[t, t]`: 0 on/below the diagonal, -inf above.
fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
    causal_mask_at(0, t, dev)
}

/// Additive causal mask `[t_new, offset + t_new]` for `t_new` queries at absolute
/// positions `offset..offset+t_new` attending over all `offset+t_new` keys.
/// Query row `i` (absolute position `offset+i`) may attend key column `j` iff
/// `j <= offset + i` (causal); entries above that are `-inf`. With `offset = 0`
/// this is the square mask; with one new query over a full cache it is all-zeros.
fn causal_mask_at(offset: usize, t_new: usize, dev: &Device) -> Result<Tensor> {
    let total = offset + t_new;
    let mut data = vec![0f32; t_new * total];
    for i in 0..t_new {
        let q_abs = offset + i;
        for j in (q_abs + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (t_new, total), dev)
}
