//! Shared Candle primitives for the s2 (Qwen3-4B `fish_qwen3_omni`) backend.
//!
//! Structurally identical to `s1::nn` — the same name-indexed weight bag, RMSNorm /
//! linear / embedding helpers, the Fish **interleaved** RoPE (`apply_rotary_emb`,
//! GPT-J pairing), the GQA self-attention block, and a per-layer KV cache. It is
//! duplicated here (rather than shared) so the s2 backend is self-contained inside
//! `src/s2/` per the wave's confinement rule.
//!
//! The one Qwen3 delta from s1's Llama backbone is already expressed through
//! [`AttnShape`] flags: `qk_norm` (per-head QK-RMSNorm before RoPE) is **on** for the
//! slow backbone, and `qkv_bias`/`o_bias` are read from the config. Per `s2-pro`'s
//! published `config.json` (`fish_qwen3` / `fish_qwen3_audio_decoder`), the real
//! Qwen3-3 design has **no** qkv/o bias and **does** carry QK-RMSNorm — so the slow AR
//! runs `qk_norm=true, qkv_bias=false, o_bias=false`, and the fast audio decoder runs
//! all three `false`. The block here honours whatever the resolved config says.
//
// PARITY: confirm on-box that the `fish_qwen3` attention truly omits qkv/o bias and
// applies per-head QK-RMSNorm exactly as Qwen3 (i.e. RMSNorm over head_dim, no learned
// bias, applied to q and k *before* RoPE). config.json says attention_qkv_bias=false,
// attention_o_bias=false, attention_qk_norm=true for text_config.

use candle_core::{Device, DType, Module, Result, Tensor, D};
use std::collections::HashMap;

/// A name → `Tensor` weight bag (the loaded checkpoint), plus the device and the
/// **compute dtype** `dt` every weight is stored in.
///
/// `dt` selects the precision of the whole forward: **f32** on CPU (the parity build —
/// `linear` is the plain `x @ Wᵀ` reference matmul) and **bf16** on CUDA (so the 4.4B
/// LM fits a 12 GB GPU). The published `s2-pro` LM ships bf16 across two shards;
/// [`super::load`] casts every shard to `dt` here.
///
// PARITY: bf16 (`dt == BF16`) is the GPU-fit path and may diverge numerically from the
// f32 CPU parity path. To stay stable, every variance/normalisation reduction below is
// computed in f32 internally (cast x→f32, reduce, normalise, cast back to `dt`) and the
// final logits are returned in f32 for the sampler. When `dt == F32` every such cast is
// an identity, so the CPU path is byte-unchanged.
pub struct Weights {
    pub map: HashMap<String, Tensor>,
    pub dev: Device,
    /// The dtype every weight is stored in and the transformer runs in (f32 or bf16).
    pub dt: DType,
    /// Optional **int4 store**: `name -> QMatMul` for the big 2-D projections.
    ///
    /// This is a `QMatMul`, **not** dequant-on-fetch. The CosyVoice quantizers store a
    /// `QTensor` and dequantize the whole weight every time the forward asks for it,
    /// which is why their `--quantized` help warns it is slow. `QMatMul::forward`
    /// instead dispatches into candle's fused quantized kernels
    /// (`mul_mat_vec_via_q8_1` at batch<=8, `mul_mat_via_q8_1` above), which read the
    /// int4 blocks directly and never materialise the f32 weight.
    ///
    /// Present only on a `load_quantized` build; empty means the dense path.
    pub qmap: HashMap<String, candle_core::quantized::QMatMul>,
}

impl Weights {
    /// Fetch a weight by name (clone of the stored f32 tensor).
    pub fn g(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
    }

    /// Whether `name` is present in the bag.
    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// `x @ Wᵀ (+ b)` for a `[.., in]` input and a `[out, in]` weight `wname`
    /// (the reference `nn.Linear` / `F.linear`).
    pub fn linear(&self, x: &Tensor, wname: &str, bias: Option<&str>) -> Result<Tensor> {
        // int4 store first: QMatMul already encodes Wᵀ, so it takes x directly.
        // candle's quantized CUDA kernels require f32 activations, which is why the
        // quantized build runs the whole forward in f32 (see `load_quantized`).
        let y = match self.qmap.get(wname) {
            Some(q) => q.forward(x)?,
            None => {
                let w = self.g(wname)?;
                // Dense fallback: a weight may be stored narrower than the compute dtype
                // (see `linear_w`). Match dtypes on the activation side.
                if w.dtype() == x.dtype() {
                    x.broadcast_matmul(&w.t()?)?
                } else {
                    let y = x.to_dtype(w.dtype())?.broadcast_matmul(&w.t()?)?;
                    y.to_dtype(x.dtype())?
                }
            }
        };
        match bias {
            Some(b) => y.broadcast_add(&self.g(b)?),
            None => Ok(y),
        }
    }

    /// `x @ Wᵀ` against an explicit weight tensor (for the tied LM head, whose weight
    /// is the input-embedding table rather than a dedicated `output.weight`).
    pub fn linear_w(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        // The int4 build runs f32 activations but keeps the 778 MB embedding table —
        // which doubles as the tied LM head — in bf16. Cast the ACTIVATION down to the
        // weight's dtype for the matmul and lift the result back, never the table: `x`
        // is [.., dim] and the table is [vocab, dim], so casting the table per step
        // would move ~800 MB every token to save casting a few KB.
        let xd = x.dtype();
        let wd = w.dtype();
        if xd == wd {
            return x.broadcast_matmul(&w.t()?);
        }
        let y = x.to_dtype(wd)?.broadcast_matmul(&w.t()?)?;
        y.to_dtype(xd)
    }

    /// RMSNorm: `x * rsqrt(mean(x², -1) + eps) * weight`. The normalisation is computed
    /// in f32 (the reference casts `x.float()` inside `_norm`), then the result is cast
    /// back to `x`'s dtype and the weight (in `dt`) is re-applied. For the f32 CPU path
    /// the f32 round-trip is an identity.
    // PARITY: bf16-stability — the mean-of-squares reduction stays in f32 regardless of `dt`.
    pub fn rms_norm(&self, x: &Tensor, wname: &str, eps: f64) -> Result<Tensor> {
        let w = self.g(wname)?;
        rms_norm_w(x, &w, eps)
    }

    /// Gather rows `ids` of the embedding table `table` → `[ids.len(), dim]`.
    pub fn embedding(&self, table: &str, ids: &[u32]) -> Result<Tensor> {
        let t = self.g(table)?;
        let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.dev)?;
        let rows = t.index_select(&idx, 0)?;
        // The table may be stored narrower than the compute dtype: the int4 build runs
        // f32 activations (candle's quantized kernels require f32) but keeps the 778 MB
        // embedding table in bf16, because it is gathered rather than multiplied and a
        // f32 copy would cost more VRAM than the quantization saves. Cast the gathered
        // rows only — `ids.len()` of them, not the whole table. A no-op when they match.
        if rows.dtype() == self.dt {
            Ok(rows)
        } else {
            rows.to_dtype(self.dt)
        }
    }
}

/// Bare RMSNorm against an explicit weight tensor (used inside per-head QK-norm,
/// where the weight is a `[head_dim]` vector, not a named-bag lookup).
///
/// The variance reduction + normalisation run in f32 for bf16-stability; the normalised
/// activation is cast back to `x`'s dtype before the (dtype-`dt`) weight multiply. When
/// `x` is already f32 every cast is an identity, so the CPU path is byte-unchanged.
pub fn rms_norm_w(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xf.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(dt)?;
    xn.broadcast_mul(w)
}

/// Precompute the Fish RoPE cos/sin tables `[seq_len, head_dim/2]`.
///
/// Mirrors `precompute_freqs_cis`: `inv_freq[i] = base^-(2i/head_dim)`,
/// `angle[t,i] = t * inv_freq[i]`, and the cos/sin are the real/imag of
/// `polar(1, angle)`. The interleaved [`apply_rotary_emb`] consumes these.
///
/// The angles are always computed in f32 (precision), then the cos/sin tables are cast
/// to `dt` so they can be consumed by `dt`-typed activations — Candle errors on a mixed
/// f32-table × bf16-x op. For `dt == F32` the cast is an identity (CPU path unchanged).
pub fn precompute_rope(
    seq_len: usize,
    head_dim: usize,
    base: f64,
    dev: &Device,
    dt: DType,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / (base as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let mut cos = vec![0f32; seq_len * half];
    let mut sin = vec![0f32; seq_len * half];
    for t in 0..seq_len {
        for i in 0..half {
            let a = (t as f32) * inv_freq[i];
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq_len, half), dev)?.to_dtype(dt)?,
        Tensor::from_vec(sin, (seq_len, half), dev)?.to_dtype(dt)?,
    ))
}

/// Fish **interleaved** rotary embedding on `x` shaped `[b, t, h, d]`, with cos/sin
/// slices `[t, d/2]` aligned to `x`'s `t` positions.
///
/// Reproduces `apply_rotary_emb`: reshape the last dim into `(d/2, 2)` adjacent
/// pairs, rotate each pair by its angle, and re-flatten. (Contrast with the HF
/// `rotate_half` convention, which pairs `i` with `i + d/2`.)
//
// PARITY: confirm `fish_qwen3` uses the same interleaved (GPT-J) RoPE pairing as the
// rest of the Fish stack rather than HF Qwen3's split-half `rotate_half`. The Fish
// reference `precompute_freqs_cis`/`apply_rotary_emb` are interleaved; if the s2 export
// kept HF Qwen3 attention verbatim it would be split-half. Confirm on-box.
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, t, h, d) = x.dims4()?;
    let half = d / 2;
    let xs = x.reshape((b, t, h, half, 2))?;
    let x0 = xs.narrow(4, 0, 1)?.reshape((b, t, h, half))?;
    let x1 = xs.narrow(4, 1, 1)?.reshape((b, t, h, half))?;
    let cos = cos.reshape((1, t, 1, half))?;
    let sin = sin.reshape((1, t, 1, half))?;
    let o0 = x0.broadcast_mul(&cos)?.sub(&x1.broadcast_mul(&sin)?)?;
    let o1 = x1.broadcast_mul(&cos)?.add(&x0.broadcast_mul(&sin)?)?;
    let out = Tensor::stack(&[&o0, &o1], 4)?.reshape((b, t, h, d))?;
    Ok(out)
}

/// Batched Fish interleaved rotary embedding: like [`apply_rotary_emb`], but the cos/sin
/// slices are **per-batch-element** `[b, t, d/2]` rather than the shared `[t, d/2]`. This
/// is what the left-padded batched generation path needs: every sample in the batch has a
/// DIFFERENT RoPE position id at the same physical column (its real-token positions skip
/// the sample's left-pad), so the rotation angle differs across the batch dim.
///
/// The pairing/rotation math is byte-identical to [`apply_rotary_emb`]; only the cos/sin
/// rank (and hence the broadcast over heads) changes. For `b == 1` with a position-aligned
/// table this would reduce to the single-sample function, so the two cannot disagree on
/// the batch=1 path (which keeps calling [`apply_rotary_emb`], unchanged).
pub fn apply_rotary_emb_batched(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, t, h, d) = x.dims4()?;
    let half = d / 2;
    let xs = x.reshape((b, t, h, half, 2))?;
    let x0 = xs.narrow(4, 0, 1)?.reshape((b, t, h, half))?;
    let x1 = xs.narrow(4, 1, 1)?.reshape((b, t, h, half))?;
    // cos/sin are [b, t, half] → [b, t, 1, half] so they broadcast over the head axis.
    let cos = cos.reshape((b, t, 1, half))?;
    let sin = sin.reshape((b, t, 1, half))?;
    let o0 = x0.broadcast_mul(&cos)?.sub(&x1.broadcast_mul(&sin)?)?;
    let o1 = x1.broadcast_mul(&cos)?.add(&x0.broadcast_mul(&sin)?)?;
    let out = Tensor::stack(&[&o0, &o1], 4)?.reshape((b, t, h, d))?;
    Ok(out)
}

/// GQA: expand `[b, kv, t, d]` KV heads so each serves `n` query heads → `[b, kv*n, t, d]`.
///
/// The result is always **contiguous** (the `n > 1` reshape materialises it; the `n == 1`
/// MHA short-circuit forces it). That matters now that [`KvCache`] hands back a `narrow`ed
/// view of a preallocated slab rather than a freshly `cat`-ed contiguous buffer: without
/// the `n == 1` `contiguous()` the downstream `matmul` would see a different batch stride
/// than it did under the old cat-per-step cache. Keeping the layout byte-identical keeps
/// the gemm configuration — and therefore the numerics — byte-identical too.
pub fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return x.contiguous();
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv, n, t, d))?
        .reshape((b, kv * n, t, d))
}

/// Additive causal mask `[t_new, offset + t_new]`: query row `i` (absolute position
/// `offset + i`) may attend key `j` iff `j <= offset + i`, else `-inf`.
/// Built in f32 then cast to `dt` so it adds cleanly onto `dt`-typed attention scores
/// (bf16 represents `-inf` exactly). For `dt == F32` the cast is an identity.
pub fn causal_mask_at(offset: usize, t_new: usize, dev: &Device, dt: DType) -> Result<Tensor> {
    let total = offset + t_new;
    let mut data = vec![0f32; t_new * total];
    for i in 0..t_new {
        let q_abs = offset + i;
        for j in (q_abs + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (t_new, total), dev)?.to_dtype(dt)
}

/// Growth granularity, in cached positions, for a [`KvCache`] built without an explicit
/// capacity. The slabs are allocated to a multiple of this and only reallocated when a
/// step would overflow, so a `T`-step generation reallocates `≈ T / KV_CHUNK` times
/// instead of every single step — while over-allocating by at most `KV_CHUNK - 1`
/// positions (36 layers × 8 kv-heads × 128 × bf16 × 255 ≈ 37 MB for the s2 slow AR at
/// batch 1), which matters on a 12 GB card holding a 10.4 GB model.
const KV_CHUNK: usize = 256;

/// One layer's running K/V for incremental autoregressive decoding. K is stored
/// **post-RoPE** at the keys' true absolute positions, shaped `[b, n_kv, seq, head_dim]`
/// (pre-`repeat_kv`); `repeat_kv` is applied on read, never stored.
///
/// ## Storage: a preallocated slab, not a per-step `cat`
///
/// Each layer owns one contiguous slab `[b, n_kv, cap, head_dim]`. A step writes its
/// `t_new` new positions **in place** at offset `len` with `slice_set` and hands back a
/// `narrow`ed view over `0..len + t_new`. The previous design rebuilt the whole cache
/// with `Tensor::cat` on every layer of every step, which copies `O(T²)` bytes over a
/// `T`-step generation and transiently doubles the cache allocation; this one copies
/// `O(T)` (plus `O(T² / KV_CHUNK)` for the rare growth reallocations).
///
/// The **contents** are bit-identical to the cat design — `slice_set` and `cat` both
/// perform an exact element copy, and [`repeat_kv`] re-materialises a contiguous
/// `[b, n_head, len, head_dim]` from the view exactly as it did from the cat output — so
/// every downstream gemm sees the same data in the same layout. The batched left-padded
/// path needs no special handling: left-padding lives in the RoPE position ids and the
/// additive mask, never in the cache layout, so physical column `c` means the same thing
/// under either storage scheme.
pub struct KvCache {
    /// Per-layer preallocated slabs `[b, n_kv, cap, head_dim]` (allocated on first use).
    kv: Vec<Option<(Tensor, Tensor)>>,
    /// Number of cached positions actually written.
    len: usize,
    /// Positions currently allocated in every live slab (0 before the first append).
    cap: usize,
    /// Exact capacity requested by [`KvCache::with_capacity`]; `0` = grow in `KV_CHUNK`s.
    hint: usize,
}

impl KvCache {
    /// A fresh cache for `n_layers` decoder layers, sized on demand in [`KV_CHUNK`]
    /// position steps. Use [`KvCache::with_capacity`] when the exact final length is
    /// known up front (the fast AR: one position per codebook).
    pub fn new(n_layers: usize) -> Self {
        Self::with_capacity(n_layers, 0)
    }

    /// A fresh cache for `n_layers` layers preallocated to exactly `capacity` positions
    /// on first append. It still grows if `capacity` turns out to be too small.
    pub fn with_capacity(n_layers: usize, capacity: usize) -> Self {
        Self {
            kv: (0..n_layers).map(|_| None).collect(),
            len: 0,
            cap: 0,
            hint: capacity,
        }
    }

    /// Number of cached positions (== the next token's absolute position).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Positions currently allocated per layer (the slab depth). Test/diagnostic hook.
    #[allow(dead_code)] // exercised by the sizing tests; kept as a diagnostic accessor.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// The slab depth to allocate so that `needed` positions fit: the requested capacity
    /// when it is large enough, else `needed` rounded up to a whole [`KV_CHUNK`].
    fn target_cap(&self, needed: usize) -> usize {
        if self.hint >= needed {
            self.hint
        } else {
            needed.div_ceil(KV_CHUNK) * KV_CHUNK
        }
    }

    /// An all-zero slab shaped like `like` `[b, n_kv, _, head_dim]` but `cap` deep.
    fn empty_slab(like: &Tensor, cap: usize) -> Result<Tensor> {
        let (b, h, _, d) = like.dims4()?;
        Tensor::zeros((b, h, cap, d), like.dtype(), like.device())
    }

    /// Reallocate every live slab to `new_cap` positions, carrying the existing contents
    /// over. Runs `O(log T)` times per generation, never per step.
    fn grow(&mut self, new_cap: usize) -> Result<()> {
        for slot in self.kv.iter_mut() {
            if let Some((k, v)) = slot.take() {
                // `k`/`v` are the FULL contiguous old slabs, so a single `slice_set` at
                // offset 0 carries them (and the unread tail) into the new ones.
                let nk = Self::empty_slab(&k, new_cap)?;
                nk.slice_set(&k, 2, 0)?;
                let nv = Self::empty_slab(&v, new_cap)?;
                nv.slice_set(&v, 2, 0)?;
                *slot = Some((nk, nv));
            }
        }
        self.cap = new_cap;
        Ok(())
    }

    /// Append the new (post-RoPE) `k`/`v` slabs `[b, n_kv, t_new, head_dim]` for
    /// `layer`, returning the full cached `(k, v)` covering positions `0..len+t_new`.
    pub fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let t_new = k.dim(2)?;
        let end = self.len + t_new;
        if end > self.cap {
            let new_cap = self.target_cap(end);
            self.grow(new_cap)?;
        }
        if self.kv[layer].is_none() {
            let slab_k = Self::empty_slab(k, self.cap)?;
            let slab_v = Self::empty_slab(v, self.cap)?;
            self.kv[layer] = Some((slab_k, slab_v));
        }
        // `slice_set` requires both sides contiguous; the callers already hand over
        // contiguous k/v, so this is a clone in the hot path.
        let (slab_k, slab_v) = self.kv[layer].as_ref().expect("slab allocated above");
        slab_k.slice_set(&k.contiguous()?, 2, self.len)?;
        slab_v.slice_set(&v.contiguous()?, 2, self.len)?;
        Ok((slab_k.narrow(2, 0, end)?, slab_v.narrow(2, 0, end)?))
    }

    /// Advance the shared length by `t_new` (call once per forward, after all layers
    /// have appended their slabs).
    pub fn advance(&mut self, t_new: usize) {
        self.len += t_new;
    }
}

/// Per-attention-block geometry (shared schema for the slow + fast transformers and
/// the codec's bottleneck transformer).
#[derive(Clone, Copy)]
pub struct AttnShape {
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub qkv_bias: bool,
    pub o_bias: bool,
    pub qk_norm: bool,
    pub eps: f64,
}

/// One GQA self-attention block (the reference `Attention.forward`), KV-cached.
///
/// `prefix` is the module path (e.g. `layers.3.attention`). `x` is `[b, t_new, dim]`;
/// `cos`/`sin` are `[t_new, head_dim/2]` for the queries' absolute positions; `mask`
/// (when `Some`) is the additive causal mask `[t_new, offset + t_new]`. Computes
/// `wqkv` → split → reshape → (optional per-head QK-RMSNorm) → interleaved RoPE →
/// cache-append → GQA-expand → scaled-dot-product attention → `wo`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    w: &Weights,
    prefix: &str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mask: Option<&Tensor>,
    shape: AttnShape,
    cache: &mut KvCache,
    layer: usize,
) -> Result<Tensor> {
    let (b, t, _) = x.dims3()?;
    let AttnShape {
        n_head,
        n_local_heads,
        head_dim,
        qkv_bias,
        o_bias,
        qk_norm,
        eps,
    } = shape;
    let q_size = n_head * head_dim;
    let kv_size = n_local_heads * head_dim;

    let qkv_bias_name = format!("{prefix}.wqkv.bias");
    let qkv = w.linear(
        x,
        &format!("{prefix}.wqkv.weight"),
        if qkv_bias { Some(qkv_bias_name.as_str()) } else { None },
    )?; // [b, t, q_size + 2*kv_size]

    let q = qkv.narrow(D::Minus1, 0, q_size)?;
    let k = qkv.narrow(D::Minus1, q_size, kv_size)?;
    let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?;

    let q = q.reshape((b, t, n_head, head_dim))?;
    let k = k.reshape((b, t, n_local_heads, head_dim))?;
    let v = v.reshape((b, t, n_local_heads, head_dim))?;

    // Optional per-head QK-RMSNorm (Qwen3-style; on for the s2 slow backbone, off for
    // the s2 fast audio decoder and the codec bottleneck).
    let (q, k) = if qk_norm {
        let qn = w.g(&format!("{prefix}.q_norm.weight"))?;
        let kn = w.g(&format!("{prefix}.k_norm.weight"))?;
        (rms_norm_w(&q, &qn, eps)?, rms_norm_w(&k, &kn, eps)?)
    } else {
        (q, k)
    };

    let q = apply_rotary_emb(&q, cos, sin)?;
    let k = apply_rotary_emb(&k, cos, sin)?;

    let q = q.transpose(1, 2)?.contiguous()?; // [b, n_head, t, hd]
    let k = k.transpose(1, 2)?.contiguous()?; // [b, n_local, t, hd]
    let v = v.transpose(1, 2)?.contiguous()?;

    let (k_full, v_full) = cache.append(layer, &k, &v)?;
    let k_full = repeat_kv(&k_full, n_head / n_local_heads)?;
    let v_full = repeat_kv(&v_full, n_head / n_local_heads)?;

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k_full.transpose(2, 3)?.contiguous()?)? * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let ctx = probs.matmul(&v_full)?; // [b, n_head, t, hd]
    let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, q_size))?;

    let o_bias_name = format!("{prefix}.wo.bias");
    w.linear(
        &ctx,
        &format!("{prefix}.wo.weight"),
        if o_bias { Some(o_bias_name.as_str()) } else { None },
    )
}

/// Batched GQA self-attention for the left-padded generation path. Identical in every
/// respect to [`attention`] except:
///   * `cos`/`sin` are **per-sample** `[b, t_new, head_dim/2]` (consumed by
///     [`apply_rotary_emb_batched`]) instead of the shared `[t_new, head_dim/2]`, because
///     each batch element's real-token RoPE positions skip its own left-pad;
///   * `mask` is the **per-sample** additive mask `[b, 1, t_new, total]` (causal AND the
///     left-pad key positions masked to `-inf` for each sample independently), which
///     broadcasts over the head axis onto the `[b, n_head, t_new, total]` scores.
///
/// Batch elements never attend across the batch dim (attention is per-element), so a
/// finished/frozen sample's stale K/V can never leak into another sample. The single
/// `attention` path is left untouched, so the batch=1 code is byte-for-byte unchanged.
#[allow(clippy::too_many_arguments)]
pub fn attention_batched(
    w: &Weights,
    prefix: &str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mask: Option<&Tensor>,
    shape: AttnShape,
    cache: &mut KvCache,
    layer: usize,
) -> Result<Tensor> {
    let (b, t, _) = x.dims3()?;
    let AttnShape {
        n_head,
        n_local_heads,
        head_dim,
        qkv_bias,
        o_bias,
        qk_norm,
        eps,
    } = shape;
    let q_size = n_head * head_dim;
    let kv_size = n_local_heads * head_dim;

    let qkv_bias_name = format!("{prefix}.wqkv.bias");
    let qkv = w.linear(
        x,
        &format!("{prefix}.wqkv.weight"),
        if qkv_bias { Some(qkv_bias_name.as_str()) } else { None },
    )?; // [b, t, q_size + 2*kv_size]

    let q = qkv.narrow(D::Minus1, 0, q_size)?;
    let k = qkv.narrow(D::Minus1, q_size, kv_size)?;
    let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?;

    let q = q.reshape((b, t, n_head, head_dim))?;
    let k = k.reshape((b, t, n_local_heads, head_dim))?;
    let v = v.reshape((b, t, n_local_heads, head_dim))?;

    let (q, k) = if qk_norm {
        let qn = w.g(&format!("{prefix}.q_norm.weight"))?;
        let kn = w.g(&format!("{prefix}.k_norm.weight"))?;
        (rms_norm_w(&q, &qn, eps)?, rms_norm_w(&k, &kn, eps)?)
    } else {
        (q, k)
    };

    let q = apply_rotary_emb_batched(&q, cos, sin)?;
    let k = apply_rotary_emb_batched(&k, cos, sin)?;

    let q = q.transpose(1, 2)?.contiguous()?; // [b, n_head, t, hd]
    let k = k.transpose(1, 2)?.contiguous()?; // [b, n_local, t, hd]
    let v = v.transpose(1, 2)?.contiguous()?;

    let (k_full, v_full) = cache.append(layer, &k, &v)?;
    let k_full = repeat_kv(&k_full, n_head / n_local_heads)?;
    let v_full = repeat_kv(&v_full, n_head / n_local_heads)?;

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k_full.transpose(2, 3)?.contiguous()?)? * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let ctx = probs.matmul(&v_full)?; // [b, n_head, t, hd]
    let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, q_size))?;

    let o_bias_name = format!("{prefix}.wo.bias");
    w.linear(
        &ctx,
        &format!("{prefix}.wo.weight"),
        if o_bias { Some(o_bias_name.as_str()) } else { None },
    )
}

/// SwiGLU feed-forward (the reference `FeedForward`): `w2(silu(w1(x)) * w3(x))`.
pub fn swiglu(w: &Weights, prefix: &str, x: &Tensor) -> Result<Tensor> {
    let g = w.linear(x, &format!("{prefix}.w1.weight"), None)?;
    let u = w.linear(x, &format!("{prefix}.w3.weight"), None)?;
    let act = candle_nn::ops::silu(&g)?.mul(&u)?;
    w.linear(&act, &format!("{prefix}.w2.weight"), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-optimisation `KvCache::append`: rebuild the whole cache with `Tensor::cat`
    /// on every step. Kept here as the reference the slab cache must reproduce exactly.
    struct CatCache {
        kv: Vec<Option<(Tensor, Tensor)>>,
        len: usize,
    }

    impl CatCache {
        fn new(n_layers: usize) -> Self {
            Self {
                kv: (0..n_layers).map(|_| None).collect(),
                len: 0,
            }
        }
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
        fn advance(&mut self, t_new: usize) {
            self.len += t_new;
        }
    }

    fn ramp(b: usize, h: usize, t: usize, d: usize, seed: f32, dev: &Device) -> Tensor {
        let n = b * h * t * d;
        let data: Vec<f32> = (0..n).map(|i| seed + i as f32 * 0.5).collect();
        Tensor::from_vec(data, (b, h, t, d), dev).unwrap()
    }

    /// Drive both caches through the same prefill + `steps` decode steps and assert the
    /// returned `(k, v)` are element-for-element equal at every layer of every step.
    fn assert_matches(b: usize, layers: usize, prefill: usize, steps: usize, cap_hint: usize) {
        let dev = Device::Cpu;
        let (h, d) = (2usize, 4usize);
        let mut slab = KvCache::with_capacity(layers, cap_hint);
        let mut cat = CatCache::new(layers);

        for step in 0..=steps {
            let t_new = if step == 0 { prefill } else { 1 };
            for l in 0..layers {
                let seed = (step * 100 + l) as f32;
                let k = ramp(b, h, t_new, d, seed, &dev);
                let v = ramp(b, h, t_new, d, seed + 0.25, &dev);
                let (sk, sv) = slab.append(l, &k, &v).unwrap();
                let (ck, cv) = cat.append(l, &k, &v).unwrap();
                assert_eq!(sk.dims(), ck.dims(), "k dims, step {step} layer {l}");
                assert_eq!(sv.dims(), cv.dims(), "v dims, step {step} layer {l}");
                let sk: Vec<f32> = sk.flatten_all().unwrap().to_vec1().unwrap();
                let ck: Vec<f32> = ck.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(sk, ck, "k contents, step {step} layer {l}");
                let sv: Vec<f32> = sv.flatten_all().unwrap().to_vec1().unwrap();
                let cv: Vec<f32> = cv.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(sv, cv, "v contents, step {step} layer {l}");
            }
            slab.advance(t_new);
            cat.advance(t_new);
            assert_eq!(slab.len(), cat.len);
        }
    }

    #[test]
    fn slab_cache_matches_cat_cache_single_sample() {
        // Stays inside the first KV_CHUNK: no growth reallocation at all.
        assert_matches(1, 3, 5, 20, 0);
    }

    #[test]
    fn slab_cache_matches_cat_cache_across_growth() {
        // Prefill lands just under KV_CHUNK, then the decode steps walk across the
        // KV_CHUNK boundary and force a growth reallocation mid-generation.
        assert_matches(1, 2, KV_CHUNK - 3, 8, 0);
    }

    #[test]
    fn slab_cache_matches_cat_cache_batched() {
        // The left-padded batched path only ever differs in the batch dim (padding lives
        // in the RoPE ids and the mask), so b > 1 must behave identically too.
        assert_matches(4, 2, 7, 12, 0);
    }

    #[test]
    fn slab_cache_matches_cat_cache_exact_capacity() {
        // The fast-AR shape: capacity known up front and never exceeded.
        assert_matches(3, 2, 1, 9, 10);
    }

    #[test]
    fn exact_capacity_is_honoured_and_never_grows() {
        let dev = Device::Cpu;
        let mut c = KvCache::with_capacity(2, 10);
        for step in 0..10 {
            for l in 0..2 {
                let k = ramp(1, 2, 1, 4, step as f32, &dev);
                c.append(l, &k, &k).unwrap();
            }
            c.advance(1);
            assert_eq!(c.capacity(), 10, "exact capacity must not be rounded or grown");
        }
        assert_eq!(c.len(), 10);
    }

    #[test]
    fn chunked_capacity_grows_only_at_chunk_boundaries() {
        let dev = Device::Cpu;
        let mut c = KvCache::new(1);
        assert_eq!(c.capacity(), 0);
        // Prefill of KV_CHUNK - 1 positions → one chunk allocated.
        let k = ramp(1, 2, KV_CHUNK - 1, 4, 0.0, &dev);
        c.append(0, &k, &k).unwrap();
        c.advance(KV_CHUNK - 1);
        assert_eq!(c.capacity(), KV_CHUNK);
        // The step that exactly fills the chunk must NOT reallocate.
        let one = ramp(1, 2, 1, 4, 1.0, &dev);
        c.append(0, &one, &one).unwrap();
        c.advance(1);
        assert_eq!(c.capacity(), KV_CHUNK);
        // The next one must, and by exactly one more chunk.
        c.append(0, &one, &one).unwrap();
        c.advance(1);
        assert_eq!(c.capacity(), 2 * KV_CHUNK);
    }

    #[test]
    fn repeat_kv_output_is_contiguous_from_a_narrowed_cache_view() {
        let dev = Device::Cpu;
        let mut c = KvCache::new(1);
        let k = ramp(1, 2, 3, 4, 0.0, &dev);
        let (kf, _) = c.append(0, &k, &k).unwrap();
        // The cache hands back a narrow view over a KV_CHUNK-deep slab, so it is NOT
        // contiguous — the point of the repeat_kv contract is that its output always is.
        assert!(!kf.is_contiguous());
        assert!(repeat_kv(&kf, 1).unwrap().is_contiguous());
        assert!(repeat_kv(&kf, 4).unwrap().is_contiguous());
        // ... and n == 1 must still be an exact passthrough of the values.
        let a: Vec<f32> = repeat_kv(&kf, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = kf.contiguous().unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, b);
    }
}

