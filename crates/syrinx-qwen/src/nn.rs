//! Qwen3 transformer primitives: RMSNorm, rotary embedding, GQA attention with
//! per-head q/k normalisation, SwiGLU, and a preallocated KV cache.
//!
//! ## Why this is not shared with `syrinx-fish`
//!
//! Fish s2 is also a Qwen3 backbone, but the two differ in ways that make a shared
//! implementation more dangerous than a duplicated one:
//!
//! * **RoPE convention.** Fish uses the *interleaved* pairing `(x[2i], x[2i+1])`.
//!   Qwen3-TTS uses HuggingFace's *half-split* `rotate_half` — `x1 = x[..d/2]`,
//!   `x2 = x[d/2..]`, `cat(-x2, x1)`. Swapping them yields audio that sounds like
//!   speech and is wrong, which is the worst failure mode available.
//! * **Projections.** Fish fuses `wqkv`; Qwen3-TTS ships separate `q_proj`/`k_proj`/
//!   `v_proj` with a `q_norm`/`k_norm` applied per head *before* RoPE.
//! * **Licensing.** These weights are Apache-2.0; the Fish crate's are not. Keeping the
//!   crates independent means an Apache-only consumer needs no Fish code at all.
//!
//! ## M-RoPE
//!
//! The talker config carries `rope_scaling = {interleaved: true, mrope_section:
//! [24,20,20]}` — Qwen's 3-D multimodal RoPE. It is **provably inert here**: every
//! `position_ids` in the talker is built as `arange(...).expand(3, ...)`, so the
//! temporal/height/width rows are identical copies. `apply_interleaved_rope` then
//! overwrites strided slices of row 0 with values from rows 1 and 2 — which, when all
//! three rows are equal, changes nothing. So M-RoPE reduces exactly to the 1-D
//! half-split rotation implemented below. (A vision-conditioned path would need the
//! general form; TTS has no vision inputs.)

use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;

/// A name → tensor weight bag plus the device and compute dtype.
pub struct Weights {
    pub map: HashMap<String, Tensor>,
    pub dev: Device,
    pub dt: DType,
}

impl Weights {
    pub fn g(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// `x @ Wᵀ (+ b)` for a `[.., in]` input and an `[out, in]` weight.
    pub fn linear(&self, x: &Tensor, wname: &str, bias: Option<&str>) -> Result<Tensor> {
        let w = self.g(wname)?;
        let y = x.broadcast_matmul(&w.t()?)?;
        match bias {
            Some(b) => y.broadcast_add(&self.g(b)?),
            None => Ok(y),
        }
    }

    /// Gather rows `ids` of an embedding table, cast to the compute dtype.
    pub fn embedding(&self, table: &str, ids: &[u32]) -> Result<Tensor> {
        let t = self.g(table)?;
        let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.dev)?;
        let rows = t.index_select(&idx, 0)?;
        if rows.dtype() == self.dt {
            Ok(rows)
        } else {
            rows.to_dtype(self.dt)
        }
    }

    pub fn rms_norm(&self, x: &Tensor, wname: &str, eps: f64) -> Result<Tensor> {
        rms_norm_w(x, &self.g(wname)?, eps)
    }
}

/// RMSNorm against an explicit weight: `x * rsqrt(mean(x²) + eps) * w`.
///
/// The reduction runs in f32 regardless of the compute dtype — a bf16 mean-of-squares
/// over a 1024-wide vector loses enough precision to shift the normalised activation.
/// For `dt == F32` every cast is an identity.
pub fn rms_norm_w(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xf.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(dt)?;
    xn.broadcast_mul(w)
}

/// Precompute `cos`/`sin` for the half-split rotation, shape `[seq_len, head_dim]`.
///
/// HuggingFace builds `emb = cat(freqs, freqs)` so that channel `i` and channel
/// `i + head_dim/2` share an angle — the pairing `rotate_half` assumes.
pub fn precompute_rope(
    seq_len: usize,
    head_dim: usize,
    base: f64,
    dev: &Device,
    dt: DType,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1f32 / (base as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let mut cos = vec![0f32; seq_len * head_dim];
    let mut sin = vec![0f32; seq_len * head_dim];
    for t in 0..seq_len {
        for i in 0..half {
            let a = (t as f32) * inv[i];
            let (c, s) = (a.cos(), a.sin());
            cos[t * head_dim + i] = c;
            cos[t * head_dim + half + i] = c;
            sin[t * head_dim + i] = s;
            sin[t * head_dim + half + i] = s;
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq_len, head_dim), dev)?.to_dtype(dt)?,
        Tensor::from_vec(sin, (seq_len, head_dim), dev)?.to_dtype(dt)?,
    ))
}

/// `cat(-x2, x1)` over the last dim — HuggingFace's `rotate_half`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, d - half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply the half-split rotation to `[b, h, t, d]` with `cos`/`sin` `[t, d]`.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, t, d) = x.dims4()?;
    let cos = cos.narrow(0, 0, t)?.reshape((1, 1, t, d))?;
    let sin = sin.narrow(0, 0, t)?.reshape((1, 1, t, d))?;
    x.broadcast_mul(&cos)?
        .add(&rotate_half(x)?.broadcast_mul(&sin)?)
}

/// Expand `[b, kv, t, d]` to `[b, kv*n, t, d]` for grouped-query attention.
pub fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return x.contiguous();
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv, n, t, d))?
        .reshape((b, kv * n, t, d))
}

/// Additive causal mask `[t_new, offset + t_new]`: query `i` (absolute `offset + i`)
/// may attend key `j` iff `j <= offset + i`.
pub fn causal_mask_at(offset: usize, t_new: usize, dev: &Device, dt: DType) -> Result<Tensor> {
    let total = offset + t_new;
    let mut m = vec![0f32; t_new * total];
    for i in 0..t_new {
        for j in 0..total {
            if j > offset + i {
                m[i * total + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(m, (t_new, total), dev)?.to_dtype(dt)
}

/// A per-layer preallocated KV cache.
///
/// Grows in fixed chunks and writes in place rather than re-concatenating: the
/// `Tensor::cat`-per-token design this replaces copies the whole cache on every step,
/// which is O(T²) traffic over a generation.
pub struct KvCache {
    kv: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
    cap: usize,
}

/// Positions added per growth step.
const CACHE_CHUNK: usize = 256;

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self { kv: (0..n_layers).map(|_| None).collect(), len: 0, cap: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append `k`/`v` `[b, n_kv, t_new, d]` for `layer`, returning the full cached
    /// `(k, v)` covering `0..len + t_new`.
    pub fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, nkv, t_new, d) = k.dims4()?;
        let need = self.len + t_new;
        if need > self.cap {
            let newcap = need.div_ceil(CACHE_CHUNK) * CACHE_CHUNK;
            for slot in self.kv.iter_mut() {
                *slot = match slot.take() {
                    None => None,
                    Some((ck, cv)) => {
                        let grow = |t: Tensor| -> Result<Tensor> {
                            let (b, nkv, _, d) = t.dims4()?;
                            let z = Tensor::zeros((b, nkv, newcap, d), t.dtype(), t.device())?;
                            z.slice_assign(&[0..b, 0..nkv, 0..self.len, 0..d], &t.narrow(2, 0, self.len)?)
                        };
                        Some((grow(ck)?, grow(cv)?))
                    }
                };
            }
            self.cap = newcap;
        }
        let slot = &mut self.kv[layer];
        if slot.is_none() {
            let z = |t: &Tensor| Tensor::zeros((b, nkv, self.cap, d), t.dtype(), t.device());
            *slot = Some((z(k)?, z(v)?));
        }
        let (ck, cv) = slot.as_ref().unwrap();
        let nk = ck.slice_assign(&[0..b, 0..nkv, self.len..self.len + t_new, 0..d], k)?;
        let nv = cv.slice_assign(&[0..b, 0..nkv, self.len..self.len + t_new, 0..d], v)?;
        *slot = Some((nk.clone(), nv.clone()));
        Ok((
            nk.narrow(2, 0, self.len + t_new)?,
            nv.narrow(2, 0, self.len + t_new)?,
        ))
    }

    /// Advance the write cursor after every layer has appended `t_new`.
    pub fn advance(&mut self, t_new: usize) {
        self.len += t_new;
    }
}

/// One attention block's shape.
#[derive(Debug, Clone, Copy)]
pub struct AttnShape {
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f64,
}

/// Qwen3 self-attention: separate q/k/v projections, per-head RMSNorm on q and k
/// **before** RoPE, GQA, causal mask, then `o_proj`.
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
    let (b, t, _dim) = x.dims3()?;
    let AttnShape { n_head, n_kv, head_dim, eps } = shape;

    let q = w.linear(x, &format!("{prefix}.q_proj.weight"), None)?;
    let k = w.linear(x, &format!("{prefix}.k_proj.weight"), None)?;
    let v = w.linear(x, &format!("{prefix}.v_proj.weight"), None)?;

    // [b, t, n, d] then per-head norm (over the head_dim axis), then [b, n, t, d].
    let q = q.reshape((b, t, n_head, head_dim))?;
    let k = k.reshape((b, t, n_kv, head_dim))?;
    let q = rms_norm_w(&q, &w.g(&format!("{prefix}.q_norm.weight"))?, eps)?;
    let k = rms_norm_w(&k, &w.g(&format!("{prefix}.k_norm.weight"))?, eps)?;
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.reshape((b, t, n_kv, head_dim))?.transpose(1, 2)?.contiguous()?;

    let offset = cache.len();
    let cos_s = cos.narrow(0, offset, t)?;
    let sin_s = sin.narrow(0, offset, t)?;
    let q = apply_rope(&q, &cos_s, &sin_s)?;
    let k = apply_rope(&k, &cos_s, &sin_s)?;

    let (k, v) = cache.append(layer, &k, &v)?;
    let k = repeat_kv(&k, n_head / n_kv)?;
    let v = repeat_kv(&v, n_head / n_kv)?;

    let scale = 1f64 / (head_dim as f64).sqrt();
    let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    let att = match mask {
        Some(m) => att.broadcast_add(m)?,
        None => att,
    };
    let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
    let out = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, n_head * head_dim))?;
    w.linear(&out, &format!("{prefix}.o_proj.weight"), None)
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
pub fn swiglu(w: &Weights, prefix: &str, x: &Tensor) -> Result<Tensor> {
    let gate = w.linear(x, &format!("{prefix}.gate_proj.weight"), None)?;
    let up = w.linear(x, &format!("{prefix}.up_proj.weight"), None)?;
    let h = (candle_nn::ops::silu(&gate)? * up)?;
    w.linear(&h, &format!("{prefix}.down_proj.weight"), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

    /// `rotate_half` must be HuggingFace's `cat(-x2, x1)` half-split, NOT the
    /// interleaved `(x[2i], x[2i+1])` pairing the Fish port uses. Both produce
    /// plausible-sounding audio; only one is right for these weights.
    #[test]
    fn rotate_half_is_the_hf_split_not_interleaved() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 1, 1, 6), &dev()).unwrap();
        let got: Vec<f32> = rotate_half(&x).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // half-split: [-4, -5, -6, 1, 2, 3]
        assert_eq!(got, vec![-4., -5., -6., 1., 2., 3.]);
        // the interleaved convention would give [-2, 1, -4, 3, -6, 5]
        assert_ne!(got, vec![-2., 1., -4., 3., -6., 5.]);
    }

    /// cos/sin must duplicate each frequency across `i` and `i + head_dim/2`, which is
    /// what makes the half-split pairing rotate a consistent angle per channel pair.
    #[test]
    fn rope_tables_duplicate_each_frequency_across_the_split() {
        let d = 8usize;
        let (cos, sin) = precompute_rope(4, d, 1_000_000.0, &dev(), DType::F32).unwrap();
        let c: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
        let s: Vec<f32> = sin.flatten_all().unwrap().to_vec1().unwrap();
        for t in 0..4 {
            for i in 0..d / 2 {
                assert_eq!(c[t * d + i], c[t * d + d / 2 + i], "cos t={t} i={i}");
                assert_eq!(s[t * d + i], s[t * d + d / 2 + i], "sin t={t} i={i}");
            }
        }
        // position 0 is the identity rotation
        assert!(c[..d].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!(s[..d].iter().all(|v| v.abs() < 1e-6));
    }

    /// RoPE is a rotation: it must preserve the norm of each head vector.
    #[test]
    fn rope_preserves_norm() {
        let (b, h, t, d) = (1, 2, 5, 8);
        let n = b * h * t * d;
        let v: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) - 3.0).collect();
        let x = Tensor::from_vec(v, (b, h, t, d), &dev()).unwrap();
        let (cos, sin) = precompute_rope(t, d, 1_000_000.0, &dev(), DType::F32).unwrap();
        let y = apply_rope(&x, &cos, &sin).unwrap();
        let nx: f32 = x.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        let ny: f32 = y.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!((nx - ny).abs() / nx < 1e-5, "norm changed: {nx} -> {ny}");
    }

    /// GQA expansion must repeat each kv head contiguously, so head `i` of the expanded
    /// tensor reads kv head `i / n`.
    #[test]
    fn repeat_kv_expands_contiguously() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 1, 2), &dev()).unwrap();
        let y = repeat_kv(&x, 2).unwrap();
        assert_eq!(y.dims(), &[1, 4, 1, 2]);
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![1., 2., 1., 2., 3., 4., 3., 4.]);
    }

    /// The mask must permit exactly the keys at or before each query's absolute
    /// position, including when the query block starts at a cache offset.
    #[test]
    fn causal_mask_respects_the_cache_offset() {
        let m = causal_mask_at(3, 2, &dev(), DType::F32).unwrap();
        assert_eq!(m.dims(), &[2, 5]);
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        // row 0 is absolute position 3 -> keys 0..=3 allowed, key 4 blocked
        assert_eq!(&v[0..4], &[0., 0., 0., 0.]);
        assert!(v[4].is_infinite() && v[4] < 0.0);
        // row 1 is absolute position 4 -> all five keys allowed
        assert!(v[5..10].iter().all(|x| *x == 0.0));
    }

    /// The slab cache must return exactly what was written, across a growth boundary.
    #[test]
    fn kv_cache_returns_what_was_written_across_growth() {
        let (b, nkv, d) = (1, 2, 4);
        let mut c = KvCache::new(1);
        let mut expect: Vec<f32> = Vec::new();
        // push past CACHE_CHUNK so the grow path runs
        for step in 0..(CACHE_CHUNK + 5) {
            let vals: Vec<f32> = (0..b * nkv * d).map(|i| (step * 100 + i) as f32).collect();
            expect.extend(vals.iter().copied());
            let k = Tensor::from_vec(vals.clone(), (b, nkv, 1, d), &dev()).unwrap();
            let (gk, _) = c.append(0, &k, &k).unwrap();
            c.advance(1);
            assert_eq!(gk.dim(2).unwrap(), step + 1, "cache length after step {step}");
        }
        let (gk, _) = {
            let last: Vec<f32> = vec![0.0; b * nkv * d];
            let t = Tensor::from_vec(last, (b, nkv, 1, d), &dev()).unwrap();
            let r = c.append(0, &t, &t).unwrap();
            r
        };
        // element [b, kv, step, d] must equal what step wrote
        let got: Vec<f32> = gk
            .narrow(2, 0, CACHE_CHUNK + 5)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(got, expect, "cache contents diverged");
    }
}
