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

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

const HIDDEN: usize = 896;
const N_LAYERS: usize = 24;
const N_HEADS: usize = 14;
const N_KV: usize = 2;
const HEAD_DIM: usize = 64;
const EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

/// The real Qwen2-0.5B LM + CosyVoice2 `llm_decoder`, loaded from fp32 safetensors.
pub struct Qwen2Lm {
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl Qwen2Lm {
    /// Load the converted fp32 checkpoint (`llm_fp32.safetensors`) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        // Normalise to f32 so the forward is a clean fp32 reference match.
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self { w, dev })
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
    fn linear(&self, x: &Tensor, wname: &str, bias: Option<&str>) -> Result<Tensor> {
        let w = self.g(wname)?;
        let y = x.broadcast_matmul(&w.t()?)?;
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
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), dev)?;
    let pos: Vec<f32> = (0..t).map(|i| i as f32).collect();
    let pos = Tensor::from_vec(pos, (t,), dev)?;
    let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [t, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [t, head_dim]
    Ok((emb.cos()?, emb.sin()?))
}

/// Additive causal mask `[t, t]`: 0 on/below the diagonal, -inf above.
fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (t, t), dev)
}
