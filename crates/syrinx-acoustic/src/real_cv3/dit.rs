//! The CV3 CFM estimator: the 22-layer DiT transformer — time embedding, causal
//! conv-position embedding, rotary tables, the `DiTBlock` (AdaLN-Zero attention + AdaLN
//! tanh-GELU FF), the AdaLN-final head, and the rotary/no-affine-LN helpers. Moved
//! verbatim from `real_cv3.rs`.

use super::*;
use candle_core::D;

impl Cv3Flow {
    // ============================ DiT ESTIMATOR ============================

    /// The CV3 CFM estimator: the 22-layer DiT (`decoder.estimator.*`).
    ///
    /// Inputs are the CFG-stacked channel-first tensors `x,mu,cond: [B,80,L]`,
    /// `spks: [B,80]`, `t: [B]`. Returns `[B,80,L]`. Non-streaming / full-context:
    /// the attention mask is all-true, so no masking is applied.
    pub fn estimator(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        // Parity default: the unmasked (full-context) DiT, so existing callers and the
        // frozen `real_cv3_flow_parity` test stay byte-identical.
        self.estimator_masked(x, mu, t, spks, cond, None)
    }

    /// [`Self::estimator`] with an optional chunked-causal attention mask.
    ///
    /// `mask`, if given, is the additive `[1,1,L,L]` mask (built at the mel length `L`)
    /// threaded into every DiT block's self-attention so the 22 transformer blocks are
    /// chunk-causal (the CV3 `DiT.forward(streaming=True)` path); `None` reproduces the
    /// full-context batch path exactly.
    pub fn estimator_masked(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let p = "decoder.estimator";
        let (b, _c, l) = x.dims3()?;

        // DiT.forward: move to [B, L, C]
        let xx = x.transpose(1, 2)?.contiguous()?; // [B,L,80]
        let muu = mu.transpose(1, 2)?.contiguous()?; // [B,L,80]
        let condd = cond.transpose(1, 2)?.contiguous()?; // [B,L,80]

        // time embedding -> [B, 1024]
        let temb = self.time_embed(t, p)?;

        // InputEmbedding: proj(cat[x, cond, mu, spks_broadcast]) then conv_pos_embed
        // residual.
        let spk_b = spks.unsqueeze(1)?.broadcast_as((b, l, MEL))?.contiguous()?; // [B,L,80]
        let cat = Tensor::cat(&[&xx, &condd, &muu, &spk_b], 2)?; // [B,L,320]
        debug_assert_eq!(cat.dim(2)?, PROJ_IN);
        let mut h = self.linear(
            &cat,
            &format!("{p}.input_embed.proj.weight"),
            Some(&format!("{p}.input_embed.proj.bias")),
        )?; // [B,L,1024]
        let cpe = self.conv_pos_embed(&h, &format!("{p}.input_embed.conv_pos_embed"))?;
        h = (cpe + h)?;

        // rotary table for this sequence length (loaded inv_freq).
        let (rope_cos, rope_sin) = self.rope_tables(l, p)?; // [1,L,64] each

        // 22 DiT blocks.
        for i in 0..DIT_DEPTH {
            h = self.dit_block(&h, &temb, &rope_cos, &rope_sin, mask, &format!("{p}.transformer_blocks.{i}"))?;
        }

        // AdaLayerNormZero_Final + proj_out.
        h = self.adaln_final(&h, &temb, &format!("{p}.norm_out"))?; // [B,L,1024]
        let out = self.linear(&h, &format!("{p}.proj_out.weight"), Some(&format!("{p}.proj_out.bias")))?; // [B,L,80]
        out.transpose(1, 2)?.contiguous() // [B,80,L]
    }

    /// `TimestepEmbedding`: `SinusPositionEmbedding(256, scale=1000)` then
    /// `Linear(256->1024) -> SiLU -> Linear(1024->1024)`. `t: [B] -> [B,1024]`.
    fn time_embed(&self, t: &Tensor, p: &str) -> Result<Tensor> {
        let half = DIT_FREQ_DIM / 2; // 128
        let emb_scale = (10000f64.ln()) / (half - 1) as f64;
        let inv: Vec<f32> = (0..half).map(|i| (-(i as f64) * emb_scale).exp() as f32).collect();
        let inv = Tensor::from_vec(inv, (half,), &self.dev)?; // [128]
        let tt = (t * 1000.0)?.unsqueeze(1)?; // [B,1]
        let freqs = tt.broadcast_mul(&inv.unsqueeze(0)?)?; // [B,128]
        let emb = Tensor::cat(&[&freqs.sin()?, &freqs.cos()?], D::Minus1)?; // [B,256]
        let h = self.linear(
            &emb,
            &format!("{p}.time_embed.time_mlp.0.weight"),
            Some(&format!("{p}.time_embed.time_mlp.0.bias")),
        )?;
        let h = silu(&h)?;
        self.linear(
            &h,
            &format!("{p}.time_embed.time_mlp.2.weight"),
            Some(&format!("{p}.time_embed.time_mlp.2.bias")),
        )
    }

    /// `CausalConvPositionEmbedding`: permute -> padL(k-1) -> grouped conv1 (k=31,
    /// groups=16) -> Mish -> padL(k-1) -> grouped conv2 -> Mish -> permute back.
    /// Input/out `[B, L, 1024]`.
    fn conv_pos_embed(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?; // [B,1024,L]
        let pad = CONV_POS_K - 1; // 30, padded on the LEFT (causal)
        let h = pad_time(&xt, pad, 0)?;
        let h = conv1d(
            &h,
            &self.g(&format!("{p}.conv1.0.weight"))?,
            Some(&self.g(&format!("{p}.conv1.0.bias"))?),
            1,
            CONV_POS_GROUPS,
        )?;
        let h = mish(&h)?;
        let h = pad_time(&h, pad, 0)?;
        let h = conv1d(
            &h,
            &self.g(&format!("{p}.conv2.0.weight"))?,
            Some(&self.g(&format!("{p}.conv2.0.bias"))?),
            1,
            CONV_POS_GROUPS,
        )?;
        let h = mish(&h)?;
        h.transpose(1, 2)?.contiguous() // [B,L,1024]
    }

    /// Build the rotary cos/sin tables `[1, L, 64]` for `forward_from_seq_len(L)`.
    ///
    /// `inv_freq[i] = 1/(10000^(2i/64))` (loaded from the checkpoint, i in 0..32). The
    /// reference interleaves each frequency twice: `freqs[p] = [θ0,θ0,θ1,θ1,...]` with
    /// `θ_i = p * inv_freq[i]`, so adjacent channel pairs share an angle (GPT-J style).
    fn rope_tables(&self, l: usize, p: &str) -> Result<(Tensor, Tensor)> {
        let inv = self.g(&format!("{p}.rotary_embed.inv_freq"))?; // [32]
        let inv: Vec<f32> = inv.to_dtype(DType::F32)?.to_vec1()?;
        let half = inv.len(); // 32
        let mut cos = vec![0f32; l * DIT_HEAD_DIM];
        let mut sin = vec![0f32; l * DIT_HEAD_DIM];
        for pos in 0..l {
            for i in 0..half {
                let theta = pos as f32 * inv[i];
                let (s, c) = (theta.sin(), theta.cos());
                cos[pos * DIT_HEAD_DIM + 2 * i] = c;
                cos[pos * DIT_HEAD_DIM + 2 * i + 1] = c;
                sin[pos * DIT_HEAD_DIM + 2 * i] = s;
                sin[pos * DIT_HEAD_DIM + 2 * i + 1] = s;
            }
        }
        let cos = Tensor::from_vec(cos, (1, l, DIT_HEAD_DIM), &self.dev)?;
        let sin = Tensor::from_vec(sin, (1, l, DIT_HEAD_DIM), &self.dev)?;
        Ok((cos, sin))
    }

    /// One `DiTBlock`: AdaLN-Zero modulated self-attention + AdaLN-modulated tanh-GELU
    /// FF, both gated by the time embedding `t`.
    fn dit_block(
        &self,
        x: &Tensor,
        temb: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: Option<&Tensor>,
        p: &str,
    ) -> Result<Tensor> {
        // attn_norm = AdaLayerNormZero: returns modulated `norm` + the 4 mlp/gate params.
        let (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.adaln_zero(x, temb, &format!("{p}.attn_norm"))?;
        let attn = self.dit_attn(&norm, rope_cos, rope_sin, mask, &format!("{p}.attn"))?;
        // x = x + gate_msa[:,None] * attn
        let x = (x + gate_msa.unsqueeze(1)?.broadcast_mul(&attn)?)?;

        // ff_norm (LayerNorm no-affine eps 1e-6) * (1 + scale_mlp[:,None]) + shift_mlp[:,None]
        let ffn = ln_noaffine(&x, LN_EPS_DIT)?;
        let one_plus = (scale_mlp.unsqueeze(1)? + 1.0)?; // [B,1,1024]
        let ffn = (ffn.broadcast_mul(&one_plus)?.broadcast_add(&shift_mlp.unsqueeze(1)?))?;
        let ff = self.dit_ff(&ffn, &format!("{p}.ff"))?;
        // x = x + gate_mlp[:,None] * ff
        x + gate_mlp.unsqueeze(1)?.broadcast_mul(&ff)?
    }

    /// `AdaLayerNormZero`: `emb = linear(silu(t))` (1024 -> 6144), chunk into
    /// `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`; returns the
    /// modulated, no-affine-LN'd `x` plus `(gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
    fn adaln_zero(
        &self,
        x: &Tensor,
        temb: &Tensor,
        p: &str,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let e = silu(temb)?;
        let e = self.linear(&e, &format!("{p}.linear.weight"), Some(&format!("{p}.linear.bias")))?; // [B,6144]
        let shift_msa = e.narrow(1, 0, DIT_DIM)?;
        let scale_msa = e.narrow(1, DIT_DIM, DIT_DIM)?;
        let gate_msa = e.narrow(1, 2 * DIT_DIM, DIT_DIM)?;
        let shift_mlp = e.narrow(1, 3 * DIT_DIM, DIT_DIM)?;
        let scale_mlp = e.narrow(1, 4 * DIT_DIM, DIT_DIM)?;
        let gate_mlp = e.narrow(1, 5 * DIT_DIM, DIT_DIM)?;
        // norm(x) * (1 + scale_msa[:,None]) + shift_msa[:,None]
        let xn = ln_noaffine(x, LN_EPS_DIT)?;
        let one_plus = (scale_msa.unsqueeze(1)? + 1.0)?;
        let norm = xn.broadcast_mul(&one_plus)?.broadcast_add(&shift_msa.unsqueeze(1)?)?;
        Ok((norm, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }

    /// `AdaLayerNormZero_Final`: `emb = linear(silu(t))` (1024 -> 2048) chunked into
    /// `(scale, shift)`; `x = norm(x) * (1 + scale)[:,None,:] + shift[:,None,:]`.
    fn adaln_final(&self, x: &Tensor, temb: &Tensor, p: &str) -> Result<Tensor> {
        let e = silu(temb)?;
        let e = self.linear(&e, &format!("{p}.linear.weight"), Some(&format!("{p}.linear.bias")))?; // [B,2048]
        let scale = e.narrow(1, 0, DIT_DIM)?;
        let shift = e.narrow(1, DIT_DIM, DIT_DIM)?;
        let xn = ln_noaffine(x, LN_EPS_DIT)?;
        let one_plus = (scale.unsqueeze(1)? + 1.0)?;
        xn.broadcast_mul(&one_plus)?.broadcast_add(&shift.unsqueeze(1)?)
    }

    /// DiT self-attention (`AttnProcessor`). Heads=16, head_dim=64, scale=1/sqrt(64).
    ///
    /// IMPORTANT — faithful to the reference (empirically confirmed on box): rotary is
    /// applied to the **full `[B,N,1024]` projection BEFORE the head reshape**, with a
    /// 64-wide freqs table (`rot_dim=64`). So only channels `[0:64]` are rotated (== head
    /// 0 after reshape); channels `[64:1024]` pass through unrotated. (CV3's
    /// `AttnProcessor` rotates pre-view, unlike standard F5-TTS which rotates post-view
    /// per head.) Instrumenting the real `apply_rotary_pos_emb` confirmed exactly
    /// **64/1024 query channels change** (query ndim=3, rot_dim=64), and `use_xpos=False`
    /// so the xpos scale is the identity 1.0.
    fn dit_attn(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        mask: Option<&Tensor>,
        p: &str,
    ) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let q = self.linear(x, &format!("{p}.to_q.weight"), Some(&format!("{p}.to_q.bias")))?; // [B,N,1024]
        let k = self.linear(x, &format!("{p}.to_k.weight"), Some(&format!("{p}.to_k.bias")))?;
        let v = self.linear(x, &format!("{p}.to_v.weight"), Some(&format!("{p}.to_v.bias")))?;
        // rope on the first 64 channels of q/k (pre-reshape).
        let q = apply_rope_first64(&q, rope_cos, rope_sin)?;
        let k = apply_rope_first64(&k, rope_cos, rope_sin)?;

        let h = DIT_HEADS;
        let dk = DIT_HEAD_DIM;
        let q = q.reshape((b, n, h, dk))?.transpose(1, 2)?.contiguous()?; // [B,H,N,dk]
        let k = k.reshape((b, n, h, dk))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, n, h, dk))?.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (dk as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [B,H,N,N]
        // Chunked-causal mask (streaming): the additive `[1,1,N,N]` 0/-inf mask is added to
        // the scores before softmax so a finalized frame never attends to the future — the
        // CV3 `DiT.forward(streaming=True)` `add_optional_chunk_mask` path. `None` ⇒ the
        // full-context batch path (byte-unchanged).
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = softmax_last(&scores)?;
        let ctx = probs.matmul(&v)?; // [B,H,N,dk]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, n, h * dk))?;
        // to_out.0 (Linear w/ bias); to_out.1 is dropout (off).
        self.linear(&ctx, &format!("{p}.to_out.0.weight"), Some(&format!("{p}.to_out.0.bias")))
    }

    /// DiT `FeedForward`: `Linear(1024->2048) -> GELU(tanh) -> Linear(2048->1024)`.
    fn dit_ff(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let h = self.linear(x, &format!("{p}.ff.0.0.weight"), Some(&format!("{p}.ff.0.0.bias")))?;
        let h = gelu_tanh(&h)?;
        self.linear(&h, &format!("{p}.ff.2.weight"), Some(&format!("{p}.ff.2.bias")))
    }
}

/// LayerNorm over the last dim with NO affine params (`elementwise_affine=False`).
fn ln_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// Apply interleaved (GPT-J style) rotary to the **first 64 channels** of `x`
/// `[B,N,C]` (C >= 64), leaving channels `[64:]` unchanged. `cos`/`sin` are
/// `[1,N,64]` and broadcast over the batch.
///
/// `rot[..2i]   = a_{2i}*cos_i - a_{2i+1}*sin_i`,
/// `rot[..2i+1] = a_{2i+1}*cos_i + a_{2i}*sin_i` — produced by `x*cos +
/// rotate_half(x)*sin`, with `rotate_half([a0,a1,a2,a3,..]) = [-a1,a0,-a3,a2,..]`.
fn apply_rope_first64(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let c = x.dim(2)?;
    debug_assert!(c >= DIT_HEAD_DIM);
    let x_rot = x.narrow(2, 0, DIT_HEAD_DIM)?.contiguous()?; // [B,N,64]
    let rh = rotate_half(&x_rot)?; // [B,N,64]
    let rotated = (x_rot.broadcast_mul(cos)? + rh.broadcast_mul(sin)?)?; // [B,N,64]
    if c > DIT_HEAD_DIM {
        let x_pass = x.narrow(2, DIT_HEAD_DIM, c - DIT_HEAD_DIM)?.contiguous()?;
        Tensor::cat(&[&rotated, &x_pass], 2)?.contiguous()
    } else {
        Ok(rotated)
    }
}

/// `rotate_half([a0,a1,a2,a3,..]) = [-a1,a0,-a3,a2,..]` (pairs adjacent channels).
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let (b, n, c) = x.dims3()?;
    let r = x.reshape((b, n, c / 2, 2))?;
    let x1 = r.narrow(3, 0, 1)?; // even channels  [B,N,c/2,1]
    let x2 = r.narrow(3, 1, 1)?; // odd channels
    let stacked = Tensor::cat(&[&x2.neg()?, &x1], 3)?; // [-odd, even] -> [B,N,c/2,2]
    stacked.reshape((b, n, c))
}

fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
}

fn mish(x: &Tensor) -> Result<Tensor> {
    // x * tanh(softplus(x)); softplus = ln(1 + exp(x)).
    let sp = (x.exp()? + 1.0)?.log()?;
    x * sp.tanh()?
}

/// `nn.GELU(approximate="tanh")` — candle's `gelu` is the tanh approximation.
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(x, D::Minus1)
}
