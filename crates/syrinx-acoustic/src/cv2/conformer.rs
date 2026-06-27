//! `UpsampleConformerEncoder` — the CV2 flow's conformer-style encoder: linear
//! subsampling, espnet relative positional encoding, the pre-lookahead layer, the 2×
//! upsample, the conformer layers and their rel-pos multi-headed attention. Moved
//! verbatim from `real.rs`.

use super::*;

impl Flow {
    // ============================ ENCODER ============================

    /// `UpsampleConformerEncoder.forward` (streaming=False, full context, no mask).
    ///
    /// The parity default: a thin pass-through to [`Self::encoder_masked`] with no chunk
    /// masks, so existing callers (and the frozen parity tests) keep the unmasked
    /// full-context behavior byte-for-byte.
    pub fn encoder(&self, emb: &Tensor) -> Result<Tensor> {
        self.encoder_masked(emb, None, None)
    }

    /// `UpsampleConformerEncoder.forward` with optional chunked-causal attention masks.
    ///
    /// `m1` masks the **first-stage** conformer layers (token length `T`); `m2` masks the
    /// **up-stage** layers (post-2×-upsample length `2T`). Both are additive `[1,1,L,L]`
    /// masks (see [`add_optional_chunk_mask`]); `None` ⇒ that stage is unmasked. The two
    /// stages have different sequence lengths (the upsample 2×'s it), so they need two
    /// separate masks built at their own lengths — a single mask cannot serve both.
    pub fn encoder_masked(
        &self,
        emb: &Tensor,
        m1: Option<&Tensor>,
        m2: Option<&Tensor>,
    ) -> Result<Tensor> {
        // embed: Linear -> LayerNorm(1e-5); pos enc multiplies x by sqrt(d_model)
        let mut xs = self.subsample(emb, "encoder.embed")?; // [1, T, 512]
        let t = xs.dim(1)?;
        let pos = self.rel_pos_emb(t)?; // [1, 2T-1, 512]
        xs = (xs * (ENC_DIM as f64).sqrt())?;

        // pre-lookahead layer
        xs = self.pre_lookahead(&xs)?;

        // first-stage conformer layers (mask m1, at token length T)
        for l in 0..N_ENC {
            xs = self.conformer_layer(&xs, &pos, m1, &format!("encoder.encoders.{l}"))?;
        }

        // upsample: transpose to [1,512,T], Upsample1D stride 2, transpose back
        let xc = xs.transpose(1, 2)?.contiguous()?; // [1,512,T]
        let up = self.upsample1d(&xc)?; // [1,512,2T]
        xs = up.transpose(1, 2)?.contiguous()?; // [1,2T,512]

        // up_embed + pos for the new (2T) length
        xs = self.subsample(&xs, "encoder.up_embed")?;
        let t2 = xs.dim(1)?;
        let pos2 = self.rel_pos_emb(t2)?;
        xs = (xs * (ENC_DIM as f64).sqrt())?;

        // up-stage conformer layers (mask m2, at post-upsample length 2T)
        for l in 0..N_UPENC {
            xs = self.conformer_layer(&xs, &pos2, m2, &format!("encoder.up_encoders.{l}"))?;
        }

        // after_norm
        self.layer_norm(&xs, "encoder.after_norm.weight", "encoder.after_norm.bias", LN_EPS)
    }

    /// LinearNoSubsampling: Linear(idim->odim) then LayerNorm(eps 1e-5). (dropout off)
    fn subsample(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let y = self.linear(x, &format!("{prefix}.out.0.weight"), Some(&format!("{prefix}.out.0.bias")))?;
        self.layer_norm(&y, &format!("{prefix}.out.1.weight"), &format!("{prefix}.out.1.bias"), LN_EPS)
    }

    /// EspnetRelPositionalEncoding position table for sequence length `t`:
    /// returns `[1, 2t-1, d_model]` (positive part reversed ++ negative part).
    fn rel_pos_emb(&self, t: usize) -> Result<Tensor> {
        let d = ENC_DIM;
        let half = d / 2;
        // div_term[i] = exp(2i * -(ln10000/d)) for i in 0..half
        let scale = -(10000f64.ln()) / d as f64;
        let div: Vec<f32> = (0..half).map(|i| ((2 * i) as f64 * scale).exp() as f32).collect();
        // pe_positive / pe_negative rows for positions 0..t
        let mut pos = vec![0f32; t * d];
        let mut neg = vec![0f32; t * d];
        for p in 0..t {
            for i in 0..half {
                let ap = p as f32 * div[i];
                pos[p * d + 2 * i] = ap.sin();
                pos[p * d + 2 * i + 1] = ap.cos();
                neg[p * d + 2 * i] = (-ap).sin();
                neg[p * d + 2 * i + 1] = (-ap).cos();
            }
        }
        // pe_positive = flip(pos, dim0); pe_negative = neg[1:]; pe = cat -> [2t-1, d]
        let mut pe = vec![0f32; (2 * t - 1) * d];
        for r in 0..t {
            // reversed positive: row r of pe is pos row (t-1-r)
            let src = (t - 1 - r) * d;
            pe[r * d..r * d + d].copy_from_slice(&pos[src..src + d]);
        }
        // negative part neg[1..t] -> output rows t..2t-1
        for r in 1..t {
            let out_row = (t - 1) + r; // rows t .. 2t-2
            let src = r * d;
            pe[out_row * d..out_row * d + d].copy_from_slice(&neg[src..src + d]);
        }
        let pe = Tensor::from_vec(pe, (2 * t - 1, d), &self.dev)?;
        // position_encoding(offset=0, size=t): slice center +- t
        // pe full length is 2t-1; center = (2t-1)//2 = t-1.
        // slice [center - t + 1 : center + t] = [0 : 2t-1] -> the whole thing.
        pe.unsqueeze(0) // [1, 2t-1, d]
    }

    /// PreLookaheadLayer (inference, no context): pad right `pre_lookahead_len`,
    /// conv1 (k=pre+1) + leaky_relu(0.01), pad left k2-1, conv2 (k=3), + residual.
    fn pre_lookahead(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?; // [1, 512, T]
        // pad right by PRE_LOOKAHEAD with zeros
        let padded = pad_time(&xt, 0, PRE_LOOKAHEAD)?;
        let c1 = conv1d(&padded, &self.g("encoder.pre_lookahead_layer.conv1.weight")?,
                        Some(&self.g("encoder.pre_lookahead_layer.conv1.bias")?), 1)?;
        let c1 = leaky_relu(&c1, 0.01)?;
        // pad left by (kernel2 - 1) = 2
        let padded2 = pad_time(&c1, 2, 0)?;
        let c2 = conv1d(&padded2, &self.g("encoder.pre_lookahead_layer.conv2.weight")?,
                        Some(&self.g("encoder.pre_lookahead_layer.conv2.bias")?), 1)?;
        let out = c2.transpose(1, 2)?.contiguous()?; // [1, T, 512]
        out + x
    }

    /// Upsample1D (encoder up_layer, stride 2): nearest interpolate x2 along time,
    /// pad left by stride*2 = 4, conv k=5 stride 1.
    fn upsample1d(&self, x: &Tensor) -> Result<Tensor> {
        let up = upsample_nearest_time(x, 2)?; // [1,512,2T]
        let padded = pad_time(&up, 4, 0)?;
        conv1d(&padded, &self.g("encoder.up_layer.conv.weight")?,
               Some(&self.g("encoder.up_layer.conv.bias")?), 1)
    }

    /// One ConformerEncoderLayer (no macaron, no cnn module), normalize_before:
    /// x = x + selfattn(norm_mha(x)); x = x + ffn(norm_ff(x)).
    fn conformer_layer(&self, x: &Tensor, pos: &Tensor, mask: Option<&Tensor>, p: &str) -> Result<Tensor> {
        let res = x.clone();
        let xn = self.layer_norm(x, &format!("{p}.norm_mha.weight"), &format!("{p}.norm_mha.bias"), LN_EPS_CONF)?;
        let att = self.rel_self_attn(&xn, pos, mask, &format!("{p}.self_attn"))?;
        let x = (res + att)?;
        let res = x.clone();
        let xn = self.layer_norm(&x, &format!("{p}.norm_ff.weight"), &format!("{p}.norm_ff.bias"), LN_EPS_CONF)?;
        let ff = self.conformer_ffn(&xn, &format!("{p}.feed_forward"))?;
        res + ff
    }

    /// PositionwiseFeedForward: w_2(silu(w_1(x))). (swish == silu)
    fn conformer_ffn(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let h = self.linear(x, &format!("{p}.w_1.weight"), Some(&format!("{p}.w_1.bias")))?;
        let h = silu(&h)?;
        self.linear(&h, &format!("{p}.w_2.weight"), Some(&format!("{p}.w_2.bias")))
    }

    /// RelPositionMultiHeadedAttention (espnet rel-pos). `mask`, if given, is an additive
    /// `[1,1,t,t]` chunk-causal mask added to the scores before softmax (full context when
    /// `None`).
    fn rel_self_attn(&self, x: &Tensor, pos: &Tensor, mask: Option<&Tensor>, p: &str) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = ENC_HEADS;
        let dk = ENC_HEAD_DIM;
        // q,k,v: linear then [b, h, t, dk]
        let q = self.linear(x, &format!("{p}.linear_q.weight"), Some(&format!("{p}.linear_q.bias")))?;
        let k = self.linear(x, &format!("{p}.linear_k.weight"), Some(&format!("{p}.linear_k.bias")))?;
        let v = self.linear(x, &format!("{p}.linear_v.weight"), Some(&format!("{p}.linear_v.bias")))?;
        let q = q.reshape((b, t, h, dk))?; // [b,t,h,dk]
        let k = k.reshape((b, t, h, dk))?.transpose(1, 2)?.contiguous()?; // [b,h,t,dk]
        let v = v.reshape((b, t, h, dk))?.transpose(1, 2)?.contiguous()?; // [b,h,t,dk]

        // p projection of pos_emb: linear_pos (no bias), [1, 2t-1, d] -> [1, h, 2t-1, dk]
        let pt = pos.dim(1)?;
        let p_proj = self.linear(pos, &format!("{p}.linear_pos.weight"), None)?;
        let p_proj = p_proj.reshape((1, pt, h, dk))?.transpose(1, 2)?.contiguous()?; // [1,h,2t-1,dk]

        // pos_bias_u/v: [h, dk]
        let bias_u = self.g(&format!("{p}.pos_bias_u"))?.reshape((1, h, 1, dk))?;
        let bias_v = self.g(&format!("{p}.pos_bias_v"))?.reshape((1, h, 1, dk))?;
        // q is [b,t,h,dk]; (q + bias).transpose(1,2) -> [b,h,t,dk]
        let qh = q.transpose(1, 2)?.contiguous()?; // [b,h,t,dk]
        let q_u = qh.broadcast_add(&bias_u)?; // [b,h,t,dk]
        let q_v = qh.broadcast_add(&bias_v)?;

        // matrix_ac = q_u @ k^T : [b,h,t,t]
        let ac = q_u.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        // matrix_bd = q_v @ p^T : [b,h,t,2t-1], then rel_shift -> [b,h,t,t]
        let p_bcast = p_proj.broadcast_as((b, h, pt, dk))?.contiguous()?;
        let bd = q_v.matmul(&p_bcast.transpose(2, 3)?.contiguous()?)?; // [b,h,t,2t-1]
        let bd = rel_shift(&bd, t)?; // [b,h,t,t]
        let scale = 1.0 / (dk as f64).sqrt();
        let scores = ((ac + bd)? * scale)?; // [b,h,t,t]
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = softmax_last(&scores)?;
        let ctx = probs.matmul(&v)?; // [b,h,t,dk]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, h * dk))?;
        self.linear(&ctx, &format!("{p}.linear_out.weight"), Some(&format!("{p}.linear_out.bias")))
    }
}

/// Nearest-neighbour upsample along the time (last) dim by integer `factor`.
///
/// Implemented as an `index_select` over the gather indices `[0,0,..,1,1,..]`
/// (each input position repeated `factor` times, in order) rather than Candle's
/// `upsample_nearest1d`. The two are numerically **identical** for an integer
/// factor — nearest-neighbour upsample *is* this exact integer repeat — but
/// `upsample_nearest1d` has no CUDA kernel in candle 0.8 ("upsample-nearest1d is
/// not supported on cuda"), whereas `index_select` runs on both CPU and CUDA.
/// So this keeps the CPU result bit-for-bit unchanged while letting the flow run
/// on GPU.
fn upsample_nearest_time(x: &Tensor, factor: usize) -> Result<Tensor> {
    let t = x.dim(2)?;
    // gather indices: position p maps to inputs [p,p,..] (factor copies), in order.
    let mut idx: Vec<u32> = Vec::with_capacity(t * factor);
    for p in 0..t {
        for _ in 0..factor {
            idx.push(p as u32);
        }
    }
    let idx = Tensor::from_vec(idx, (t * factor,), x.device())?;
    x.index_select(&idx, 2)
}

fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    // max(x, 0) + slope * min(x, 0)
    let pos = x.relu()?;
    let neg = (x - &pos)?; // == min(x,0)
    pos + (neg * slope)?
}

/// espnet rel_shift: input `[b,h,t,2t-1]` -> `[b,h,t,t]`.
fn rel_shift(x: &Tensor, t: usize) -> Result<Tensor> {
    let (b, h, tq, n) = x.dims4()?; // n = 2t-1
    debug_assert_eq!(tq, t);
    // zero pad one column on the left along last dim
    let zpad = Tensor::zeros((b, h, tq, 1), x.dtype(), x.device())?;
    let xp = Tensor::cat(&[&zpad, x], D::Minus1)?; // [b,h,t,n+1]
    // view as [b,h,n+1,t]
    let xp = xp.reshape((b, h, n + 1, tq))?;
    // drop first row along dim2, view back as [b,h,t,n], keep first t cols
    let xp = xp.narrow(2, 1, n)?.reshape((b, h, tq, n))?;
    xp.narrow(D::Minus1, 0, n / 2 + 1)
}
