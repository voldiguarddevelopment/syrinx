//! The CV2 CFM U-Net estimator (`CausalConditionalDecoder`): time embedding, causal
//! resnet/conv blocks, the diffusers `BasicTransformerBlock` stack and its self-attention
//! + gelu feed-forward. Moved verbatim from `real.rs`.

use super::*;

impl Flow {
    /// CausalConditionalDecoder.forward (the estimator), streaming=False, no mask.
    /// Inputs are the CFG-stacked `[2, .., L]` tensors. Returns `[2, 80, L]`.
    ///
    /// Parity default: a thin pass-through to [`Self::estimator_masked`] with no chunk
    /// mask, so existing callers (and the frozen parity tests) stay byte-identical.
    pub fn estimator(&self, x: &Tensor, mu: &Tensor, t: &Tensor, spks: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.estimator_masked(x, mu, t, spks, cond, None)
    }

    /// CausalConditionalDecoder.forward with an optional chunked-causal attention mask.
    ///
    /// `mask`, if given, is the additive `[1,1,L,L]` mask (built at the mel length `L`)
    /// threaded into every `transformer_stack` so the U-Net's down/mid/up self-attention
    /// is chunk-causal; `None` reproduces the full-context path exactly.
    pub fn estimator_masked(&self, x: &Tensor, mu: &Tensor, t: &Tensor, spks: &Tensor, cond: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let l = x.dim(2)?;
        // time embedding: SinusoidalPosEmb(in_channels=320) then time_mlp
        let temb = self.time_embed(t)?; // [2, 1024]

        // pack x = cat([x, mu, spks_broadcast, cond], dim=1) -> [2, 320, L]
        let spks_b = spks.unsqueeze(2)?.broadcast_as((spks.dim(0)?, MEL, l))?.contiguous()?;
        let mut h = Tensor::cat(&[x, mu, &spks_b, cond], 1)?; // [2,320,L]

        let mut hiddens: Vec<Tensor> = Vec::new();
        // down block (channels=[256] -> single down, is_last, downsample=CausalConv1d k=3)
        h = self.causal_resnet(&h, &temb, "decoder.estimator.down_blocks.0.0")?; // [2,256,L]
        h = self.transformer_stack(&h, &temb, mask, "decoder.estimator.down_blocks.0.1")?;
        hiddens.push(h.clone());
        // downsample: CausalConv1d (pad left 2, k=3, stride 1) -> same length
        h = self.causal_conv(&h, "decoder.estimator.down_blocks.0.2", 3)?;

        // mid blocks
        for m in 0..N_MID {
            h = self.causal_resnet(&h, &temb, &format!("decoder.estimator.mid_blocks.{m}.0"))?;
            h = self.transformer_stack(&h, &temb, mask, &format!("decoder.estimator.mid_blocks.{m}.1"))?;
        }

        // up block (single, is_last, upsample=CausalConv1d k=3). input is cat(x, skip)=512
        let skip = hiddens.pop().unwrap();
        let cat = Tensor::cat(&[&h, &skip], 1)?; // [2,512,L]
        h = self.causal_resnet(&cat, &temb, "decoder.estimator.up_blocks.0.0")?; // [2,256,L]
        h = self.transformer_stack(&h, &temb, mask, "decoder.estimator.up_blocks.0.1")?;
        h = self.causal_conv(&h, "decoder.estimator.up_blocks.0.2", 3)?;

        // final_block (CausalBlock1D) + final_proj (Conv1d k=1)
        h = self.causal_block(&h, "decoder.estimator.final_block")?;
        self.conv1x1(&h, "decoder.estimator.final_proj")
    }

    /// SinusoidalPosEmb(dim=320, scale=1000) + TimestepEmbedding (silu).
    fn time_embed(&self, t: &Tensor) -> Result<Tensor> {
        let dim = 320usize;
        let half = dim / 2;
        let emb_scale = (10000f64.ln()) / (half - 1) as f64;
        let inv: Vec<f32> = (0..half).map(|i| (-(i as f64) * emb_scale).exp() as f32).collect();
        let inv = Tensor::from_vec(inv, (half,), &self.dev)?; // [160]
        // emb = scale * t[:,None] * inv[None,:] ; t is [2]
        let tt = (t * 1000.0)?.unsqueeze(1)?; // [2,1]
        let freqs = tt.broadcast_mul(&inv.unsqueeze(0)?)?; // [2,160]
        let emb = Tensor::cat(&[&freqs.sin()?, &freqs.cos()?], D::Minus1)?; // [2,320]
        // time_mlp: linear_1 -> silu -> linear_2
        let h = self.linear(&emb, "decoder.estimator.time_mlp.linear_1.weight",
                            Some("decoder.estimator.time_mlp.linear_1.bias"))?;
        let h = silu(&h)?;
        self.linear(&h, "decoder.estimator.time_mlp.linear_2.weight",
                    Some("decoder.estimator.time_mlp.linear_2.bias"))
    }

    /// CausalResnetBlock1D: block1 + mlp(time) ; block2 ; + res_conv(x).
    /// (mask is all-ones so omitted.)
    fn causal_resnet(&self, x: &Tensor, temb: &Tensor, p: &str) -> Result<Tensor> {
        let h = self.causal_block(x, &format!("{p}.block1"))?;
        // mlp: Mish then Linear(time_emb_dim -> dim_out); add as [.., dim_out, 1]
        let m = mish(temb)?;
        let m = self.linear(&m, &format!("{p}.mlp.1.weight"), Some(&format!("{p}.mlp.1.bias")))?; // [2,dim_out]
        let h = h.broadcast_add(&m.unsqueeze(2)?)?;
        let h = self.causal_block(&h, &format!("{p}.block2"))?;
        let res = self.conv1x1(x, &format!("{p}.res_conv"))?;
        h + res
    }

    /// CausalBlock1D: CausalConv1d(k=3) -> LayerNorm(over channels) -> Mish.
    /// The norm is a Transpose->LayerNorm->Transpose (LN over the channel dim).
    fn causal_block(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let c = self.causal_conv(x, &format!("{p}.block.0"), 3)?; // [b, dim_out, L]
        // LayerNorm over channels: transpose to [b, L, dim_out], LN, transpose back
        let xt = c.transpose(1, 2)?.contiguous()?;
        let xn = self.layer_norm(&xt, &format!("{p}.block.2.weight"), &format!("{p}.block.2.bias"), LN_EPS)?;
        let xn = xn.transpose(1, 2)?.contiguous()?;
        mish(&xn)
    }

    /// CausalConv1d: left-pad time by (k-1), conv1d stride 1, valid.
    fn causal_conv(&self, x: &Tensor, p: &str, k: usize) -> Result<Tensor> {
        let w = self.g(&format!("{p}.weight"))?;
        let b = self.g(&format!("{p}.bias"))?;
        let padded = pad_time(x, k - 1, 0)?;
        conv1d(&padded, &w, Some(&b), 1)
    }

    /// Conv1d kernel-size 1 (== linear over channels). weight `[out,in,1]`.
    fn conv1x1(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let w = self.g(&format!("{p}.weight"))?; // [out,in,1]
        let b = self.g(&format!("{p}.bias"))?;
        let w2 = w.squeeze(2)?; // [out,in]
        // x: [b,in,L] -> [b,L,in] @ [in,out] -> [b,L,out] -> [b,out,L]
        let xt = x.transpose(1, 2)?.contiguous()?;
        let y = xt.broadcast_matmul(&w2.t()?)?.broadcast_add(&b)?;
        y.transpose(1, 2)?.contiguous()
    }

    /// N_TB BasicTransformerBlocks. Input/out are channel-first `[b, c, L]`;
    /// rearranged to `[b, L, c]` internally.
    fn transformer_stack(&self, x: &Tensor, temb: &Tensor, mask: Option<&Tensor>, p: &str) -> Result<Tensor> {
        let _ = temb; // norm here is plain LayerNorm (no ada-norm); timestep unused
        let mut h = x.transpose(1, 2)?.contiguous()?; // [b, L, c]
        for i in 0..N_TB {
            h = self.basic_transformer_block(&h, mask, &format!("{p}.{i}"))?;
        }
        h.transpose(1, 2)?.contiguous()
    }

    /// diffusers BasicTransformerBlock (self-attn only, layer_norm, gelu FF).
    fn basic_transformer_block(&self, x: &Tensor, mask: Option<&Tensor>, p: &str) -> Result<Tensor> {
        // 1. self-attn: norm1 -> attn1 -> +residual
        let n1 = self.layer_norm(x, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"), LN_EPS)?;
        let att = self.diffusers_attn(&n1, mask, &format!("{p}.attn1"))?;
        let x = (att + x)?;
        // 3. feed-forward: norm3 -> ff -> +residual
        let n3 = self.layer_norm(&x, &format!("{p}.norm3.weight"), &format!("{p}.norm3.bias"), LN_EPS)?;
        let ff = self.gelu_ff(&n3, &format!("{p}.ff"))?;
        ff + x
    }

    /// diffusers Attention (self-attn, no bias on q/k/v, bias on to_out.0). `mask`, if
    /// given, is an additive `[1,1,t,t]` chunk-causal mask added to the scores before
    /// softmax (full context when `None`).
    fn diffusers_attn(&self, x: &Tensor, mask: Option<&Tensor>, p: &str) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = EST_HEADS;
        let dk = EST_HEAD_DIM;
        let q = self.linear(x, &format!("{p}.to_q.weight"), None)?;
        let k = self.linear(x, &format!("{p}.to_k.weight"), None)?;
        let v = self.linear(x, &format!("{p}.to_v.weight"), None)?;
        let q = q.reshape((b, t, h, dk))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, t, h, dk))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, t, h, dk))?.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (dk as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = softmax_last(&scores)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, h * dk))?;
        // to_out.0 is Linear (with bias), to_out.1 is dropout (off)
        self.linear(&ctx, &format!("{p}.to_out.0.weight"), Some(&format!("{p}.to_out.0.bias")))
    }

    /// diffusers FeedForward with act_fn='gelu': net.0 = GELU(Linear proj) then gelu,
    /// net.2 = Linear out. (net.1 is dropout, off.)
    fn gelu_ff(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        // GELU module: proj Linear then gelu activation.
        let h = self.linear(x, &format!("{p}.net.0.proj.weight"), Some(&format!("{p}.net.0.proj.bias")))?;
        let h = gelu(&h)?;
        self.linear(&h, &format!("{p}.net.2.weight"), Some(&format!("{p}.net.2.bias")))
    }
}

fn mish(x: &Tensor) -> Result<Tensor> {
    // x * tanh(softplus(x)); softplus = ln(1+exp(x)) (use log1p via exp)
    let sp = (x.exp()? + 1.0)?.log()?;
    x * sp.tanh()?
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    // diffusers GELU default (exact, erf-based)
    x.gelu_erf()
}
