//! Real CosyVoice2 flow-matching mel decoder via Candle (the acoustic component's
//! real-weights parity track — the most complex Syrinx component).
//!
//! Reproduces `CausalMaskedDiffWithXvec` (the `flow:` block in cosyvoice2.yaml):
//! speech tokens + a 192-d speaker embedding -> an 80-dim mel spectrogram, through
//! an `UpsampleConformerEncoder` (conformer-style, 2x upsample) and a
//! conditional flow-matching (CFM) decoder that integrates a fixed-step Euler ODE
//! over a U-Net estimator. Everything is deterministic at a fixed seed + fixed step
//! count (the CFM noise is a frozen buffer baked into the checkpoint's design).
//!
//! Gated behind the `real` cargo feature + on-disk fp32 weights; the parity test
//! skips cleanly when the weights/reference are absent (the device-bound recipe),
//! and runs CPU fp32 for real where they exist. This file targets the
//! single-utterance, non-streaming, full-context inference path (no prompt), which
//! is what the reference dumper records — under that path all padding masks are
//! all-true, so attention is unmasked.

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

const ENC_DIM: usize = 512; // encoder hidden
const ENC_HEADS: usize = 8;
const ENC_HEAD_DIM: usize = ENC_DIM / ENC_HEADS; // 64
const MEL: usize = 80;
const EST_HEADS: usize = 8;
const EST_HEAD_DIM: usize = 64;
const N_ENC: usize = 6; // first-stage conformer layers
const N_UPENC: usize = 4; // upsample-stage conformer layers
const N_MID: usize = 12; // estimator mid blocks
const N_TB: usize = 4; // transformer blocks per down/mid/up group
const PRE_LOOKAHEAD: usize = 3;
const LN_EPS: f64 = 1e-5;
const LN_EPS_CONF: f64 = 1e-12; // conformer layernorms use eps 1e-12

/// The real CosyVoice2 flow `CausalMaskedDiffWithXvec`, loaded from fp32 safetensors.
pub struct Flow {
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl Flow {
    /// Load the converted fp32 checkpoint (`flow_fp32.safetensors`) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
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

    /// `x @ W^T (+ b)` for `[.., in]` input and `[out, in]` weight.
    fn linear(&self, x: &Tensor, w: &str, b: Option<&str>) -> Result<Tensor> {
        let weight = self.g(w)?;
        let y = x.broadcast_matmul(&weight.t()?)?;
        match b {
            Some(bn) => y.broadcast_add(&self.g(bn)?),
            None => Ok(y),
        }
    }

    /// LayerNorm over the last dim with explicit weight/bias and eps.
    fn layer_norm(&self, x: &Tensor, w: &str, b: &str, eps: f64) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
        xn.broadcast_mul(&self.g(w)?)?.broadcast_add(&self.g(b)?)
    }

    // ---- the public forward: token + xvec -> mel [1, 80, 2T] ----

    /// Full flow forward for a single utterance (no prompt), 10 Euler steps.
    /// `token`: i64 `[1, T]`; `embedding`: f32 `[1, 192]`. Returns mel `[1, 80, 2T]`.
    pub fn forward(&self, token: &Tensor, embedding: &Tensor, n_timesteps: usize) -> Result<Tensor> {
        let spk = self.spk_proj(embedding)?; // [1, 80]
        let emb = self.input_embedding(token)?; // [1, T, 512]
        let h = self.encoder(&emb)?; // [1, 2T, 512]
        let mu = self.linear(&h, "encoder_proj.weight", Some("encoder_proj.bias"))?; // [1, 2T, 80]
        let mu_t = mu.transpose(1, 2)?.contiguous()?; // [1, 80, 2T]
        self.cfm_solve(&mu_t, &spk, n_timesteps)
    }

    /// Public passthrough to the internal linear (`x @ W^T (+b)`), for tests that
    /// validate the `encoder_proj` stage in isolation.
    pub fn real_linear_pub(&self, x: &Tensor, w: &str, b: Option<&str>) -> Result<Tensor> {
        self.linear(x, w, b)
    }

    /// xvec projection: L2-normalize over dim 1, then affine 192 -> 80.
    pub fn spk_proj(&self, embedding: &Tensor) -> Result<Tensor> {
        let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normed = embedding.broadcast_div(&norm)?;
        self.linear(&normed, "spk_embed_affine_layer.weight", Some("spk_embed_affine_layer.bias"))
    }

    /// Token -> input embedding `[1, T, 512]` (mask is all-ones, so a plain lookup).
    pub fn input_embedding(&self, token: &Tensor) -> Result<Tensor> {
        let table = self.g("input_embedding.weight")?; // [6561, 512]
        let (b, t) = token.dims2()?;
        let idx = token.reshape((b * t,))?;
        let emb = table.index_select(&idx, 0)?; // [b*t, 512]
        emb.reshape((b, t, ENC_DIM))
    }

    // ============================ ENCODER ============================

    /// `UpsampleConformerEncoder.forward` (streaming=False, full context, no mask).
    pub fn encoder(&self, emb: &Tensor) -> Result<Tensor> {
        // embed: Linear -> LayerNorm(1e-5); pos enc multiplies x by sqrt(d_model)
        let mut xs = self.subsample(emb, "encoder.embed")?; // [1, T, 512]
        let t = xs.dim(1)?;
        let pos = self.rel_pos_emb(t)?; // [1, 2T-1, 512]
        xs = (xs * (ENC_DIM as f64).sqrt())?;

        // pre-lookahead layer
        xs = self.pre_lookahead(&xs)?;

        // first-stage conformer layers
        for l in 0..N_ENC {
            xs = self.conformer_layer(&xs, &pos, &format!("encoder.encoders.{l}"))?;
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

        for l in 0..N_UPENC {
            xs = self.conformer_layer(&xs, &pos2, &format!("encoder.up_encoders.{l}"))?;
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
    fn conformer_layer(&self, x: &Tensor, pos: &Tensor, p: &str) -> Result<Tensor> {
        let res = x.clone();
        let xn = self.layer_norm(x, &format!("{p}.norm_mha.weight"), &format!("{p}.norm_mha.bias"), LN_EPS_CONF)?;
        let att = self.rel_self_attn(&xn, pos, &format!("{p}.self_attn"))?;
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

    /// RelPositionMultiHeadedAttention (espnet rel-pos), no mask (full context).
    fn rel_self_attn(&self, x: &Tensor, pos: &Tensor, p: &str) -> Result<Tensor> {
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
        let probs = softmax_last(&scores)?;
        let ctx = probs.matmul(&v)?; // [b,h,t,dk]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, h * dk))?;
        self.linear(&ctx, &format!("{p}.linear_out.weight"), Some(&format!("{p}.linear_out.bias")))
    }

    // ============================ CFM / ESTIMATOR ============================

    /// CausalConditionalCFM.solve_euler with CFG. `mu`: [1,80,L], `spk`: [1,80].
    /// Noise z is the frozen design buffer; here we read it from the reference so
    /// the ODE is bit-reproducible (the buffer is a fixed seed-0 randn, baked in).
    pub fn cfm_solve_with_noise(&self, mu: &Tensor, spk: &Tensor, z0: &Tensor, n_timesteps: usize) -> Result<Tensor> {
        let l = mu.dim(2)?;
        // t_span (cosine schedule)
        let mut tvals = vec![0f32; n_timesteps + 1];
        for (i, s) in tvals.iter_mut().enumerate() {
            let lin = i as f32 / n_timesteps as f32;
            *s = 1.0 - (lin * 0.5 * std::f32::consts::PI).cos();
        }
        let mut x = z0.clone(); // [1,80,L]
        let mut t = tvals[0];
        let cfg = 0.7f64; // inference_cfg_rate
        for step in 1..=n_timesteps {
            // CFG batch of 2: index 0 carries mu/spks/cond, index 1 is zeros.
            // x_in[:] = x  -> both rows are x
            let x_in = Tensor::cat(&[&x, &x], 0)?; // [2,80,L]
            let mu0 = mu.clone();
            let mu1 = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
            let mu_in = Tensor::cat(&[&mu0, &mu1], 0)?; // [2,80,L]
            let spk0 = spk.clone();
            let spk1 = Tensor::zeros((1, MEL), DType::F32, &self.dev)?;
            let spk_in = Tensor::cat(&[&spk0, &spk1], 0)?; // [2,80]
            let cond_in = Tensor::zeros((2, MEL, l), DType::F32, &self.dev)?; // no prompt
            let t_in = Tensor::from_vec(vec![t, t], (2,), &self.dev)?;
            let dphi = self.estimator(&x_in, &mu_in, &t_in, &spk_in, &cond_in)?; // [2,80,L]
            let real = dphi.narrow(0, 0, 1)?;
            let uncond = dphi.narrow(0, 1, 1)?;
            // (1+cfg)*real - cfg*uncond
            let dphi_dt = ((real * (1.0 + cfg))? - (uncond * cfg)?)?;
            let dt = tvals[step] - t;
            x = (x + (dphi_dt * dt as f64)?)?;
            t = tvals[step];
        }
        Ok(x)
    }

    /// CausalConditionalCFM.solve_euler with CFG **and a non-trivial `cond`** — the
    /// zero-shot prompt path. Identical to [`Self::cfm_solve_with_noise`] except the
    /// conditioning signal is the caller-supplied `cond` `[1, 80, L]` (the prompt mel
    /// prepended, zeros after) rather than all-zeros. The unconditioned CFG branch
    /// (index 1) keeps `cond = 0`, mirroring `solve_euler`'s `cond_in[0] = cond`.
    pub fn cfm_solve_with_cond(
        &self,
        mu: &Tensor,
        spk: &Tensor,
        cond: &Tensor,
        z0: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        let l = mu.dim(2)?;
        let mut tvals = vec![0f32; n_timesteps + 1];
        for (i, s) in tvals.iter_mut().enumerate() {
            let lin = i as f32 / n_timesteps as f32;
            *s = 1.0 - (lin * 0.5 * std::f32::consts::PI).cos();
        }
        let mut x = z0.clone(); // [1,80,L]
        let mut t = tvals[0];
        let cfg = 0.7f64; // inference_cfg_rate
        let cond_zero = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        for step in 1..=n_timesteps {
            let x_in = Tensor::cat(&[&x, &x], 0)?; // [2,80,L]
            let mu0 = mu.clone();
            let mu1 = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
            let mu_in = Tensor::cat(&[&mu0, &mu1], 0)?; // [2,80,L]
            let spk0 = spk.clone();
            let spk1 = Tensor::zeros((1, MEL), DType::F32, &self.dev)?;
            let spk_in = Tensor::cat(&[&spk0, &spk1], 0)?; // [2,80]
            // cond[0] = prompt cond, cond[1] = zeros (the CFG-dropped branch).
            let cond_in = Tensor::cat(&[cond, &cond_zero], 0)?; // [2,80,L]
            let t_in = Tensor::from_vec(vec![t, t], (2,), &self.dev)?;
            let dphi = self.estimator(&x_in, &mu_in, &t_in, &spk_in, &cond_in)?; // [2,80,L]
            let real = dphi.narrow(0, 0, 1)?;
            let uncond = dphi.narrow(0, 1, 1)?;
            let dphi_dt = ((real * (1.0 + cfg))? - (uncond * cfg)?)?;
            let dt = tvals[step] - t;
            x = (x + (dphi_dt * dt as f64)?)?;
            t = tvals[step];
        }
        Ok(x)
    }

    /// Full zero-shot prompt-conditioned `flow.inference` (the CosyVoice2 path).
    ///
    /// Mirrors `CausalMaskedDiffWithXvec.inference(streaming=False, finalize=True)`:
    /// concatenate `prompt_token ++ token`, encode the whole thing, project to mu,
    /// build the CFM `cond` by prepending the prompt mel `prompt_feat`, solve the
    /// 10-step Euler ODE feeding the pinned noise `z`, then drop the prompt-mel
    /// prefix so only the generated mel is returned.
    ///
    /// - `prompt_token`: i64 `[1, Tp]`, `token`: i64 `[1, Tg]`
    /// - `prompt_feat`: f32 `[1, Mp, 80]` (the prompt mel; `Mp == 2*Tp`)
    /// - `embedding`: f32 `[1, 192]`
    /// - `z`: f32 `[1, 80, 2*(Tp+Tg)]` — the model's fixed `rand_noise` slice.
    ///
    /// Returns the generated mel `[1, 80, 2*Tg]`.
    pub fn forward_zero_shot(
        &self,
        prompt_token: &Tensor,
        token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        let spk = self.spk_proj(embedding)?; // [1, 80]

        // concat prompt + gen tokens, embed, encode (full context, no mask).
        let tok_cat = Tensor::cat(&[prompt_token, token], 1)?; // [1, Tp+Tg]
        let emb = self.input_embedding(&tok_cat)?; // [1, T, 512]
        let h = self.encoder(&emb)?; // [1, 2T, 512]
        let mu = self.linear(&h, "encoder_proj.weight", Some("encoder_proj.bias"))?; // [1, 2T, 80]
        let mu_t = mu.transpose(1, 2)?.contiguous()?; // [1, 80, 2T]

        let total = mu_t.dim(2)?; // 2*(Tp+Tg)
        let mel_len1 = prompt_feat.dim(1)?; // == 2*Tp
        let mel_len2 = total - mel_len1; // == 2*Tg

        // cond: prompt mel prepended ([1, 80, mel_len1]), zeros after -> [1,80,total].
        let prompt_ct = prompt_feat.transpose(1, 2)?.contiguous()?; // [1, 80, mel_len1]
        let cond_tail = Tensor::zeros((1, MEL, mel_len2), DType::F32, &self.dev)?;
        let cond = Tensor::cat(&[&prompt_ct, &cond_tail], 2)?; // [1, 80, total]

        let mel_full = self.cfm_solve_with_cond(&mu_t, &spk, &cond, z, n_timesteps)?; // [1,80,total]
        // drop the prompt-mel prefix; keep only the generated mel.
        mel_full.narrow(2, mel_len1, mel_len2)
    }

    /// Convenience: solve using the design noise buffer reconstructed via the
    /// reference fixture is preferred; this variant uses a provided z explicitly.
    pub fn cfm_solve(&self, mu: &Tensor, spk: &Tensor, n_timesteps: usize) -> Result<Tensor> {
        // Without the frozen randn buffer we cannot reproduce z; callers that need
        // bit-parity must pass the reference noise via cfm_solve_with_noise.
        let l = mu.dim(2)?;
        let z = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        self.cfm_solve_with_noise(mu, spk, &z, n_timesteps)
    }

    /// CausalConditionalDecoder.forward (the estimator), streaming=False, no mask.
    /// Inputs are the CFG-stacked `[2, .., L]` tensors. Returns `[2, 80, L]`.
    pub fn estimator(&self, x: &Tensor, mu: &Tensor, t: &Tensor, spks: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let l = x.dim(2)?;
        // time embedding: SinusoidalPosEmb(in_channels=320) then time_mlp
        let temb = self.time_embed(t)?; // [2, 1024]

        // pack x = cat([x, mu, spks_broadcast, cond], dim=1) -> [2, 320, L]
        let spks_b = spks.unsqueeze(2)?.broadcast_as((spks.dim(0)?, MEL, l))?.contiguous()?;
        let mut h = Tensor::cat(&[x, mu, &spks_b, cond], 1)?; // [2,320,L]

        let mut hiddens: Vec<Tensor> = Vec::new();
        // down block (channels=[256] -> single down, is_last, downsample=CausalConv1d k=3)
        h = self.causal_resnet(&h, &temb, "decoder.estimator.down_blocks.0.0")?; // [2,256,L]
        h = self.transformer_stack(&h, &temb, "decoder.estimator.down_blocks.0.1")?;
        hiddens.push(h.clone());
        // downsample: CausalConv1d (pad left 2, k=3, stride 1) -> same length
        h = self.causal_conv(&h, "decoder.estimator.down_blocks.0.2", 3)?;

        // mid blocks
        for m in 0..N_MID {
            h = self.causal_resnet(&h, &temb, &format!("decoder.estimator.mid_blocks.{m}.0"))?;
            h = self.transformer_stack(&h, &temb, &format!("decoder.estimator.mid_blocks.{m}.1"))?;
        }

        // up block (single, is_last, upsample=CausalConv1d k=3). input is cat(x, skip)=512
        let skip = hiddens.pop().unwrap();
        let cat = Tensor::cat(&[&h, &skip], 1)?; // [2,512,L]
        h = self.causal_resnet(&cat, &temb, "decoder.estimator.up_blocks.0.0")?; // [2,256,L]
        h = self.transformer_stack(&h, &temb, "decoder.estimator.up_blocks.0.1")?;
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
    fn transformer_stack(&self, x: &Tensor, temb: &Tensor, p: &str) -> Result<Tensor> {
        let _ = temb; // norm here is plain LayerNorm (no ada-norm); timestep unused
        let mut h = x.transpose(1, 2)?.contiguous()?; // [b, L, c]
        for i in 0..N_TB {
            h = self.basic_transformer_block(&h, &format!("{p}.{i}"))?;
        }
        h.transpose(1, 2)?.contiguous()
    }

    /// diffusers BasicTransformerBlock (self-attn only, layer_norm, gelu FF).
    fn basic_transformer_block(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        // 1. self-attn: norm1 -> attn1 -> +residual
        let n1 = self.layer_norm(x, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"), LN_EPS)?;
        let att = self.diffusers_attn(&n1, &format!("{p}.attn1"))?;
        let x = (att + x)?;
        // 3. feed-forward: norm3 -> ff -> +residual
        let n3 = self.layer_norm(&x, &format!("{p}.norm3.weight"), &format!("{p}.norm3.bias"), LN_EPS)?;
        let ff = self.gelu_ff(&n3, &format!("{p}.ff"))?;
        ff + x
    }

    /// diffusers Attention (self-attn, no bias on q/k/v, bias on to_out.0).
    fn diffusers_attn(&self, x: &Tensor, p: &str) -> Result<Tensor> {
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

/// Deterministic zero-shot **token2wav** glue: speech tokens -> mel -> audio.
///
/// Reproduces `CosyVoice2Model.token2wav` (non-streaming, single utterance) for the
/// zero-shot path: the prompt-conditioned flow ([`Flow::forward_zero_shot`]) yields
/// the generated mel, which the HiFT vocoder ([`syrinx_vocoder::real::HiftVocoder::decode`])
/// turns into a 24 kHz waveform. Both stochastic inputs are pinned and fed in:
/// the CFM noise `z` (the flow's fixed `rand_noise` slice) and the HiFT source STFT
/// `s_stft` (the SineGen source has a random initial phase in the real model, so the
/// reference captures it). Returns the waveform `[1, L]`.
#[allow(clippy::too_many_arguments)]
pub fn token2wav(
    flow: &Flow,
    vocoder: &syrinx_vocoder::real::HiftVocoder,
    prompt_token: &Tensor,
    token: &Tensor,
    prompt_feat: &Tensor,
    embedding: &Tensor,
    z: &Tensor,
    s_stft: &Tensor,
    n_timesteps: usize,
) -> Result<Tensor> {
    let mel = flow.forward_zero_shot(prompt_token, token, prompt_feat, embedding, z, n_timesteps)?;
    vocoder.decode(&mel, s_stft)
}

// ============================ free fns ============================

/// Pad the last (time) dim with zeros: `left` then `right`.
fn pad_time(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let mut y = x.clone();
    if left > 0 {
        let dims = x.dims().to_vec();
        let mut sh = dims.clone();
        sh[2] = left;
        let z = Tensor::zeros(sh.as_slice(), x.dtype(), x.device())?;
        y = Tensor::cat(&[&z, &y], 2)?;
    }
    if right > 0 {
        let mut sh = y.dims().to_vec();
        sh[2] = right;
        let z = Tensor::zeros(sh.as_slice(), x.dtype(), x.device())?;
        y = Tensor::cat(&[&y, &z], 2)?;
    }
    Ok(y)
}

/// 1D convolution, stride `s`, no padding (caller pads). weight `[out,in,k]`.
fn conv1d(x: &Tensor, w: &Tensor, b: Option<&Tensor>, s: usize) -> Result<Tensor> {
    let y = x.conv1d(w, 0, s, 1, 1)?;
    match b {
        Some(bias) => y.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?),
        None => Ok(y),
    }
}

/// Nearest-neighbour upsample along the time (last) dim by integer `factor`.
fn upsample_nearest_time(x: &Tensor, factor: usize) -> Result<Tensor> {
    x.upsample_nearest1d(x.dim(2)? * factor)
}

fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    // max(x, 0) + slope * min(x, 0)
    let pos = x.relu()?;
    let neg = (x - &pos)?; // == min(x,0)
    pos + (neg * slope)?
}

fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
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

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(x, D::Minus1)
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
