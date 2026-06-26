//! Real CosyVoice3 flow-matching mel decoder via Candle (`CausalMaskedDiffWithDiT`).
//!
//! The CV3 flow differs from CV2 (`CausalMaskedDiffWithXvec`, see [`crate::real`]) in
//! two places, and reuses the rest:
//!
//!   * **Front-end (token -> mu).** No conformer encoder. The token id is looked up in
//!     an `Embedding(6561, 80)` (note: 80-d, *not* CV2's 512-d), passed through a single
//!     `PreLookaheadLayer` (conv1 k=4 left-context + conv2 k=3 + residual), then
//!     `repeat_interleave(2)` along time and transposed to `mu` `[1, 80, 2T]`.
//!   * **CFM estimator.** A **22-layer DiT transformer** (`dim=1024`, `16` heads,
//!     `dim_head=64`, rotary position emb, AdaLN-Zero time conditioning, tanh-GELU FF
//!     of inner width `2048`) replaces CV2's U-Net. This is the hard part.
//!
//! The CFM Euler/CFG *wrapper* is byte-identical in structure to CV2's `solve_euler`
//! (10 cosine-schedule steps, CFG batch-of-2 with `cfg_rate = 0.7`, frozen noise `z`
//! consumed verbatim); only the estimator it calls changed, so it is re-expressed here
//! around the DiT rather than shared by reference (CV2's `crate::real` stays byte-frozen).
//!
//! Gated behind the `real` feature + on-disk fp32 weights; the parity test
//! (`tests/real_cv3_flow_parity.rs`) skips cleanly when the weights/reference are
//! absent and runs CPU fp32 where they exist. Single-utterance, non-streaming,
//! full-context inference (the path the reference dumper records): all padding masks
//! are all-true, so attention is unmasked.

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

// ---- CV3 flow dimensions (from build_flow / flow_fp32.safetensors shapes) ----
const VOCAB: usize = 6561; // input_embedding rows
const MEL: usize = 80; // output mel channels == input_embedding cols == spk_proj out
const TOKEN_MEL_RATIO: usize = 2; // repeat_interleave factor
const PRE_LOOKAHEAD: usize = 3; // pre_lookahead_len
const SPK_DIM: usize = 192; // raw xvec dim

// ---- DiT estimator dimensions ----
const DIT_DIM: usize = 1024; // transformer hidden
const DIT_DEPTH: usize = 22; // number of DiTBlocks
const DIT_HEADS: usize = 16; // attention heads (1024 / 64)
const DIT_HEAD_DIM: usize = 64; // == rotary dim
const DIT_FREQ_DIM: usize = 256; // SinusPositionEmbedding width for the time embed
const PROJ_IN: usize = MEL * 2 + MEL + MEL; // input_embed.proj in: x|cond|mu|spks = 320
const CONV_POS_K: usize = 31; // CausalConvPositionEmbedding kernel
const CONV_POS_GROUPS: usize = 16; // grouped conv

const LN_EPS_DIT: f64 = 1e-6; // DiT's elementwise_affine=False LayerNorms
const CFG_RATE: f64 = 0.7; // inference_cfg_rate

/// The real CosyVoice3 flow `CausalMaskedDiffWithDiT`, loaded from fp32 safetensors.
pub struct Cv3Flow {
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl Cv3Flow {
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

    // ======================== FRONT-END (token -> mu) ========================

    /// xvec projection: L2-normalize over dim 1, then affine 192 -> 80. `[1,192]->[1,80]`.
    pub fn spk_proj(&self, embedding: &Tensor) -> Result<Tensor> {
        debug_assert_eq!(embedding.dim(1).unwrap_or(SPK_DIM), SPK_DIM);
        let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normed = embedding.broadcast_div(&norm)?;
        self.linear(&normed, "spk_embed_affine_layer.weight", Some("spk_embed_affine_layer.bias"))
    }

    /// Token -> input embedding `[1, T, 80]` (single-utterance: mask all-ones, plain
    /// lookup of `Embedding(6561, 80)` after `clamp(min=0)`).
    pub fn input_embedding(&self, token: &Tensor) -> Result<Tensor> {
        let table = self.g("input_embedding.weight")?; // [6561, 80]
        let (b, t) = token.dims2()?;
        // clamp(min=0): tokens are already non-negative ids; clamp for fidelity.
        let idx = token
            .to_dtype(DType::I64)?
            .clamp(0i64, (VOCAB - 1) as i64)?
            .reshape((b * t,))?;
        let emb = table.index_select(&idx, 0)?; // [b*t, 80]
        emb.reshape((b, t, MEL))
    }

    /// `PreLookaheadLayer` (inference, no streaming context). Input/out `[1, T, 80]`.
    ///
    /// transpose -> pad RIGHT `pre_lookahead_len` (=3) -> conv1 (k=4, 80->1024) ->
    /// leaky_relu(0.01) -> pad LEFT (k2-1)=2 -> conv2 (k=3, 1024->80) -> transpose ->
    /// + residual.
    pub fn pre_lookahead(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?; // [1, 80, T]
        let padded = pad_time(&xt, 0, PRE_LOOKAHEAD)?; // pad right by 3
        let c1 = conv1d(
            &padded,
            &self.g("pre_lookahead_layer.conv1.weight")?,
            Some(&self.g("pre_lookahead_layer.conv1.bias")?),
            1,
            1,
        )?;
        let c1 = leaky_relu(&c1, 0.01)?;
        let padded2 = pad_time(&c1, 2, 0)?; // pad left by k2-1 = 2
        let c2 = conv1d(
            &padded2,
            &self.g("pre_lookahead_layer.conv2.weight")?,
            Some(&self.g("pre_lookahead_layer.conv2.bias")?),
            1,
            1,
        )?;
        let out = c2.transpose(1, 2)?.contiguous()?; // [1, T, 80]
        out + x
    }

    /// Full token -> mu front-end: `input_embedding -> pre_lookahead ->
    /// repeat_interleave(2, time) -> transpose`. Returns `mu` `[1, 80, 2T]`.
    pub fn token_to_mu(&self, token: &Tensor) -> Result<Tensor> {
        let emb = self.input_embedding(token)?; // [1, T, 80]
        let h_pre = self.pre_lookahead(&emb)?; // [1, T, 80]
        let h = repeat_interleave_time(&h_pre, TOKEN_MEL_RATIO)?; // [1, 2T, 80]
        h.transpose(1, 2)?.contiguous() // mu [1, 80, 2T]
    }

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
            h = self.dit_block(&h, &temb, &rope_cos, &rope_sin, &format!("{p}.transformer_blocks.{i}"))?;
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
        p: &str,
    ) -> Result<Tensor> {
        // attn_norm = AdaLayerNormZero: returns modulated `norm` + the 4 mlp/gate params.
        let (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.adaln_zero(x, temb, &format!("{p}.attn_norm"))?;
        let attn = self.dit_attn(&norm, rope_cos, rope_sin, &format!("{p}.attn"))?;
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
    /// IMPORTANT — faithful to the reference: rotary is applied to the **full
    /// `[B,N,1024]` projection BEFORE the head reshape**, with a 64-wide freqs table
    /// (`rot_dim=64`). So only channels `[0:64]` are rotated (== head 0 after reshape);
    /// channels `[64:1024]` pass through unrotated. (CV3's `AttnProcessor` rotates pre-
    /// view, unlike standard F5-TTS which rotates post-view per head — see module note.)
    fn dit_attn(&self, x: &Tensor, rope_cos: &Tensor, rope_sin: &Tensor, p: &str) -> Result<Tensor> {
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

    // ============================ CFM SOLVE ============================

    /// CV3 CFM Euler solve with CFG (reuses the CV2 wrapper structure around the DiT).
    ///
    /// `mu,cond: [1,80,L]`, `spk: [1,80]`, `z0: [1,80,L]` (frozen noise consumed
    /// verbatim). 10 cosine-schedule steps, CFG batch-of-2 (`idx0` carries
    /// mu/spk/cond, `idx1` is zeros), `cfg_rate=0.7`. Returns `mel_full [1,80,L]`.
    pub fn cfm_solve(
        &self,
        mu: &Tensor,
        spk: &Tensor,
        cond: &Tensor,
        z0: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        let l = mu.dim(2)?;
        // t_span: cosine schedule 1 - cos(linspace(0,1,n+1) * 0.5*pi).
        let mut tvals = vec![0f32; n_timesteps + 1];
        for (i, s) in tvals.iter_mut().enumerate() {
            let lin = i as f32 / n_timesteps as f32;
            *s = 1.0 - (lin * 0.5 * std::f32::consts::PI).cos();
        }
        let mut x = z0.clone(); // [1,80,L]
        let mut t = tvals[0];
        let zero_mu = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        let zero_spk = Tensor::zeros((1, MEL), DType::F32, &self.dev)?;
        let zero_cond = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        for step in 1..=n_timesteps {
            let x_in = Tensor::cat(&[&x, &x], 0)?; // [2,80,L]
            let mu_in = Tensor::cat(&[mu, &zero_mu], 0)?; // [2,80,L]
            let spk_in = Tensor::cat(&[spk, &zero_spk], 0)?; // [2,80]
            let cond_in = Tensor::cat(&[cond, &zero_cond], 0)?; // [2,80,L]
            let t_in = Tensor::from_vec(vec![t, t], (2,), &self.dev)?;
            let dphi = self.estimator(&x_in, &mu_in, &t_in, &spk_in, &cond_in)?; // [2,80,L]
            let real = dphi.narrow(0, 0, 1)?;
            let uncond = dphi.narrow(0, 1, 1)?;
            let dphi_dt = ((real * (1.0 + CFG_RATE))? - (uncond * CFG_RATE)?)?;
            let dt = tvals[step] - t;
            x = (x + (dphi_dt * dt as f64)?)?;
            t = tvals[step];
        }
        Ok(x)
    }

    /// Full zero-shot CV3 flow (`CausalMaskedDiffWithDiT.inference`, non-streaming).
    ///
    /// `prompt_token`/`token`: i64 `[1,Tp]`/`[1,Tg]`; `prompt_feat`: f32 `[1,Mp,80]`
    /// (`Mp == 2*Tp`); `embedding`: f32 `[1,192]`; `z`: f32 `[1,80,2*(Tp+Tg)]`.
    /// Returns generated mel `[1,80,2*Tg]` (the prompt-mel prefix dropped).
    pub fn forward(
        &self,
        prompt_token: &Tensor,
        token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        let spk = self.spk_proj(embedding)?; // [1,80]
        let tok_cat = Tensor::cat(&[prompt_token, token], 1)?; // [1, Tp+Tg]
        let mu = self.token_to_mu(&tok_cat)?; // [1,80, 2*(Tp+Tg)]

        let total = mu.dim(2)?;
        let mel_len1 = prompt_feat.dim(1)?; // 2*Tp
        let mel_len2 = total - mel_len1; // 2*Tg

        let prompt_ct = prompt_feat.transpose(1, 2)?.contiguous()?; // [1,80,mel_len1]
        let cond_tail = Tensor::zeros((1, MEL, mel_len2), DType::F32, &self.dev)?;
        let cond = Tensor::cat(&[&prompt_ct, &cond_tail], 2)?; // [1,80,total]

        let mel_full = self.cfm_solve(&mu, &spk, &cond, z, n_timesteps)?; // [1,80,total]
        mel_full.narrow(2, mel_len1, mel_len2) // drop prompt-mel prefix
    }
}

// ============================ free fns ============================

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

/// Repeat each time step `factor` times along dim 1: `[B,T,C] -> [B,factor*T,C]`.
/// (`torch.repeat_interleave(factor, dim=1)`: `[f0,f0,f1,f1,...]`.)
fn repeat_interleave_time(x: &Tensor, factor: usize) -> Result<Tensor> {
    let t = x.dim(1)?;
    let mut idx: Vec<u32> = Vec::with_capacity(t * factor);
    for p in 0..t {
        for _ in 0..factor {
            idx.push(p as u32);
        }
    }
    let idx = Tensor::from_vec(idx, (t * factor,), x.device())?;
    x.index_select(&idx, 1)
}

/// Pad the last (time) dim with zeros: `left` then `right`.
fn pad_time(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let mut y = x.clone();
    if left > 0 {
        let mut sh = x.dims().to_vec();
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

/// 1D convolution, stride `s`, groups `groups`, no padding (caller pads).
/// weight `[out, in/groups, k]`.
fn conv1d(x: &Tensor, w: &Tensor, b: Option<&Tensor>, s: usize, groups: usize) -> Result<Tensor> {
    let y = x.conv1d(w, 0, s, 1, groups)?;
    match b {
        Some(bias) => y.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?),
        None => Ok(y),
    }
}

fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = (x - &pos)?; // == min(x,0)
    pos + (neg * slope)?
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
