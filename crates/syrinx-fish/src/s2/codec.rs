//! The s2 **446M EVA-GAN / causal-DAC** RVQ codec: `[10, T]` codes ↔ 44.1 kHz waveform.
//!
//! ⚠ THIS IS THE RISKIEST, MOST PARITY-UNCONFIRMED MODULE IN THE s2 BACKEND. ⚠
//! There is no in-repo reference for the s2 codec (the RVQ lives in the external `dac`
//! package and the generator is a custom EVA-GAN), and the box is offline so the exact
//! `codec.pth` key layout / channel schedule / block dilations cannot be verified. The
//! math below is **real and complete** (real causal convs, real factorized RVQ, real
//! sliding-window transformer), reconstructed from the S2 technical report
//! (arXiv:2603.08823) plus the s1 modded-DAC as a structural starting point — but EVERY
//! structural choice marked `// PARITY:` MUST be reconciled against the real `codec.pth`
//! on-box. Treat any numeric output as unvalidated until then.
//!
//! ## What the tech report fixes (and we implement)
//! * RVQ: 10 codebooks (1 **semantic** + 9 **residual**), factorized to `codebook_dim`,
//!   the Descript-Audio-Codec (DAC) quantizer math (`in_proj` → L2 nearest codebook →
//!   `out_proj`), exactly as s1's modded-DAC.
//! * Strictly **causal** convolutions everywhere (masked / left-padded), for streaming.
//! * **Mimi-style** causal sliding-window Transformer blocks **both before and after**
//!   the RVQ layers (the "bottleneck"). We run a `pre_rvq` transformer on the encoder
//!   side (after downsample, before quantize) and a `post_rvq` transformer on the decode
//!   side (after `from_codes`, before the generator). Applied only when the checkpoint
//!   carries those weights (so a codec without an explicit bottleneck still loads).
//! * Downsampling = standard DAC encoder (**512×**) + extra **ConvNeXt-V2 (4×)** ⇒
//!   total **2048×** ⇒ ~21 Hz frame rate at 44.1 kHz.
//! * Decoder = the **EVA-GAN** generator structure (replaces the DAC decoder), causal.
//!
//! ## Least-confident parts (be honest)
//! 1. The EVA-GAN generator's exact block layout — MRF residual kernel/dilation sets,
//!    channel schedule, and whether it uses Snake or anti-aliased (AMP) activations.
//!    Implemented here as the DAC-style causal upsampler (Snake + ConvTranspose + dilated
//!    ResidualUnits); the real EVA-GAN generator differs and is `// PARITY:`-flagged.
//! 2. The Mimi bottleneck transformer depth / window size / placement.
//! 3. Every `codec.pth` weight key name (no reference state-dict offline).

use candle_core::{DType, Device, Result, Tensor, D};

use super::nn::{attention, precompute_rope, swiglu, AttnShape, KvCache, Weights};
use crate::common::config::CodecConfig;

// --- s2 EVA-GAN / causal-DAC structural constants -----------------------------
// PARITY: confirm every constant here against `s2-pro/codec.pth` + its config on-box.

/// Encoder base channels (`encoder_dim`). PARITY: confirm on-box.
const ENCODER_DIM: usize = 64;
/// Number of **residual** RVQ codebooks (the semantic codebook is separate).
const N_RESIDUAL: usize = 9; // PARITY: confirm n_codebooks == 1 + 9 on-box.
/// The factorized down/up-sample factors (extra ConvNeXt-V2 ×4 on top of DAC's 512×),
/// product == ×4. PARITY: confirm downsample_factor on-box.
const DOWNSAMPLE_FACTOR: [usize; 2] = [2, 2];
/// Mimi-style bottleneck transformer depth (one stack pre-RVQ, one post-RVQ).
/// PARITY: confirm the bottleneck layer count on-box (Mimi uses 8; s2 unknown).
const BOTTLENECK_LAYERS: usize = 8;
/// Bottleneck transformer head dim. PARITY: confirm on-box.
const BOTTLENECK_HEAD_DIM: usize = 64;
/// Causal sliding-window attention window (`window_size`). PARITY: confirm on-box.
const WINDOW_SIZE: usize = 250;
/// Codec transformer RoPE base. PARITY: confirm on-box.
const DAC_ROPE_BASE: f64 = 10_000.0;
/// Codec transformer / norm epsilon. PARITY: confirm on-box.
const DAC_NORM_EPS: f64 = 1e-5;
/// `Snake1d` numerical epsilon (`(alpha + 1e-9).reciprocal()`).
const SNAKE_EPS: f64 = 1e-9;
/// `F.normalize` epsilon (p=2).
const NORM_EPS: f64 = 1e-12;

/// The loaded s2 EVA-GAN / causal-DAC codec.
pub struct EvaGanDac {
    w: Weights,
    cfg: CodecConfig,
}

impl EvaGanDac {
    /// Build from a loaded codec weight bag + the resolved codec geometry.
    pub fn new(w: Weights, cfg: CodecConfig) -> Self {
        Self { w, cfg }
    }

    fn dev(&self) -> &Device {
        &self.w.dev
    }

    // --- causal conv primitives (CausalConvNet / CausalTransConvNet) -----------

    /// `CausalConvNet.forward`: left-pad by `kernel_eff - stride` (+ the alignment
    /// `extra_padding`) with zeros, then a plain conv with no internal padding.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv1d(
        &self,
        x: &Tensor,
        wname: &str,
        bname: &str,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        let weight = self.w.g(wname)?;
        let bias = self.w.g(bname)?;
        let kernel_eff = (kernel - 1) * dilation + 1;
        let pad = kernel_eff.saturating_sub(stride);
        let length = x.dim(D::Minus1)?;
        let extra = extra_padding(length, kernel_eff, stride, pad);
        let xp = x.pad_with_zeros(D::Minus1, pad, extra)?;
        let y = xp.conv1d(&weight, 0, stride, dilation, groups)?;
        let b = bias.reshape((1, bias.dim(0)?, 1))?;
        y.broadcast_add(&b)
    }

    /// `CausalTransConvNet.forward`: `ConvTranspose1d` then unpad `(left, right)` where
    /// `right = kernel - stride`, `left = 0`.
    fn causal_transpose1d(
        &self,
        x: &Tensor,
        wname: &str,
        bname: &str,
        kernel: usize,
        stride: usize,
    ) -> Result<Tensor> {
        let weight = self.w.g(wname)?; // [Cin, Cout, K]
        let bias = self.w.g(bname)?;
        let y = x.conv_transpose1d(&weight, 0, 0, stride, 1, 1)?;
        let b = bias.reshape((1, bias.dim(0)?, 1))?;
        let y = y.broadcast_add(&b)?;
        let pad = kernel.saturating_sub(stride);
        let right = pad;
        let left = pad - right; // == 0
        let len = y.dim(D::Minus1)?;
        let kept = len - left - right;
        y.narrow(D::Minus1, left, kept)
    }

    /// `Snake1d`: `x + (alpha + 1e-9)^{-1} * sin(alpha * x)^2`, channel-wise alpha.
    fn snake(&self, x: &Tensor, alpha_name: &str) -> Result<Tensor> {
        let a = self.w.g(alpha_name)?;
        let c = a.elem_count();
        let alpha = a.reshape((1, c, 1))?;
        let xa = x.broadcast_mul(&alpha)?;
        let s = xa.sin()?.sqr()?;
        let inv = alpha.affine(1.0, SNAKE_EPS)?.recip()?;
        x.add(&s.broadcast_mul(&inv)?)
    }

    /// `ResidualUnit`: `Snake → Conv(k7, dilation) → Snake → Conv(k1)`, added back to
    /// the (causal-trimmed) input.
    fn residual_unit(&self, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
        let y = self.snake(x, &format!("{prefix}.block.0.alpha"))?;
        let y = self.causal_conv1d(
            &y,
            &format!("{prefix}.block.1.conv.weight"),
            &format!("{prefix}.block.1.conv.bias"),
            7,
            1,
            dilation,
            1,
        )?;
        let y = self.snake(&y, &format!("{prefix}.block.2.alpha"))?;
        let y = self.causal_conv1d(
            &y,
            &format!("{prefix}.block.3.conv.weight"),
            &format!("{prefix}.block.3.conv.bias"),
            1,
            1,
            1,
            1,
        )?;
        let lx = x.dim(D::Minus1)?;
        let ly = y.dim(D::Minus1)?;
        let x = if lx > ly {
            x.narrow(D::Minus1, 0, ly)? // causal: drop the right
        } else {
            x.clone()
        };
        x.add(&y)
    }

    /// `ConvNeXtV2Block`: depthwise causal `Conv(k7)` → channels-last `LayerNorm` →
    /// `Linear → GELU → (GRN) → Linear` → `gamma` scale → residual.
    //
    // PARITY: the s2 downsampler uses **ConvNeXt-V2**, which adds a Global Response
    // Normalization (GRN) between the two pointwise convs. We apply GRN when the block
    // ships `{prefix}.grn.gamma`/`.beta`; otherwise this is a ConvNeXt-V1 block. Confirm
    // the exact ConvNeXt variant + whether `gamma` (layer scale) is present on-box.
    fn convnext(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let c = x.dim(1)?;
        let h = self.causal_conv1d(
            x,
            &format!("{prefix}.dwconv.conv.weight"),
            &format!("{prefix}.dwconv.conv.bias"),
            7,
            1,
            1,
            c, // depthwise
        )?;
        let h = h.permute((0, 2, 1))?.contiguous()?; // [B, T, C]
        let h = layer_norm(
            &h,
            &self.w.g(&format!("{prefix}.norm.weight"))?,
            &self.w.g(&format!("{prefix}.norm.bias"))?,
            1e-6,
        )?;
        let h = self.w.linear(
            &h,
            &format!("{prefix}.pwconv1.weight"),
            Some(&format!("{prefix}.pwconv1.bias")),
        )?;
        let h = h.gelu_erf()?;
        // ConvNeXt-V2 GRN (optional — present only in V2 blocks).
        let h = if self.w.has(&format!("{prefix}.grn.gamma")) {
            self.grn(&h, prefix)?
        } else {
            h
        };
        let h = self.w.linear(
            &h,
            &format!("{prefix}.pwconv2.weight"),
            Some(&format!("{prefix}.pwconv2.bias")),
        )?;
        let h = if self.w.has(&format!("{prefix}.gamma")) {
            let gamma = self.w.g(&format!("{prefix}.gamma"))?.reshape((1, 1, c))?;
            h.broadcast_mul(&gamma)?
        } else {
            h
        };
        let h = h.permute((0, 2, 1))?.contiguous()?; // [B, C, T]
        x.add(&h)
    }

    /// ConvNeXt-V2 Global Response Normalization over the time axis (channels-last
    /// `[B, T, C]`): `gx = ‖x‖_2(dim=T)`, `nx = gx / mean(gx, dim=C)`, then
    /// `gamma * (x * nx) + beta + x`.
    fn grn(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let gx = x.sqr()?.sum_keepdim(1)?.sqrt()?; // [B, 1, C]
        let nx = gx.broadcast_div(&(gx.mean_keepdim(D::Minus1)? + 1e-6)?)?; // [B, 1, C]
        let gamma = self.w.g(&format!("{prefix}.grn.gamma"))?;
        let beta = self.w.g(&format!("{prefix}.grn.beta"))?;
        let c = x.dim(D::Minus1)?;
        let gamma = gamma.reshape((1, 1, c))?;
        let beta = beta.reshape((1, 1, c))?;
        let scaled = x.broadcast_mul(&nx)?.broadcast_mul(&gamma)?.broadcast_add(&beta)?;
        x.add(&scaled)
    }

    // --- factorized RVQ `from_codes` (decode) ---------------------------------

    /// One quantizer's `from_codes`: gather the raw codebook embedding for `codes`,
    /// then `out_proj` (1×1 conv `cbdim → latent`). Returns `[1, latent, T]`.
    fn decode_codebook(&self, prefix: &str, codes: &[u32]) -> Result<Tensor> {
        let t = codes.len();
        let cb = self.w.g(&format!("{prefix}.codebook.weight"))?; // [size, cbdim]
        let idx = Tensor::from_vec(codes.to_vec(), (t,), self.dev())?;
        let zp = cb.index_select(&idx, 0)?; // [T, cbdim]
        let zp = zp.transpose(0, 1)?.unsqueeze(0)?.contiguous()?; // [1, cbdim, T]
        self.causal_conv1d(
            &zp,
            &format!("{prefix}.out_proj.weight"),
            &format!("{prefix}.out_proj.bias"),
            1,
            1,
            1,
            1,
        )
    }

    /// `upsample`: 2 × (`ConvTranspose(factor)` + `ConvNeXtBlock`), restoring ×4 the
    /// time resolution. Built in reversed `downsample_factor` order (key index 0 first).
    fn upsample(&self, z: &Tensor) -> Result<Tensor> {
        let mut z = z.clone();
        for (s, &factor) in DOWNSAMPLE_FACTOR.iter().rev().enumerate() {
            z = self.causal_transpose1d(
                &z,
                &format!("upsample.{s}.0.conv.weight"),
                &format!("upsample.{s}.0.conv.bias"),
                factor,
                factor,
            )?;
            z = self.convnext(&z, &format!("upsample.{s}.1"))?;
        }
        Ok(z)
    }

    /// The **EVA-GAN generator** decode path (causal). Structurally a DAC-style causal
    /// upsampler: first conv → per-stride `DecoderBlock` (`Snake → ConvTranspose →
    /// 3 dilated {1,3,9} ResidualUnits`) → Snake → conv → Tanh.
    //
    // PARITY: this is the s1 modded-DAC decoder shape used as a stand-in for the EVA-GAN
    // generator. The real EVA-GAN generator differs (multi-receptive-field residual
    // blocks, a different channel schedule, possibly anti-aliased/AMP activations rather
    // than plain Snake, and a different final activation). Reconcile the block layout +
    // every key name against `codec.pth` on-box before trusting the waveform.
    fn run_generator(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.causal_conv1d(
            z,
            "decoder.model.0.conv.weight",
            "decoder.model.0.conv.bias",
            7,
            1,
            1,
            1,
        )?;
        for (i, &stride) in self.cfg.decoder_rates.iter().enumerate() {
            let prefix = format!("decoder.model.{}", i + 1);
            x = self.snake(&x, &format!("{prefix}.block.0.alpha"))?;
            x = self.causal_transpose1d(
                &x,
                &format!("{prefix}.block.1.conv.weight"),
                &format!("{prefix}.block.1.conv.bias"),
                2 * stride,
                stride,
            )?;
            x = self.residual_unit(&x, &format!("{prefix}.block.2"), 1)?;
            x = self.residual_unit(&x, &format!("{prefix}.block.3"), 3)?;
            x = self.residual_unit(&x, &format!("{prefix}.block.4"), 9)?;
        }
        let final_idx = self.cfg.decoder_rates.len() + 1;
        x = self.snake(&x, &format!("decoder.model.{final_idx}.alpha"))?;
        x = self.causal_conv1d(
            &x,
            &format!("decoder.model.{}.conv.weight", final_idx + 1),
            &format!("decoder.model.{}.conv.bias", final_idx + 1),
            7,
            1,
            1,
            1,
        )?;
        x.tanh()
    }

    /// Decode a `[num_codebooks, T]` (or `[1, num_codebooks, T]`) code matrix to a mono
    /// `[n_samples]` waveform.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let codes = if codes.rank() == 3 {
            codes.squeeze(0)?
        } else {
            codes.clone()
        };
        let n_cb = codes.dim(0)?;
        let t = codes.dim(1)?;
        let host: Vec<u32> = codes.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?;
        let row = |r: usize| -> Vec<u32> { (0..t).map(|c| host[r * t + c]).collect() };

        let sem_max = (self.cfg.semantic_size - 1) as u32;
        let res_max = (self.cfg.residual_size - 1) as u32;

        // Factorized RVQ from_codes: semantic codebook 0 + 9 residual codebooks.
        let sem: Vec<u32> = row(0).iter().map(|&c| c.min(sem_max)).collect();
        let mut z = self.decode_codebook("semantic_quantizer.quantizers.0", &sem)?;
        for i in 0..(n_cb - 1) {
            let res: Vec<u32> = row(i + 1).iter().map(|&c| c.min(res_max)).collect();
            let zr = self.decode_codebook(&format!("quantizer.quantizers.{i}"), &res)?;
            z = (z + zr)?;
        }

        // Decoder-side Mimi bottleneck (after RVQ), then ConvNeXt upsample, then the
        // EVA-GAN generator.
        let z = self.maybe_bottleneck(&z, "decoder_transformer")?;
        let z = self.upsample(&z)?;
        let wav = self.run_generator(&z)?; // [1, 1, L]
        wav.reshape((wav.dim(D::Minus1)?,))
    }

    // --- Encoder + RVQ analysis (encode / cloning) ----------------------------

    /// One encoder stage (`EncoderBlock`): 3 dilated `{1,3,9}` ResidualUnits → Snake →
    /// downsample conv.
    fn encoder_block(&self, x: &Tensor, prefix: &str, stride: usize) -> Result<Tensor> {
        let mut x = self.residual_unit(x, &format!("{prefix}.block.0"), 1)?;
        x = self.residual_unit(&x, &format!("{prefix}.block.1"), 3)?;
        x = self.residual_unit(&x, &format!("{prefix}.block.2"), 9)?;
        x = self.snake(&x, &format!("{prefix}.block.3.alpha"))?;
        x = self.causal_conv1d(
            &x,
            &format!("{prefix}.block.4.conv.weight"),
            &format!("{prefix}.block.4.conv.bias"),
            2 * stride,
            stride,
            1,
            1,
        )?;
        Ok(x)
    }

    /// The causal DAC `Encoder` forward (512×): first conv → 4 `EncoderBlock`s → Snake →
    /// latent conv. The Mimi bottleneck + ConvNeXt downsample are applied by the caller.
    fn run_encoder(&self, wav: &Tensor) -> Result<Tensor> {
        let mut x = self.causal_conv1d(
            wav,
            "encoder.block.0.conv.weight",
            "encoder.block.0.conv.bias",
            7,
            1,
            1,
            1,
        )?;
        let mut dim = ENCODER_DIM;
        for (i, &stride) in self.cfg.encoder_rates.iter().enumerate() {
            dim *= 2;
            let _ = dim;
            x = self.encoder_block(&x, &format!("encoder.block.{}", i + 1), stride)?;
        }
        x = self.snake(
            &x,
            &format!("encoder.block.{}.alpha", self.cfg.encoder_rates.len() + 1),
        )?;
        self.causal_conv1d(
            &x,
            &format!("encoder.block.{}.conv.weight", self.cfg.encoder_rates.len() + 2),
            &format!("encoder.block.{}.conv.bias", self.cfg.encoder_rates.len() + 2),
            3,
            1,
            1,
            1,
        )
    }

    /// `downsample`: 2 × (`Conv(factor, stride=factor)` + `ConvNeXtBlock`), ×4 reduction.
    fn downsample(&self, z: &Tensor) -> Result<Tensor> {
        let mut z = z.clone();
        for (s, &factor) in DOWNSAMPLE_FACTOR.iter().enumerate() {
            z = self.causal_conv1d(
                &z,
                &format!("downsample.{s}.0.conv.weight"),
                &format!("downsample.{s}.0.conv.bias"),
                factor,
                factor,
                1,
                1,
            )?;
            z = self.convnext(&z, &format!("downsample.{s}.1"))?;
        }
        Ok(z)
    }

    /// One quantizer's analysis step: `in_proj` → L2-normalized nearest-codebook
    /// search → `(codes, out_proj(decode_code(codes)))`. Returns `(codes[T], z_q[1,latent,T])`.
    fn quantize_one(&self, prefix: &str, z: &Tensor) -> Result<(Vec<u32>, Tensor)> {
        let ze = self.causal_conv1d(
            z,
            &format!("{prefix}.in_proj.weight"),
            &format!("{prefix}.in_proj.bias"),
            1,
            1,
            1,
            1,
        )?; // [1, cbdim, T]
        let t = ze.dim(D::Minus1)?;
        let enc = ze.squeeze(0)?.transpose(0, 1)?.contiguous()?; // [T, cbdim]
        let enc = l2_normalize(&enc)?;
        let cb = self.w.g(&format!("{prefix}.codebook.weight"))?; // [size, cbdim]
        let cbn = l2_normalize(&cb)?;
        let sim = enc.matmul(&cbn.t()?)?; // [T, size]
        let size = cb.dim(0)?;
        let sim_host: Vec<f32> = sim.flatten_all()?.to_vec1()?;
        let mut codes = vec![0u32; t];
        for (ti, code) in codes.iter_mut().enumerate() {
            let row = &sim_host[ti * size..(ti + 1) * size];
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            *code = best as u32;
        }
        let zq = self.decode_codebook(prefix, &codes)?;
        Ok((codes, zq))
    }

    /// Encode a mono `[n_samples]` (or `[1, 1, n_samples]`) 44.1 kHz waveform to a
    /// `[num_codebooks, T]` code matrix (the reference cloning path).
    pub fn encode(&self, wav: &Tensor) -> Result<Tensor> {
        let wav = match wav.rank() {
            1 => wav.reshape((1, 1, wav.dim(0)?))?,
            2 => wav.unsqueeze(1)?,
            _ => wav.clone(),
        };
        // Pad to a multiple of the total hop (frame_hop == 2048×).
        let length = wav.dim(D::Minus1)?;
        let fl = self.cfg.frame_hop;
        let right = (fl - (length % fl)) % fl;
        let wav = if right > 0 {
            wav.pad_with_zeros(D::Minus1, 0, right)?
        } else {
            wav
        };

        let z = self.run_encoder(&wav)?; // [1, latent, T_512]
        let z = self.downsample(&z)?; // [1, latent, T_2048]
        // Encoder-side Mimi bottleneck (before RVQ).
        let z = self.maybe_bottleneck(&z, "encoder_transformer")?;

        // Factorized RVQ: 1 semantic codebook, then 9 residual codebooks on the residual.
        let (sem_codes, z_sem) = self.quantize_one("semantic_quantizer.quantizers.0", &z)?;
        let mut residual = (z - z_sem)?;
        let mut rows: Vec<Vec<u32>> = vec![sem_codes];
        for i in 0..N_RESIDUAL {
            let (codes, zq) = self.quantize_one(&format!("quantizer.quantizers.{i}"), &residual)?;
            residual = (residual - zq)?;
            rows.push(codes);
        }

        let t = rows[0].len();
        let n = rows.len();
        let mut flat = vec![0u32; n * t];
        for (ci, r) in rows.iter().enumerate() {
            for (ti, &c) in r.iter().enumerate() {
                flat[ci * t + ti] = c;
            }
        }
        Tensor::from_vec(flat, (n, t), self.dev())
    }

    // --- Mimi-style causal sliding-window transformer bottleneck --------------

    /// Apply the Mimi bottleneck transformer at `prefix` (`{prefix}.layers.N.*`) iff the
    /// checkpoint carries it; otherwise identity. Channels-first in/out; channels-last
    /// window-causal RoPE transformer inside (`Attention` + `LayerScale`d residual +
    /// SwiGLU), matching the s1 codec's `WindowLimitedTransformer`.
    fn maybe_bottleneck(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        if !self.w.has(&format!("{prefix}.layers.0.attention.wqkv.weight"))
            && !self.w.has(&format!("{prefix}.layers.0.attention.wq.weight"))
        {
            // No explicit bottleneck weights → treat as identity (real codec may fold the
            // bottleneck elsewhere). PARITY: confirm the bottleneck key prefix on-box.
            return Ok(x.clone());
        }
        let dim = x.dim(1)?;
        let xt = x.transpose(1, 2)?.contiguous()?; // [B, T, dim]
        let t = xt.dim(1)?;
        let head_dim = BOTTLENECK_HEAD_DIM;
        let n_head = dim / head_dim;
        let (cos, sin) = precompute_rope(t, head_dim, DAC_ROPE_BASE, self.dev())?;
        let mask = window_causal_mask(t, WINDOW_SIZE, self.dev())?;
        let shape = AttnShape {
            n_head,
            n_local_heads: n_head,
            head_dim,
            qkv_bias: false,
            o_bias: false,
            qk_norm: false,
            eps: DAC_NORM_EPS,
        };
        let mut cache = KvCache::new(BOTTLENECK_LAYERS);
        let mut h = xt;
        for l in 0..BOTTLENECK_LAYERS {
            let p = format!("{prefix}.layers.{l}");
            if !self.w.has(&format!("{p}.attention_norm.weight")) {
                break; // fewer layers than the default cap → stop at the last present.
            }
            let r = h.clone();
            let hn = self.w.rms_norm(&h, &format!("{p}.attention_norm.weight"), DAC_NORM_EPS)?;
            let a = attention(
                &self.w,
                &format!("{p}.attention"),
                &hn,
                &cos,
                &sin,
                Some(&mask),
                shape,
                &mut cache,
                l,
            )?;
            // Optional LayerScale on the attention branch.
            let a = if self.w.has(&format!("{p}.attention_layer_scale.gamma")) {
                let g = self
                    .w
                    .g(&format!("{p}.attention_layer_scale.gamma"))?
                    .reshape((1, 1, dim))?;
                a.broadcast_mul(&g)?
            } else {
                a
            };
            h = (r + a)?;
            let r = h.clone();
            let hn = self.w.rms_norm(&h, &format!("{p}.ffn_norm.weight"), DAC_NORM_EPS)?;
            let f = swiglu(&self.w, &format!("{p}.feed_forward"), &hn)?;
            let f = if self.w.has(&format!("{p}.ffn_layer_scale.gamma")) {
                let g2 = self
                    .w
                    .g(&format!("{p}.ffn_layer_scale.gamma"))?
                    .reshape((1, 1, dim))?;
                f.broadcast_mul(&g2)?
            } else {
                f
            };
            h = (r + f)?;
        }
        let h = if self.w.has(&format!("{prefix}.norm.weight")) {
            self.w.rms_norm(&h, &format!("{prefix}.norm.weight"), DAC_NORM_EPS)?
        } else {
            h
        };
        h.transpose(1, 2)?.contiguous()
    }
}

// --- free helpers -------------------------------------------------------------

/// `get_extra_padding_for_conv1d`: the right-side alignment padding for a causal conv.
fn extra_padding(length: usize, kernel_eff: usize, stride: usize, padding_total: usize) -> usize {
    let n_frames =
        (length as f64 - kernel_eff as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal =
        (n_frames.ceil() - 1.0) * stride as f64 + (kernel_eff as f64 - padding_total as f64);
    let extra = ideal - length as f64;
    if extra <= 0.0 {
        0
    } else {
        extra.round() as usize
    }
}

/// Channels-last `LayerNorm` over the last dim.
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    xn.broadcast_mul(w)?.broadcast_add(b)
}

/// `F.normalize(x, p=2, dim=-1)` with eps `1e-12`.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    x.broadcast_div(&norm.affine(1.0, NORM_EPS)?)
}

/// Additive window-limited causal mask `[t, t]`: query `i` may attend key `j` iff
/// `i - window + 1 <= j <= i`, else `-inf`.
fn window_causal_mask(t: usize, window: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            let too_late = j > i;
            let too_old = i >= window && j < i - window + 1;
            if too_late || too_old {
                data[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (t, t), dev)
}
