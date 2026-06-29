//! The s2 **causal-DAC** RVQ codec: `[10, T]` codes ↔ 44.1 kHz waveform.
//!
//! Reconciled against the REAL `codec.pth` state dict (541 keys, dumped on-box). The
//! top-level prefixes are `encoder.`, `quantizer.`, `decoder.`; the codec is a strictly
//! causal Descript-Audio-Codec (DAC) — Snake + dilated ResidualUnits + weight-normed
//! convs — with a ConvNeXt down/up-sampler and full-causal Transformer bottlenecks.
//!
//! ## Confirmed structure (mapped key-for-key)
//! * **Encoder** `encoder.block.{0..6}`: `block.0` first conv (1→64, k7); `block.{1..4}`
//!   are DAC EncoderBlocks (3 dilated{1,3,9} ResidualUnits → Snake → strided downsample
//!   conv), channel schedule 64→128→256→512→1024, strides [2,4,8,8] (×512); `block.5`
//!   Snake; `block.6` latent conv (1024→1024, k3). The **deepest** block carries a
//!   4-layer Transformer at `block.4.block.5` (applied after that block's downsample
//!   conv, at 1024-dim / 512× resolution).
//! * **Quantizer** `quantizer.*`: 1 semantic RVQ (`semantic_quantizer.quantizers.0`,
//!   codebook 4096×8) + 9 residual RVQs (`quantizer.quantizers.0..8`, codebook 1024×8),
//!   each factorized (`in_proj` 1024→8 / `out_proj` 8→1024, DAC `in_proj` → L2 nearest
//!   codebook → `out_proj` math). A ConvNeXt ×4 down/up-sampler (`downsample`/`upsample`
//!   .{0,1}, each = plain strided conv + ConvNeXt block) sits around the RVQ, and two
//!   8-layer full-causal Transformers (`pre_module` after downsample/before RVQ,
//!   `post_module` after RVQ/before upsample) form the bottleneck.
//! * **Decoder** `decoder.model.{0..6}`: DAC Decoder (first conv 1024→1536 → 4
//!   DecoderBlocks Snake→ConvTranspose→3 dilated{1,3,9} ResidualUnits, channels
//!   1536→768→384→192→96, strides [8,8,4,2] → Snake → final conv 96→1, k7) + a final
//!   paramless `Tanh` (index 7, not in the state dict).
//!
//! ## Weight-norm (folded at load, see `load.rs`)
//! * RVQ `in_proj`/`out_proj`: old `weight_g`/`weight_v`. Encoder/decoder convs: new
//!   `parametrizations.weight.original{0,1}`. Plain convs (`downsample/upsample.*.0.conv`,
//!   ConvNeXt `dwconv`/`pwconv`, the Transformer linears) have no weight-norm.
//!
//! ## Least-confident parts (`// PARITY:`-flagged below)
//! 1. The bottleneck/encoder Transformer is **full causal** (the skipped `causal_mask`
//!    bool buffers are triangular, matching the fish reference `register_buffer`); the
//!    RoPE base (10_000) and RMSNorm eps are best-effort.
//! 2. Whether the encode-side RVQ residual order / semantic+residual summation exactly
//!    matches the reference `DownsampleResidualVectorQuantize.forward`.

use candle_core::{DType, Device, Result, Tensor, D};

use super::nn::{attention, causal_mask_at, precompute_rope, swiglu, AttnShape, KvCache, Weights};
use crate::common::config::CodecConfig;

// --- s2 causal-DAC structural constants ---------------------------------------
// These are now reconciled against the REAL `codec.pth` state dict (541 keys): the
// top-level prefixes are `encoder.`, `quantizer.`, `decoder.`, and the codec is a
// strictly-causal Descript-DAC (Snake + dilated ResidualUnits) with a ConvNeXt
// down/up-sampler and full-causal Transformer bottlenecks (`quantizer.pre_module`/
// `post_module`) plus a Transformer inside the deepest encoder block (`block.4.block.5`).

/// Encoder base channels (`encoder.block.0.conv` out = 64).
const ENCODER_DIM: usize = 64;
/// Number of **residual** RVQ codebooks (`quantizer.quantizer.quantizers.0..8`); the
/// semantic codebook (`quantizer.semantic_quantizer.quantizers.0`) is separate.
const N_RESIDUAL: usize = 9;
/// The ConvNeXt down/up-sample factors (`quantizer.downsample.{0,1}` k2/stride2 each),
/// product == ×4 on top of the DAC encoder's 512× ⇒ 2048× total hop.
const DOWNSAMPLE_FACTOR: [usize; 2] = [2, 2];
/// Codec Transformer head dim (`freqs_cis (.., 32, 2)` ⇒ head_dim/2 == 32 ⇒ 64; the
/// fused `attention.wqkv (3072, 1024)` ⇒ q=k=v=1024 ⇒ 16 heads × 64).
const TF_HEAD_DIM: usize = 64;
/// Codec Transformer RoPE base. PARITY: the fish reference uses 10_000 for the codec
/// Transformer; confirm on-box (the LM backbone uses a larger base).
const TF_ROPE_BASE: f64 = 10_000.0;
/// Codec Transformer / norm epsilon. PARITY: confirm the codec RMSNorm eps on-box.
const TF_NORM_EPS: f64 = 1e-5;
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
        // PARITY: the response-norm reduction runs in f32 for bf16-stability; `nx` is
        // cast back to `x`'s dtype before the elementwise combine. Identity for f32.
        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let gx = xf.sqr()?.sum_keepdim(1)?.sqrt()?; // [B, 1, C]
        let nx = gx
            .broadcast_div(&(gx.mean_keepdim(D::Minus1)? + 1e-6)?)?
            .to_dtype(dt)?; // [B, 1, C]
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
                &format!("quantizer.upsample.{s}.0.conv.weight"),
                &format!("quantizer.upsample.{s}.0.conv.bias"),
                factor,
                factor,
            )?;
            z = self.convnext(&z, &format!("quantizer.upsample.{s}.1"))?;
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

        // Factorized RVQ from_codes: semantic codebook 0 + 9 residual codebooks, all
        // summed into the shared 1024-dim latent.
        let sem: Vec<u32> = row(0).iter().map(|&c| c.min(sem_max)).collect();
        let mut z = self.decode_codebook("quantizer.semantic_quantizer.quantizers.0", &sem)?;
        for i in 0..(n_cb - 1) {
            let res: Vec<u32> = row(i + 1).iter().map(|&c| c.min(res_max)).collect();
            let zr =
                self.decode_codebook(&format!("quantizer.quantizer.quantizers.{i}"), &res)?;
            z = (z + zr)?;
        }

        // Decode bottleneck: `post_module` Transformer (after RVQ, before upsample), then
        // the ConvNeXt upsample, then the DAC decoder.
        let z = self.transformer("quantizer.post_module", &z)?;
        let z = self.upsample(&z)?;
        let wav = self.run_generator(&z)?; // [1, 1, L]
        // PARITY: return the waveform in f32 regardless of the compute dtype — the WAV
        // writer / `to_vec1::<f32>` consumers expect f32. Identity on the f32 CPU path.
        wav.reshape((wav.dim(D::Minus1)?,))?.to_dtype(DType::F32)
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
    /// latent conv. The deepest EncoderBlock additionally carries a Transformer at
    /// `encoder.block.<i>.block.5` (applied after its downsample conv). The ConvNeXt
    /// downsample + the `pre_module` bottleneck are applied by the caller (encode).
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
            let bp = format!("encoder.block.{}", i + 1);
            x = self.encoder_block(&x, &bp, stride)?;
            // The deepest EncoderBlock embeds a full-causal Transformer (`block.5`),
            // applied after the downsample conv at 1024-dim / 512× resolution.
            if self.w.has(&format!("{bp}.block.5.norm.weight")) {
                x = self.transformer(&format!("{bp}.block.5"), &x)?;
            }
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
                &format!("quantizer.downsample.{s}.0.conv.weight"),
                &format!("quantizer.downsample.{s}.0.conv.bias"),
                factor,
                factor,
                1,
                1,
            )?;
            z = self.convnext(&z, &format!("quantizer.downsample.{s}.1"))?;
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
        // PARITY: the nearest-codebook argmin reads f32 similarities — cast up before the
        // host copy (bf16 `to_vec1::<f32>` would fail and the argmin wants f32 precision).
        let sim_host: Vec<f32> = sim.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
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
        // Encode-side bottleneck: `pre_module` Transformer (after downsample, before RVQ).
        let z = self.transformer("quantizer.pre_module", &z)?;

        // Factorized RVQ: 1 semantic codebook, then 9 residual codebooks on the residual.
        let (sem_codes, z_sem) =
            self.quantize_one("quantizer.semantic_quantizer.quantizers.0", &z)?;
        let mut residual = (z - z_sem)?;
        let mut rows: Vec<Vec<u32>> = vec![sem_codes];
        for i in 0..N_RESIDUAL {
            let (codes, zq) =
                self.quantize_one(&format!("quantizer.quantizer.quantizers.{i}"), &residual)?;
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

    // --- full-causal RoPE Transformer bottleneck ------------------------------

    /// Apply the codec Transformer rooted at `prefix` (`{prefix}.layers.N.*` + a final
    /// `{prefix}.norm`). Used for `quantizer.pre_module` / `quantizer.post_module` (8
    /// layers) and the deepest encoder block's `encoder.block.4.block.5` (4 layers); the
    /// layer count is discovered from the checkpoint. Channels-first in/out; inside it is
    /// a channels-last pre-norm RoPE Transformer (RMSNorm → `Attention` → LayerScale
    /// residual → RMSNorm → SwiGLU → LayerScale residual), matching the fish reference
    /// `TransformerBlock`. Attention is **full causal** (the `causal_mask` bool buffer in
    /// the checkpoint is a triangular `register_buffer`, recomputed here since Candle's
    /// pickle reader skips BoolStorage).
    //
    // PARITY: RoPE base (`TF_ROPE_BASE`) + RMSNorm eps (`TF_NORM_EPS`) are best-effort;
    // the `freqs_cis` buffer in the checkpoint is ignored in favour of recomputation.
    fn transformer(&self, prefix: &str, x: &Tensor) -> Result<Tensor> {
        // Discover the layer count from the checkpoint (8 for pre/post_module, 4 for the
        // encoder block.4 Transformer).
        let mut n_layers = 0usize;
        while self
            .w
            .has(&format!("{prefix}.layers.{n_layers}.attention_norm.weight"))
        {
            n_layers += 1;
        }
        if n_layers == 0 {
            return Ok(x.clone());
        }
        let dim = x.dim(1)?;
        let xt = x.transpose(1, 2)?.contiguous()?; // [B, T, dim]
        let t = xt.dim(1)?;
        let head_dim = TF_HEAD_DIM;
        let n_head = dim / head_dim;
        let dt = self.w.dt;
        let (cos, sin) = precompute_rope(t, head_dim, TF_ROPE_BASE, self.dev(), dt)?;
        let mask = causal_mask_at(0, t, self.dev(), dt)?; // full lower-triangular causal mask
        let shape = AttnShape {
            n_head,
            n_local_heads: n_head,
            head_dim,
            qkv_bias: false,
            o_bias: false,
            qk_norm: false,
            eps: TF_NORM_EPS,
        };
        let mut cache = KvCache::new(n_layers);
        let mut h = xt;
        for l in 0..n_layers {
            let p = format!("{prefix}.layers.{l}");
            let r = h.clone();
            let hn = self.w.rms_norm(&h, &format!("{p}.attention_norm.weight"), TF_NORM_EPS)?;
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
            let g = self
                .w
                .g(&format!("{p}.attention_layer_scale.gamma"))?
                .reshape((1, 1, dim))?;
            let a = a.broadcast_mul(&g)?;
            h = (r + a)?;
            let r = h.clone();
            let hn = self.w.rms_norm(&h, &format!("{p}.ffn_norm.weight"), TF_NORM_EPS)?;
            let f = swiglu(&self.w, &format!("{p}.feed_forward"), &hn)?;
            let g2 = self
                .w
                .g(&format!("{p}.ffn_layer_scale.gamma"))?
                .reshape((1, 1, dim))?;
            let f = f.broadcast_mul(&g2)?;
            h = (r + f)?;
        }
        let h = self.w.rms_norm(&h, &format!("{prefix}.norm.weight"), TF_NORM_EPS)?;
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
//
// PARITY: the mean/variance reduction runs in f32 for bf16-stability; the normalised
// activation is cast back to `x`'s dtype before the (dtype-`dt`) affine. Identity for f32.
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xc.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(dt)?;
    xn.broadcast_mul(w)?.broadcast_add(b)
}

/// `F.normalize(x, p=2, dim=-1)` with eps `1e-12`.
//
// PARITY: the L2 norm reduction runs in f32 for bf16-stability, then the result is cast
// back to `x`'s dtype. Identity for the f32 CPU path.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let norm = xf.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    xf.broadcast_div(&norm.affine(1.0, NORM_EPS)?)?.to_dtype(dt)
}
