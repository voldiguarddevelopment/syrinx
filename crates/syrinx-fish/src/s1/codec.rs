//! The s1 **modded-DAC** RVQ codec: `[10, T]` codes ↔ 44.1 kHz waveform.
//!
//! Ports `modded_dac.py` (`DAC`, `Encoder`/`Decoder`, `Snake1d`, `ResidualUnit`,
//! `CausalWNConv1d`/`CausalWNConvTranspose1d`, `WindowLimitedTransformer`) and
//! `fish_speech/models/dac/rvq.py` (`DownsampleResidualVectorQuantize` — factorized
//! RVQ with `codebook_dim=8`, a 1-codebook **semantic** quantizer + a 9-codebook
//! **residual** quantizer, and a ×4 down/up-sample pair).
//!
//! ## Decode (the priority path: `from_indices`)
//! `quantizer.decode(codes)` → `upsample(×4)` → `Decoder`:
//!   * factorized RVQ `from_codes`: per codebook, gather the raw codebook embedding
//!     `[cbdim]`, `out_proj` (1×1 conv `cbdim → latent`), sum semantic + 9 residual;
//!   * `upsample`: 2 × (`ConvTranspose` ×2 + `ConvNeXtBlock`) — restores the latent
//!     time resolution the decoder expects;
//!   * `Decoder`: a causal conv stack — `Conv(k7)`, 4 `DecoderBlock`s
//!     (`Snake → ConvTranspose(stride) → 3× dilated {1,3,9} ResidualUnit`), `Snake`,
//!     `Conv(k7)`, `Tanh`. **The decoder's window-limited transformer modules are
//!     disabled in the reference** (commented out of `DecoderBlock`), so decode is
//!     purely convolutional.
//!
//! ## Encode (reference cloning: `encode`)
//! `Encoder` (causal conv stack + per-stage window-limited transformers) →
//! `downsample(×4)` → factorized RVQ analysis (L2-normalized nearest-codebook search)
//! → `[10, T]` codes. Implemented in full; the per-stage transformer-layer counts are
//! the only on-box-confirmable numbers (`// PARITY:` below).
//!
//! ## Weight-norm
//! Every `WN*` conv ships weight-norm-parametrized (`weight_g`/`weight_v` or
//! `parametrizations.weight.original{0,1}`). [`super::load`] **folds** those into a
//! plain `.weight` at conversion time, so this module sees ordinary conv weights.

use candle_core::{DType, Device, Result, Tensor, D};

use super::nn::{attention, precompute_rope, swiglu, AttnShape, KvCache, Weights};
use crate::common::config::CodecConfig;

// --- modded-DAC structural constants (s1 `modded_dac_vq.yaml`) ----------------
// PARITY: confirm the DAC channel dims + per-stage transformer-layer counts from
// `openaudio-s1-mini/codec.pth` + `configs/modded_dac_vq.yaml` on-box.

/// Encoder base channels (`encoder_dim`).  // PARITY: confirm encoder_dim on-box.
const ENCODER_DIM: usize = 64;
// NOTE: the decoder base width (`decoder_dim` ≈ 1536) and the RVQ latent width
// (`latent_dim = encoder_dim · 2^len(encoder_rates)` = 1024) are read directly from
// the loaded conv weights, so they need no standalone constant — but both are on the
// // PARITY: list to confirm against `codec.pth` on-box.
/// Number of **residual** RVQ codebooks (the semantic codebook is separate).
const N_RESIDUAL: usize = 9; // PARITY: confirm n_codebooks on-box.
/// The factorized down/up-sample factors (`downsample_factor`), product == ×4.
const DOWNSAMPLE_FACTOR: [usize; 2] = [2, 2]; // PARITY: confirm downsample_factor.
/// Per-encoder-stage window-limited-transformer layer counts (`n_transformer_layers`,
/// one per `encoder_rates` stage). PARITY: the reference `Encoder` default is
/// `[0, 0, 4, 4]`; confirm the s1 codec's value on-box. Decode never uses these.
const ENCODER_TRANSFORMER_LAYERS: [usize; 4] = [0, 0, 4, 4];
/// Window-limited-attention window (`window_size`).  // PARITY: confirm on-box.
const WINDOW_SIZE: usize = 512;
/// modded-DAC transformer RoPE base.  // PARITY: confirm rope_base on-box.
const DAC_ROPE_BASE: f64 = 10_000.0;
/// modded-DAC transformer RMSNorm epsilon.  // PARITY: confirm norm_eps on-box.
const DAC_NORM_EPS: f64 = 1e-5;
/// `Snake1d` numerical epsilon (`(alpha + 1e-9).reciprocal()`).
const SNAKE_EPS: f64 = 1e-9;
/// `F.normalize` epsilon (p=2).
const NORM_EPS: f64 = 1e-12;

/// The loaded modded-DAC codec.
pub struct ModdedDac {
    w: Weights,
    cfg: CodecConfig,
}

impl ModdedDac {
    /// Build from a loaded codec weight bag + the resolved codec geometry.
    pub fn new(w: Weights, cfg: CodecConfig) -> Self {
        Self { w, cfg }
    }

    fn dev(&self) -> &Device {
        &self.w.dev
    }

    // --- conv primitives (CausalConvNet / CausalTransConvNet) -----------------

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
    /// `right = ceil(kernel - stride)`, `left = (kernel - stride) - right`.
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
        let right = pad; // ceil(pad) for integer pad
        let left = pad - right; // == 0
        let len = y.dim(D::Minus1)?;
        let kept = len - left - right;
        y.narrow(D::Minus1, left, kept)
    }

    /// `Snake1d`: `x + (alpha + 1e-9)^{-1} * sin(alpha * x)^2`, channel-wise alpha
    /// (stored `[1, C, 1]`; `alpha_logscale=False`).
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

    /// `ConvNeXtBlock`: depthwise causal `Conv(k7)` → channels-last `LayerNorm` →
    /// `Linear → GELU → Linear` → `gamma` scale → residual.
    fn convnext(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let c = x.dim(1)?;
        let h = self.causal_conv1d(
            x,
            &format!("{prefix}.dwconv.conv.weight"),
            &format!("{prefix}.dwconv.conv.bias"),
            7,
            1,
            1,
            c, // depthwise: groups = channels
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
        let h = self.w.linear(
            &h,
            &format!("{prefix}.pwconv2.weight"),
            Some(&format!("{prefix}.pwconv2.bias")),
        )?;
        let gamma = self.w.g(&format!("{prefix}.gamma"))?.reshape((1, 1, c))?;
        let h = h.broadcast_mul(&gamma)?;
        let h = h.permute((0, 2, 1))?.contiguous()?; // [B, C, T]
        x.add(&h)
    }

    // --- factorized RVQ `from_codes` (decode) ---------------------------------

    /// One quantizer's `from_codes`: gather the raw codebook embedding for `codes`
    /// (length `T`), then `out_proj` (1×1 conv `cbdim → latent`). Returns `[1, latent, T]`.
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

    /// The causal-conv `Decoder` (transformer modules disabled in the reference).
    fn run_decoder(&self, z: &Tensor) -> Result<Tensor> {
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

        // Semantic codebook 0 (clamped to its 4096 range), 9 residual codebooks (1024).
        let sem_max = (self.cfg.semantic_size - 1) as u32;
        let res_max = (self.cfg.residual_size - 1) as u32;

        let sem: Vec<u32> = row(0).iter().map(|&c| c.min(sem_max)).collect();
        let mut z = self.decode_codebook("semantic_quantizer.quantizers.0", &sem)?;
        for i in 0..(n_cb - 1) {
            let res: Vec<u32> = row(i + 1).iter().map(|&c| c.min(res_max)).collect();
            let zr = self.decode_codebook(&format!("quantizer.quantizers.{i}"), &res)?;
            z = (z + zr)?;
        }
        // post_module is Identity in the reference.
        let z = self.upsample(&z)?;
        let wav = self.run_decoder(&z)?; // [1, 1, L]
        wav.reshape((wav.dim(D::Minus1)?,))
    }

    // --- Encoder + RVQ analysis (encode) --------------------------------------

    /// One encoder stage (`EncoderBlock`): 3 dilated `{1,3,9}` ResidualUnits → Snake →
    /// downsample conv → optional window-limited transformer.
    fn encoder_block(
        &self,
        x: &Tensor,
        prefix: &str,
        stride: usize,
        n_t_layer: usize,
        out_dim: usize,
    ) -> Result<Tensor> {
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
        if n_t_layer > 0 {
            x = self.window_transformer(&x, &format!("{prefix}.block.5"), n_t_layer, out_dim)?;
        }
        Ok(x)
    }

    /// `WindowLimitedTransformer`: channels-first → window-causal RoPE transformer
    /// (`Attention` + `LayerScale`d residual + SwiGLU) → channels-first. `input_proj` /
    /// `output_proj` are `Identity` (s1 stages keep `input_dim == config.dim`).
    fn window_transformer(
        &self,
        x: &Tensor,
        prefix: &str,
        n_layer: usize,
        dim: usize,
    ) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?; // [B, T, dim]
        let t = xt.dim(1)?;
        let head_dim = 64;
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
        let mut cache = KvCache::new(n_layer);
        let mut h = xt;
        for l in 0..n_layer {
            let p = format!("{prefix}.layers.{l}");
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
            let g = self.w.g(&format!("{p}.attention_layer_scale.gamma"))?.reshape((1, 1, dim))?;
            let a = a.broadcast_mul(&g)?;
            h = (r + a)?;
            let r = h.clone();
            let hn = self.w.rms_norm(&h, &format!("{p}.ffn_norm.weight"), DAC_NORM_EPS)?;
            let f = swiglu(&self.w, &format!("{p}.feed_forward"), &hn)?;
            let g2 = self.w.g(&format!("{p}.ffn_layer_scale.gamma"))?.reshape((1, 1, dim))?;
            let f = f.broadcast_mul(&g2)?;
            h = (r + f)?;
        }
        let h = self.w.rms_norm(&h, &format!("{prefix}.norm.weight"), DAC_NORM_EPS)?;
        h.transpose(1, 2)?.contiguous()
    }

    /// The `Encoder` forward: first conv → 4 `EncoderBlock`s → Snake → latent conv.
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
            let n_t = *ENCODER_TRANSFORMER_LAYERS.get(i).unwrap_or(&0);
            x = self.encoder_block(&x, &format!("encoder.block.{}", i + 1), stride, n_t, dim)?;
        }
        x = self.snake(&x, &format!("encoder.block.{}.alpha", self.cfg.encoder_rates.len() + 1))?;
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
        // decode_code: raw (unnormalized) codebook embedding → out_proj.
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
        // Pad to a multiple of frame_length (= hop * 4 = frame_hop).
        let length = wav.dim(D::Minus1)?;
        let fl = self.cfg.frame_hop;
        let right = (fl - (length % fl)) % fl;
        let wav = if right > 0 {
            wav.pad_with_zeros(D::Minus1, 0, right)?
        } else {
            wav
        };

        let z = self.run_encoder(&wav)?; // [1, latent, T_lat]
        let z = self.downsample(&z)?; // [1, latent, T_ds]

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

/// Channels-last `LayerNorm` over the last dim: `(x - mean) / sqrt(var + eps) * w + b`.
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
