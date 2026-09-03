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

/// Left context, in codec frames, that the synthesis conv stack needs for a chunk of its
/// output to be **identical** to the one-shot decode.
///
/// Everything after the `post_module` bottleneck ([`EvaGanDac::synthesize`]: the ConvNeXt
/// `upsample` + the DAC generator) is strictly causal and contains no right-side padding
/// (every stride-1 causal conv has `extra_padding == 0`, and `causal_transpose1d` trims
/// exactly `kernel - stride` from the right), so an output sample depends only on frames
/// at or before it. Propagating the dependency interval backwards through the exact op
/// list —
///
/// ```text
///   upsample:  2 × [ ConvTranspose(k=2,s=2) ; ConvNeXt dwconv k7 d1 ]
///   generator: conv k7 d1
///              4 × [ ConvTranspose(k=2s,s) for s in 8,8,4,2 ; 3 × conv k7 d∈{1,3,9} ]
///              conv k7 d1
/// ```
///
/// — gives a **10-frame** left dependency (frame `-10` still reaches output sample `+298`;
/// frame `-11` stops at sample `-1750`), independent of the chunk size, with zero right
/// context. 16 is that bound plus margin; over-supplying context is free of correctness
/// risk (it only replaces zero-padding with true history) and costs 6 frames of recompute.
const DECODE_LEFT_CTX_FRAMES: usize = 16;

/// Left context, in frames, for the chunked part of the encoder.
///
/// Only [`EvaGanDac::encoder_convs`] is chunked — the first conv plus the four
/// `EncoderBlock`s. Walking its causal receptive field forward (`block.0.conv` k7, then
/// per block three dilated {1,3,9} k7 residual units and a k`2s`/s downsample, strides
/// 2·4·8·8):
///
/// ```text
///   encoder_convs   6 442 samples = 3.146 frames @ 2048 hop  ->  4 frames suffice
/// ```
///
/// Everything after it — the deepest block's Transformer, the k3 latent conv and
/// `quantizer.downsample` — runs one-shot over the concatenated result, so it
/// contributes no chunk-boundary dependency. (An earlier revision chunked those too and
/// needed 12.65 frames; the Transformer made that wrong at any finite context, which is
/// why the split exists.)
///
/// 16 is the 4-frame requirement with generous margin, at negligible cost: peak scales
/// with `chunk + ctx`, so 64+16 vs 64+4 is ~17 %. `encode_left_ctx_matches_receptive_field`
/// pins the real boundary by sweeping the constant and asserting it flips at 4.
const ENCODE_LEFT_CTX_FRAMES: usize = 16;

/// Default encode chunk length in frames. Bounds the encoder's `im2col` spike, which
/// otherwise scales with the whole reference: candle materialises `L × C_in × k`
/// elements per conv1d, so a 35 s reference at C=192/k7 costs 4.2 GB in bf16 (8.4 GB in
/// f32) in one allocation — enough to OOM a 12 GB card that has 8 GB free.
/// `SYRINX_FISH_CODEC_ENCODE_CHUNK` overrides; `0` selects the one-shot path.
const DEFAULT_ENCODE_CHUNK_FRAMES: usize = 64;

/// Resolve the encode chunk length from `SYRINX_FISH_CODEC_ENCODE_CHUNK`.
pub fn encode_chunk_frames_env() -> usize {
    std::env::var("SYRINX_FISH_CODEC_ENCODE_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_ENCODE_CHUNK_FRAMES)
}

/// Default synthesis chunk length in frames (≈ 3.0 s of audio at the 21.5 Hz frame rate).
///
/// Peak decode VRAM is ~4.2 MiB per in-flight frame (dominated by candle's `im2col`
/// buffer for the k7 dilated convs of the last two decoder blocks, which is `7 ×` the
/// activation), so a chunk bounds the spike at roughly
/// `4.2 MiB × (DEFAULT_DECODE_CHUNK_FRAMES + DECODE_LEFT_CTX_FRAMES)` ≈ 340 MiB regardless
/// of utterance length, for a 25 % recompute overhead on the codec stage only.
/// Override with `SYRINX_FISH_CODEC_CHUNK_FRAMES` (`0` selects the one-shot path).
const DEFAULT_DECODE_CHUNK_FRAMES: usize = 64;

/// Resolve the synthesis chunk length: `SYRINX_FISH_CODEC_CHUNK_FRAMES` if it parses,
/// else [`DEFAULT_DECODE_CHUNK_FRAMES`]. `0` means "one-shot" (the parity reference).
pub fn decode_chunk_frames_env() -> usize {
    std::env::var("SYRINX_FISH_CODEC_CHUNK_FRAMES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_DECODE_CHUNK_FRAMES)
}

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
        // Clamp against THIS table's own height, which is the only ceiling that is always
        // right. The reference clamps here too (`DownsampleResidualVectorQuantize.decode`),
        // and the two tables differ: the semantic codebook has 4096 entries but every
        // residual codebook has 1024. `CodecConfig::residual_size` is the fast AR head's
        // logit width (4096) — a *sampling* bound, not a table height — so using it here
        // let a residual code in 1024..=4095 reach `index_select` out of bounds.
        let ceiling = (cb.dim(0)? - 1) as u32;
        let clamped: Vec<u32> = codes.iter().map(|&c| c.min(ceiling)).collect();
        let idx = Tensor::from_vec(clamped, (t,), self.dev())?;
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

    /// Factorized RVQ `from_codes` + the `post_module` bottleneck: `[num_codebooks, T]`
    /// codes → the post-bottleneck latent `[1, latent, T]` that [`Self::synthesize`]
    /// turns into a waveform.
    ///
    /// This half is **not** chunkable: `post_module` is a full-causal Transformer whose
    /// position `t` attends over all of `0..=t`, so truncating its context would change
    /// the result. It is also cheap — `[1, 1024, T]` activations and a `T × T` attention
    /// map, well under 10 MiB for a minute of audio — so it always runs over the whole
    /// utterance.
    fn codes_to_latent(&self, codes: &Tensor) -> Result<Tensor> {
        let codes = if codes.rank() == 3 {
            codes.squeeze(0)?
        } else {
            codes.clone()
        };
        let n_cb = codes.dim(0)?;
        let t = codes.dim(1)?;
        let host: Vec<u32> = codes.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?;
        let row = |r: usize| -> Vec<u32> { (0..t).map(|c| host[r * t + c]).collect() };

        // Factorized RVQ from_codes: semantic codebook 0 + 9 residual codebooks, all
        // summed into the shared 1024-dim latent. Range-clamping lives in
        // `decode_codebook`, against each table's own height — see the note there.
        let mut z = self.decode_codebook("quantizer.semantic_quantizer.quantizers.0", &row(0))?;
        for i in 0..(n_cb - 1) {
            let zr =
                self.decode_codebook(&format!("quantizer.quantizer.quantizers.{i}"), &row(i + 1))?;
            z = (z + zr)?;
        }

        // Decode bottleneck: `post_module` Transformer (after RVQ, before upsample).
        self.transformer("quantizer.post_module", &z)
    }

    /// The strictly-causal synthesis stack: ConvNeXt `upsample` (×4) → DAC generator
    /// (×512) → `[1, 1, L * frame_hop]` for an `[1, latent, L]` latent.
    ///
    /// Every stage preserves the length exactly (`stride-1` causal convs are length-
    /// preserving with `extra_padding == 0`; `causal_transpose1d` yields exactly `L *
    /// stride`), so the output is exactly `L * self.cfg.frame_hop` samples. This is the
    /// only part of decode whose peak memory grows with the utterance length, and the
    /// only part [`Self::decode_chunked`] splits.
    fn synthesize(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.upsample(z)?;
        self.run_generator(&z) // [1, 1, L]
    }

    /// Decode a `[num_codebooks, T]` (or `[1, num_codebooks, T]`) code matrix to a mono
    /// `[n_samples]` waveform.
    ///
    /// Runs the chunked synthesis path with [`decode_chunk_frames_env`] frames per chunk
    /// so peak VRAM is bounded independently of the utterance length. Set
    /// `SYRINX_FISH_CODEC_CHUNK_FRAMES=0` (or call [`Self::decode_oneshot`]) for the
    /// one-shot reference path.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        self.decode_chunked(codes, decode_chunk_frames_env())
    }

    /// The **one-shot** decode — the parity reference. Materialises the whole waveform in
    /// a single pass through the generator; peak VRAM grows linearly with `T`.
    // Reachable in-tree only from the equivalence tests below and, at runtime, via
    // `SYRINX_FISH_CODEC_CHUNK_FRAMES=0`; kept as a named entry point so the parity path
    // can be called directly without touching the environment.
    #[allow(dead_code)]
    pub fn decode_oneshot(&self, codes: &Tensor) -> Result<Tensor> {
        self.decode_chunked(codes, 0)
    }

    /// Decode with the synthesis stack split into `chunk_frames`-frame pieces
    /// (`0` == one-shot).
    ///
    /// Each chunk `[a, b)` is synthesised from latent frames `[a - ctx, b)` with
    /// `ctx = `[`DECODE_LEFT_CTX_FRAMES`], and only the `[a, b)` portion of its output is
    /// kept. Because the stack is strictly causal with a proven 10-frame left dependency
    /// and no right dependency, the kept samples are the **same values** the one-shot path
    /// computes: the only thing the chunk boundary changes is what lies further left than
    /// the receptive field.
    ///
    /// (Exact in the mathematical sense. Bit-for-bit equality additionally requires the
    /// GEMM backend to accumulate identically for the two shapes; cuBLAS may pick a
    /// different tiling/split-k for a different `m`, so on CUDA expect agreement to
    /// rounding, not necessarily to the last bit. On CPU the reduction order is fixed by
    /// the output element, so it is bit-exact there.)
    pub fn decode_chunked(&self, codes: &Tensor, chunk_frames: usize) -> Result<Tensor> {
        let z = self.codes_to_latent(codes)?; // [1, latent, T]
        let t = z.dim(D::Minus1)?;

        // PARITY: return the waveform in f32 regardless of the compute dtype — the WAV
        // writer / `to_vec1::<f32>` consumers expect f32. Identity on the f32 CPU path.
        if chunk_frames == 0 || t <= chunk_frames {
            let wav = self.synthesize(&z)?; // [1, 1, L]
            return wav.reshape((wav.dim(D::Minus1)?,))?.to_dtype(DType::F32);
        }

        let hop = self.cfg.frame_hop;
        let mut parts: Vec<Tensor> = Vec::new();
        let mut a = 0usize;
        while a < t {
            let b = (a + chunk_frames).min(t);
            let lo = a.saturating_sub(DECODE_LEFT_CTX_FRAMES);
            let zc = z.narrow(D::Minus1, lo, b - lo)?.contiguous()?;
            let wav = self.synthesize(&zc)?; // [1, 1, (b - lo) * hop]
            let n = wav.dim(D::Minus1)?;
            let off = (a - lo) * hop;
            let keep = (b - a) * hop;
            if n < off + keep {
                return Err(candle_core::Error::Msg(format!(
                    "codec chunk synthesis produced {n} samples, expected at least \
                     {} for frames [{lo}, {b}) — synthesis stack is not length-exact",
                    off + keep
                )));
            }
            parts.push(
                wav.narrow(D::Minus1, off, keep)?
                    .reshape((keep,))?
                    .to_dtype(DType::F32)?,
            );
            a = b;
        }
        Tensor::cat(&parts, 0)
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
        let x = self.encoder_convs(wav)?;
        self.encoder_tail(&x)
    }

    /// The **chunkable** half of the encoder: the first conv and the four
    /// `EncoderBlock`s, i.e. everything that runs at (a fraction of) the full waveform
    /// rate and therefore dominates the `im2col` spike. Strictly causal convolutions
    /// only — no attention — so a finite left context reproduces it exactly.
    fn encoder_convs(&self, wav: &Tensor) -> Result<Tensor> {
        let mut x = self.causal_conv1d(
            wav,
            "encoder.block.0.conv.weight",
            "encoder.block.0.conv.bias",
            7,
            1,
            1,
            1,
        )?;
        for (i, &stride) in self.cfg.encoder_rates.iter().enumerate() {
            let bp = format!("encoder.block.{}", i + 1);
            x = self.encoder_block(&x, &bp, stride)?;
        }
        Ok(x)
    }

    /// The **one-shot** half of the encoder: the deepest block's full-causal Transformer
    /// (`encoder.block.<n>.block.5`), then Snake + the latent conv.
    ///
    /// This CANNOT be chunked. Causal attention at position `t` reads every position
    /// `0..=t`, so its dependency is unbounded to the left and no finite context
    /// reproduces it — chunking here changes the values, which the RVQ then turns into
    /// different codes. It is also cheap: it runs at 512× reduced rate, where the
    /// sequence is short and the cost is attention, not an `im2col` buffer.
    fn encoder_tail(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.cfg.encoder_rates.len();
        let bp = format!("encoder.block.{n}");
        let x = if self.w.has(&format!("{bp}.block.5.norm.weight")) {
            self.transformer(&format!("{bp}.block.5"), x)?
        } else {
            x.clone()
        };
        let x = self.snake(&x, &format!("encoder.block.{}.alpha", n + 1))?;
        self.causal_conv1d(
            &x,
            &format!("encoder.block.{}.conv.weight", n + 2),
            &format!("encoder.block.{}.conv.bias", n + 2),
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
    /// Run the encoder's length-scaled conv stack (`run_encoder` → `downsample`) over
    /// `wav` in `chunk_frames`-frame pieces, returning the full `[1, latent, T]` latent.
    /// `chunk_frames == 0` runs it one-shot (the parity reference).
    ///
    /// Chunk `[a, b)` is computed from samples `[(a - ctx) * hop, b * hop)` with
    /// `ctx = `[`ENCODE_LEFT_CTX_FRAMES`], and only the `[a, b)` latent frames are kept.
    /// The stack is strictly causal with a 12.65-frame left dependency and no right
    /// dependency, so a kept frame sees its entire receptive field and is the value the
    /// one-shot path computes. (Exact mathematically; on CUDA, agreement to rounding
    /// rather than bit-for-bit, since cuBLAS may tile a different `m` differently.)
    fn encode_latent_chunked(&self, wav: &Tensor, chunk_frames: usize) -> Result<Tensor> {
        self.encode_latent_ctx(wav, chunk_frames, ENCODE_LEFT_CTX_FRAMES)
    }

    /// [`Self::encode_latent_chunked`] with an explicit left context, so the equivalence
    /// test can sweep it and prove where the receptive field actually ends.
    fn encode_latent_ctx(
        &self,
        wav: &Tensor,
        chunk_frames: usize,
        ctx_frames: usize,
    ) -> Result<Tensor> {
        let hop = self.cfg.frame_hop;
        let n = wav.dim(D::Minus1)?;
        debug_assert_eq!(n % hop, 0, "encode input must be hop-aligned");
        let t = n / hop; // latent frames this waveform yields

        if chunk_frames == 0 || t <= chunk_frames {
            let z = self.run_encoder(wav)?;
            return self.downsample(&z);
        }


        // Chunk ONLY the conv cascade. `encoder_convs` outputs at 512× reduction, so a
        // frame there is `hop / 512` steps; the tail (attention) and `downsample` then
        // run once over the concatenated result.
        let sub = hop / 512;
        let mut parts: Vec<Tensor> = Vec::new();
        let mut a = 0usize;
        while a < t {
            let b = (a + chunk_frames).min(t);
            let lo = a.saturating_sub(ctx_frames);
            let seg = wav
                .narrow(D::Minus1, lo * hop, (b - lo) * hop)?
                .contiguous()?;
            let x = self.encoder_convs(&seg)?; // [1, C, (b - lo) * sub]
            let have = x.dim(D::Minus1)?;
            let off = (a - lo) * sub;
            let keep = (b - a) * sub;
            if have < off + keep {
                return Err(candle_core::Error::Msg(format!(
                    "codec chunk encode produced {have} steps, expected at least \
                     {} for frames [{lo}, {b}) — encoder is not length-exact",
                    off + keep
                )));
            }
            parts.push(x.narrow(D::Minus1, off, keep)?);
            a = b;
        }
        let x = Tensor::cat(&parts, D::Minus1)?.contiguous()?;
        let z = self.encoder_tail(&x)?;
        self.downsample(&z)
    }

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
        // The reference wav arrives as f32, but the encoder convs run in the codec compute
        // dtype (`dt` = f32 on CPU / bf16 on CUDA); cast so conv1d dtypes match.
        let wav = wav.to_dtype(self.w.dt)?;

        // The length-scaled conv stack (`run_encoder` + `downsample`) runs in chunks;
        // `pre_module` and the RVQ then see the whole latent. This is the mirror of the
        // decode split: there, `codes_to_latent`/`post_module` stays one-shot and only
        // the generator is chunked. `pre_module` is full-causal attention over the
        // sequence, so it must not be chunked — but it runs at 1/2048 the sample rate,
        // where length costs nothing.
        let z = self.encode_latent_chunked(&wav, encode_chunk_frames_env())?;
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

// --- chunked-decode equivalence ------------------------------------------------
//
// These run on the **CPU f32** path against a synthetic small-channel codec built with
// the REAL structural geometry (`decoder_rates = [8, 8, 4, 2]`, `frame_hop = 2048`, the
// ×4 ConvNeXt upsample) — only the channel widths are shrunk, and the widths do not
// enter the chunking arithmetic. They pin the two properties [`EvaGanDac::decode_chunked`]
// rests on: the synthesis stack is length-exact (`L` frames → `L * frame_hop` samples),
// and its left dependency is inside [`DECODE_LEFT_CTX_FRAMES`] with no right dependency —
// so chunk boundaries are invisible in the output.
//
// NOTE (placement): the repo convention puts frozen Ratchet tests at the repo root
// `tests/`. These are not frozen criterion tests, and hosting them at the root would
// require making `s2::codec` / `s2::nn` public — files outside this change's ownership —
// so they live here as crate unit tests. Run with:
//     cargo test -p syrinx-fish --features real s2::codec
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic LCG → weights in `(-scale, scale)`, so the fixture is reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        }
    }

    /// A synthetic codec with the real decode geometry at small channel widths.
    struct Fixture {
        codec: EvaGanDac,
        n_cb: usize,
    }

    const LATENT: usize = 8;
    const CB_DIM: usize = 4;
    const SEM_SIZE: usize = 16;
    const RES_SIZE: usize = 8;
    const N_CB: usize = 10;
    /// `decoder.model.{0..4}` channel schedule (model.0 out, then one per decoder rate).
    const DEC_CH: [usize; 5] = [16, 8, 8, 4, 4];

    /// A deterministic `[1, 1, t * hop]` test waveform for the encode tests.
    fn ramp_wav(f: &Fixture, t: usize) -> Tensor {
        let hop = f.codec.cfg.frame_hop;
        let n = t * hop;
        let mut rng = Lcg(0xC0FF_EE00_1234_5678);
        let v: Vec<f32> = (0..n).map(|_| rng.next() * 0.5).collect();
        Tensor::from_vec(v, (1, 1, n), f.codec.dev()).unwrap()
    }

    fn fixture() -> Fixture {
        let dev = Device::Cpu;
        let dt = DType::F32;
        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        let put = |map: &mut HashMap<String, Tensor>,
                       rng: &mut Lcg,
                       name: &str,
                       dims: &[usize],
                       scale: f32| {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = (0..n).map(|_| rng.next() * scale).collect();
            map.insert(name.to_string(), Tensor::from_vec(v, dims, &dev).unwrap());
        };

        // --- RVQ from_codes: 1 semantic + 9 residual factorized quantizers -------
        put(&mut map, &mut rng, "quantizer.semantic_quantizer.quantizers.0.codebook.weight", &[SEM_SIZE, CB_DIM], 1.0);
        put(&mut map, &mut rng, "quantizer.semantic_quantizer.quantizers.0.out_proj.weight", &[LATENT, CB_DIM, 1], 0.5);
        put(&mut map, &mut rng, "quantizer.semantic_quantizer.quantizers.0.out_proj.bias", &[LATENT], 0.1);
        for i in 0..(N_CB - 1) {
            let p = format!("quantizer.quantizer.quantizers.{i}");
            put(&mut map, &mut rng, &format!("{p}.codebook.weight"), &[RES_SIZE, CB_DIM], 1.0);
            put(&mut map, &mut rng, &format!("{p}.out_proj.weight"), &[LATENT, CB_DIM, 1], 0.3);
            put(&mut map, &mut rng, &format!("{p}.out_proj.bias"), &[LATENT], 0.1);
        }
        // `quantizer.post_module` is deliberately absent: `transformer` discovers zero
        // layers and is the identity, so the fixture isolates the causal conv stack —
        // which is the only part `decode_chunked` splits (the bottleneck Transformer
        // always runs over the whole utterance, in both paths).

        // --- quantizer.upsample: 2 × (ConvTranspose(k=2,s=2) + ConvNeXt) ---------
        for s in 0..2 {
            let p = format!("quantizer.upsample.{s}");
            put(&mut map, &mut rng, &format!("{p}.0.conv.weight"), &[LATENT, LATENT, 2], 0.4);
            put(&mut map, &mut rng, &format!("{p}.0.conv.bias"), &[LATENT], 0.05);
            put(&mut map, &mut rng, &format!("{p}.1.dwconv.conv.weight"), &[LATENT, 1, 7], 0.4);
            put(&mut map, &mut rng, &format!("{p}.1.dwconv.conv.bias"), &[LATENT], 0.05);
            put(&mut map, &mut rng, &format!("{p}.1.norm.weight"), &[LATENT], 1.0);
            put(&mut map, &mut rng, &format!("{p}.1.norm.bias"), &[LATENT], 0.1);
            put(&mut map, &mut rng, &format!("{p}.1.pwconv1.weight"), &[4 * LATENT, LATENT], 0.4);
            put(&mut map, &mut rng, &format!("{p}.1.pwconv1.bias"), &[4 * LATENT], 0.05);
            put(&mut map, &mut rng, &format!("{p}.1.pwconv2.weight"), &[LATENT, 4 * LATENT], 0.4);
            put(&mut map, &mut rng, &format!("{p}.1.pwconv2.bias"), &[LATENT], 0.05);
            put(&mut map, &mut rng, &format!("{p}.1.gamma"), &[LATENT], 0.5);
        }

        // --- decoder.model.*: DAC generator, real strides [8, 8, 4, 2] -----------
        let rates = [8usize, 8, 4, 2];
        put(&mut map, &mut rng, "decoder.model.0.conv.weight", &[DEC_CH[0], LATENT, 7], 0.3);
        put(&mut map, &mut rng, "decoder.model.0.conv.bias", &[DEC_CH[0]], 0.05);
        for (i, &s) in rates.iter().enumerate() {
            let p = format!("decoder.model.{}", i + 1);
            let (ci, co) = (DEC_CH[i], DEC_CH[i + 1]);
            put(&mut map, &mut rng, &format!("{p}.block.0.alpha"), &[1, ci, 1], 0.5);
            put(&mut map, &mut rng, &format!("{p}.block.1.conv.weight"), &[ci, co, 2 * s], 0.3);
            put(&mut map, &mut rng, &format!("{p}.block.1.conv.bias"), &[co], 0.05);
            for ru in 2..5 {
                let q = format!("{p}.block.{ru}");
                put(&mut map, &mut rng, &format!("{q}.block.0.alpha"), &[1, co, 1], 0.5);
                put(&mut map, &mut rng, &format!("{q}.block.1.conv.weight"), &[co, co, 7], 0.3);
                put(&mut map, &mut rng, &format!("{q}.block.1.conv.bias"), &[co], 0.05);
                put(&mut map, &mut rng, &format!("{q}.block.2.alpha"), &[1, co, 1], 0.5);
                put(&mut map, &mut rng, &format!("{q}.block.3.conv.weight"), &[co, co, 1], 0.3);
                put(&mut map, &mut rng, &format!("{q}.block.3.conv.bias"), &[co], 0.05);
            }
        }
        let last = DEC_CH[4];
        put(&mut map, &mut rng, "decoder.model.5.alpha", &[1, last, 1], 0.5);
        put(&mut map, &mut rng, "decoder.model.6.conv.weight", &[1, last, 7], 0.3);
        put(&mut map, &mut rng, "decoder.model.6.conv.bias", &[1], 0.02);

        // --- encoder side (for the chunked-encode equivalence tests) -----------
        // Same GEOMETRY as the real codec — kernel sizes, strides, dilations and the
        // [2,4,8,8] rate schedule all match, which is what the receptive field depends
        // on — at fixture channel widths. Channels do not affect the left-context
        // derivation, so the constant proven here is the one that ships.
        // The deepest width must be a multiple of TF_HEAD_DIM (64): `transformer`
        // derives `n_head = dim / TF_HEAD_DIM`, so a narrower block would give 0 heads.
        const ENC_CH: [usize; 5] = [ENCODER_DIM, 8, 8, 8, 64];
        put(&mut map, &mut rng, "encoder.block.0.conv.weight", &[ENC_CH[0], 1, 7], 0.3);
        put(&mut map, &mut rng, "encoder.block.0.conv.bias", &[ENC_CH[0]], 0.05);
        for (i, &stride) in [2usize, 4, 8, 8].iter().enumerate() {
            let bp = format!("encoder.block.{}", i + 1);
            let cin = ENC_CH[i];
            let cout = ENC_CH[i + 1];
            // 3 dilated residual units at the block's INPUT width
            for (u, _d) in [1usize, 3, 9].iter().enumerate() {
                put(&mut map, &mut rng, &format!("{bp}.block.{u}.block.0.alpha"), &[1, cin, 1], 0.5);
                put(&mut map, &mut rng, &format!("{bp}.block.{u}.block.1.conv.weight"), &[cin, cin, 7], 0.2);
                put(&mut map, &mut rng, &format!("{bp}.block.{u}.block.1.conv.bias"), &[cin], 0.05);
                put(&mut map, &mut rng, &format!("{bp}.block.{u}.block.2.alpha"), &[1, cin, 1], 0.5);
                put(&mut map, &mut rng, &format!("{bp}.block.{u}.block.3.conv.weight"), &[cin, cin, 1], 0.2);
                put(&mut map, &mut rng, &format!("{bp}.block.{u}.block.3.conv.bias"), &[cin], 0.05);
            }
            put(&mut map, &mut rng, &format!("{bp}.block.3.alpha"), &[1, cin, 1], 0.5);
            put(&mut map, &mut rng, &format!("{bp}.block.4.conv.weight"), &[cout, cin, 2 * stride], 0.2);
            put(&mut map, &mut rng, &format!("{bp}.block.4.conv.bias"), &[cout], 0.05);
        }
        // A REAL Transformer in the deepest encoder block. The fixture previously had
        // none, so `run_encoder` skipped it and the chunked-encode test passed while the
        // real codec — which does have one — produced different codes. Full-causal
        // attention has an unbounded left dependency, so its presence is exactly what
        // forces the encoder split; the test is only meaningful with it wired.
        {
            // dim == TF_HEAD_DIM => exactly one attention head at fixture width.
            const TF_D: usize = ENC_CH[4];
            let p5 = "encoder.block.4.block.5";
            put(&mut map, &mut rng, &format!("{p5}.norm.weight"), &[TF_D], 1.0);
            let l0 = format!("{p5}.layers.0");
            put(&mut map, &mut rng, &format!("{l0}.attention_norm.weight"), &[TF_D], 1.0);
            put(&mut map, &mut rng, &format!("{l0}.attention.wqkv.weight"), &[3 * TF_D, TF_D], 0.2);
            put(&mut map, &mut rng, &format!("{l0}.attention.wo.weight"), &[TF_D, TF_D], 0.2);
            put(&mut map, &mut rng, &format!("{l0}.attention_layer_scale.gamma"), &[TF_D], 0.3);
            put(&mut map, &mut rng, &format!("{l0}.ffn_norm.weight"), &[TF_D], 1.0);
            put(&mut map, &mut rng, &format!("{l0}.feed_forward.w1.weight"), &[2 * TF_D, TF_D], 0.2);
            put(&mut map, &mut rng, &format!("{l0}.feed_forward.w3.weight"), &[2 * TF_D, TF_D], 0.2);
            put(&mut map, &mut rng, &format!("{l0}.feed_forward.w2.weight"), &[TF_D, 2 * TF_D], 0.2);
            put(&mut map, &mut rng, &format!("{l0}.ffn_layer_scale.gamma"), &[TF_D], 0.3);
        }
        put(&mut map, &mut rng, "encoder.block.5.alpha", &[1, ENC_CH[4], 1], 0.5);
        put(&mut map, &mut rng, "encoder.block.6.conv.weight", &[LATENT, ENC_CH[4], 3], 0.2);
        put(&mut map, &mut rng, "encoder.block.6.conv.bias", &[LATENT], 0.05);
        // `quantizer.downsample`: 2 x (conv k=f stride=f + ConvNeXt)
        for (i, &f) in DOWNSAMPLE_FACTOR.iter().enumerate() {
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.0.conv.weight"), &[LATENT, LATENT, f], 0.3);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.0.conv.bias"), &[LATENT], 0.05);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.dwconv.conv.weight"), &[LATENT, 1, 7], 0.2);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.dwconv.conv.bias"), &[LATENT], 0.05);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.norm.weight"), &[LATENT], 1.0);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.norm.bias"), &[LATENT], 0.05);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.pwconv1.weight"), &[LATENT * 2, LATENT], 0.2);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.pwconv1.bias"), &[LATENT * 2], 0.05);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.pwconv2.weight"), &[LATENT, LATENT * 2], 0.2);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.pwconv2.bias"), &[LATENT], 0.05);
            put(&mut map, &mut rng, &format!("quantizer.downsample.{i}.1.gamma"), &[LATENT], 0.5);
        }

        let cfg = CodecConfig {
            num_codebooks: N_CB,
            semantic_size: SEM_SIZE,
            residual_size: RES_SIZE,
            codebook_dim: CB_DIM,
            sample_rate: 44_100,
            frame_hop: 2048,
            encoder_rates: vec![2, 4, 8, 8],
            decoder_rates: rates.to_vec(),
        };
        Fixture {
            codec: EvaGanDac::new(Weights { map, dev, dt, qmap: HashMap::new() }, cfg),
            n_cb: N_CB,
        }
    }

    fn codes(f: &Fixture, t: usize) -> Tensor {
        let mut rng = Lcg(0xC0DE_0F15_4321_0007);
        let mut v = vec![0u32; f.n_cb * t];
        for (i, c) in v.iter_mut().enumerate() {
            let size = if i < t { SEM_SIZE } else { RES_SIZE } as f32;
            *c = (((rng.next() + 1.0) * 0.5 * size) as u32).min(size as u32 - 1);
        }
        Tensor::from_vec(v, (f.n_cb, t), &Device::Cpu).unwrap()
    }

    /// The synthesis stack is length-exact: `T` frames → `T * frame_hop` samples.
    #[test]
    fn oneshot_decode_is_length_exact() {
        let f = fixture();
        for t in [1usize, 5, 17, 40] {
            let wav = f.codec.decode_oneshot(&codes(&f, t)).unwrap();
            assert_eq!(wav.dims(), &[t * 2048], "T={t}");
        }
    }

    /// The fixture must not be degenerate — a saturated (all-`±1`) or constant output
    /// would make the equivalence assertions below vacuous.
    #[test]
    fn fixture_output_is_non_degenerate() {
        let f = fixture();
        let v: Vec<f32> = f.codec.decode_oneshot(&codes(&f, 8)).unwrap().to_vec1().unwrap();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        assert!(var > 1e-8, "output is constant (var={var})");
        let saturated = v.iter().filter(|x| x.abs() > 0.999).count();
        assert!(
            saturated * 2 < v.len(),
            "output is tanh-saturated ({saturated}/{} samples)",
            v.len()
        );
    }

    /// Chunked synthesis reproduces the one-shot waveform exactly, for every chunk
    /// length — including chunks far smaller than `DECODE_LEFT_CTX_FRAMES`, chunks that
    /// do not divide `T`, and chunks at/over `T` (which take the one-shot branch).
    #[test]
    fn chunked_decode_matches_oneshot() {
        let f = fixture();
        let t = 40usize;
        let c = codes(&f, t);
        let want: Vec<f32> = f.codec.decode_oneshot(&c).unwrap().to_vec1().unwrap();
        for chunk in [2usize, 3, 8, 13, 16, 32, 39, 40, 64] {
            let got: Vec<f32> = f.codec.decode_chunked(&c, chunk).unwrap().to_vec1().unwrap();
            assert_eq!(got.len(), want.len(), "chunk={chunk}: length");
            let (i, d) = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert_eq!(d, 0.0, "chunk={chunk}: sample {i} differs by {d}");
        }
    }

    /// Chunked ENCODE reproduces the one-shot latent exactly, for every chunk length —
    /// including chunks smaller than the left context, chunks that do not divide `T`,
    /// and chunks at/over `T` (which take the one-shot branch).
    #[test]
    fn chunked_encode_matches_oneshot() {
        let f = fixture();
        let t = 48usize;
        let wav = ramp_wav(&f, t);
        let want = f.codec.encode_latent_chunked(&wav, 0).unwrap();
        let want_v: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        for chunk in [2usize, 3, 8, 13, 16, 32, 47, 48, 96] {
            let got = f.codec.encode_latent_chunked(&wav, chunk).unwrap();
            assert_eq!(got.dims(), want.dims(), "chunk={chunk}: latent shape");
            let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            let (i, d) = got_v
                .iter()
                .zip(&want_v)
                .map(|(g, w)| (g - w).abs())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert_eq!(d, 0.0, "chunk={chunk}: latent {i} differs by {d}");
        }
    }

    /// The NEGATIVE CONTROL for [`ENCODE_LEFT_CTX_FRAMES`].
    ///
    /// Re-runs the chunked encode with a deliberately truncated left context and asserts
    /// it *fails* below the derived 12.65-frame receptive field and *passes* at and above
    /// it. Without this, an under-sized constant would corrupt the reference silently —
    /// the decode equivalent showed a context one short still matching to 6e-6, which no
    /// tolerance-based assert would have caught.
    #[test]
    fn encode_left_ctx_matches_receptive_field() {
        let f = fixture();
        let t = 48usize;
        let wav = ramp_wav(&f, t);
        let want: Vec<f32> = f
            .codec
            .encode_latent_chunked(&wav, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let worst_with_ctx = |ctx: usize| -> f32 {
            let got: Vec<f32> = f
                .codec
                .encode_latent_ctx(&wav, 8, ctx)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            got.iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max)
        };
        // encoder_convs receptive field = 6 442 samples / 2048 = 3.146 frames, so 4 is
        // the first sufficient context and 3 must still be visibly wrong.
        for ctx in [0usize, 1, 2, 3] {
            assert!(
                worst_with_ctx(ctx) > 0.0,
                "ctx={ctx} is below the 3.146-frame receptive field but matched exactly \
                 — the derivation or the chunking is wrong"
            );
        }
        for ctx in [4usize, 8, 16] {
            assert_eq!(worst_with_ctx(ctx), 0.0, "ctx={ctx} should be sufficient");
        }
        assert!(
            ENCODE_LEFT_CTX_FRAMES >= 4,
            "the shipped constant must cover the receptive field"
        );
    }

    /// `decode` (the default path) agrees with the one-shot reference, and the env
    /// override selects the one-shot path at `0`.
    #[test]
    fn default_decode_matches_oneshot() {
        let f = fixture();
        let c = codes(&f, 40);
        let want: Vec<f32> = f.codec.decode_oneshot(&c).unwrap().to_vec1().unwrap();
        let got: Vec<f32> = f.codec.decode(&c).unwrap().to_vec1().unwrap();
        assert_eq!(got, want);
        let zero: Vec<f32> = f.codec.decode_chunked(&c, 0).unwrap().to_vec1().unwrap();
        assert_eq!(zero, want);
    }
}
