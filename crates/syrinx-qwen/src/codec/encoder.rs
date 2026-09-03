//! The **analysis** half of the Qwen3-TTS 12 Hz tokenizer: waveform -> RVQ codes.
//!
//! This is the path the `*-Base` checkpoints use to clone a voice from a ~3 s reference
//! clip. It is the exact reverse of [`crate::codec::rvq`]'s synthesis side and it is
//! **not** a Qwen-specific design: `Qwen3TTSTokenizerV2Encoder` subclasses HuggingFace's
//! `MimiModel` and blanks out `upsample` / `decoder_transformer` / `decoder`, so the
//! encode stack is stock Mimi (`transformers/models/mimi/modeling_mimi.py`). Everything
//! below was read out of that file and out of the checkpoint header, never inferred.
//!
//! ```text
//!   wav [1, 1, L]  @24 kHz
//!     -> encoder            SEANet conv cascade, strides 4·5·6·8 = 960x   -> [1, 512, ceil(L/960)]
//!     -> encoder_transformer  8-layer causal Transformer bottleneck       -> same shape
//!     -> downsample         causal conv k=4 s=2, `replicate` padding      -> [1, 512, ceil(L/1920)]
//!     -> quantizer          split RVQ: 1 semantic + 15 acoustic codes     -> [16][T]
//! ```
//!
//! # Reference semantics that are easy to get wrong
//!
//! * **The split RVQ is parallel, not serial.** `MimiSplitResidualVectorQuantizer::encode`
//!   feeds the *same* `embeddings` to both the semantic and the acoustic stack — the
//!   acoustic stack does **not** start from the semantic residual. Only *within* a stack
//!   is the quantisation residual. (Decode mirrors it: `rvq_first(...) + rvq_rest(...)`.)
//! * **No normalisation before the search.** `MimiEuclideanCodebook::quantize` is
//!   `torch.cdist(x, embed, p=2).argmin(-1)` on the raw projected vectors. The sibling
//!   Fish RVQ L2-normalises both sides first; this one must not. Each stack does apply
//!   its own `input_proj` (a bias-free `Conv1d(512, 256, 1)`) before the search.
//! * **The codebook is an EMA quotient.** `embed = embed_sum / cluster_usage.clamp(1e-5)`.
//!   The reconstruction is not duplicated here — see [`load_codebooks`].
//! * **The tensor names differ from the decode side.** Encode ships
//!   `...quantizer.{semantic,acoustic}_residual_vector_quantizer.layers.{l}.codebook.embed_sum`;
//!   decode ships `...vq.layers.{l}._codebook.embedding_sum`. Same formula, different keys,
//!   because encode is upstream `MimiModel` and decode is Qwen's own rewrite.
//! * **The checkpoint carries 32 acoustic-side quantizers but only 16 are used.**
//!   `encoder_valid_num_quantizers = 16` and the reference slices `audio_codes[:, :16]`
//!   *after* running all 32. RVQ layers are strictly sequential within a stack, so
//!   computing only the first 15 acoustic layers yields bit-identical codes for those 15.
//! * **`sliding_window: 250` is inert.** `MimiTransformerModel` builds its mask with
//!   `create_causal_mask` (never `create_sliding_window_causal_mask`), and neither the
//!   eager nor the sdpa attention applies `self.sliding_window`; only the
//!   `flash_attention_2` class would. The published default is sdpa, so the bottleneck is
//!   **full** causal attention. This is what forces the stack split below.
//!
//! # Memory: why the conv cascade is chunked
//!
//! candle's `conv1d` materialises an `L * C_in * k` im2col buffer. The cascade runs at
//! the full 24 kHz rate, and its widest early layer (`layers.3`, k=8, C_in=64) costs
//! `L * 2048` bytes in f32 — 1.7 GB in one allocation for a 35 s reference. That precise
//! failure OOMed the sibling Fish s2 port, so the cascade runs in chunks with a derived
//! left context and the peak becomes a function of the chunk, not of the clip.
//!
//! The `encoder_transformer` **cannot** be inside that chunking: causal attention at
//! position `t` reads every position `0..=t`, so no finite left context reproduces it.
//! It sits between the cascade and `downsample`, so the stack is split exactly there —
//! chunk the cascade, then run transformer + downsample + quantizer once over the
//! concatenated result. They run at 1/960 the sample rate, where length costs nothing.
//!
//! # Verification status
//!
//! Unlike most of this workspace, the encode stack **is** confirmable off-box: it is
//! f32 and CPU-cheap on both sides. [`tests::matches_the_python_reference`] runs the real
//! 682 MB `Qwen3-TTS-Tokenizer-12Hz` checkpoint against a dump from HuggingFace's own
//! `MimiModel` pieces and has been executed on the dev box (CPU, no GPU):
//!
//! * 5-frame clip (hop-aligned) and 32-frame clip (60 297 samples — deliberately ragged,
//!   giving an **odd** 63-step cascade output so `downsample`'s right-side replicate pad
//!   is exercised);
//! * every code identical (16 x 5 and 16 x 32), through the one-shot cascade **and**
//!   through a 2-step chunked cascade;
//! * latent max abs difference 1.2e-4 and 2.5e-4 against magnitudes of order 5-10 — pure
//!   f32 reassociation.
//!
//! `load_state_dict` on the reference reported zero missing and zero unexpected keys for
//! the 225 `encoder.*` tensors, so the name/shape map below is the checkpoint's own.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, D};

use crate::codec::rvq::Rvq;
use crate::nn::{causal_mask_at, precompute_rope, apply_rope, Weights};

/// Left context, in 960-sample cascade steps, that a chunk of the conv cascade needs for
/// its kept output to be **identical** to the one-shot result.
///
/// Every conv in the cascade is causal: with left padding `P = k_eff - s` (where
/// `k_eff = (k-1)*d + 1`), output index `i` reads original inputs
/// `[i*s - k_eff + s, (i+1)*s - 1]` — nothing to its right. Propagating that interval
/// backwards through the exact op list
///
/// ```text
///   conv k7 s1
///   4 x [ resnet(conv k3 s1 ; conv k1 s1) ; conv k(2r) s(r) ]   for r in 4, 5, 6, 8
///   conv k3 s1
/// ```
///
/// gives a **3320-sample** left dependency, constant in the output index (output 0 reaches
/// back to -3320, output 10 to +6280 = 10*960 - 3320). That is 3.458 steps of 960, so
/// **4 steps is the first sufficient context**. 16 is that bound with generous margin;
/// over-supplying context only replaces zero-padding with true history, and peak memory
/// scales with `chunk + ctx` so 128+16 costs 12.5 % over 128+4.
/// [`tests::left_ctx_matches_the_receptive_field`] sweeps the constant and asserts the
/// flip happens at 4 — without that negative control an undersized value would corrupt
/// the reference silently.
const ENCODE_LEFT_CTX_STEPS: usize = 16;

/// The derived receptive field of the conv cascade, in input samples. Pinned as a named
/// constant so the test that sweeps [`ENCODE_LEFT_CTX_STEPS`] can state what it defends.
pub const CASCADE_RECEPTIVE_FIELD_SAMPLES: usize = 3320;

/// Default cascade chunk length in 960-sample steps (128 * 960 = 5.12 s of audio).
///
/// Bounds the im2col spike at `(chunk + ctx) * 960 * 2048` bytes ~= 283 MB regardless of
/// reference length, against 1.7 GB for a one-shot 35 s clip.
/// `SYRINX_QWEN_CODEC_ENCODE_CHUNK` overrides; `0` selects the one-shot path.
const DEFAULT_ENCODE_CHUNK_STEPS: usize = 128;

/// Resolve the cascade chunk length from `SYRINX_QWEN_CODEC_ENCODE_CHUNK`.
pub fn encode_chunk_steps_env() -> usize {
    std::env::var("SYRINX_QWEN_CODEC_ENCODE_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_ENCODE_CHUNK_STEPS)
}

/// Padding mode of one causal conv, mirroring `MimiConv1d.pad_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PadMode {
    /// `config.pad_mode = "constant"` — the whole encoder cascade.
    Zero,
    /// `pad_mode="replicate"`, passed explicitly for `downsample` only.
    Replicate,
}

/// Geometry of the tokenizer's encode stack, parsed from the published `config.json`.
///
/// Read from the file, never defaulted from memory: the sibling Fish s1 port hand-wrote
/// this kind of geometry and got seven of nine fields wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct MimiEncoderConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    /// First conv's kernel (`kernel_size`).
    pub kernel_size: usize,
    /// Final cascade conv's kernel (`last_kernel_size`).
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub num_residual_layers: usize,
    /// Residual-block bottleneck divisor (`hidden = dim / compress`).
    pub compress: usize,
    /// Decoder-side upsampling ratios; the encoder strides are these **reversed**.
    pub upsampling_ratios: Vec<usize>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub layer_scale_initial_scale: f64,
    pub sampling_rate: usize,
    /// `_frame_rate` — 12.5 Hz. The cascade alone lands at `encodec_frame_rate` (25 Hz);
    /// `downsample` halves it.
    pub frame_rate: f64,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    /// `vector_quantization_hidden_dimension`; a projection exists iff it differs from
    /// `hidden_size`.
    pub vq_hidden_dim: usize,
    pub num_semantic_quantizers: usize,
    /// Top-level `encoder_valid_num_quantizers` — how many codes the talker consumes.
    pub valid_num_quantizers: usize,
}

fn ju(v: &serde_json::Value, k: &str) -> Option<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize)
}
fn jf(v: &serde_json::Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

impl MimiEncoderConfig {
    /// Parse the published `Qwen3-TTS-Tokenizer-12Hz/config.json`.
    pub fn from_json(json: &str) -> std::result::Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("parse tokenizer config.json: {e}"))?;
        let e = v
            .get("encoder_config")
            .ok_or("tokenizer config.json: no `encoder_config`")?;
        let need = |k: &str| ju(e, k).ok_or_else(|| format!("encoder_config: missing `{k}`"));
        let ratios: Vec<usize> = e
            .get("upsampling_ratios")
            .and_then(|x| x.as_array())
            .ok_or("encoder_config: missing `upsampling_ratios`")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        if ratios.is_empty() || ratios.contains(&0) {
            return Err("encoder_config: bad `upsampling_ratios`".into());
        }
        Ok(Self {
            audio_channels: need("audio_channels")?,
            num_filters: need("num_filters")?,
            kernel_size: need("kernel_size")?,
            last_kernel_size: need("last_kernel_size")?,
            residual_kernel_size: need("residual_kernel_size")?,
            dilation_growth_rate: need("dilation_growth_rate")?,
            num_residual_layers: need("num_residual_layers")?,
            compress: need("compress")?,
            upsampling_ratios: ratios,
            hidden_size: need("hidden_size")?,
            num_hidden_layers: need("num_hidden_layers")?,
            num_attention_heads: need("num_attention_heads")?,
            num_key_value_heads: need("num_key_value_heads")?,
            head_dim: need("head_dim")?,
            intermediate_size: need("intermediate_size")?,
            norm_eps: jf(e, "norm_eps").ok_or("encoder_config: missing `norm_eps`")?,
            rope_theta: jf(e, "rope_theta").ok_or("encoder_config: missing `rope_theta`")?,
            layer_scale_initial_scale: jf(e, "layer_scale_initial_scale").unwrap_or(0.01),
            sampling_rate: need("sampling_rate")?,
            frame_rate: jf(e, "_frame_rate").ok_or("encoder_config: missing `_frame_rate`")?,
            codebook_size: need("codebook_size")?,
            codebook_dim: need("codebook_dim")?,
            vq_hidden_dim: need("vector_quantization_hidden_dimension")?,
            num_semantic_quantizers: need("num_semantic_quantizers")?,
            valid_num_quantizers: ju(&v, "encoder_valid_num_quantizers")
                .ok_or("tokenizer config.json: missing `encoder_valid_num_quantizers`")?,
        })
    }

    /// Cascade stride product (`960`): the rate the `encoder_transformer` runs at.
    pub fn cascade_hop(&self) -> usize {
        self.upsampling_ratios.iter().product()
    }

    /// `MimiConfig.encodec_frame_rate` — the cascade's own frame rate, 25 Hz.
    pub fn encodec_frame_rate(&self) -> usize {
        self.sampling_rate.div_ceil(self.cascade_hop())
    }

    /// `downsample` stride: always 2 in `MimiModel.__init__`.
    pub fn downsample_stride(&self) -> usize {
        2
    }

    /// `downsample` kernel: `2 * int(encodec_frame_rate / frame_rate)` = 4.
    pub fn downsample_kernel(&self) -> usize {
        2 * ((self.encodec_frame_rate() as f64 / self.frame_rate) as usize)
    }

    /// Samples per emitted code frame: `cascade_hop * downsample_stride` = 1920.
    pub fn frame_hop(&self) -> usize {
        self.cascade_hop() * self.downsample_stride()
    }

    /// Acoustic quantizers actually run: `valid_num_quantizers - num_semantic_quantizers`.
    pub fn valid_acoustic_quantizers(&self) -> usize {
        self.valid_num_quantizers
            .saturating_sub(self.num_semantic_quantizers)
    }
}

/// Reconstruct one encode-side RVQ stack's codebooks **through [`Rvq::load`]**.
///
/// The EMA quotient `embed_sum / cluster_usage.clamp(1e-5)` is the single most
/// consequential formula in this codec — using the raw accumulator would rescale every
/// centroid by a different 2x-50x factor and still produce audio-shaped output. It is
/// therefore not re-derived here. The only obstacle to calling [`Rvq::load`] directly is
/// naming: it expects the decode-side keys
/// `{p}.vq.layers.{l}._codebook.{embedding_sum,cluster_usage}` while the encode side
/// ships `{p}.layers.{l}.codebook.{embed_sum,cluster_usage}`. So this builds a tiny
/// alias bag (candle `Tensor` clones are refcount bumps, not copies) and hands that over.
fn load_codebooks(w: &Weights, prefix: &str, n_layers: usize) -> Result<Rvq> {
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for l in 0..n_layers {
        let src = format!("{prefix}.layers.{l}.codebook");
        let dst = format!("alias.vq.layers.{l}._codebook");
        map.insert(
            format!("{dst}.embedding_sum"),
            w.g(&format!("{src}.embed_sum"))?,
        );
        map.insert(
            format!("{dst}.cluster_usage"),
            w.g(&format!("{src}.cluster_usage"))?,
        );
    }
    let alias = Weights { map, dev: w.dev.clone(), dt: w.dt };
    Rvq::load(&alias, "alias", n_layers)
}

/// One residual-vector-quantizer stack (semantic or acoustic) on the analysis side.
struct RvqStack {
    /// `input_proj.weight`, `[vq_dim, hidden, 1]` — a bias-free kernel-1 conv.
    input_proj: Tensor,
    /// Reconstructed centroids, via [`load_codebooks`].
    rvq: Rvq,
    /// `||centroid||^2` per layer, `[1, codebook_size]`, precomputed for the search.
    sq_norms: Vec<Tensor>,
}

impl RvqStack {
    fn load(w: &Weights, prefix: &str, n_layers: usize) -> Result<Self> {
        let input_proj = w
            .g(&format!("{prefix}.input_proj.weight"))?
            .to_dtype(DType::F32)?;
        let rvq = load_codebooks(w, prefix, n_layers)?;
        let mut sq_norms = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let cb = rvq
                .codebook(l)
                .ok_or_else(|| candle_core::Error::Msg(format!("{prefix}: no codebook {l}")))?;
            sq_norms.push(cb.sqr()?.sum_keepdim(D::Minus1)?.t()?.contiguous()?);
        }
        Ok(Self { input_proj, rvq, sq_norms })
    }

    /// `MimiResidualVectorQuantizer.encode`: project, then per layer take the nearest
    /// centroid, emit its index and subtract it from the running residual.
    ///
    /// `emb` is `[1, hidden, T]` in f32. Returns `[n_layers][T]`.
    fn encode(&self, emb: &Tensor) -> Result<Vec<Vec<u32>>> {
        // `input_proj` is Conv1d(hidden, vq_dim, 1, bias=False): squeeze the kernel axis
        // and it is a channel-wise matmul.
        let pw = self.input_proj.squeeze(D::Minus1)?; // [vq_dim, hidden]
        let x = emb.squeeze(0)?.t()?.contiguous()?; // [T, hidden]
        let mut residual = x.matmul(&pw.t()?)?; // [T, vq_dim]

        let mut rows = Vec::with_capacity(self.rvq.n_layers());
        for l in 0..self.rvq.n_layers() {
            let cb = self.rvq.codebook(l).unwrap(); // [size, dim]
            // argmin over ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2; ||x||^2 is constant
            // across candidates, so it drops out. torch.cdist takes the same mm-based
            // route for inputs this size (compute_mode `use_mm_for_euclid_dist_if_necessary`).
            let dots = residual.matmul(&cb.t()?)?; // [T, size]
            let scores = self.sq_norms[l].broadcast_sub(&(dots * 2.0)?)?;
            let codes = argmin_rows(&scores)?;
            rows.push(codes.clone());

            let idx = Tensor::from_vec(codes, (residual.dim(0)?,), residual.device())?;
            let quantized = cb.index_select(&idx, 0)?; // [T, dim]
            residual = (residual - quantized)?;
        }
        Ok(rows)
    }
}

/// `MimiSplitResidualVectorQuantizer.encode`.
///
/// **Both stacks read the same `emb`.** The acoustic stack does not start from the
/// semantic residual — it is a second, independent RVQ over the original embedding,
/// exactly as decode adds the two reconstructions rather than nesting them.
fn quantize_split(sem: &RvqStack, aco: &RvqStack, emb: &Tensor) -> Result<Vec<Vec<u32>>> {
    // PARITY: `MimiEuclideanCodebook.quantize` casts to float32 before `cdist`, and the
    // published encoder_config pins `"dtype": "float32"`. The search runs in f32 here
    // regardless of the compute dtype; a bf16 search could flip borderline argmins into
    // different codes.
    let emb = emb.to_dtype(DType::F32)?;
    let mut rows = sem.encode(&emb)?;
    rows.extend(aco.encode(&emb)?);
    Ok(rows)
}

/// Row-wise argmin, first index on ties (matching `torch.argmin` on CPU).
fn argmin_rows(scores: &Tensor) -> Result<Vec<u32>> {
    let (t, size) = scores.dims2()?;
    let host: Vec<f32> = scores.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut out = vec![0u32; t];
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &host[i * size..(i + 1) * size];
        let mut best = 0usize;
        let mut best_v = f32::INFINITY;
        for (j, &v) in row.iter().enumerate() {
            if v < best_v {
                best_v = v;
                best = j;
            }
        }
        *slot = best as u32;
    }
    Ok(out)
}

/// The loaded Mimi encode stack.
pub struct MimiEncoder {
    w: Weights,
    cfg: MimiEncoderConfig,
    semantic: RvqStack,
    acoustic: RvqStack,
}

const ENC: &str = "encoder.encoder";
const TF: &str = "encoder.encoder_transformer";
const DOWN: &str = "encoder.downsample";
const QUANT: &str = "encoder.quantizer";

impl MimiEncoder {
    /// Wrap a loaded weight bag. Only the `valid_num_quantizers` codebooks the talker
    /// consumes are reconstructed; the checkpoint's remaining acoustic layers are never
    /// touched (RVQ layers are sequential, so truncation is exact).
    pub fn new(w: Weights, cfg: MimiEncoderConfig) -> Result<Self> {
        let semantic = RvqStack::load(
            &w,
            &format!("{QUANT}.semantic_residual_vector_quantizer"),
            cfg.num_semantic_quantizers,
        )?;
        let acoustic = RvqStack::load(
            &w,
            &format!("{QUANT}.acoustic_residual_vector_quantizer"),
            cfg.valid_acoustic_quantizers(),
        )?;
        Ok(Self { w, cfg, semantic, acoustic })
    }

    pub fn config(&self) -> &MimiEncoderConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.w.dev.clone()
    }

    // --- causal convolution ---------------------------------------------------

    /// `MimiConv1d.forward`: pad left by `k_eff - stride` plus the right-side
    /// `extra_padding`, then convolve.
    ///
    /// `extra_padding` reduces algebraically: the reference computes
    /// `ceil((L - k_eff + padding_total)/s) * s + k_eff - padding_total - L`, and with
    /// `padding_total == k_eff - s` that is exactly `ceil(L/s)*s - L`. Doing the integer
    /// form avoids the reference's float `ceil`.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv1d(
        &self,
        x: &Tensor,
        wname: &str,
        bias: Option<&str>,
        k: usize,
        dilation: usize,
        stride: usize,
        pad: PadMode,
    ) -> Result<Tensor> {
        let k_eff = (k - 1) * dilation + 1;
        let padding_total = k_eff - stride;
        let len = x.dim(D::Minus1)?;
        let extra = len.div_ceil(stride) * stride - len;
        let xp = match pad {
            PadMode::Zero => x.pad_with_zeros(D::Minus1, padding_total, extra)?,
            PadMode::Replicate => x.pad_with_same(D::Minus1, padding_total, extra)?,
        };
        let kernel = self.w.g(wname)?;
        let y = xp.conv1d(&kernel, 0, stride, dilation, 1)?;
        match bias {
            None => Ok(y),
            Some(b) => {
                let bt = self.w.g(b)?;
                y.broadcast_add(&bt.reshape((1, bt.dim(0)?, 1))?)
            }
        }
    }

    /// `MimiResnetBlock`: `ELU -> conv(k=residual_kernel_size, d) -> ELU -> conv(k=1)`,
    /// added to the (identity) shortcut. `use_conv_shortcut` is false in this checkpoint,
    /// so the shortcut carries no weights.
    fn resnet_block(&self, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
        let h = x.elu(1.0)?;
        let h = self.causal_conv1d(
            &h,
            &format!("{prefix}.block.1.conv.weight"),
            Some(&format!("{prefix}.block.1.conv.bias")),
            self.cfg.residual_kernel_size,
            dilation,
            1,
            PadMode::Zero,
        )?;
        let h = h.elu(1.0)?;
        let h = self.causal_conv1d(
            &h,
            &format!("{prefix}.block.3.conv.weight"),
            Some(&format!("{prefix}.block.3.conv.bias")),
            1,
            1,
            1,
            PadMode::Zero,
        )?;
        x + h
    }

    /// `MimiEncoder.forward`: the SEANet conv cascade. `[1, channels, L]` -> `[1, hidden,
    /// ceil(L / cascade_hop)]`.
    ///
    /// Module indices follow the reference's `nn.ModuleList` construction exactly: index
    /// 0 is the input conv, then per reversed ratio `num_residual_layers` resnet blocks,
    /// an ELU and a stride-`r` conv, then a final ELU and the `last_kernel_size` conv.
    fn cascade(&self, wav: &Tensor) -> Result<Tensor> {
        let mut x = self.causal_conv1d(
            wav,
            &format!("{ENC}.layers.0.conv.weight"),
            Some(&format!("{ENC}.layers.0.conv.bias")),
            self.cfg.kernel_size,
            1,
            1,
            PadMode::Zero,
        )?;
        let mut idx = 1usize;
        for &ratio in self.cfg.upsampling_ratios.iter().rev() {
            for j in 0..self.cfg.num_residual_layers {
                let dilation = self.cfg.dilation_growth_rate.pow(j as u32);
                x = self.resnet_block(&x, &format!("{ENC}.layers.{idx}"), dilation)?;
                idx += 1;
            }
            x = x.elu(1.0)?;
            idx += 1; // the ELU occupies a module slot
            x = self.causal_conv1d(
                &x,
                &format!("{ENC}.layers.{idx}.conv.weight"),
                Some(&format!("{ENC}.layers.{idx}.conv.bias")),
                ratio * 2,
                1,
                ratio,
                PadMode::Zero,
            )?;
            idx += 1;
        }
        x = x.elu(1.0)?;
        idx += 1;
        self.causal_conv1d(
            &x,
            &format!("{ENC}.layers.{idx}.conv.weight"),
            Some(&format!("{ENC}.layers.{idx}.conv.bias")),
            self.cfg.last_kernel_size,
            1,
            1,
            PadMode::Zero,
        )
    }

    // --- the transformer bottleneck ------------------------------------------

    /// `nn.LayerNorm` over the last axis (mean **and** variance, plus a bias) — not the
    /// RMSNorm the Qwen decoder stacks use. The checkpoint settles it: these layers ship
    /// `input_layernorm.bias`.
    fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let mean = xf.mean_keepdim(D::Minus1)?;
        let xc = xf.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = xc
            .broadcast_div(&(var + self.cfg.norm_eps)?.sqrt()?)?
            .to_dtype(dt)?;
        xn.broadcast_mul(&self.w.g(&format!("{prefix}.weight"))?)?
            .broadcast_add(&self.w.g(&format!("{prefix}.bias"))?)
    }

    /// `MimiAttention`: plain multi-head causal attention. No `q_norm`/`k_norm` (Mimi has
    /// none — which is why [`crate::nn::attention`] cannot be reused), no projection
    /// biases (`attention_bias: false`), RoPE on q and k, scale `1/sqrt(head_dim)`.
    fn attention(&self, x: &Tensor, prefix: &str, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let (nh, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let q = self
            .w
            .linear(x, &format!("{prefix}.q_proj.weight"), None)?
            .reshape((b, t, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .w
            .linear(x, &format!("{prefix}.k_proj.weight"), None)?
            .reshape((b, t, nkv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .w
            .linear(x, &format!("{prefix}.v_proj.weight"), None)?
            .reshape((b, t, nkv, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let k = crate::nn::repeat_kv(&k, nh / nkv)?;
        let v = crate::nn::repeat_kv(&v, nh / nkv)?;

        let scale = 1f64 / (hd as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        let out = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, nh * hd))?;
        self.w.linear(&out, &format!("{prefix}.o_proj.weight"), None)
    }

    /// `MimiTransformerModel`: `num_hidden_layers` of pre-norm attention + MLP, each
    /// residual branch scaled elementwise by a learnt `LayerScale`. No final norm and no
    /// input/output projection — the checkpoint carries exactly 12 tensors per layer and
    /// nothing else under this prefix.
    ///
    /// **One-shot by construction.** Attention here is full causal (see the module
    /// header on `sliding_window`), so this must see the whole sequence.
    fn transformer(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, t) = x.dims3()?;
        let mut h = x.transpose(1, 2)?.contiguous()?; // [1, T, hidden]
        let (cos, sin) = precompute_rope(
            t,
            self.cfg.head_dim,
            self.cfg.rope_theta,
            &self.w.dev,
            h.dtype(),
        )?;
        let mask = causal_mask_at(0, t, &self.w.dev, h.dtype())?;
        for l in 0..self.cfg.num_hidden_layers {
            let p = format!("{TF}.layers.{l}");
            let residual = h.clone();
            let n = self.layer_norm(&h, &format!("{p}.input_layernorm"))?;
            let a = self.attention(&n, &format!("{p}.self_attn"), &cos, &sin, &mask)?;
            let a = a.broadcast_mul(&self.w.g(&format!("{p}.self_attn_layer_scale.scale"))?)?;
            h = (residual + a)?;

            let residual = h.clone();
            let n = self.layer_norm(&h, &format!("{p}.post_attention_layernorm"))?;
            // `MimiMLP` is fc1 -> gelu -> fc2, NOT SwiGLU: `hidden_act` is "gelu", which
            // `ACT2FN` maps to the exact erf GELU, not the tanh approximation.
            let m = self.w.linear(&n, &format!("{p}.mlp.fc1.weight"), None)?.gelu_erf()?;
            let m = self.w.linear(&m, &format!("{p}.mlp.fc2.weight"), None)?;
            let m = m.broadcast_mul(&self.w.g(&format!("{p}.mlp_layer_scale.scale"))?)?;
            h = (residual + m)?;
        }
        h.transpose(1, 2)?.contiguous()
    }

    /// `MimiModel.downsample`: a bias-free causal conv, kernel 4, stride 2, and — unlike
    /// every conv in the cascade — **`replicate`** padding, passed explicitly at
    /// construction. 25 Hz -> 12.5 Hz.
    fn downsample(&self, x: &Tensor) -> Result<Tensor> {
        self.causal_conv1d(
            x,
            &format!("{DOWN}.conv.weight"),
            None,
            self.cfg.downsample_kernel(),
            1,
            self.cfg.downsample_stride(),
            PadMode::Replicate,
        )
    }

    // --- the chunked cascade --------------------------------------------------

    /// Run the conv cascade over `wav` in `chunk_steps`-step pieces (a step is
    /// `cascade_hop` input samples), returning the full `[1, hidden, ceil(L/hop)]`.
    /// `chunk_steps == 0` runs it one-shot — the parity reference.
    ///
    /// Chunk `[a, b)` is computed from samples `[(a - ctx) * hop, min(b * hop, L))` and
    /// only steps `[a, b)` are kept. Every conv is causal with the
    /// [`CASCADE_RECEPTIVE_FIELD_SAMPLES`] left dependency and no right dependency, so a
    /// kept step sees its entire receptive field and equals the one-shot value exactly.
    /// The final chunk runs to the true end of the waveform, so its per-conv
    /// `extra_padding` matches the one-shot's (the two input lengths differ by a multiple
    /// of every cumulative stride).
    fn cascade_ctx(&self, wav: &Tensor, chunk_steps: usize, ctx_steps: usize) -> Result<Tensor> {
        let hop = self.cfg.cascade_hop();
        let len = wav.dim(D::Minus1)?;
        let steps = len.div_ceil(hop);

        if chunk_steps == 0 || steps <= chunk_steps {
            return self.cascade(wav);
        }

        let mut parts: Vec<Tensor> = Vec::new();
        let mut a = 0usize;
        while a < steps {
            let b = (a + chunk_steps).min(steps);
            let lo = a.saturating_sub(ctx_steps);
            let start = lo * hop;
            let end = (b * hop).min(len);
            let seg = wav.narrow(D::Minus1, start, end - start)?.contiguous()?;
            let x = self.cascade(&seg)?;
            let have = x.dim(D::Minus1)?;
            let off = a - lo;
            let keep = b - a;
            if have < off + keep {
                return Err(candle_core::Error::Msg(format!(
                    "cascade chunk produced {have} steps, expected at least {} for \
                     steps [{lo}, {b}) — the cascade is not length-exact",
                    off + keep
                )));
            }
            parts.push(x.narrow(D::Minus1, off, keep)?);
            a = b;
        }
        Tensor::cat(&parts, D::Minus1)?.contiguous()
    }

    // --- the quantizer --------------------------------------------------------

    /// See [`quantize_split`].
    fn quantize(&self, emb: &Tensor) -> Result<Vec<Vec<u32>>> {
        quantize_split(&self.semantic, &self.acoustic, emb)
    }

    // --- the public entry points ---------------------------------------------

    /// Encode a mono 24 kHz waveform to `[valid_num_quantizers][frames]` codes.
    ///
    /// Accepts `[L]`, `[1, L]` or `[1, 1, L]`. Frame count is `ceil(L / frame_hop)`,
    /// matching the reference's `-(-n_samples // encode_downsample_rate)`.
    pub fn encode(&self, wav: &Tensor) -> Result<Vec<Vec<u32>>> {
        self.encode_with(wav, encode_chunk_steps_env())
    }

    /// [`Self::encode`] with the cascade forced one-shot — the parity reference path.
    pub fn encode_oneshot(&self, wav: &Tensor) -> Result<Vec<Vec<u32>>> {
        self.encode_with(wav, 0)
    }

    /// [`Self::encode`] with an explicit cascade chunk length.
    pub fn encode_with(&self, wav: &Tensor, chunk_steps: usize) -> Result<Vec<Vec<u32>>> {
        let z = self.latent_with(wav, chunk_steps)?;
        self.quantize(&z)
    }

    /// The pre-quantizer latent `[1, hidden, frames]`: cascade (chunked) -> transformer
    /// -> downsample. Exposed so the chunk-equivalence tests can compare the continuous
    /// value, where a discrepancy the argmin would have hidden still shows.
    pub fn latent_with(&self, wav: &Tensor, chunk_steps: usize) -> Result<Tensor> {
        self.latent_ctx(wav, chunk_steps, ENCODE_LEFT_CTX_STEPS)
    }

    /// [`Self::latent_with`] with an explicit left context, so the negative-control test
    /// can sweep it and pin where the receptive field actually ends.
    fn latent_ctx(&self, wav: &Tensor, chunk_steps: usize, ctx_steps: usize) -> Result<Tensor> {
        let wav = match wav.rank() {
            1 => wav.reshape((1, 1, wav.dim(0)?))?,
            2 => wav.unsqueeze(1)?,
            _ => wav.clone(),
        };
        let wav = wav.to_dtype(self.w.dt)?;
        let x = self.cascade_ctx(&wav, chunk_steps, ctx_steps)?;
        let x = self.transformer(&x)?;
        self.downsample(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real rate schedule (ratios 8·6·5·4 -> 960x, then downsample 2 -> 1920x) at toy
    /// channel widths. The receptive field depends only on kernels, dilations and
    /// strides, so the chunking arithmetic these tests defend is the shipped one.
    ///
    /// The `encoder_transformer` is deliberately present. In the sibling Fish port a
    /// fixture that omitted it let the encoder silently skip the bottleneck, and a broken
    /// chunking scheme passed.
    const FIXTURE_JSON: &str = r#"{
      "encoder_valid_num_quantizers": 4,
      "encoder_config": {
        "_frame_rate": 12.5,
        "audio_channels": 1,
        "codebook_dim": 4,
        "codebook_size": 5,
        "compress": 2,
        "dilation_growth_rate": 2,
        "head_dim": 4,
        "hidden_size": 8,
        "intermediate_size": 16,
        "kernel_size": 7,
        "last_kernel_size": 3,
        "layer_scale_initial_scale": 0.01,
        "norm_eps": 1e-05,
        "num_attention_heads": 2,
        "num_filters": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_residual_layers": 1,
        "num_semantic_quantizers": 1,
        "residual_kernel_size": 3,
        "rope_theta": 10000.0,
        "sampling_rate": 24000,
        "upsampling_ratios": [8, 6, 5, 4],
        "vector_quantization_hidden_dimension": 4
      }
    }"#;

    /// Deterministic pseudo-random fill, so every weight is distinct and non-degenerate.
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    struct Fx {
        enc: MimiEncoder,
        cfg: MimiEncoderConfig,
    }

    fn put(map: &mut HashMap<String, Tensor>, name: &str, dims: &[usize], seed: u64) {
        let n: usize = dims.iter().product();
        map.insert(
            name.to_string(),
            Tensor::from_vec(fill(n, seed), dims, &Device::Cpu).unwrap(),
        );
    }

    fn fixture() -> Fx {
        let cfg = MimiEncoderConfig::from_json(FIXTURE_JSON).unwrap();
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut seed = 1u64;
        let mut s = || {
            seed += 1;
            seed
        };

        // --- cascade: mirror the reference's module indexing exactly ---------
        let nf = cfg.num_filters;
        put(&mut m, &format!("{ENC}.layers.0.conv.weight"), &[nf, cfg.audio_channels, cfg.kernel_size], s());
        put(&mut m, &format!("{ENC}.layers.0.conv.bias"), &[nf], s());
        let mut idx = 1usize;
        let mut scaling = 1usize;
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let dim = scaling * nf;
            let hidden = dim / cfg.compress;
            for _ in 0..cfg.num_residual_layers {
                put(&mut m, &format!("{ENC}.layers.{idx}.block.1.conv.weight"), &[hidden, dim, cfg.residual_kernel_size], s());
                put(&mut m, &format!("{ENC}.layers.{idx}.block.1.conv.bias"), &[hidden], s());
                put(&mut m, &format!("{ENC}.layers.{idx}.block.3.conv.weight"), &[dim, hidden, 1], s());
                put(&mut m, &format!("{ENC}.layers.{idx}.block.3.conv.bias"), &[dim], s());
                idx += 1;
            }
            idx += 1; // ELU
            put(&mut m, &format!("{ENC}.layers.{idx}.conv.weight"), &[dim * 2, dim, ratio * 2], s());
            put(&mut m, &format!("{ENC}.layers.{idx}.conv.bias"), &[dim * 2], s());
            idx += 1;
            scaling *= 2;
        }
        idx += 1; // ELU
        put(&mut m, &format!("{ENC}.layers.{idx}.conv.weight"), &[cfg.hidden_size, scaling * nf, cfg.last_kernel_size], s());
        put(&mut m, &format!("{ENC}.layers.{idx}.conv.bias"), &[cfg.hidden_size], s());

        // --- transformer -----------------------------------------------------
        let h = cfg.hidden_size;
        for l in 0..cfg.num_hidden_layers {
            let p = format!("{TF}.layers.{l}");
            for n in ["input_layernorm", "post_attention_layernorm"] {
                put(&mut m, &format!("{p}.{n}.weight"), &[h], s());
                put(&mut m, &format!("{p}.{n}.bias"), &[h], s());
            }
            for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                put(&mut m, &format!("{p}.self_attn.{n}.weight"), &[h, h], s());
            }
            put(&mut m, &format!("{p}.mlp.fc1.weight"), &[cfg.intermediate_size, h], s());
            put(&mut m, &format!("{p}.mlp.fc2.weight"), &[h, cfg.intermediate_size], s());
            put(&mut m, &format!("{p}.self_attn_layer_scale.scale"), &[h], s());
            put(&mut m, &format!("{p}.mlp_layer_scale.scale"), &[h], s());
        }

        // --- downsample ------------------------------------------------------
        put(&mut m, &format!("{DOWN}.conv.weight"), &[h, h, cfg.downsample_kernel()], s());

        // --- quantizer -------------------------------------------------------
        for (name, n) in [
            ("semantic_residual_vector_quantizer", cfg.num_semantic_quantizers),
            ("acoustic_residual_vector_quantizer", cfg.valid_acoustic_quantizers()),
        ] {
            let p = format!("{QUANT}.{name}");
            put(&mut m, &format!("{p}.input_proj.weight"), &[cfg.vq_hidden_dim, h, 1], s());
            put(&mut m, &format!("{p}.output_proj.weight"), &[h, cfg.vq_hidden_dim, 1], s());
            for l in 0..n {
                put(&mut m, &format!("{p}.layers.{l}.codebook.embed_sum"), &[cfg.codebook_size, cfg.codebook_dim], s());
                // usage in [0.5, 1.5) so the EMA quotient is a real, per-row rescale
                let u: Vec<f32> = fill(cfg.codebook_size, s()).iter().map(|v| v + 1.0).collect();
                m.insert(
                    format!("{p}.layers.{l}.codebook.cluster_usage"),
                    Tensor::from_vec(u, (cfg.codebook_size,), &Device::Cpu).unwrap(),
                );
            }
        }

        let w = Weights { map: m, dev: Device::Cpu, dt: DType::F32 };
        let enc = MimiEncoder::new(w, cfg.clone()).unwrap();
        Fx { enc, cfg }
    }

    /// A deterministic non-degenerate waveform of `steps` cascade steps.
    fn wave(fx: &Fx, steps: usize) -> Tensor {
        let n = steps * fx.cfg.cascade_hop();
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.0007).sin() * 0.6 + ((i as f32) * 0.013).cos() * 0.3)
            .collect();
        Tensor::from_vec(v, (n,), &Device::Cpu).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    // --- geometry -------------------------------------------------------------

    /// The published tokenizer config, parsed for the fields this module reads.
    #[test]
    fn parses_the_published_tokenizer_config() {
        let c = MimiEncoderConfig::from_json(
            r#"{"encoder_valid_num_quantizers":16,"encoder_config":{
                "_frame_rate":12.5,"audio_channels":1,"codebook_dim":256,"codebook_size":2048,
                "compress":2,"dilation_growth_rate":2,"head_dim":64,"hidden_size":512,
                "intermediate_size":2048,"kernel_size":7,"last_kernel_size":3,
                "layer_scale_initial_scale":0.01,"norm_eps":1e-05,"num_attention_heads":8,
                "num_filters":64,"num_hidden_layers":8,"num_key_value_heads":8,
                "num_residual_layers":1,"num_semantic_quantizers":1,"residual_kernel_size":3,
                "rope_theta":10000.0,"sampling_rate":24000,"upsampling_ratios":[8,6,5,4],
                "vector_quantization_hidden_dimension":256}}"#,
        )
        .unwrap();
        assert_eq!(c.cascade_hop(), 960);
        assert_eq!(c.encodec_frame_rate(), 25);
        assert_eq!(c.downsample_kernel(), 4);
        assert_eq!(c.downsample_stride(), 2);
        // `encode_downsample_rate` from the top-level config.
        assert_eq!(c.frame_hop(), 1920);
        // 1 semantic + 15 acoustic, out of the 32 the checkpoint carries.
        assert_eq!(c.valid_acoustic_quantizers(), 15);
        assert_eq!(c.hidden_size, 512);
        assert_eq!(c.num_attention_heads * c.head_dim, 512, "q/k/v/o_proj width");
    }

    /// The frame count must be `ceil(L / 1920)` for lengths on both sides of a frame
    /// boundary, matching the reference's `-(-n // encode_downsample_rate)` trim.
    #[test]
    fn frame_count_is_ceil_of_the_hop() {
        let fx = fixture();
        let hop = fx.cfg.frame_hop();
        for (samples, want) in [
            (hop, 1usize),
            (hop + 1, 2),
            (2 * hop - 1, 2),
            (2 * hop, 2),
            (2 * hop + 1, 3),
        ] {
            let v = Tensor::from_vec(vec![0.1f32; samples], (samples,), &Device::Cpu).unwrap();
            let z = fx.enc.latent_with(&v, 0).unwrap();
            assert_eq!(z.dim(D::Minus1).unwrap(), want, "L={samples}");
            let codes = fx.enc.encode_oneshot(&v).unwrap();
            assert_eq!(codes[0].len(), want, "L={samples}: codes");
        }
    }

    /// Every configured quantizer emits one in-range code per frame.
    #[test]
    fn emits_one_in_range_code_per_quantizer_per_frame() {
        let fx = fixture();
        let codes = fx.enc.encode_oneshot(&wave(&fx, 8)).unwrap();
        assert_eq!(codes.len(), fx.cfg.valid_num_quantizers);
        for row in &codes {
            assert_eq!(row.len(), 4, "8 cascade steps -> 4 code frames");
            assert!(row.iter().all(|c| (*c as usize) < fx.cfg.codebook_size));
        }
    }

    // --- the RVQ analysis semantics ------------------------------------------

    /// The codebook the search runs against is the EMA quotient, and it comes from
    /// [`Rvq`] rather than a second copy of the formula.
    #[test]
    fn codebook_is_the_ema_quotient() {
        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "p.layers.0.codebook.embed_sum".into(),
            Tensor::from_vec(vec![2f32, 4., 6., 8.], (2, 2), &dev).unwrap(),
        );
        m.insert(
            "p.layers.0.codebook.cluster_usage".into(),
            Tensor::from_vec(vec![2f32, 4.], (2,), &dev).unwrap(),
        );
        let w = Weights { map: m, dev, dt: DType::F32 };
        let rvq = load_codebooks(&w, "p", 1).unwrap();
        assert_eq!(flat(rvq.codebook(0).unwrap()), vec![1.0, 2.0, 1.5, 2.0]);
        // the raw accumulator would be [2,4,6,8] — a different rescale per row
        assert_ne!(flat(rvq.codebook(0).unwrap()), vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// The nearest-centroid search is plain Euclidean on the projected vectors — no L2
    /// normalisation — and the residual is subtracted between layers.
    ///
    /// Layer 0's codebook is `[[0,0],[10,0]]` and layer 1's is `[[0,0],[0,1]]`. The input
    /// `[9, 4]` is nearer `[10,0]` (distance 4.12) than `[0,0]` (9.85), so layer 0 emits
    /// 1 — while a *cosine* search would prefer `[0,0]`... which has no direction, so the
    /// discriminating case is the second point `[1, 9]`: Euclidean picks `[0,0]`
    /// (distance 9.06 vs 12.7) even though its projection onto `[10,0]` is positive.
    /// After subtracting, layer 1 sees `[-1, 4]` and `[1, 9]` and picks `[0,1]` for both.
    #[test]
    fn search_is_euclidean_on_unnormalised_vectors() {
        let dev = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        // identity input_proj so the fed embedding is what the search sees
        m.insert(
            "p.input_proj.weight".into(),
            Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2, 1), &dev).unwrap(),
        );
        for (l, rows) in [(0usize, vec![0f32, 0., 10., 0.]), (1, vec![0f32, 0., 0., 1.])] {
            m.insert(
                format!("p.layers.{l}.codebook.embed_sum"),
                Tensor::from_vec(rows, (2, 2), &dev).unwrap(),
            );
            m.insert(
                format!("p.layers.{l}.codebook.cluster_usage"),
                Tensor::from_vec(vec![1f32, 1.], (2,), &dev).unwrap(),
            );
        }
        let w = Weights { map: m, dev: dev.clone(), dt: DType::F32 };
        let stack = RvqStack::load(&w, "p", 2).unwrap();
        // [1, channels=2, T=2]: points [9,4] and [1,9]
        let emb = Tensor::from_vec(vec![9f32, 1., 4., 9.], (1, 2, 2), &dev).unwrap();
        let rows = stack.encode(&emb).unwrap();
        assert_eq!(rows[0], vec![1u32, 0], "layer 0 nearest centroid");
        assert_eq!(rows[1], vec![1u32, 1], "layer 1 sees the residual");
    }

    /// **The acoustic stack reads the original embedding, not the semantic residual.**
    ///
    /// `MimiSplitResidualVectorQuantizer::encode` calls both stacks on the same tensor;
    /// only *within* a stack is the quantisation residual. Getting this wrong yields
    /// plausible codes that reconstruct the wrong voice, so it is pinned on a case built
    /// to separate the two readings exactly.
    ///
    /// Both stacks use an identity `input_proj`, the semantic codebook is
    /// `[[0,0], [10,0]]` and the acoustic one is `[[9,0], [-1,0]]`. For the embedding
    /// `[9, 0]`:
    ///
    /// * semantic picks index 1 (`[10,0]`, distance 1, against 9) — so the residual a
    ///   *serial* chain would hand on is `[-1, 0]`;
    /// * the shipped **parallel** reading gives acoustic the untouched `[9, 0]` and it
    ///   picks index 0 (distance 0, against 10);
    /// * the **serial** misreading gives it `[-1, 0]` and it picks index 1.
    #[test]
    fn acoustic_stack_is_parallel_to_semantic_not_serial() {
        fn stack(rows: Vec<f32>) -> RvqStack {
            let dev = Device::Cpu;
            let mut m: HashMap<String, Tensor> = HashMap::new();
            m.insert(
                "p.input_proj.weight".into(),
                Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2, 1), &dev).unwrap(),
            );
            m.insert(
                "p.layers.0.codebook.embed_sum".into(),
                Tensor::from_vec(rows, (2, 2), &dev).unwrap(),
            );
            m.insert(
                "p.layers.0.codebook.cluster_usage".into(),
                Tensor::from_vec(vec![1f32, 1.], (2,), &dev).unwrap(),
            );
            let w = Weights { map: m, dev, dt: DType::F32 };
            RvqStack::load(&w, "p", 1).unwrap()
        }
        let sem = stack(vec![0f32, 0., 10., 0.]);
        let aco = stack(vec![9f32, 0., -1., 0.]);
        let emb = Tensor::from_vec(vec![9f32, 0.], (1, 2, 1), &Device::Cpu).unwrap();

        let got = quantize_split(&sem, &aco, &emb).unwrap();
        assert_eq!(got, vec![vec![1u32], vec![0u32]], "parallel split RVQ");

        // what a serial (nested-residual) chain would have produced
        let cb = sem.rvq.codebook(0).unwrap();
        let idx = Tensor::from_vec(vec![1u32], (1,), &Device::Cpu).unwrap();
        let q = cb.index_select(&idx, 0).unwrap().t().unwrap().contiguous().unwrap();
        let residual = (emb - q.unsqueeze(0).unwrap()).unwrap();
        assert_eq!(aco.encode(&residual).unwrap(), vec![vec![1u32]], "serial reading");
    }

    /// The whole-encoder path agrees with the two stacks run independently on the latent
    /// — i.e. `quantize` really is the parallel split, at real geometry.
    #[test]
    fn encode_splits_semantic_and_acoustic_over_the_same_latent() {
        let fx = fixture();
        let z = fx.enc.latent_with(&wave(&fx, 12), 0).unwrap();
        let zf = z.to_dtype(DType::F32).unwrap();
        let all = fx.enc.quantize(&z).unwrap();
        assert_eq!(all[..fx.cfg.num_semantic_quantizers].to_vec(), fx.enc.semantic.encode(&zf).unwrap());
        assert_eq!(all[fx.cfg.num_semantic_quantizers..].to_vec(), fx.enc.acoustic.encode(&zf).unwrap());
    }

    // --- chunking -------------------------------------------------------------

    /// Chunked encode reproduces the one-shot latent **exactly**, for every chunk length —
    /// including chunks smaller than the left context, chunks that do not divide the step
    /// count, and chunks at/over it (which take the one-shot branch).
    #[test]
    fn chunked_latent_matches_oneshot() {
        let fx = fixture();
        let steps = 24usize;
        let wav = wave(&fx, steps);
        let want = fx.enc.latent_with(&wav, 0).unwrap();
        let want_v = flat(&want);
        for chunk in [2usize, 3, 5, 8, 13, 16, 23, 24, 48] {
            let got = fx.enc.latent_with(&wav, chunk).unwrap();
            assert_eq!(got.dims(), want.dims(), "chunk={chunk}: latent shape");
            let (i, d) = flat(&got)
                .iter()
                .zip(&want_v)
                .map(|(g, w)| (g - w).abs())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert_eq!(d, 0.0, "chunk={chunk}: latent {i} differs by {d}");
        }
    }

    /// The same, through the argmin: chunking must not move a single code.
    #[test]
    fn chunked_codes_match_oneshot() {
        let fx = fixture();
        let wav = wave(&fx, 24);
        let want = fx.enc.encode_oneshot(&wav).unwrap();
        for chunk in [2usize, 3, 7, 16, 24, 48] {
            assert_eq!(fx.enc.encode_with(&wav, chunk).unwrap(), want, "chunk={chunk}");
        }
    }

    /// The NEGATIVE CONTROL for [`ENCODE_LEFT_CTX_STEPS`].
    ///
    /// Re-runs the chunked cascade with a deliberately truncated left context and asserts
    /// it *fails* below the derived 3320-sample receptive field and *passes* at and above
    /// it. Without this an undersized constant would corrupt the reference silently — in
    /// the sibling Fish port a context one step short still agreed to 6e-6, which no
    /// tolerance-based assert would have caught.
    #[test]
    fn left_ctx_matches_the_receptive_field() {
        let fx = fixture();
        let wav = wave(&fx, 24);
        let want = flat(&fx.enc.latent_ctx(&wav, 0, ENCODE_LEFT_CTX_STEPS).unwrap());
        let worst = |ctx: usize| -> f32 {
            flat(&fx.enc.latent_ctx(&wav, 8, ctx).unwrap())
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max)
        };
        let hop = fx.cfg.cascade_hop();
        let bound = CASCADE_RECEPTIVE_FIELD_SAMPLES.div_ceil(hop);
        assert_eq!(bound, 4, "3320 samples / 960 -> 4 steps");
        for ctx in 0..bound {
            assert!(
                worst(ctx) > 0.0,
                "ctx={ctx} is below the {CASCADE_RECEPTIVE_FIELD_SAMPLES}-sample receptive \
                 field but matched exactly — the derivation or the chunking is wrong"
            );
        }
        for ctx in [bound, bound + 1, 8, 16] {
            assert_eq!(worst(ctx), 0.0, "ctx={ctx} should be sufficient");
        }
        assert!(
            ENCODE_LEFT_CTX_STEPS >= bound,
            "the shipped constant must cover the receptive field"
        );
    }

    /// The env override selects the chunk length, and `0` selects the one-shot path.
    /// (Serialised with the other env test by running both assertions here.)
    #[test]
    fn env_override_selects_the_chunk_length() {
        let key = "SYRINX_QWEN_CODEC_ENCODE_CHUNK";
        std::env::remove_var(key);
        assert_eq!(encode_chunk_steps_env(), DEFAULT_ENCODE_CHUNK_STEPS);
        std::env::set_var(key, "7");
        assert_eq!(encode_chunk_steps_env(), 7);
        std::env::set_var(key, "0");
        assert_eq!(encode_chunk_steps_env(), 0);
        // a value that does not parse falls back to the default rather than panicking
        std::env::set_var(key, "not-a-number");
        assert_eq!(encode_chunk_steps_env(), DEFAULT_ENCODE_CHUNK_STEPS);
        std::env::remove_var(key);
    }

    /// The transformer really runs: zeroing its layer scales (the only path from the
    /// attention/MLP branches into the residual stream) must change the latent. This is
    /// the guard the Fish port lacked — a fixture whose bottleneck was silently skipped.
    #[test]
    fn the_transformer_bottleneck_is_in_the_path() {
        let fx = fixture();
        let wav = wave(&fx, 8);
        let with = flat(&fx.enc.latent_with(&wav, 0).unwrap());

        let mut m = fx.enc.w.map.clone();
        for l in 0..fx.cfg.num_hidden_layers {
            for n in ["self_attn_layer_scale", "mlp_layer_scale"] {
                m.insert(
                    format!("{TF}.layers.{l}.{n}.scale"),
                    Tensor::zeros((fx.cfg.hidden_size,), DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        let w = Weights { map: m, dev: Device::Cpu, dt: DType::F32 };
        let flat_enc = MimiEncoder::new(w, fx.cfg.clone()).unwrap();
        let without = flat(&flat_enc.latent_with(&wav, 0).unwrap());
        assert_ne!(with, without, "the bottleneck contributed nothing");
    }

    // --- parity against the Python reference ---------------------------------

    /// Read `n` little-endian f32 from a raw dump.
    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }

    /// **Numeric parity against the HuggingFace `MimiModel` encode stack.**
    ///
    /// SKIPs unless both env vars are set, mirroring the on-box convention in
    /// `scripts/test-all.env`:
    ///
    /// * `SYRINX_QWEN_TOKENIZER_DIR` — the published `Qwen3-TTS-Tokenizer-12Hz` folder
    ///   (`config.json` + `model.safetensors`).
    /// * `SYRINX_QWEN_ENCODER_REF` — a folder holding `ref_meta.json`
    ///   (`{samples, channels, frames, n_q}`), `ref_wav.f32`, `ref_latent.f32`
    ///   (`[channels, frames]`, row-major) and `ref_codes.u32` (`[n_q, frames]`),
    ///   produced by running the reference:
    ///
    /// ```python
    /// enc_cfg = MimiConfig(**cfg["encoder_config"])          # minus dtype/version keys
    /// m = Qwen3TTSTokenizerV2Encoder(enc_cfg).eval()         # loads encoder.* strict
    /// emb = m.encoder(wav[None, None])
    /// tf  = m.encoder_transformer(emb.transpose(1, 2), return_dict=True)[0].transpose(1, 2)
    /// lat = m.downsample(tf)
    /// codes = m.quantizer.encode(lat).transpose(0, 1)[0][:16]
    /// ```
    ///
    /// This runs on CPU — no GPU needed — so it is the one part of this port that can be
    /// confirmed off-box. It checks the latent (continuous, where any drift shows) *and*
    /// the codes (discrete, where an argmin flip shows), through both the one-shot and
    /// the chunked cascade.
    #[test]
    fn matches_the_python_reference() {
        let (Ok(dir), Ok(refdir)) = (
            std::env::var("SYRINX_QWEN_TOKENIZER_DIR"),
            std::env::var("SYRINX_QWEN_ENCODER_REF"),
        ) else {
            eprintln!("SKIP matches_the_python_reference: set SYRINX_QWEN_TOKENIZER_DIR and SYRINX_QWEN_ENCODER_REF");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let refdir = std::path::PathBuf::from(refdir);

        let cfg = MimiEncoderConfig::from_json(
            &std::fs::read_to_string(dir.join("config.json")).unwrap(),
        )
        .unwrap();
        let map = candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu).unwrap();
        let w = Weights { map, dev: Device::Cpu, dt: DType::F32 };
        let enc = MimiEncoder::new(w, cfg.clone()).unwrap();

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(refdir.join("ref_meta.json")).unwrap())
                .unwrap();
        let frames = meta["frames"].as_u64().unwrap() as usize;
        let channels = meta["channels"].as_u64().unwrap() as usize;
        let n_q = meta["n_q"].as_u64().unwrap() as usize;

        let wav_v = read_f32(&refdir.join("ref_wav.f32"));
        let wav = Tensor::from_vec(wav_v.clone(), (wav_v.len(),), &Device::Cpu).unwrap();

        let want_lat = read_f32(&refdir.join("ref_latent.f32"));
        let got_lat = flat(&enc.latent_with(&wav, 0).unwrap());
        assert_eq!(got_lat.len(), channels * frames, "latent shape");
        let worst = got_lat
            .iter()
            .zip(&want_lat)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        // f32 CPU throughout on both sides; the residual gap is reassociation in the
        // conv/matmul reductions only.
        eprintln!("latent max abs diff {worst}");
        assert!(worst < 2e-3, "latent max abs diff {worst}");

        let want_codes: Vec<u32> = std::fs::read(refdir.join("ref_codes.u32"))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        assert_eq!(want_codes.len(), n_q * frames);
        for (chunk, label) in [(0usize, "one-shot"), (2, "chunked")] {
            let got = enc.encode_with(&wav, chunk).unwrap();
            assert_eq!(got.len(), cfg.valid_num_quantizers, "{label}: quantizer count");
            for (q, row) in got.iter().enumerate() {
                assert_eq!(
                    row.as_slice(),
                    &want_codes[q * frames..(q + 1) * frames],
                    "{label}: quantizer {q}"
                );
            }
        }
    }

    /// `downsample` pads with `replicate`, not zeros: a constant-valued input must stay
    /// at the value the conv gives for a full window, which zero-padding would not.
    #[test]
    fn downsample_uses_replicate_padding() {
        let fx = fixture();
        let h = fx.cfg.hidden_size;
        let t = 6usize;
        let x = Tensor::from_vec(vec![1f32; h * t], (1, h, t), &Device::Cpu).unwrap();
        let got = flat(&fx.enc.downsample(&x).unwrap());
        // With replicate padding every window is all-ones, so every output channel is the
        // sum of its kernel and every frame is identical.
        let kw = fx.enc.w.g(&format!("{DOWN}.conv.weight")).unwrap();
        let want_ch = flat(&kw.sum(D::Minus1).unwrap().sum(D::Minus1).unwrap());
        let frames = got.len() / h;
        assert_eq!(frames, t / 2);
        for c in 0..h {
            for f in 0..frames {
                let d = (got[c * frames + f] - want_ch[c]).abs();
                assert!(d < 1e-5, "channel {c} frame {f}: {d}");
            }
        }
    }
}
