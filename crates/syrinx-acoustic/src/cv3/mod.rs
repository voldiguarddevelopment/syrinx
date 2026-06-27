//! Real CosyVoice3 flow-matching mel decoder via Candle (`CausalMaskedDiffWithDiT`).
//!
//! The CV3 flow differs from CV2 (`CausalMaskedDiffWithXvec`, see [`crate::cv2`]) in
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
//! around the DiT rather than shared by reference (CV2's `crate::cv2` stays byte-frozen).
//!
//! Gated behind the `real` feature + on-disk fp32 weights; the parity test
//! (`tests/real_cv3_flow_parity.rs`) skips cleanly when the weights/reference are
//! absent and runs CPU fp32 where they exist. Single-utterance, non-streaming,
//! full-context inference (the path the reference dumper records): all padding masks
//! are all-true, so attention is unmasked.
//!
//! ## Module layout (pure structural split; see the per-file docs)
//! * [`frontend`] — token -> mu (`input_embedding` + `pre_lookahead` + `repeat_interleave`).
//! * [`dit`] — the 22-layer DiT estimator (blocks, AdaLN, rotary, conv-pos-embed, attn/FF).
//! * [`cfm`] — the CFM Euler/CFG solver.
//! * [`streaming`] — [`Cv3StreamCfg`], the chunk mask, and the streaming flow forward.
//!
//! This `mod.rs` keeps the [`Cv3Flow`] struct + its fields, the constants, the loading +
//! `linear`/weight-fetch helpers, the public non-streaming [`Cv3Flow::forward`], and the
//! shared `pad_time`/`conv1d` free fns — all visible to the submodules above (which are
//! descendants) without widening any visibility.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{safetensors, DType, Device, Module, Result, Tensor};
use std::collections::HashMap;

mod cfm;
mod dit;
mod frontend;
mod streaming;

// Re-export the streaming config so the public API stays at `real_cv3::Cv3StreamCfg`.
pub use streaming::Cv3StreamCfg;

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
///
/// Two precisions share one struct + one forward, exactly like the CV2 [`crate::cv2::Flow`]:
///   * **fp32 (default, parity)** — [`Cv3Flow::load`], every weight kept f32 in `w`;
///     `linear` is the plain `x @ Wᵀ` reference matmul. Byte-unchanged.
///   * **int4 (`load_quantized`)** — every plain-`linear()` weight (the DiT's large 2-D
///     linears: per-block attention `to_q/k/v` + `to_out`, the tanh-GELU FF
///     `1024→2048`/`2048→1024`, the AdaLN modulation Linears (`attn_norm`/`norm_out`),
///     `input_embed.proj`, `proj_out`, the time-MLP, and `spk_embed_affine`) is quantized
///     to GGML `Q4_0` and lives in `qmm` as a [`QMatMul`]; `linear` dispatches to it. The
///     `PreLookahead`/`ConvPosEmb` conv kernels, all biases, and the `input_embedding`
///     token table stay f32 in `w` (none are plain `x @ Wᵀ` matmuls). Forward math is
///     otherwise identical.
pub struct Cv3Flow {
    w: HashMap<String, Tensor>,
    /// Quantized `linear()` weights (int4 `Q4_0`), keyed by the same name as the fp32
    /// weight. Empty for the fp32 build; populated by [`Cv3Flow::load_quantized`].
    qmm: HashMap<String, QMatMul>,
    /// Sum of the `QTensor` storage bytes realized by quantization (0 in the fp32 build).
    quant_bytes: usize,
    dev: Device,
}

/// Realized weight footprint of a loaded [`Cv3Flow`], split into the int4-quantized
/// `linear()` weights and the retained dense (f32) part (conv kernels, biases, the
/// `input_embedding` table). `total_bytes` is the headline size number.
#[derive(Debug, Clone, Copy)]
pub struct Cv3FlowFootprint {
    /// Bytes held by the `Q4_0` quantized `linear()` weights (0 in the fp32 build).
    pub quant_bytes: usize,
    /// Bytes held by the retained dense f32 weights.
    pub dense_bytes: usize,
    /// Number of `linear()` weights quantized to int4.
    pub n_quantized: usize,
}

impl Cv3FlowFootprint {
    /// Total realized weight bytes (`quant + dense`).
    pub fn total_bytes(&self) -> usize {
        self.quant_bytes + self.dense_bytes
    }
    /// Total realized weight footprint in mebibytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
}

impl Cv3Flow {
    /// Load the converted fp32 checkpoint (`flow_fp32.safetensors`) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self { w, qmm: HashMap::new(), quant_bytes: 0, dev })
    }

    /// Load the same `flow_fp32.safetensors`, but quantize every plain-`linear()` weight to
    /// **int4** (GGML `Q4_0`) — the README size goal, mirroring [`crate::cv2::Flow::load_quantized`].
    ///
    /// Quantized (one `QMatMul` each): every 2-D weight whose inner (`in_features`) dim is a
    /// multiple of the 32-element `Q4_0` block — which is exactly the DiT's `linear()`
    /// weights (attention `to_q/k/v`/`to_out`, the tanh-GELU FF, the AdaLN `attn_norm`/
    /// `norm_out` modulation Linears, `input_embed.proj`, `proj_out`, the time-MLP) plus
    /// `spk_embed_affine`. These are true `x @ Wᵀ` matmuls, so `QMatMul::forward` is the
    /// same op with an int4 weight.
    ///
    /// Kept dense (f32): the `PreLookahead` + `ConvPosEmb` conv kernels (3-D), all biases
    /// and the `rotary_embed.inv_freq` table (1-D), and the `input_embedding` token table
    /// `[6561, 80]` (an `index_select` gather, and its inner dim 80 isn't block-aligned
    /// anyway). A generic 2-D-block-aligned test (rather than a name allowlist) means no
    /// large dense matmul can be silently left in f32.
    ///
    /// int4 trades accuracy for size; the forward is otherwise byte-identical to
    /// [`Cv3Flow::load`]. ⚠️ This is an opt-in **size**, not speed, tradeoff — same caveat as
    /// the CV2 path. The on-box SIM-o eval measures the quality cost.
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        let mut qmm = HashMap::new();
        let mut quant_bytes = 0usize;
        for (k, v) in raw {
            let vf = v.to_dtype(DType::F32)?;
            let dims = vf.dims();
            let inner = *dims.last().unwrap_or(&0);
            // The `input_embedding` table is a row-lookup gather, never a `linear()` matmul:
            // keep it dense even though it is 2-D. (Its inner dim 80 isn't 32-aligned, so it
            // would be skipped regardless — the explicit guard documents the intent.)
            let is_embed_table = k == "input_embedding.weight";
            if !is_embed_table
                && dims.len() == 2
                && inner % GgmlDType::Q4_0.block_size() == 0
            {
                let qt = QTensor::quantize(&vf, GgmlDType::Q4_0)?;
                quant_bytes += qt.storage_size_in_bytes();
                qmm.insert(k, QMatMul::from_qtensor(qt)?);
                continue;
            }
            // Conv kernels (3-D), biases / inv_freq (1-D), and the embed table stay dense f32.
            w.insert(k, vf);
        }
        Ok(Self { w, qmm, quant_bytes, dev })
    }

    /// Realized weight footprint (quantized + dense) of this loaded flow.
    pub fn footprint(&self) -> Cv3FlowFootprint {
        let dense_bytes: usize = self
            .w
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        Cv3FlowFootprint {
            quant_bytes: self.quant_bytes,
            dense_bytes,
            n_quantized: self.qmm.len(),
        }
    }

    fn g(&self, name: &str) -> Result<Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
            .cloned()
    }

    /// `x @ W^T (+ b)` for `[.., in]` input and `[out, in]` weight.
    ///
    /// When a quantized `QMatMul` exists for `w` (the int4 build) it computes the same
    /// `x @ Wᵀ` with an int4 weight (QMatMul requires a contiguous f32 input); otherwise it
    /// is the dense fp32 matmul. The bias, when present, is always added in f32. The fp32
    /// build has no `qmm` entry, so its path is byte-for-byte the original reference matmul.
    fn linear(&self, x: &Tensor, w: &str, b: Option<&str>) -> Result<Tensor> {
        let y = if let Some(qm) = self.qmm.get(w) {
            qm.forward(&x.contiguous()?)?
        } else {
            let weight = self.g(w)?;
            x.broadcast_matmul(&weight.t()?)?
        };
        match b {
            Some(bn) => y.broadcast_add(&self.g(bn)?),
            None => Ok(y),
        }
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

// ============================ shared math free fns ============================

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
