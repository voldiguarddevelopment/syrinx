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
//!
//! ## Module layout (pure structural split; see the per-file docs)
//! * [`load`] — fp32 / int4 loading, footprint, the quant-weight predicate.
//! * [`conformer`] — `UpsampleConformerEncoder` (subsample, rel-pos, pre-lookahead,
//!   upsample, conformer layers, rel-pos attention).
//! * [`cfm`] — the CFM Euler/CFG solver variants.
//! * [`estimator`] — the U-Net CFM estimator (`CausalConditionalDecoder`).
//! * [`streaming`] — [`StreamCfg`], the chunk-mask, the streaming flow forward, and the
//!   streaming/non-streaming `token2wav` glue.
//!
//! This `mod.rs` keeps the [`Flow`] struct + its fields, the constants, the core
//! `linear`/`layer_norm`/weight-fetch helpers, the public non-streaming forwards, and
//! the shared math free fns — all visible to the submodules above (which are
//! descendants) without widening any visibility.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use std::collections::HashMap;

mod cfm;
mod conformer;
mod estimator;
mod load;
mod streaming;

// Re-export the streaming surface so the public API stays at `real::*`
// (`real::StreamCfg`, `real::token2wav`, ...), unchanged by the split.
pub use streaming::{
    token2wav, token2wav_streaming, AudioChunk, StreamCfg, StreamSourceFn, MEL_CACHE_LEN,
    SOURCE_CACHE_LEN, SOURCE_PER_MEL, TOKEN_MEL_RATIO,
};

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
/// UpsampleConformerEncoder `up_layer.stride`: the encoder 2x-upsamples between its
/// first-stage and up-stage conformer blocks, so the up-stage sequence length (and
/// hence its chunk size) is `ENC_UPSAMPLE`× the first-stage's.
const ENC_UPSAMPLE: usize = 2;

/// The real CosyVoice2 flow `CausalMaskedDiffWithXvec`, loaded from fp32 safetensors.
///
/// Two precisions share this one struct and one forward, exactly like the LM:
///   * **fp32 (default, parity)** — every weight kept in `w` as f32; [`Flow::load`],
///     byte-unchanged. `linear` is the plain `x @ Wᵀ` reference matmul.
///   * **int4 (`load_quantized`)** — every plain-`linear()` weight (the conformer
///     `linear_q/k/v/pos/out` + `feed_forward.w_1/w_2`, the estimator transformer
///     `to_q/k/v` + `to_out` + gelu-FF, the `time_mlp`/resnet `mlp.1`, the
///     `spk_embed_affine` / `encoder_proj` / subsample projections) is quantized to GGML
///     `Q4_0` and lives in `qmm` as [`QMatMul`]; `linear` dispatches to it. Conv kernels,
///     LayerNorm/`pos_bias`/biases and the `input_embedding` lookup stay f32 in `w` (they
///     are not plain `x @ Wᵀ` matmuls). All forward math is otherwise identical.
pub struct Flow {
    w: HashMap<String, Tensor>,
    /// Quantized `linear()` weights (int4), keyed by the same name as the fp32 weight.
    /// Empty for the fp32 build; populated by [`Flow::load_quantized`].
    qmm: HashMap<String, QMatMul>,
    /// Sum of the `QTensor` storage bytes realized by quantization (0 in the fp32 build).
    quant_bytes: usize,
    dev: Device,
}

/// Realized weight footprint of a loaded [`Flow`], split into the int4-quantized
/// `linear()` weights and the retained dense (f32) part (conv kernels, norms, biases,
/// `input_embedding`). `total_bytes` is the headline size number.
#[derive(Debug, Clone, Copy)]
pub struct FlowFootprint {
    /// Bytes held by the `Q4_0` quantized `linear()` weights (0 in the fp32 build).
    pub quant_bytes: usize,
    /// Bytes held by the retained dense f32 weights.
    pub dense_bytes: usize,
    /// Number of `linear()` weights quantized to int4.
    pub n_quantized: usize,
}

impl FlowFootprint {
    /// Total realized weight bytes (`quant + dense`).
    pub fn total_bytes(&self) -> usize {
        self.quant_bytes + self.dense_bytes
    }
    /// Total realized weight footprint in mebibytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
}

impl Flow {
    fn g(&self, name: &str) -> Result<Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
            .cloned()
    }

    /// `x @ W^T (+ b)` for `[.., in]` input and `[out, in]` weight.
    ///
    /// When a quantized `QMatMul` exists for `w` (the int4 build) it computes the same
    /// `x @ Wᵀ` with an int4 weight (QMatMul requires a contiguous f32 input); otherwise
    /// it is the dense fp32 matmul. The bias, when present, is always added in f32. The
    /// fp32 build has no `qmm` entry, so its path is byte-for-byte the original matmul.
    fn linear(&self, x: &Tensor, w: &str, b: Option<&str>) -> Result<Tensor> {
        let y = if let Some(qm) = self.qmm.get(w) {
            qm.forward(&x.contiguous()?)?
        } else {
            let weight = self.g(w)?;
            if weight.dtype() == DType::F32 {
                x.broadcast_matmul(&weight.t()?)?
            } else {
                // f16 dense fallback (non-block-aligned weight): upcast for the matmul.
                x.broadcast_matmul(&weight.to_dtype(DType::F32)?.t()?)?
            }
        };
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
}

// ============================ shared math free fns ============================

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

fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(x, D::Minus1)
}
