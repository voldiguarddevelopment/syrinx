//! Real **CosyVoice3** HiFT vocoder forward via Candle (CPU) — the
//! `CausalHiFTGenerator` mel -> 24 kHz waveform path, parity-verified against the
//! dumped CV3 reference. Gated behind the `real` cargo feature + on-disk weights,
//! mirroring [`crate::real`] (the CV2 port).
//!
//! ## What CV3 changes vs CV2 ([`crate::real::HiftVocoder`])
//! The architecture is the same HiFTNet (NSF + iSTFTNet) skeleton — `conv_pre`,
//! 3 upsample stages with source/F0 STFT fusion, Snake ResBlocks, a `conv_post`
//! magnitude/phase head and an iSTFT — but two deltas matter:
//!
//! 1. **Causal convolutions.** Every conv is a `CausalConv1d`
//!    (`cosyvoice/transformer/convolution.py`), which pads *entirely on one side*
//!    rather than symmetrically. For kernel `k`, dilation `d`, stride 1 the total
//!    pad is `causal_padding = (k*d - d) - ... = d*(k-1)` (the source formula
//!    `int((k*d-d)/2)*2 + (k+1)%2` reduces to exactly `d*(k-1)` for every kernel we
//!    use). It is placed on the **left** (`causal_type='left'`) for all convs
//!    except `conv_pre` and `f0_predictor.condnet[0]`, which look ahead and pad on
//!    the **right** (`causal_type='right'`, `conv_pre_look_right=4`). Output length
//!    equals input length (`assert x.shape[2] == input_timestep`).
//!    The upsample stages are **nearest-neighbour upsample + a left-causal conv**
//!    (`CausalConv1dUpsample`), not the CV2 `ConvTranspose1d`.
//!
//! 2. **float64 f0_predictor.** `CausalConvRNNF0Predictor` is run in float64
//!    (`self.f0_predictor.to(torch.float64)`; "precision is crucial for causal
//!    inference"), so the whole condnet + classifier is computed in f64 here.
//!
//! ## Weight loading
//! Unlike the CV2 checkpoint (folded offline), `hift_fp32.safetensors` here still
//! carries the `weight_norm` parametrization (`...parametrizations.weight.original0`
//! = g `[out,1,1]`, `original1` = v `[out,in,k]`). This loader folds them on fetch:
//! `weight = g * v / ||v||` with the norm over all dims but `0` — and folds in the
//! **requested dtype** so the f64 f0 path matches torch's `to(float64)` recompute.
//! Plain (non-parametrized) weights — `source_downs.*`, `classifier`,
//! `m_source.l_linear` — are used as-is.
//!
//! ## Determinism / inputs
//! As in CV2, the source/F0 branch's `SineGen` is stochastic (seeded Gaussian +
//! random phase) and not reproducible across torch/Candle RNGs. The reference dumps
//! the SineGen **`source`** waveform `[1,1,97920]`; [`Cv3Hift::decode`] consumes it
//! verbatim, computes its STFT (`_stft`), and reproduces the deterministic decode.
//!
//! ## Module layout
//! Pure structural split of the original single `real_cv3.rs` (logic byte-unchanged):
//!   * this `mod.rs` — the [`Cv3Hift`] struct, loaders, `weight_norm` fold, the
//!     `CausalConv1d` primitive;
//!   * [`f0`] — the float64 `f0_predictor` + `m_source` merge weights;
//!   * [`stft`] — the source `_stft` and the iSTFT head;
//!   * [`decode`] — the causal upsample/fusion/ResBlock stack + `decode`;
//!   * [`quant`] — the int4 `Q4_0` quantizer.

use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

mod decode;
mod f0;
mod quant;
mod stft;

use quant::{maybe_quantize_weight, QStore};

const N_FFT: usize = 16;
const HOP: usize = 4;
const N_BINS: usize = N_FFT / 2 + 1; // 9 (onesided)
const AUDIO_LIMIT: f64 = 0.99;
const LRELU_SLOPE: f64 = 0.1;
const SNAKE_EPS: f64 = 1e-9;
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
/// ResBlock dilations (`resblock_dilation_sizes = [[1,3,5], ...]`).
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
/// `conv_pre_look_right`: `conv_pre` is `CausalConv1d(k=look_right+1, type='right')`.
const CONV_PRE_LOOK_RIGHT: usize = 4;

/// The real CosyVoice3 (`CausalHiFTGenerator`) HiFT vocoder, loaded from the
/// `weight_norm`-parametrized fp32 safetensors checkpoint.
///
/// Two precisions share one struct + one forward, like the CV2 [`crate::real::HiftVocoder`]:
///   * **fp32 (default, parity)** — [`Cv3Hift::load`], every tensor kept in `raw`
///     (folded/cast per-fetch), byte-unchanged.
///   * **int4 (`load_quantized`)** — the large decode conv kernels (the `ups` upsample
///     convs, the Snake `ResBlock` `convs1/2`, `source_resblocks`, and `conv_post`) are
///     `weight_norm`-folded then stored `Q4_0` in `qw` and dequantized on fetch; biases,
///     Snake `alpha`s, the non-aligned `conv_pre`/`source_downs` kernels, and the **entire
///     float64 `f0_predictor`** stay dense in `raw` (the f0 path's precision is load-bearing,
///     so it is never quantized).
pub struct Cv3Hift {
    /// Raw safetensors tensors (kept f32; folded/cast per-fetch). In the int4 build the
    /// quantized convs' `parametrizations.weight.original0/1` are removed from here (they
    /// move into `qw`), so dense and quantized bytes are never double-counted.
    raw: HashMap<String, Tensor>,
    /// Q4_0 dequant-on-fetch weights, keyed by the logical `"<prefix>.weight"` name (empty
    /// in the fp32 build).
    qw: HashMap<String, QStore>,
    /// Sum of the `QTensor` storage bytes realized by quantization (0 in the fp32 build).
    quant_bytes: usize,
    dev: Device,
}

/// Realized weight footprint of a loaded [`Cv3Hift`], split into the int4-quantized and
/// retained dense (f32) parts.
#[derive(Debug, Clone, Copy)]
pub struct Cv3HiftFootprint {
    /// Bytes held by the `Q4_0` quantized weights (0 in the fp32 build).
    pub quant_bytes: usize,
    /// Bytes held by the retained dense tensors (biases, Snake alphas, small/odd kernels,
    /// the float64 f0_predictor, and any un-folded `weight_norm` g/v of un-quantized convs).
    pub dense_bytes: usize,
    /// Number of weights quantized to int4.
    pub n_quantized: usize,
}

impl Cv3HiftFootprint {
    /// Total realized weight bytes (`quant + dense`).
    pub fn total_bytes(&self) -> usize {
        self.quant_bytes + self.dense_bytes
    }
    /// Total realized weight footprint in mebibytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
}

impl Cv3Hift {
    /// Load `hift_fp32.safetensors` (still `weight_norm`-parametrized) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = candle_core::safetensors::load(path, &dev)?;
        Ok(Self { raw, qw: HashMap::new(), quant_bytes: 0, dev })
    }

    /// Load the same checkpoint but quantize the large decode conv kernels to GGML `Q4_0`
    /// (dequant-on-fetch) for a smaller vocoder footprint — the README size goal, mirroring
    /// the CV2 [`crate::real::HiftVocoder::load_quantized`].
    ///
    /// Each `weight_norm`-parametrized conv (`g = original0`, `v = original1`) is folded
    /// (`weight = g·v/‖v‖`) to its logical `[out,in,k]` kernel; if that kernel is large
    /// (≥ [`quant::QUANT_MIN_ELEMS`]) and its flattened `in·k` inner dim is 32-aligned, it is
    /// reshaped to a 2-D block store, int4 quantized into `qw`, and its raw `g`/`v` are
    /// dropped from `raw`. On lookup [`Cv3Hift::weight`] dequantizes + reshapes back, so the
    /// `decode` math is byte-for-byte the fp32 path on the dequantized weight.
    ///
    /// Kept dense (f32, in `raw`): all biases, the Snake `alpha`s, the non-32-aligned
    /// `conv_pre` (`in·k = 400`) and the plain `source_downs` kernels, `m_source.l_linear`,
    /// and — crucially — the **whole `f0_predictor`** (run in float64; "precision is crucial
    /// for causal inference"), which is excluded from quantization by name so the f64 f0 path
    /// is byte-unchanged.
    ///
    /// NOTE this **does** quantize convolution kernels (not just linears) — the only way the
    /// conv-dominated vocoder yields a real size win; int4 on conv kernels trades quality
    /// for size. ⚠️ Opt-in **size**, not speed (dequant-on-fetch). The on-box SIM-o eval is
    /// the arbiter, and the fp32 [`load`](Self::load) path stays available for full quality.
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        let mut raw = candle_core::safetensors::load(path, &dev)?;
        let mut qw: HashMap<String, QStore> = HashMap::new();
        let mut quant_bytes = 0usize;

        // Every parametrized conv weight is signalled by its `...original1` (the `v` tensor).
        let bases: Vec<String> = raw
            .keys()
            .filter_map(|k| {
                k.strip_suffix(".parametrizations.weight.original1")
                    .map(|b| b.to_string())
            })
            .collect();

        for base in bases {
            // The f0_predictor runs in float64; never quantize it (keep full precision).
            if base.starts_with("f0_predictor") {
                continue;
            }
            let g_key = format!("{base}.parametrizations.weight.original0");
            let v_key = format!("{base}.parametrizations.weight.original1");
            let (g, v) = match (raw.get(&g_key), raw.get(&v_key)) {
                (Some(g), Some(v)) => (g.clone(), v.clone()),
                _ => continue,
            };
            let folded = fold_weight_norm(&g, &v, DType::F32)?; // [out,in,k]
            if let Some(qs) = maybe_quantize_weight(&folded)? {
                quant_bytes += qs.qt.storage_size_in_bytes();
                qw.insert(format!("{base}.weight"), qs);
                // Drop the raw g/v so dense and quantized bytes are never double-counted.
                raw.remove(&g_key);
                raw.remove(&v_key);
            }
        }

        Ok(Self { raw, qw, quant_bytes, dev })
    }

    /// Realized weight footprint (quantized + dense) of this loaded vocoder.
    pub fn footprint(&self) -> Cv3HiftFootprint {
        let dense_bytes: usize = self
            .raw
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        Cv3HiftFootprint {
            quant_bytes: self.quant_bytes,
            dense_bytes,
            n_quantized: self.qw.len(),
        }
    }

    /// Fetch a plain (non-parametrized) tensor cast to `dtype`.
    fn raw_t(&self, name: &str, dtype: DType) -> Result<Tensor> {
        self.raw
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor: {name}")))?
            .to_dtype(dtype)
    }

    /// Resolve a conv `weight` in `dtype`, folding `weight_norm` if parametrized.
    ///
    /// `name` is the logical `"<prefix>.weight"`. If
    /// `"<prefix>.parametrizations.weight.original0/1"` exist they are folded:
    /// `weight = g * v / ||v||_2`, the norm taken over every dim except `0`
    /// (torch `weight_norm(dim=0)`). Done in `dtype` to mirror torch's
    /// `module.to(float64)` recompute on the f0 path. Otherwise the plain tensor.
    fn weight(&self, name: &str, dtype: DType) -> Result<Tensor> {
        // int4 dequant-on-fetch (quantized build): the large decode conv kernels live in
        // `qw` keyed by the logical `".weight"` name. Dequantize (-> f32), restore the
        // original `[out,in,k]` shape, cast to `dtype`. The f0_predictor is never quantized,
        // so its float64 fetches never hit this branch and stay full precision.
        if let Some(q) = self.qw.get(name) {
            return q
                .qt
                .dequantize(&self.dev)?
                .reshape(q.shape.clone())?
                .to_dtype(dtype);
        }
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let g_key = format!("{base}.parametrizations.weight.original0");
        let v_key = format!("{base}.parametrizations.weight.original1");
        match (self.raw.get(&g_key), self.raw.get(&v_key)) {
            (Some(g), Some(v)) => {
                let g = g.to_dtype(dtype)?; // [out,1,1]
                let v = v.to_dtype(dtype)?; // [out,in,k]
                // ||v||_2 over (in, k), keepdim -> [out,1,1].
                let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
                v.broadcast_mul(&g)?.broadcast_div(&norm)
            }
            _ => self.raw_t(name, dtype),
        }
    }

    /// `CausalConv1d` (stride 1): pad `dilation*(k-1)` entirely on one side
    /// (`left` true = `causal_type='left'`, false = `'right'`), conv with no
    /// internal padding. Output length equals input length.
    fn causal_conv(
        &self,
        x: &Tensor,
        wname: &str,
        bname: &str,
        dilation: usize,
        left: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let w = self.weight(wname, dtype)?; // [out,in,k]
        let b = self.raw_t(bname, dtype)?; // [out]
        let k = w.dim(2)?;
        let pad = dilation * (k - 1);
        let xp = if left {
            x.pad_with_zeros(2, pad, 0)?
        } else {
            x.pad_with_zeros(2, 0, pad)?
        };
        let y = xp.conv1d(&w, 0, 1, dilation, 1)?;
        let b = b.reshape((1, b.dim(0)?, 1))?;
        y.broadcast_add(&b)
    }
}

/// Fold a `weight_norm` parametrization to its logical kernel in `dtype`:
/// `weight = g · v / ‖v‖`, the L2 norm taken over every dim except `0` (torch
/// `weight_norm(dim=0)`). `g` is `[out,1,1]`, `v` is `[out,in,k]`; returns `[out,in,k]`.
/// The same math [`Cv3Hift::weight`] applies on the fp32/f64 fetch path, hoisted here so
/// [`Cv3Hift::load_quantized`] can fold-then-quantize once at load.
fn fold_weight_norm(g: &Tensor, v: &Tensor, dtype: DType) -> Result<Tensor> {
    let g = g.to_dtype(dtype)?; // [out,1,1]
    let v = v.to_dtype(dtype)?; // [out,in,k]
    let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?; // [out,1,1]
    v.broadcast_mul(&g)?.broadcast_div(&norm)
}
