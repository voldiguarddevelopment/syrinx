//! Real CosyVoice2 speaker encoder (CAM++ / campplus) forward via Candle — the real
//! model behind the toy reference parity, mirroring the LM `real.rs` recipe.
//!
//! The encoder ships only as `campplus.onnx` (no PyTorch source in CosyVoice), so this
//! is a hand-port of the 3D-Speaker / WeSpeaker **CAM++** architecture recovered from
//! the ONNX graph + initializer hierarchy. It maps an 80-dim kaldi fbank of a 16 kHz
//! reference clip (`[B, T, 80]`, mean-normalised over time by the caller) to a fixed
//! **192-d x-vector** speaker embedding (`[B, 192]`).
//!
//! Gated behind the `real` cargo feature and a weights path on disk (the fp32
//! safetensors exported from the ONNX initializers — too large to vendor). The parity
//! test skips cleanly when the weights/fixture are absent (device-bound task recipe).
//!
//! Architecture (recovered from `campplus.onnx`, 3206 nodes / 617 initializers):
//!
//! - **FCM head** (`head`): input `[B,T,80]` -> transpose `[B,80,T]` -> unsqueeze
//!   `[B,1,80,T]`; conv2d(1->32, 3x3, stride(1,1)) + ReLU; two BasicResBlocks
//!   (`layer1`: 32ch, first block stride(2,1) w/ 1x1 shortcut, second stride(1,1)
//!   identity; `layer2`: same with its own stride(2,1) downsample); conv2d(32->32,
//!   3x3, stride(2,1)) + ReLU. Freq 80 -> 10 over three stride-2 freq downsamples;
//!   reshape `[B, 32*10=320, T]`.
//! - **xvector** (CAM-DenseTDNN): `tdnn` conv1d(320->128, k=5, pad=2, **stride=2**,
//!   bias) downsamples time; `block1` (12 CAMDenseTDNNLayers, growth 32, dilation 1),
//!   `transit1` BN+ReLU then conv1d(512->256, k=1, no bias); `block2` (24 layers,
//!   dilation 2), `transit2` (1024->512); `block3` (16 layers, dilation 2), `transit3`
//!   BN+ReLU then conv1d(1024->512, k=1, **bias**); `out_nonlinear` ReLU; statistics
//!   pooling (mean ++ std over time -> 1024); `dense` conv1d(1024->192, k=1, no bias)
//!   then a final BatchNorm1d with **affine=False** (running stats only, eps 1e-5).
//!
//! Each **CAMDenseTDNNLayer** (`tdnndN`): `y = relu(BN(x))`; `y = relu(linear1(y))`
//! (conv1d k=1, in->128, bias); `y = cam_layer(y)`; output is `cat([x, y], dim=1)`
//! (dense growth). **CAMLayer**: `local = linear_local(y)` (conv1d k=3, dilation d,
//! pad d, in->growth, **no bias**); a context vector `ctx = mean_t(y) + seg_pool(y)`
//! where `seg_pool` average-pools time in segments of 100 (ceil mode) and broadcasts
//! back; `m = sigmoid(linear2(relu(linear1(ctx))))` (conv1d k=1, 128->64->32, bias);
//! output `local * m`.
//!
//! All BatchNorm is inference-mode: `(x - mean)/sqrt(var+eps) * gamma + beta`,
//! eps 1e-5. All matmuls/accumulations are f32 (CPU), enough for the 1e-3 parity bar.
//!
//! ## Module layout
//! Pure structural split of the original single `real.rs` (logic byte-unchanged):
//!   * this `mod.rs` — the [`CamPlus`] struct, loaders, primitive conv/BatchNorm ops;
//!   * [`tdnn`] — the FCM head, CAM-DenseTDNN blocks and the `forward` pass;
//!   * [`pooling`] — statistics pooling + CAM segment pooling;
//!   * [`quant`] — the int4 `Q4_0` quantizer + the parity-diagnostic forwards.

use candle_core::{safetensors, DType, Device, Result, Tensor};
use std::collections::HashMap;

mod pooling;
mod quant;
mod tdnn;

use quant::{maybe_quantize_weight, QStore};

const BN_EPS: f64 = 1e-5;
const SEG: usize = 100; // CAM segment pooling window (kernel=stride=100, ceil mode)

/// Per-block CAMDenseTDNN config: (num layers, dilation, kernel padding).
/// dilation 1 -> pad 1 (block1); dilation 2 -> pad 2 (block2/3). kernel size is 3.
const BLOCKS: [(&str, usize, usize); 3] = [
    ("block1", 12, 1), // dilation 1
    ("block2", 24, 2), // dilation 2
    ("block3", 16, 2), // dilation 2
];

/// The real CAM++ speaker encoder, loaded from fp32 safetensors (ONNX-exported).
pub struct CamPlus {
    w: HashMap<String, Tensor>,
    /// Q4_0 dequant-on-fetch weights (empty in the fp32 / f64 builds).
    qw: HashMap<String, QStore>,
    /// Sum of the `QTensor` storage bytes realized by quantization (0 in the fp32 build).
    quant_bytes: usize,
    dev: Device,
    dtype: DType,
}

/// Realized weight footprint of a loaded [`CamPlus`], split int4-quantized vs dense f32.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerFootprint {
    /// Bytes held by the `Q4_0` quantized weights (0 in the fp32 build).
    pub quant_bytes: usize,
    /// Bytes held by the retained dense f32 weights (biases, BN stats, small kernels).
    pub dense_bytes: usize,
    /// Number of weights quantized to int4.
    pub n_quantized: usize,
}

impl SpeakerFootprint {
    /// Total realized weight bytes (`quant + dense`).
    pub fn total_bytes(&self) -> usize {
        self.quant_bytes + self.dense_bytes
    }
    /// Total realized weight footprint in mebibytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
}

impl CamPlus {
    /// Load the fp32 weight dump (`campplus_weights.safetensors`) onto `dev` for the
    /// fp32 forward (`load`), or in f64 for a higher-precision accumulation path
    /// (`load_with_dtype`). The campplus weights are stored fp32; the model box is CPU
    /// so f64 is the fully-precise reference that fp32 onnxruntime itself approximates.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        Self::load_with_dtype(path, dev, DType::F32)
    }

    /// Load with an explicit compute dtype. `DType::F64` makes the whole forward
    /// accumulate in double precision (the conv/matmul gemm runs f64), which lands
    /// closer to the exact math than either fp32 path's accumulation order.
    pub fn load_with_dtype(path: &str, dev: Device, dtype: DType) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(dtype)?);
        }
        Ok(Self { w, qw: HashMap::new(), quant_bytes: 0, dev, dtype })
    }

    /// Load the campplus weights but quantize the large block-aligned weight matrices to
    /// GGML `Q4_0` (dequant-on-fetch) for a smaller speaker footprint (quant pass goal #4).
    ///
    /// Quantized: every weight whose flattened `[out, rest]` matrix has a 32-aligned inner
    /// dim and ≥ [`quant::QUANT_MIN_ELEMS`] elements — the CAM-DenseTDNN conv1d kernels, the
    /// FCM head conv2d kernels, and the transit/dense linears (the footprint bulk). They are
    /// reshaped to a 2-D block store, int4 quantized, and reconstructed to the original
    /// shape on lookup, so the forward is byte-for-byte the f32 path on the dequantized
    /// weight. Biases, BatchNorm running stats and small/odd kernels stay dense f32.
    ///
    /// Compute dtype is F32 (the dequantized weights are f32); the high-precision f64
    /// reference path stays on the fp32/f64 [`load`](Self::load) loaders. int4 trades
    /// quality for size; the on-box SIM-o eval is the arbiter.
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        let mut qw = HashMap::new();
        let mut quant_bytes = 0usize;
        for (k, v) in raw {
            let vf = v.to_dtype(DType::F32)?;
            if let Some(qs) = maybe_quantize_weight(&vf)? {
                quant_bytes += qs.qt.storage_size_in_bytes();
                qw.insert(k, qs);
            } else {
                w.insert(k, vf);
            }
        }
        Ok(Self { w, qw, quant_bytes, dev, dtype: DType::F32 })
    }

    /// Realized weight footprint (quantized + dense) of this loaded encoder.
    pub fn footprint(&self) -> SpeakerFootprint {
        let dense_bytes: usize = self
            .w
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        SpeakerFootprint {
            quant_bytes: self.quant_bytes,
            dense_bytes,
            n_quantized: self.qw.len(),
        }
    }

    fn g(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.w.get(name) {
            return Ok(t.clone());
        }
        if let Some(q) = self.qw.get(name) {
            // Dequantize (-> f32) and restore the original logical shape.
            return q.qt.dequantize(&self.dev)?.reshape(q.shape.clone());
        }
        Err(candle_core::Error::Msg(format!("missing weight: {name}")))
    }

    fn has(&self, name: &str) -> bool {
        self.w.contains_key(name) || self.qw.contains_key(name)
    }

    // ---- primitive ops -------------------------------------------------------

    /// conv2d with explicit padding/stride; `x` is `[B, Cin, H, W]`, weight
    /// `[Cout, Cin, kh, kw]`, optional bias `[Cout]`.
    fn conv2d(
        &self,
        x: &Tensor,
        wname: &str,
        bias: Option<&str>,
        pad: usize,
        stride: (usize, usize),
    ) -> Result<Tensor> {
        // candle conv2d takes a single (pad, stride, dilation, groups). The head uses
        // symmetric padding (1,1) on both H and W, so a scalar pad is exact here.
        let w = self.g(wname)?;
        // candle's conv2d only supports a single stride; the head needs stride(sh, sw)
        // = (2,1) on some convs. Emulate asymmetric stride by conv at stride 1 then
        // strided-slice on H — but cleaner: do conv with stride 1 then subsample H.
        let (sh, sw) = stride;
        let mut y = if sh == sw {
            x.conv2d(&w, pad, sh, 1, 1)?
        } else {
            // stride 1 conv with same padding, then take every sh-th row on H, sw-th on W.
            let y = x.conv2d(&w, pad, 1, 1, 1)?;
            let y = subsample(&y, 2, sh)?; // H axis
            subsample(&y, 3, sw)? // W axis
        };
        if let Some(b) = bias {
            let bt = self.g(b)?.reshape((1, (), 1, 1))?;
            y = y.broadcast_add(&bt)?;
        }
        Ok(y)
    }

    /// conv1d; `x` is `[B, Cin, T]`, weight `[Cout, Cin, k]`, optional bias `[Cout]`.
    fn conv1d(
        &self,
        x: &Tensor,
        wname: &str,
        bias: Option<&str>,
        pad: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let w = self.g(wname)?;
        let mut y = x.conv1d(&w, pad, stride, dilation, 1)?;
        if let Some(b) = bias {
            let bt = self.g(b)?.reshape((1, (), 1))?;
            y = y.broadcast_add(&bt)?;
        }
        Ok(y)
    }

    /// Inference BatchNorm over channel dim of a `[B, C, *]` tensor.
    /// `(x - running_mean)/sqrt(running_var + eps) * gamma + beta`. When the affine
    /// params (`weight`/`bias`) are absent, affine=False (gamma=1, beta=0).
    fn batch_norm(&self, x: &Tensor, prefix: &str, rank: usize) -> Result<Tensor> {
        let mean = self.g(&format!("{prefix}.running_mean"))?;
        let var = self.g(&format!("{prefix}.running_var"))?;
        // reshape stats to broadcast over [B, C, ...]
        let shape: Vec<usize> = std::iter::once(1usize)
            .chain(std::iter::once(mean.dim(0)?))
            .chain(std::iter::repeat(1).take(rank - 2))
            .collect();
        let mean = mean.reshape(shape.clone())?;
        let var = var.reshape(shape.clone())?;
        let denom = (var + BN_EPS)?.sqrt()?;
        let mut y = x.broadcast_sub(&mean)?.broadcast_div(&denom)?;
        if self.has(&format!("{prefix}.weight")) {
            let gamma = self.g(&format!("{prefix}.weight"))?.reshape(shape.clone())?;
            let beta = self.g(&format!("{prefix}.bias"))?.reshape(shape)?;
            y = y.broadcast_mul(&gamma)?.broadcast_add(&beta)?;
        }
        Ok(y)
    }

    /// Convenience: the encoder's input device (so callers can stage tensors).
    pub fn device(&self) -> &Device {
        &self.dev
    }
}

// ---- free helpers -----------------------------------------------------------

/// Take every `step`-th index along `dim` starting at 0 (emulates strided conv on an
/// axis candle's single-stride conv can't address independently). `start_pad` mirrors
/// the "same"-padded stride-2 behaviour: candle pads symmetrically, ONNX/PyTorch take
/// output index `o` from input `o*stride`, so index 0,stride,2*stride,... is correct.
fn subsample(x: &Tensor, dim: usize, step: usize) -> Result<Tensor> {
    if step == 1 {
        return Ok(x.clone());
    }
    let n = x.dim(dim)?;
    let idx: Vec<u32> = (0..n).step_by(step).map(|i| i as u32).collect();
    let index = Tensor::from_vec(idx.clone(), idx.len(), x.device())?;
    x.index_select(&index, dim)
}
