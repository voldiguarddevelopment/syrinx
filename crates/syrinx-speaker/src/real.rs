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

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

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
    dev: Device,
    dtype: DType,
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
        Ok(Self { w, dev, dtype })
    }

    fn g(&self, name: &str) -> Result<Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
            .cloned()
    }

    fn has(&self, name: &str) -> bool {
        self.w.contains_key(name)
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

    // ---- FCM head ------------------------------------------------------------

    /// One BasicResBlock: `relu(BN-free conv path + shortcut)` (the campplus head
    /// uses bias convs, no BN inside the residual blocks per the ONNX graph).
    /// `conv1` (stride `s`, pad 1) -> ReLU -> `conv2` (stride 1, pad 1); shortcut is a
    /// 1x1 conv (stride `s`) when `has_shortcut`, else identity.
    fn res_block(&self, x: &Tensor, prefix: &str, s: usize, has_shortcut: bool) -> Result<Tensor> {
        let y = self.conv2d(x, &format!("{prefix}.conv1.weight"), Some(&format!("{prefix}.conv1.bias")), 1, (s, 1))?;
        let y = y.relu()?;
        let y = self.conv2d(&y, &format!("{prefix}.conv2.weight"), Some(&format!("{prefix}.conv2.bias")), 1, (1, 1))?;
        let sc = if has_shortcut {
            self.conv2d(
                x,
                &format!("{prefix}.shortcut.shortcut.0.weight"),
                Some(&format!("{prefix}.shortcut.shortcut.0.bias")),
                0,
                (s, 1),
            )?
        } else {
            x.clone()
        };
        (y + sc)?.relu()
    }

    /// FCM head: `[B, T, 80]` -> `[B, 320, T']`.
    fn head(&self, fbank: &Tensor) -> Result<Tensor> {
        // cast input to the compute dtype, then [B,T,80] -> [B,80,T] -> [B,1,80,T]
        let fbank = fbank.to_dtype(self.dtype)?;
        let x = fbank.transpose(1, 2)?.unsqueeze(1)?.contiguous()?;
        // conv1: 1->32, 3x3, stride(1,1), pad 1, +ReLU
        let x = self
            .conv2d(&x, "head.conv1.weight", Some("head.conv1.bias"), 1, (1, 1))?
            .relu()?;
        // layer1: block0 (stride 2, shortcut), block1 (stride 1, identity)
        let x = self.res_block(&x, "head.layer1.layer1.0", 2, true)?;
        let x = self.res_block(&x, "head.layer1.layer1.1", 1, false)?;
        // layer2: block0 (stride 2, shortcut), block1 (stride 1, identity)
        let x = self.res_block(&x, "head.layer2.layer2.0", 2, true)?;
        let x = self.res_block(&x, "head.layer2.layer2.1", 1, false)?;
        // conv2: 32->32, 3x3, stride(2,1), pad 1, +ReLU
        let x = self
            .conv2d(&x, "head.conv2.weight", Some("head.conv2.bias"), 1, (2, 1))?
            .relu()?;
        // x: [B, 32, 10, T] -> reshape [B, 320, T]
        let (b, c, f, t) = x.dims4()?;
        x.reshape((b, c * f, t))
    }

    // ---- CAM-DenseTDNN -------------------------------------------------------

    /// CAMLayer: `y` `[B,128,T]` -> `[B, GROWTH, T]`.
    fn cam_layer(&self, y: &Tensor, prefix: &str, dilation: usize, pad: usize) -> Result<Tensor> {
        // local conv: k=3, dilation, pad, 128->GROWTH, no bias
        let local = self.conv1d(y, &format!("{prefix}.linear_local.weight"), None, pad, 1, dilation)?;
        // global context: mean over time, keepdim -> [B,128,1]
        let context_global = y.mean_keepdim(D::Minus1)?;
        // segment context: average-pool time in windows of SEG (ceil mode), broadcast back
        let context_seg = seg_pool_broadcast(y, SEG)?; // [B,128,T]
        // context = global + seg  (global broadcasts over T)
        let context = context_seg.broadcast_add(&context_global)?; // [B,128,T]
        // attention MLP: 128->64 (relu) ->32, k=1 convs with bias, then sigmoid
        let m = self.conv1d(&context, &format!("{prefix}.linear1.weight"), Some(&format!("{prefix}.linear1.bias")), 0, 1, 1)?;
        let m = m.relu()?;
        let m = self.conv1d(&m, &format!("{prefix}.linear2.weight"), Some(&format!("{prefix}.linear2.bias")), 0, 1, 1)?;
        let m = sigmoid(&m)?;
        local.broadcast_mul(&m)
    }

    /// One CAMDenseTDNNLayer: dense growth, returns `cat([x, branch], dim=1)`.
    fn tdnnd(&self, x: &Tensor, prefix: &str, dilation: usize, pad: usize) -> Result<Tensor> {
        // nonlinear1: BN -> ReLU
        let y = self.batch_norm(x, &format!("{prefix}.nonlinear1.batchnorm"), 3)?;
        let y = y.relu()?;
        // linear1: conv1d k=1, in->128, bias
        let y = self.conv1d(&y, &format!("{prefix}.linear1.weight"), Some(&format!("{prefix}.linear1.bias")), 0, 1, 1)?;
        // nonlinear2: ReLU
        let y = y.relu()?;
        // cam_layer -> [B, GROWTH, T]
        let branch = self.cam_layer(&y, &format!("{prefix}.cam_layer"), dilation, pad)?;
        // dense concat along channel
        Tensor::cat(&[x, &branch], 1)
    }

    fn dense_block(&self, x: &Tensor, block: &str, n: usize, dilation: usize, pad: usize) -> Result<Tensor> {
        let mut h = x.clone();
        for i in 1..=n {
            h = self.tdnnd(&h, &format!("xvector.{block}.tdnnd{i}"), dilation, pad)?;
        }
        Ok(h)
    }

    /// TransitLayer: BN -> ReLU -> conv1d(k=1). `has_bias` per the ONNX graph.
    fn transit(&self, x: &Tensor, prefix: &str, has_bias: bool) -> Result<Tensor> {
        let y = self.batch_norm(x, &format!("{prefix}.nonlinear.batchnorm"), 3)?;
        let y = y.relu()?;
        let bias = if has_bias { Some(format!("{prefix}.linear.bias")) } else { None };
        self.conv1d(&y, &format!("{prefix}.linear.weight"), bias.as_deref(), 0, 1, 1)
    }

    /// Full forward: `[B, T, 80]` fbank -> `[B, 192]` x-vector.
    pub fn forward(&self, fbank: &Tensor) -> Result<Tensor> {
        // FCM head -> [B, 320, T']
        let x = self.head(fbank)?;
        // tdnn: linear (conv1d 320->128, k=5, pad=2, stride=2, bias) then nonlinear ReLU
        let x = self.conv1d(&x, "xvector.tdnn.linear.weight", Some("xvector.tdnn.linear.bias"), 2, 2, 1)?;
        let x = x.relu()?;
        // dense blocks + transits
        let x = self.dense_block(&x, BLOCKS[0].0, BLOCKS[0].1, 1, BLOCKS[0].2)?;
        let x = self.transit(&x, "xvector.transit1", false)?;
        let x = self.dense_block(&x, BLOCKS[1].0, BLOCKS[1].1, 2, BLOCKS[1].2)?;
        let x = self.transit(&x, "xvector.transit2", false)?;
        let x = self.dense_block(&x, BLOCKS[2].0, BLOCKS[2].1, 2, BLOCKS[2].2)?;
        // transit3: BN -> ReLU -> conv1d(k=1, bias) -> 512
        let x = self.transit(&x, "xvector.transit3", true)?;
        // out_nonlinear: ReLU
        let x = x.relu()?;
        // statistics pooling: [mean_t(x) ++ std_t(x)] over time -> [B, 2*C]
        let pooled = stats_pool(&x)?; // [B, 1024]
        // dense: conv1d k=1, 1024->192, no bias. Reshape pooled -> [B,1024,1].
        let p = pooled.unsqueeze(2)?; // [B,1024,1]
        let d = self.conv1d(&p, "xvector.dense.linear.weight", None, 0, 1, 1)?; // [B,192,1]
        // final BN (affine=False)
        let d = self.batch_norm(&d, "xvector.dense.nonlinear.batchnorm", 3)?;
        // squeeze time -> [B, 192], back to f32 for the parity comparison
        d.squeeze(2)?.to_dtype(DType::F32)
    }

    /// Convenience: the encoder's input device (so callers can stage tensors).
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Staged forward returning labelled intermediates for parity diagnostics:
    /// `head` `[B,320,T]`, `tdnn` `[B,128,T']`, `transit1` `[B,256,T']`,
    /// `out_nonlinear` `[B,512,T']`, `stats` `[B,1024]`, `dense_pre_bn` `[B,192,1]`.
    pub fn forward_staged(&self, fbank: &Tensor) -> Result<Vec<(String, Tensor)>> {
        let mut s = Vec::new();
        let x = self.head(fbank)?;
        s.push(("head".into(), x.clone()));
        let x = self.conv1d(&x, "xvector.tdnn.linear.weight", Some("xvector.tdnn.linear.bias"), 2, 2, 1)?;
        s.push(("tdnn".into(), x.clone()));
        let x = self.dense_block(&x, BLOCKS[0].0, BLOCKS[0].1, 1, BLOCKS[0].2)?;
        let x = self.transit(&x, "xvector.transit1", false)?;
        s.push(("transit1".into(), x.clone()));
        let x = self.dense_block(&x, BLOCKS[1].0, BLOCKS[1].1, 2, BLOCKS[1].2)?;
        let x = self.transit(&x, "xvector.transit2", false)?;
        let x = self.dense_block(&x, BLOCKS[2].0, BLOCKS[2].1, 2, BLOCKS[2].2)?;
        let x = self.transit(&x, "xvector.transit3", true)?;
        let x = x.relu()?;
        s.push(("out_nonlinear".into(), x.clone()));
        let pooled = stats_pool(&x)?;
        s.push(("stats".into(), pooled.clone()));
        let p = pooled.unsqueeze(2)?;
        let d = self.conv1d(&p, "xvector.dense.linear.weight", None, 0, 1, 1)?;
        s.push(("dense_pre_bn".into(), d.clone()));
        Ok(s)
    }

    /// Per-layer block1 outputs (the 12 dense concats) for parity localisation.
    pub fn block1_stages(&self, fbank: &Tensor) -> Result<Vec<Tensor>> {
        let x = self.head(fbank)?;
        let x = self.conv1d(&x, "xvector.tdnn.linear.weight", Some("xvector.tdnn.linear.bias"), 2, 2, 1)?;
        let mut out = Vec::new();
        let mut h = x;
        for i in 1..=12 {
            h = self.tdnnd(&h, &format!("xvector.block1.tdnnd{i}"), 1, 1)?;
            out.push(h.clone());
        }
        Ok(out)
    }
}

// ---- free helpers -----------------------------------------------------------

/// Sigmoid via candle (`1 / (1 + exp(-x))`).
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(x)
}

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

/// Segment average pooling over time with window `seg` (kernel=stride=seg, ceil mode),
/// then broadcast each segment's mean back across its frames -> `[B, C, T]`.
///
/// ONNX path: Pad(0) -> AveragePool(kernel=seg, stride=seg, ceil_mode=1) -> the pooled
/// `[B,C,nseg]` is expanded back to T by `nearest`-style repeat (each output frame t
/// maps to segment `t // seg`). The pad is zero-width here (Constant pad value list is
/// all zeros), so AveragePool with ceil mode over T frames yields
/// `nseg = ceil(T/seg)` segments; the last (partial) segment averages only its real
/// frames (count_include_pad has no effect since pad width is 0).
fn seg_pool_broadcast(x: &Tensor, seg: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let nseg = t.div_ceil(seg);
    // Per-segment mean. Build by averaging each window (the last may be partial).
    let mut means: Vec<Tensor> = Vec::with_capacity(nseg);
    for s in 0..nseg {
        let start = s * seg;
        let len = seg.min(t - start);
        let win = x.narrow(2, start, len)?; // [B,C,len]
        means.push(win.mean_keepdim(2)?); // [B,C,1]
    }
    let seg_means = Tensor::cat(&means, 2)?; // [B,C,nseg]
    // Broadcast back: output frame t -> segment t/seg. Build a gather index.
    let idx: Vec<u32> = (0..t).map(|i| (i / seg) as u32).collect();
    let index = Tensor::from_vec(idx, t, x.device())?;
    let out = seg_means.index_select(&index, 2)?; // [B,C,T]
    debug_assert_eq!(out.dims(), &[b, c, t]);
    Ok(out)
}

/// Statistics pooling: concat of time-mean and time-std over a `[B, C, T]` tensor ->
/// `[B, 2C]`. Matches the ONNX `stats` block exactly:
///   mean   = ReduceMean_t(x)
///   popvar = ReduceMean_t((x - mean)^2)
///   var    = popvar * T / (T - 1)            (the **unbiased / sample** estimator:
///            ONNX does Mul by N then Div by (N-1))
///   std    = sqrt(var)
/// and concatenates `[mean, std]` along the channel axis.
fn stats_pool(x: &Tensor) -> Result<Tensor> {
    let t = x.dim(D::Minus1)? as f64;
    let mean = x.mean_keepdim(D::Minus1)?; // [B,C,1]
    let centered = x.broadcast_sub(&mean)?;
    let popvar = centered.sqr()?.mean(D::Minus1)?; // [B,C]
    // unbiased: popvar * T/(T-1)
    let scale = t / (t - 1.0);
    let var = (popvar * scale)?;
    // numerical guard: clamp tiny negatives from fp rounding before sqrt
    let var = var.relu()?;
    let std = var.sqrt()?;
    let mean = mean.squeeze(D::Minus1)?; // [B,C]
    Tensor::cat(&[&mean, &std], 1)
}
