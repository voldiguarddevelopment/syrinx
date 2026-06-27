//! FCM head, CAM-DenseTDNN blocks and the full `forward` pass of the CAM++ encoder.
//! Split out of `real.rs` unchanged; the methods extend [`super::CamPlus`].

use candle_core::{DType, Result, Tensor, D};

use super::pooling::{seg_pool_broadcast, stats_pool};
use super::{CamPlus, BLOCKS, SEG};

impl CamPlus {
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
    pub(super) fn head(&self, fbank: &Tensor) -> Result<Tensor> {
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
    pub(super) fn tdnnd(&self, x: &Tensor, prefix: &str, dilation: usize, pad: usize) -> Result<Tensor> {
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

    pub(super) fn dense_block(&self, x: &Tensor, block: &str, n: usize, dilation: usize, pad: usize) -> Result<Tensor> {
        let mut h = x.clone();
        for i in 1..=n {
            h = self.tdnnd(&h, &format!("xvector.{block}.tdnnd{i}"), dilation, pad)?;
        }
        Ok(h)
    }

    /// TransitLayer: BN -> ReLU -> conv1d(k=1). `has_bias` per the ONNX graph.
    pub(super) fn transit(&self, x: &Tensor, prefix: &str, has_bias: bool) -> Result<Tensor> {
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
}

// ---- free helpers -----------------------------------------------------------

/// Sigmoid via candle (`1 / (1 + exp(-x))`).
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(x)
}
