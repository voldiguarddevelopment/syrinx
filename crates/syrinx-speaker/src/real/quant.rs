//! int4 `Q4_0` quantizer for the CAM++ weights and the parity-diagnostic forwards
//! (`forward_staged` / `block1_stages`). Split out of `real.rs` unchanged.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Result, Tensor};

use super::pooling::stats_pool;
use super::{CamPlus, BLOCKS};

/// Smallest weight (in elements) worth quantizing in [`CamPlus::load_quantized`]. Below
/// this the Q4_0 per-block scale overhead dominates; tiny params stay exact f32.
pub(super) const QUANT_MIN_ELEMS: usize = 4096;

/// A `Q4_0`-quantized weight stored for **dequant-on-fetch** (see [`HiftVocoder`]'s twin):
/// the forward is unchanged, only the resident storage is int4; the original logical
/// shape (conv kernels are 3-D/4-D) is restored on lookup.
pub(super) struct QStore {
    pub(super) qt: QTensor,
    pub(super) shape: Vec<usize>,
}

/// Decide whether a weight `vf` is a large block-aligned matrix worth quantizing to
/// `Q4_0` (dequant-on-fetch), returning its [`QStore`] if so, else `None` (keep f32).
///
/// A conv kernel `[out, in, ...]` (or a 2-D linear `[out, in]`) flattens to `[out, rest]`
/// where `rest` is the product of all but the first dim; it is quantized when `rest` is a
/// multiple of the 32-element `Q4_0` block and the tensor has ≥ [`QUANT_MIN_ELEMS`]
/// elements. 1-D weights (biases, BatchNorm stats) and small/odd kernels stay dense.
pub(super) fn maybe_quantize_weight(vf: &Tensor) -> Result<Option<QStore>> {
    let dims = vf.dims().to_vec();
    if dims.len() < 2 || vf.elem_count() < QUANT_MIN_ELEMS {
        return Ok(None);
    }
    let out = dims[0];
    let inner: usize = dims[1..].iter().product();
    if inner % GgmlDType::Q4_0.block_size() != 0 {
        return Ok(None);
    }
    let mat = vf.reshape((out, inner))?;
    let qt = QTensor::quantize(&mat, GgmlDType::Q4_0)?;
    Ok(Some(QStore { qt, shape: dims }))
}

impl CamPlus {
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
