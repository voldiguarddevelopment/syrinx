//! The int4 `Q4_0` weight quantizer for the HiFT vocoder (dequant-on-fetch storage).
//! Split out of `real.rs` unchanged.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Result, Tensor};

/// Smallest weight (in elements) worth quantizing in [`super::HiftVocoder::load_quantized`].
/// Below this the Q4_0 per-block scale overhead dominates and the tiny weight is left
/// f32 (keeps small/sensitive params — the head `conv_post`-adjacent tensors — exact).
pub(super) const QUANT_MIN_ELEMS: usize = 4096;

/// A `Q4_0`-quantized weight stored for **dequant-on-fetch**: the forward is unchanged
/// (it asks [`super::HiftVocoder::g`] for an f32 tensor and gets one back), only the
/// *resident* storage is the int4 `QTensor`. The original logical shape (conv kernels are
/// 3-D `[out,in,k]`) is kept so `g` can restore it after dequantizing the 2-D block store.
pub(super) struct QStore {
    pub(super) qt: QTensor,
    pub(super) shape: Vec<usize>,
}

/// Decide whether a weight `vf` is a large block-aligned matrix worth quantizing to
/// `Q4_0` (dequant-on-fetch), returning its [`QStore`] if so, else `None` (keep f32).
///
/// A conv kernel `[out,in,k]` (or a 2-D linear `[out,in]`) flattens to `[out, in*k]`;
/// it is quantized when `in*k` is a multiple of the 32-element `Q4_0` block and the
/// tensor has at least [`QUANT_MIN_ELEMS`] elements. 1-D weights (biases, Snake `alpha`s)
/// and small/odd kernels stay dense.
pub(super) fn maybe_quantize_weight(vf: &Tensor) -> Result<Option<QStore>> {
    let dims = vf.dims().to_vec();
    if dims.len() < 2 || vf.elem_count() < QUANT_MIN_ELEMS {
        return Ok(None);
    }
    let out = dims[0];
    let inner: usize = dims[1..].iter().product(); // in * k (conv) or in (linear)
    if inner % GgmlDType::Q4_0.block_size() != 0 {
        return Ok(None);
    }
    let mat = vf.reshape((out, inner))?; // 2-D block store
    let qt = QTensor::quantize(&mat, GgmlDType::Q4_0)?;
    Ok(Some(QStore { qt, shape: dims }))
}
