//! The int4 `Q4_0` weight quantizer for the CV3 HiFT vocoder (dequant-on-fetch storage).
//! Split out of `real_cv3.rs` unchanged.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Result, Tensor};

/// Smallest weight (in elements) worth quantizing in [`super::Cv3Hift::load_quantized`].
/// Below this the Q4_0 per-block scale overhead dominates; tiny params stay exact f32
/// (matches the CV2 [`crate::real::HiftVocoder`] threshold).
pub(super) const QUANT_MIN_ELEMS: usize = 4096;

/// A `Q4_0`-quantized weight stored for **dequant-on-fetch** (the CV3 twin of the CV2
/// [`crate::real::HiftVocoder`]'s `QStore`): the forward is unchanged — it asks
/// [`super::Cv3Hift::weight`] for an f32 tensor and gets one back — only the *resident*
/// storage is the int4 `QTensor`. The original logical conv shape `[out,in,k]` is kept so
/// `weight` can restore it after dequantizing the 2-D block store.
pub(super) struct QStore {
    pub(super) qt: QTensor,
    pub(super) shape: Vec<usize>,
}

/// Decide whether a folded conv kernel `vf` is a large block-aligned matrix worth quantizing
/// to `Q4_0` (dequant-on-fetch), returning its [`QStore`] if so, else `None` (keep dense).
///
/// A conv kernel `[out,in,k]` flattens to `[out, in·k]`; it is quantized when `in·k` is a
/// multiple of the 32-element `Q4_0` block and the kernel has ≥ [`QUANT_MIN_ELEMS`]
/// elements. Small or non-32-aligned kernels stay dense (matches the CV2 twin).
pub(super) fn maybe_quantize_weight(vf: &Tensor) -> Result<Option<QStore>> {
    let dims = vf.dims().to_vec();
    if dims.len() < 2 || vf.elem_count() < QUANT_MIN_ELEMS {
        return Ok(None);
    }
    let out = dims[0];
    let inner: usize = dims[1..].iter().product(); // in * k
    if inner % GgmlDType::Q4_0.block_size() != 0 {
        return Ok(None);
    }
    let mat = vf.reshape((out, inner))?; // 2-D block store
    let qt = QTensor::quantize(&mat, GgmlDType::Q4_0)?;
    Ok(Some(QStore { qt, shape: dims }))
}
