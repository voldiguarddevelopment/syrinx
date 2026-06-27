//! Pooling helpers for the CAM++ encoder: CAM segment pooling (inside `cam_layer`) and
//! the final statistics pooling. Split out of `real.rs` unchanged.

use candle_core::{Result, Tensor, D};

/// Segment average pooling over time with window `seg` (kernel=stride=seg, ceil mode),
/// then broadcast each segment's mean back across its frames -> `[B, C, T]`.
///
/// ONNX path: Pad(0) -> AveragePool(kernel=seg, stride=seg, ceil_mode=1) -> the pooled
/// `[B,C,nseg]` is expanded back to T by `nearest`-style repeat (each output frame t
/// maps to segment `t // seg`). The pad is zero-width here (Constant pad value list is
/// all zeros), so AveragePool with ceil mode over T frames yields
/// `nseg = ceil(T/seg)` segments; the last (partial) segment averages only its real
/// frames (count_include_pad has no effect since pad width is 0).
pub(super) fn seg_pool_broadcast(x: &Tensor, seg: usize) -> Result<Tensor> {
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
pub(super) fn stats_pool(x: &Tensor) -> Result<Tensor> {
    let t = x.dim(D::Minus1)? as f64;
    let mean = x.mean_keepdim(D::Minus1)?; // [B,C,1]
    let centered = x.broadcast_sub(&mean)?;
    let popvar = centered.sqr()?.mean(D::Minus1)?; // [B,C]
    // unbiased: popvar * T/(T-1). For T==1 the unbiased correction is undefined
    // (T-1==0 → +inf → 0*inf == NaN x-vector); a single frame has no variance, so scale
    // to 0 → std 0. Byte-identical for any real (T>1) reference clip.
    let scale = if t > 1.0 { t / (t - 1.0) } else { 0.0 };
    let var = (popvar * scale)?;
    // numerical guard: clamp tiny negatives from fp rounding before sqrt
    let var = var.relu()?;
    let std = var.sqrt()?;
    let mean = mean.squeeze(D::Minus1)?; // [B,C]
    Tensor::cat(&[&mean, &std], 1)
}
