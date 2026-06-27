//! Embedding-table quantization scheme + per-row quantizers, and the realized
//! [`Footprint`] report.
//!
//! Split out verbatim from the original single-file `real` port. The big linear weights
//! quantize to GGML `Q4_0` `QMatMul`s inline in `load::load_quantized`; this module owns
//! the *embedding-table* quantizers ([`quantize_embed`] + the int8/int4 paths that build a
//! [`super::QEmbed`]), the [`EmbedScheme`] selector + [`DEFAULT_EMBED_SCHEME`] default, and
//! the [`Footprint`] accounting type. `quantize_embed` is `pub(super)` (called by the
//! loader); the int8/int4 builders stay private.

use super::QEmbed;
use candle_core::{DType, Result, Tensor, D};

/// The per-row quantization scheme for an embedding table.
///
/// Both are symmetric, per-row, dequant-on-gather quantizers; they differ only in the
/// bit width (and so the storage and the quality cost the on-box SIM-o eval measures).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedScheme {
    /// 8-bit: one byte per weight (`q+128`), `scale = max(|row|)/127`. Half the f16 table.
    Int8,
    /// 4-bit: two weights packed per byte (`q+8` nibbles), `scale = max(|row|)/7`. A
    /// quarter of the f16 table — half of [`EmbedScheme::Int8`] — at a higher quality cost.
    Int4,
}

/// The **default** embedding-table quantization for [`super::Qwen2Lm::load_quantized`].
///
/// int4 is the default (the README 4-bit footprint track — realized ≈388 MB CV2 / ≈488 MB
/// CV3, the early ~270 MB budget under-counted the Qwen2-0.5B body: it halves the int8
/// embed bulk — ~136 → ~68 MB for the 151936×896 token table). Flip this to [`EmbedScheme::Int8`] to
/// trade size back for embedding fidelity; the on-box SIM-o eval measures the difference.
pub const DEFAULT_EMBED_SCHEME: EmbedScheme = EmbedScheme::Int4;

/// Per-row symmetric quantize an `[V, H]` f32 embedding table for dequant-on-gather,
/// at the requested bit width. Each row carries its own scale; an all-zero row's
/// `+1e-12` keeps the `0/0` finite ⇒ it dequantizes back to zeros.
pub(super) fn quantize_embed(table: &Tensor, scheme: EmbedScheme) -> Result<QEmbed> {
    match scheme {
        EmbedScheme::Int8 => quantize_embed_int8(table),
        // int4 needs an even row width to pack two-per-byte; fall back to int8 otherwise
        // (no Qwen2/CosyVoice2 embed table has an odd hidden dim — 896 is even).
        EmbedScheme::Int4 if table.dim(D::Minus1)? % 2 == 0 => quantize_embed_int4(table),
        EmbedScheme::Int4 => quantize_embed_int8(table),
    }
}

/// Per-row symmetric int8-quantize: weights `round(row/scale)` clamped to `[-127,127]`,
/// `scale = max(|row|)/127`, stored `+128` as u8 (`[V, H]`).
fn quantize_embed_int8(table: &Tensor) -> Result<QEmbed> {
    let (_v, h) = table.dims2()?;
    let amax = table.abs()?.max_keepdim(D::Minus1)?; // [V, 1]
    let scale = ((amax / 127.0)? + 1e-12)?; // [V, 1], f32; +eps guards an all-zero row
    let q = table
        .broadcast_div(&scale)?
        .round()?
        .clamp(-127f32, 127f32)?; // [V, H], integer-valued f32 in [-127,127]
    let q = (q + 128.0)?.to_dtype(DType::U8)?; // store offset by +128 (range [1,255])
    let bytes = q.elem_count() + scale.elem_count() * DType::F32.size_in_bytes();
    Ok(QEmbed { scheme: EmbedScheme::Int8, q, scale, h, bytes })
}

/// Per-row symmetric int4-quantize: weights `round(row/scale)` clamped to `[-7,7]`,
/// `scale = max(|row|)/7`, two weights packed per byte as nibbles `q+8` (`[V, H/2]`).
/// `H` must be even (caller guarantees via [`quantize_embed`]).
fn quantize_embed_int4(table: &Tensor) -> Result<QEmbed> {
    let (v, h) = table.dims2()?;
    let amax = table.abs()?.max_keepdim(D::Minus1)?; // [V, 1]
    let scale = ((amax / 7.0)? + 1e-12)?; // [V, 1], f32; +eps guards an all-zero row
    // integer-valued f32 in [-7,7], flattened row-major to pack on the host.
    let qf: Vec<f32> = table
        .broadcast_div(&scale)?
        .round()?
        .clamp(-7f32, 7f32)?
        .flatten_all()?
        .to_vec1()?;
    let hp = h / 2;
    let mut packed = vec![0u8; v * hp];
    for i in 0..v {
        for j in 0..hp {
            let lo = (qf[i * h + 2 * j] as i32 + 8) as u8 & 0x0F; // element 2j -> low nibble
            let hi = (qf[i * h + 2 * j + 1] as i32 + 8) as u8 & 0x0F; // element 2j+1 -> high
            packed[i * hp + j] = lo | (hi << 4);
        }
    }
    let q = Tensor::from_vec(packed, (v, hp), table.device())?;
    let bytes = q.elem_count() + scale.elem_count() * DType::F32.size_in_bytes();
    Ok(QEmbed { scheme: EmbedScheme::Int4, q, scale, h, bytes })
}

/// Realized on-disk-equivalent footprint of a loaded [`super::Qwen2Lm`], split into the
/// quantized (int4) and dense (f16 embed + f32 norm/bias) parts. `total_bytes` is what
/// the model actually occupies for its weights, the headline number for the README's
/// size goal.
#[derive(Debug, Clone, Copy)]
pub struct Footprint {
    /// Bytes held by the `Q4_0` quantized linear weights (0 in the fp32 build).
    pub quant_bytes: usize,
    /// Bytes held by the per-row quantized embedding tables ([`DEFAULT_EMBED_SCHEME`],
    /// int4 by default; 0 in the fp32 build, where the embeds live in `dense_bytes` as f32).
    pub embed_bytes: usize,
    /// Bytes held by the retained dense weights (norms/biases f32, plus the f32 embeds
    /// in the fp32 build).
    pub dense_bytes: usize,
    /// Number of weights that were quantized to int4.
    pub n_quantized: usize,
}

impl Footprint {
    /// Total realized weight bytes (`quant + embed + dense`).
    pub fn total_bytes(&self) -> usize {
        self.quant_bytes + self.embed_bytes + self.dense_bytes
    }
    /// Total realized weight footprint in mebibytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
}
