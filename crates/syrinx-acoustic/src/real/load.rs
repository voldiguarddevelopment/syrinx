//! fp32 / int4 loading, the realized-footprint accessor, and the predicate that
//! decides which `linear()` weights are quantized. Moved verbatim from `real.rs`.

use super::*;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::safetensors;

impl Flow {
    /// Load the converted fp32 checkpoint (`flow_fp32.safetensors`) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self { w, qmm: HashMap::new(), quant_bytes: 0, dev })
    }

    /// Load the same `flow_fp32.safetensors`, but quantize every plain-`linear()` weight
    /// to **int4** (GGML `Q4_0`) — the README size goal, mirroring [`Qwen2Lm::load_quantized`].
    ///
    /// Quantized (one `QMatMul` each): every weight name [`is_quant_linear_flow`] flags —
    /// the conformer attention/FFN projections, the estimator transformer + gelu-FF, the
    /// time/resnet MLP linears, and the `spk_embed`/`encoder_proj`/subsample projections.
    /// These are all true `x @ Wᵀ` matmuls, so `QMatMul::forward` is the same op with an
    /// int4 weight. `Q4_0` runs 32-wide blocks along the inner (`in_features`) dim; a
    /// weight whose inner dim is not a multiple of 32 is left dense in f16 (recorded).
    ///
    /// Kept dense (f32): conv1d kernels, LayerNorm weights, `pos_bias_u/v`, all biases,
    /// and the `input_embedding` lookup table — none are plain `x @ Wᵀ` matmuls.
    ///
    /// int4 trades accuracy for size; the forward is otherwise byte-identical to
    /// [`Flow::load`]. The on-box SIM-o eval measures the quality cost.
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        let mut qmm = HashMap::new();
        let mut quant_bytes = 0usize;
        for (k, v) in raw {
            let vf = v.to_dtype(DType::F32)?;
            if is_quant_linear_flow(&k) {
                let dims = vf.dims();
                let inner = *dims.last().unwrap_or(&0);
                // Q4_0 needs the inner dim to be a multiple of the 32-element block.
                if dims.len() == 2 && inner % GgmlDType::Q4_0.block_size() == 0 {
                    let qt = QTensor::quantize(&vf, GgmlDType::Q4_0)?;
                    quant_bytes += qt.storage_size_in_bytes();
                    qmm.insert(k, QMatMul::from_qtensor(qt)?);
                    continue;
                }
                // Not block-aligned: keep dense in f16 (none expected for these dims).
                w.insert(k, vf.to_dtype(DType::F16)?);
                continue;
            }
            w.insert(k, vf);
        }
        Ok(Self { w, qmm, quant_bytes, dev })
    }

    /// Realized weight footprint (quantized + dense) of this loaded flow.
    pub fn footprint(&self) -> FlowFootprint {
        let dense_bytes: usize = self
            .w
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        FlowFootprint {
            quant_bytes: self.quant_bytes,
            dense_bytes,
            n_quantized: self.qmm.len(),
        }
    }
}

/// Per-`linear()` weight suffixes that are real `x @ Wᵀ` matmuls in the flow, and so
/// are quantized to int4 in [`Flow::load_quantized`]. Every name here is passed to
/// [`Flow::linear`]; conv kernels, LayerNorm weights, `pos_bias_u/v`, biases and the
/// `input_embedding` lookup are excluded (they are not plain matmuls).
const QUANT_LINEAR_FLOW_SUFFIXES: [&str; 16] = [
    // conformer RelPositionMultiHeadedAttention
    ".linear_q.weight",
    ".linear_k.weight",
    ".linear_v.weight",
    ".linear_pos.weight",
    ".linear_out.weight",
    // conformer PositionwiseFeedForward
    ".w_1.weight",
    ".w_2.weight",
    // estimator diffusers BasicTransformerBlock self-attn + gelu FF
    ".to_q.weight",
    ".to_k.weight",
    ".to_v.weight",
    ".to_out.0.weight",
    ".net.0.proj.weight",
    ".net.2.weight",
    // estimator resnet time-MLP (mlp.1) and time_mlp (linear_1/linear_2)
    ".mlp.1.weight",
    ".linear_1.weight",
    ".linear_2.weight",
];

/// Whether `name` is a plain `linear()` weight the flow quantizes to int4: any suffix
/// in [`QUANT_LINEAR_FLOW_SUFFIXES`], or the two standalone projections
/// (`spk_embed_affine_layer` / `encoder_proj`) and the LinearNoSubsampling `out.0`.
fn is_quant_linear_flow(name: &str) -> bool {
    name == "spk_embed_affine_layer.weight"
        || name == "encoder_proj.weight"
        || name.ends_with(".out.0.weight")
        || QUANT_LINEAR_FLOW_SUFFIXES.iter().any(|s| name.ends_with(s))
}
