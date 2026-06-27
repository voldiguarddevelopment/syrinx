//! Checkpoint loading (fp32 parity + int4 quantized) and the dense-weight footprint
//! instrumentation.
//!
//! Split out verbatim from the original single-file `real` port: [`super::Qwen2Lm::load`]
//! / [`super::Qwen2Lm::load_quantized`] (the weight-wiring paths, including the inline
//! `Q4_0` `QMatMul` quantization of the big linears) plus [`super::Qwen2Lm::footprint`] /
//! [`super::Qwen2Lm::dense_breakdown`]. The two private weight-classification predicates
//! (`is_unused_text_head`, `is_embedding_table`) are used only by `load_quantized` and stay
//! local. Embedding-table quantization lives in `super::quant`.

use super::quant::quantize_embed;
use super::{Footprint, Qwen2Lm, DEFAULT_EMBED_SCHEME};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{safetensors, DType, Device, Result};
use std::collections::HashMap;

impl Qwen2Lm {
    /// Load the converted fp32 checkpoint (`llm_fp32.safetensors`) onto `dev`.
    ///
    /// This is the parity build: every weight is normalised to f32 and the forward is a
    /// clean fp32 reference match. Use [`Qwen2Lm::load_quantized`] for the int4 build.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        // Normalise to f32 so the forward is a clean fp32 reference match.
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self {
            w,
            qmm: HashMap::new(),
            qembed: HashMap::new(),
            quant_bytes: 0,
            dev,
        })
    }

    /// Load the same `llm_fp32.safetensors`, but quantize the big linear weights to
    /// **int4** (GGML `Q4_0`) for a ~4× smaller LM footprint (the README size goal).
    ///
    /// Quantized (one `QMatMul` each): every layer's `q/k/v/o_proj` and
    /// `gate/up/down_proj`, plus the `llm_decoder` output head — these are all true
    /// `x @ Wᵀ` matmuls, so `QMatMul::forward` is numerically the same op as the fp32
    /// `broadcast_matmul`, just with an int4 weight. `Q4_0` blocks of 32 run along each
    /// weight's inner (`in_features`) dim; every Qwen2 inner dim here (896 and the MLP
    /// intermediate 4864) is a multiple of 32, so all quantize cleanly. Any weight whose
    /// inner dim is **not** a multiple of 32 is left dense in f16 (recorded, none occur
    /// for these dims).
    ///
    /// Quantized per-row (symmetric, [`super::QEmbed`]) at [`DEFAULT_EMBED_SCHEME`] — **int4** by
    /// default: the embedding tables (`embed_tokens` / `llm_embedding` / `speech_embedding`).
    /// These are `index_select` gathers, not matmuls, so they are *not* `QMatMul`s — the
    /// gathered rows are dequantized on lookup (see [`Qwen2Lm::embed_rows`]). int4 stores
    /// the 151936×896 token table at ~68 MB (two 4-bit weights/byte + a tiny per-row
    /// scale), a quarter of the f16 table; int8 (~136 MB) is available via the scheme const.
    ///
    /// **Dropped:** the untied Qwen2 `lm_head` (text-token output projection) — a full
    /// `[vocab, 896]` f32 matrix ≈ 519.6 MB that the speech path never uses (it decodes
    /// via `llm_decoder`). It is the entire post-int4 "dense remainder"; dropping it is
    /// the single biggest size win.
    ///
    /// Kept dense: the RMSNorm weights (f32, tiny) and the attention q/k/v **biases**
    /// (f32, tiny).
    ///
    /// int4 trades accuracy for size; the forward is otherwise identical, so the
    /// quantized logits track but do not equal the fp32 logits (see the root
    /// `real_lm_quant` test, which measures argmax agreement + the realized footprint).
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        let mut qmm = HashMap::new();
        let mut qembed = HashMap::new();
        let mut quant_bytes = 0usize;
        for (k, v) in raw {
            // DROP the unused Qwen2 text-token output head (`lm_head`). The checkpoint
            // exports it untied — a full `[vocab=151936, 896]` f32 matrix ≈ 519.6 MB,
            // the entire "dense remainder" of the post-int4 footprint. CosyVoice2's
            // speech path produces speech tokens through `llm_decoder` and *never* calls
            // `lm_head` (see `forward_logits`), so it is dead weight here: not loaded,
            // not quantized. This is the single biggest size win. (fp32 `load()` keeps
            // it, so the parity path is byte-unchanged.)
            if is_unused_text_head(&k) {
                continue;
            }
            let vf = v.to_dtype(DType::F32)?;
            // Embedding tables -> per-row dequant-on-gather lookup (int4 by default; see
            // `DEFAULT_EMBED_SCHEME`). These are `index_select` gathers, not matmuls.
            if is_embedding_table(&k) {
                qembed.insert(k, quantize_embed(&vf, DEFAULT_EMBED_SCHEME)?);
                continue;
            }
            // Every remaining 2-D weight is a real `x @ Wᵀ` matmul (the per-layer
            // q/k/v/o + gate/up/down projections and the `llm_decoder` head), so it
            // quantizes to GGML `Q4_0` as a `QMatMul`. A generic 2-D test (rather than a
            // name allowlist) means *no* large dense matmul can be silently left in f32 —
            // any future weight is caught here. `Q4_0` runs 32-wide blocks along the
            // inner (`in_features`) dim; the few weights whose inner dim is not a multiple
            // of 32 stay dense in f16 (none occur for Qwen2: 896 and 4864 are both ×32).
            let dims = vf.dims();
            let inner = *dims.last().unwrap_or(&0);
            if dims.len() == 2 && inner % GgmlDType::Q4_0.block_size() == 0 {
                let qt = QTensor::quantize(&vf, GgmlDType::Q4_0)?;
                quant_bytes += qt.storage_size_in_bytes();
                qmm.insert(k, QMatMul::from_qtensor(qt)?);
                continue;
            }
            if dims.len() == 2 {
                // 2-D but not block-aligned: keep dense in f16 (none occur for Qwen2).
                w.insert(k, vf.to_dtype(DType::F16)?);
                continue;
            }
            // 1-D weights (RMSNorm weights, q/k/v biases): stay f32 (tiny).
            w.insert(k, vf);
        }
        Ok(Self {
            w,
            qmm,
            qembed,
            quant_bytes,
            dev,
        })
    }

    /// Realized weight footprint (quantized + dense bytes) of this loaded model.
    pub fn footprint(&self) -> Footprint {
        let dense_bytes: usize = self
            .w
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        let embed_bytes: usize = self.qembed.values().map(|e| e.bytes).sum();
        Footprint {
            quant_bytes: self.quant_bytes,
            embed_bytes,
            dense_bytes,
            n_quantized: self.qmm.len(),
        }
    }

    /// Per-tensor `(name, bytes)` of every retained **dense** weight, largest first.
    ///
    /// The instrumentation that surfaces exactly what (if anything) is *not* quantized —
    /// the tool used to identify the ~520 MB `lm_head` remainder. In the quantized build
    /// this should be only the tiny RMSNorm weights + attention biases; any large entry
    /// here is an un-quantized matmul that should be investigated.
    pub fn dense_breakdown(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self
            .w
            .iter()
            .map(|(k, t)| (k.clone(), t.elem_count() * t.dtype().size_in_bytes()))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

// --- weight classification for the int4 (Q4_0) quantized build ---------------

/// Whether `name` is the unused Qwen2 text-token output head (`lm_head`).
///
/// CosyVoice2 decodes *speech* tokens through `llm_decoder`; the base Qwen2's `lm_head`
/// (text vocabulary) is never on the speech forward path, yet the checkpoint exports it
/// untied as a full `[vocab, hidden]` matrix — the ~520 MB "dense remainder". The
/// quantized build drops it (see [`Qwen2Lm::load_quantized`]). Matched by substring so
/// it catches whatever module prefix the export uses (`llm.model.lm_head.weight` etc.);
/// nothing on the live forward path contains `lm_head` (it uses `llm_decoder`,
/// `embed_tokens`, `layers`, `norm`, `*_embedding`).
fn is_unused_text_head(name: &str) -> bool {
    name.contains("lm_head")
}

/// Whether `name` is an embedding lookup table (kept as an f16 gather, not a matmul).
fn is_embedding_table(name: &str) -> bool {
    name == "llm.model.model.embed_tokens.weight"
        || name == "llm_embedding.weight"
        || name == "speech_embedding.weight"
}
