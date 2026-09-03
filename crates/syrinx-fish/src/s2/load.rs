//! s2 weight loading: the **sharded** Qwen3 LM (`model-0000{1,2}-of-00002.safetensors`
//! + `model.safetensors.index.json`, bf16) and the **`codec.pth`** EVA-GAN/DAC codec.
//!
//! Two sources, two readers:
//!   * the LM ships as bf16 safetensors split across shards. [`load_lm`] reads the
//!     `model.safetensors.index.json` shard map (falling back to globbing
//!     `model-*-of-*.safetensors`, then a single `model.safetensors`), loads every
//!     shard, key-remaps the Qwen3 layout to the fish-native module names this backend's
//!     `nn`/`slow_ar`/`fast_ar` expect, fuses split `wq/wk/wv` → `wqkv`, and casts to f32.
//!   * the codec ships as a torch pickle `codec.pth`. [`load_codec`] reads it via
//!     `candle_core::pickle` (no on-box conversion required), strips a leading
//!     `generator.` prefix if present, folds weight-norm, and casts to the compute
//!     dtype. It takes a [`CodecParts`] selector so the encode-side stack (needed only
//!     once, to turn a reference clip into prompt codes) need not stay resident for the
//!     whole of generation.
//!
//! The remap is keyed to the REAL `s2-pro` layout (dumped on-box, 358 tensors): the slow
//! AR under `text_model.model.*` (qkv pre-fused, tied head) and the fast AR under
//! `audio_decoder.*` (no QK-norm; shared value table + per-codebook MCF table + output
//! head). HF-Qwen3 names and bare fish-native names are still accepted as fallbacks.

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::nn::Weights;

/// Load the sharded Qwen3 LM checkpoint from `dir` into a [`Weights`] bag in `dt`.
///
/// `dt` is the **compute dtype**: `F32` for the CPU parity path (byte-unchanged) or
/// `BF16` for the CUDA-fit path (the 4.4B LM is ~9 GB in bf16 vs ~18 GB in f32, so bf16
/// fits a 12 GB GPU). Each shard tensor is cast to `dt` as it is read, keeping peak load
/// memory at the bf16 footprint rather than a transient f32 blow-up. The published LM
/// carries no weight-norm, so `fold_weight_norm` is a no-op here.
pub fn load_lm(dir: &Path, dev: Device, dt: DType) -> Result<Weights> {
    let shard_files = resolve_shards(dir)?;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for file in &shard_files {
        let raw = safetensors::load(file, &dev)?;
        for (k, v) in raw {
            if k.contains("audio_tower") || k.contains("visual") {
                // Drop any multimodal-encoder tensors the TTS path never uses.
                continue;
            }
            let key = match remap_qwen3_key(&k) {
                Some(nk) => nk,
                None => continue,
            };
            map.insert(key, v.to_dtype(dt)?);
        }
    }
    fuse_qkv(&mut map)?;
    fold_weight_norm(&mut map)?;
    Ok(Weights { map, dev, dt, qmap: HashMap::new() })
}

/// Which half of the codec to materialise.
///
/// The `codec.pth` state dict splits cleanly along the direction of travel, and the two
/// halves are close to the same size (bf16: encode-side 417 MB, decode-side 369 MB):
///
///   * **encode-side** — `encoder.*`, `quantizer.downsample.*`, `quantizer.pre_module.*`.
///     Read only by the codec's `encode`, i.e. exactly once per run, to turn a reference
///     clip into cloning codes.
///   * **decode-side** — `decoder.*`, `quantizer.upsample.*`, `quantizer.post_module.*`.
///     Read only by the codec's `decode`.
///   * **shared** — the factorized RVQ (`quantizer.quantizer.*`,
///     `quantizer.semantic_quantizer.*`, 0.6 MB in bf16): `in_proj` + codebook on the
///     encode side, codebook + `out_proj` on the decode side. Kept in BOTH bags.
///
/// Anything unrecognised falls into both bags, so an unexpected key can never go missing.
///
/// Keeping the encode side out of the resident set for the whole of generation is worth
/// ~417 MB on a 12 GB card — roughly 40% of the working headroom left over after the
/// 9.12 GB bf16 LM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CodecParts {
    /// Decode + shared only (drops `encoder.*` / `downsample` / `pre_module`).
    Decode,
    /// Encode + shared only (drops `decoder.*` / `upsample` / `post_module`).
    Encode,
}

/// Top-level prefixes read **only** by the codec's `encode` path.
const ENCODE_ONLY: [&str; 3] = ["encoder.", "quantizer.downsample.", "quantizer.pre_module."];
/// Top-level prefixes read **only** by the codec's `decode` path.
const DECODE_ONLY: [&str; 3] = ["decoder.", "quantizer.upsample.", "quantizer.post_module."];

impl CodecParts {
    /// Whether a (prefix-stripped) state-dict key belongs in this bag.
    fn wants(self, key: &str) -> bool {
        match self {
            CodecParts::Decode => !ENCODE_ONLY.iter().any(|p| key.starts_with(p)),
            CodecParts::Encode => !DECODE_ONLY.iter().any(|p| key.starts_with(p)),
        }
    }
}

/// Whether a state-dict key is a weight-norm *component* (`g`/`v`) that
/// [`fold_weight_norm`] will consume. Those must be materialised in f32 — folding in
/// bf16 loses precision on the ‖v‖ reduction. Everything else can be cast straight to
/// the compute dtype at read time.
fn is_weight_norm_part(key: &str) -> bool {
    key.ends_with(".weight_g")
        || key.ends_with(".weight_v")
        || key.ends_with(".parametrizations.weight.original0")
        || key.ends_with(".parametrizations.weight.original1")
}

/// Load the `codec.pth` EVA-GAN/DAC checkpoint into a [`Weights`] bag in `dt`, keeping
/// only the tensors `parts` asks for.
///
/// Weight-norm is **folded in f32** (folding in bf16 loses precision on the ‖v‖
/// reduction), so only the `g`/`v` components are materialised in f32; every other tensor
/// is cast straight to `dt` as it is read. That is byte-identical to the previous
/// "everything to f32, fold, then cast the whole bag" order — `f32 → f32 → bf16` and
/// `f32 → bf16` are the same single rounding, and the checkpoint's handful of bf16
/// buffers round-trip through f32 exactly — but it drops the load-time transient from
/// 1572 MB to ~939 MB for the full bag (and less again for one half). For `dt == F32`
/// every cast is an identity, leaving the CPU parity path byte-unchanged.
///
/// NOTE: candle's pickle reader has no `Bool` dtype, so it *skips* (with a
/// `skipping: ...` line on stderr) this checkpoint's three `causal_mask` bool buffers —
/// 302 MB on disk, of which `encoder.block.4.block.5.causal_mask` alone is a
/// [16384, 16384] 268 MB triangle. They never reach the device, and `codec::transformer`
/// recomputes the mask with `causal_mask_at`, so nothing is lost.
pub fn load_codec(path: &str, dev: Device, dt: DType, parts: CodecParts) -> Result<Weights> {
    // `candle_core::pickle::read_all` reads a torch `.pth` directly (CPU tensors).
    let tensors = candle_core::pickle::read_all(path)?;
    let has_generator = tensors.iter().any(|(k, _)| k.contains("generator."));
    let mut map: HashMap<String, Tensor> = HashMap::with_capacity(tensors.len());
    for (k, v) in tensors {
        let key = if has_generator {
            match k.strip_prefix("generator.") {
                Some(s) => s.to_string(),
                None => continue,
            }
        } else {
            k
        };
        // Drop the half this bag does not need BEFORE the tensor touches the device.
        if !parts.wants(&key) {
            continue;
        }
        // Weight-norm components stay f32 for the (f32) fold; everything else goes
        // straight to the compute dtype.
        let target = if is_weight_norm_part(&key) {
            DType::F32
        } else {
            dt
        };
        let v = v.to_device(&dev)?.to_dtype(target)?;
        map.insert(key, v);
    }
    fold_weight_norm(&mut map)?;
    // PARITY: fold first (f32), then cast the folded weights to the compute dtype. The
    // rest of the bag is already in `dt`, so the `!= dt` guard makes this a no-op for
    // them — and for the whole bag when `dt == F32` (the CPU parity path).
    for v in map.values_mut() {
        if v.dtype() != dt {
            *v = v.to_dtype(dt)?;
        }
    }
    Ok(Weights { map, dev, dt, qmap: HashMap::new() })
}

/// Resolve the LM shard files: prefer the `model.safetensors.index.json` weight map,
/// then any `model-*-of-*.safetensors`, then a single `model.safetensors`.
fn resolve_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let json = std::fs::read_to_string(&index)
            .map_err(|e| candle_core::Error::Msg(format!("read index json: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| candle_core::Error::Msg(format!("parse index json: {e}")))?;
        let mut files: HashSet<String> = HashSet::new();
        if let Some(wm) = v.get("weight_map").and_then(|m| m.as_object()) {
            for shard in wm.values() {
                if let Some(s) = shard.as_str() {
                    files.insert(s.to_string());
                }
            }
        }
        if !files.is_empty() {
            let mut out: Vec<std::path::PathBuf> =
                files.into_iter().map(|f| dir.join(f)).collect();
            out.sort();
            return Ok(out);
        }
    }
    // Glob fallback: model-00001-of-00002.safetensors, ...
    let mut globbed: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for ent in rd.flatten() {
            let name = ent.file_name().to_string_lossy().to_string();
            if name.starts_with("model-") && name.ends_with(".safetensors") {
                globbed.push(ent.path());
            }
        }
    }
    if !globbed.is_empty() {
        globbed.sort();
        return Ok(globbed);
    }
    // Single-file fallback.
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(candle_core::Error::Msg(format!(
        "no LM safetensors shards found in {}",
        dir.display()
    )))
}

/// Remap a `fish_qwen3_omni` weight key to this backend's fish-native module name.
/// Returns `None` to drop a key. Fish-native keys are passed through unchanged.
///
/// The REAL `s2-pro` checkpoint (358 tensors) uses two top prefixes:
///   * `text_model.model.*` — the Qwen3-4B **slow** AR (36 layers). Its per-layer tails
///     are already fish-native (`attention.wqkv` — pre-FUSED q4096+k1024+v1024,
///     `attention.{q,k}_norm`, `attention_norm`, `ffn_norm`, `feed_forward.{w1,w2,w3}`).
///     The LM head is **tied** (no `lm_head` tensor ships) → the slow head reuses
///     `text_model.model.embeddings.weight`.
///   * `audio_decoder.*` — the 4-layer **fast** AR. Same block tails as the slow AR but
///     WITHOUT `q_norm`/`k_norm`, plus the shared value table (`embeddings`), the
///     per-codebook MCF offset table (`codebook_embeddings`, 10×4096, consumed by the
///     SLOW embed), the final `norm` and the `output` head.
fn remap_qwen3_key(k: &str) -> Option<String> {
    // ---- The REAL s2-pro namespace (text_model.* slow, audio_decoder.* fast) -------
    // Slow backbone: strip `text_model.model.layers.N.` → `layers.N.` (tail unchanged).
    if let Some(rest) = k.strip_prefix("text_model.model.layers.") {
        let (n, tail) = rest.split_once('.')?;
        return Some(format!("layers.{n}.{tail}"));
    }
    match k {
        "text_model.model.embeddings.weight" => return Some("embeddings.weight".to_string()),
        "text_model.model.norm.weight" => return Some("norm.weight".to_string()),
        // The head is tied (no lm_head ships); honour an explicit head if a future
        // export adds one.
        "text_model.lm_head.weight" => return Some("output.weight".to_string()),
        _ => {}
    }
    // Fast AR (audio decoder): strip `audio_decoder.layers.N.` → `fast_layers.N.`.
    if let Some(rest) = k.strip_prefix("audio_decoder.layers.") {
        let (n, tail) = rest.split_once('.')?;
        return Some(format!("fast_layers.{n}.{tail}"));
    }
    match k {
        "audio_decoder.embeddings.weight" => return Some("fast_embeddings.weight".to_string()),
        "audio_decoder.codebook_embeddings.weight" => {
            return Some("codebook_embeddings.weight".to_string())
        }
        "audio_decoder.norm.weight" => return Some("fast_norm.weight".to_string()),
        "audio_decoder.output.weight" => return Some("fast_output.weight".to_string()),
        _ => {}
    }

    // ---- Fish-native passthrough + legacy HF-Qwen3 fallback ------------------------
    // Already fish-native (the fish-speech `DualARTransformer` state dict) → pass through.
    if k.starts_with("layers.")
        || k.starts_with("fast_layers.")
        || k == "embeddings.weight"
        || k == "codebook_embeddings.weight"
        || k == "fast_embeddings.weight"
        || k == "norm.weight"
        || k == "fast_norm.weight"
        || k == "fast_output.weight"
        || k == "output.weight"
        || k.starts_with("fast_project_in.")
    {
        return Some(k.to_string());
    }

    // Top-level HF tensors.
    match k {
        "model.embed_tokens.weight" => return Some("embeddings.weight".to_string()),
        "model.norm.weight" => return Some("norm.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        // The fish audio decoder's shared table + per-codebook table + projection — the
        // names below are best-effort (PARITY) for an HF-style export.
        "codebook_embeddings.weight" => return Some("codebook_embeddings.weight".to_string()),
        _ => {}
    }

    // Per-layer slow backbone: model.layers.{N}.<...>
    if let Some(rest) = k.strip_prefix("model.layers.") {
        let (n, tail) = rest.split_once('.')?;
        let mapped = remap_layer_tail(tail, "layers", n)?;
        return Some(mapped);
    }

    // Audio decoder (fast AR): a handful of plausible HF prefixes → fast_* names.
    for pfx in ["model.audio_decoder.layers.", "audio_decoder.layers.", "fast_transformer.layers."] {
        if let Some(rest) = k.strip_prefix(pfx) {
            let (n, tail) = rest.split_once('.')?;
            return remap_layer_tail(tail, "fast_layers", n);
        }
    }
    for (pfx, dst) in [
        ("model.audio_decoder.embeddings.weight", "fast_embeddings.weight"),
        ("audio_decoder.embeddings.weight", "fast_embeddings.weight"),
        ("model.audio_decoder.norm.weight", "fast_norm.weight"),
        ("audio_decoder.norm.weight", "fast_norm.weight"),
        ("model.audio_decoder.output.weight", "fast_output.weight"),
        ("audio_decoder.output.weight", "fast_output.weight"),
        ("model.codebook_embeddings.weight", "codebook_embeddings.weight"),
        ("model.fast_project_in.weight", "fast_project_in.weight"),
        ("model.fast_project_in.bias", "fast_project_in.bias"),
        ("fast_project_in.weight", "fast_project_in.weight"),
        ("fast_project_in.bias", "fast_project_in.bias"),
    ] {
        if k == pfx {
            return Some(dst.to_string());
        }
    }

    // Unknown key: keep it under its original name (folding/fusing ignore unknowns). A
    // genuinely unused tensor is harmless in the bag.
    Some(k.to_string())
}

/// Remap one per-layer tail (`self_attn.q_proj.weight`, `input_layernorm.weight`, …)
/// onto `<base>.<n>.<fish-name>`.
fn remap_layer_tail(tail: &str, base: &str, n: &str) -> Option<String> {
    let p = format!("{base}.{n}");
    let mapped = match tail {
        "input_layernorm.weight" => format!("{p}.attention_norm.weight"),
        "post_attention_layernorm.weight" => format!("{p}.ffn_norm.weight"),
        "self_attn.q_proj.weight" => format!("{p}.attention.wq.weight"),
        "self_attn.q_proj.bias" => format!("{p}.attention.wq.bias"),
        "self_attn.k_proj.weight" => format!("{p}.attention.wk.weight"),
        "self_attn.k_proj.bias" => format!("{p}.attention.wk.bias"),
        "self_attn.v_proj.weight" => format!("{p}.attention.wv.weight"),
        "self_attn.v_proj.bias" => format!("{p}.attention.wv.bias"),
        "self_attn.o_proj.weight" => format!("{p}.attention.wo.weight"),
        "self_attn.o_proj.bias" => format!("{p}.attention.wo.bias"),
        "self_attn.q_norm.weight" => format!("{p}.attention.q_norm.weight"),
        "self_attn.k_norm.weight" => format!("{p}.attention.k_norm.weight"),
        "mlp.gate_proj.weight" => format!("{p}.feed_forward.w1.weight"),
        "mlp.up_proj.weight" => format!("{p}.feed_forward.w3.weight"),
        "mlp.down_proj.weight" => format!("{p}.feed_forward.w2.weight"),
        _ => return None,
    };
    Some(mapped)
}

/// Apply the reference `Attention.load_hook`: fuse split `<p>.wq/.wk/.wv` (weight and,
/// if present, bias) into `<p>.wqkv`.
fn fuse_qkv(map: &mut HashMap<String, Tensor>) -> Result<()> {
    let wq_keys: Vec<String> = map
        .keys()
        .filter(|k| k.ends_with(".wq.weight"))
        .cloned()
        .collect();
    for wq_key in wq_keys {
        let base = wq_key.trim_end_matches(".wq.weight").to_string();
        // weights
        let wk_key = format!("{base}.wk.weight");
        let wv_key = format!("{base}.wv.weight");
        let wqkv_key = format!("{base}.wqkv.weight");
        if !map.contains_key(&wqkv_key) {
            if let (Some(wq), Some(wk), Some(wv)) =
                (map.get(&wq_key), map.get(&wk_key), map.get(&wv_key))
            {
                let fused = Tensor::cat(&[wq, wk, wv], 0)?;
                map.insert(wqkv_key, fused);
                map.remove(&wq_key);
                map.remove(&wk_key);
                map.remove(&wv_key);
            }
        }
        // biases (Qwen3-3 omits them, but fuse if a variant ships them).
        let bq = format!("{base}.wq.bias");
        let bk = format!("{base}.wk.bias");
        let bv = format!("{base}.wv.bias");
        let bqkv = format!("{base}.wqkv.bias");
        if !map.contains_key(&bqkv) {
            if let (Some(q), Some(k), Some(v)) = (map.get(&bq), map.get(&bk), map.get(&bv)) {
                let fused = Tensor::cat(&[q, k, v], 0)?;
                map.insert(bqkv, fused);
                map.remove(&bq);
                map.remove(&bk);
                map.remove(&bv);
            }
        }
    }
    Ok(())
}

/// Fold any weight-norm parametrization (`<p>.weight_g`/`<p>.weight_v`, or the newer
/// `<p>.parametrizations.weight.original{0,1}`) into a plain `<p>.weight`.
fn fold_weight_norm(map: &mut HashMap<String, Tensor>) -> Result<()> {
    let mut triples: Vec<(String, String, String)> = Vec::new();
    for k in map.keys() {
        if let Some(base) = k.strip_suffix(".weight_g") {
            triples.push((format!("{base}.weight"), k.clone(), format!("{base}.weight_v")));
        } else if let Some(base) = k.strip_suffix(".parametrizations.weight.original0") {
            triples.push((
                format!("{base}.weight"),
                k.clone(),
                format!("{base}.parametrizations.weight.original1"),
            ));
        }
    }
    for (out_key, g_key, v_key) in triples {
        let (g, v) = match (map.get(&g_key), map.get(&v_key)) {
            (Some(g), Some(v)) => (g.clone(), v.clone()),
            _ => continue,
        };
        let folded = fold_one(&g, &v)?;
        map.insert(out_key, folded);
        map.remove(&g_key);
        map.remove(&v_key);
    }
    Ok(())
}

/// `weight = v * g / ‖v‖`, norm over all dims except dim 0.
fn fold_one(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dims = v.dims().to_vec();
    let out = dims[0];
    let inner: usize = dims[1..].iter().product();
    let v2 = v.reshape((out, inner))?;
    let norm = v2.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [out, 1]
    let g2 = g.reshape((out, 1))?;
    let scale = g2.broadcast_div(&norm)?;
    let w2 = v2.broadcast_mul(&scale)?;
    w2.reshape(dims)
}

/// Smallest weight (in elements) worth quantizing. Below this the Q4_0 per-block
/// scale overhead dominates and the tensor is left dense.
const QUANT_MIN_ELEMS: usize = 4096;

/// Quantize the big 2-D projections of an already-loaded LM bag to int4 `Q4_0`,
/// moving them from the dense `map` into the `qmap` as `QMatMul`.
///
/// What is quantized: every 2-D `[out, in]` weight whose `in` is a multiple of the
/// 32-element Q4_0 block and which has at least [`QUANT_MIN_ELEMS`] elements — i.e.
/// the attention `wq/wk/wv/wo` and the SwiGLU `w1/w2/w3` of all 36 slow + 4 fast
/// layers. That is where essentially all of the 9.1 GB lives.
///
/// What is deliberately left dense:
/// * **the token-embedding table** — it is read by `index_select` (a row gather),
///   not by a matmul, and candle's `QTensor` cannot be gathered from. It stays
///   dense and is the single largest remaining tensor.
/// * **norm weights and biases** — 1-D, tiny, and numerically sensitive.
///
/// Quantization is per-tensor and lossy; the caller owns deciding that the accuracy
/// cost is acceptable. Nothing here changes the dense path.
pub fn quantize_lm(w: &mut Weights) -> Result<()> {
    // Kernel choice for the quantized matmuls (CUDA only). candle's default is the
    // `*_via_q8_1` family, which re-quantizes the activation into a scratch buffer on
    // every call; measured here, that scratch costs ~1.3 GB of resident VRAM over the
    // dense path. `SYRINX_FISH_DMMV=1` switches to the fused `dequantize_mul_mat_vec`
    // kernel, which reads the int4 blocks straight into the dot product with no
    // activation buffer — trading some speed for memory. Both are numerically valid;
    // which one wins depends on whether the box is VRAM- or latency-bound.
    #[cfg(feature = "cuda")]
    if std::env::var("SYRINX_FISH_DMMV").is_ok() {
        candle_core::quantized::cuda::set_force_dmmv(true);
        eprintln!("syrinx-fish s2: SYRINX_FISH_DMMV=1 -> fused dmmv kernel (lower VRAM)");
    }

    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    let names: Vec<String> = w
        .map
        .keys()
        .filter(|k| k.ends_with(".weight"))
        .cloned()
        .collect();
    let mut n_q = 0usize;
    let mut bytes_before = 0usize;
    let mut q_bytes_total = 0usize;
    for name in names {
        // The embedding table is gathered from, not multiplied by: leave it dense.
        if name.contains("embed") || name.contains("tok_embeddings") {
            continue;
        }
        let t = match w.map.get(&name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let dims = t.dims().to_vec();
        if dims.len() != 2 || t.elem_count() < QUANT_MIN_ELEMS {
            continue;
        }
        if dims[1] % GgmlDType::Q4_0.block_size() != 0 {
            continue;
        }
        // QTensor::quantize wants f32 input regardless of the stored dtype.
        let f32t = t.to_dtype(DType::F32)?;
        let qt = QTensor::quantize(&f32t, GgmlDType::Q4_0)?;
        q_bytes_total += qt.storage_size_in_bytes();
        w.qmap.insert(name.clone(), QMatMul::from_qtensor(qt)?);
        w.map.remove(&name);
        bytes_before += t.elem_count() * t.dtype().size_in_bytes();
        n_q += 1;
    }
    // Actual resident bytes, not an estimate: sum the int4 block storage and whatever
    // stayed dense. Printed so a VRAM budget can be checked against reality instead of
    // arithmetic (SYRINX_FISH_MEM_REPORT=1).
    if std::env::var("SYRINX_FISH_MEM_REPORT").is_ok() {
        let q_bytes: usize = q_bytes_total;
        let mut dense: Vec<(String, usize)> = w
            .map
            .iter()
            .map(|(k, t)| (k.clone(), t.elem_count() * t.dtype().size_in_bytes()))
            .collect();
        dense.sort_by_key(|(_, b)| std::cmp::Reverse(*b));
        let d_bytes: usize = dense.iter().map(|(_, b)| *b).sum();
        eprintln!(
            "syrinx-fish s2 MEM: int4 {:.0} MB + dense {:.0} MB = {:.0} MB resident (LM only)",
            q_bytes as f64 / 1e6,
            d_bytes as f64 / 1e6,
            (q_bytes + d_bytes) as f64 / 1e6
        );
        for (k, b) in dense.iter().take(8) {
            eprintln!("  dense {:>8.1} MB  {k}", *b as f64 / 1e6);
        }
    }
    eprintln!(
        "syrinx-fish s2: quantized {n_q} projections to Q4_0 ({:.2} GB dense -> ~{:.2} GB int4)",
        bytes_before as f64 / 1e9,
        bytes_before as f64 / 1e9 * 4.5 / 16.0
    );
    Ok(())
}
