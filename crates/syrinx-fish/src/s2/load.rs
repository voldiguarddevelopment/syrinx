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
//!     `generator.` prefix if present, folds weight-norm, and casts to f32.
//!
//! PARITY: the exact published key layout is unconfirmed offline. The remap below handles
//! BOTH (a) HF-Qwen3 names (`model.layers.N.self_attn.q_proj`, `model.embed_tokens`, …)
//! and (b) fish-native names (`layers.N.attention.wqkv`, `embeddings`, …) by passing the
//! latter through unchanged. Confirm which the real `s2-pro` checkpoint uses on-box, and
//! confirm the audio-decoder / codebook-embedding key prefixes (the least-certain remap).

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::nn::Weights;

/// Load the sharded Qwen3 LM checkpoint from `dir` into a [`Weights`] bag (f32).
pub fn load_lm(dir: &Path, dev: Device) -> Result<Weights> {
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
            map.insert(key, v.to_dtype(DType::F32)?);
        }
    }
    fuse_qkv(&mut map)?;
    fold_weight_norm(&mut map)?;
    Ok(Weights { map, dev })
}

/// Load the `codec.pth` EVA-GAN/DAC checkpoint into a [`Weights`] bag (f32).
pub fn load_codec(path: &str, dev: Device) -> Result<Weights> {
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
        // Move to the target device + normalise to f32 (the parity build).
        let v = v.to_device(&dev)?.to_dtype(DType::F32)?;
        map.insert(key, v);
    }
    fold_weight_norm(&mut map)?;
    Ok(Weights { map, dev })
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

/// Remap a Qwen3 / `fish_qwen3_omni` weight key to this backend's fish-native module
/// name. Returns `None` to drop a key. Fish-native keys are passed through unchanged.
//
// PARITY: this remap is the single most checkpoint-layout-sensitive piece of the s2
// loader. It assumes HF-Qwen3 naming for the slow backbone and a `fast`/`codebook`
// naming for the audio decoder; confirm the real prefixes on-box and extend as needed.
fn remap_qwen3_key(k: &str) -> Option<String> {
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
