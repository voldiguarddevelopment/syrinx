//! s1 weight loading: the LM (`model.safetensors`) and codec (`codec.safetensors`)
//! checkpoints → name-indexed f32 [`Weights`] bags.
//!
//! Fish ships the s1 weights as torch `.pth`. The **on-box prep** converts them to
//! safetensors and folds every weight-norm parametrization into a plain `.weight`
//! (`torch.save` → `safetensors`, with `remove_parametrizations` / a manual
//! `g·v/‖v‖` fold). This loader reads the resulting safetensors and key-remaps the
//! reference module prefixes, mirroring `BaseTransformer.from_pretrained` and the
//! codec's `generator.`-prefix strip in `load_codec_model`.
//!
//! For robustness the loader **also folds weight-norm at load** when the converter
//! left the `weight_g`/`weight_v` (or `parametrizations.weight.original{0,1}`) pairs
//! in place — so either a pre-folded or a raw-parametrized safetensors loads cleanly.
//!
//! PARITY: the exact converted key layout (whether `wqkv` ships fused or as
//! `wq`/`wk`/`wv`, and whether the converter pre-folds weight-norm) is confirmed
//! against the real checkpoint on-box; both forms are handled here.

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

use super::nn::Weights;

/// Load the s1 LM checkpoint into a [`Weights`] bag.
///
/// Key remap (mirrors `from_pretrained`): a leading `model.` prefix is stripped and
/// any `audio_*` weight is dropped (the s1 path never uses the audio projector). The
/// reference `Attention.load_hook` (fuse `wq`/`wk`/`wv` → `wqkv`) is applied, and
/// weight-norm pairs (none expected on the LM) are folded if present.
pub fn load_lm(path: &str, dev: Device) -> Result<Weights> {
    let raw = safetensors::load(path, &dev)?;
    let mut map: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        if k.contains("audio_") {
            continue;
        }
        let key = k.strip_prefix("model.").map(str::to_string).unwrap_or(k);
        map.insert(key, v.to_dtype(DType::F32)?);
    }
    fuse_qkv(&mut map)?;
    fold_weight_norm(&mut map)?;
    Ok(Weights { map, dev })
}

/// Load the s1 modded-DAC codec checkpoint into a [`Weights`] bag.
///
/// Key remap (mirrors `load_codec_model`): if any key carries a `generator.` prefix,
/// keep only those keys and strip the prefix. Weight-norm pairs are folded if the
/// converter left them in place.
pub fn load_codec(path: &str, dev: Device) -> Result<Weights> {
    let raw = safetensors::load(path, &dev)?;
    let has_generator = raw.keys().any(|k| k.contains("generator."));
    let mut map: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let key = if has_generator {
            match k.strip_prefix("generator.") {
                Some(s) => s.to_string(),
                None => continue, // keep only the generator submodule
            }
        } else {
            k
        };
        map.insert(key, v.to_dtype(DType::F32)?);
    }
    fold_weight_norm(&mut map)?;
    Ok(Weights { map, dev })
}

/// Apply the reference `Attention.load_hook`: wherever a checkpoint stores the
/// attention projection split as `<p>.wq.weight` / `<p>.wk.weight` / `<p>.wv.weight`,
/// fuse them into `<p>.wqkv.weight = cat([wq, wk, wv], dim=0)`.
fn fuse_qkv(map: &mut HashMap<String, Tensor>) -> Result<()> {
    let wq_keys: Vec<String> = map
        .keys()
        .filter(|k| k.ends_with(".wq.weight"))
        .cloned()
        .collect();
    for wq_key in wq_keys {
        let base = wq_key.trim_end_matches(".wq.weight").to_string();
        let wk_key = format!("{base}.wk.weight");
        let wv_key = format!("{base}.wv.weight");
        let wqkv_key = format!("{base}.wqkv.weight");
        if map.contains_key(&wqkv_key) {
            continue;
        }
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
    Ok(())
}

/// Fold any weight-norm parametrization (`<p>.weight_g`/`<p>.weight_v`, or the newer
/// `<p>.parametrizations.weight.original0`/`original1`) into a plain `<p>.weight`.
///
/// `weight = v * (g / ‖v‖)` with the norm taken over every dim except dim 0 (the
/// `torch.nn.utils.weight_norm` default `dim=0`).
fn fold_weight_norm(map: &mut HashMap<String, Tensor>) -> Result<()> {
    // Collect (out_weight_key, g_key, v_key) triples for both naming schemes.
    let mut triples: Vec<(String, String, String)> = Vec::new();
    for k in map.keys() {
        if let Some(base) = k.strip_suffix(".weight_g") {
            triples.push((
                format!("{base}.weight"),
                k.clone(),
                format!("{base}.weight_v"),
            ));
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
    let scale = g2.broadcast_div(&norm)?; // [out, 1]
    let w2 = v2.broadcast_mul(&scale)?; // [out, inner]
    w2.reshape(dims)
}
