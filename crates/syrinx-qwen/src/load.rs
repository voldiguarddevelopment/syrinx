//! Checkpoint loading and, first, **verification**.
//!
//! The s1-mini port in `syrinx-fish` shipped a loader whose expected tensor names and
//! shapes had never been checked against a real checkpoint; the mismatch only surfaced
//! as a runtime shape error after the weights were finally downloaded. So this module
//! leads with [`verify_checkpoint`], which asserts that every tensor the parsed
//! [`Qwen3TtsConfig`] implies is actually present, with the shape the config predicts —
//! a check that is cheap, runs off the safetensors header alone, and turns that whole
//! class of bug into a clear error naming the offending tensor.

use std::collections::BTreeMap;
use std::path::Path;

use crate::config::{Qwen3TtsConfig, TransformerConfig};

/// A tensor the config says must exist, and the shape it must have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expected {
    pub name: String,
    pub shape: Vec<usize>,
}

fn e(name: String, shape: Vec<usize>) -> Expected {
    Expected { name, shape }
}

/// Every per-layer tensor of one Qwen3 decoder stack rooted at `prefix`.
fn decoder_tensors(prefix: &str, c: &TransformerConfig, out: &mut Vec<Expected>) {
    let h = c.hidden_size;
    let q = c.num_attention_heads * c.head_dim;
    let kv = c.num_key_value_heads * c.head_dim;
    for l in 0..c.num_hidden_layers {
        let p = format!("{prefix}.layers.{l}");
        out.push(e(format!("{p}.input_layernorm.weight"), vec![h]));
        out.push(e(format!("{p}.post_attention_layernorm.weight"), vec![h]));
        out.push(e(format!("{p}.self_attn.q_proj.weight"), vec![q, h]));
        out.push(e(format!("{p}.self_attn.k_proj.weight"), vec![kv, h]));
        out.push(e(format!("{p}.self_attn.v_proj.weight"), vec![kv, h]));
        out.push(e(format!("{p}.self_attn.o_proj.weight"), vec![h, q]));
        // Qwen3 normalises q and k per head before RoPE.
        out.push(e(format!("{p}.self_attn.q_norm.weight"), vec![c.head_dim]));
        out.push(e(format!("{p}.self_attn.k_norm.weight"), vec![c.head_dim]));
        out.push(e(format!("{p}.mlp.gate_proj.weight"), vec![c.intermediate_size, h]));
        out.push(e(format!("{p}.mlp.up_proj.weight"), vec![c.intermediate_size, h]));
        out.push(e(format!("{p}.mlp.down_proj.weight"), vec![h, c.intermediate_size]));
    }
    out.push(e(format!("{prefix}.norm.weight"), vec![h]));
}

/// The full tensor manifest implied by `cfg`, excluding the speaker encoder (whose
/// internal shape is not described by `config.json`).
pub fn expected_tensors(cfg: &Qwen3TtsConfig) -> Vec<Expected> {
    let mut v = Vec::new();
    let t = &cfg.talker;

    decoder_tensors("talker.model", t, &mut v);
    // Text enters at `text_embed_dim` and is projected down to the talker width.
    v.push(e(
        "talker.model.text_embedding.weight".into(),
        vec![cfg.text_vocab_size, cfg.text_embed_dim],
    ));
    v.push(e(
        "talker.text_projection.linear_fc1.weight".into(),
        vec![cfg.text_embed_dim, cfg.text_embed_dim],
    ));
    v.push(e("talker.text_projection.linear_fc1.bias".into(), vec![cfg.text_embed_dim]));
    v.push(e(
        "talker.text_projection.linear_fc2.weight".into(),
        vec![t.hidden_size, cfg.text_embed_dim],
    ));
    v.push(e("talker.text_projection.linear_fc2.bias".into(), vec![t.hidden_size]));
    // Codec group 0: embedding in, head out, both over the talker's codec vocabulary.
    v.push(e(
        "talker.model.codec_embedding.weight".into(),
        vec![t.vocab_size, t.hidden_size],
    ));
    v.push(e("talker.codec_head.weight".into(), vec![t.vocab_size, t.hidden_size]));

    let cp = &cfg.code_predictor;
    decoder_tensors("talker.code_predictor.model", cp, &mut v);
    // The talker->predictor width bridge ("small" backbone -> multi-token-prediction
    // head). It exists ONLY when the two stacks differ in width: the 1.7B is talker
    // 2048 / predictor 1024 and ships it; the 0.6B is 1024/1024 and does not. Deriving
    // its presence from the widths — rather than hardcoding it — is what keeps one
    // loader correct for both sizes.
    if t.hidden_size != cp.hidden_size {
        v.push(e(
            "talker.code_predictor.small_to_mtp_projection.weight".into(),
            vec![cp.hidden_size, t.hidden_size],
        ));
        v.push(e(
            "talker.code_predictor.small_to_mtp_projection.bias".into(),
            vec![cp.hidden_size],
        ));
    }
    for i in 0..cfg.code_predictor_heads() {
        // NOTE the width: the predictor's codec_embedding is at the TALKER's hidden
        // size, not its own. It embeds a code into the space the talker's hidden state
        // lives in; only the predictor's layers and `lm_head` run at `cp.hidden_size`.
        // On the 0.6B both are 1024 so the distinction is invisible — the 1.7B (talker
        // 2048, predictor 1024) is what makes it observable, which is why the verifier
        // runs over every published checkpoint and not just the small one.
        v.push(e(
            format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
            vec![cp.vocab_size, t.hidden_size],
        ));
        v.push(e(
            format!("talker.code_predictor.lm_head.{i}.weight"),
            vec![cp.vocab_size, cp.hidden_size],
        ));
    }
    v
}

/// What a verification run found.
#[derive(Debug, Default)]
pub struct VerifyReport {
    pub checked: usize,
    /// Tensors the config requires that the checkpoint does not contain.
    pub missing: Vec<String>,
    /// `(name, expected, actual)` for tensors present with the wrong shape.
    pub mismatched: Vec<(String, Vec<usize>, Vec<usize>)>,
    /// Tensors in the checkpoint that nothing in the config accounts for. Not an error
    /// on its own (the speaker encoder lives here), but a silent dumping ground is how
    /// dead weight goes unnoticed, so they are surfaced.
    pub unaccounted: Vec<String>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.mismatched.is_empty()
    }
}

/// Verify `dir/model.safetensors` against `cfg`, reading only the safetensors header.
#[cfg(feature = "real")]
pub fn verify_checkpoint(dir: impl AsRef<Path>, cfg: &Qwen3TtsConfig) -> std::io::Result<VerifyReport> {
    let path = dir.as_ref().join("model.safetensors");
    let bytes = std::fs::read(&path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
    let have: BTreeMap<String, Vec<usize>> = st
        .tensors()
        .into_iter()
        .map(|(n, t)| (n, t.shape().to_vec()))
        .collect();

    let mut r = VerifyReport::default();
    let want = expected_tensors(cfg);
    for x in &want {
        r.checked += 1;
        match have.get(&x.name) {
            None => r.missing.push(x.name.clone()),
            Some(s) if *s != x.shape => {
                r.mismatched.push((x.name.clone(), x.shape.clone(), s.clone()))
            }
            Some(_) => {}
        }
    }
    let wanted: std::collections::BTreeSet<&str> = want.iter().map(|x| x.name.as_str()).collect();
    r.unaccounted = have
        .keys()
        .filter(|k| !wanted.contains(k.as_str()))
        .cloned()
        .collect();
    Ok(r)
}

// ---------------------------------------------------------------------------------------
// Materialising the checkpoint (appended: see `crate::model` for the consumer).
// ---------------------------------------------------------------------------------------

/// Read `dir/model.safetensors` into a name → tensor map on `dev`, casting each tensor to
/// `dt` **as it is read**.
///
/// Casting per tensor rather than after the whole bag is loaded keeps peak memory at the
/// target footprint instead of holding a full extra copy. The published checkpoints are
/// bf16 (1.83 GB for the 0.6B, 3.86 GB for the 1.7B), so `dt = F32` roughly doubles that —
/// which is fine on CPU (the parity path) and is exactly what must NOT be hardcoded for
/// CUDA. See [`crate::model::Qwen3Tts::load`] for the device-derived choice.
#[cfg(feature = "real")]
pub fn load_tensors(
    dir: impl AsRef<Path>,
    dev: &candle_core::Device,
    dt: candle_core::DType,
) -> candle_core::Result<std::collections::HashMap<String, candle_core::Tensor>> {
    let path = dir.as_ref().join("model.safetensors");
    let raw = candle_core::safetensors::load(&path, dev)?;
    let mut map = std::collections::HashMap::with_capacity(raw.len());
    // Consuming `raw` by value drops each source tensor right after its cast, so the bf16
    // original is freed incrementally instead of both bags being resident at once.
    for (k, v) in raw {
        let cast = v.to_dtype(dt)?;
        map.insert(k, cast);
    }
    Ok(map)
}

/// Prefix that separates the code predictor's tensors from the talker's.
///
/// The two stacks are disjoint by name — `talker.model.*` / `talker.text_projection.*` /
/// `talker.codec_head.*` versus `talker.code_predictor.*` — so the bag can be split rather
/// than duplicated. (Cloning a `Tensor` is an `Arc` bump, not a copy, but splitting keeps
/// each stack's `Weights::g` misses honest: a talker typo cannot silently resolve against a
/// predictor tensor.)
pub const CODE_PREDICTOR_PREFIX: &str = "talker.code_predictor.";

/// Split a loaded bag into `(talker, code_predictor)` by [`CODE_PREDICTOR_PREFIX`].
///
/// The speaker encoder's tensors (`speaker_encoder.*`, present only on `-Base`) belong to
/// neither stack and stay with the talker half, where the speaker module can read them.
#[cfg(feature = "real")]
pub fn split_stacks(
    map: std::collections::HashMap<String, candle_core::Tensor>,
) -> (
    std::collections::HashMap<String, candle_core::Tensor>,
    std::collections::HashMap<String, candle_core::Tensor>,
) {
    let mut talker = std::collections::HashMap::new();
    let mut predictor = std::collections::HashMap::new();
    for (k, v) in map {
        if k.starts_with(CODE_PREDICTOR_PREFIX) {
            predictor.insert(k, v);
        } else {
            talker.insert(k, v);
        }
    }
    (talker, predictor)
}
