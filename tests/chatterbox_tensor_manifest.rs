//! `syrinx-chatterbox` — the tensor manifest, checked against the **real published
//! checkpoints** with no weights loaded and no GPU.
//!
//! # What the fixtures are, and are not
//!
//! `tests/golden/chatterbox/index/` holds the safetensors **headers** of
//! `t3_turbo_v1.safetensors` (1.9 GB) and `ve.safetensors` (5.7 MB), reduced to
//! `{name: {shape, dtype}}`. A safetensors file begins with a `u64` header length and
//! then that many bytes of JSON index, so a 29 KB and a 1.3 KB HTTP range request
//! produced both — **no weight data was downloaded, and none is committed**.
//!
//! # What is certified here
//!
//! That `syrinx_chatterbox::manifest` names every tensor in those two checkpoints, with
//! exactly the right shape, and claims none that is absent. That is the whole defence
//! against the failure `syrinx-fish`'s s1-mini loader shipped: expected names and shapes
//! that had never met a real checkpoint, surfacing as a shape panic after a 9 GB
//! download.
//!
//! # What is NOT certified
//!
//! Any numeric content, and any *meaning*. Nothing here has been loaded or multiplied.
//! The manifest is a statement of what the config implies and the headers confirm; that
//! a `c_attn` really fuses q, k and v in that order, or that `cond_enc.spkr_enc` really
//! consumes the voice encoder's output, is Phase 1 and needs weights, a GPU and a
//! reference dump. `s3gen*.safetensors` is out of scope entirely — nothing in
//! `t3_turbo_v1.yaml` describes it, so there is nothing to derive.

use std::collections::{BTreeMap, BTreeSet};

use syrinx_chatterbox::config::T3Config;
use syrinx_chatterbox::manifest::{
    expected_t3_tensors, expected_ve_tensors, BACKBONE_PREFIX, PER_BLOCK_TENSORS, VE_LSTM_HIDDEN,
    VE_LSTM_LAYERS, VE_MEL_BINS,
};
use syrinx_chatterbox::Expected;

const YAML: &str = include_str!("golden/chatterbox/config/t3_turbo_v1.yaml");
const T3_INDEX: &str = include_str!("golden/chatterbox/index/t3_turbo_v1.json");
const VE_INDEX: &str = include_str!("golden/chatterbox/index/ve.json");

fn cfg() -> T3Config {
    T3Config::from_yaml(YAML).expect("the shipped t3_turbo_v1.yaml must parse")
}

/// `name -> (shape, dtype)` as the published safetensors header declares it.
fn entries(index: &str, what: &str) -> BTreeMap<String, (Vec<usize>, String)> {
    let v: serde_json::Value =
        serde_json::from_str(index).unwrap_or_else(|e| panic!("{what}: {e}"));
    v.as_object()
        .unwrap_or_else(|| panic!("{what}: index is not an object"))
        .iter()
        .map(|(k, e)| {
            let shape: Vec<usize> = e["shape"]
                .as_array()
                .unwrap_or_else(|| panic!("{what}: {k}: no shape"))
                .iter()
                .map(|d| d.as_u64().expect("shape dim") as usize)
                .collect();
            (k.clone(), (shape, e["dtype"].as_str().expect("dtype").to_string()))
        })
        .collect()
}

fn shapes(index: &str, what: &str) -> BTreeMap<String, Vec<usize>> {
    entries(index, what).into_iter().map(|(k, v)| (k, v.0)).collect()
}

/// Every derived name is present with the derived shape, and nothing in the checkpoint is
/// left unexplained. Both directions, or the check is half a check.
fn assert_manifest_is_exact(want: &[Expected], have: &BTreeMap<String, Vec<usize>>, what: &str) {
    let missing: Vec<&str> = want
        .iter()
        .filter(|x| !have.contains_key(&x.name))
        .map(|x| x.name.as_str())
        .collect();
    assert!(missing.is_empty(), "{what}: {} tensors missing: {missing:?}", missing.len());

    let mismatched: Vec<String> = want
        .iter()
        .filter_map(|x| {
            let got = have.get(&x.name)?;
            (*got != x.shape).then(|| format!("{}: want {:?} got {got:?}", x.name, x.shape))
        })
        .collect();
    assert!(mismatched.is_empty(), "{what}: shape mismatches: {mismatched:?}");

    let claimed: BTreeSet<&str> = want.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(claimed.len(), want.len(), "{what}: duplicate names in the manifest");

    let unaccounted: Vec<&String> =
        have.keys().filter(|k| !claimed.contains(k.as_str())).collect();
    assert!(unaccounted.is_empty(), "{what}: tensors nothing accounts for: {unaccounted:?}");

    assert_eq!(want.len(), have.len(), "{what}: manifest size vs checkpoint size");
}

// ============================================================================ T3

/// The load-bearing test: the T3 manifest matches `t3_turbo_v1.safetensors` exactly.
#[test]
fn the_t3_manifest_matches_the_published_checkpoint_exactly() {
    let have = shapes(T3_INDEX, "t3");
    let want = expected_t3_tensors(&cfg());
    assert_manifest_is_exact(&want, &have, "t3");

    // Non-trivial by construction: 24 blocks x 12 tensors, plus the 11 around them.
    assert_eq!(have.len(), 299);
    assert_eq!(want.len(), 24 * PER_BLOCK_TENSORS + 11);
}

/// **The backbone is a 24-block GPT-2, and the checkpoint is what says so.** The YAML
/// declares 30 layers and a Llama; block 23 exists and block 24 does not, which is the
/// off-by-six that a config-trusting loader would have shipped.
#[test]
fn the_checkpoint_has_exactly_the_blocks_the_preset_predicts_and_not_the_declared_ones() {
    let c = cfg();
    let have = shapes(T3_INDEX, "t3");

    assert_eq!(c.backbone.n_layers, 24);
    assert_eq!(c.declared_transformer_layers, 30);

    let last = c.backbone.n_layers - 1;
    assert!(
        have.contains_key(&format!("tfmr.h.{last}.attn.c_attn.weight")),
        "block {last} must exist"
    );
    assert!(
        !have.contains_key(&format!("tfmr.h.{}.attn.c_attn.weight", c.backbone.n_layers)),
        "there must be no block {}",
        c.backbone.n_layers
    );
    assert!(
        !have.contains_key(&format!(
            "tfmr.h.{}.attn.c_attn.weight",
            c.declared_transformer_layers - 1
        )),
        "the declared 30-layer geometry must NOT be present"
    );

    // Nothing Llama-shaped is in the file: no RMSNorm-only blocks, no gate/up/down MLP,
    // no separate q/k/v. The names are GPT-2's.
    for llama in ["tfmr.layers.0.self_attn.q_proj.weight", "tfmr.h.0.mlp.gate_proj.weight"] {
        assert!(!have.contains_key(llama), "{llama} must not exist — this is not a Llama");
    }
    for gpt2 in ["tfmr.h.0.attn.c_attn.weight", "tfmr.h.0.ln_1.bias", "tfmr.h.0.mlp.c_fc.bias"] {
        assert!(have.contains_key(gpt2), "{gpt2} must exist — this is a GPT-2");
    }
}

/// GPT-2 `Conv1D` stores its weight `[in, out]`, the transpose of `nn.Linear`. A port
/// that misses this loads a silently transposed model. Pinned against the real header,
/// and pinned as an asymmetry: `c_fc` is `[c, ffn]` and `c_proj` is `[ffn, c]`, so the
/// two cannot both be right under the other convention.
#[test]
fn the_conv1d_weights_are_stored_in_by_out() {
    let c = cfg();
    let have = shapes(T3_INDEX, "t3");
    let (w, ffn) = (c.backbone.n_channels, c.backbone.ffn_dim);

    assert_eq!(have.get("tfmr.h.0.attn.c_attn.weight"), Some(&vec![w, 3 * w]));
    assert_eq!(have.get("tfmr.h.0.attn.c_attn.bias"), Some(&vec![3 * w]));
    assert_eq!(have.get("tfmr.h.0.mlp.c_fc.weight"), Some(&vec![w, ffn]));
    assert_eq!(have.get("tfmr.h.0.mlp.c_proj.weight"), Some(&vec![ffn, w]));

    // …whereas `cond_enc.spkr_enc` is a real `nn.Linear` and IS `[out, in]`. The two
    // conventions coexist in one file, which is exactly why this is worth a test.
    assert_eq!(
        have.get("cond_enc.spkr_enc.weight"),
        Some(&vec![w, c.speaker.embed_size])
    );
    assert_ne!(
        have.get("cond_enc.spkr_enc.weight"),
        Some(&vec![c.speaker.embed_size, w]),
        "if this ever passes, the speaker projection changed convention"
    );
}

/// The three config-derived table heights, each traced to the field it comes from.
/// Changing the field must move the manifest — otherwise the derivation is decorative.
#[test]
fn the_table_heights_track_the_config_fields_they_come_from() {
    let c = cfg();
    let have = shapes(T3_INDEX, "t3");
    let w = c.backbone.n_channels;

    // text_tokens_dict_size -> three tables
    for name in ["tfmr.wte.weight", "text_emb.weight", "text_head.weight"] {
        assert_eq!(have.get(name), Some(&vec![c.text.dict_size, w]), "{name}");
    }
    // max_total_tokens -> the learned position table, which is what
    // `input_pos_emb: handled_internally_by_backbone` means
    assert_eq!(have.get("tfmr.wpe.weight"), Some(&vec![c.max_total_tokens, w]));
    assert!(!have.contains_key("text_pos_emb.emb.weight"), "no separate text positions");
    assert!(!have.contains_key("speech_pos_emb.emb.weight"), "no separate speech positions");
    // speech_tokens_dict_size -> two tables and one bias
    assert_eq!(have.get("speech_emb.weight"), Some(&vec![c.speech.dict_size, w]));
    assert_eq!(have.get("speech_head.weight"), Some(&vec![c.speech.dict_size, w]));
    assert_eq!(have.get("speech_head.bias"), Some(&vec![c.speech.dict_size]));
    // …and the text head has NO bias, unlike the speech head. Asymmetric, so a loader
    // cannot treat the two heads alike.
    assert!(!have.contains_key("text_head.bias"));

    // Move each field and the manifest moves with it.
    let moved = |from: &str, to: &str| {
        assert!(YAML.contains(from), "fixture does not contain {from:?}");
        let y = YAML.replacen(from, to, 1);
        let m: BTreeMap<String, Vec<usize>> = expected_t3_tensors(&T3Config::from_yaml(&y).unwrap())
            .into_iter()
            .map(|x| (x.name, x.shape))
            .collect();
        m
    };
    let m = moved("max_total_tokens: 8196", "max_total_tokens: 4096");
    assert_eq!(m.get("tfmr.wpe.weight"), Some(&vec![4096, w]));
    let m = moved("speaker_embed_size: 256", "speaker_embed_size: 192");
    assert_eq!(m.get("cond_enc.spkr_enc.weight"), Some(&vec![w, 192]));
    assert_eq!(m.get("cond_enc.spkr_enc.bias"), Some(&vec![w]), "the bias is the width, not the embed");
}

/// The backbone/head split is total and unambiguous over the real names: everything
/// under `tfmr.` is the stock GPT-2, everything else is T3's own wrapping.
#[test]
fn the_backbone_prefix_splits_the_manifest_cleanly() {
    assert_eq!(BACKBONE_PREFIX, "tfmr.");
    let want = expected_t3_tensors(&cfg());
    let (backbone, wrapper): (Vec<&Expected>, Vec<&Expected>) =
        want.iter().partition(|x| x.name.starts_with(BACKBONE_PREFIX));

    // 24 blocks x 12, plus ln_f (2), wte and wpe.
    assert_eq!(backbone.len(), 24 * PER_BLOCK_TENSORS + 4);
    // text_emb, text_head, speech_emb, speech_head.weight, speech_head.bias, and the two
    // speaker-conditioning tensors.
    assert_eq!(wrapper.len(), 7);
    assert_eq!(backbone.len() + wrapper.len(), want.len(), "the split loses nothing");
    assert!(wrapper.iter().all(|x| !x.name.starts_with(BACKBONE_PREFIX)));
}

/// Every T3 tensor ships as F32 — no BF16 anywhere, unlike the Qwen checkpoints. 1.9 GB
/// for 479M parameters is the arithmetic that confirms it, and a port that assumed BF16
/// would halve its buffer.
#[test]
fn the_published_t3_weights_are_all_f32() {
    let e = entries(T3_INDEX, "t3");
    assert!(!e.is_empty());
    let odd: Vec<&String> = e.iter().filter(|(_, (_, d))| d != "F32").map(|(k, _)| k).collect();
    assert!(odd.is_empty(), "non-F32 tensors: {odd:?}");
}

// ============================================================================ VE

/// The voice-encoder manifest matches `ve.safetensors` exactly.
#[test]
fn the_ve_manifest_matches_the_published_checkpoint_exactly() {
    let have = shapes(VE_INDEX, "ve");
    let want = expected_ve_tensors(&cfg());
    assert_manifest_is_exact(&want, &have, "ve");
    assert_eq!(have.len(), 16);
    assert_eq!(want.len(), VE_LSTM_LAYERS * 4 + 4);
}

/// **`ve_hidden_size: 768` in the YAML is dead.** The checkpoint's LSTM is 256 wide, and
/// the manifest must take that from the checkpoint-read constant, not the file. Both
/// sides: the real hidden size is asserted present, the declared one asserted absent.
#[test]
fn the_declared_ve_hidden_size_is_not_what_the_checkpoint_ships() {
    assert!(YAML.contains("ve_hidden_size: 768"), "the file really does say 768");
    assert_eq!(VE_LSTM_HIDDEN, 256);

    let have = shapes(VE_INDEX, "ve");
    assert_eq!(have.get("lstm.weight_hh_l0"), Some(&vec![4 * VE_LSTM_HIDDEN, VE_LSTM_HIDDEN]));
    assert_ne!(
        have.get("lstm.weight_hh_l0"),
        Some(&vec![4 * 768, 768]),
        "a 768-wide LSTM is not what ships"
    );
}

/// The first LSTM layer consumes mel bins and every later one consumes hidden state —
/// both sides of that branch, against the real header.
#[test]
fn the_first_lstm_layer_takes_mel_bins_and_the_rest_take_hidden_state() {
    let have = shapes(VE_INDEX, "ve");
    let gates = 4 * VE_LSTM_HIDDEN;

    assert_eq!(VE_MEL_BINS, 40);
    assert_ne!(VE_MEL_BINS, VE_LSTM_HIDDEN, "the branch would be unobservable otherwise");
    assert_eq!(have.get("lstm.weight_ih_l0"), Some(&vec![gates, VE_MEL_BINS]));
    for l in 1..VE_LSTM_LAYERS {
        assert_eq!(
            have.get(&format!("lstm.weight_ih_l{l}")),
            Some(&vec![gates, VE_LSTM_HIDDEN]),
            "layer {l} consumes hidden state"
        );
    }
    assert!(
        !have.contains_key(&format!("lstm.weight_ih_l{VE_LSTM_LAYERS}")),
        "there are exactly {VE_LSTM_LAYERS} layers"
    );
}

/// The one config-derived shape in the VE manifest is the projection, and it tracks
/// `speaker_embed_size` — which is also the width `cond_enc.spkr_enc` consumes, so the
/// two checkpoints agree on the handoff.
#[test]
fn the_voice_encoder_projection_meets_the_t3_speaker_input() {
    let c = cfg();
    let ve = shapes(VE_INDEX, "ve");
    let t3 = shapes(T3_INDEX, "t3");

    assert_eq!(ve.get("proj.weight"), Some(&vec![c.speaker.embed_size, VE_LSTM_HIDDEN]));
    assert_eq!(ve.get("proj.bias"), Some(&vec![c.speaker.embed_size]));
    // ve out width == t3 spkr_enc in width. This is the join between the two files.
    let spkr = t3.get("cond_enc.spkr_enc.weight").expect("cond_enc.spkr_enc.weight");
    assert_eq!(spkr[1], ve.get("proj.weight").unwrap()[0], "ve output feeds spkr_enc input");

    let y = YAML.replacen("speaker_embed_size: 256", "speaker_embed_size: 192", 1);
    let moved: BTreeMap<String, Vec<usize>> =
        expected_ve_tensors(&T3Config::from_yaml(&y).unwrap())
            .into_iter()
            .map(|x| (x.name, x.shape))
            .collect();
    assert_eq!(moved.get("proj.weight"), Some(&vec![192, VE_LSTM_HIDDEN]));
    assert_eq!(moved.get("lstm.weight_hh_l0"), Some(&vec![4 * VE_LSTM_HIDDEN, VE_LSTM_HIDDEN]),
        "the LSTM is checkpoint-read and must NOT move with the config");
}
