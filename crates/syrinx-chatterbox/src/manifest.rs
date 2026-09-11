//! The tensor manifest: every weight name and shape the config implies, so that a
//! failed load is a named diagnostic instead of a shape panic deep in a forward pass.
//!
//! This is the same defence `syrinx-qwen`'s `load::expected_tensors` provides, and it
//! exists for the same recorded reason: the `syrinx-fish` s1-mini port shipped a loader
//! whose expected names and shapes had never been checked against a real checkpoint, and
//! the mismatch only surfaced after the weights were finally downloaded.
//!
//! # How much of this is verified
//!
//! **The names and shapes are checked against the shipped checkpoints' safetensors
//! headers**, committed under `tests/golden/chatterbox/index/` and asserted by
//! `tests/chatterbox_tensor_manifest.rs`. A safetensors header is a JSON index of
//! `{name: {shape, dtype}}` — no weight data — so that check costs nothing and runs on
//! the model-free board.
//!
//! **No numeric content is verified.** Nothing here has been loaded, multiplied or
//! compared against the upstream Python. This is a statement of what the configuration
//! *implies* the files contain, confirmed against what their headers *say* they contain.
//! Whether the tensors mean what a port assumes they mean is Phase 1, and needs weights,
//! a GPU and a reference dump.
//!
//! # GPT-2 Conv1D weight layout
//!
//! The T3 backbone is a HuggingFace `GPT2Model`, whose linear layers are `Conv1D`, not
//! `nn.Linear`: the weight is stored **`[in, out]`**, the transpose of what Candle's
//! `Linear` expects. That is why `attn.c_attn.weight` below is `[c, 3c]` and
//! `mlp.c_proj.weight` is `[ffn, c]`. A port that loads these straight into a `Linear`
//! gets a shape error at best and a silently transposed model at worst.
//!
//! # Out of scope
//!
//! `s3gen.safetensors` / `s3gen_meanflow.safetensors`. Their geometry appears nowhere in
//! `t3_turbo_v1.yaml`, so there is nothing to derive; see [`crate`].

use crate::config::T3Config;

/// A tensor the config says must exist, and the shape it must have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expected {
    pub name: String,
    pub shape: Vec<usize>,
}

fn e(name: String, shape: Vec<usize>) -> Expected {
    Expected { name, shape }
}

/// Prefix of everything belonging to the GPT-2 backbone proper, as opposed to T3's own
/// heads, embeddings and speaker conditioning around it.
pub const BACKBONE_PREFIX: &str = "tfmr.";

/// Tensors per GPT-2 block.
pub const PER_BLOCK_TENSORS: usize = 12;

/// The manifest for `t3_turbo_v1.safetensors`.
pub fn expected_t3_tensors(cfg: &T3Config) -> Vec<Expected> {
    let c = cfg.backbone.n_channels;
    let ffn = cfg.backbone.ffn_dim;
    let mut v = Vec::new();

    for i in 0..cfg.backbone.n_layers {
        let p = format!("tfmr.h.{i}");
        v.push(e(format!("{p}.ln_1.weight"), vec![c]));
        v.push(e(format!("{p}.ln_1.bias"), vec![c]));
        // One fused qkv projection: Conv1D `[in, out]`, out = 3 * width.
        v.push(e(format!("{p}.attn.c_attn.weight"), vec![c, 3 * c]));
        v.push(e(format!("{p}.attn.c_attn.bias"), vec![3 * c]));
        v.push(e(format!("{p}.attn.c_proj.weight"), vec![c, c]));
        v.push(e(format!("{p}.attn.c_proj.bias"), vec![c]));
        v.push(e(format!("{p}.ln_2.weight"), vec![c]));
        v.push(e(format!("{p}.ln_2.bias"), vec![c]));
        v.push(e(format!("{p}.mlp.c_fc.weight"), vec![c, ffn]));
        v.push(e(format!("{p}.mlp.c_fc.bias"), vec![ffn]));
        v.push(e(format!("{p}.mlp.c_proj.weight"), vec![ffn, c]));
        v.push(e(format!("{p}.mlp.c_proj.bias"), vec![c]));
    }
    v.push(e("tfmr.ln_f.weight".into(), vec![c]));
    v.push(e("tfmr.ln_f.bias".into(), vec![c]));
    // The backbone's own token and position tables. `wpe` is why `max_total_tokens` is
    // load-bearing, and its presence is what `input_pos_emb:
    // handled_internally_by_backbone` means: there are no separate text/speech position
    // embeddings.
    v.push(e("tfmr.wte.weight".into(), vec![cfg.text.dict_size, c]));
    v.push(e("tfmr.wpe.weight".into(), vec![cfg.max_total_tokens, c]));

    // T3's own text side. `text_head` has no bias; `speech_head` does.
    v.push(e("text_emb.weight".into(), vec![cfg.text.dict_size, c]));
    v.push(e("text_head.weight".into(), vec![cfg.text.dict_size, c]));
    // T3's own speech side.
    v.push(e("speech_emb.weight".into(), vec![cfg.speech.dict_size, c]));
    v.push(e("speech_head.weight".into(), vec![cfg.speech.dict_size, c]));
    v.push(e("speech_head.bias".into(), vec![cfg.speech.dict_size]));
    // The speaker embedding enters here, and only here: an `nn.Linear` (NOT a Conv1D, so
    // `[out, in]`) from the voice encoder's 256 dims to the backbone width.
    v.push(e(
        "cond_enc.spkr_enc.weight".into(),
        vec![c, cfg.speaker.embed_size],
    ));
    v.push(e("cond_enc.spkr_enc.bias".into(), vec![c]));
    v
}

/// LSTM layers in the voice encoder.
///
/// This and the two constants below are **read off the shipped `ve.safetensors` header**,
/// not derived from `t3_turbo_v1.yaml`. The YAML's `ve_hidden_size: 768` is a dead field
/// — the checkpoint's `lstm.weight_hh_l0` is `[1024, 256]`, i.e. 4 gates x 256, so the
/// hidden size is 256. Only [`T3Config::speaker`]'s `embed_size` is config-derived.
pub const VE_LSTM_LAYERS: usize = 3;
/// Hidden width of each voice-encoder LSTM layer (checkpoint-read; see
/// [`VE_LSTM_LAYERS`]).
pub const VE_LSTM_HIDDEN: usize = 256;
/// Mel filterbank channels the voice encoder consumes (checkpoint-read).
pub const VE_MEL_BINS: usize = 40;
/// An LSTM packs input, forget, cell and output gates into one weight.
const LSTM_GATES: usize = 4;

/// The manifest for `ve.safetensors` — the 3-layer LSTM voice encoder whose 256-dim
/// output is what `cond_enc.spkr_enc` consumes.
///
/// Note that `s3gen.safetensors` carries a *different* `speaker_encoder.*` network for
/// the token-to-mel decoder. These are not the same thing.
pub fn expected_ve_tensors(cfg: &T3Config) -> Vec<Expected> {
    let h = VE_LSTM_HIDDEN;
    let gates = LSTM_GATES * h;
    let mut v = Vec::new();
    for l in 0..VE_LSTM_LAYERS {
        // Layer 0 consumes mel bins; every later layer consumes the previous hidden state.
        let input = if l == 0 { VE_MEL_BINS } else { h };
        v.push(e(format!("lstm.weight_ih_l{l}"), vec![gates, input]));
        v.push(e(format!("lstm.weight_hh_l{l}"), vec![gates, h]));
        v.push(e(format!("lstm.bias_ih_l{l}"), vec![gates]));
        v.push(e(format!("lstm.bias_hh_l{l}"), vec![gates]));
    }
    // The only config-derived shape in this manifest.
    v.push(e("proj.weight".into(), vec![cfg.speaker.embed_size, h]));
    v.push(e("proj.bias".into(), vec![cfg.speaker.embed_size]));
    // GE2E training parameters. Shipped, unused at inference, and named here so the
    // checkpoint has nothing left unaccounted for.
    v.push(e("similarity_weight".into(), vec![1]));
    v.push(e("similarity_bias".into(), vec![1]));
    v
}
