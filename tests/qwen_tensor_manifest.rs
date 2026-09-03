//! `syrinx-qwen` — the loader's tensor manifest, checked against the **real published
//! checkpoints** with no weights loaded and no GPU.
//!
//! `syrinx_qwen::load::expected_tensors` derives, from the parsed `config.json` alone,
//! every tensor name and shape the talker + code predictor must contain. That derivation
//! is the whole defence against the failure `crates/syrinx-qwen/src/load.rs` was written
//! to prevent: *"the s1-mini port in `syrinx-fish` shipped a loader whose expected tensor
//! names and shapes had never been checked against a real checkpoint; the mismatch only
//! surfaced as a runtime shape error after the weights were finally downloaded."*
//!
//! **What is certified here:** the manifest matches all five published checkpoints
//! exactly — every derived name exists, every derived shape agrees, and the only tensors
//! the manifest does not account for are the 76 `speaker_encoder.*` of the two `-Base`
//! checkpoints. Plus the two config-derived branches inside the manifest (the width
//! bridge, the per-group head count) on both sides.
//!
//! **What is not:** any numeric content. The fixtures under `tests/golden/qwen/index/`
//! are the safetensors **header** — `{name: {shape, dtype}}` — and carry no weight data,
//! which is exactly why this runs everywhere and must never SKIP. Loading those tensors
//! for real is `syrinx_qwen::load::verify_checkpoint` and needs the 1.8–3.9 GB files; see
//! `docs/backends/QWEN_PORT_STATUS.md`.

use std::collections::{BTreeMap, BTreeSet};

use syrinx_qwen::config::{Qwen3TtsConfig, QwenVariant};
use syrinx_qwen::load::{expected_tensors, Expected, CODE_PREDICTOR_PREFIX};

// ---------------------------------------------------------------------------- fixtures

/// One checkpoint: name, `config.json`, safetensors header index, variant.
struct Ckpt {
    name: &'static str,
    config: &'static str,
    index: &'static str,
    variant: QwenVariant,
}

macro_rules! ckpt {
    ($dir:literal, $variant:expr) => {
        Ckpt {
            name: $dir,
            config: include_str!(concat!("golden/qwen/config/", $dir, ".json")),
            index: include_str!(concat!("golden/qwen/index/", $dir, ".json")),
            variant: $variant,
        }
    };
}

fn checkpoints() -> Vec<Ckpt> {
    vec![
        ckpt!("Qwen3-TTS-12Hz-0.6B-Base", QwenVariant::Base),
        ckpt!("Qwen3-TTS-12Hz-0.6B-CustomVoice", QwenVariant::CustomVoice),
        ckpt!("Qwen3-TTS-12Hz-1.7B-Base", QwenVariant::Base),
        ckpt!("Qwen3-TTS-12Hz-1.7B-CustomVoice", QwenVariant::CustomVoice),
        ckpt!("Qwen3-TTS-12Hz-1.7B-VoiceDesign", QwenVariant::VoiceDesign),
    ]
}

impl Ckpt {
    fn cfg(&self) -> Qwen3TtsConfig {
        Qwen3TtsConfig::from_json(self.config).unwrap_or_else(|e| panic!("{}: {e}", self.name))
    }

    /// `name -> shape` as the published `model.safetensors` header declares it.
    fn shapes(&self) -> BTreeMap<String, Vec<usize>> {
        self.entries()
            .into_iter()
            .map(|(k, v)| (k, v.0))
            .collect()
    }

    /// `name -> (shape, dtype)`.
    fn entries(&self) -> BTreeMap<String, (Vec<usize>, String)> {
        let v: serde_json::Value =
            serde_json::from_str(self.index).unwrap_or_else(|e| panic!("{}: index: {e}", self.name));
        v.as_object()
            .unwrap_or_else(|| panic!("{}: index is not an object", self.name))
            .iter()
            .map(|(k, e)| {
                let shape: Vec<usize> = e["shape"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{}: {k}: no shape", self.name))
                    .iter()
                    .map(|d| d.as_u64().expect("shape dim") as usize)
                    .collect();
                let dtype = e["dtype"].as_str().expect("dtype").to_string();
                (k.clone(), (shape, dtype))
            })
            .collect()
    }
}

// ------------------------------------------------------- the manifest vs the checkpoints

/// The load-bearing test: for every published checkpoint, every tensor the config implies
/// is present with exactly the derived shape.
#[test]
fn the_manifest_matches_every_published_checkpoint_exactly() {
    for c in checkpoints() {
        let cfg = c.cfg();
        let have = c.shapes();
        let want = expected_tensors(&cfg);

        let missing: Vec<&str> = want
            .iter()
            .filter(|x| !have.contains_key(&x.name))
            .map(|x| x.name.as_str())
            .collect();
        assert!(missing.is_empty(), "{}: {} tensors missing: {missing:?}", c.name, missing.len());

        let mismatched: Vec<String> = want
            .iter()
            .filter_map(|x| {
                let got = have.get(&x.name)?;
                (*got != x.shape).then(|| format!("{}: want {:?} got {got:?}", x.name, x.shape))
            })
            .collect();
        assert!(mismatched.is_empty(), "{}: shape mismatches: {mismatched:?}", c.name);

        // The manifest is non-trivial: an empty or tiny one would satisfy the two checks
        // above vacuously. The 0.6B implies 402 tensors and the 1.7B 404 (the two extra
        // are the width bridge's weight + bias).
        let n = if cfg.talker.hidden_size == cfg.code_predictor.hidden_size { 402 } else { 404 };
        assert_eq!(want.len(), n, "{}: manifest size", c.name);
        assert!(
            want.len() as f64 >= 0.8 * have.len() as f64,
            "{}: the manifest accounts for only {} of {} checkpoint tensors",
            c.name,
            want.len(),
            have.len()
        );
    }
}

/// Nothing in the checkpoint is left unexplained except the speaker encoder, which
/// `expected_tensors` documents as out of scope ("whose internal shape is not described by
/// `config.json`"). Both sides: the clone-capable checkpoints have exactly 76 such
/// tensors, the others have none at all.
#[test]
fn the_only_unaccounted_tensors_are_the_speaker_encoder() {
    for c in checkpoints() {
        let cfg = c.cfg();
        let have = c.shapes();
        let want: BTreeSet<&str> = expected_tensors(&cfg).iter().map(|x| leak(&x.name)).collect();

        let unaccounted: Vec<&String> = have.keys().filter(|k| !want.contains(k.as_str())).collect();
        let non_speaker: Vec<&&String> = unaccounted
            .iter()
            .filter(|k| !k.starts_with("speaker_encoder."))
            .collect();
        assert!(
            non_speaker.is_empty(),
            "{}: tensors nothing accounts for: {non_speaker:?}",
            c.name
        );

        if cfg.variant.supports_voice_clone() {
            assert_eq!(
                unaccounted.len(),
                76,
                "{}: the -Base checkpoints carry a 76-tensor ECAPA-TDNN speaker encoder",
                c.name
            );
            assert_eq!(c.variant, QwenVariant::Base, "{}", c.name);
        } else {
            assert!(
                unaccounted.is_empty(),
                "{}: a non-clone checkpoint must carry no speaker encoder, found {}",
                c.name,
                unaccounted.len()
            );
        }
    }
}

/// `expected_tensors` names each tensor once. A duplicated name would make the
/// missing/mismatch counts above lie about coverage.
#[test]
fn the_manifest_names_are_unique() {
    for c in checkpoints() {
        let want = expected_tensors(&c.cfg());
        let unique: BTreeSet<&str> = want.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(unique.len(), want.len(), "{}: duplicate names in the manifest", c.name);
    }
}

// ------------------------------------------------------------- the two derived branches

/// The width bridge exists exactly when the talker and the predictor differ in width —
/// derived from the config, never hardcoded. Both sides, against the real checkpoints:
/// the 0.6B (1024/1024) must NOT expect it and must not ship it; the 1.7B (2048/1024)
/// must expect it and must ship it, at `[predictor, talker]`.
#[test]
fn the_width_bridge_tracks_the_two_stack_widths() {
    const BRIDGE_W: &str = "talker.code_predictor.small_to_mtp_projection.weight";
    const BRIDGE_B: &str = "talker.code_predictor.small_to_mtp_projection.bias";

    for c in checkpoints() {
        let cfg = c.cfg();
        let have = c.shapes();
        let want: BTreeMap<&str, &Vec<usize>> =
            expected_tensors(&cfg).iter().map(|x| (leak(&x.name), leak_shape(&x.shape))).collect();

        let widths_differ = cfg.talker.hidden_size != cfg.code_predictor.hidden_size;
        assert_eq!(
            want.contains_key(BRIDGE_W),
            widths_differ,
            "{}: bridge expectation vs widths {} / {}",
            c.name,
            cfg.talker.hidden_size,
            cfg.code_predictor.hidden_size
        );
        assert_eq!(want.contains_key(BRIDGE_B), widths_differ, "{}: bridge bias", c.name);
        // The checkpoint itself agrees — this is what makes it a fact and not a
        // self-consistent guess.
        assert_eq!(have.contains_key(BRIDGE_W), widths_differ, "{}: checkpoint bridge", c.name);

        if widths_differ {
            assert_eq!(
                *want[BRIDGE_W],
                vec![cfg.code_predictor.hidden_size, cfg.talker.hidden_size],
                "{}: the bridge narrows talker -> predictor, not the reverse",
                c.name
            );
            assert_eq!(*want[BRIDGE_B], vec![cfg.code_predictor.hidden_size], "{}", c.name);
        }
    }
}

/// The predictor's per-group tables: `num_code_groups - 1` of each, `codec_embedding.{i}`
/// at the TALKER's width and `lm_head.{i}` at the predictor's. On the 0.6B those two
/// widths coincide; the 1.7B is what makes the distinction observable, so both are
/// asserted against the real headers.
#[test]
fn the_per_group_tables_use_the_two_different_widths() {
    for c in checkpoints() {
        let cfg = c.cfg();
        let have = c.shapes();
        let n = cfg.code_predictor_heads();
        assert_eq!(n, 15, "{}", c.name);

        for i in 0..n {
            let emb = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
            let head = format!("talker.code_predictor.lm_head.{i}.weight");
            assert_eq!(
                have.get(&emb),
                Some(&vec![cfg.code_predictor.vocab_size, cfg.talker.hidden_size]),
                "{}: {emb} is at the TALKER width",
                c.name
            );
            assert_eq!(
                have.get(&head),
                Some(&vec![cfg.code_predictor.vocab_size, cfg.code_predictor.hidden_size]),
                "{}: {head} is at the PREDICTOR width",
                c.name
            );
        }
        // …and there is no head 15: group 0 is the talker's `codec_head`.
        assert!(
            !have.contains_key(&format!("talker.code_predictor.lm_head.{n}.weight")),
            "{}: the checkpoint carries exactly {n} residual heads",
            c.name
        );
        assert!(
            have.contains_key("talker.codec_head.weight"),
            "{}: group 0 comes from the talker",
            c.name
        );
    }
}

/// Layer coverage: the manifest walks `0..num_hidden_layers` of both stacks and stops
/// there — one layer short or one layer long is the classic off-by-one.
#[test]
fn both_stacks_are_covered_layer_for_layer() {
    for c in checkpoints() {
        let cfg = c.cfg();
        let have = c.shapes();
        for (prefix, tc) in [
            ("talker.model", &cfg.talker),
            ("talker.code_predictor.model", &cfg.code_predictor),
        ] {
            let last = tc.num_hidden_layers - 1;
            assert!(
                have.contains_key(&format!("{prefix}.layers.{last}.self_attn.q_proj.weight")),
                "{}: {prefix} is missing its last layer {last}",
                c.name
            );
            assert!(
                !have.contains_key(&format!(
                    "{prefix}.layers.{}.self_attn.q_proj.weight",
                    tc.num_hidden_layers
                )),
                "{}: {prefix} has MORE layers than num_hidden_layers says",
                c.name
            );
            // Qwen3 normalises q and k per head before RoPE — the tensors SpeechBrain-style
            // ports forget.
            assert_eq!(
                have.get(&format!("{prefix}.layers.0.self_attn.q_norm.weight")),
                Some(&vec![tc.head_dim]),
                "{}: {prefix} q_norm",
                c.name
            );
            assert_eq!(
                have.get(&format!("{prefix}.layers.0.self_attn.k_norm.weight")),
                Some(&vec![tc.head_dim]),
                "{}: {prefix} k_norm",
                c.name
            );
            assert!(have.contains_key(&format!("{prefix}.norm.weight")), "{}: final norm", c.name);
        }
        assert_eq!(cfg.talker.num_hidden_layers, 28, "{}", c.name);
        assert_eq!(cfg.code_predictor.num_hidden_layers, 5, "{}", c.name);
    }
}

// ---------------------------------------------------------------------- the stack split

/// `split_stacks` divides the bag on `CODE_PREDICTOR_PREFIX`. The split must be total and
/// unambiguous over the real names: nothing in the talker half may start with the prefix,
/// and the predictor half must hold every per-layer tensor of the 5-layer stack.
#[test]
fn the_prefix_splits_the_manifest_into_two_disjoint_stacks() {
    assert_eq!(CODE_PREDICTOR_PREFIX, "talker.code_predictor.");
    for c in checkpoints() {
        let cfg = c.cfg();
        let want = expected_tensors(&cfg);
        let (predictor, talker): (Vec<&Expected>, Vec<&Expected>) =
            want.iter().partition(|x| x.name.starts_with(CODE_PREDICTOR_PREFIX));

        assert!(
            talker.iter().all(|x| !x.name.starts_with(CODE_PREDICTOR_PREFIX)),
            "{}: the halves overlap",
            c.name
        );
        assert!(
            talker.iter().all(|x| x.name.starts_with("talker.")),
            "{}: every talker tensor is under `talker.`",
            c.name
        );
        assert_eq!(talker.len() + predictor.len(), want.len(), "{}: the split loses nothing", c.name);

        // 5 layers x 11 per-layer tensors + the final norm + 15 embeddings + 15 heads
        // (+ 2 bridge tensors when the widths differ).
        let bridge = if cfg.talker.hidden_size == cfg.code_predictor.hidden_size { 0 } else { 2 };
        assert_eq!(predictor.len(), 5 * 11 + 1 + 15 + 15 + bridge, "{}: predictor half", c.name);
        // 28 layers x 11 + final norm + text embedding + 4 projection tensors + codec
        // embedding + codec head.
        assert_eq!(talker.len(), 28 * 11 + 1 + 1 + 4 + 1 + 1, "{}: talker half", c.name);

        // The speaker encoder is in neither half of the MANIFEST (it is not derived from
        // the config at all), which is why `split_stacks` files it with the talker bag.
        assert!(
            !want.iter().any(|x| x.name.starts_with("speaker_encoder.")),
            "{}: the manifest must not claim the speaker encoder",
            c.name
        );
    }
}

/// Every talker/predictor tensor in the published checkpoints is BF16 — the fact
/// `Qwen3Tts::load` relies on when it upcasts to F32 on CPU and keeps BF16 on CUDA. A
/// checkpoint that shipped F32 would double the load footprint silently.
#[test]
fn the_published_talker_weights_are_bf16() {
    for c in checkpoints() {
        let entries = c.entries();
        assert!(!entries.is_empty(), "{}: empty index", c.name);
        let odd: Vec<&String> = entries
            .iter()
            .filter(|(_, (_, dt))| dt != "BF16")
            .map(|(k, _)| k)
            .collect();
        assert!(odd.is_empty(), "{}: non-BF16 tensors: {odd:?}", c.name);
    }
}

// ------------------------------------------------------------------------------ helpers

/// The manifest is owned by a temporary in several tests; these keep string/shape
/// references usable inside a `BTreeMap` built from it. Deliberately tiny and only ever
/// called with manifest data (a few hundred entries per checkpoint, five checkpoints).
fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn leak_shape(s: &[usize]) -> &'static Vec<usize> {
    Box::leak(Box::new(s.to_vec()))
}
