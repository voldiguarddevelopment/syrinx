//! **Chatterbox Turbo** (ResembleAI) — Phase 0 of a *candidate* port.
//!
//! ## Status: candidate, not adopted
//!
//! Qwen3-TTS (`syrinx-qwen`) is the TTS path and the fallback. Nothing in this crate
//! adopts Chatterbox, and nothing here renders audio: there is no model code, no Candle,
//! no weight loading and no GPU path. What there is, is the part of a port that is a
//! **pure function of the shipped configuration files** — the geometry, the tokenizer
//! contract, and the tensor manifest those two imply — so that the expensive half can
//! start from a checked foundation instead of a guess.
//!
//! `docs/backends/CHATTERBOX_PORT_SCOPE.md` is the scope document. The one fact that
//! decides whether this is even allowed to ship is the licence: the model repository's
//! card carries `license: mit`.
//!
//! ## Why there is no `real` feature
//!
//! `docs/backends/QWEN_PORT_STATUS.md` §5 gap 5 records a mistake worth not repeating:
//! `syrinx-qwen` put `suppressed_ids`, `PromptConfig`, `MimiEncoderConfig::from_json` and
//! `DecoderConfig::from_json` — all pure functions of a config file — inside modules that
//! import Candle, so they ended up behind the `real` feature gate and could not join the
//! model-free board. They became untestable in the place testing is cheapest.
//!
//! So this crate has **no features at all**. Everything it contains is model-free by
//! construction and runs on the default board (`GROUP_chatterbox`, in `FAMILY_free`).
//! When Phase 1 adds a Candle forward pass it gets a `real` feature and its own modules;
//! [`config`], [`tokenizer`], [`contract`] and [`manifest`] must stay outside it.
//!
//! ## What the shipped files actually say
//!
//! Read from `t3_turbo_v1.yaml` and the tokenizer JSONs, and cross-checked against the
//! safetensors **headers** of `t3_turbo_v1.safetensors` and `ve.safetensors` (the header
//! is a JSON index of names, shapes and dtypes; fetching it costs 29 KB and 1.3 KB, and
//! carries no weight data). Those two indices are committed under
//! `tests/golden/chatterbox/index/` and the manifest is checked against them.
//!
//! **The T3 backbone is GPT-2, not Llama, and it has 24 layers, not 30.** The YAML's
//! `llama_config_name: Llama_520M` and `n_transformer_layers: 30` are both dead fields —
//! `t3_turbo_v1.yaml` is a training-config *superset* covering several models in
//! Resemble's stack, and most of its ~300 keys belong to other ones (`dcnar_*`, `rvc_*`,
//! `taco_*`, `hooli*`, `voc*`). The shipped checkpoint's tensors are
//! `tfmr.h.{0..23}.attn.c_attn.*` — a HuggingFace `GPT2Model` with the Conv1D
//! `[in, out]` weight layout — which is `gpt_transformer_type: gpt2-medium`, the field
//! that IS load-bearing. [`config::T3Config::from_yaml`] therefore takes its layer
//! count, head count, width and FFN width from the *preset*, cross-checks the two
//! declared numbers that agree (`n_transformer_heads`, `n_gpt_channels`), and records
//! the contradicting one as [`config::T3Config::declared_transformer_layers`] so the
//! trap is written down in code rather than in a comment.
//!
//! The load-bearing keys, and nothing else, are listed on [`config::T3Config`].
//!
//! ## Two speaker encoders, not one
//!
//! `ve.safetensors` is a 3-layer LSTM voice encoder (Real-Time-Voice-Cloning lineage,
//! 40 mel bins in, 256-dim embedding out) whose output is what T3 conditions on via
//! `cond_enc.spkr_enc`. `s3gen.safetensors` carries a **second, unrelated**
//! `speaker_encoder.*` (937 tensors, CAMPPlus-shaped) for the token→mel decoder. A port
//! that assumes one speaker path will be wrong.
//!
//! ## What Phase 0 deliberately does not cover
//!
//! `s3gen.safetensors` / `s3gen_meanflow.safetensors` (2489 / 2491 tensors:
//! `speaker_encoder` 937, `flow` 1122, `mel2wav` 328, `tokenizer` 104). Their geometry
//! is not described by `t3_turbo_v1.yaml` at all — it is hard-coded in the upstream
//! Python — so there is nothing to *derive*, and a manifest for them would be a
//! transcription of a header rather than a statement the config implies. Phase 1.

pub mod config;
pub mod contract;
pub mod manifest;
pub mod tokenizer;

pub use config::T3Config;
pub use contract::ModelContract;
pub use manifest::Expected;
pub use tokenizer::{Tag, TokenizerContract};

/// The model repository this crate describes.
pub const REPO: &str = "ResembleAI/chatterbox-turbo";

/// The repository's declared licence (`license: mit` in the model card's front matter).
/// The reason a second backend family is discussable at all — see `docs/LICENSES.md`.
pub const LICENSE: &str = "MIT";
