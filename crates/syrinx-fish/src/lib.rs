//! # syrinx-fish — pure-Rust / Candle port of Fish Audio's open TTS models
//!
//! This crate ports **Fish Audio**'s open dual-AR text-to-speech stack to pure
//! Rust on [Candle], covering two checkpoints that share one architecture but differ
//! in backbone and codec:
//!
//! * [`FishVariant::S1Mini`] — `openaudio-s1-mini` (0.5B). Slow AR = a Llama-style
//!   `DualARTransformer`; codec = modded-DAC; weights `.pth`; tiktoken tokenizer.
//! * [`FishVariant::S2Pro`] — `s2-pro` (5B). Slow AR = Qwen3-4B (`fish_qwen3_omni`:
//!   QK-RMSNorm, GQA, qkv/o bias); fast AR = a 4-layer ~400M head with a single
//!   shared embedding table (codebook identity carried by RoPE position) + MCF
//!   fusion; codec = a new 446M EVA-GAN / causal-DAC; weights = sharded safetensors
//!   + `codec.pth`; Qwen3 155k BPE tokenizer.
//!
//! ## The shared idea: dual-AR + RVQ
//!
//! BOTH variants are **dual-AR**. A "slow" autoregressive transformer predicts one
//! **semantic** codebook token per ~21.5 Hz frame; a small "fast" autoregressive
//! transformer then expands that frame into the remaining residual RVQ codes
//! (**10 codebooks total** = 1 semantic-derived + 9 residual). The `[10, T]` code
//! matrix is decoded by an RVQ codec to a **44.1 kHz** waveform. Emotion/style tags
//! like `[happy]` or `(whisper)` are **plain text** — there is no special token path
//! for them; they flow through the tokenizer like any other characters.
//!
//! ## Module map
//!
//! * [`common::config`] — [`common::config::FishConfig`], the variant-agnostic config
//!   (dual-AR transformer + fast-AR + codec), with [`common::config::FishConfig::s1_mini`]
//!   / [`common::config::FishConfig::s2_pro`] constructors and a `config.json` loader.
//! * [`common::dualar`] — the [`common::dualar::DualArBackend`] trait (the contract the
//!   s1/s2 backends implement) + the variant-agnostic driver loop
//!   ([`common::dualar::drive`]) that produces the `[10, T]` code matrix.
//! * [`common::codec`] — the [`common::codec::RvqCodec`] trait (codes → waveform, and
//!   reference audio → codes for voice cloning).
//! * [`common::sampling`] — temperature / top-p / top-k / repetition-penalty / RAS
//!   sampling on a seeded `SplitMix64` PRNG (the CV sampler idiom).
//! * [`common::audio`] — 44.1 kHz wav write + band-limited resample helpers.
//! * `s1` / `s2` — the per-variant backend bodies, implemented by the two later waves.
//!
//! ## ⚠ Parity status
//!
//! This is the **foundation wave**: the crate, config, shared traits, dual-AR driver,
//! sampler, audio I/O, and a run-script skeleton. The math here is **real** but
//! **parity-UNVERIFIED** — the build box is offline, so it is checked only by
//! `cargo build`. Every spot whose exact numeric value must be confirmed against the
//! Python reference on hardware is flagged with a `// PARITY:` comment. Nothing here
//! is a stub: where a value is unknown it is a best-effort default flagged for on-box
//! confirmation, never a fake constant standing in for missing work.
//!
//! ## License
//!
//! NON-COMMERCIAL. Fish Audio publishes the `openaudio-s1-mini` / `s2-pro` weights
//! under a CC-BY-NC-SA-style license. This crate is the inference implementation; any
//! use of the published checkpoints inherits that non-commercial restriction.
//!
//! [Candle]: https://github.com/huggingface/candle

use serde::{Deserialize, Serialize};

pub mod common;
pub mod s1;
pub mod s2;

/// The two Fish Audio checkpoints this crate ports. They share the dual-AR + RVQ
/// concept and the `[10, T]`-code / 44.1 kHz-waveform pipeline, and differ only in the
/// slow-AR backbone, the fast-AR head, the codec, and the tokenizer/weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FishVariant {
    /// `openaudio-s1-mini` (0.5B): Llama-style `DualARTransformer` + modded-DAC,
    /// `.pth` weights, tiktoken tokenizer.
    S1Mini,
    /// `s2-pro` (5B): Qwen3-4B slow AR + 4-layer fast AR + 446M causal-DAC codec,
    /// sharded-safetensors + `codec.pth` weights, Qwen3 155k BPE tokenizer.
    S2Pro,
}

impl FishVariant {
    /// The canonical checkpoint directory name (`run-fish.sh <name>` and on-disk
    /// `checkpoints/<name>/`).
    pub fn dir_name(self) -> &'static str {
        match self {
            FishVariant::S1Mini => "openaudio-s1-mini",
            FishVariant::S2Pro => "s2-pro",
        }
    }

    /// Parse the CLI/identifier spelling (`s1-mini` / `s2-pro`).
    pub fn from_id(s: &str) -> Option<Self> {
        match s {
            "s1-mini" | "s1" | "openaudio-s1-mini" => Some(FishVariant::S1Mini),
            "s2-pro" | "s2" => Some(FishVariant::S2Pro),
            _ => None,
        }
    }
}
