//! **Qwen3-TTS** — a pure-Rust / Candle port of Alibaba's Qwen3-TTS family.
//!
//! ## Licensing
//!
//! The published Qwen3-TTS checkpoints are **Apache-2.0**. That is the practical reason
//! this crate exists alongside [`syrinx-fish`](../syrinx_fish/index.html), whose
//! upstream weights are non-commercial: the two ports cover different licence needs, not
//! different quality tiers.
//!
//! ## Architecture
//!
//! Qwen3-TTS is a **dual-AR** stack, the same shape as the Fish s2 backend:
//!
//! | stage | what it does | Fish s2 analogue |
//! |-------|--------------|------------------|
//! | `talker.model` | 28-layer Qwen3 decoder; one semantic code per frame | slow AR |
//! | `talker.code_predictor` | 5-layer decoder; the remaining 15 RVQ groups per frame | fast AR |
//! | `speaker_encoder` | x-vector over a reference clip (the `-Base` clone path) | — |
//! | `Qwen3-TTS-Tokenizer-12Hz` | separate checkpoint: RVQ codec, 16 quantizers | EVA-GAN/DAC codec |
//!
//! Frame rate is **12.5 Hz** (`encode_downsample_rate` 1920 at 24 kHz), against Fish's
//! 21.5 Hz — fewer, wider frames, and a non-DiT decoder.
//!
//! ## What this crate does NOT do
//!
//! Qwen3-TTS has **no inline emotion tags**. There is no `[sad]` / `(whispering)`
//! mechanism to port. Emotional control is a per-utterance natural-language `instruct`
//! string, and only the `CustomVoice` / `VoiceDesign` checkpoints accept one — the
//! `Base` checkpoints clone a voice but take no instruction. A tagged corpus does not
//! transfer to this model unchanged; see [`config::QwenVariant::supports_instruct`].
//!
//! Also note the language set is **10 languages and does not include Polish**
//! (`auto, chinese, english, french, german, italian, japanese, korean, portuguese,
//! russian, spanish`), confirmed from the model's own `get_supported_languages()`.

pub mod config;
#[cfg(feature = "real")]
pub mod nn;
#[cfg(feature = "real")]
pub mod talker;
#[cfg(feature = "real")]
pub mod code_predictor;
#[cfg(feature = "real")]
pub mod codec;
#[cfg(feature = "real")]
pub mod speaker;
#[cfg(feature = "real")]
pub mod tokenizer;
#[cfg(feature = "real")]
pub mod prompt;
pub mod load;
#[cfg(feature = "real")]
pub mod model;
pub mod sampling;

pub use config::{Qwen3TtsConfig, QwenVariant, TransformerConfig};

/// Output sample rate of the 12 Hz tokenizer's decoder (Hz).
pub const SAMPLE_RATE_24K: u32 = 24_000;
/// Samples per code frame (`decode_upsample_rate`): 24000 / 1920 = 12.5 Hz.
pub const FRAME_HOP: usize = 1920;
