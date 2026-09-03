//! The Qwen3-TTS 12 Hz tokenizer: a Mimi-family RVQ codec shipped as its own
//! checkpoint (`Qwen3-TTS-Tokenizer-12Hz`, 682 MB), separate from the talker.
//!
//! 24 kHz in and out, `encode_downsample_rate` / `decode_upsample_rate` 1920 — so one
//! code frame is 1920 samples and the frame rate is 12.5 Hz. Decode is
//! `codes -> RVQ -> pre_transformer -> pre_conv -> upsample x4 -> decoder stack x480`.

pub mod rvq;
pub mod encoder;
pub mod decoder;
