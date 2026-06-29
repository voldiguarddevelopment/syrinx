//! syrinx-stt — pure-Rust / Candle speech-to-text (Whisper).
//!
//! The mirror image of the Syrinx TTS stack: where the other crates turn
//! `text + voice -> audio`, this one turns `audio -> text`. It makes Syrinx
//! bidirectional and — more usefully for the harness — gives the TTS tests a
//! NATIVE intelligibility oracle: synthesize text, transcribe the result, and
//! compare with [`wer`]. That replaces the external `faster-whisper` dependency
//! the eval scripts used to shell out to.
//!
//! The real work is OpenAI Whisper, reused from
//! [`candle_transformers::models::whisper`] (the `model::Whisper`
//! encoder/decoder + the `audio` log-mel front end) — we do NOT reimplement
//! Whisper. The decode loop mirrors the official `candle-examples/whisper`
//! example (SOT / language / transcribe / no-timestamps prompt, greedy decode
//! with a temperature fallback, non-speech-token suppression).
//!
//! Like the TTS crates, the Candle backend is behind the default-on `real`
//! feature; `--no-default-features` yields a Candle-free crate that still
//! exposes the pure-Rust [`wer`] helper.

// The word-error-rate helper is pure Rust (no Candle) — always available so the
// TTS tests can assert intelligibility even in a model-free build.
mod wermetric;
pub use wermetric::wer;

// The real Whisper STT path (feature `real`, default-on).
#[cfg(feature = "real")]
mod stt;
#[cfg(feature = "real")]
pub use stt::{Segment, Stt, SttError, Transcript};
