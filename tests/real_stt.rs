//! Real Whisper STT test — transcribes a known clip with the pure-Rust
//! `syrinx-stt` engine and (optionally) scores it against a reference with WER.
//! Env-gated on the on-box Whisper model + a test WAV; skips cleanly in CI.
//!
//!   SYRINX_STT_MODEL_DIR=<dir with config.json+tokenizer.json+model.safetensors> \
//!   SYRINX_STT_WAV=<clip.wav> [SYRINX_STT_REF="known transcript"] \
//!   [SYRINX_STT_LANG=en] [SYRINX_STT_WER_MAX=0.5] \
//!   cargo test --features real --release --test real_stt -- --nocapture
//!
//! Download a Whisper model (e.g. base) with:
//!   hf download openai/whisper-base --local-dir <dir>
#![cfg(feature = "real")]

use std::path::Path;

use syrinx_serve::wavio;
use syrinx_stt::{wer, Stt};

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

/// Pure-Rust WER sanity — always runs (no model needed). Guards the oracle the
/// TTS tests lean on.
#[test]
fn wer_metric_is_sane() {
    assert_eq!(wer("the quick brown fox", "the quick brown fox"), 0.0);
    assert_eq!(wer("Hello, World!", "hello world"), 0.0);
    assert!((wer("a b c d", "a b c x") - 0.25).abs() < 1e-6);
}

/// Real transcription — env-gated; skips off-box.
#[test]
fn real_stt_transcribes_clip() {
    let model_dir = match env("SYRINX_STT_MODEL_DIR") {
        Some(d) => d,
        None => {
            eprintln!(
                "skipping real_stt: set SYRINX_STT_MODEL_DIR to a Whisper model dir \
                 (config.json + tokenizer.json + model.safetensors)"
            );
            return;
        }
    };
    let wav = match env("SYRINX_STT_WAV") {
        Some(w) => w,
        None => {
            eprintln!("skipping real_stt: set SYRINX_STT_WAV to a test clip WAV");
            return;
        }
    };

    let (samples_16k, _r24) = wavio::read_ref_wav(Path::new(&wav)).expect("read test WAV");

    let stt = Stt::load(&model_dir, candle_core::Device::Cpu).expect("load Whisper model");
    let transcript = stt
        .transcribe_lang(&samples_16k, 16_000, env("SYRINX_STT_LANG").as_deref())
        .expect("transcribe clip");

    eprintln!(
        "real_stt: language={:?} segments={} text={:?}",
        transcript.language,
        transcript.segments.len(),
        transcript.text
    );

    assert!(
        !transcript.text.trim().is_empty(),
        "transcript should be non-empty for a speech clip"
    );

    // If a reference transcript is given, assert intelligibility via WER.
    if let Some(reference) = env("SYRINX_STT_REF") {
        let threshold: f32 = env("SYRINX_STT_WER_MAX")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let score = wer(&reference, &transcript.text);
        eprintln!("real_stt: WER {score:.4} (threshold {threshold:.4})");
        assert!(
            score <= threshold,
            "WER {score:.4} exceeds threshold {threshold:.4} \
             (reference {reference:?} vs transcript {:?})",
            transcript.text
        );
    }
}
