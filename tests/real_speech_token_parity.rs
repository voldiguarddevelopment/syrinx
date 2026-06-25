//! Real CosyVoice2 **prompt speech-token** parity — the real-weights track,
//! buildable on the model box. Mirrors `CosyVoiceFrontEnd._extract_speech_token`:
//! a reference 16 kHz mono wav is turned into the `prompt_speech_token` id
//! sequence (whisper log-mel computed in Rust, then `speech_tokenizer_v2.onnx`
//! run through ONNX Runtime).
//!
//! Gated on the `real` feature AND env vars pointing at the ONNX model + the
//! Python reference dump (both too large / model-box-only to vendor). Skips
//! cleanly when absent, like the other device-bound parity tests.
//!
//!   SYRINX_STOK_ONNX=/root/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
//!   SYRINX_STOK_REF=/root/parity/frontend/stok_ref.safetensors \
//!   cargo test -p syrinx-workspace-scaffold-tests --features real \
//!       --test real_speech_token_parity -- --nocapture
//!
//! The reference dump (`stok_ref.safetensors`) holds, per clip `{a,b}`:
//!   * `wav_{a,b}`    : [N] f32  16 kHz mono waveform,
//!   * `mel_{a,b}`    : [128,T]  whisper log-mel (staged check),
//!   * `tokens_{a,b}` : [T_tok]  i32 reference speech-token ids.

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};
use std::path::Path;
use syrinx_frontend::speech_token::{self, SpeechTokenizer};

/// Pull a f32 1-D tensor out of the safetensors dump as a `Vec<f32>`.
fn vec_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Pull an i32 1-D tensor out of the dump as a `Vec<i32>`.
fn vec_i32(t: &Tensor) -> Vec<i32> {
    t.to_dtype(candle_core::DType::I64)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

/// Hamming-style edit count: positions where the two equal-length id sequences
/// differ. Used only for a diagnostic message; the assertion demands exact match.
fn mismatch_count(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
fn real_speech_token_matches_reference_exactly() {
    let (onnx, refpath) = match (
        std::env::var("SYRINX_STOK_ONNX").ok(),
        std::env::var("SYRINX_STOK_REF").ok(),
    ) {
        (Some(o), Some(r)) if Path::new(&o).exists() && Path::new(&r).exists() => (o, r),
        _ => {
            eprintln!(
                "SKIP real_speech_token parity: set SYRINX_STOK_ONNX (speech_tokenizer_v2.onnx) \
                 + SYRINX_STOK_REF (stok_ref.safetensors) to the on-disk model-box fixtures"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let r = candle_core::safetensors::load(&refpath, &dev).expect("load reference fixture");

    let mut tok = SpeechTokenizer::load(&onnx).expect("load speech_tokenizer_v2.onnx via ort");

    // Each clip present in the dump is checked independently.
    for clip in ["a", "b"] {
        let wav = match r.get(&format!("wav_{clip}")) {
            Some(t) => t,
            None => continue,
        };
        let expected = vec_i32(r.get(&format!("tokens_{clip}")).expect("ref has tokens"));
        let samples = vec_f32(wav);

        // --- Staged check 1: Rust whisper log-mel matches the reference mel. ---
        if let Some(mel_ref_t) = r.get(&format!("mel_{clip}")) {
            let ref_dims = mel_ref_t.dims().to_vec();
            assert_eq!(ref_dims.len(), 2, "clip {clip}: ref mel must be [128,T]");
            let mel_ref = vec_f32(mel_ref_t); // row-major [n_mels, T]
            let (mel_got, n_mels, n_frames) =
                speech_token::log_mel_flat(&samples).expect("rust log-mel");
            assert_eq!(
                (n_mels, n_frames),
                (ref_dims[0], ref_dims[1]),
                "clip {clip}: mel shape ({n_mels},{n_frames}) != ref ({},{})",
                ref_dims[0],
                ref_dims[1]
            );
            let mut maxd = 0.0f32;
            for (g, e) in mel_got.iter().zip(mel_ref.iter()) {
                maxd = maxd.max((g - e).abs());
            }
            eprintln!("clip {clip}: whisper log-mel max-abs-diff = {maxd:.3e}");
            // The mel feeds an FSQ-quantised encoder; tokens are exact only if the
            // mel matches tightly. 1e-3 is comfortably below the quantiser's grid.
            assert!(
                maxd < 1e-3,
                "clip {clip}: log-mel max-abs diff {maxd:.3e} exceeds 1e-3"
            );
        }

        // --- Staged check 2: end-to-end token ids match the reference exactly. ---
        let got = tok.tokens_from_wav(&samples).expect("rust speech tokens");

        eprintln!(
            "clip {clip}: ntok rust={} ref={}",
            got.len(),
            expected.len()
        );
        assert_eq!(
            got.len(),
            expected.len(),
            "clip {clip}: token count {} != ref {}",
            got.len(),
            expected.len()
        );
        let diffs = mismatch_count(&got, &expected);
        if diffs != 0 {
            let first = got
                .iter()
                .zip(expected.iter())
                .position(|(x, y)| x != y)
                .unwrap();
            eprintln!(
                "clip {clip}: {diffs} token mismatch(es); first at idx {first}: rust={} ref={}",
                got[first], expected[first]
            );
        }
        assert_eq!(
            got, expected,
            "clip {clip}: speech-token ids diverge from reference ({diffs} positions differ)"
        );
    }
}
