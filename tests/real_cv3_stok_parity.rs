//! Real **CosyVoice3 prompt speech-token** parity (the v3 tokenizer port).
//!
//! Mirrors `CosyVoice3`'s `_extract_speech_token` — which is the CosyVoice2 method
//! unchanged: a 16 kHz mono reference wav -> `whisper.log_mel_spectrogram(n_mels=128)`
//! computed in Rust -> `speech_tokenizer_v3.onnx` run through ONNX Runtime -> the flat
//! int32 `prompt_speech_token` id sequence used for zero-shot voice cloning.
//!
//! The v3 ONNX graph I/O was inspected (`onnx.load`) and is byte-identical to v2:
//!   inputs  `feats: f32 [1,128,T]`, `feats_length: i32 [1]`
//!   output  `indices` (int32 token ids)
//! so the exact v2 whisper-log-mel + `ort` session path is reused; only the model
//! file swaps to v3. Speech tokens are DISCRETE ids, so the parity target is an
//! EXACT id match (a single trailing tail-boundary token is tolerated -> >=99%
//! agreement); the test prints the agreement and any mismatched positions.
//!
//! Gated on the `real` feature AND env vars pointing at the v3 ONNX + the verified
//! CV3 reference dump (both model-box-only, too large to vendor). Skips cleanly when
//! absent, like the other device-bound parity tests. The maintainer runs it on-box
//! (the box runs the actual `ort` inference; this repo only builds the path):
//!
//!   SYRINX_CV3_STOK_ONNX=/root/models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx \
//!   SYRINX_CV3_STOK_REF=/root/parity-cv3/e2e/ref.safetensors \
//!   cargo test -p syrinx-workspace-scaffold-tests --features real \
//!       --test real_cv3_stok_parity -- --nocapture
//!
//! The reference dump (`/root/parity-cv3/e2e/ref.safetensors`) holds the standard
//! zero-shot ref clip (`/root/CosyVoice/asset/zero_shot_prompt.wav` @16k):
//!   * `prompt_wav_16k`      : [1, 55680] f32  16 kHz mono waveform (the tokenizer input),
//!   * `prompt_speech_token` : [1, 87]    i64  the v3 reference token ids.

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};
use std::path::Path;
use syrinx_frontend::speech_token::SpeechTokenizer;

/// Pull a f32 tensor out of the safetensors dump as a flat `Vec<f32>`.
fn vec_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Pull an integer tensor (dumped i64) out of the dump as a flat `Vec<i32>`.
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

#[test]
fn real_cv3_speech_token_matches_reference() {
    let (onnx, refpath) = match (
        std::env::var("SYRINX_CV3_STOK_ONNX").ok(),
        std::env::var("SYRINX_CV3_STOK_REF").ok(),
    ) {
        (Some(o), Some(r)) if Path::new(&o).exists() && Path::new(&r).exists() => (o, r),
        _ => {
            eprintln!(
                "SKIP real_cv3_stok parity: set SYRINX_CV3_STOK_ONNX (speech_tokenizer_v3.onnx) \
                 + SYRINX_CV3_STOK_REF (/root/parity-cv3/e2e/ref.safetensors)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let r = candle_core::safetensors::load(&refpath, &dev).expect("load CV3 reference fixture");

    // The 16 kHz tokenizer input is embedded in the dump (the exact wav that produced
    // the reference ids). An explicit `prompt_wav_16k` key is required.
    let wav_t = r
        .get("prompt_wav_16k")
        .expect("CV3 ref must contain prompt_wav_16k [1,N] f32");
    let samples = vec_f32(wav_t);

    let expected = vec_i32(
        r.get("prompt_speech_token")
            .expect("CV3 ref must contain prompt_speech_token [1,T_tok]"),
    );

    let mut tok =
        SpeechTokenizer::load_cv3(&onnx).expect("load speech_tokenizer_v3.onnx via ort");
    let got = tok
        .tokens_from_wav(&samples)
        .expect("rust CV3 speech tokens");

    eprintln!(
        "CV3 stok: wav={} samples ({:.2}s); ntok rust={} ref={}",
        samples.len(),
        samples.len() as f32 / 16000.0,
        got.len(),
        expected.len(),
    );

    // Discrete-id agreement over the overlapping prefix. A single trailing boundary
    // token (length differing by 1) is tolerated; everything else must match exactly.
    let overlap = got.len().min(expected.len());
    let mut mismatches: Vec<(usize, i32, i32)> = Vec::new();
    for i in 0..overlap {
        if got[i] != expected[i] {
            mismatches.push((i, got[i], expected[i]));
        }
    }
    let agree = overlap - mismatches.len();
    let denom = got.len().max(expected.len()).max(1);
    let agreement = agree as f32 / denom as f32;

    if !mismatches.is_empty() {
        eprintln!("CV3 stok mismatched positions (idx rust ref):");
        for (i, g, e) in mismatches.iter().take(32) {
            eprintln!("  {i}: {g} != {e}");
        }
    }
    eprintln!(
        "CV3 stok id agreement = {agree}/{denom} = {:.4} ({} mismatch, len_diff {})",
        agreement,
        mismatches.len(),
        (got.len() as isize - expected.len() as isize).abs(),
    );

    if got.len() == expected.len() && mismatches.is_empty() {
        // Best case: exact match.
        return;
    }

    // Otherwise accept only a single tail-boundary discrepancy: >=99% id agreement
    // AND the length differs by at most one token.
    let len_diff = (got.len() as isize - expected.len() as isize).abs();
    assert!(
        agreement >= 0.99 && len_diff <= 1,
        "CV3 speech-token parity failed: agreement {agreement:.4} (<0.99) or len_diff {len_diff} (>1); \
         {} interior mismatches",
        mismatches.len()
    );
}
