//! Real CosyVoice2 HiFT-vocoder parity (the model-box / real-weights track).
//!
//! Gated on the `real` feature AND env vars pointing at the folded fp32 checkpoint
//! + the Python reference dump (both too large to vendor — they live on the model
//! box). Skips cleanly when absent, like the device-bound task recipe.
//!
//!   SYRINX_HIFT_WEIGHTS=/root/parity/vocoder/hift_fp32.safetensors \
//!   SYRINX_HIFT_REF=/root/parity/vocoder/ref.safetensors \
//!   cargo test -p syrinx-vocoder --features real -- --nocapture

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_vocoder::cv2::HiftVocoder;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn real_hift_decode_matches_reference_within_1e_3() {
    let (weights, reference) = match (
        std::env::var("SYRINX_HIFT_WEIGHTS").ok(),
        std::env::var("SYRINX_HIFT_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => (w, r),
        _ => {
            eprintln!(
                "SKIP real_hift parity: set SYRINX_HIFT_WEIGHTS + SYRINX_HIFT_REF to the \
                 on-disk fp32 fixtures (folded hift.pt + reference dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let hift = HiftVocoder::load(&weights, dev.clone()).expect("load fp32 weights");

    let r = candle_core::safetensors::load(&reference, &dev).expect("load reference fixture");
    let get = |k: &str| {
        r.get(k)
            .unwrap_or_else(|| panic!("reference has {k}"))
            .to_dtype(DType::F32)
            .unwrap()
    };
    let mel = get("mel"); // [1, 80, 32]
    let s_stft = get("s_stft"); // [1, 18, T]
    let expected_wav = get("waveform"); // [1, 15360]

    // Stage 1: the deterministic F0 predictor (isolates conv-stack vs. head bugs).
    // F0 is in Hz (here up to ~4 kHz), so parity is judged *relative* to its scale,
    // not against the waveform's [-1, 1] absolute tolerance.
    let exp_f0 = get("f0");
    let f0 = hift.f0_predict(&mel).expect("f0_predict");
    assert_eq!(f0.dims(), exp_f0.dims(), "f0 shape mismatch");
    let f0_diff = max_abs_diff(&f0, &exp_f0);
    let f0_scale = exp_f0
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let f0_rel = f0_diff / f0_scale;
    eprintln!("f0       max-abs-diff = {f0_diff:.3e}   relative = {f0_rel:.3e}");
    assert!(f0_rel < 1e-3, "f0 relative diff {f0_rel:.3e} exceeds 1e-3");

    // Stage 2: the full decode(mel, s_stft) -> waveform path (conv/upsample/resblock
    // stack + iSTFT head + audio clamp).
    let wav = hift.decode(&mel, &s_stft).expect("decode");
    assert_eq!(wav.dims(), expected_wav.dims(), "waveform shape mismatch");
    let d = max_abs_diff(&wav, &expected_wav);
    eprintln!("waveform max-abs-diff = {d:.3e}");
    assert!(d < 1e-3, "waveform max-abs diff {d:.3e} exceeds the 1e-3 parity tolerance");
}
