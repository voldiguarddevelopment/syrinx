//! Real **CosyVoice3** HiFT-vocoder (`CausalHiFTGenerator`) parity — the model-box /
//! real-weights track. The CV3 deltas vs CV2 are causal convolutions and a float64
//! f0_predictor (see `crates/syrinx-vocoder/src/real_cv3.rs`).
//!
//! Gated on the `real` feature AND env vars pointing at the `weight_norm`-parametrized
//! fp32 checkpoint + the CV3 reference dump (both too large to vendor — they live on
//! the model box). Skips cleanly when absent, like the device-bound task recipe.
//!
//!   SYRINX_CV3_HIFT_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/hift_fp32.safetensors \
//!   SYRINX_CV3_HIFT_REF=/root/parity-cv3/hift/ref.safetensors \
//!   cargo test -p syrinx-vocoder --features real --test real_cv3_hift_parity -- --nocapture

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_vocoder::real_cv3::Cv3Hift;

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

fn abs_max(a: &Tensor) -> f32 {
    a.abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn real_cv3_hift_decode_matches_reference_within_1e_3() {
    let (weights, reference) = match (
        std::env::var("SYRINX_CV3_HIFT_WEIGHTS").ok(),
        std::env::var("SYRINX_CV3_HIFT_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => (w, r),
        _ => {
            eprintln!(
                "SKIP real_cv3_hift parity: set SYRINX_CV3_HIFT_WEIGHTS + SYRINX_CV3_HIFT_REF \
                 to the on-box fp32 checkpoint + CV3 reference dump"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let hift = Cv3Hift::load(&weights, dev.clone()).expect("load CV3 weights");

    let r = candle_core::safetensors::load(&reference, &dev).expect("load reference fixture");
    let get = |k: &str| {
        r.get(k)
            .unwrap_or_else(|| panic!("reference has {k}"))
            .to_dtype(DType::F32)
            .unwrap()
    };
    let mel = get("mel"); // [1, 80, 204]
    let source = get("source"); // [1, 1, 97920]
    let exp_f0 = get("f0"); // [1, 204]
    let exp_s_stft = get("s_stft"); // [1, 18, 24481]
    let exp_audio = get("audio"); // [1, 97920]

    // Stage 1 — the float64 f0_predictor (tightest; isolates causal/f64 handling).
    // F0 is in Hz, so judge relative to its scale rather than the audio's [-1,1] tol.
    let f0 = hift.f0_predict(&mel).expect("f0_predict");
    assert_eq!(f0.dims(), exp_f0.dims(), "f0 shape mismatch");
    let f0_diff = max_abs_diff(&f0, &exp_f0);
    let f0_rel = f0_diff / abs_max(&exp_f0);
    eprintln!("f0       max-abs-diff = {f0_diff:.3e}   relative = {f0_rel:.3e}");
    assert!(f0_rel < 1e-3, "f0 relative diff {f0_rel:.3e} exceeds 1e-3");

    // Stage 2 — _stft(source) -> s_stft.
    let s_stft = hift.stft(&source).expect("stft");
    assert_eq!(s_stft.dims(), exp_s_stft.dims(), "s_stft shape mismatch");
    let s_diff = max_abs_diff(&s_stft, &exp_s_stft);
    eprintln!("s_stft   max-abs-diff = {s_diff:.3e}");
    assert!(s_diff < 1e-3, "s_stft max-abs diff {s_diff:.3e} exceeds 1e-3");

    // Stage 3 — full causal decode(mel, source) -> audio (the headline number).
    let audio = hift.decode(&mel, &source).expect("decode");
    assert_eq!(audio.dims(), exp_audio.dims(), "audio shape mismatch");
    let a_diff = max_abs_diff(&audio, &exp_audio);
    eprintln!("audio    max-abs-diff = {a_diff:.3e}");
    assert!(a_diff < 1e-3, "audio max-abs diff {a_diff:.3e} exceeds the 1e-3 parity tolerance");
}
