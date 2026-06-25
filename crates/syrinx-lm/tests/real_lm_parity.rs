//! Real CosyVoice2 LM-forward parity (the GPU / real-weights track — blocked task
//! line T-02.0x, now buildable on the model box).
//!
//! Gated on the `real` feature AND env vars pointing at the converted fp32
//! checkpoint + the Python reference dump (both too large to vendor — they live on
//! the GPU box). Skips cleanly when absent, like the device-bound task recipe.
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_LM_REF=/root/parity/lm_ref_fp32.safetensors \
//!   cargo test -p syrinx-lm --features real -- --nocapture

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_lm::real::Qwen2Lm;

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

/// Argmax over the vocab at the last position of a `[1, T, V]` logit tensor.
fn argmax_last(t: &Tensor) -> u32 {
    let tt = t.dim(1).unwrap();
    t.narrow(1, tt - 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .argmax(0)
        .unwrap()
        .to_scalar::<u32>()
        .unwrap()
}

#[test]
fn real_lm_forward_matches_reference_within_1e_3() {
    let (weights, reference) = match (
        std::env::var("SYRINX_LM_WEIGHTS").ok(),
        std::env::var("SYRINX_LM_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => (w, r),
        _ => {
            eprintln!(
                "SKIP real_lm_forward parity: set SYRINX_LM_WEIGHTS + SYRINX_LM_REF to the \
                 on-disk fp32 fixtures (CosyVoice2 weights + reference dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let lm = Qwen2Lm::load(&weights, dev.clone()).expect("load fp32 weights");

    let r = candle_core::safetensors::load(&reference, &dev).expect("load reference fixture");
    let embeds = r
        .get("input_embeds")
        .expect("reference has input_embeds")
        .to_dtype(DType::F32)
        .unwrap();
    let expected = r
        .get("logits")
        .expect("reference has logits")
        .to_dtype(DType::F32)
        .unwrap();

    // Staged check: hidden state first (isolates transformer vs. decoder-head bugs).
    if let Some(exp_hidden) = r.get("hidden") {
        let hidden = lm.forward_hidden(&embeds).expect("forward_hidden");
        let exp_hidden = exp_hidden.to_dtype(DType::F32).unwrap();
        assert_eq!(hidden.dims(), exp_hidden.dims(), "hidden shape mismatch");
        eprintln!("hidden  max-abs-diff = {:.3e}", max_abs_diff(&hidden, &exp_hidden));
    }

    let logits = lm.forward_logits(&embeds).expect("forward_logits");
    assert_eq!(logits.dims(), expected.dims(), "logit shape mismatch");

    let d = max_abs_diff(&logits, &expected);
    eprintln!(
        "logits  max-abs-diff = {:.3e}   argmax ref={} ours={}",
        d,
        argmax_last(&expected),
        argmax_last(&logits)
    );
    assert!(d < 1e-3, "logits max-abs diff {d:.3e} exceeds the 1e-3 parity tolerance");
}
