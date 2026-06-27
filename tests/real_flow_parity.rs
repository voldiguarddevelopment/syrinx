//! Real CosyVoice2 flow-matching mel parity (the real-weights track, buildable on
//! the model box). Gated on the `real` feature AND env vars pointing at the
//! converted fp32 checkpoint + the Python reference dump (both too large to vendor
//! — they live on the model box). Skips cleanly when absent.
//!
//!   SYRINX_FLOW_WEIGHTS=/root/models/CosyVoice2-0.5B/flow_fp32.safetensors \
//!   SYRINX_FLOW_REF=/root/parity/acoustic/ref.safetensors \
//!   cargo test -p syrinx-acoustic --features real -- --nocapture
//!
//! Stages (each isolates a failure region): encoder output, mu (encoder_proj),
//! one estimator call, then the full 10-step Euler ODE mel. The spec pins mel at
//! 1e-2.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_acoustic::cv2::Flow;

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

fn get(r: &std::collections::HashMap<String, Tensor>, k: &str) -> Tensor {
    r.get(k)
        .unwrap_or_else(|| panic!("reference missing `{k}`"))
        .to_dtype(DType::F32)
        .unwrap()
}

#[test]
fn real_flow_mel_matches_reference_within_1e_2() {
    let (weights, reference) = match (
        std::env::var("SYRINX_FLOW_WEIGHTS").ok(),
        std::env::var("SYRINX_FLOW_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => (w, r),
        _ => {
            eprintln!(
                "SKIP real_flow parity: set SYRINX_FLOW_WEIGHTS + SYRINX_FLOW_REF to the \
                 on-disk fp32 fixtures (flow weights + reference dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let flow = Flow::load(&weights, dev.clone()).expect("load fp32 flow weights");
    let r = candle_core::safetensors::load(&reference, &dev).expect("load reference fixture");

    let token = get(&r, "token").to_dtype(DType::I64).unwrap();
    let embedding = get(&r, "embedding");

    // ---- Stage: speaker projection ----
    let spk = flow.spk_proj(&embedding).expect("spk_proj");
    let exp_spk = get(&r, "spk_proj");
    eprintln!("spk_proj  max-abs-diff = {:.3e}", max_abs_diff(&spk, &exp_spk));

    // ---- Stage: input embedding ----
    let emb = flow.input_embedding(&token).expect("input_embedding");
    let exp_emb = get(&r, "input_emb");
    assert_eq!(emb.dims(), exp_emb.dims(), "input_emb shape");
    eprintln!("input_emb max-abs-diff = {:.3e}", max_abs_diff(&emb, &exp_emb));

    // ---- Stage: encoder output ----
    let enc = flow.encoder(&emb).expect("encoder");
    let exp_enc = get(&r, "encoder_out");
    assert_eq!(enc.dims(), exp_enc.dims(), "encoder_out shape");
    let d_enc = max_abs_diff(&enc, &exp_enc);
    eprintln!("encoder   max-abs-diff = {:.3e}", d_enc);

    // ---- Stage: mu (encoder_proj) ----
    let mu = flow
        .real_linear_pub(&enc, "encoder_proj.weight", Some("encoder_proj.bias"))
        .expect("encoder_proj");
    let exp_mu = get(&r, "mu");
    eprintln!("mu        max-abs-diff = {:.3e}", max_abs_diff(&mu, &exp_mu));

    // ---- Stage: single estimator call (step 1) ----
    let est_x = get(&r, "est_step1_x");
    let est_mu = get(&r, "est_step1_mu");
    let est_t = get(&r, "est_step1_t");
    let est_spks = get(&r, "est_step1_spks");
    let est_cond = get(&r, "est_step1_cond");
    let est_out = flow
        .estimator(&est_x, &est_mu, &est_t, &est_spks, &est_cond)
        .expect("estimator");
    let exp_est = get(&r, "est_step1_out");
    assert_eq!(est_out.dims(), exp_est.dims(), "estimator shape");
    let d_est = max_abs_diff(&est_out, &exp_est);
    eprintln!("estimator max-abs-diff = {:.3e}", d_est);

    // ---- Stage: full ODE mel (uses the reference frozen noise z) ----
    let mu_t = exp_mu.transpose(1, 2).unwrap().contiguous().unwrap(); // [1,80,L]
    let z = get(&r, "z");
    let mel = flow
        .cfm_solve_with_noise(&mu_t, &exp_spk, &z, 10)
        .expect("cfm_solve");
    let exp_mel = get(&r, "mel");
    assert_eq!(mel.dims(), exp_mel.dims(), "mel shape");
    let d_mel = max_abs_diff(&mel, &exp_mel);
    eprintln!("mel       max-abs-diff = {:.3e}", d_mel);

    assert!(d_enc < 1e-3, "encoder max-abs diff {d_enc:.3e} exceeds 1e-3");
    assert!(d_est < 1e-3, "estimator max-abs diff {d_est:.3e} exceeds 1e-3");
    assert!(d_mel < 1e-2, "mel max-abs diff {d_mel:.3e} exceeds the 1e-2 parity tolerance");
}
