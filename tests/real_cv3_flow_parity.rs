//! Real CosyVoice3 flow-matching mel parity (`CausalMaskedDiffWithDiT`), buildable on
//! the model box. Gated on the `real` feature AND env vars pointing at the converted
//! fp32 CV3 flow checkpoint + the Python reference dump (both too large to vendor —
//! they live on the model box). Skips cleanly when absent.
//!
//!   SYRINX_CV3_FLOW_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/flow_fp32.safetensors \
//!   SYRINX_CV3_FLOW_REF=/root/parity-cv3/flow/ref.safetensors \
//!   cargo test --features real --test real_cv3_flow_parity -- --nocapture
//!
//! Two tests:
//!   * `real_cv3_flow_estimator_matches_reference` — the **DiT in isolation** (one
//!     forward, no Euler loop) against `est_step1_*`. The critical 22-layer anchor; run
//!     this first to debug the transformer alone.
//!   * `real_cv3_flow_mel_matches_reference` — every stage in fail-fast order:
//!     spk_proj, input_emb, pre_lookahead_out, mu, est_step1, mel_full, mel.
//!
//! Tolerances: the exact front-end stages (spk_proj/input_emb/pre_lookahead/mu) hold
//! tight, < 1e-4 (on box: 1.2e-7 / 0.0 / 8.3e-6). The deep 22-layer DiT stages
//! (`est_step1_out`, `mel_full`, `mel`) use **4e-3** — the empirical fp32 accumulation
//! floor, not a loose pass. A torch-internal fp32-vs-fp64 run of the *same* DiT on the
//! *same* fixture already differs by 1.338e-3 at the output: its hidden states diverge
//! to ~0.15 purely from rounding (the final AdaLN+proj compresses that back), and the
//! per-block diff grows smoothly (1e-4 -> 2.6e-3 -> ... -> 1.5e-1) with no single-op
//! jump. So 1e-3 is *below* the fp32 noise floor — unreachable by any faithful fp32
//! port; candle-vs-torch lands at 2.27e-3 (est) / 1.49e-3 (mel). 4e-3 passes a faithful
//! port with margin yet still fails any real structural error, which — given that
//! 0.15 hidden-state sensitivity — would push the output well past 1e-2.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use syrinx_acoustic::real_cv3::Cv3Flow;

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

fn get(r: &HashMap<String, Tensor>, k: &str) -> Tensor {
    r.get(k)
        .unwrap_or_else(|| panic!("reference missing `{k}`"))
        .to_dtype(DType::F32)
        .unwrap()
}

fn fixtures() -> Option<(String, String)> {
    match (
        std::env::var("SYRINX_CV3_FLOW_WEIGHTS").ok(),
        std::env::var("SYRINX_CV3_FLOW_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => Some((w, r)),
        _ => {
            eprintln!(
                "SKIP real_cv3_flow parity: set SYRINX_CV3_FLOW_WEIGHTS + SYRINX_CV3_FLOW_REF \
                 to the on-disk fp32 fixtures (CV3 flow weights + reference dump)"
            );
            None
        }
    }
}

/// The DiT estimator alone, against `est_step1_*` (CFG batch-2: idx0 real, idx1 zeros).
/// The single most important anchor for the 22-layer port — no Euler loop.
#[test]
fn real_cv3_flow_estimator_matches_reference() {
    let Some((weights, reference)) = fixtures() else {
        return;
    };
    let dev = Device::Cpu;
    let flow = Cv3Flow::load(&weights, dev.clone()).expect("load CV3 flow weights");
    let r = candle_core::safetensors::load(&reference, &dev).expect("load reference fixture");

    let x = get(&r, "est_step1_x");
    let mu = get(&r, "est_step1_mu");
    let t = get(&r, "est_step1_t");
    let spks = get(&r, "est_step1_spks");
    let cond = get(&r, "est_step1_cond");
    let out = flow.estimator(&x, &mu, &t, &spks, &cond).expect("estimator");
    let exp = get(&r, "est_step1_out");
    assert_eq!(out.dims(), exp.dims(), "est_step1_out shape");
    let d = max_abs_diff(&out, &exp);
    eprintln!("est_step1_out  max-abs-diff = {d:.3e}  (shape {:?})", out.dims());
    // 4e-3 = the fp32 accumulation floor for this 22-layer DiT (see module doc): torch's
    // own fp32-vs-fp64 output diff is 1.338e-3; candle-vs-torch is 2.27e-3. A real bug
    // would exceed 1e-2 given the ~0.15 hidden-state sensitivity.
    assert!(d < 4e-3, "DiT est_step1_out max-abs diff {d:.3e} exceeds the 4e-3 fp32 floor");
}

/// Every CV3 flow stage in fail-fast order, ending at the full 10-step Euler mel.
#[test]
fn real_cv3_flow_mel_matches_reference() {
    let Some((weights, reference)) = fixtures() else {
        return;
    };
    let dev = Device::Cpu;
    let flow = Cv3Flow::load(&weights, dev.clone()).expect("load CV3 flow weights");
    let r = candle_core::safetensors::load(&reference, &dev).expect("load reference fixture");

    let embedding = get(&r, "embedding");
    let token_cat = get(&r, "token_cat").to_dtype(DType::I64).unwrap();

    // ---- Stage 1: speaker projection (normalize + linear, ~0) ----
    let spk = flow.spk_proj(&embedding).expect("spk_proj");
    let exp_spk = get(&r, "spk_proj");
    assert_eq!(spk.dims(), exp_spk.dims(), "spk_proj shape");
    let d_spk = max_abs_diff(&spk, &exp_spk);
    eprintln!("spk_proj          max-abs-diff = {d_spk:.3e}");

    // ---- Stage 2: input embedding (lookup + mask, ~0) ----
    let emb = flow.input_embedding(&token_cat).expect("input_embedding");
    let exp_emb = get(&r, "input_emb");
    assert_eq!(emb.dims(), exp_emb.dims(), "input_emb shape");
    let d_emb = max_abs_diff(&emb, &exp_emb);
    eprintln!("input_emb         max-abs-diff = {d_emb:.3e}");

    // ---- Stage 3: PreLookaheadLayer (conv path, < 1e-4) ----
    let pre = flow.pre_lookahead(&exp_emb).expect("pre_lookahead");
    let exp_pre = get(&r, "pre_lookahead_out");
    assert_eq!(pre.dims(), exp_pre.dims(), "pre_lookahead_out shape");
    let d_pre = max_abs_diff(&pre, &exp_pre);
    eprintln!("pre_lookahead_out max-abs-diff = {d_pre:.3e}");

    // ---- Stage 4: full token -> mu (repeat_interleave + transpose, < 1e-4) ----
    let mu = flow.token_to_mu(&token_cat).expect("token_to_mu");
    let exp_mu = get(&r, "mu");
    assert_eq!(mu.dims(), exp_mu.dims(), "mu shape");
    let d_mu = max_abs_diff(&mu, &exp_mu);
    eprintln!("mu                max-abs-diff = {d_mu:.3e}");

    // ---- Stage 5: single DiT estimator call (step 1) ----
    let est_x = get(&r, "est_step1_x");
    let est_mu = get(&r, "est_step1_mu");
    let est_t = get(&r, "est_step1_t");
    let est_spks = get(&r, "est_step1_spks");
    let est_cond = get(&r, "est_step1_cond");
    let est_out = flow
        .estimator(&est_x, &est_mu, &est_t, &est_spks, &est_cond)
        .expect("estimator");
    let exp_est = get(&r, "est_step1_out");
    assert_eq!(est_out.dims(), exp_est.dims(), "est_step1_out shape");
    let d_est = max_abs_diff(&est_out, &exp_est);
    eprintln!("est_step1_out     max-abs-diff = {d_est:.3e}");

    // ---- Stage 6: full 10-step Euler ODE (consume frozen z + reference cond) ----
    let z = get(&r, "z");
    let cond = get(&r, "cond");
    let mel_full = flow.cfm_solve(&exp_mu, &exp_spk, &cond, &z, 10).expect("cfm_solve");
    let exp_mel_full = get(&r, "mel_full");
    assert_eq!(mel_full.dims(), exp_mel_full.dims(), "mel_full shape");
    let d_mel_full = max_abs_diff(&mel_full, &exp_mel_full);
    eprintln!("mel_full          max-abs-diff = {d_mel_full:.3e}");

    // ---- mel = mel_full[:, :, mel_len1:] (drop the prompt-mel prefix) ----
    let mel_len1 = exp_mel_full.dim(2).unwrap() - get(&r, "mel").dim(2).unwrap();
    let mel = mel_full.narrow(2, mel_len1, get(&r, "mel").dim(2).unwrap()).unwrap();
    let exp_mel = get(&r, "mel");
    assert_eq!(mel.dims(), exp_mel.dims(), "mel shape");
    let d_mel = max_abs_diff(&mel, &exp_mel);
    eprintln!("mel               max-abs-diff = {d_mel:.3e}");

    assert!(d_spk < 1e-4, "spk_proj diff {d_spk:.3e} exceeds 1e-4");
    assert!(d_emb < 1e-4, "input_emb diff {d_emb:.3e} exceeds 1e-4");
    assert!(d_pre < 1e-4, "pre_lookahead_out diff {d_pre:.3e} exceeds 1e-4");
    assert!(d_mu < 1e-4, "mu diff {d_mu:.3e} exceeds 1e-4");
    // Deep-DiT stages: 4e-3 fp32 accumulation floor (module doc). Front-end stays exact.
    assert!(d_est < 4e-3, "est_step1_out diff {d_est:.3e} exceeds the 4e-3 fp32 floor");
    assert!(d_mel_full < 4e-3, "mel_full diff {d_mel_full:.3e} exceeds the 4e-3 fp32 floor");
    assert!(d_mel < 4e-3, "mel diff {d_mel:.3e} exceeds the 4e-3 fp32 floor");
}
