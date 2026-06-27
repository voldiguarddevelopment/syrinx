//! Real **CosyVoice3** LM-forward parity (the first CV3 component port — its anchor test).
//!
//! Two deterministic, RNG-free checks against the verified reference dump:
//!   1. **Input-embeds exactness.** `Cv3Lm::build_lm_input(text_token, prompt_speech_token)`
//!      must reproduce the reference `input_embeds` `[1, 126, 896]` bit-for-bit (it is pure
//!      embedding lookups + concat: `sos_emb || embed_tokens(text) || task_id_emb ||
//!      speech_embedding(prompt_speech)`, with `sos`/`task_id` drawn from `speech_embedding`).
//!   2. **Teacher-forced logit parity.** Feed the reference `teacher_embeds` `[1, 227, 896]`
//!      through the Qwen2 body in ONE forward → bias-free `llm_decoder` → take the 32
//!      per-step logits at positions `t0-1 .. t0-1+32` and compare to `step_logits`
//!      `[32, 6761]`. The Python teacher-vs-incremental maxabs was 1.9e-5; we allow < 1e-3
//!      abs for candle-vs-torch fp32 gemm.
//!
//! Gated on the `real` feature AND env vars pointing at the on-box fp32 checkpoint + dump
//! (both too large to vendor — they live on the GPU box). Skips cleanly when absent, like
//! the device-bound task recipe.
//!
//!   SYRINX_CV3_LM_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/llm_fp32.safetensors \
//!   SYRINX_CV3_LM_REF=/root/parity-cv3/lm/ref.safetensors \
//!   cargo test -p syrinx-lm --features real --test real_cv3_lm_parity -- --nocapture

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_lm::cv3::Cv3Lm;

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

/// Read a CV3 ref integer tensor (dumped as I32, loaded by candle as I64) as `Vec<u32>`.
fn ids(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect()
}

/// Argmax over the vocab of a single `[V]` logit row.
fn argmax_row(t: &Tensor) -> u32 {
    t.argmax(0).unwrap().to_scalar::<u32>().unwrap()
}

#[test]
fn real_cv3_lm_forward_matches_reference_within_1e_3() {
    let (weights, reference) = match (
        std::env::var("SYRINX_CV3_LM_WEIGHTS").ok(),
        std::env::var("SYRINX_CV3_LM_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => (w, r),
        _ => {
            eprintln!(
                "SKIP real_cv3_lm_forward parity: set SYRINX_CV3_LM_WEIGHTS + \
                 SYRINX_CV3_LM_REF to the on-disk fp32 fixtures (CosyVoice3 \
                 llm_fp32.safetensors + the parity-cv3/lm/ref.safetensors dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let lm = Cv3Lm::load(&weights, dev.clone()).expect("load CV3 fp32 weights");

    let r = candle_core::safetensors::load(&reference, &dev).expect("load CV3 reference fixture");
    let f32t = |k: &str| {
        r.get(k)
            .unwrap_or_else(|| panic!("reference missing {k}"))
            .to_dtype(DType::F32)
            .unwrap()
    };

    let input_embeds = f32t("input_embeds"); // [1, 126, 896]
    let teacher_embeds = f32t("teacher_embeds"); // [1, 227, 896]
    let step_logits = f32t("step_logits"); // [32, 6761]
    let text_token = ids(r.get("text_token").expect("reference has text_token"));
    let prompt_speech_token = ids(r.get("prompt_speech_token").expect("has prompt_speech_token"));
    let t0 = ids(r.get("t0").expect("reference has t0"))[0] as usize;

    // ---- Check 1: input-embedding assembly is exact (pure lookups + concat). ----
    let built = lm
        .build_lm_input(&text_token, &prompt_speech_token)
        .expect("build_lm_input");
    assert_eq!(
        built.dims(),
        input_embeds.dims(),
        "input-embeds shape mismatch: got {:?} vs ref {:?}",
        built.dims(),
        input_embeds.dims()
    );
    let d_in = max_abs_diff(&built, &input_embeds);
    eprintln!("input_embeds  max-abs-diff = {d_in:.3e}  (expect 0)");
    assert!(
        d_in <= 1e-6,
        "input-embeds assembly diff {d_in:.3e} is not bit-exact (sos/task/text/speech \
         concat must match the reference exactly)"
    );

    // ---- Check 2: teacher-forced per-position logit parity. ----
    let n = step_logits.dim(0).unwrap(); // 32
    let got = lm
        .teacher_forced_logits(&teacher_embeds, t0, n)
        .expect("teacher_forced_logits"); // [n, 6761]
    assert_eq!(
        got.dims(),
        step_logits.dims(),
        "step-logits shape mismatch: got {:?} vs ref {:?}",
        got.dims(),
        step_logits.dims()
    );
    let d = max_abs_diff(&got, &step_logits);
    let got0 = got.narrow(0, 0, 1).unwrap().flatten_all().unwrap();
    let ref0 = step_logits.narrow(0, 0, 1).unwrap().flatten_all().unwrap();
    eprintln!(
        "step_logits   max-abs-diff = {d:.3e}   row0 argmax ref={} ours={}",
        argmax_row(&ref0),
        argmax_row(&got0)
    );
    assert!(
        d < 1e-3,
        "CV3 teacher-forced logits max-abs diff {d:.3e} exceeds the 1e-3 parity tolerance"
    );
}
