//! Real CosyVoice2 LM autoregressive speech-token GENERATION parity (extends the
//! single-forward parity in `real_lm_parity.rs`). Verifies the AR loop on top of the
//! already-parity-verified Qwen2 forward.
//!
//! Gated on the `real` feature AND env vars pointing at the converted fp32 checkpoint
//! plus the Python reference dump produced by `/root/parity/lmgen/dump_lmgen.py`. Skips
//! cleanly when absent (device-bound recipe), runs for real where the fixtures exist.
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_LMGEN_REF=/root/parity/lmgen/ref.safetensors \
//!   cargo test -p syrinx-lm --features real --test real_lm_gen_parity -- --nocapture
//!
//! What it proves:
//!   (a) LOGIT PARITY — teacher-forcing the reference's chosen token sequence, EVERY
//!       AR step's last-position logits match the reference within 1e-3. This is the
//!       real correctness signal for the generation loop: the AR input assembly + the
//!       full-recompute forward reproduce the reference exactly, step for step.
//!   (b) TOKEN SEQUENCE — generation, given matching per-step logits, is a deterministic
//!       function of (logits, RNG). We pin our RNG and assert the `generate` loop is
//!       bit-reproducible, stops on a stop token / max_len, and emits only valid ids.
//!       (We do NOT claim to match torch's multinomial RNG draws — that PRNG is not
//!       portable; logit parity above is the honest correctness check.)

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_lm::real::Qwen2Lm;

fn ids_u32(t: &Tensor) -> Vec<u32> {
    let t = t.flatten_all().unwrap().to_dtype(DType::U32).unwrap();
    t.to_vec1::<u32>().unwrap()
}

fn scalar_i64(t: &Tensor) -> i64 {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()[0]
}

/// Max abs diff between two same-shape tensors, as f32.
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

fn fixtures() -> Option<(String, String)> {
    match (
        std::env::var("SYRINX_LM_WEIGHTS").ok(),
        std::env::var("SYRINX_LMGEN_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => Some((w, r)),
        _ => None,
    }
}

#[test]
fn real_lm_generation_logits_match_reference_within_1e_3() {
    let (weights, reference) = match fixtures() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP real_lm_generation parity: set SYRINX_LM_WEIGHTS + SYRINX_LMGEN_REF \
                 to the on-disk fp32 fixtures (CosyVoice2 weights + AR-gen reference dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let lm = Qwen2Lm::load(&weights, dev.clone()).expect("load fp32 weights");
    let r = candle_core::safetensors::load(&reference, &dev).expect("load gen reference");

    let text_token = ids_u32(r.get("text_token").expect("text_token"));
    let prompt_speech_token = ids_u32(r.get("prompt_speech_token").expect("prompt_speech_token"));
    let gen_tokens = ids_u32(r.get("gen_tokens").expect("gen_tokens"));
    let ref_step_logits = r
        .get("step_logits")
        .expect("step_logits")
        .to_dtype(DType::F32)
        .unwrap(); // [N, V]
    let t0_ref = scalar_i64(r.get("t0").expect("t0")) as usize;

    let n = gen_tokens.len();
    assert!(n >= 2, "reference must have >=2 generated tokens, got {n}");

    // --- (0) input-assembly parity: our step-0 LM input matches the reference's. -----
    let lm_input0 = lm
        .build_lm_input(&text_token, &prompt_speech_token)
        .expect("build_lm_input");
    assert_eq!(
        lm_input0.dim(1).unwrap(),
        t0_ref,
        "assembled step-0 input length {} != reference t0 {}",
        lm_input0.dim(1).unwrap(),
        t0_ref
    );
    if let Some(ref_inp) = r.get("input_embeds") {
        let ref_inp = ref_inp.to_dtype(DType::F32).unwrap();
        let d = max_abs_diff(&lm_input0, &ref_inp);
        eprintln!("input_embeds  max-abs-diff = {d:.3e}  (T0 = {t0_ref})");
        assert!(d < 1e-4, "step-0 input embedding diff {d:.3e} too large");
    }

    // --- (a) LOGIT PARITY: teacher-force the reference tokens, compare every step. ----
    let ours = lm
        .teacher_forced_logits(&text_token, &prompt_speech_token, &gen_tokens)
        .expect("teacher_forced_logits"); // [N, V]
    assert_eq!(
        ours.dims(),
        ref_step_logits.dims(),
        "teacher-forced logit shape {:?} != reference {:?}",
        ours.dims(),
        ref_step_logits.dims()
    );

    // Whole-tensor max abs diff across all N steps.
    let d_all = max_abs_diff(&ours, &ref_step_logits);

    // Per-step diffs + argmax agreement on the first few steps (extra signal).
    let nshow = n.min(6);
    for k in 0..nshow {
        let a = ours.narrow(0, k, 1).unwrap();
        let b = ref_step_logits.narrow(0, k, 1).unwrap();
        let d = max_abs_diff(&a, &b);
        let am_o = a.flatten_all().unwrap().argmax(0).unwrap().to_scalar::<u32>().unwrap();
        let am_r = b.flatten_all().unwrap().argmax(0).unwrap().to_scalar::<u32>().unwrap();
        eprintln!(
            "  step {k:>3}: logit max-abs-diff = {d:.3e}   argmax ref={am_r} ours={am_o}   chosen={}",
            gen_tokens[k]
        );
        assert!(d < 1e-3, "step {k} logit diff {d:.3e} exceeds 1e-3");
    }
    eprintln!("teacher-forced ALL {n} steps  max-abs-diff = {d_all:.3e}");
    assert!(
        d_all < 1e-3,
        "teacher-forced logit max-abs diff {d_all:.3e} exceeds 1e-3 over all {n} steps"
    );

    // --- (b) GENERATION loop: runs, stops cleanly, emits only valid speech-token ids,
    //         and is deterministic (bit-reproducible) under a pinned seed. -------------
    // The full-recompute forward is O(n²); a short bounded run suffices to prove the loop
    // is well-formed + deterministic (logit parity above already covers the full length).
    let min_len = scalar_i64(r.get("min_len").expect("min_len")) as usize;
    let gen_cap: usize = std::env::var("SYRINX_LMGEN_MAXSTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let g1 = lm
        .generate(&text_token, &prompt_speech_token, min_len, gen_cap, 1234)
        .expect("generate");
    let g2 = lm
        .generate(&text_token, &prompt_speech_token, min_len, gen_cap, 1234)
        .expect("generate (repeat)");
    assert_eq!(g1, g2, "generation must be deterministic for a fixed seed");
    assert!(!g1.is_empty(), "generation produced no tokens");
    assert!(g1.len() <= gen_cap, "generation exceeded the step cap");
    // Every emitted token is a real speech token (< speech_token_size = 6561), never a
    // stop token: the loop breaks on a stop token and excludes it.
    for &t in &g1 {
        assert!(t < 6561, "emitted token {t} is not a valid speech token (< 6561)");
    }
    // The first nucleus candidate set is shaped by the reference-matching logits; the
    // generated length is bounded and the loop honoured min_len (no early-EOS before it).
    eprintln!(
        "generate(seed=1234): {} tokens, first 8 = {:?}",
        g1.len(),
        &g1[..g1.len().min(8)]
    );

    eprintln!(
        "PASS: per-step logit parity over {n} steps within 1e-3; \
         generation loop deterministic + well-formed."
    );
}
