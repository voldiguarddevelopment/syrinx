//! Real CosyVoice2 LM **int4 (Q4_0) quantization** check — the README size-goal track.
//!
//! Loads the same converted fp32 checkpoint twice: once as the fp32 parity LM
//! (`Qwen2Lm::load`) and once as the int4 quantized LM (`Qwen2Lm::load_quantized`,
//! which quantizes every layer's `q/k/v/o_proj` + `gate/up/down_proj` and the
//! `llm_decoder` head to GGML `Q4_0`, keeps the embed tables as an f16 lookup, and
//! the norms/biases f32). Then it:
//!
//!   (a) REPORTS the realized weight footprint of each (the int4 footprint vs the
//!       ~2449 MB fp32 LM — the size win),
//!   (b) feeds the **same** fp32-derived input embeddings through both forwards and
//!       asserts every logit is FINITE, and
//!   (c) measures the int4-vs-fp32 per-position top-1 (argmax) agreement over a short
//!       prompt and asserts it is reasonable (≥ 60% — int4 is NOT argmax-exact; the
//!       real quality signal is the on-box SIM-o eval, not a tight logit bound here).
//!
//! Gated on the `real` feature AND `SYRINX_LM_WEIGHTS` pointing at the on-disk fp32
//! checkpoint (too large to vendor — lives on the GPU box). Skips cleanly when absent,
//! mirroring the device-bound task recipe.
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   cargo test --features real --test real_lm_quant -- --nocapture

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};
use std::path::Path;
use syrinx_lm::cv2::Qwen2Lm;

/// fp32 reference footprint of the CosyVoice2 LM checkpoint (from the README budget).
const FP32_LM_MB: f64 = 2449.0;

/// All-finite check over an arbitrary-rank tensor.
fn all_finite(t: &Tensor) -> bool {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())
}

/// Per-position argmax over the vocab of a `[1, T, V]` logit tensor -> `Vec<u32>` of len T.
fn argmax_per_pos(logits: &Tensor) -> Vec<u32> {
    let t = logits.dim(1).unwrap();
    let mut out = Vec::with_capacity(t);
    for i in 0..t {
        let row = logits.narrow(1, i, 1).unwrap().flatten_all().unwrap();
        out.push(row.argmax(0).unwrap().to_scalar::<u32>().unwrap());
    }
    out
}

#[test]
fn int4_lm_footprint_and_argmax_agreement() {
    let weights = match std::env::var("SYRINX_LM_WEIGHTS").ok() {
        Some(w) if Path::new(&w).exists() => w,
        _ => {
            eprintln!(
                "SKIP real_lm_quant: set SYRINX_LM_WEIGHTS to the on-disk fp32 \
                 CosyVoice2 LM checkpoint (llm_fp32.safetensors)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let lm_f = Qwen2Lm::load(&weights, dev.clone()).expect("load fp32 LM");
    let lm_q = Qwen2Lm::load_quantized(&weights, dev.clone()).expect("load int4 LM");

    // --- (a) footprint report -------------------------------------------------
    let fp = lm_f.footprint();
    let qp = lm_q.footprint();
    eprintln!(
        "footprint  fp32 (this load) = {:.1} MB   int4 = {:.1} MB \
         ({} quantized weights; quant {:.1} MB + dense {:.1} MB)   README fp32 LM = {:.0} MB",
        fp.total_mb(),
        qp.total_mb(),
        qp.n_quantized,
        qp.quant_bytes as f64 / (1024.0 * 1024.0),
        qp.dense_bytes as f64 / (1024.0 * 1024.0),
        FP32_LM_MB,
    );
    // The int4 build must actually be smaller than the fp32 load and the README budget.
    assert!(qp.total_bytes() < fp.total_bytes(), "int4 not smaller than fp32 load");
    assert!(qp.total_mb() < FP32_LM_MB, "int4 footprint not below the fp32 budget");
    assert!(qp.n_quantized > 0, "no weights were quantized");

    // --- (b)+(c) finite logits + argmax agreement over a short prompt ---------
    // A short synthetic text-token prompt (small, in-vocab ids); empty prompt speech.
    // Build the input embeds ONCE from the fp32 model and feed the SAME tensor to both
    // forwards, so the comparison isolates the int4 weight effect (not the f16 embed).
    let text_token: Vec<u32> = (10u32..42u32).collect(); // 32 ids -> 34 positions
    let embeds = lm_f.build_lm_input(&text_token, &[]).expect("build lm input");

    let logits_f = lm_f.forward_logits(&embeds).expect("fp32 forward");
    let logits_q = lm_q.forward_logits(&embeds).expect("int4 forward");
    assert_eq!(logits_f.dims(), logits_q.dims(), "logit shape mismatch");
    assert!(all_finite(&logits_f), "fp32 logits not all finite");
    assert!(all_finite(&logits_q), "int4 logits not all finite");

    let am_f = argmax_per_pos(&logits_f);
    let am_q = argmax_per_pos(&logits_q);
    let agree = am_f.iter().zip(&am_q).filter(|(a, b)| a == b).count();
    let frac = agree as f64 / am_f.len() as f64;
    eprintln!(
        "argmax agreement int4-vs-fp32 = {agree}/{} = {:.1}%",
        am_f.len(),
        100.0 * frac
    );
    assert!(
        frac >= 0.60,
        "int4-vs-fp32 top-1 agreement {:.1}% below the 60% sanity floor",
        100.0 * frac
    );

    // Smoke: the quantized generation loop runs and emits only valid, in-range ids.
    let gen = lm_q
        .generate(&text_token, &[], 1, 8, 0)
        .expect("int4 generate");
    eprintln!("int4 generate produced {} tokens", gen.len());
    assert!(gen.iter().all(|&t| t < 6561), "generated id out of speech-token range");
}
