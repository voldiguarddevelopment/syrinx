//! Real **CosyVoice3** LM **int4 (Q4_0) quantization** forward check — the CV3 twin of
//! `tests/real_lm_quant.rs` (which does the same for the CV2 LM).
//!
//! Loads the same converted CV3 fp32 checkpoint twice: once as the fp32 parity LM
//! ([`Cv3Lm::load`]) and once as the int4 quantized LM ([`Cv3Lm::load_quantized`], which
//! quantizes the Qwen2 body's big linears + the bias-free `llm_decoder` head to GGML
//! `Q4_0` and the embedding tables via per-row dequant-on-gather). Then it:
//!
//!   (a) builds the input embeds ONCE from the fp32 model (`build_lm_input`) so the
//!       comparison isolates the int4 weight effect (not the quantized embed), feeds the
//!       SAME tensor through both forwards, and asserts every logit is FINITE, and
//!   (b) measures the int4-vs-fp32 per-position top-1 (argmax) agreement over a short
//!       prompt and asserts it is reasonable (>= 60% — int4 is NOT argmax-exact; the real
//!       quality signal is the on-box SIM-o eval, not a tight logit bound here).
//!
//! Gated on the `real` feature AND `SYRINX_CV3_LM_WEIGHTS` pointing at the on-disk fp32
//! CV3 checkpoint (too large to vendor — lives on the GPU box). Skips cleanly when absent,
//! mirroring the device-bound task recipe.
//!
//!   SYRINX_CV3_LM_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/llm_fp32.safetensors \
//!   cargo test --features real --test real_cv3_quant_parity -- --nocapture

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};
use std::path::Path;
use syrinx_lm::cv3::Cv3Lm;

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
fn cv3_int4_lm_finite_and_argmax_agreement() {
    let weights = match std::env::var("SYRINX_CV3_LM_WEIGHTS").ok() {
        Some(w) if Path::new(&w).exists() => w,
        _ => {
            eprintln!(
                "SKIP real_cv3_quant_parity: set SYRINX_CV3_LM_WEIGHTS to the on-disk fp32 \
                 CosyVoice3 LM checkpoint (llm_fp32.safetensors)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let lm_f = Cv3Lm::load(&weights, dev.clone()).expect("load CV3 fp32 LM");
    let lm_q = Cv3Lm::load_quantized(&weights, dev.clone()).expect("load CV3 int4 LM");

    // Build the input embeds ONCE from the fp32 model and feed the SAME tensor to both
    // forwards, so the comparison isolates the int4 weight effect. A short in-vocab text
    // prompt with an empty prompt-speech prefix.
    let text_token: Vec<u32> = (10u32..42u32).collect(); // 32 ids
    let embeds = lm_f.build_lm_input(&text_token, &[]).expect("build CV3 lm input");

    let logits_f = lm_f.forward_logits(&embeds).expect("CV3 fp32 forward");
    let logits_q = lm_q.forward_logits(&embeds).expect("CV3 int4 forward");
    assert_eq!(logits_f.dims(), logits_q.dims(), "CV3 logit shape mismatch");
    assert!(all_finite(&logits_f), "CV3 fp32 logits not all finite");
    assert!(all_finite(&logits_q), "CV3 int4 logits not all finite");

    let am_f = argmax_per_pos(&logits_f);
    let am_q = argmax_per_pos(&logits_q);
    let agree = am_f.iter().zip(&am_q).filter(|(a, b)| a == b).count();
    let frac = agree as f64 / am_f.len() as f64;
    eprintln!(
        "CV3 argmax agreement int4-vs-fp32 = {agree}/{} = {:.1}%",
        am_f.len(),
        100.0 * frac
    );
    assert!(
        frac >= 0.60,
        "CV3 int4-vs-fp32 top-1 agreement {:.1}% below the 60% sanity floor",
        100.0 * frac
    );
}
