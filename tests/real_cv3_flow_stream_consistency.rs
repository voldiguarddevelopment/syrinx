//! CV3 streaming-flow self-consistency — the core proof that the chunked-causal DiT mask
//! makes finalized mel frames stable (no future leakage), which is what makes CV3 streaming
//! faithful. The CV3 analogue of `real_flow_stream_consistency.rs` (CV2).
//!
//! Claim: with the mask, `Cv3Flow::forward_zero_shot_streaming(token[..N-k])` and
//! `...(token[..N])` produce **identical** mel for the finalized frames (everything but the
//! trailing chunk) — removing future tokens can't change an already-finalized frame. The
//! non-causal batch `Cv3Flow::forward` does NOT have this property.
//!
//! This mirrors the on-box CV3 reference check in `cosyvoice/flow/flow.py`'s `__main__`
//! (`pred_gt` vs per-chunk `pred_chunk`, `static_chunk_size`), but as a Syrinx-internal
//! full-vs-truncated stability check that needs no per-chunk reference dump.
//!
//!   SYRINX_CV3_FLOW_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/flow_fp32.safetensors \
//!   SYRINX_CV3_E2E_REF=/root/parity-cv3/e2e/ref.safetensors \
//!   cargo test --features real --release --test real_cv3_flow_stream_consistency -- --nocapture
#![cfg(feature = "real")]

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use syrinx_acoustic::real_cv3::{Cv3Flow, Cv3StreamCfg};

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

fn get(r: &HashMap<String, Tensor>, k: &str) -> Tensor {
    r.get(k).unwrap_or_else(|| panic!("missing fixture key {k}")).clone()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64
}

const TOKEN_MEL_RATIO: usize = 2;

#[test]
fn cv3_streaming_flow_finalized_frames_are_stable() {
    // Uses the FLOW fixture (it carries the CFM noise `z`, which the e2e fixture omits)
    // — `z` is required to drive the streaming flow deterministically.
    let (weights, reference) = match (env("SYRINX_CV3_FLOW_WEIGHTS"), env("SYRINX_CV3_FLOW_REF")) {
        (Some(w), Some(r)) => (w, r),
        _ => {
            eprintln!(
                "SKIP real_cv3_flow_stream_consistency: set SYRINX_CV3_FLOW_WEIGHTS + SYRINX_CV3_FLOW_REF"
            );
            return;
        }
    };
    let dev = Device::Cpu;
    let flow = Cv3Flow::load(&weights, dev.clone()).expect("load CV3 flow weights");
    let r = candle_core::safetensors::load(&reference, &dev).expect("load CV3 e2e fixture");

    let prompt_token = get(&r, "prompt_speech_token").to_dtype(DType::I64).unwrap();
    let token = get(&r, "speech_token").to_dtype(DType::I64).unwrap();
    let prompt_feat = get(&r, "prompt_feat");
    let embedding = get(&r, "embedding");
    let z = get(&r, "z");

    let pt = prompt_token.dim(1).unwrap();
    let tg = token.dim(1).unwrap();

    // Drop the last k tokens; the earlier finalized frames must be unaffected.
    let k = 60usize.min(tg / 3).max(1);
    let tg_trunc = tg - k;
    let token_trunc = token.narrow(1, 0, tg_trunc).unwrap();
    let z_trunc = z
        .narrow(2, 0, TOKEN_MEL_RATIO * (pt + tg_trunc))
        .unwrap()
        .contiguous()
        .unwrap();

    eprintln!("[cv3-stream] prompt={pt} gen={tg} -> truncate {k} -> gen_trunc={tg_trunc}");

    let cfg = Cv3StreamCfg::cosyvoice3();

    // --- masked streaming flow: full vs truncated ---
    let mel_full = flow
        .forward_zero_shot_streaming(&prompt_token, &token, &prompt_feat, &embedding, &z, 10, cfg)
        .expect("streaming full");
    let mel_trunc = flow
        .forward_zero_shot_streaming(
            &prompt_token, &token_trunc, &prompt_feat, &embedding, &z_trunc, 10, cfg,
        )
        .expect("streaming trunc");

    // Finalized region = the truncated mel minus its trailing chunk.
    let trunc_len = mel_trunc.dim(2).unwrap();
    let finalized = trunc_len.saturating_sub(cfg.est_chunk);
    assert!(finalized > 0, "truncated mel too short to have a finalized region");
    let d_masked = max_abs_diff(
        &mel_full.narrow(2, 0, finalized).unwrap(),
        &mel_trunc.narrow(2, 0, finalized).unwrap(),
    );
    eprintln!("[cv3-stream] MASKED finalized-frame diff (n={finalized}) = {d_masked:.3e}");

    // --- sanity: the non-causal batch path does NOT keep finalized frames stable ---
    let nc_full = flow
        .forward(&prompt_token, &token, &prompt_feat, &embedding, &z, 10)
        .expect("non-causal full");
    let nc_trunc = flow
        .forward(&prompt_token, &token_trunc, &prompt_feat, &embedding, &z_trunc, 10)
        .expect("non-causal trunc");
    let d_nc = max_abs_diff(
        &nc_full.narrow(2, 0, finalized).unwrap(),
        &nc_trunc.narrow(2, 0, finalized).unwrap(),
    );
    eprintln!("[cv3-stream] NON-CAUSAL finalized-frame diff (n={finalized}) = {d_nc:.3e}");

    assert!(
        d_masked < 1e-2,
        "CV3 masked finalized frames not stable under truncation: {d_masked:.3e}"
    );
    assert!(
        d_nc > d_masked * 5.0,
        "CV3 non-causal should leak far more than masked: nc={d_nc:.3e} masked={d_masked:.3e}"
    );
    eprintln!(
        "[cv3-stream] PASS: the DiT chunk mask makes finalized frames stable ({:.0}x more stable than non-causal)",
        d_nc / d_masked.max(1e-9)
    );
}
