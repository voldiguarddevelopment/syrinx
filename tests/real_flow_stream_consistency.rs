//! Streaming-flow self-consistency — the core proof that the chunked-causal mask makes
//! finalized frames stable (no future leakage), which is what makes streaming faithful.
//!
//! Claim: with the mask, `forward_zero_shot_streaming(token[..N-k])` and
//! `forward_zero_shot_streaming(token[..N])` produce **identical** mel for the finalized
//! frames (everything but the trailing chunk) — removing future tokens can't change an
//! already-finalized frame. The non-causal `forward_zero_shot` does NOT have this property.
//!
//!   SYRINX_FLOW_WEIGHTS=/root/models/CosyVoice2-0.5B/flow_fp32.safetensors \
//!   SYRINX_E2E_REF=/root/parity/e2e/ref.safetensors \
//!   cargo test --features real --release --test real_flow_stream_consistency -- --nocapture
#![cfg(feature = "real")]

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use syrinx_acoustic::cv2::{Flow, StreamCfg};

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
fn streaming_flow_finalized_frames_are_stable() {
    let (weights, reference) = match (env("SYRINX_FLOW_WEIGHTS"), env("SYRINX_E2E_REF")) {
        (Some(w), Some(r)) => (w, r),
        _ => {
            eprintln!("SKIP real_flow_stream_consistency: set SYRINX_FLOW_WEIGHTS + SYRINX_E2E_REF");
            return;
        }
    };
    let dev = Device::Cpu;
    let flow = Flow::load(&weights, dev.clone()).expect("load flow weights");
    let r = candle_core::safetensors::load(&reference, &dev).expect("load e2e fixture");

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

    eprintln!("[stream] prompt={pt} gen={tg} -> truncate {k} -> gen_trunc={tg_trunc}");

    // --- masked streaming flow: full vs truncated ---
    let mel_full = flow
        .forward_zero_shot_streaming(&prompt_token, &token, &prompt_feat, &embedding, &z, 10, StreamCfg::cosyvoice2())
        .expect("streaming full");
    let mel_trunc = flow
        .forward_zero_shot_streaming(&prompt_token, &token_trunc, &prompt_feat, &embedding, &z_trunc, 10, StreamCfg::cosyvoice2())
        .expect("streaming trunc");

    // Finalized region = the truncated mel minus its trailing chunk.
    let cfg = StreamCfg::cosyvoice2();
    let trunc_len = mel_trunc.dim(2).unwrap();
    let finalized = trunc_len.saturating_sub(cfg.est_chunk);
    assert!(finalized > 0, "truncated mel too short to have a finalized region");
    let d_masked = max_abs_diff(
        &mel_full.narrow(2, 0, finalized).unwrap(),
        &mel_trunc.narrow(2, 0, finalized).unwrap(),
    );
    eprintln!("[stream] MASKED finalized-frame diff (n={finalized}) = {d_masked:.3e}");

    // --- sanity: the non-causal path does NOT keep finalized frames stable ---
    let nc_full = flow
        .forward_zero_shot(&prompt_token, &token, &prompt_feat, &embedding, &z, 10)
        .expect("non-causal full");
    let nc_trunc = flow
        .forward_zero_shot(&prompt_token, &token_trunc, &prompt_feat, &embedding, &z_trunc, 10)
        .expect("non-causal trunc");
    let d_nc = max_abs_diff(
        &nc_full.narrow(2, 0, finalized).unwrap(),
        &nc_trunc.narrow(2, 0, finalized).unwrap(),
    );
    eprintln!("[stream] NON-CAUSAL finalized-frame diff (n={finalized}) = {d_nc:.3e}");

    assert!(
        d_masked < 1e-2,
        "masked finalized frames not stable under truncation: {d_masked:.3e}"
    );
    assert!(
        d_nc > d_masked * 5.0,
        "non-causal should leak far more than masked: nc={d_nc:.3e} masked={d_masked:.3e}"
    );
    eprintln!(
        "[stream] PASS: the chunk mask makes finalized frames stable ({:.0}x more stable than non-causal)",
        d_nc / d_masked.max(1e-9)
    );
}
