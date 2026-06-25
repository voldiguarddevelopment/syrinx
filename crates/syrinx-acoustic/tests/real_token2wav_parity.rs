//! End-to-end **token2wav** zero-shot parity (the real-weights track, buildable on
//! the model box). Gated on the `real` feature AND env vars pointing at the
//! converted fp32 checkpoints + the Python e2e reference dump (all too large to
//! vendor — they live on the model box). Skips cleanly when absent.
//!
//!   SYRINX_FLOW_WEIGHTS=/root/models/CosyVoice2-0.5B/flow_fp32.safetensors \
//!   SYRINX_HIFT_WEIGHTS=/root/parity/vocoder/hift_fp32.safetensors \
//!   SYRINX_E2E_REF=/root/parity/e2e/ref.safetensors \
//!   cargo test -p syrinx-acoustic --features real -- --nocapture
//!
//! Stages (each isolates a failure region): the prompt-conditioned encoder mu, the
//! prompt-conditioned full ODE mel (with the prompt prefix), the generated mel
//! (prefix dropped) — pinned at 1e-2 — then the final audio through the real HiFT
//! vocoder fed the pinned source STFT — pinned at 1e-3 relative.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_acoustic::real::{token2wav, Flow};
use syrinx_vocoder::real::HiftVocoder;

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

/// max |a-b| / (max|b| + eps): a scale-relative error for the audio waveform.
fn max_rel_diff(a: &Tensor, b: &Tensor) -> f32 {
    let d = max_abs_diff(a, b);
    let scale = b
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    d / (scale + 1e-12)
}

fn get(r: &std::collections::HashMap<String, Tensor>, k: &str) -> Tensor {
    r.get(k)
        .unwrap_or_else(|| panic!("reference missing `{k}`"))
        .to_dtype(DType::F32)
        .unwrap()
}

#[test]
fn real_token2wav_zero_shot_matches_reference() {
    let (fw, hw, refp) = match (
        std::env::var("SYRINX_FLOW_WEIGHTS").ok(),
        std::env::var("SYRINX_HIFT_WEIGHTS").ok(),
        std::env::var("SYRINX_E2E_REF").ok(),
    ) {
        (Some(f), Some(h), Some(r))
            if Path::new(&f).exists() && Path::new(&h).exists() && Path::new(&r).exists() =>
        {
            (f, h, r)
        }
        _ => {
            eprintln!(
                "SKIP real_token2wav parity: set SYRINX_FLOW_WEIGHTS + SYRINX_HIFT_WEIGHTS \
                 + SYRINX_E2E_REF to the on-disk fp32 fixtures"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let flow = Flow::load(&fw, dev.clone()).expect("load fp32 flow weights");
    let vocoder = HiftVocoder::load(&hw, dev.clone()).expect("load fp32 hift weights");
    let r = candle_core::safetensors::load(&refp, &dev).expect("load e2e reference");

    let prompt_token = get(&r, "prompt_speech_token").to_dtype(DType::I64).unwrap();
    let token = get(&r, "speech_token").to_dtype(DType::I64).unwrap();
    let prompt_feat = get(&r, "prompt_feat"); // [1, Mp, 80]
    let embedding = get(&r, "embedding"); // [1, 192]
    let z = get(&r, "z"); // [1, 80, total]
    let s_stft = get(&r, "s_stft"); // [1, 18, T]

    // ---- Stage: prompt-conditioned mu (encoder over prompt++gen tokens) ----
    let spk = flow.spk_proj(&embedding).expect("spk_proj");
    let tok_cat = Tensor::cat(&[&prompt_token, &token], 1).unwrap();
    let emb = flow.input_embedding(&tok_cat).expect("input_embedding");
    let enc = flow.encoder(&emb).expect("encoder");
    let exp_enc = get(&r, "encoder_out");
    assert_eq!(enc.dims(), exp_enc.dims(), "encoder_out shape");
    let d_enc = max_abs_diff(&enc, &exp_enc);
    eprintln!("encoder(prompt) max-abs-diff = {:.3e}", d_enc);

    let mu = flow
        .real_linear_pub(&enc, "encoder_proj.weight", Some("encoder_proj.bias"))
        .expect("encoder_proj");
    let exp_mu = get(&r, "mu");
    let d_mu = max_abs_diff(&mu, &exp_mu);
    eprintln!("mu(prompt)      max-abs-diff = {:.3e}", d_mu);

    // ---- Stage: full ODE mel WITH prompt cond (includes the prompt prefix) ----
    let mu_t = exp_mu.transpose(1, 2).unwrap().contiguous().unwrap(); // [1,80,total]
    let cond = get(&r, "cond"); // [1,80,total] prompt mel prepended
    let mel_full = flow
        .cfm_solve_with_cond(&mu_t, &spk, &cond, &z, 10)
        .expect("cfm_solve_with_cond");
    let exp_mel_full = get(&r, "mel_full");
    assert_eq!(mel_full.dims(), exp_mel_full.dims(), "mel_full shape");
    let d_mel_full = max_abs_diff(&mel_full, &exp_mel_full);
    eprintln!("mel_full        max-abs-diff = {:.3e}", d_mel_full);

    // ---- Stage: generated mel via the public zero-shot forward (prefix dropped) ----
    let mel = flow
        .forward_zero_shot(&prompt_token, &token, &prompt_feat, &embedding, &z, 10)
        .expect("forward_zero_shot");
    let exp_mel = get(&r, "mel");
    assert_eq!(mel.dims(), exp_mel.dims(), "mel shape");
    let d_mel = max_abs_diff(&mel, &exp_mel);
    eprintln!("mel(generated)  max-abs-diff = {:.3e}", d_mel);

    // ---- Stage: end-to-end token2wav audio (pinned z + source STFT) ----
    let audio = token2wav(
        &flow,
        &vocoder,
        &prompt_token,
        &token,
        &prompt_feat,
        &embedding,
        &z,
        &s_stft,
        10,
    )
    .expect("token2wav");
    let exp_audio = get(&r, "audio");
    assert_eq!(audio.dims(), exp_audio.dims(), "audio shape");
    let d_audio_abs = max_abs_diff(&audio, &exp_audio);
    let d_audio_rel = max_rel_diff(&audio, &exp_audio);
    eprintln!(
        "audio           max-abs-diff = {:.3e}  max-rel-diff = {:.3e}",
        d_audio_abs, d_audio_rel
    );

    assert!(d_enc < 1e-3, "encoder(prompt) max-abs diff {d_enc:.3e} exceeds 1e-3");
    assert!(
        d_mel < 1e-2,
        "generated mel max-abs diff {d_mel:.3e} exceeds the 1e-2 parity tolerance"
    );
    assert!(
        d_audio_rel < 1e-3,
        "audio max-rel diff {d_audio_rel:.3e} exceeds the 1e-3 relative parity tolerance"
    );
}
