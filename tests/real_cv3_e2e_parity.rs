//! End-to-end **CosyVoice3** synthesizer parity + functional smoke (the CV3 capstone).
//!
//! Wires the four parity-verified CV3 component ports — the v3 speech tokenizer, the
//! `Cv3Lm`, the `Cv3Flow` DiT mel decoder, and the `Cv3Hift` vocoder — plus the reused
//! CV2 frontend pieces (Qwen BPE tokenizer, kaldi-fbank + CAM++ speaker, 24 kHz matcha
//! prompt-mel) into one `text + reference voice -> 24 kHz audio` pipeline
//! (`syrinx_serve::synth_cv3::Cv3Synthesizer`) and checks it three ways.
//!
//! Gated on the `real` feature AND env vars pointing at the on-box fp32 CV3 weights +
//! assets + the coherent e2e reference dump. Skips cleanly when any is absent
//! (device-bound recipe; the maintainer runs it on the model box):
//!
//!   SYRINX_CV3_LM_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/llm_fp32.safetensors \
//!   SYRINX_CV3_SPK_WEIGHTS=/root/parity/speaker/campplus_weights.safetensors \
//!   SYRINX_CV3_FLOW_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/flow_fp32.safetensors \
//!   SYRINX_CV3_HIFT_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/hift_fp32.safetensors \
//!   SYRINX_CV3_TOK_JSON=/root/parity/frontend/tokenizer.json \
//!   SYRINX_CV3_STOK_ONNX=/root/models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx \
//!   SYRINX_CV3_E2E_REF=/root/parity-cv3/e2e/ref.safetensors \
//!   cargo test --features real --test real_cv3_e2e_parity -- --nocapture
//!
//! The e2e ref (`/root/parity-cv3/e2e/ref.safetensors`) holds, for the standard
//! zero-shot prompt clip: `prompt_wav_16k` [1,N] f32 (16 kHz), the reference
//! `prompt_speech_token` (v3 ids), `prompt_feat` [1,Mp,80], `embedding` [1,192],
//! `speech_token` (the generated semantic tokens), `mel` [1,80,2*Tg], `audio`, and the
//! `tts_text_token`/`prompt_text_token`. If it also carries a frozen CFM noise (key
//! `z`) and/or a 24 kHz prompt (`prompt_wav_24k`), the tighter checks below use them.
//!
//! ## What it proves
//!
//! 1. **Frontend parity** — from `prompt_wav_16k` (and a 24 kHz prompt: the ref's
//!    `prompt_wav_24k` if present, else a windowed-sinc resample), the synthesizer's
//!    frontend reproduces the reference conditioning: `prompt_token` exact ids,
//!    `embedding` within 1e-4, `prompt_feat` within 1e-3.
//! 2. **Deterministic chain anchor** — feeding the reference `prompt_speech_token` +
//!    `speech_token` + `prompt_feat` + `embedding` (and the frozen `z`) through
//!    `Cv3Flow` reproduces the reference `mel` within 4e-3 (the DiT fp32 floor). The LM
//!    is verified separately (its logits); the HiFT separately (hift/ref) — so this
//!    isolates frontend->flow. Asserted only when the ref carries `z`; otherwise the
//!    mel is checked for shape + finiteness with a diagnostic (no false failure).
//! 3. **Full synthesize smoke** — `Cv3Synthesizer::synthesize` runs the REAL chain
//!    (live LM gen + flow + hift + a produced source) and yields finite, non-silent
//!    24 kHz audio of plausible length. Bit-parity vs the fixture `audio` is NOT
//!    expected (LM sampling + SineGen RNG); the deterministic anchor is the parity signal.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use syrinx_serve::synth_cv3::{Cv3SynthConfig, Cv3SynthInputs, Cv3Synthesizer};

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

fn wav(r: &HashMap<String, Tensor>, k: &str) -> Vec<f32> {
    get(r, k).flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn ids_i64(t: &Tensor) -> Vec<i64> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
}

/// Representative zero-shot texts for the functional smoke. Bit-parity is not expected
/// on the live path, so any text exercises the chain; the reference voice clip drives
/// the clone. (The exact CV3 dump strings are not embedded in the ref — only their
/// token ids are — so the smoke uses these representative strings.)
const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
const TTS_TEXT: &str = "收到好友从远方寄来的生日礼物。";

struct Env {
    cfg: Cv3SynthConfig,
    e2e_ref: String,
}

fn env() -> Option<Env> {
    let v = |k: &str| std::env::var(k).ok().filter(|p| Path::new(p).exists());
    Some(Env {
        cfg: Cv3SynthConfig {
            lm_weights: v("SYRINX_CV3_LM_WEIGHTS")?,
            spk_weights: v("SYRINX_CV3_SPK_WEIGHTS")?,
            flow_weights: v("SYRINX_CV3_FLOW_WEIGHTS")?,
            hift_weights: v("SYRINX_CV3_HIFT_WEIGHTS")?,
            tokenizer_json: v("SYRINX_CV3_TOK_JSON")?,
            speech_tokenizer_onnx: v("SYRINX_CV3_STOK_ONNX")?,
        },
        e2e_ref: v("SYRINX_CV3_E2E_REF")?,
    })
}

#[test]
fn real_cv3_e2e_frontend_chain_and_smoke() {
    let env = match env() {
        Some(e) => e,
        None => {
            eprintln!(
                "SKIP real_cv3_e2e: set SYRINX_CV3_LM_WEIGHTS, SYRINX_CV3_SPK_WEIGHTS, \
                 SYRINX_CV3_FLOW_WEIGHTS, SYRINX_CV3_HIFT_WEIGHTS, SYRINX_CV3_TOK_JSON, \
                 SYRINX_CV3_STOK_ONNX, SYRINX_CV3_E2E_REF to the on-box CV3 fixtures"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let r = candle_core::safetensors::load(&env.e2e_ref, &dev).expect("load CV3 e2e ref");

    // The 16 kHz prompt the reference conditioning was derived from.
    let ref_wav_16k = wav(&r, "prompt_wav_16k");
    // The 24 kHz prompt for `prompt_feat`. CV3's `_extract_speech_feat` runs the matcha
    // mel on `load_wav(prompt, 24000)` — the EXACT reference-resampled 24 kHz prompt. A
    // local 16k->24k resample misaligns STFT frames and blows the log-mel diff (~5.8 was
    // observed on box from the resample path), so the dump's `prompt_wav_24k` MUST be used
    // when present; the resample is only a loud last-resort fallback. The eprintln below
    // records which path ran, so a box failure cleanly isolates resample-vs-mel.
    let ref_wav_24k = match r.get("prompt_wav_24k") {
        Some(_) => {
            let w = wav(&r, "prompt_wav_24k");
            eprintln!(
                "prompt_wav_24k: using EXACT reference 24 kHz prompt ({} samples, {:.2}s)",
                w.len(),
                w.len() as f32 / 24_000.0
            );
            w
        }
        None => {
            let w = syrinx_serve::wavio::resample(&ref_wav_16k, 16_000, 24_000);
            eprintln!(
                "WARN: e2e ref has no `prompt_wav_24k`; resampled the 16 kHz prompt 16k->24k \
                 (windowed-sinc, {} samples). prompt_feat will likely exceed 1e-3 due to STFT \
                 frame misalignment — add the exact `load_wav(prompt, 24000)` to the dump.",
                w.len()
            );
            w
        }
    };

    let mut synth = Cv3Synthesizer::load(&env.cfg).expect("load all CV3 sub-models");

    // ---- (1) frontend parity ----
    let cond = synth
        .prompt_cond(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
        .expect("prompt_cond");

    // prompt_token: exact ids (after the same %2 alignment the reference applies).
    let ref_prompt_token = ids_i64(&get(&r, "prompt_speech_token"));
    let our_prompt_token = ids_i64(&cond.prompt_token);
    assert_eq!(
        our_prompt_token, ref_prompt_token,
        "CV3 prompt speech-token ids must match the reference exactly\n ours={our_prompt_token:?}\n ref ={ref_prompt_token:?}"
    );
    eprintln!("prompt_token: EXACT match ({} ids)", our_prompt_token.len());

    // embedding (CAM++ x-vector): within 1e-4.
    let ref_emb = get(&r, "embedding"); // [1, 192]
    assert_eq!(cond.spk_embedding.dims(), ref_emb.dims(), "embedding shape");
    let d_emb = max_abs_diff(&cond.spk_embedding, &ref_emb);
    eprintln!("embedding   max-abs-diff = {d_emb:.3e}");
    assert!(d_emb < 1e-4, "CV3 embedding diff {d_emb:.3e} exceeds 1e-4");

    // prompt_feat: within 1e-3 ([1, Mp, 80] frame-major).
    let ref_pf = get(&r, "prompt_feat");
    assert_eq!(
        cond.prompt_feat.dims(),
        ref_pf.dims(),
        "prompt_feat shape: ours={:?} ref={:?}",
        cond.prompt_feat.dims(),
        ref_pf.dims()
    );
    let d_pf = max_abs_diff(&cond.prompt_feat, &ref_pf);
    eprintln!(
        "prompt_feat max-abs-diff = {d_pf:.3e}  (shape {:?})",
        cond.prompt_feat.dims()
    );
    assert!(d_pf < 1e-3, "CV3 prompt_feat diff {d_pf:.3e} exceeds 1e-3");

    // ---- (2) deterministic chain anchor: frontend -> flow -> mel ----
    // Feed the reference prompt/speech tokens + conditioning straight through Cv3Flow
    // (no LM sampling) and compare the mel to the reference mel.
    let ref_prompt_tok_t = get(&r, "prompt_speech_token").to_dtype(DType::I64).unwrap();
    let ref_speech_tok_t = get(&r, "speech_token").to_dtype(DType::I64).unwrap();
    let ref_mel = get(&r, "mel"); // [1, 80, 2*Tg]

    // The frozen CFM noise the reference mel was produced with (the flow's fixed
    // `rand_noise` slice). Use it when the dump carries it; else a zeros init (a valid
    // ODE start that will NOT reproduce the reference trajectory — diagnostic only).
    let total = 2 * (ref_prompt_tok_t.dim(1).unwrap() + ref_speech_tok_t.dim(1).unwrap());
    let (z, have_z) = match r.get("z") {
        Some(_) => (get(&r, "z"), true),
        None => (
            Tensor::zeros((1, 80, total), DType::F32, &dev).unwrap(),
            false,
        ),
    };

    let mel = synth
        .flow_from_reference_tokens(
            &ref_prompt_tok_t,
            &ref_speech_tok_t,
            &ref_pf,
            &ref_emb,
            &z,
            10,
        )
        .expect("flow_from_reference_tokens");
    assert_eq!(mel.dims(), ref_mel.dims(), "anchor mel shape");
    let mel_vals: Vec<f32> = mel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        mel_vals.iter().all(|x| x.is_finite()),
        "anchor mel has non-finite values"
    );
    let d_mel = max_abs_diff(&mel, &ref_mel);
    eprintln!("anchor mel  max-abs-diff = {d_mel:.3e}  (have_z={have_z})");
    if have_z {
        // 4e-3 = the DiT fp32 accumulation floor (see real_cv3_flow_parity).
        assert!(
            d_mel < 4e-3,
            "CV3 deterministic-chain mel diff {d_mel:.3e} exceeds the 4e-3 fp32 floor"
        );
    } else {
        eprintln!(
            "note: e2e ref has no frozen CFM noise `z`; the anchor ran with a zeros init, \
             which will not reproduce the reference trajectory. Checked shape + finiteness \
             only. Add `z` (the flow's rand_noise slice) to the dump for the 4e-3 anchor."
        );
    }

    // ---- (3) full synthesize smoke: live LM generation -> finite non-silent audio ----
    let live = synth
        .synthesize(
            TTS_TEXT,
            PROMPT_TEXT,
            &ref_wav_16k,
            &ref_wav_24k,
            // Cap live LM generation (KV-cached but still costly on CPU for a smoke);
            // overridable via SYRINX_CV3_SYNTH_MAXSTEPS.
            &Cv3SynthInputs {
                lm_seed: 1234,
                max_gen_steps: Some(
                    std::env::var("SYRINX_CV3_SYNTH_MAXSTEPS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(40),
                ),
                ..Default::default()
            },
        )
        .expect("live CV3 synthesize");
    assert!(!live.is_empty(), "live CV3 synthesis produced no audio");
    assert!(
        live.iter().all(|x| x.is_finite()),
        "live CV3 audio has non-finite samples"
    );
    let rms =
        (live.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / live.len() as f64).sqrt();
    let peak = live.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let dur = live.len() as f64 / 24_000.0;
    eprintln!(
        "live CV3 synth: {} samples ({dur:.2}s @24k)  rms={rms:.5}  peak={peak:.4}",
        live.len()
    );
    assert!(rms > 1e-4, "live CV3 audio is silent (rms {rms:.2e})");
    assert!(rms < 1.0, "live CV3 audio rms {rms:.3} implausibly hot");
    assert!(
        peak <= 0.99 + 1e-4,
        "live CV3 audio exceeds the 0.99 audio limit (peak {peak:.4})"
    );
    assert!(dur > 0.2, "live CV3 audio implausibly short ({dur:.3}s)");

    // optional: write the live wav for manual listening.
    if let Ok(path) = std::env::var("SYRINX_CV3_SYNTH_OUT") {
        syrinx_serve::wavio::write_wav_24k(&path, &live).expect("write wav");
        eprintln!("wrote live CV3 synth wav to {path}");
    }

    eprintln!("PASS: CV3 frontend parity, deterministic chain anchor, functional live synthesis.");
}
