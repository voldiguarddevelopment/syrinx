//! End-to-end **synthesizer** parity + functional smoke (the `real`-feature
//! capstone). Wires the five parity-verified Syrinx components — frontend
//! tokenizer + feature extractor + speech-token ONNX, CAM++ speaker encoder, Qwen2
//! LM, flow decoder, HiFT vocoder — into one `text + reference voice -> 24 kHz
//! audio` pipeline (`syrinx_serve::synth::Synthesizer`) and checks it two ways.
//!
//! Gated on the `real` feature AND env vars pointing at the on-box fp32 weights +
//! assets + the coherent e2e reference dump (`/root/parity/e2e/ref.safetensors`,
//! from `dump_e2e.py`). Skips cleanly when any is absent (device-bound recipe).
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_SPK_WEIGHTS=/root/parity/speaker/campplus_weights.safetensors \
//!   SYRINX_FLOW_WEIGHTS=/root/models/CosyVoice2-0.5B/flow_fp32.safetensors \
//!   SYRINX_HIFT_WEIGHTS=/root/parity/vocoder/hift_fp32.safetensors \
//!   SYRINX_TOK_JSON=/root/parity/frontend/tokenizer.json \
//!   SYRINX_STOK_ONNX=/root/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
//!   SYRINX_FEAT_REF=/root/parity/frontend/feat_ref.safetensors \
//!   SYRINX_E2E_REF=/root/parity/e2e/ref.safetensors \
//!   cargo test --features real --test real_synth_e2e -- --nocapture
//!
//! ## What it proves
//!
//! 1. **Prompt-side parity** — from the prompt wav (the reference-resampled 16 kHz +
//!    24 kHz waveforms carried by `feat_ref`), the synthesizer's frontend half
//!    reproduces each conditioning value the e2e reference recorded:
//!    `prompt_token` exact, `spk` within 1e-3, `prompt_feat` within 1e-3.
//! 2. **Deterministic full-chain audio parity** — feeding the reference's *pinned*
//!    generated `speech_token` + CFM `z` + HiFT source `s_stft` through `token2wav`,
//!    the final 24 kHz audio matches the reference within ~1e-3 relative. (Live LM
//!    sampling RNG is not bit-portable — the per-step logit parity covers that — so
//!    the tokens are pinned here for a deterministic check.)
//! 3. **Functional live path** — `synthesize` with LIVE LM generation (its own
//!    pinned-PRNG sampling) returns finite, non-silent 24 kHz audio of a sane length.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use syrinx_serve::synth::{SynthConfig, SynthInputs, Synthesizer};

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

/// max |a-b| / (max|b| + eps): scale-relative waveform error.
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
    t.flatten_all().unwrap().to_dtype(DType::I64).unwrap().to_vec1::<i64>().unwrap()
}

/// CosyVoice2 e2e reference texts (must match `dump_e2e.py`).
const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
const TTS_TEXT: &str = "收到好友从远方寄来的生日礼物。";

struct Env {
    cfg: SynthConfig,
    feat_ref: String,
    e2e_ref: String,
}

fn env() -> Option<Env> {
    let v = |k: &str| std::env::var(k).ok().filter(|p| Path::new(p).exists());
    Some(Env {
        cfg: SynthConfig {
            lm_weights: v("SYRINX_LM_WEIGHTS")?,
            spk_weights: v("SYRINX_SPK_WEIGHTS")?,
            flow_weights: v("SYRINX_FLOW_WEIGHTS")?,
            hift_weights: v("SYRINX_HIFT_WEIGHTS")?,
            tokenizer_json: v("SYRINX_TOK_JSON")?,
            speech_tokenizer_onnx: v("SYRINX_STOK_ONNX")?,
        },
        feat_ref: v("SYRINX_FEAT_REF")?,
        e2e_ref: v("SYRINX_E2E_REF")?,
    })
}

#[test]
fn real_synth_e2e_deterministic_parity_and_functional() {
    let env = match env() {
        Some(e) => e,
        None => {
            eprintln!(
                "SKIP real_synth_e2e: set SYRINX_LM_WEIGHTS, SYRINX_SPK_WEIGHTS, \
                 SYRINX_FLOW_WEIGHTS, SYRINX_HIFT_WEIGHTS, SYRINX_TOK_JSON, \
                 SYRINX_STOK_ONNX, SYRINX_FEAT_REF, SYRINX_E2E_REF to the on-box fixtures"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let feat = candle_core::safetensors::load(&env.feat_ref, &dev).expect("load feat_ref");
    let r = candle_core::safetensors::load(&env.e2e_ref, &dev).expect("load e2e ref");

    // The reference-resampled prompt waveforms (so only the feature math is exercised).
    let ref_wav_16k = wav(&feat, "wav16_a");
    let ref_wav_24k = wav(&feat, "wav24_a");

    let mut synth = Synthesizer::load(&env.cfg).expect("load all sub-models");

    // ---- (1) prompt-side parity ----
    let cond = synth
        .prompt_cond(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
        .expect("prompt_cond");

    // prompt_token: exact match (after the same %2 alignment the reference applies).
    let ref_prompt_token = ids_i64(&get(&r, "prompt_speech_token").to_dtype(DType::I64).unwrap());
    let our_prompt_token = ids_i64(&cond.prompt_token);
    assert_eq!(
        our_prompt_token, ref_prompt_token,
        "prompt speech-token ids must match the reference exactly\n ours={our_prompt_token:?}\n ref ={ref_prompt_token:?}"
    );
    eprintln!("prompt_token: EXACT match ({} ids)", our_prompt_token.len());

    // spk embedding: within 1e-3 (the flow embedding the reference fed spk_proj).
    let ref_emb = get(&r, "embedding"); // [1, 192]
    assert_eq!(cond.spk_embedding.dims(), ref_emb.dims(), "spk embedding shape");
    let d_spk = max_abs_diff(&cond.spk_embedding, &ref_emb);
    eprintln!("spk embedding  max-abs-diff = {d_spk:.3e}");
    assert!(d_spk < 1e-3, "spk embedding diff {d_spk:.3e} exceeds 1e-3");

    // prompt_feat: within 1e-3 ([1, Mp, 80] frame-major).
    let ref_pf = get(&r, "prompt_feat");
    assert_eq!(cond.prompt_feat.dims(), ref_pf.dims(), "prompt_feat shape");
    let d_pf = max_abs_diff(&cond.prompt_feat, &ref_pf);
    eprintln!("prompt_feat    max-abs-diff = {d_pf:.3e}");
    assert!(d_pf < 1e-3, "prompt_feat diff {d_pf:.3e} exceeds 1e-3");

    // text_token: deterministic + non-empty (prompt_text ++ tts_text). The e2e ref
    // does not dump text_token (the LM stays on GPU there); we assert the tokenizer
    // is exact-reproducible across calls and the prompt/tts split is consistent.
    let cond2 = synth
        .prompt_cond(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
        .expect("prompt_cond repeat");
    assert_eq!(cond.text_token, cond2.text_token, "text_token must be deterministic");
    assert!(!cond.text_token.is_empty(), "text_token is empty");
    assert!(cond.prompt_text_len > 0 && cond.prompt_text_len < cond.text_token.len(),
        "prompt_text_len {} not a proper prefix of text_token len {}",
        cond.prompt_text_len, cond.text_token.len());
    eprintln!(
        "text_token: {} ids (prompt_text {} + tts_text {})",
        cond.text_token.len(),
        cond.prompt_text_len,
        cond.text_token.len() - cond.prompt_text_len
    );

    // ---- (2) deterministic full-chain audio parity (pinned tokens + z + source) ----
    let pinned_tokens = ids_i64(&get(&r, "speech_token").to_dtype(DType::I64).unwrap());
    let z = get(&r, "z"); // [1, 80, total]
    let s_stft = get(&r, "s_stft"); // [1, 18, T]
    let inputs = SynthInputs {
        pinned_speech_token: Some(pinned_tokens.clone()),
        z: Some(z),
        s_stft: Some(s_stft),
        ..Default::default()
    };

    let speech_token = Tensor::from_vec(pinned_tokens.clone(), (1, pinned_tokens.len()), &dev).unwrap();
    let audio = synth
        .token2wav(&cond, &speech_token, &inputs)
        .expect("token2wav (pinned)"); // [1, L]
    let ref_audio = get(&r, "audio");
    assert_eq!(audio.dims(), ref_audio.dims(), "audio shape");
    let d_audio_abs = max_abs_diff(&audio, &ref_audio);
    let d_audio_rel = max_rel_diff(&audio, &ref_audio);
    eprintln!(
        "deterministic audio  max-abs-diff = {d_audio_abs:.3e}  max-rel-diff = {d_audio_rel:.3e}  (L={})",
        audio.dims()[1]
    );
    assert!(
        d_audio_rel < 2e-3,
        "deterministic audio max-rel diff {d_audio_rel:.3e} exceeds the ~1e-3 relative parity bar"
    );

    // ---- (3) functional live path: live LM generation -> finite non-silent audio ----
    let live = synth
        .synthesize(
            TTS_TEXT,
            PROMPT_TEXT,
            &ref_wav_16k,
            &ref_wav_24k,
            // Cap the live generation: the LM has no KV cache (O(n²) per step), so the
            // real ratio (~text*20) is impractically slow on CPU for a smoke test. A cap
            // still exercises live sampling end to end and yields sane-length audio.
            // Overridable via SYRINX_SYNTH_MAXSTEPS.
            &SynthInputs {
                lm_seed: 1234,
                max_gen_steps: Some(
                    std::env::var("SYRINX_SYNTH_MAXSTEPS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(40),
                ),
                ..Default::default()
            },
        )
        .expect("live synthesize");
    assert!(!live.is_empty(), "live synthesis produced no audio");
    assert!(live.iter().all(|x| x.is_finite()), "live audio has non-finite samples");
    let rms = (live.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / live.len() as f64).sqrt();
    let peak = live.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let dur = live.len() as f64 / 24_000.0;
    eprintln!("live synth: {} samples ({dur:.2}s @24k)  rms={rms:.5}  peak={peak:.4}", live.len());
    assert!(rms > 1e-4, "live audio is silent (rms {rms:.2e})");
    assert!(peak <= 0.99 + 1e-4, "live audio exceeds the 0.99 audio limit (peak {peak:.4})");
    // sane length: at least ~0.2s of audio for a non-trivial utterance.
    assert!(dur > 0.2, "live audio implausibly short ({dur:.3}s)");

    // optional: write the live wav for manual listening.
    if let Ok(path) = std::env::var("SYRINX_SYNTH_OUT") {
        write_wav(&path, &live, 24_000);
        eprintln!("wrote live synth wav to {path}");
    }

    eprintln!("PASS: prompt-side parity, deterministic audio parity, functional live synthesis.");
}

/// Minimal 16-bit PCM WAV writer (mono) for manual listening.
fn write_wav(path: &str, samples: &[f32], sr: u32) {
    use std::io::Write;
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    let byte_rate = sr * 2;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sr.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &x in samples {
        let v = (x.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(&bytes);
    }
}
