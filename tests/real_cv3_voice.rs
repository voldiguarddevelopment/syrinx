//! On-box CV3 **voice layer** test: extract reusable [`Voice`]s from a real reference
//! clip, blend them, persist via a [`VoiceLibrary`], and synthesize from the cached
//! conditioning — proving the voice path reproduces the ref path and yields finite,
//! non-silent audio.
//!
//! Env-gated on the CV3 weights + the e2e reference dump (which carries the prompt wav);
//! it SKIPs cleanly off-box. Run on the model box with:
//!
//! ```bash
//! SYRINX_CV3_LM_WEIGHTS=…/llm_fp32.safetensors \
//! SYRINX_CV3_SPK_WEIGHTS=…/campplus_weights.safetensors \
//! SYRINX_CV3_FLOW_WEIGHTS=…/flow_fp32.safetensors \
//! SYRINX_CV3_HIFT_WEIGHTS=…/hift_fp32.safetensors \
//! SYRINX_CV3_TOK_JSON=…/tokenizer.json \
//! SYRINX_CV3_STOK_ONNX=…/speech_tokenizer_v3.onnx \
//! SYRINX_CV3_E2E_REF=…/e2e/ref.safetensors \
//!   cargo test --test real_cv3_voice -- --nocapture
//! ```

#![cfg(feature = "real")]

use std::path::Path;

use candle_core::{Device, Tensor};
use syrinx_serve::synth_cv3::{Cv3SynthConfig, Cv3SynthInputs, Cv3Synthesizer};
use syrinx_serve::voice::{Voice, VoiceLibrary};

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

/// Pull a 1-D f32 waveform out of the e2e ref dump.
fn wav(map: &std::collections::HashMap<String, Tensor>, key: &str) -> Vec<f32> {
    map.get(key)
        .unwrap_or_else(|| panic!("e2e ref missing `{key}`"))
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
fn cv3_voice_extract_blend_persist_and_synthesize() {
    let Some(env) = env() else {
        eprintln!(
            "SKIP real_cv3_voice: set SYRINX_CV3_LM_WEIGHTS, SYRINX_CV3_SPK_WEIGHTS, \
             SYRINX_CV3_FLOW_WEIGHTS, SYRINX_CV3_HIFT_WEIGHTS, SYRINX_CV3_TOK_JSON, \
             SYRINX_CV3_STOK_ONNX, SYRINX_CV3_E2E_REF to the on-box CV3 fixtures"
        );
        return;
    };

    let dev = Device::Cpu;
    let r = candle_core::safetensors::load(&env.e2e_ref, &dev).expect("load CV3 e2e ref");
    let ref_wav_16k = wav(&r, "prompt_wav_16k");
    let ref_wav_24k = match r.get("prompt_wav_24k") {
        Some(_) => wav(&r, "prompt_wav_24k"),
        None => syrinx_serve::wavio::resample(&ref_wav_16k, 16_000, 24_000),
    };

    let mut synth = Cv3Synthesizer::load(&env.cfg).expect("load CV3 sub-models");

    // ---- (1) extract a reusable Voice once via prompt_cond. ----
    let voice_a = Voice::from_reference(
        &mut synth,
        &ref_wav_16k,
        &ref_wav_24k,
        PROMPT_TEXT,
        "ref_a",
    )
    .expect("extract voice A")
    .with_source("e2e/ref:prompt_wav_16k");
    assert_eq!(voice_a.speaker_embedding.dims(), &[1, 192], "embedding shape");
    assert!(!voice_a.prompt_token.is_empty(), "prompt tokens extracted");

    // A second genuinely-extracted voice: a gain-scaled copy of the clip (a valid
    // waveform → its own CAM++ embedding / prompt tokens), so the blend is non-trivial.
    let ref16_b: Vec<f32> = ref_wav_16k.iter().map(|&x| x * 0.8).collect();
    let ref24_b: Vec<f32> = ref_wav_24k.iter().map(|&x| x * 0.8).collect();
    let voice_b = Voice::from_reference(&mut synth, &ref16_b, &ref24_b, PROMPT_TEXT, "ref_b")
        .expect("extract voice B");

    // ---- (2) blend timbres; result is a valid Voice carrying A's clip-tied cond. ----
    let blended = Voice::blend(&[(&voice_a, 0.7), (&voice_b, 0.3)])
        .expect("blend")
        .with_name("ab_blend");
    let be = blended.embedding_vec().unwrap();
    let bnorm = be.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    assert!((bnorm - 1.0).abs() < 1e-4, "blended embedding must be L2-normalized");
    assert_eq!(blended.prompt_token, voice_a.prompt_token, "blend carries base cond");

    // ---- (3) library round-trip on real tensors. ----
    let dir = std::env::temp_dir().join(format!("syrinx_cv3_voicelib_{}", std::process::id()));
    let lib = VoiceLibrary::open(&dir).expect("open lib");
    lib.save(&voice_a).expect("save A");
    lib.save(&blended).expect("save blend");
    assert_eq!(lib.list().unwrap(), vec!["ab_blend".to_string(), "ref_a".to_string()]);
    let reloaded = lib.load("ref_a").expect("reload A");
    let oe: Vec<f32> = voice_a.speaker_embedding.flatten_all().unwrap().to_vec1().unwrap();
    let ne: Vec<f32> = reloaded.speaker_embedding.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(oe, ne, "library embedding round-trip byte-exact");

    // ---- (4) the cached-conditioning path reproduces the ref path EXACTLY when the
    //          stochastic inputs are pinned/seeded identically. ----
    let pinned = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    let inputs = || Cv3SynthInputs {
        pinned_speech_token: Some(pinned.clone()),
        lm_seed: 7,
        ..Default::default()
    };
    let ref_audio = synth
        .synthesize(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k, &inputs())
        .expect("ref synthesize");
    let voice_audio = synth
        .synthesize_with_voice(TTS_TEXT, &voice_a, &inputs())
        .expect("voice synthesize");
    assert_eq!(
        ref_audio.len(),
        voice_audio.len(),
        "voice path length must equal ref path"
    );
    let d = ref_audio
        .iter()
        .zip(&voice_audio)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        d < 1e-6,
        "synthesize_with_voice must reproduce synthesize from the same clip (max diff {d:.3e})"
    );

    // ---- (5) blended voice synthesizes finite, non-silent audio. ----
    let cap = Cv3SynthInputs {
        lm_seed: 7,
        max_gen_steps: Some(40),
        ..Default::default()
    };
    let audio = synth
        .synthesize_with_voice(TTS_TEXT, &blended, &cap)
        .expect("blended synthesize");
    assert!(!audio.is_empty(), "blended audio non-empty");
    assert!(audio.iter().all(|x| x.is_finite()), "blended audio finite");
    let peak = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(peak > 1e-4, "blended audio must be non-silent (peak {peak:.3e})");
    eprintln!("blended voice: {} samples, peak {peak:.3}", audio.len());

    std::fs::remove_dir_all(&dir).ok();
}
