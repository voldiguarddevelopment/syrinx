//! On-box CV3 **inline emotion-tagging** test: `synthesize_tagged("[happy] … [sad] …")`
//! must produce finite, non-silent audio whose length equals the sum of the per-segment
//! renders minus the boundary cross-fade — proving the parse -> per-segment instruct
//! synth -> equal-power concat pipeline actually runs end-to-end on the real model.
//!
//! Env-gated on the CV3 weights + the e2e reference dump (which carries the prompt wav);
//! SKIPs cleanly off-box. The model-free parser / registry / cross-fade math is covered
//! separately by `tests/emotion_tags.rs`. Run on the model box with:
//!
//! ```bash
//! SYRINX_CV3_LM_WEIGHTS=…/llm_fp32.safetensors \
//! SYRINX_CV3_SPK_WEIGHTS=…/campplus_weights.safetensors \
//! SYRINX_CV3_FLOW_WEIGHTS=…/flow_fp32.safetensors \
//! SYRINX_CV3_HIFT_WEIGHTS=…/hift_fp32.safetensors \
//! SYRINX_CV3_TOK_JSON=…/tokenizer.json \
//! SYRINX_CV3_STOK_ONNX=…/speech_tokenizer_v3.onnx \
//! SYRINX_CV3_E2E_REF=…/e2e/ref.safetensors \
//!   cargo test --test real_cv3_emotion -- --nocapture
//! ```

#![cfg(feature = "real")]

use std::path::Path;

use candle_core::{Device, Tensor};
use syrinx_serve::emotion::{EmotionRegistry, DEFAULT_XFADE_SAMPLES};
use syrinx_serve::synth_cv3::{Cv3SynthConfig, Cv3SynthInputs, Cv3Synthesizer};

const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
const HAPPY_TEXT: &str = "今天天气真好呀";
const SAD_TEXT: &str = "可惜你要走了";

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

fn wav(map: &std::collections::HashMap<String, Tensor>, key: &str) -> Vec<f32> {
    map.get(key)
        .unwrap_or_else(|| panic!("e2e ref missing `{key}`"))
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn finite_nonsilent(audio: &[f32], label: &str) {
    assert!(!audio.is_empty(), "{label}: empty audio");
    assert!(
        audio.iter().all(|x| x.is_finite()),
        "{label}: non-finite samples"
    );
    let peak = audio.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(peak > 1e-3, "{label}: audio is silent (peak {peak:.3e})");
}

#[test]
fn cv3_synthesize_tagged_concatenates_emotion_segments() {
    let Some(env) = env() else {
        eprintln!(
            "SKIP real_cv3_emotion: set SYRINX_CV3_LM_WEIGHTS, SYRINX_CV3_SPK_WEIGHTS, \
             SYRINX_CV3_FLOW_WEIGHTS, SYRINX_CV3_HIFT_WEIGHTS, SYRINX_CV3_TOK_JSON, \
             SYRINX_CV3_STOK_ONNX, SYRINX_CV3_E2E_REF to the on-box CV3 fixtures"
        );
        return;
    };

    let dev = Device::Cpu;
    let r = candle_core::safetensors::load(&env.e2e_ref, &dev).expect("load CV3 e2e ref");
    let ref16 = wav(&r, "prompt_wav_16k");
    let ref24 = match r.get("prompt_wav_24k") {
        Some(_) => wav(&r, "prompt_wav_24k"),
        None => syrinx_serve::wavio::resample(&ref16, 16_000, 24_000),
    };

    let mut synth = Cv3Synthesizer::load(&env.cfg).expect("load CV3 sub-models");
    let registry = EmotionRegistry::default();
    let inputs = Cv3SynthInputs::default();

    // --- the tagged render: a happy span then a sad span. ---
    let tagged = format!("[happy] {HAPPY_TEXT} [sad] {SAD_TEXT}");
    let out = synth
        .synthesize_tagged(&tagged, PROMPT_TEXT, &ref16, &ref24, &registry, &inputs)
        .expect("synthesize_tagged");
    finite_nonsilent(&out, "tagged");

    // --- the two segments rendered independently (same fixed-seed instruct path). ---
    let happy = synth
        .synthesize_instruct(HAPPY_TEXT, registry.instruct("happy").unwrap(), &ref16, &ref24)
        .expect("instruct happy");
    let sad = synth
        .synthesize_instruct(SAD_TEXT, registry.instruct("sad").unwrap(), &ref16, &ref24)
        .expect("instruct sad");
    finite_nonsilent(&happy, "happy seg");
    finite_nonsilent(&sad, "sad seg");

    // The tagged waveform is the two segments joined with ONE equal-power cross-fade, so
    // its length is exactly len(happy)+len(sad) - overlap, overlap clamped to the shorter
    // side. `synthesize_instruct` is fixed-seed, so the per-segment renders match those the
    // tagged path made internally.
    let overlap = DEFAULT_XFADE_SAMPLES.min(happy.len()).min(sad.len());
    let expected = happy.len() + sad.len() - overlap;
    assert_eq!(
        out.len(),
        expected,
        "tagged length {} != sum {}+{} - xfade {}",
        out.len(),
        happy.len(),
        sad.len(),
        overlap
    );
}
