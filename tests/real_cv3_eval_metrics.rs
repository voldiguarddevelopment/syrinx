//! Real CV3 eval-metrics test — runs the measured SIM-o / WER / MOS / RTF / TTFB
//! metrics through the real **CosyVoice3** `Cv3Synthesizer`. Env-gated on the on-box
//! CV3 weights + a reference WAV; skips cleanly in CI (no Candle fixtures present).
//!
//! Mirrors the CV2 `real_eval_metrics.rs`, with the CV3 synthesizer + `SYRINX_CV3_*`
//! weight env vars swapped in. The WER/MOS helper scripts and the SIM-o speaker path
//! are reused unchanged, so the maintainer sees the CV3 quality numbers in the same
//! `{sim_o,wer,mos_proxy,ttfb_ms,rtf}` JSON shape.
//!
//!   SYRINX_CV3_LM_WEIGHTS=…  SYRINX_CV3_SPK_WEIGHTS=…  SYRINX_CV3_FLOW_WEIGHTS=… \
//!   SYRINX_CV3_HIFT_WEIGHTS=… SYRINX_CV3_TOK_JSON=…    SYRINX_CV3_STOK_ONNX=… \
//!   SYRINX_EVAL_REF_WAV=<ref.wav> [SYRINX_SYNTH_MAXSTEPS=N] \
//!   [SYRINX_WER_HELPER="micromamba run -n syrinx python scripts/eval_wer.py"] \
//!   [SYRINX_MOS_HELPER="micromamba run -n syrinx python scripts/eval_mos.py"] \
//!   cargo test --features real --release --test real_cv3_eval_metrics -- --nocapture
#![cfg(feature = "real")]

use std::path::Path;

use syrinx_eval::real::{evaluate_cv3, EvalInput};
use syrinx_serve::synth_cv3::{Cv3SynthConfig, Cv3Synthesizer};
use syrinx_serve::wavio;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

#[test]
fn real_cv3_eval_metrics_are_measured() {
    let cfg = match (
        env("SYRINX_CV3_LM_WEIGHTS"),
        env("SYRINX_CV3_SPK_WEIGHTS"),
        env("SYRINX_CV3_FLOW_WEIGHTS"),
        env("SYRINX_CV3_HIFT_WEIGHTS"),
        env("SYRINX_CV3_TOK_JSON"),
        env("SYRINX_CV3_STOK_ONNX"),
    ) {
        (Some(lm), Some(spk), Some(flow), Some(hift), Some(tok), Some(stok)) => Cv3SynthConfig {
            lm_weights: lm,
            spk_weights: spk,
            flow_weights: flow,
            hift_weights: hift,
            tokenizer_json: tok,
            speech_tokenizer_onnx: stok,
        },
        _ => {
            eprintln!(
                "skipping real_cv3_eval_metrics: set SYRINX_CV3_*_WEIGHTS + SYRINX_CV3_TOK_JSON + SYRINX_CV3_STOK_ONNX"
            );
            return;
        }
    };
    let ref_wav = match env("SYRINX_EVAL_REF_WAV") {
        Some(p) => p,
        None => {
            eprintln!("skipping real_cv3_eval_metrics: set SYRINX_EVAL_REF_WAV to a reference voice WAV");
            return;
        }
    };

    let mut synth = Cv3Synthesizer::load(&cfg).expect("load CV3 synthesizer");
    let (r16, r24) = wavio::read_ref_wav(Path::new(&ref_wav)).expect("read reference WAV");
    let max_steps = env("SYRINX_SYNTH_MAXSTEPS").and_then(|s| s.parse::<usize>().ok());

    let input = EvalInput {
        text: "收到好友从远方寄来的生日礼物。",
        prompt_text: "希望你以后能够做的比我还好呦。",
        ref_wav_16k: &r16,
        ref_wav_24k: &r24,
    };
    let m = evaluate_cv3(&mut synth, &input, max_steps).expect("evaluate_cv3");
    println!("real cv3 eval metrics: {}", m.to_json());

    // SIM-o must be a real cosine in [-1, 1] (not the zero-norm fallback), and a
    // same-voice clone must be positively correlated with its reference.
    let sim_o = m.sim_o.expect("sim_o measured");
    assert!(
        sim_o.is_finite() && (-1.0..=1.0).contains(&sim_o) && sim_o != 0.0,
        "sim_o not a valid measured cosine: {sim_o}"
    );
    assert!(sim_o > 0.0, "sim_o non-positive for a clone of the same voice: {sim_o}");

    // RTF + TTFB are positive, finite wall-clock measurements (CV3 is batch-only, so
    // TTFB is the full synthesis wall-time — a real number, never a stub constant).
    let rtf = m.rtf.expect("rtf measured");
    assert!(rtf.is_finite() && rtf > 0.0, "rtf must be a positive number: {rtf}");
    let ttfb = m.ttfb_ms.expect("ttfb_ms measured");
    assert!(ttfb.is_finite() && ttfb > 0.0, "ttfb_ms must be a positive number: {ttfb}");

    // WER is measured only when the Whisper ASR helper is configured (SYRINX_WER_HELPER);
    // otherwise it is an honest null. A valid error rate sits in [0, 2].
    match (env("SYRINX_WER_HELPER"), m.wer) {
        (Some(_), Some(w)) => assert!(
            w.is_finite() && (0.0..=2.0).contains(&w),
            "wer out of range: {w}"
        ),
        (Some(_), None) => eprintln!("note: SYRINX_WER_HELPER set but the helper returned no rate"),
        (None, w) => assert!(w.is_none(), "wer must be null without SYRINX_WER_HELPER: {w:?}"),
    }
    // MOS-proxy is measured only when the UTMOS helper is configured (SYRINX_MOS_HELPER);
    // otherwise it is an honest null. A valid MOS sits in [1, 5].
    match (env("SYRINX_MOS_HELPER"), m.mos_proxy) {
        (Some(_), Some(mos)) => assert!(
            mos.is_finite() && (1.0..=5.0).contains(&mos),
            "mos_proxy out of [1,5]: {mos}"
        ),
        (Some(_), None) => eprintln!("note: SYRINX_MOS_HELPER set but the helper returned no score"),
        (None, m) => assert!(m.is_none(), "mos_proxy must be null without SYRINX_MOS_HELPER: {m:?}"),
    }
}
