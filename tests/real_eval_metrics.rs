//! Real eval-metrics test — runs the measured SIM-o / RTF / TTFB metrics through
//! the real CosyVoice2 `Synthesizer`. Env-gated on the on-box weights + a reference
//! WAV; skips cleanly in CI (no Candle fixtures present).
//!
//!   SYRINX_LM_WEIGHTS=… SYRINX_SPK_WEIGHTS=… SYRINX_FLOW_WEIGHTS=… \
//!   SYRINX_HIFT_WEIGHTS=… SYRINX_TOK_JSON=… SYRINX_STOK_ONNX=… \
//!   SYRINX_EVAL_REF_WAV=<ref.wav> [SYRINX_SYNTH_MAXSTEPS=N] \
//!   cargo test --features real --release --test real_eval_metrics -- --nocapture
#![cfg(feature = "real")]

use std::path::Path;

use syrinx_eval::real::{evaluate, EvalInput};
use syrinx_serve::synth::{SynthConfig, Synthesizer};
use syrinx_serve::wavio;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

#[test]
fn real_eval_metrics_are_measured() {
    let cfg = match (
        env("SYRINX_LM_WEIGHTS"),
        env("SYRINX_SPK_WEIGHTS"),
        env("SYRINX_FLOW_WEIGHTS"),
        env("SYRINX_HIFT_WEIGHTS"),
        env("SYRINX_TOK_JSON"),
        env("SYRINX_STOK_ONNX"),
    ) {
        (Some(lm), Some(spk), Some(flow), Some(hift), Some(tok), Some(stok)) => SynthConfig {
            lm_weights: lm,
            spk_weights: spk,
            flow_weights: flow,
            hift_weights: hift,
            tokenizer_json: tok,
            speech_tokenizer_onnx: stok,
        },
        _ => {
            eprintln!("skipping real_eval_metrics: set SYRINX_*_WEIGHTS + SYRINX_TOK_JSON + SYRINX_STOK_ONNX");
            return;
        }
    };
    let ref_wav = match env("SYRINX_EVAL_REF_WAV") {
        Some(p) => p,
        None => {
            eprintln!("skipping real_eval_metrics: set SYRINX_EVAL_REF_WAV to a reference voice WAV");
            return;
        }
    };

    let mut synth = Synthesizer::load(&cfg).expect("load synthesizer");
    let (r16, r24) = wavio::read_ref_wav(Path::new(&ref_wav)).expect("read reference WAV");
    let max_steps = env("SYRINX_SYNTH_MAXSTEPS").and_then(|s| s.parse::<usize>().ok());

    let input = EvalInput {
        text: "收到好友从远方寄来的生日礼物。",
        prompt_text: "希望你以后能够做的比我还好呦。",
        ref_wav_16k: &r16,
        ref_wav_24k: &r24,
    };
    let m = evaluate(&mut synth, &input, max_steps).expect("evaluate");
    println!("real eval metrics: {}", m.to_json());

    // SIM-o must be a real cosine in [-1, 1] (not the zero-norm fallback), and a
    // same-voice clone must be positively correlated with its reference.
    let sim_o = m.sim_o.expect("sim_o measured");
    assert!(
        sim_o.is_finite() && (-1.0..=1.0).contains(&sim_o) && sim_o != 0.0,
        "sim_o not a valid measured cosine: {sim_o}"
    );
    assert!(sim_o > 0.0, "sim_o non-positive for a clone of the same voice: {sim_o}");

    // RTF + TTFB are positive, finite wall-clock measurements.
    let rtf = m.rtf.expect("rtf measured");
    assert!(rtf.is_finite() && rtf > 0.0, "rtf must be a positive number: {rtf}");
    let ttfb = m.ttfb_ms.expect("ttfb_ms measured");
    assert!(ttfb.is_finite() && ttfb > 0.0, "ttfb_ms must be a positive number: {ttfb}");

    // The un-implemented metrics are honest nulls, never fake constants.
    assert!(m.wer.is_none(), "wer must be null until an ASR model is wired");
    assert!(m.mos_proxy.is_none(), "mos_proxy must be null until a MOS model is wired");
}
