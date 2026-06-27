//! Multilingual / cross-lingual eval — runs the measured metric suite across more
//! than one language and aggregates, so cross-lingual clone quality is measurable.
//!
//! Cross-lingual = clone a voice in one language and synthesize text in *another*;
//! the zero-shot voice carries across. The suite below pairs the on-box Chinese
//! reference (`zero_shot_prompt.wav`) with a Chinese target (native zh) **and** an
//! English target (zh voice → en text, the cross-lingual case). An optional third
//! native-English case is wired behind its own env vars for when an English
//! reference clip + transcript are supplied (e.g. `asset/cross_lingual_prompt.wav`).
//!
//! Env-gated on the on-box weights + a reference WAV; skips cleanly in CI.
//!
//!   SYRINX_LM_WEIGHTS=… SYRINX_SPK_WEIGHTS=… SYRINX_FLOW_WEIGHTS=… \
//!   SYRINX_HIFT_WEIGHTS=… SYRINX_TOK_JSON=… SYRINX_STOK_ONNX=… \
//!   SYRINX_EVAL_REF_WAV=<zh_ref.wav> \
//!   [SYRINX_WER_HELPER="micromamba run -n syrinx python scripts/eval_wer.py"] \
//!   [SYRINX_EVAL_REF_WAV_EN=<en_ref.wav> SYRINX_EVAL_PROMPT_TEXT_EN="<transcript>"] \
//!   [SYRINX_SYNTH_MAXSTEPS=N] \
//!   cargo test --features real --release --test real_eval_multilingual -- --nocapture
#![cfg(feature = "real")]

use std::path::Path;

use syrinx_eval::metrics::{aggregate, evaluate_suite, EvalCase, Metrics};
use syrinx_serve::synth::{SynthConfig, Synthesizer};
use syrinx_serve::wavio;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

fn fmt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4}")).unwrap_or_else(|| "null".to_string())
}

/// Assert every *present* metric is finite and within its valid range; `None` is an
/// honest "not measured" and is allowed.
fn assert_metrics_in_range(label: &str, m: &Metrics) {
    if let Some(s) = m.sim_o {
        assert!(
            s.is_finite() && (-1.0..=1.0).contains(&s),
            "{label}: sim_o out of [-1,1]: {s}"
        );
    }
    if let Some(w) = m.wer {
        assert!(
            w.is_finite() && (0.0..=2.0).contains(&w),
            "{label}: wer out of [0,2]: {w}"
        );
    }
    if let Some(mos) = m.mos_proxy {
        assert!(
            mos.is_finite() && (1.0..=5.0).contains(&mos),
            "{label}: mos_proxy out of [1,5]: {mos}"
        );
    }
    if let Some(t) = m.ttfb_ms {
        assert!(t.is_finite() && t > 0.0, "{label}: ttfb_ms must be > 0: {t}");
    }
    if let Some(r) = m.rtf {
        assert!(r.is_finite() && r > 0.0, "{label}: rtf must be > 0: {r}");
    }
}

#[test]
fn real_eval_multilingual_suite() {
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
            eprintln!("skipping real_eval_multilingual: set SYRINX_*_WEIGHTS + SYRINX_TOK_JSON + SYRINX_STOK_ONNX");
            return;
        }
    };
    let zh_ref = match env("SYRINX_EVAL_REF_WAV") {
        Some(p) => p,
        None => {
            eprintln!("skipping real_eval_multilingual: set SYRINX_EVAL_REF_WAV to the zh reference voice WAV");
            return;
        }
    };

    let mut synth = if env("SYRINX_QUANT").is_some() {
        eprintln!("real_eval_multilingual: using the int4-quantized LM (load_quantized)");
        Synthesizer::load_quantized(&cfg).expect("load quantized synthesizer")
    } else {
        Synthesizer::load(&cfg).expect("load synthesizer")
    };
    let max_steps = env("SYRINX_SYNTH_MAXSTEPS").and_then(|s| s.parse::<usize>().ok());

    // The Chinese reference voice (zero_shot_prompt.wav) + its known transcript.
    let (zh16, zh24) = wavio::read_ref_wav(Path::new(&zh_ref)).expect("read zh reference WAV");
    const ZH_PROMPT: &str = "希望你以后能够做的比我还好呦。";

    // Cases 1 & 2 share the zh reference voice; case 2 is cross-lingual (zh voice
    // synthesizing English text), so its WER ASR runs in English.
    let mut cases: Vec<EvalCase> = vec![
        EvalCase {
            lang: "zh",
            text: "收到好友从远方寄来的生日礼物。",
            prompt_text: ZH_PROMPT,
            ref_wav_16k: &zh16,
            ref_wav_24k: &zh24,
        },
        EvalCase {
            lang: "en",
            text: "The quick brown fox jumps over the lazy dog near the river bank.",
            prompt_text: ZH_PROMPT,
            ref_wav_16k: &zh16,
            ref_wav_24k: &zh24,
        },
    ];

    // Optional native-English case: an English reference clip + its transcript. The
    // box ships asset/cross_lingual_prompt.wav (an English voice) but not its
    // transcript, so this case is wired behind env vars rather than hardcoded — the
    // maintainer supplies the transcript. Keep the buffers alive past the call.
    let en_ref = (env("SYRINX_EVAL_REF_WAV_EN"), env("SYRINX_EVAL_PROMPT_TEXT_EN"));
    let en_bufs;
    if let (Some(p), Some(en_prompt)) = (&en_ref.0, &en_ref.1) {
        let (en16, en24) = wavio::read_ref_wav(Path::new(p)).expect("read en reference WAV");
        en_bufs = (en16, en24, en_prompt.clone());
        cases.push(EvalCase {
            lang: "en",
            text: "She sells sea shells by the sea shore on a bright sunny day.",
            prompt_text: &en_bufs.2,
            ref_wav_16k: &en_bufs.0,
            ref_wav_24k: &en_bufs.1,
        });
    }

    let n = cases.len();
    let results = evaluate_suite(&mut synth, &cases, max_steps);

    // Per-case + aggregate table.
    println!(
        "\n{:<40} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "case", "sim_o", "wer", "mos", "ttfb_ms", "rtf"
    );
    for (label, m) in &results {
        println!(
            "{:<40} {:>8} {:>8} {:>8} {:>10} {:>8}",
            label,
            fmt(m.sim_o),
            fmt(m.wer),
            fmt(m.mos_proxy),
            fmt(m.ttfb_ms),
            fmt(m.rtf)
        );
    }
    let agg = aggregate(&results);
    println!(
        "{:<40} {:>8} {:>8} {:>8} {:>10} {:>8}\n",
        "AGGREGATE (mean)",
        fmt(agg.sim_o),
        fmt(agg.wer),
        fmt(agg.mos_proxy),
        fmt(agg.ttfb_ms),
        fmt(agg.rtf)
    );

    // Every case must have produced a row (a synthesis failure shortens the vector).
    assert_eq!(results.len(), n, "a case failed to evaluate; see stderr above");

    // Every present metric is finite + in range, per case and in aggregate.
    for (label, m) in &results {
        assert_metrics_in_range(label, m);
        // SIM-o is always measured by evaluate; a real clone is positively correlated.
        let s = m.sim_o.expect("sim_o measured");
        assert!(s > 0.0, "{label}: sim_o non-positive for a voice clone: {s}");
    }
    assert_metrics_in_range("AGGREGATE", &agg);
    assert!(agg.sim_o.is_some(), "aggregate sim_o must be present");
}
