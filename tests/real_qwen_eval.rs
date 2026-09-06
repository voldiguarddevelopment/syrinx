//! The `syrinx-eval` Qwen hookup, on real weights.
//!
//! `docs/backends/QWEN_PORT_STATUS.md` recorded "no `syrinx-eval` hookup" as the last
//! unwired piece of the Qwen line. `syrinx_eval::qwen` closes it: it drives the serve
//! Qwen engine and scores the result with the in-tree Whisper, with no Python anywhere in
//! the loop — C4.1 requires own-runtime models only, and `metrics`'s CosyVoice-era
//! functions still shell out to `scripts/eval_wer.py`.
//!
//! Three things are worth asserting on real weights rather than assuming:
//!
//! 1. **It scores.** A plain render comes back intelligible with a finite RTF.
//! 2. **The WER reference is the PLAN's clean text, not the caller's input.** This is the
//!    subtle one. `[whisper] Come closer...` is planned down to `Come closer...` plus an
//!    instruction; scoring against the raw input would charge the engine for markup it is
//!    forbidden to speak, so a perfectly correct cued render would look broken. The cued
//!    case here must score as well as the plain one — if it does not, the reference is
//!    being taken from the wrong place.
//! 3. **Activation refuses rather than inventing a number.** `QwenRequest` carries no
//!    seed and the engine renders with `DriveParams::default()`, so every call is
//!    bit-identical. A permutation test over identical groups calls ANY difference
//!    significant, which is exactly the naive comparison `syrinx_eval::acoustic` exists to
//!    reject. The hookup detects that and errors; this pins the refusal so nobody
//!    "fixes" it into a silent number later.
//!
//! Gated on `SYRINX_QWEN_CV_DIR`, `SYRINX_QWEN_TOK_DIR` and `SYRINX_STT_MODEL_DIR`;
//! skips cleanly without them.

#![cfg(feature = "real")]

use candle_core::Device;

use syrinx_cue::BackendId;
use syrinx_eval::qwen::{evaluate_one, measure_activation, QwenCase};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};
use syrinx_stt::Stt;

const PLAIN: &str = "Come closer, I have something to tell you.";
const CUED: &str = "[whisper] Come closer, I have something to tell you.";

/// Measured, not guessed: the plain render below scores 0.000 against whisper-base on this
/// box. 0.35 leaves room for a small model on synthetic speech while still failing loudly
/// if the engine emits the wrong words — or speaks cue markup, which shows up here as
/// inserted tokens.
const MAX_WER: f64 = 0.35;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

#[test]
fn qwen_eval_scores_real_renders_natively() {
    let (Some(dir), Some(tok_dir), Some(stt_dir)) = (
        env_path("SYRINX_QWEN_CV_DIR"),
        env_path("SYRINX_QWEN_TOK_DIR"),
        env_path("SYRINX_STT_MODEL_DIR"),
    ) else {
        eprintln!(
            "SKIP real_qwen_eval: needs SYRINX_QWEN_CV_DIR, SYRINX_QWEN_TOK_DIR and \
             SYRINX_STT_MODEL_DIR"
        );
        return;
    };

    // CPU/f32. The scoring here is behavioural rather than numerical, but the board builds
    // without the cuda feature and a checkpoint on CPU is the portable path.
    let dev = Device::Cpu;
    let backend = BackendId::Qwen17bCustomVoice;
    let engine = QwenModelEngine::load(
        backend,
        &dir,
        &tok_dir,
        dev.clone(),
        QwenVoice::Preset("serena".into()),
    )
    .expect("load the Qwen engine");
    let oracle = Stt::load(&stt_dir, Device::Cpu).expect("load Whisper");

    // 1. A plain render scores.
    let plain = evaluate_one(&engine, backend, &oracle, QwenCase { id: "plain", text: PLAIN, lang: "en" })
        .expect("evaluate plain");
    eprintln!(
        "[qwen-eval] plain:  wer {:.3}  rtf {:.2}  {:.2}s  {} segment(s)  {:?}",
        plain.wer, plain.rtf, plain.audio_secs, plain.segments, plain.transcript
    );
    assert!(plain.wer.is_finite() && plain.rtf.is_finite(), "non-finite metrics");
    assert!(plain.rtf > 0.0, "rtf must be positive, got {}", plain.rtf);
    assert!(plain.audio_secs > 0.5, "implausibly short render: {}s", plain.audio_secs);
    assert!(plain.wer <= MAX_WER, "plain render unintelligible: WER {}", plain.wer);
    assert_eq!(plain.segments, 1, "a cue-free line must not split");

    // 2. The cued render is scored against the CLEAN text, so it scores as well as the
    //    plain one. Scoring against the raw input would penalise it for the markup it is
    //    required NOT to speak.
    let cued = evaluate_one(&engine, backend, &oracle, QwenCase { id: "cued", text: CUED, lang: "en" })
        .expect("evaluate cued");
    eprintln!(
        "[qwen-eval] cued:   wer {:.3}  rtf {:.2}  {:.2}s  instruct {:?}  {:?}",
        cued.wer, cued.rtf, cued.audio_secs, cued.instruct, cued.transcript
    );
    assert!(
        cued.wer <= MAX_WER,
        "cued render scored {} — if the plain case passed, the WER reference is being \
         taken from the raw input instead of the plan's clean text",
        cued.wer
    );
    assert!(
        cued.instruct.is_some(),
        "1.7B-CustomVoice honours instructions, so the cue must have produced one"
    );

    // 3. Activation now MEASURES, because the engine can vary its draw.
    //
    // This assertion used to require the opposite — that `measure_activation` refuse —
    // and its panic message said what to do when a seed arrived: "delete this assertion
    // and gate the real numbers instead". Per-render seed control landed in d1d0ceb, so
    // that is what this now does. The refusal path is still covered, by the engine-side
    // `honors_seed` contract and its unit tests.
    let n = 4; // the minimum at which the exact permutation test can reject at alpha=0.05
    let (m, detail) = measure_activation(
        &engine,
        backend,
        &oracle,
        QwenCase { id: "act", text: CUED, lang: "en" },
        n,
        0.05,
    )
    .expect("engine honours seeds, so activation is measurable");

    eprintln!("[qwen-eval] activation: activated={} {}", m.activated, detail);
    eprintln!("[qwen-eval]   wer cued {:.3} vs plain {:.3}", m.wer, m.baseline_wer);

    // What is asserted is that the measurement is SOUND, not what it concluded. Whether
    // this checkpoint moves for this cue is a finding, and turning a finding into a
    // required outcome is how a gate starts lying.
    assert_eq!(m.case_id, "act");
    assert_eq!(m.backend, backend.as_str());
    assert!(m.wer.is_finite() && m.baseline_wer.is_finite(), "non-finite WER");
    assert!(
        detail.contains("p="),
        "the detail line must carry the p-value that justifies the verdict: {detail}"
    );
    // A cue that "activates" only by destroying intelligibility is not a success.
    assert!(
        m.wer <= MAX_WER,
        "cued render unintelligible (WER {}), so any activation verdict is worthless",
        m.wer
    );
}
