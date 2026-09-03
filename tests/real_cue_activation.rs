//! **The C4.2 certification run**: measure real cue activation on a real backend.
//!
//! `tests/cue_activation_gate.rs` proves the harness aggregates and gates correctly given
//! measurements. `tests/cue_activation_measure.rs` proves the `activated` flag means what
//! it claims. Neither produces a single real number — per ledger amendment A23, "no GPU run
//! has produced real activation measurements for the frozen set, and until it happens no
//! claim is made about real activation rates". This test is that run.
//!
//! ## What it does
//!
//! For every non-`neutral` case in the frozen cue set, on the Fish s2-pro backend:
//!
//!   1. The **cued** text is the author's string verbatim. s2-pro is [`Inline::Open`], so
//!      the lowering pass-through is byte-identical — what the author typed is exactly what
//!      the backend receives, and re-deriving it here would only risk disagreeing with
//!      `syrinx-cue`.
//!   2. The **control** text is `lower_full(...).text` — the same sentence with the cue
//!      stripped, whitespace-normalized. It is also the WER reference, because the cue must
//!      never be spoken.
//!   3. Both are rendered `n` times with distinct seeds. The dual-AR loop is bit-
//!      reproducible per seed, so the spread across seeds is the model's own sampling
//!      noise, measured rather than assumed.
//!   4. [`syrinx_eval::acoustic::activation_test`] decides `activated` by an exact
//!      permutation test against that noise; the native Whisper oracle (`syrinx-stt`, no
//!      Python at inference, as C4.1 requires) gives the WER of each render.
//!   5. [`evaluate_activation`] aggregates per kind and emits the JSON.
//!
//! ## What it asserts, and what it deliberately does not
//!
//! It asserts the run is **structurally sound**: every case measured, every number finite,
//! the JSON written. It does **not** assert the C4.2 thresholds by default, because those
//! thresholds have never been checked against reality — asserting them now would either
//! rubber-stamp whatever the model happens to do, or fail the board for a model-quality
//! reason nobody has yet agreed is a regression. Certifying them means looking at the first
//! real numbers and *then* deciding. Set `SYRINX_CUE_ACTIVATION_ENFORCE=1` to turn the
//! thresholds into assertions once that decision is made.
//!
//! It also cannot say whether `[sad]` sounds sad. That is perceptual, and per `CLAUDE.md`
//! perceptual judgements are never automated into a green here.
//!
//! ## Running it
//!
//! Long and GPU-bound, so it is opt-in on `SYRINX_CUE_ACTIVATION_OUT` and skips cleanly
//! otherwise — it must never fire during a routine board run:
//!
//! ```text
//! source scripts/test-all.env
//! SYRINX_CUE_ACTIVATION_OUT=.opt-reports/cue-activation.json \
//!   scripts/run-isolated.sh cargo test --features "real cuda" --release \
//!   --test real_cue_activation -- --nocapture
//! ```
//!
//! `SYRINX_CUE_ACTIVATION_N` sets renders per condition (default 4 — the smallest `n`
//! whose exact test can reject at α=0.05), `SYRINX_CUE_ACTIVATION_CASES` caps the case
//! count for a pilot, and `SYRINX_FISH_MAXFRAMES` caps generated frames per render.

#![cfg(feature = "real")]

use std::io::Write;
use std::time::Instant;

use candle_core::{Device, Tensor};

use syrinx_eval::acoustic::{activation_test, features, min_n_for_alpha, Features};
use syrinx_eval::activation::{evaluate_activation, CueCase, Measurement, Thresholds};
use syrinx_fish::common::audio as fish_audio;
use syrinx_fish::common::codec::RvqCodec;
use syrinx_fish::common::dualar::DriveParams;
use syrinx_fish::s2::S2Pro;
use syrinx_stt::{wer, Stt};

const SET: &str = include_str!("golden/cue_eval/cue_set.jsonl");
const BACKEND: &str = "fish-s2-pro";
const ALPHA: f64 = 0.05;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

fn env_path(k: &str) -> Option<String> {
    env(k).filter(|p| std::path::Path::new(p).exists())
}

fn env_usize(k: &str, default: usize) -> usize {
    env(k).and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// Collapse runs of whitespace. Stripping a cue leaves the gap it occupied, and neither
/// the backend nor the WER reference should carry that artifact.
fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn real_cue_activation_run() {
    let Some(out_path) = env("SYRINX_CUE_ACTIVATION_OUT") else {
        eprintln!(
            "SKIP real_cue_activation: set SYRINX_CUE_ACTIVATION_OUT to a JSON path to run \
             the C4.2 certification (GPU-bound and long; never part of a routine board run)"
        );
        return;
    };
    let Some(s2_dir) = env_path("SYRINX_FISH_S2_DIR") else {
        eprintln!("SKIP real_cue_activation: set SYRINX_FISH_S2_DIR to the s2-pro checkpoint dir");
        return;
    };
    let Some(stt_dir) = env_path("SYRINX_STT_MODEL_DIR") else {
        eprintln!("SKIP real_cue_activation: set SYRINX_STT_MODEL_DIR to the Whisper model dir");
        return;
    };
    let Some(ref_wav) = env_path("SYRINX_FISH_REF_WAV") else {
        eprintln!(
            "SKIP real_cue_activation: set SYRINX_FISH_REF_WAV — the run clones one fixed \
             reference voice so the voice is held constant across every case"
        );
        return;
    };

    let n_per_condition = env_usize("SYRINX_CUE_ACTIVATION_N", 4);
    assert!(
        n_per_condition >= min_n_for_alpha(ALPHA),
        "SYRINX_CUE_ACTIVATION_N={n_per_condition} cannot reject at alpha={ALPHA}; \
         the exact test needs at least {} renders per condition, and a smaller n would \
         report 0% activation no matter how the model behaves",
        min_n_for_alpha(ALPHA)
    );

    let all_cases = CueCase::parse_set(SET);
    assert!(!all_cases.is_empty(), "frozen cue set failed to parse");
    let limit = env_usize("SYRINX_CUE_ACTIVATION_CASES", all_cases.len());

    // s2-pro on the selected device; Whisper stays on CPU so it does not compete for the
    // card's 12 GB with a ~10 GB bf16 model.
    let dev = syrinx_serve::synth::pick_device(
        env("SYRINX_FISH_DEVICE").and_then(|s| s.trim().parse::<usize>().ok()),
    );
    eprintln!("[activation] device={dev:?} n={n_per_condition} alpha={ALPHA}");

    let t0 = Instant::now();
    let mut model = S2Pro::load(&s2_dir, dev.clone()).expect("load s2-pro");
    let stt = Stt::load(&stt_dir, Device::Cpu).expect("load Whisper");
    eprintln!("[activation] models loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let sr = <S2Pro as RvqCodec>::sample_rate(&model);

    // One reference voice, encoded once and reused: holding the speaker constant keeps the
    // cue as the only thing that varies between the two conditions.
    let ref_samples = fish_audio::read_ref_wav_44k(&ref_wav).expect("read SYRINX_FISH_REF_WAV");
    let ref_tensor =
        Tensor::from_vec(ref_samples.clone(), ref_samples.len(), &dev).expect("ref tensor");
    let ref_codes = model.encode_reference(&ref_tensor).expect("encode reference");
    let ref_text = env("SYRINX_FISH_REF_TEXT").unwrap_or_default();

    let max_frames = env_usize("SYRINX_FISH_MAXFRAMES", 512);
    let vocab = syrinx_cue::Vocab::embedded().expect("cue vocab");
    let caps = syrinx_cue::BackendId::FishS2Pro.caps().expect("s2-pro caps");

    let mut measurements: Vec<Measurement> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let run_start = Instant::now();

    for (i, case) in all_cases.iter().take(limit).enumerate() {
        let doc = syrinx_cue::parse_any(&case.text, &vocab, &syrinx_cue::ParseOptions::default())
            .expect("parse frozen case");
        let lowered = syrinx_cue::lower_full(&doc, &caps, &vocab);
        let plain = squeeze(&lowered.text);
        let cued = squeeze(&case.text);

        // A `neutral` control carries no cue, so the two conditions would be the same text.
        // It still contributes its baseline WER, which is the point of the control group.
        let is_neutral = case.kind == "neutral";

        let mut render = |text: &str, seed_base: u64| -> Vec<Vec<f32>> {
            (0..n_per_condition)
                .map(|k| {
                    let params = DriveParams {
                        seed: seed_base + k as u64,
                        max_new_frames: max_frames,
                        ..Default::default()
                    };
                    model
                        .synthesize_cloned(&ref_text, &ref_codes, text, &params)
                        .unwrap_or_else(|e| panic!("synthesize {text:?} seed {k}: {e}"))
                })
                .collect()
        };

        // Distinct seed bases per case and per condition: identical seeds across conditions
        // would pair the draws, which the unpaired permutation test does not assume.
        let base = (i as u64 + 1) * 1000;
        let cued_wavs = if is_neutral { Vec::new() } else { render(&cued, base) };
        let plain_wavs = render(&plain, base + 500);

        let transcribe_wer = |wavs: &[Vec<f32>]| -> f64 {
            let scores: Vec<f64> = wavs
                .iter()
                .map(|w| {
                    let t = stt
                        .transcribe_lang(w, sr, Some(case.lang.as_str()))
                        .expect("transcribe");
                    f64::from(wer(&plain, &t.text))
                })
                .collect();
            scores.iter().sum::<f64>() / scores.len().max(1) as f64
        };

        let baseline_wer = transcribe_wer(&plain_wavs);

        let (activated, cued_wer, detail) = if is_neutral {
            // No cue to activate; the control's own WER is both sides of the delta.
            (false, baseline_wer, "neutral control".to_string())
        } else {
            let feats = |wavs: &[Vec<f32>]| -> Vec<Features> {
                wavs.iter().map(|w| features(w, sr)).collect()
            };
            let outcome = activation_test(&feats(&cued_wavs), &feats(&plain_wavs), ALPHA)
                .expect("group sizes were validated above");
            (
                outcome.activated,
                transcribe_wer(&cued_wavs),
                format!("p={:.4} effect={:.3}", outcome.p_value, outcome.effect),
            )
        };

        assert!(baseline_wer.is_finite(), "{}: non-finite baseline WER", case.id);
        assert!(cued_wer.is_finite(), "{}: non-finite cued WER", case.id);

        eprintln!(
            "[activation] {:>3}/{} {:<34} {:<8} act={:<5} wer={:.3} base={:.3}  {}",
            i + 1,
            limit,
            case.id,
            case.kind,
            activated,
            cued_wer,
            baseline_wer,
            detail
        );

        measurements.push(Measurement {
            case_id: case.id.clone(),
            backend: BACKEND.to_string(),
            activated,
            wer: cued_wer,
            baseline_wer,
        });
    }

    let elapsed = run_start.elapsed().as_secs_f64();
    let measured = all_cases.iter().take(limit).count();
    assert_eq!(
        measurements.len(),
        measured,
        "every case in the run must produce a measurement; missing: {skipped:?}"
    );
    skipped.clear();

    let report = evaluate_activation(
        &all_cases,
        &measurements,
        &[BACKEND.to_string()],
        Thresholds::default(),
    );

    assert!(!report.cells.is_empty(), "no cells aggregated from {} measurements", measurements.len());
    for c in &report.cells {
        assert!(
            c.activation_rate.is_finite() && (0.0..=1.0).contains(&c.activation_rate),
            "{}/{}: activation_rate out of range: {}",
            c.backend,
            c.kind,
            c.activation_rate
        );
        assert!(c.wer_delta.is_finite(), "{}/{}: non-finite wer_delta", c.backend, c.kind);
    }

    let json = report.to_json();
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = std::fs::File::create(&out_path).expect("create output JSON");
    f.write_all(json.as_bytes()).expect("write output JSON");

    eprintln!("\n[activation] {measured} cases in {elapsed:.1}s -> {out_path}");
    eprintln!("{json}");
    for v in &report.violations {
        eprintln!("[activation] threshold not met: {v}");
    }

    if env("SYRINX_CUE_ACTIVATION_ENFORCE").as_deref() == Some("1") {
        assert!(
            report.passed(),
            "C4.2 thresholds not met: {:?}",
            report.violations.iter().map(|v| v.to_string()).collect::<Vec<_>>()
        );
    } else {
        eprintln!(
            "[activation] thresholds NOT enforced (set SYRINX_CUE_ACTIVATION_ENFORCE=1 once \
             the numbers above have been reviewed and the limits agreed)"
        );
    }
}
