//! **C4.2′ — the Qwen capability-profile activation run.** ADR-0003.
//!
//! `real_cue_activation` certifies the *research* path: it hard-wires
//! `const BACKEND = "fish-s2-pro"` and gates event activation on `Inline::Open` backends.
//! After the Qwen redirection that criterion is orphaned — the only `Inline::Open` backend
//! is research-licensed, and every Qwen checkpoint is `inline = none` with
//! `event = unsupported`, so event cues are filtered before a render happens. This is the
//! shipping-path replacement, and it asserts a different thing: not a floor, but that the
//! *instrument* is sound.
//!
//! Four clauses, pre-registered in ADR-0003:
//!
//!   A1  the `accepted` checkpoint renders bit-identically (a control on OUR filter)
//!   A2  the plain arm agrees with itself (A/A calibration, at the NOMINAL alpha)
//!   A3  a delivery-neutral sham does not activate  <- what makes the rest mean anything
//!   B   a pre-registered sentinel activates        <- only once earned by replication
//!   C   the WER veto, carried over unchanged
//!
//! **First run enforces nothing.** Clause B has no sentinel until a run earns one, and
//! ADR-0003 records "clause B not enabled" as an acceptable outcome *in advance*, so the
//! run cannot quietly become a search for a sentinel. Set `SYRINX_C42_ENFORCE=1` only once
//! a sentinel has been pinned.
//!
//! Opt-in, GPU-bound, ~26 min. Never part of a routine board.
//!
//!     source scripts/test-all.env
//!     SYRINX_C42_OUT=.opt-reports/c42-qwen.json \
//!     SYRINX_EMOTION2VEC_ONNX=/data/models/emotion2vec-plus-large/model.onnx \
//!       ./scripts/test-all.sh --test real_cue_activation_qwen

#![cfg(all(feature = "real", feature = "cuda"))]

use std::collections::BTreeMap;

use candle_core::Device;
use syrinx_cue::BackendId;
use syrinx_eval::acoustic::{activation_test, features, min_n_for_headroom, Features};
use syrinx_eval::activation::CueCase;
use syrinx_eval::contrast::{
    evaluate_contrast, Arm, ArmContrast, ContrastThresholds, PipelineControl,
};
use syrinx_serve::qwen::{plan, QwenEngine, QwenRequest};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};
use syrinx_stt::{wer, Stt};

const SET: &str = include_str!("golden/cue_eval/cue_set.jsonl");

/// The six pre-registered sentinel candidates (ADR-0003). English only: both the WER
/// oracle and the judge are strongest there, and mixing languages into a six-case gate
/// would confound the one axis it can actually resolve.
const SENTINELS: [&str; 6] = [
    "en-emotion-sad-mid",
    "en-emotion-happy-leading",
    "en-emotion-angry-trailing",
    "en-style-whisper-leading",
    "en-style-shout-mid",
    "en-emotion-calm-mid",
];

/// Delivery-neutral, length-matched. If this activates, every number in the run is
/// uninterpretable — the contrast is measuring prompt perturbation, not content.
const SHAM: &str = "Read the sentence that follows";

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}
fn env_usize(k: &str, d: usize) -> usize {
    env(k).and_then(|v| v.trim().parse().ok()).unwrap_or(d)
}
fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn real_cue_activation_qwen_run() {
    let Some(out_path) = env("SYRINX_C42_OUT") else {
        eprintln!(
            "SKIP real_cue_activation_qwen: set SYRINX_C42_OUT to a JSON path to run the \
             C4.2' certification (GPU-bound, ~26 min; never part of a routine board)"
        );
        return;
    };
    let (Some(cv), Some(tok)) = (env("SYRINX_QWEN_CV_DIR"), env("SYRINX_QWEN_TOK_DIR")) else {
        eprintln!("SKIP real_cue_activation_qwen: SYRINX_QWEN_CV_DIR / _TOK_DIR unset");
        return;
    };
    let small = env("SYRINX_QWEN_CV_DIR_0_6B");

    let n = env_usize("SYRINX_C42_N", 8);
    let enforce = env("SYRINX_C42_ENFORCE").is_some();
    let sentinel = env("SYRINX_C42_SENTINEL");

    let cases: Vec<CueCase> = CueCase::parse_set(SET);
    let chosen: Vec<&CueCase> =
        cases.iter().filter(|c| SENTINELS.contains(&c.id.as_str())).collect();
    assert_eq!(
        chosen.len(),
        SENTINELS.len(),
        "the frozen set is missing a pre-registered sentinel: have {:?}",
        chosen.iter().map(|c| &c.id).collect::<Vec<_>>()
    );

    // Bonferroni over every contrast the run makes: 3 per case (cue-vs-plain, sham-vs-plain,
    // cue-vs-sham) plus one A/A per case.
    let comparisons = chosen.len() * 4;
    let th = ContrastThresholds {
        alpha: 0.05,
        comparisons,
        max_aa_activations: None,
        max_wer_delta: 0.5,
        sentinel: sentinel.clone(),
    };
    let alpha = th.corrected_alpha();
    // POWER, not feasibility. `min_n_for_alpha` says whether the test can EVER reject;
    // at that n the only way to clear the bar is the single most extreme labeling. The
    // [sad] tuning round of 2026-09-08 was wrecked by exactly that confusion, and the
    // first clean run of THIS gate then repeated it at 13.4x headroom while three of six
    // cases sat within 1.2x of the bar — the regime where power decides the verdict.
    let floor = min_n_for_headroom(alpha, 20.0);
    assert!(
        n >= floor,
        "n={n} has too little power for alpha={alpha:.5}: 20x headroom needs n>={floor}. \
         Reporting \"no activation\" from an underpowered run is a false negative dressed \
         as a certification."
    );
    eprintln!(
        "[c42'] {} cases, n={n}, alpha 0.05/{comparisons} = {alpha:.5} (n floor {floor}), \
         enforce={enforce}",
        chosen.len()
    );

    let dev = Device::new_cuda(env_usize("SYRINX_QWEN_DEVICE", 0)).expect("cuda");
    let backend = BackendId::Qwen17bCustomVoice;
    let engine =
        QwenModelEngine::load(backend, &cv, &tok, dev.clone(), QwenVoice::Preset("serena".into()))
            .expect("load 1.7B");
    assert!(engine.honors_seed(), "the engine must honour seeds or every take is one draw");
    // Whisper is optional: with no model dir the WER veto has nothing to say, and the run
    // reports that rather than silently scoring 0.0 and passing clause C for free.
    let stt = env("SYRINX_STT_MODEL_DIR").and_then(|d| Stt::load(&d, Device::Cpu).ok());
    if stt.is_none() {
        eprintln!("  [C] SYRINX_STT_MODEL_DIR unset — the WER veto is NOT exercised this run");
    }

    let mut contrasts: Vec<ArmContrast> = Vec::new();
    let mut not_applicable: Vec<String> = Vec::new();
    let mut lowering_bugs: Vec<String> = Vec::new();
    let mut seen_texts: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut report: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    for case in &chosen {
        let p = plan(backend, &case.text).expect("plan");
        // The segment that CARRIES the instruct, not the first one. A `mid` or `trailing`
        // cue makes `pass_hoist` split the utterance, and segment 0 is then the text BEFORE
        // the cue, which by construction has no instruct. Reading `segments.first()` made
        // four of six sentinels report `not_applicable` on the first run -- a fabricated
        // non-measurement, which is the failure ADR-0003 §5 exists to prevent, arriving
        // from the opposite direction. The cue scopes a span; that span is what to measure.
        let carrier = p.segments.iter().find(|s| s.instruct.is_some());
        let Some((clean, instruct)) =
            carrier.map(|s| (squeeze(&s.text), s.instruct.clone().expect("carrier has one")))
        else {
            // Nothing carried an instruct. That is legitimate ONLY if the backend cannot
            // express this kind; if caps say it can, the cue was dropped by a lowering bug
            // and calling it "not applicable" would bury exactly the defect this gate is
            // for. ADR-0003 §5 names both halves and the runner must tell them apart.
            let caps = backend.caps().expect("caps");
            let kind = match case.kind.as_str() {
                "emotion" => Some(syrinx_cue::ir::CueKind::Emotion {
                    label: case.label.clone(),
                    intensity: 1.0,
                }),
                "style" => Some(syrinx_cue::ir::CueKind::Style { label: case.label.clone() }),
                "event" => Some(syrinx_cue::ir::CueKind::Event { label: case.label.clone() }),
                _ => None,
            };
            let expressible = kind.as_ref().is_some_and(|k| caps.can_express(k));
            if expressible {
                eprintln!(
                    "  {:<28} LOWERING BUG: caps say {} is expressible, but no segment \
                     carries an instruct",
                    case.id, case.kind
                );
                lowering_bugs.push(case.id.clone());
            } else {
                eprintln!("  {:<28} not_applicable ({} unsupported here)", case.id, case.kind);
                not_applicable.push(case.id.clone());
            }
            continue;
        };

        let render = |ins: Option<&str>, base: u64| -> Vec<Vec<f32>> {
            (0..n)
                .map(|i| {
                    let req = QwenRequest {
                        mode: p.mode,
                        text: &clean,
                        instruct: ins,
                        instruct_effect: p.instruct_effect,
                    };
                    engine.render_seeded(&req, base + i as u64).expect("render")
                })
                .collect()
        };

        // 2n plain: the first n serve as the control arm, the second n as the A/A partner.
        let plain_a = render(None, 0);
        let plain_b = render(None, 1_000);
        let cued = render(Some(&instruct), 0);
        let sham = render(Some(SHAM), 0);

        let f = |w: &Vec<Vec<f32>>| w.iter().map(|s| features(s, 24_000)).collect::<Vec<Features>>();
        let (fa, fb, fc, fs) = (f(&plain_a), f(&plain_b), f(&cued), f(&sham));

        // WER on the cued arm against the plan's CLEAN text — never the caller's input,
        // which still carries the markup.
        let (mut w_cued, mut w_plain) = (0.0f64, 0.0f64);
        if let Some(s) = stt.as_ref() {
            let tr = |w: &Vec<f32>| {
                s.transcribe(w, 24_000).map(|t| t.text).unwrap_or_default()
            };
            w_cued = f64::from(wer(&clean, &tr(&cued[0])));
            w_plain = f64::from(wer(&clean, &tr(&plain_a[0])));
        }

        let mk = |arm: Arm, against: Arm, a: &[Features], b: &[Features]| ArmContrast {
            case_id: case.id.clone(),
            backend: format!("{backend:?}"),
            arm,
            against,
            p_value: activation_test(a, b, alpha).map(|o| o.p_value).unwrap_or(1.0),
            effect: activation_test(a, b, alpha).map(|o| o.effect).unwrap_or(0.0),
            wer: w_cued.max(w_plain),
            baseline_wer: w_plain,
        };
        let cvp = mk(Arm::Cue, Arm::Plain, &fc, &fa);
        let svp = mk(Arm::Sham, Arm::Plain, &fs, &fa);
        let cvs = mk(Arm::Cue, Arm::Sham, &fc, &fs);
        // A/A depends only on the TEXT: two cases over the same sentence render the same
        // plain audio and produce the identical A/A p-value. Counting it once per case
        // inflates the activation count against a budget derived from the case count --
        // the first run reported 2 activations from ONE underlying measurement, because
        // two sentinels share a sentence.
        let aa = mk(Arm::ControlA, Arm::ControlB, &fa, &fb);
        let first_for_text = seen_texts.insert(clean.clone());

        eprintln!(
            "  {:<28} cue/plain {:.4}  sham/plain {:.4}  cue/sham {:.4}  A/A {:.4}  wer {:.3}",
            case.id, cvp.p_value, svp.p_value, cvs.p_value, aa.p_value, w_cued
        );
        report.insert(
            case.id.clone(),
            serde_json::json!({
                "instruct": instruct,
                "cue_vs_plain": cvp.p_value, "sham_vs_plain": svp.p_value,
                "cue_vs_sham": cvs.p_value, "a_a": aa.p_value,
                "wer_cued": w_cued, "wer_plain": w_plain,
            }),
        );
        contrasts.extend([cvp, svp, cvs]);
        if first_for_text {
            contrasts.push(aa);
        }
    }

    // The 1.7B is no longer needed and holds ~4.5 GB of VRAM. Loading the 0.6B on top of
    // it OOM'd on a box whose GPUs are shared with other work, so it is dropped explicitly
    // rather than at end of scope.
    drop(engine);

    // Clause A1: the `accepted` checkpoint must render bit-identically. Two renders, and a
    // stronger statement than any p-value — but a control on OUR OWN `honors_instruct`
    // filter, not on the model, and ADR-0003 says so rather than overclaiming.
    let mut controls = Vec::new();
    if let Some(dir) = small {
        let e = QwenModelEngine::load(
            BackendId::Qwen06bCustomVoice,
            &dir,
            &tok,
            dev,
            QwenVoice::Preset("serena".into()),
        )
        .expect("load 0.6B");
        let p = plan(BackendId::Qwen06bCustomVoice, "[angry] You told me it was handled.")
            .expect("plan");
        let clean = squeeze(&p.text);
        // Written out rather than built by a closure: a closure returning `QwenRequest<'_>`
        // cannot relate the borrow of its argument to the borrow in its return type.
        let plain_req = QwenRequest {
            mode: p.mode,
            text: &clean,
            instruct: None,
            instruct_effect: p.instruct_effect,
        };
        let cued_req = QwenRequest {
            mode: p.mode,
            text: &clean,
            instruct: Some("Speak in an angry tone"),
            instruct_effect: p.instruct_effect,
        };
        let a = e.render_seeded(&plain_req, 0).expect("render");
        let b = e.render_seeded(&cued_req, 0).expect("render");
        let identical = a == b;
        eprintln!("  [A1] 0.6B cued == plain: {identical}");
        controls.push(PipelineControl {
            backend: "qwen3-0.6b-customvoice".into(),
            renders_identical: identical,
        });
    } else {
        eprintln!("  [A1] SKIPPED — SYRINX_QWEN_CV_DIR_0_6B unset (the pipeline control did not run)");
    }

    assert!(
        lowering_bugs.is_empty(),
        "cues the backend declares expressible were dropped before rendering: {lowering_bugs:?} \
         — a lowering bug, not a null result, and it must not be reported as one"
    );
    let verdict = evaluate_contrast(&contrasts, &controls, &not_applicable, &th);
    eprintln!("\n[c42'] corrected alpha {:.5}", verdict.corrected_alpha);
    eprintln!("[c42'] content-activated (beat plain AND sham): {:?}", verdict.content_activated);
    eprintln!("[c42'] perturbation-only (beat plain, not sham): {:?}", verdict.perturbation_only);
    eprintln!("[c42'] not applicable: {} case(s)", verdict.not_applicable.len());
    for v in &verdict.violations {
        eprintln!("[c42'] VIOLATION {v:?}");
    }

    let json = serde_json::json!({
        "alpha": verdict.corrected_alpha, "n": n, "enforced": enforce,
        "sentinel": sentinel,
        "content_activated": verdict.content_activated,
        "perturbation_only": verdict.perturbation_only,
        "not_applicable": verdict.not_applicable,
        "violations": format!("{:?}", verdict.violations),
        "cases": report,
    });
    std::fs::create_dir_all(std::path::Path::new(&out_path).parent().unwrap_or_else(|| std::path::Path::new("."))).ok();
    std::fs::write(&out_path, serde_json::to_string_pretty(&json).unwrap()).expect("write");
    eprintln!("[c42'] wrote {out_path}");

    if enforce {
        assert!(
            verdict.passed(),
            "C4.2' FAILED: {:?}",
            verdict.violations
        );
    } else {
        eprintln!(
            "\n[c42'] NOT ENFORCED. ADR-0003 records this first run as measurement only: a \
             sentinel must clear p <= alpha AND replicate on a disjoint seed block before \
             it may be pinned, and \"clause B not enabled\" is an acceptable outcome. Set \
             SYRINX_C42_ENFORCE=1 with SYRINX_C42_SENTINEL once one is earned."
        );
    }
}
