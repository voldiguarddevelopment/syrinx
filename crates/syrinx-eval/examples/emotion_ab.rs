//! The emotion A/B, with a real noise model.
//!
//! The three pairs in `renders/2026-09-06-qwen-emotion-ab/` were n=1 per condition, so a
//! single render's delta could not be told from one lucky draw. Two of the three verdicts
//! came back "cannot tell" against a floor estimated by sliding a window over the *same*
//! render — a stand-in for sampling noise, not a measurement of it.
//!
//! Per-render seed control (d1d0ceb) removes the need for the stand-in. This renders each
//! condition `n` times at distinct seeds and asks two separate questions:
//!
//!   1. **Did the delivery change at all, beyond this model's own variation?**
//!      `syrinx_eval::acoustic`'s exact permutation test over 11 acoustic dimensions.
//!      This is deliberately blind to WHICH way it moved.
//!   2. **Did it change toward the named emotion?** The affect judge's probability for the
//!      cued class, cued mean vs plain mean, compared against the spread ACROSS SEEDS
//!      within each condition — a real noise estimate.
//!
//! Both matter and neither substitutes for the other. A backend that answered `[sad]` by
//! shouting passes (1) and fails (2); a backend that ignores the cue fails both. Question
//! (2) is reported, never asserted — per CLAUDE.md, intended emotion stays a human call,
//! and the judge is a weak one for most classes (see docs/LICENSES.md).
//!
//! Usage: emotion_ab <talker-dir> <tokenizer-dir> [n]

use candle_core::Device;
use syrinx_cue::BackendId;
use syrinx_eval::acoustic::{activation_test, features};
use syrinx_eval::affect::{ravdess8_spec, ravdess_label_for_cue, score_resampled, OnnxJudge};
use syrinx_serve::qwen::{plan, QwenEngine, QwenRequest};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};

const CASES: [(&str, &str, &str); 3] = [
    ("happy", "[happy]", "We finally heard back, and the news is better than we hoped."),
    ("sad", "[sad]", "I waited by the window until the last light went out."),
    ("angry", "[angry]", "You told me it was handled. You looked me in the eye and said it was handled."),
];

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (talker, tok) = (&a[0], &a[1]);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse().ok()) {
        Some(i) => Device::new_cuda(i)?,
        None => Device::Cpu,
    };
    let backend = BackendId::Qwen17bCustomVoice;
    let engine = QwenModelEngine::load(
        backend,
        talker,
        tok,
        dev.clone(),
        QwenVoice::Preset("serena".into()),
    )?;
    eprintln!("[ab] device {:?}  n={n} per condition  honors_seed={}", dev.location(), engine.honors_seed());
    if !engine.honors_seed() {
        return Err("engine does not honour seeds — every take would be the same draw".into());
    }

    let mut judge = match std::env::var("SYRINX_AFFECT_ONNX") {
        Ok(p) => Some(OnnxJudge::load(&p, ravdess8_spec())?),
        Err(_) => {
            eprintln!("[ab] SYRINX_AFFECT_ONNX unset — acoustic question only, no affect judge");
            None
        }
    };

    for (label, cue, text) in CASES {
        let p = plan(backend, &format!("{cue} {text}"))?;
        let clean = p.text.split_whitespace().collect::<Vec<_>>().join(" ");
        let instruct = p.segments.first().and_then(|s| s.instruct.clone());

        let mut render = |ins: Option<&str>| -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
            (0..n)
                .map(|i| {
                    let req = QwenRequest {
                        mode: p.mode,
                        text: &clean,
                        instruct: ins,
                        instruct_effect: p.instruct_effect,
                    };
                    Ok(engine.render_seeded(&req, i as u64)?)
                })
                .collect()
        };
        let plain = render(None)?;
        let cued = render(instruct.as_deref())?;

        // 1. did it change at all, beyond this model's own variation?
        let f = |w: &Vec<Vec<f32>>| w.iter().map(|s| features(s, 24_000)).collect::<Vec<_>>();
        let out = activation_test(&f(&cued), &f(&plain), 0.05)
            .ok_or("activation_test refused — n too small for alpha=0.05")?;

        // 2. did it change toward the named emotion?
        let mut affect = String::from("(no judge)");
        if let Some(j) = judge.as_mut() {
            // Our cue id is not the judge's label vocabulary; `ravdess_label_for_cue` owns
            // that mapping so the example does not invent a second one.
            let jl = ravdess_label_for_cue(label)
                .ok_or_else(|| format!("no RAVDESS class maps to cue {label:?}"))?;
            let mut probs = |w: &Vec<Vec<f32>>| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
                w.iter()
                    .map(|s| {
                        let sc = score_resampled(j, s, 24_000)?;
                        Ok(f64::from(sc.get(jl).unwrap_or(f32::NAN)))
                    })
                    .collect()
            };
            let (pc, pp) = (probs(&cued)?, probs(&plain)?);
            let (mc, mp) = (mean(&pc), mean(&pp));
            // The noise floor is now measured, not simulated: the spread ACROSS SEEDS
            // within each condition, pooled.
            let noise = (sd(&pc).powi(2) + sd(&pp).powi(2)).sqrt();
            let verdict = if (mc - mp).abs() <= noise {
                "inside the seed noise — cannot tell"
            } else if mc > mp {
                "TOWARD the cued class"
            } else {
                "AWAY from the cued class"
            };
            affect = format!(
                "p({jl}) {mp:.3} -> {mc:.3}  delta {:+.3}  seed-noise +/-{noise:.3}  {verdict}",
                mc - mp
            );
        }

        println!(
            "{label:>6}  changed={:<5} p={:.4} effect={:.2}   {affect}",
            out.activated, out.p_value, out.effect
        );
    }
    Ok(())
}
