//! Does the instruct's LANGUAGE matter — and is any of it more than prompt perturbation?
//!
//! Two questions, one run, because they share their renders.
//!
//! **1. Language.** `InstructLang::default()` is `Zh`, with the note that "CV3 follows
//! Chinese instruct prompts best (the on-box A/B)". But `SplitOptions::default()` sets
//! `En` (hoist.rs:32), so every Qwen render ever made by this tree has used the English
//! phrase — on a model whose upstream instruct examples are overwhelmingly Chinese. If Zh
//! moves the audio where En does not, then the "2 of 3 cues do nothing" result is a
//! language-selection default and not a phrasing problem at all.
//!
//! **2. Perturbation.** `emotion_ab.rs` contrasts cued against plain, so the two arms
//! differ in prompt LENGTH and CONTENT, not only in meaning. `assemble_text_mode` prepends
//! the instruct block as text tokens, so at a fixed seed a longer prompt gives a different
//! AR trajectory whether or not the model attaches any meaning to the words. A model that
//! treated the instruct as pure noise would still reject the cued-vs-plain null. That
//! makes `[sad]`'s p=0.0019 uninterpretable on its own.
//!
//! The fix is a **sham arm** per language: a delivery-neutral instruction of comparable
//! length. The interesting contrast is then `cue vs sham`, not `cue vs plain`:
//!
//!   cue vs plain   — an instruction of some kind changed the audio  (the old, weak claim)
//!   sham vs plain  — ANY instruction changes the audio              (the confound, measured)
//!   cue vs sham    — the DELIVERY CONTENT changed the audio         (the claim we want)
//!
//! If sham-vs-plain activates and cue-vs-sham does not, the cue is doing nothing that a
//! meaningless string of the same size would not do, and every activation number this
//! project has recorded needs re-reading. That is the outcome this example exists to be
//! able to report.
//!
//! Reported, never asserted — per CLAUDE.md, intended emotion stays a human call, and the
//! current judge is weak on 6 of 8 classes (docs/backends/AFFECT_JUDGES.md).
//!
//! Usage: instruct_lang_ab <talker-dir> <tokenizer-dir> [n]

use candle_core::Device;
use syrinx_cue::BackendId;
use syrinx_eval::acoustic::{activation_test, features, Features};
use syrinx_eval::affect::{ravdess8_spec, ravdess_label_for_cue, score_resampled, OnnxJudge};
use syrinx_serve::qwen::{plan, QwenEngine, QwenRequest};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};

/// label, cue, text. Same three sentences as `emotion_ab.rs`, so the En arm reproduces the
/// recorded n=8 result and acts as this run's own control on the harness.
const CASES: [(&str, &str, &str); 3] = [
    ("happy", "[happy]", "We finally heard back, and the news is better than we hoped."),
    ("sad", "[sad]", "I waited by the window until the last light went out."),
    ("angry", "[angry]", "You told me it was handled. You looked me in the eye and said it was handled."),
];

/// The Chinese phrasings, taken verbatim from `syrinx_cue::legacy_emotion::DEFAULT_EMOTIONS`
/// rather than invented here — that table is what a `lang: Zh` config would actually ship.
const ZH: [(&str, &str); 3] = [
    ("happy", "用开心愉悦的语气说"),
    ("sad", "用悲伤难过的语气说"),
    ("angry", "用愤怒生气的语气说"),
];

/// Delivery-neutral instructions, length-matched to the real ones. These must carry no
/// information about HOW to speak — only that speaking is what is wanted. If a sham
/// activates, the contrast is measuring prompt perturbation.
const SHAM_EN: &str = "Read the sentence that follows";
const SHAM_ZH: &str = "请朗读下面这句话";

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

/// One acoustic contrast, formatted. `alpha` is already Bonferroni-corrected by the caller.
fn contrast(name: &str, a: &[Features], b: &[Features], alpha: f64) -> String {
    match activation_test(a, b, alpha) {
        Some(o) => format!(
            "{name:<16} {} p={:.4} effect={:.2}",
            if o.activated { "CHANGED" } else { "  --   " },
            o.p_value,
            o.effect
        ),
        None => format!("{name:<16} refused (n too small for alpha={alpha:.4})"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        return Err("usage: instruct_lang_ab <talker-dir> <tokenizer-dir> [n]".into());
    }
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
    if !engine.honors_seed() {
        return Err("engine does not honour seeds — every take would be the same draw".into());
    }

    // Five arms share the plain arm, so 5n renders per case, not 6n.
    const ARMS: usize = 5;
    // Seven contrasts per case; Bonferroni over the whole run so a hit means something.
    let comparisons = (CASES.len() * 7) as f64;
    let alpha = 0.05 / comparisons;
    eprintln!(
        "[lang-ab] device {:?}  n={n}/arm  {ARMS} arms x {} cases = {} renders\n\
         [lang-ab] alpha 0.05 Bonferroni/{comparisons:.0} = {alpha:.5}",
        dev.location(),
        CASES.len(),
        ARMS * n * CASES.len()
    );

    let mut judge = match std::env::var("SYRINX_AFFECT_ONNX") {
        Ok(p) => Some(OnnxJudge::load(&p, ravdess8_spec())?),
        Err(_) => {
            eprintln!("[lang-ab] SYRINX_AFFECT_ONNX unset — acoustic question only");
            None
        }
    };

    for (label, cue, text) in CASES {
        let p = plan(backend, &format!("{cue} {text}"))?;
        let clean = p.text.split_whitespace().collect::<Vec<_>>().join(" ");
        let en = p
            .segments
            .first()
            .and_then(|s| s.instruct.clone())
            .ok_or_else(|| format!("{label}: the cue lowered to no instruct at all"))?;
        let zh = ZH
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, s)| *s)
            .ok_or_else(|| format!("{label}: no Chinese phrasing"))?;

        eprintln!("\n[lang-ab] {label}: en={en:?}  zh={zh:?}");

        let render = |ins: Option<&str>| -> Result<Vec<Vec<f32>>, String> {
            (0..n)
                .map(|i| {
                    let req = QwenRequest {
                        mode: p.mode,
                        text: &clean,
                        instruct: ins,
                        instruct_effect: p.instruct_effect,
                    };
                    engine.render_seeded(&req, i as u64)
                })
                .collect()
        };

        let arms: [(&str, Vec<Vec<f32>>); ARMS] = [
            ("plain", render(None)?),
            ("en", render(Some(&en))?),
            ("zh", render(Some(zh))?),
            ("sham-en", render(Some(SHAM_EN))?),
            ("sham-zh", render(Some(SHAM_ZH))?),
        ];
        let feats: Vec<(&str, Vec<Features>)> = arms
            .iter()
            .map(|(k, w)| (*k, w.iter().map(|s| features(s, 24_000)).collect()))
            .collect();
        let get = |k: &str| -> &[Features] {
            &feats.iter().find(|(n, _)| *n == k).expect("arm present").1
        };

        println!("\n=== {label} ===");
        // The weak claim, for continuity with the recorded n=8 result.
        println!("  {}", contrast("en vs plain", get("en"), get("plain"), alpha));
        println!("  {}", contrast("zh vs plain", get("zh"), get("plain"), alpha));
        // The confound, measured rather than assumed away.
        println!("  {}", contrast("sham-en vs plain", get("sham-en"), get("plain"), alpha));
        println!("  {}", contrast("sham-zh vs plain", get("sham-zh"), get("plain"), alpha));
        // The claim we actually want: delivery content, over and above perturbation.
        println!("  {}", contrast("en vs sham-en", get("en"), get("sham-en"), alpha));
        println!("  {}", contrast("zh vs sham-zh", get("zh"), get("sham-zh"), alpha));
        // And the language question itself.
        println!("  {}", contrast("zh vs en", get("zh"), get("en"), alpha));

        if let Some(j) = judge.as_mut() {
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
            let base = probs(&arms[0].1)?;
            let (mb, sb) = (mean(&base), sd(&base));
            println!("  -- judge p({jl}), vs plain {mb:.3} (seed sd {sb:.3}) --");
            for (k, w) in arms.iter().skip(1) {
                let v = probs(w)?;
                let noise = (sd(&v).powi(2) + sb * sb).sqrt();
                let d = mean(&v) - mb;
                let verdict = if d.abs() <= noise {
                    "inside seed noise"
                } else if d > 0.0 {
                    "TOWARD cued class"
                } else {
                    "AWAY from cued class"
                };
                println!("     {k:<8} {:.3}  delta {d:+.3}  +/-{noise:.3}  {verdict}", mean(&v));
            }
        }
    }
    Ok(())
}
