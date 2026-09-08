//! Propose a better instruct phrasing for one cue label. ADR-0004.
//!
//! The loop **proposes**; a person accepts. This writes a proposal under `.opt-reports/`
//! and never touches `crates/syrinx-cue/`. A row is inert in `instruct.toml` until a human
//! writes `accepted_by`.
//!
//! Per round, per split: plain, incumbent, sham, counter-cue, and each candidate — n seeds
//! each. Every arm is re-measured in the same round at the same seeds, so nothing is
//! compared against a stored number.
//!
//! Usage: tune_instruct <talker-dir> <tokenizer-dir> <label> [n]
//! Env:   SYRINX_EMOTION2VEC_ONNX (required — the judge)
//!        SYRINX_QWEN_DEVICE, SYRINX_TUNE_OUT

use std::collections::BTreeMap;

use candle_core::Device;
use syrinx_cue::instruct::{lang_code, InstructTable};
use syrinx_cue::legacy_emotion::InstructLang;
use syrinx_cue::BackendId;
use syrinx_eval::acoustic::{activation_test, features, min_n_for_headroom, Features};
use syrinx_eval::affect::{
    emotion2vec9_spec, emotion2vec_label_for_cue, score_resampled, OnnxJudge,
};
use syrinx_eval::candidates::candidates_for;
use syrinx_eval::tune::{
    decide_per_sentence, SentenceMeasurement, Split, TuneArm, TuneMeasurement, TuneThresholds,
};
use syrinx_serve::qwen::{InstructEffect, QwenEngine, QwenMode, QwenRequest};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};

/// Sentences per split, **per label**. Sentence and cue are deliberately crossed: a phrase
/// that only works on one sentence is overfitting, and the holdout is what detects it.
///
/// Per label and semantically compatible with it, for the same reason
/// `sentence_dependence.rs` is: rendering `[sad]` over angry text would confound the cue
/// with the text's own affect in the RENDERER (the judge cannot read words, so it is
/// unaffected). There is no generic fallback.
///
/// **These sentences are fresh.** The `[angry]` round's holdout was measured by
/// `renders/2026-09-06-angry-sentences/`, which retired it — ADR-0004 §5 requires a new
/// `holdout_id` once a partition has been looked at, and looking at it is exactly what that
/// sweep did. None of the twelve below appears in either sentence-dependence sweep.
type Split3 = [&'static str; 3];

const ANGRY_TUNE: Split3 = [
    "You promised me this was finished and it is not finished.",
    "I have asked three times now and nobody has given me a straight answer.",
    "That is the second time this week and I am done being polite about it.",
];
const ANGRY_HOLDOUT: Split3 = [
    "Take your hands off that and step back.",
    "You do not get to decide that for me.",
    "I am not asking you again.",
];

const SAD_TUNE: Split3 = [
    "The house is quiet now in a way it never used to be.",
    "I keep forgetting that I cannot just call and tell her.",
    "There was no one left to turn the lights on.",
];
const SAD_HOLDOUT: Split3 = [
    "I packed the last box and closed the door behind me.",
    "Nobody came, and after a while I stopped watching the road.",
    "It rained the whole way home and I did not mind.",
];

fn splits_for(label: &str) -> Result<(Split3, Split3), String> {
    match label {
        "angry" => Ok((ANGRY_TUNE, ANGRY_HOLDOUT)),
        "sad" => Ok((SAD_TUNE, SAD_HOLDOUT)),
        other => Err(format!(
            "no sentence splits for cue {other:?} — add them to tune_instruct.rs rather than \
             reusing another cue's, which would confound the cue with the text"
        )),
    }
}

/// The counter-cue phrase per label: the phrasing for a DIFFERENT label, which should lose.
/// For `[sad]` the opposite pole is `happy`; for `[angry]` likewise, since a phrase that
/// merely raises arousal would otherwise look like anger.
/// The judge's measured cross-corpus recall on this class (CREMA-D, 180 clips,
/// `scripts/calibrate-emotion2vec.py`). ADR-0004 makes this mandatory on a tuned row: a
/// verdict quoted without it is not a verdict, and hardcoding one class's value onto
/// another's row would be worse than omitting it.
fn judge_recall_for(label: &str) -> f64 {
    match label {
        "angry" => 1.000,
        "neutral" => 1.000,
        "happy" => 0.967,
        "disgusted" => 0.967,
        "sad" => 0.867,
        "afraid" | "fearful" => 0.667,
        // Not probed by CREMA-D. NaN rather than a guess: it will show up in the proposal
        // as `NaN` and force whoever signs it to go and measure.
        _ => f64::NAN,
    }
}

fn counter_cue_for(label: &str) -> &'static str {
    match label {
        "sad" => "Speak in a happy, cheerful tone",
        _ => "Speak in a sad, sorrowful tone",
    }
}

const SHAM: &str = "Read the sentence that follows";
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
    if a.len() < 3 {
        return Err("usage: tune_instruct <talker-dir> <tokenizer-dir> <label> [n]".into());
    }
    let (talker, tok, label) = (&a[0], &a[1], a[2].clone());
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse().ok()) {
        Some(i) => Device::new_cuda(i)?,
        None => Device::Cpu,
    };
    let backend = BackendId::Qwen17bCustomVoice;
    let engine =
        QwenModelEngine::load(backend, talker, tok, dev.clone(), QwenVoice::Preset("serena".into()))?;
    if !engine.honors_seed() {
        return Err("engine does not honour seeds".into());
    }
    let mut judge = OnnxJudge::load(
        &std::env::var("SYRINX_EMOTION2VEC_ONNX")
            .map_err(|_| "SYRINX_EMOTION2VEC_ONNX must point at the exported judge")?,
        emotion2vec9_spec(),
    )?;
    let jl = emotion2vec_label_for_cue(&label)
        .ok_or_else(|| format!("the judge has no class for cue {label:?}"))?;

    let (tune_set, holdout_set) = splits_for(&label)?;
    let counter = counter_cue_for(&label);
    let table = InstructTable::shared();
    let lang = InstructLang::En;
    let incumbent = table
        .phrase(&label, lang)
        .ok_or_else(|| format!("no curated phrase for {label:?}"))?
        .to_string();

    // Candidates from the enumerable grammar (`syrinx_eval::candidates`), whose entire
    // output space is walked by `tests/cue_candidate_grammar.rs`. The previous inline
    // templates assumed every label was a predicate adjective and were 32% malformed.
    let vocab = syrinx_cue::vocab::Vocab::embedded()?;
    let candidates: Vec<String> = candidates_for(&label, &vocab, table)
        .into_iter()
        .filter(|c| c != &incumbent)
        .collect();
    if candidates.is_empty() {
        return Err(format!("the grammar produced no candidates for {label:?}").into());
    }
    eprintln!("[tune] label={label} judge-class={jl} n={n}/arm");
    eprintln!("[tune] incumbent: {incumbent:?}");
    for c in &candidates {
        eprintln!("[tune] candidate: {c:?}");
    }

    let th = TuneThresholds { alpha: 0.05, candidates: candidates.len(), ..Default::default() };
    // POWER, not just feasibility. min_n_for_alpha only asks whether the test can ever
    // reject; at that n the sole way to clear the bar is to draw the most extreme labeling
    // there is. The 2026-09-08 [sad] round ran at n=6 (4.6x headroom) and the INCUMBENT
    // scored exactly 1/462 on its best sentence and failed the other two.
    let floor = min_n_for_headroom(th.corrected_alpha(), 20.0);
    if n < floor {
        return Err(format!(
            "n={n} has too little power: alpha={:.5} needs n>={floor} for 20x headroom. \
             A gate only the most extreme draw can pass rejects working phrases and reads \
             as \"nothing is better\".",
            th.corrected_alpha()
        )
        .into());
    }
    eprintln!(
        "[tune] alpha {:.5} (Bonferroni/{})  renders: {}",
        th.corrected_alpha(),
        candidates.len(),
        // 2 splits x (plain + incumbent + sham + counter + candidates) arms
        // x sentences-per-split x n seeds. The sentence factor is easy to forget and
        // it is a 3x error in the estimate when you do.
        2 * n * (4 + candidates.len()) * tune_set.len()
    );

    // PER SENTENCE, never pooled. The first two rounds pooled 3 sentences x n seeds into
    // one permutation test and the INCUMBENT could not clear the acoustic bar (p=0.0166 /
    // 0.3580) while the same phrase scores 0.0003-0.0057 measured per sentence. Twelve
    // pooled samples doing worse than eight per-sentence samples is inflated variance, and
    // the sentence-dependence sweeps say where it comes from.
    let mut out: Vec<SentenceMeasurement> = Vec::new();
    let labels = emotion2vec9_spec().labels;

    for (split, sentences) in [(Split::Tune, tune_set), (Split::Holdout, holdout_set)] {
        eprintln!("\n[tune] === {split:?} split ===");
        for (si, text) in sentences.iter().enumerate() {
            let sid = format!("{split:?}-s{si}").to_lowercase();
            eprintln!("  [{sid}] {text:?}");

            let render = |ins: Option<&str>| -> Result<Vec<Vec<f32>>, String> {
                (0..n)
                    .map(|i| {
                        let req = QwenRequest {
                            mode: QwenMode::CustomVoice,
                            text,
                            instruct: ins,
                            instruct_effect: InstructEffect::Honored,
                        };
                        // Seeds disjoint per sentence so no two sentences share a draw.
                        engine.render_seeded(&req, (si * 1000 + i) as u64)
                    })
                    .collect()
            };
            let f = |w: &Vec<Vec<f32>>| {
                w.iter().map(|s| features(s, 24_000)).collect::<Vec<Features>>()
            };

            let plain = render(None)?;
            let fplain = f(&plain);
            let sham = render(Some(SHAM))?;
            let fsham = f(&sham);

            let series = |judge: &mut OnnxJudge, w: &[Vec<f32>]| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
                w.iter()
                    .map(|s| {
                        Ok(f64::from(
                            score_resampled(judge, s, 24_000)?.get(jl).unwrap_or(f32::NAN),
                        ))
                    })
                    .collect()
            };
            let mut label_means = |judge: &mut OnnxJudge, w: &[Vec<f32>]| -> Result<BTreeMap<String, f64>, Box<dyn std::error::Error>> {
                let mut acc: BTreeMap<String, Vec<f64>> = BTreeMap::new();
                for s in w {
                    let sc = score_resampled(judge, s, 24_000)?;
                    for l in &labels {
                        acc.entry(l.clone()).or_default().push(f64::from(sc.get(l).unwrap_or(0.0)));
                    }
                }
                Ok(acc.iter().map(|(k, v)| (k.clone(), mean(v))).collect())
            };

            let base = series(&mut judge, &plain)?;
            let (mb, sb) = (mean(&base), sd(&base));
            let base_full = label_means(&mut judge, &plain)?;

            let mut measure = |name: &str,
                               arm: TuneArm,
                               phrase: Option<&str>,
                               wav: &Vec<Vec<f32>>|
             -> Result<(), Box<dyn std::error::Error>> {
                let fw = f(wav);
                let p_plain =
                    activation_test(&fw, &fplain, th.corrected_alpha()).map(|o| o.p_value).unwrap_or(1.0);
                let p_sham =
                    activation_test(&fw, &fsham, th.corrected_alpha()).map(|o| o.p_value).unwrap_or(1.0);
                let v = series(&mut judge, wav)?;
                let delta = mean(&v) - mb;
                let noise = (sd(&v).powi(2) + sb * sb).sqrt();
                let full = label_means(&mut judge, wav)?;
                let (gain, _) = full
                    .iter()
                    .map(|(k, m)| (k.clone(), m - base_full.get(k).copied().unwrap_or(0.0)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                eprintln!(
                    "     {name:<22} p_plain={p_plain:.4} p_sham={p_sham:.4} \
                     delta={delta:+.3} +/-{noise:.3} top={gain}"
                );
                out.push(SentenceMeasurement {
                    sentence: sid.clone(),
                    m: TuneMeasurement {
                        label: label.clone(),
                        phrase: phrase.map(str::to_string),
                        arm,
                        split,
                        p_vs_plain: p_plain,
                        p_vs_sham: p_sham,
                        cued_delta: delta,
                        cued_noise: noise,
                        largest_gain_label: gain,
                        wer: 0.0,
                        speaker_similarity: 1.0,
                    },
                });
                Ok(())
            };

            measure("sham", TuneArm::Sham, None, &sham)?;
            let inc_w = render(Some(&incumbent))?;
            measure("incumbent", TuneArm::Incumbent, Some(&incumbent), &inc_w)?;
            let cc_w = render(Some(counter))?;
            measure("counter-cue", TuneArm::CounterCue, Some(counter), &cc_w)?;
            for c in &candidates {
                let w = render(Some(c))?;
                measure(&format!("cand:{}", &c[..22.min(c.len())]), TuneArm::Candidate, Some(c), &w)?;
            }
        }
    }

    // A candidate must clear the criteria on a MAJORITY of sentences on each split.
    // Pre-registered here, not chosen after seeing the numbers.
    let min_sentences = tune_set.len() / 2 + 1;
    let report = decide_per_sentence(&out, 0, &th, min_sentences);
    println!("\n=== decision ===");
    println!(
        "corrected alpha {:.5}   must clear {min_sentences} of {} sentences",
        report.corrected_alpha,
        tune_set.len()
    );
    if let Some(v) = &report.void {
        println!("ROUND VOID: {v:?}");
        println!("Nothing measured this round can be trusted. No proposal.");
        return Ok(());
    }
    for (c, cleared, per) in &report.detail {
        println!("  {:<44} cleared {cleared}/{} tune sentences", format!("{:?}", c.phrase), tune_set.len());
        for p in per.iter().filter(|p| !p.passed) {
            if let Some(r) = &p.reject {
                println!("      {:<14} {r:?}", p.sentence);
            }
        }
    }
    match &report.proposed {
        None => println!("\nNo candidate cleared every criterion. The incumbent stands."),
        Some(c) => {
            let path = std::env::var("SYRINX_TUNE_OUT")
                .unwrap_or_else(|_| format!(".opt-reports/tuned-{label}.toml"));
            // The margin is the real one, measured on the HOLDOUT split -- the split the
            // candidate was not selected on. Writing 0.0 here (as a first draft did) would
            // put a placeholder into a provenance field whose entire purpose is to record
            // what was actually measured.
            // Mean holdout delta across the sentences it was measured on -- per sentence,
            // so this is an average of independent measurements rather than one pooled
            // number, which is the whole point of the restructure.
            let hd = |phrase: &str| {
                let v: Vec<f64> = out
                    .iter()
                    .filter(|s| {
                        s.m.split == Split::Holdout && s.m.phrase.as_deref() == Some(phrase)
                    })
                    .map(|s| s.m.cued_delta)
                    .collect();
                (!v.is_empty()).then(|| mean(&v))
            };
            let margin = match (hd(&c.phrase), hd(&incumbent)) {
                (Some(a), Some(b)) => a - b,
                _ => f64::NAN,
            };
            let row = format!(
                "# PROPOSAL — inert until a human adds `accepted_by`. ADR-0004.\n\
                 [[tuned]]\nlabel = {:?}\nlang = {:?}\nbackend = \"qwen3-1.7b-customvoice\"\n\
                 phrase = {:?}\n# accepted_by = \"\"   # <- listen first, then sign\n\
                 measured_on = \"2026-09-06\"\nincumbent = {:?}\nmargin = {:.3}\n\
                 judge = \"emotion2vec/emotion2vec_plus_large\"\njudge_recall_on_class = {:.3}\n\
                 holdout_id = {:?}\nholdout_uses = 0\n\
                 notes = \"proposed by examples/tune_instruct.rs\"\n",
                c.label,
                lang_code(lang),
                c.phrase,
                incumbent,
                margin,
                judge_recall_for(&label),
                format!("{label}-2026-09-06-a"),
            );
            std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap()).ok();
            std::fs::write(&path, &row)?;
            println!("\nPROPOSED: {:?}", c.phrase);
            println!("written to {path} — INERT until a human signs `accepted_by`.");
        }
    }
    Ok(())
}
