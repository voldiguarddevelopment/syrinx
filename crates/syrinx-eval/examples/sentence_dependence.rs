//! Is a cue's effect a property of the CUE, or of the cue and the SENTENCE together?
//!
//! Every prior conclusion that "`[angry]` does not take on this checkpoint" — four
//! independent measurements — used **one sentence**: *"You told me it was handled…"*. The
//! first tuning round (`renders/2026-09-06-tune-angry/`) rendered the same incumbent phrase
//! against a different set and got a different answer: +2.386 on `angry` with `angry` as
//! top gainer, where the original sentence gives +0.577 and `neutral`.
//!
//! That was inside the noise band at n=4, so it was recorded as a hypothesis, not a result.
//! This settles it at n=8, **per sentence**, which is the part that matters: pooling across
//! sentences is exactly what hid the effect.
//!
//! Three arms per sentence — plain, cued, and a delivery-neutral sham of comparable length,
//! because ADR-0003 makes the sham mandatory: without it a longer prompt perturbing the AR
//! trajectory is indistinguishable from a cue steering delivery.
//!
//! Usage: sentence_dependence <talker-dir> <tokenizer-dir> <label> [n]

use candle_core::Device;
use syrinx_cue::instruct::InstructTable;
use syrinx_cue::legacy_emotion::InstructLang;
use syrinx_cue::BackendId;
use syrinx_eval::acoustic::{activation_test, features, min_n_for_alpha, Features};
use syrinx_eval::affect::{
    emotion2vec9_spec, emotion2vec_label_for_cue, score_resampled, OnnxJudge,
};
use syrinx_serve::qwen::{InstructEffect, QwenEngine, QwenMode, QwenRequest};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};

/// `(id, kind, text)`. Chosen to span a spectrum rather than to be a random sample: if the
/// effect is sentence-dependent, the axis it depends on should be visible in the ordering.
///
/// `accusatory-long` is the sentence every previous run used — kept as the anchor, so this
/// run can be compared to those directly rather than by assertion. The three `holdout-*`
/// entries are the tuning holdout set, included because the hypothesis was raised ON them
/// and a confirmation that avoids them would prove nothing. That retires the partition for
/// tuning purposes: ADR-0004 §5 requires a fresh `holdout_id` before the next round.
/// `(id, kind, text)` per label. Chosen to span a spectrum rather than sampled at random:
/// if the effect is sentence-dependent, the axis it depends on should be visible in the
/// ordering.
///
/// The sets are **per label and semantically compatible with it**, deliberately. Rendering
/// `[sad]` over angry text would confound the cue with the text's own affect. That is safe
/// for the *judge* — emotion2vec reads audio only and cannot see the words, which is
/// exactly why `docs/backends/AFFECT_JUDGES.md` rules out audio LLMs here — but it is not
/// safe for the *renderer*, which does see them.
///
/// The first entry of each set is the sentence every previous run of that cue used, kept as
/// an anchor so this run can be compared to those directly rather than by assertion.
const ANGRY: [(&str, &str, &str); 6] = [
    ("accusatory-long", "long accusation",
     "You told me it was handled. You looked me in the eye and said it was handled."),
    ("holdout-imperative", "short imperative",
     "Put it back exactly where you found it, right now."),
    ("holdout-confront", "direct confrontation",
     "Do not tell me to calm down when you are the reason for this."),
    ("holdout-betrayal", "quiet reproach",
     "I trusted you with one thing and you could not even manage that."),
    ("new-exasperated", "exasperated question",
     "How many times do I have to explain the same thing to you?"),
    ("new-command", "terse command",
     "Stop talking and listen to me for once."),
];

const SAD: [(&str, &str, &str); 6] = [
    ("anchor-vigil", "quiet narrative",
     "I waited by the window until the last light went out."),
    ("loss-short", "plain statement of loss",
     "She left before I could say goodbye."),
    ("resignation", "resignation",
     "I suppose there is nothing more to be done about it."),
    ("wistful-question", "wistful question",
     "Do you ever think about how things might have gone?"),
    ("terse-finality", "terse finality",
     "It is over. There is nothing left."),
    ("reflective-long", "long reflection",
     "We used to come here every summer, and now the house belongs to someone else."),
];

fn sentences_for(label: &str) -> Result<&'static [(&'static str, &'static str, &'static str)], String> {
    match label {
        "angry" => Ok(&ANGRY),
        "sad" => Ok(&SAD),
        // No generic fallback: rendering a cue over text of a different affect measures the
        // text as much as the cue, and silently picking the wrong set would look like a
        // result. Add a set deliberately.
        other => Err(format!(
            "no sentence set for cue {other:?} — add one to sentence_dependence.rs rather \
             than reusing another cue's, which would confound the cue with the text"
        )),
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
        return Err("usage: sentence_dependence <talker-dir> <tokenizer-dir> <label> [n]".into());
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
    let instruct = InstructTable::shared()
        .phrase(&label, InstructLang::En)
        .ok_or_else(|| format!("no curated phrase for {label:?}"))?
        .to_string();

    let sentences = sentences_for(&label)?;
    // Two acoustic contrasts per sentence.
    let comparisons = sentences.len() * 2;
    let alpha = 0.05 / comparisons as f64;
    let floor = min_n_for_alpha(alpha);
    if n < floor {
        return Err(format!(
            "n={n} cannot reach alpha={alpha:.5}; the exact permutation test needs n>={floor}"
        )
        .into());
    }
    eprintln!(
        "[sent] label={label} judge-class={jl} phrase={instruct:?}\n\
         [sent] n={n}  {} sentences x 3 arms = {} renders\n\
         [sent] alpha 0.05/{comparisons} = {alpha:.5} (n floor {floor})",
        sentences.len(),
        sentences.len() * 3 * n
    );

    println!(
        "\n{:<20}{:<24}{:>11}{:>12}{:>10}{:>10}  {}",
        "sentence", "kind", "cue/plain", "sham/plain", "delta", "+/-noise", "top-gain"
    );
    let mut hits = 0usize;
    for &(id, kind, text) in sentences {
        let render = |ins: Option<&str>| -> Result<Vec<Vec<f32>>, String> {
            (0..n)
                .map(|i| {
                    let req = QwenRequest {
                        mode: QwenMode::CustomVoice,
                        text,
                        instruct: ins,
                        instruct_effect: InstructEffect::Honored,
                    };
                    engine.render_seeded(&req, i as u64)
                })
                .collect()
        };
        let plain = render(None)?;
        let cued = render(Some(&instruct))?;
        let sham = render(Some(SHAM))?;

        let f = |w: &Vec<Vec<f32>>| w.iter().map(|s| features(s, 24_000)).collect::<Vec<Features>>();
        let p_cue = activation_test(&f(&cued), &f(&plain), alpha).map(|o| o.p_value).unwrap_or(1.0);
        let p_sham = activation_test(&f(&sham), &f(&plain), alpha).map(|o| o.p_value).unwrap_or(1.0);

        // Score each render ONCE into a full per-label map. The obvious spelling — a
        // closure per label — reruns the judge nine times over the same audio and needs a
        // mutable borrow it cannot have.
        let labels = emotion2vec9_spec().labels;
        let mut score_set = |w: &Vec<Vec<f32>>| -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
            w.iter()
                .map(|s| {
                    let sc = score_resampled(&mut judge, s, 24_000)?;
                    Ok(labels.iter().map(|l| f64::from(sc.get(l).unwrap_or(0.0))).collect())
                })
                .collect()
        };
        let (sp, sc_) = (score_set(&plain)?, score_set(&cued)?);
        let col = |m: &Vec<Vec<f64>>, i: usize| m.iter().map(|r| r[i]).collect::<Vec<f64>>();
        let ji = labels.iter().position(|l| l == jl).expect("cued class in vocabulary");
        let (bp, bc) = (col(&sp, ji), col(&sc_, ji));
        let delta = mean(&bc) - mean(&bp);
        let noise = (sd(&bc).powi(2) + sd(&bp).powi(2)).sqrt();

        // Which class gained most — a phrase that moves `neutral` is not steering anger.
        let mut best = (String::new(), f64::MIN);
        for (i, l) in labels.iter().enumerate() {
            let d = mean(&col(&sc_, i)) - mean(&col(&sp, i));
            if d > best.1 {
                best = (l.clone(), d);
            }
        }

        let flag = if p_cue <= alpha && delta > noise && best.0 == label { " <== BOTH" } else { "" };
        if !flag.is_empty() {
            hits += 1;
        }
        println!(
            "{id:<20}{kind:<24}{p_cue:>11.4}{p_sham:>12.4}{delta:>+10.3}{noise:>10.3}  {}{flag}",
            best.0
        );
    }
    println!(
        "\n{hits} of {} sentences show BOTH an acoustic change and a judge move toward `{label}`.",
        sentences.len()
    );
    println!(
        "A cue that were simply dead would show 0. A cue that were unconditionally live would\n\
         show all {}. Anything between is sentence-dependence, and says the earlier verdict\n\
         was a property of the sentence it was measured on.",
        sentences.len()
    );
    Ok(())
}
