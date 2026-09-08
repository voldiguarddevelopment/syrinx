//! Can Qwen's instruct channel produce a **laugh**, when the event cue channel cannot?
//!
//! `[laughs]` and the other twelve event cues are `event = unsupported` on all five Qwen
//! checkpoints. `pass_hoist` filters them via `caps.can_express` before a render happens, so
//! cued and plain are the identical request and the audio is bit-identical. Every backend
//! that *does* honour events — fish-s1-mini, fish-s2-pro, cosyvoice2/3 — is deprecated.
//!
//! But "the event channel is unsupported" is not the same claim as "the model cannot laugh".
//! Qwen has one expressive channel, an utterance-scoped natural-language instruction, and
//! nobody has asked it to laugh through that. This asks.
//!
//! # Measuring it is the hard part
//!
//! The affect judge is nine EMOTION classes; it has no laughter class, so it cannot answer
//! directly. (`SenseVoiceSmall` can — audio-event detection was its one unique capability,
//! and it was passed over precisely because no shipping backend had an event channel to
//! test. If this run says the instruct channel is one, that decision is worth revisiting.)
//!
//! So this triangulates from three model-free signals plus one weak model-based one:
//!
//!   * **duration** — a render that laughs before speaking is longer. Necessary, not
//!     sufficient: an instruct that merely slows delivery also lengthens.
//!   * **WER against the clean text** — a laugh is non-lexical audio the oracle must either
//!     transcribe as something (insertions) or skip. Either way it perturbs WER, and the
//!     direction distinguishes "extra sound" from "slower speech", which does not.
//!   * **acoustic activation** — did anything change at all, beyond seed noise.
//!   * **judge `happy` delta** — laughter is not `happy`, but a real laugh plausibly drags
//!     it. Reported, never relied on.
//!
//! **None of these can prove a laugh.** They can show whether something changed and rule
//! out the boring explanations. WAVs are written for every arm because the decisive test is
//! ears, and per CLAUDE.md that is never automated into a passing test.
//!
//! Usage: event_induction <talker-dir> <tokenizer-dir> [n]

use std::collections::BTreeMap;

use candle_core::Device;
use syrinx_cue::BackendId;
use syrinx_eval::acoustic::{activation_test, features, min_n_for_alpha, Features};
use syrinx_eval::affect::{emotion2vec9_spec, score_resampled, OnnxJudge};
use syrinx_serve::qwen::{InstructEffect, QwenEngine, QwenMode, QwenRequest};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};
use syrinx_stt::{wer, Stt};

/// Sentences a laugh could plausibly precede or colour, without the TEXT being funny —
/// the words are held identical across arms, so anything heard is the instruction.
const SENTENCES: [(&str, &str); 3] = [
    ("s0", "You are not going to believe what happened next."),
    ("s1", "I told him it would never work, and here we are."),
    ("s2", "Well, that is one way to solve the problem."),
];

/// The candidate instructions. Deliberately varied in what they ask for: a discrete laugh
/// before speech, laughter carried in the voice, and an amused manner — three different
/// things that "make it laugh" could mean.
const ARMS: [(&str, Option<&str>); 6] = [
    ("plain", None),
    ("sham", Some("Read the sentence that follows")),
    ("laugh-then", Some("Laugh briefly, then speak")),
    ("begin-laugh", Some("Begin with a short laugh, then say the line")),
    ("in-voice", Some("Speak with laughter in your voice")),
    ("amused", Some("Speak as if you are trying not to laugh")),
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
    if a.len() < 2 {
        return Err("usage: event_induction <talker-dir> <tokenizer-dir> [n]".into());
    }
    let (talker, tok) = (&a[0], &a[1]);
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse().ok()) {
        Some(i) => Device::new_cuda(i)?,
        None => Device::Cpu,
    };
    let engine = QwenModelEngine::load(
        BackendId::Qwen17bCustomVoice,
        talker,
        tok,
        dev.clone(),
        QwenVoice::Preset("serena".into()),
    )?;
    if !engine.honors_seed() {
        return Err("engine does not honour seeds".into());
    }
    let stt = std::env::var("SYRINX_STT_MODEL_DIR").ok().and_then(|d| Stt::load(&d, Device::Cpu).ok());
    if stt.is_none() {
        eprintln!("[event] SYRINX_STT_MODEL_DIR unset — the WER signal will not be measured");
    }
    let mut judge = std::env::var("SYRINX_EMOTION2VEC_ONNX")
        .ok()
        .and_then(|p| OnnxJudge::load(&p, emotion2vec9_spec()).ok());

    // Two contrasts per (sentence, instruct arm): vs plain and vs sham.
    let instructed = ARMS.len() - 2;
    let comparisons = SENTENCES.len() * instructed * 2;
    let alpha = 0.05 / comparisons as f64;
    let floor = min_n_for_alpha(alpha);
    if n < floor {
        return Err(format!("n={n} cannot reach alpha={alpha:.5}; need n>={floor}").into());
    }
    let out_dir = std::env::var("SYRINX_EVENT_OUT")
        .unwrap_or_else(|_| ".opt-reports/event-induction".to_string());
    std::fs::create_dir_all(&out_dir)?;
    eprintln!(
        "[event] n={n}  {} sentences x {} arms = {} renders  alpha 0.05/{comparisons}={alpha:.5} \
         (floor {floor})\n[event] wavs -> {out_dir}",
        SENTENCES.len(),
        ARMS.len(),
        SENTENCES.len() * ARMS.len() * n
    );

    let labels = emotion2vec9_spec().labels;
    let happy_i = labels.iter().position(|l| l == "happy").expect("happy class");

    println!(
        "\n{:<5}{:<14}{:>9}{:>10}{:>10}{:>10}{:>11}",
        "sent", "arm", "dur(s)", "d_dur", "acou p", "wer", "d_happy"
    );
    for (sid, text) in SENTENCES {
        let mut base_dur = 0.0f64;
        let mut base_feats: Vec<Features> = Vec::new();
        let mut sham_feats: Vec<Features> = Vec::new();
        let mut base_happy = 0.0f64;

        for (name, ins) in ARMS {
            let wav: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let req = QwenRequest {
                        mode: QwenMode::CustomVoice,
                        text,
                        instruct: ins,
                        instruct_effect: InstructEffect::Honored,
                    };
                    engine.render_seeded(&req, i as u64)
                })
                .collect::<Result<_, _>>()?;

            let durs: Vec<f64> = wav.iter().map(|w| w.len() as f64 / 24_000.0).collect();
            let (dm, ds) = (mean(&durs), sd(&durs));
            let f: Vec<Features> = wav.iter().map(|w| features(w, 24_000)).collect();

            // WER against the requested words. A laugh is non-lexical: the oracle either
            // invents tokens for it or drops them, and both move this.
            let w = match stt.as_ref() {
                Some(s) => {
                    let v: Vec<f64> = wav
                        .iter()
                        .map(|x| {
                            let t = s.transcribe(x, 24_000).map(|t| t.text).unwrap_or_default();
                            f64::from(wer(text, &t))
                        })
                        .collect();
                    mean(&v)
                }
                None => f64::NAN,
            };

            let happy = match judge.as_mut() {
                Some(j) => {
                    let v: Vec<f64> = wav
                        .iter()
                        .map(|x| {
                            score_resampled(j, x, 24_000)
                                .map(|s| f64::from(s.get(&labels[happy_i]).unwrap_or(0.0)))
                                .unwrap_or(f64::NAN)
                        })
                        .collect();
                    mean(&v)
                }
                None => f64::NAN,
            };

            if name == "plain" {
                base_dur = dm;
                base_feats = f.clone();
                base_happy = happy;
            }
            if name == "sham" {
                sham_feats = f.clone();
            }
            let p = if name == "plain" {
                f64::NAN
            } else {
                activation_test(&f, &base_feats, alpha).map(|o| o.p_value).unwrap_or(1.0)
            };

            // One WAV per arm per sentence, first seed, for listening.
            let path = format!("{out_dir}/{sid}-{name}.wav");
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 24_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut wr = hound::WavWriter::create(&path, spec)?;
            for s in &wav[0] {
                wr.write_sample(*s)?;
            }
            wr.finalize()?;

            println!(
                "{sid:<5}{name:<14}{dm:>9.2}{:>10}{:>10}{w:>10.3}{:>11}",
                if name == "plain" { "  --".into() } else { format!("{:+.2}", dm - base_dur) },
                if p.is_nan() { "  --".into() } else { format!("{p:.4}") },
                if happy.is_nan() || base_happy.is_nan() {
                    "  --".into()
                } else {
                    format!("{:+.3}", happy - base_happy)
                },
            );
            let _ = (&sham_feats, ds);
        }
    }
    println!(
        "\nRead this as a SCREEN, not a verdict. Duration alone cannot separate \"laughed\"\n\
         from \"spoke slower\"; WER moving with duration is the pair that distinguishes\n\
         extra non-lexical sound from slower speech. The decisive test is the WAVs."
    );
    Ok(())
}
