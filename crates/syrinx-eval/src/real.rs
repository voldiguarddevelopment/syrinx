//! Real evaluation metrics — measured, not stubbed (behind the `real` feature).
//!
//! Computes the quality/latency metrics for one `(text, reference voice)` input by
//! running the real CosyVoice2 [`Synthesizer`]:
//!   * `sim_o`    — speaker cosine (CAM++) between the reference clip and the output
//!                  (the zero-shot voice-clone fidelity signal);
//!   * `rtf`      — real-time factor: synthesis wall-time / output audio duration;
//!   * `ttfb_ms`  — time to the first streaming chunk (first-byte latency);
//!   * `wer`      — `None`: needs an ASR model (Whisper); none is on the box yet;
//!   * `mos_proxy`— `None`: needs a MOS-prediction model (UTMOS/DNSMOS); not built.
//!
//! `None` is serialized as JSON `null` — an honest "not measured", never a fake
//! constant. The three measured metrics replace the stub `0.90/0.05/4.0/...`.

use std::time::Instant;

use syrinx_serve::synth::{SynthInputs, Synthesizer};
use syrinx_serve::wavio;

const SR_24K: u32 = 24_000;
const SR_16K: u32 = 16_000;

/// The five-key metric record (matches [`crate::REQUIRED_KEYS`]); each `None`
/// serializes as JSON `null`.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub sim_o: Option<f64>,
    pub wer: Option<f64>,
    pub mos_proxy: Option<f64>,
    pub ttfb_ms: Option<f64>,
    pub rtf: Option<f64>,
}

impl Metrics {
    /// Serialize to the canonical five-key JSON object (each value a finite number
    /// or `null`), in [`crate::REQUIRED_KEYS`] order.
    pub fn to_json(&self) -> String {
        let f = |v: Option<f64>| v.map(|x| x.to_string()).unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"sim_o\":{},\"wer\":{},\"mos_proxy\":{},\"ttfb_ms\":{},\"rtf\":{}}}",
            f(self.sim_o),
            f(self.wer),
            f(self.mos_proxy),
            f(self.ttfb_ms),
            f(self.rtf)
        )
    }
}

/// One real eval input: the text to speak + the reference voice (prompt transcript
/// and its 16 kHz/24 kHz mono waveforms, as the synthesizer consumes them).
pub struct EvalInput<'a> {
    pub text: &'a str,
    pub prompt_text: &'a str,
    pub ref_wav_16k: &'a [f32],
    pub ref_wav_24k: &'a [f32],
}

/// Cosine similarity of two equal-length vectors (0 if either is zero-norm).
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Evaluate one input on `synth`, measuring `sim_o` / `rtf` / `ttfb_ms` (and
/// leaving `wer` / `mos_proxy` as honest `None`). `max_gen_steps` caps live LM
/// decoding so CPU runs stay tractable (`None` = the real ratio).
pub fn evaluate(
    synth: &mut Synthesizer,
    input: &EvalInput<'_>,
    max_gen_steps: Option<usize>,
) -> Result<Metrics, String> {
    let inputs = SynthInputs {
        lm_seed: 0,
        max_gen_steps,
        ..Default::default()
    };

    // --- RTF: full synthesis wall-time vs the rendered audio duration. ---
    // SYRINX_QUALITY_SOURCE=1 evaluates the real random-phase NSF source
    // (`synthesize_quality`) instead of the deterministic zero-phase smoke source —
    // so MOS-proxy A/Bs the two sources. Unset = the default deterministic path.
    // SYRINX_INSTRUCT=<instruction> evaluates the emotion/instruct path
    // (`synthesize_instruct`, the instruction takes the prompt-text role) — so MOS/SIM-o
    // A/B different emotions. Else SYRINX_QUALITY_SOURCE picks the random-phase source.
    let t0 = Instant::now();
    let wav = if let Some(instruct) = std::env::var("SYRINX_INSTRUCT").ok().filter(|s| !s.is_empty()) {
        eprintln!("real_eval_metrics: instruct = {instruct:?}");
        synth.synthesize_instruct(
            input.text,
            &instruct,
            input.ref_wav_16k,
            input.ref_wav_24k,
            &inputs,
        )
    } else if std::env::var("SYRINX_QUALITY_SOURCE").is_ok() {
        synth.synthesize_quality(
            input.text,
            input.prompt_text,
            input.ref_wav_16k,
            input.ref_wav_24k,
            &inputs,
            0,
        )
    } else {
        synth.synthesize(
            input.text,
            input.prompt_text,
            input.ref_wav_16k,
            input.ref_wav_24k,
            &inputs,
        )
    }
    .map_err(|e| e.to_string())?;
    let synth_secs = t0.elapsed().as_secs_f64();
    let audio_secs = wav.len() as f64 / SR_24K as f64;
    let rtf = (audio_secs > 0.0).then(|| synth_secs / audio_secs);

    // --- SIM-o: speaker cosine between the reference and the synthesized output,
    //     both embedded through the same CAM++ path. ---
    let out_16k = wavio::resample(&wav, SR_24K, SR_16K);
    let ref_emb = synth
        .speaker_embedding(input.ref_wav_16k)
        .map_err(|e| e.to_string())?;
    let out_emb = synth.speaker_embedding(&out_16k).map_err(|e| e.to_string())?;
    let ref_v: Vec<f32> = ref_emb
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(|e| e.to_string())?;
    let out_v: Vec<f32> = out_emb
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(|e| e.to_string())?;
    let sim_o = Some(cosine(&ref_v, &out_v));

    // --- TTFB: time to the first emitted streaming chunk. ---
    let t1 = Instant::now();
    let mut ttfb_ms: Option<f64> = None;
    synth
        .synthesize_streaming(
            input.text,
            input.prompt_text,
            input.ref_wav_16k,
            input.ref_wav_24k,
            &inputs,
            25,
            |_chunk| {
                if ttfb_ms.is_none() {
                    ttfb_ms = Some(t1.elapsed().as_secs_f64() * 1000.0);
                }
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;

    Ok(Metrics {
        sim_o,
        // WER (Whisper ASR) + MOS-proxy (UTMOS) via external eval helpers — QA-side only,
        // the inference path stays pure-Rust. Each is `None` when its env var is unset.
        wer: audio_helper(&wav, "SYRINX_WER_HELPER", Some(input.text)),
        mos_proxy: audio_helper(&wav, "SYRINX_MOS_HELPER", None),
        ttfb_ms,
        rtf,
    })
}

/// One multilingual eval case: a [`evaluate`] input plus the ASR `lang` hint that
/// the WER helper (`scripts/eval_wer.py`, via `SYRINX_WER_LANG`) must use for *this*
/// case's transcription. Cross-lingual is expressed by giving a reference voice in
/// one language (`prompt_text` / `ref_wav_*`) while `text` + `lang` are another: the
/// zero-shot clone carries the voice across into the target language.
pub struct EvalCase<'a> {
    /// Whisper language hint for the WER ASR of this case's output (`"zh"`, `"en"`, …).
    pub lang: &'a str,
    /// The text to synthesize (may be in a different language than the reference).
    pub text: &'a str,
    /// Transcript of the reference clip (the cloned voice), in the reference's language.
    pub prompt_text: &'a str,
    /// Reference voice waveform at 16 kHz mono (CAM++ / speech-token path).
    pub ref_wav_16k: &'a [f32],
    /// Reference voice waveform at 24 kHz mono (the synthesizer's prompt audio).
    pub ref_wav_24k: &'a [f32],
}

/// Evaluate a suite of [`EvalCase`]s and tag each result with a `lang | text` label.
///
/// A thin loop over [`evaluate`]: per case it sets `SYRINX_WER_LANG` to the case's
/// `lang` (so the Whisper WER helper transcribes in that language — the per-case
/// language wiring), runs the unchanged single-case [`evaluate`], then restores the
/// prior `SYRINX_WER_LANG`. A case whose synthesis errors is logged to stderr and
/// omitted, so a short result vector flags a failure to the caller. Aggregate the
/// returned rows with [`aggregate`].
pub fn evaluate_suite(
    synth: &mut Synthesizer,
    cases: &[EvalCase<'_>],
    max_gen_steps: Option<usize>,
) -> Vec<(String, Metrics)> {
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        // The WER helper reads SYRINX_WER_LANG from its environment; set it to this
        // case's language for the duration of the call, then restore the prior value.
        let prev = std::env::var("SYRINX_WER_LANG").ok();
        std::env::set_var("SYRINX_WER_LANG", case.lang);

        let input = EvalInput {
            text: case.text,
            prompt_text: case.prompt_text,
            ref_wav_16k: case.ref_wav_16k,
            ref_wav_24k: case.ref_wav_24k,
        };
        let label = case_label(case);
        match evaluate(synth, &input, max_gen_steps) {
            Ok(m) => results.push((label, m)),
            Err(e) => eprintln!("evaluate_suite: case `{label}` failed: {e}"),
        }

        match prev {
            Some(v) => std::env::set_var("SYRINX_WER_LANG", v),
            None => std::env::remove_var("SYRINX_WER_LANG"),
        }
    }
    results
}

/// A short, stable label for a case: `lang | <first chars of text>`.
fn case_label(case: &EvalCase<'_>) -> String {
    let snippet: String = case.text.chars().take(24).collect();
    format!("{} | {}", case.lang, snippet)
}

/// Mean of each metric across `results`, skipping `None` values per metric. A metric
/// that is `None` in every row stays `None` (no rows to average); otherwise the value
/// is the arithmetic mean of the present values for that metric.
pub fn aggregate(results: &[(String, Metrics)]) -> Metrics {
    let mean = |select: fn(&Metrics) -> Option<f64>| -> Option<f64> {
        let vals: Vec<f64> = results.iter().filter_map(|(_, m)| select(m)).collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    };
    Metrics {
        sim_o: mean(|m| m.sim_o),
        wer: mean(|m| m.wer),
        mos_proxy: mean(|m| m.mos_proxy),
        ttfb_ms: mean(|m| m.ttfb_ms),
        rtf: mean(|m| m.rtf),
    }
}

/// Run an external audio-eval helper and parse a float from its last stdout line.
/// `env_var` holds a command prefix (e.g. `"micromamba run -n syrinx python scripts/eval_mos.py"`);
/// the synth output WAV path is appended, plus `extra` (the reference text, for WER) when given.
/// Used for both WER (`SYRINX_WER_HELPER` + reference) and MOS-proxy (`SYRINX_MOS_HELPER`, no extra).
/// Returns `None` if the var is unset or the helper fails — the pure-Rust path never depends on it.
fn audio_helper(wav: &[f32], env_var: &str, extra: Option<&str>) -> Option<f64> {
    let cmd = std::env::var(env_var).ok().filter(|s| !s.is_empty())?;
    let tmp = std::env::temp_dir().join("syrinx_eval_audio.wav");
    wavio::write_wav_24k(&tmp, wav).ok()?;
    let mut parts = cmd.split_whitespace();
    let prog = parts.next()?;
    let mut command = std::process::Command::new(prog);
    command.args(parts).arg(&tmp);
    if let Some(e) = extra {
        command.arg(e);
    }
    let out = command.output().ok()?;
    if !out.status.success() {
        eprintln!(
            "syrinx-eval: {env_var} helper failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .lines()
        .last()?
        .trim()
        .parse::<f64>()
        .ok()
}
