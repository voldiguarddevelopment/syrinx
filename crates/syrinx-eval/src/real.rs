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
    let t0 = Instant::now();
    let wav = synth
        .synthesize(
            input.text,
            input.prompt_text,
            input.ref_wav_16k,
            input.ref_wav_24k,
            &inputs,
        )
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
        // WER via an external Whisper ASR helper (QA-side only — the inference path
        // stays pure-Rust); `None` when `SYRINX_WER_HELPER` is unset.
        wer: wer_via_helper(&wav, input.text),
        mos_proxy: None, // needs a MOS-prediction model (UTMOS/DNSMOS) — not built.
        ttfb_ms,
        rtf,
    })
}

/// Optional WER/CER via an external ASR helper. `SYRINX_WER_HELPER` is a command
/// prefix (e.g. `"micromamba run -n syrinx python scripts/eval_wer.py"`); the synth
/// output WAV path and the reference text are appended, and the helper prints the
/// error rate as a float on its last stdout line. Returns `None` if the var is unset
/// or the helper fails — the pure-Rust path never depends on it.
fn wer_via_helper(wav: &[f32], reference: &str) -> Option<f64> {
    let cmd = std::env::var("SYRINX_WER_HELPER").ok().filter(|s| !s.is_empty())?;
    let tmp = std::env::temp_dir().join("syrinx_wer_eval.wav");
    wavio::write_wav_24k(&tmp, wav).ok()?;
    let mut parts = cmd.split_whitespace();
    let prog = parts.next()?;
    let out = std::process::Command::new(prog)
        .args(parts)
        .arg(&tmp)
        .arg(reference)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "syrinx-eval: WER helper failed: {}",
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
