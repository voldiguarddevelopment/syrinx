//! Evaluating the Qwen3-TTS path — natively, with no Python at inference.
//!
//! ## Why this is not just another `evaluate_*`
//!
//! [`crate::metrics`]'s CosyVoice-era functions obtain WER by shelling out to
//! `scripts/eval_wer.py` through `SYRINX_WER_HELPER`, and report `None` when that helper
//! is unset. That predates `syrinx-stt`, the in-tree pure-Rust Whisper that `CLAUDE.md`
//! now names as "the native WER oracle". C4.1 is explicit that only ONNX / own-runtime
//! models may produce measurements — no Python at inference — so this module scores with
//! `syrinx-stt` directly. Nothing here shells out, and nothing returns a silent `None`
//! because a helper was missing.
//!
//! ## What it measures, and against what
//!
//! The WER reference is the **plan's clean text**, not the caller's input. Those differ
//! whenever the input carries cues: `[happy] hello` is planned down to `hello` plus an
//! instruction, and `hello` is what a correct backend says. Scoring against the raw input
//! would charge the engine for markup it is required not to speak, so a perfectly correct
//! render would look broken. It also means a cue that DID leak shows up as a WER penalty
//! rather than passing unnoticed.
//!
//! ## Activation, and why it currently REFUSES on this engine
//!
//! [`measure_activation`] feeds the same engine into [`crate::acoustic`] and
//! [`crate::activation`]. Qwen ought to be the ideal subject: it is `Inline::None` +
//! `Granularity::Utterance`, so a cue reaches it ONLY as an utterance-scoped instruction,
//! and the per-checkpoint rows say in advance which checkpoints must move and which must
//! not — a harness can be checked against known ground truth, which an open-vocabulary
//! backend never offers.
//!
//! It cannot be measured through this interface yet, and the function says so rather than
//! returning a number. [`QwenRequest`] carries **no seed**, and the engine renders with
//! `DriveParams::default()`, so every call for the same request is bit-identical. With
//! zero within-condition variance the permutation test stops estimating anything: two
//! groups of identical vectors make the observed split the most extreme of all labelings
//! by construction, so ANY deterministic difference reports `activated` at the smallest
//! attainable p. That is exactly the naive "is there a difference at all" comparison
//! [`crate::acoustic`] exists to reject — it would report ~100 % activation for a backend
//! that ignores cues entirely.
//!
//! So [`measure_activation`] checks for that degeneracy and errors. Closing it needs a
//! seed (or an equivalent draw selector) on `QwenRequest`, which belongs to
//! `syrinx-serve`; until then this is an honest refusal rather than a fabricated
//! measurement.

use std::time::Instant;

use syrinx_serve::qwen::{plan, QwenEngine, QwenPlanError, QwenRequest};
use syrinx_stt::{wer, Stt};

use crate::acoustic::{activation_test, features, Features};
use crate::activation::Measurement;

/// Qwen renders at 24 kHz mono.
pub const SAMPLE_RATE: u32 = 24_000;

/// One scored render.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenScore {
    pub id: String,
    /// WER of the transcript against the plan's clean text.
    pub wer: f64,
    /// Synthesis wall-time divided by the duration of the audio produced.
    pub rtf: f64,
    pub audio_secs: f64,
    /// Number of requests the cue layer produced (>1 means a cue forced a split).
    pub segments: usize,
    pub transcript: String,
    /// What the plan asked the backend to do, for reporting alongside the numbers.
    pub instruct: Option<String>,
}

/// One case to evaluate: an id, the authored text (cues allowed), and its language.
#[derive(Debug, Clone, Copy)]
pub struct QwenCase<'a> {
    pub id: &'a str,
    pub text: &'a str,
    pub lang: &'a str,
}

/// Render one case and score it. `oracle` is the native Whisper.
///
/// Every segment the plan produced is rendered and concatenated, because a cue-driven
/// split is one utterance to the listener and scoring the pieces separately would measure
/// something nobody hears.
pub fn evaluate_one<E: QwenEngine>(
    engine: &E,
    backend: syrinx_cue::BackendId,
    oracle: &Stt,
    case: QwenCase<'_>,
) -> Result<QwenScore, String> {
    let p = plan(backend, case.text).map_err(|e: QwenPlanError| e.to_string())?;

    let t0 = Instant::now();
    let mut samples: Vec<f32> = Vec::new();
    for seg in &p.segments {
        let body = seg.text.trim();
        if body.is_empty() {
            continue;
        }
        let req = QwenRequest {
            mode: p.mode,
            text: body,
            instruct: seg.instruct.as_deref(),
            instruct_effect: p.instruct_effect,
        };
        samples.extend(engine.render(&req)?);
    }
    let gen_secs = t0.elapsed().as_secs_f64();
    if samples.is_empty() {
        return Err(format!("{}: rendered no audio", case.id));
    }

    let audio_secs = samples.len() as f64 / f64::from(SAMPLE_RATE);
    let transcript = oracle
        .transcribe_lang(&samples, SAMPLE_RATE, Some(case.lang))
        .map_err(|e| format!("{}: transcribe: {e:?}", case.id))?
        .text;

    // Against the PLAN's text — what a correct backend actually says. See the module docs.
    let reference = p.text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(QwenScore {
        id: case.id.to_string(),
        wer: f64::from(wer(&reference, &transcript)),
        rtf: gen_secs / audio_secs,
        audio_secs,
        segments: p.segments.len(),
        transcript,
        instruct: p.segments.first().and_then(|s| s.instruct.clone()),
    })
}

/// Score a suite, keeping every case's result even if one fails.
pub fn evaluate_suite<E: QwenEngine>(
    engine: &E,
    backend: syrinx_cue::BackendId,
    oracle: &Stt,
    cases: &[QwenCase<'_>],
) -> Vec<Result<QwenScore, String>> {
    cases.iter().map(|c| evaluate_one(engine, backend, oracle, *c)).collect()
}

/// Render one text `n` times at consecutive seeds and summarize each render acoustically.
///
/// The engine owns its own seeding, so this asks for `n` independent renders rather than
/// passing a seed: what matters is that they are separate draws from the same condition,
/// which is what the permutation test needs to estimate sampling noise.
fn render_n<E: QwenEngine>(
    engine: &E,
    mode: syrinx_serve::qwen::QwenMode,
    effect: syrinx_serve::qwen::InstructEffect,
    text: &str,
    instruct: Option<&str>,
    n: usize,
) -> Result<Vec<Features>, String> {
    (0..n)
        .map(|_| {
            let req = QwenRequest { mode, text, instruct, instruct_effect: effect };
            let wav = engine.render(&req)?;
            Ok(features(&wav, SAMPLE_RATE))
        })
        .collect()
}

/// Does the cue measurably change this checkpoint's audio, beyond its own sampling noise?
///
/// Renders the cued and un-cued conditions `n` times each and applies the exact
/// permutation test in [`crate::acoustic`]. `alpha` and the minimum `n` are that module's
/// contract — a smaller `n` cannot reject at all and is refused there rather than quietly
/// answering "not activated".
///
/// The WER of both conditions is measured too, so a cue that "activates" purely by
/// destroying intelligibility is visible rather than counted as a success.
pub fn measure_activation<E: QwenEngine>(
    engine: &E,
    backend: syrinx_cue::BackendId,
    oracle: &Stt,
    case: QwenCase<'_>,
    n: usize,
    alpha: f64,
) -> Result<(Measurement, String), String> {
    let p = plan(backend, case.text).map_err(|e: QwenPlanError| e.to_string())?;
    let clean = p.text.split_whitespace().collect::<Vec<_>>().join(" ");
    let instruct = p.segments.first().and_then(|s| s.instruct.clone());

    // The control is the same sentence with no instruction at all — not a different
    // sentence, so the only thing that varies is the cue.
    // Control first, so the degeneracy check below can refuse before paying for the
    // cued group as well.
    let plain = render_n(engine, p.mode, p.instruct_effect, &clean, None, n)?;

    // Refuse if the engine is deterministic. Without within-condition spread the
    // permutation test has no noise to test against and would call any difference
    // significant — see the module docs. Checked on the CONTROL group, which shares one
    // condition by construction, so any spread there is genuine sampling noise.
    let degenerate = plain
        .windows(2)
        .all(|w| w[0].v.iter().zip(w[1].v.iter()).all(|(a, b)| (a - b).abs() < f64::EPSILON));
    if degenerate && plain.len() > 1 {
        return Err(format!(
            "{}: the engine rendered {n} bit-identical control takes, so there is no \
             sampling noise to test against. QwenRequest carries no seed and the engine \
             uses DriveParams::default(), so every render is the same draw; a permutation \
             test on identical groups reports ANY difference as activated. Refusing to \
             produce that number — give the engine a per-render seed first.",
            case.id
        ));
    }

    let cued = render_n(engine, p.mode, p.instruct_effect, &clean, instruct.as_deref(), n)?;
    let outcome = activation_test(&cued, &plain, alpha)
        .ok_or_else(|| format!("{}: n={n} cannot reject at alpha={alpha}", case.id))?;

    let score_of = |instr: Option<&str>| -> Result<f64, String> {
        let req =
            QwenRequest { mode: p.mode, text: &clean, instruct: instr, instruct_effect: p.instruct_effect };
        let wav = engine.render(&req)?;
        let t = oracle
            .transcribe_lang(&wav, SAMPLE_RATE, Some(case.lang))
            .map_err(|e| format!("{}: transcribe: {e:?}", case.id))?;
        Ok(f64::from(wer(&clean, &t.text)))
    };
    let wer_cued = score_of(instruct.as_deref())?;
    let wer_plain = score_of(None)?;

    let detail = format!(
        "p={:.4} effect={:.3} instruct={:?}",
        outcome.p_value, outcome.effect, instruct
    );
    Ok((
        Measurement {
            case_id: case.id.to_string(),
            backend: backend.as_str().to_string(),
            activated: outcome.activated,
            wer: wer_cued,
            baseline_wer: wer_plain,
        },
        detail,
    ))
}

// ---------------------------------------------------------------- speaker similarity

/// Cosine similarity between two speaker embeddings, i.e. **SIM-o**.
///
/// `CLAUDE.md` lists SIM-o among the blocked-on-human perceptual work. That grouping was
/// right when it was written and is no longer: SIM-o is not perceptual at all — it is a
/// cosine between speaker embeddings, fully objective — and it was blocked only because
/// nothing in-tree could produce the embeddings. `syrinx_qwen::speaker::SpeakerEncoder`
/// now can, anchored to the reference at 1e-6 (2048-wide) and 6e-7 (1024-wide). MOS is
/// the part that genuinely still needs ears or a MOS-prediction model.
///
/// **What this does not give you.** The encoder scoring the clone belongs to the same
/// family that produced it, so this measures "did the render land where the reference
/// lands in Qwen's own speaker space" — informative, and enough to catch a clone that
/// ignored its reference entirely, but weaker than a genuinely independent verifier. The
/// CosyVoice path in [`crate::metrics`] uses CAM++, a separate model, which is the
/// stronger arrangement. Read a high score here as "not obviously wrong" rather than as
/// proof of identity, and do not compare these numbers against CAM++ SIM-o from the
/// literature — different encoders, different scales.
///
/// Both clips must already be at the encoder's rate; `embed` refuses otherwise rather
/// than resampling silently.
pub fn speaker_similarity(
    encoder: &syrinx_qwen::speaker::SpeakerEncoder,
    reference: &[f32],
    render: &[f32],
    sample_rate: u32,
) -> Result<f64, String> {
    let a = encoder.embed(reference, sample_rate).map_err(|e| format!("embed reference: {e}"))?;
    let b = encoder.embed(render, sample_rate).map_err(|e| format!("embed render: {e}"))?;
    let av: Vec<f32> = a.flatten_all().and_then(|t| t.to_vec1()).map_err(|e| e.to_string())?;
    let bv: Vec<f32> = b.flatten_all().and_then(|t| t.to_vec1()).map_err(|e| e.to_string())?;
    if av.len() != bv.len() {
        return Err(format!("embedding widths differ: {} vs {}", av.len(), bv.len()));
    }
    let dot: f64 = av.iter().zip(&bv).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = av.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    let nb: f64 = bv.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return Err("a zero-norm embedding — the clip was probably silent".to_string());
    }
    Ok(dot / (na * nb))
}
