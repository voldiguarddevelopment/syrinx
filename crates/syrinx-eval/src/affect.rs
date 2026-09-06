//! Measuring the **affect** of a render — in Rust, with no Python at inference.
//!
//! # Why this module exists
//!
//! The cue layer lowers `[happy]` into a backend instruction, and until now nothing could
//! say whether the resulting audio actually became happier. WER (`syrinx-stt`) only checks
//! that the words survived; SIM-o only checks that the voice stayed the same person. Both
//! are silent about expression, which is the one thing the cue was for.
//!
//! This module runs a speech-emotion model over a waveform and reports what it saw. It is
//! a **screen**, not a certificate — see [the honesty section](#what-this-can-and-cannot-tell-you).
//!
//! # The judge is pluggable, deliberately
//!
//! The licence landscape for SER checkpoints is unstable and the technically-best model is
//! not always the usable one (`docs/LICENSES.md` records one such reversal: audEERING's
//! dimensional arousal/valence model was rejected on CC-BY-NC-SA, and a categorical
//! Apache-2.0 model adopted in its place). So nothing here is welded to one checkpoint:
//!
//! * [`AffectJudge`] is the interface — a name, a label vocabulary, a [`ScoreKind`], a
//!   sample rate, and `score(&[f32]) -> `[`AffectScore`].
//! * [`OnnxJudge`] is the only implementation, and it is *configured* rather than coded:
//!   [`OnnxJudgeSpec`] carries the labels, the tensor names and the output activation, so
//!   a new model is a new spec plus an `.onnx` file. [`ravdess8_spec`] is the one shipped
//!   spec; a dimensional judge would differ only in `kind` and `activation`.
//! * [`ScoreKind`] is what lets a consumer stay correct across that swap: categorical
//!   scores are a simplex (moving one class necessarily moves others), dimensional scores
//!   are independent axes. Code that reads a score without checking the kind is wrong for
//!   one of the two.
//!
//! # The adopted judge, and the trap inside it
//!
//! `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (Apache-2.0), 8 RAVDESS
//! classes. Its `config.json` claims `architectures: ["Wav2Vec2ForSequenceClassification"]`
//! and that is **false** — the checkpoint's head is `classifier.dense` + `classifier.output`,
//! which that class does not have, so `AutoModelForAudioClassification` silently discards
//! the head and runs a randomly-initialised one. The real head (mean-pool -> dense -> tanh
//! -> output) and the exact non-linearity were established by measurement in
//! `scripts/probe-affect-head.py`, and both it and the feature extractor are baked into the
//! ONNX graph by `scripts/export-affect-onnx.py`. That is why [`OnnxJudge::score`] hands the
//! graph a **raw** waveform and owns no feature-extraction code: the graph normalises
//! internally, and the export script asserts that it does.
//!
//! # What this can and cannot tell you
//!
//! `CLAUDE.md` puts "intended emotion" firmly in blocked-on-human territory, and nothing
//! here changes that. Concretely, measured on this box on 2026-09-06 by
//! `scripts/probe-affect-head.py` (180 held-out CREMA-D clips) and by
//! `tests/real_qwen_affect.rs`:
//!
//! * Cross-corpus accuracy **0.394** against a 0.167 chance baseline — above chance, but
//!   far from the 0.822 its card reports in-domain on its own training corpus. Per class
//!   it is good at `angry` (recall 0.80) and `neutral` (0.73), middling at `happy` (0.30,
//!   leaking to `surprised`) and `disgust` (0.27), and poor at `sad` (0.17, leaking to
//!   `calm`) and `fearful` (0.10).
//! * **Absolute scores are close to meaningless.** On ten seconds of ordinary neutral
//!   LibriSpeech read speech it answers `disgust` 0.57; on a real acted *sad* clip it
//!   answers `happy` 0.70. Only a delta between two clips that differ in one controlled
//!   way is worth reading — the same lesson SIM-o taught, where 0.90 looked high and was
//!   below a different speaker.
//! * The verdict is not even stable within one clip: scoring three overlapping 60 %
//!   windows of the same 3.5 s render moves the worst class by 0.24-0.54. Any delta below
//!   a clip's own window spread is noise. `examples/affect_judge.rs` computes that floor
//!   and refuses to call a direction inside it.
//! * It is a model of **acted human** speech applied to **synthetic** speech. Two domain
//!   gaps stacked. A null result means "this judge cannot tell", never "the cue did
//!   nothing".
//!
//! So: use it for screening and regression detection, report the whole probability vector
//! rather than the argmax, and never turn a direction verdict into a passing gate.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor as OrtTensor;

/// Everything that can go wrong scoring a clip.
#[derive(Debug)]
pub enum AffectError {
    /// The ONNX runtime failed to build a session or run the graph.
    Runtime(String),
    /// The graph's inputs/outputs are not what the spec says they are.
    Contract(String),
    /// The clip is empty, or too short for the model's convolutional front end.
    TooShort { samples: usize, minimum: usize },
}

impl std::fmt::Display for AffectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AffectError::Runtime(m) => write!(f, "affect: onnx runtime: {m}"),
            AffectError::Contract(m) => write!(f, "affect: graph contract: {m}"),
            AffectError::TooShort { samples, minimum } => {
                write!(f, "affect: clip is {samples} samples, need at least {minimum}")
            }
        }
    }
}

impl std::error::Error for AffectError {}

impl<R> From<ort::Error<R>> for AffectError {
    fn from(e: ort::Error<R>) -> Self {
        AffectError::Runtime(e.to_string())
    }
}

/// How to read an [`AffectScore`]'s numbers — the one thing a consumer must branch on to
/// stay correct when the judge is swapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreKind {
    /// A probability distribution over the labels: non-negative, sums to 1. Classes are
    /// coupled, so "class A rose" always means "the others fell".
    Categorical,
    /// Independent continuous axes, each roughly in 0..1 (arousal, valence, ...). One axis
    /// can move without any other moving.
    Dimensional,
}

/// One judge's verdict on one clip: a value per label, plus how to read them.
#[derive(Debug, Clone)]
pub struct AffectScore {
    /// Label names, in the same order as `values`.
    pub labels: Vec<String>,
    /// See [`ScoreKind`].
    pub kind: ScoreKind,
    /// One number per label.
    pub values: Vec<f32>,
}

impl AffectScore {
    /// The value for `label`, or `None` if this judge does not have that label.
    pub fn get(&self, label: &str) -> Option<f32> {
        self.labels.iter().position(|l| l == label).map(|i| self.values[i])
    }

    /// The highest-scoring label and its value.
    ///
    /// Provided for logging only. Prefer the whole vector: a cue that moves the right
    /// class from 0.20 to 0.40 without overtaking the leader is a real effect that the
    /// argmax throws away.
    pub fn top(&self) -> (&str, f32) {
        let mut best = 0usize;
        for i in 1..self.values.len() {
            if self.values[i] > self.values[best] {
                best = i;
            }
        }
        (self.labels[best].as_str(), self.values[best])
    }
}

/// What a judge must be able to do. See the module docs for why this is an interface.
pub trait AffectJudge {
    /// A short identifier for logs and reports (the checkpoint, essentially).
    fn name(&self) -> &str;
    /// The judge's own output vocabulary.
    fn labels(&self) -> &[String];
    /// How its numbers should be read.
    fn kind(&self) -> ScoreKind;
    /// The sample rate it expects; callers resample to this (see [`score_resampled`]).
    fn sample_rate(&self) -> u32;
    /// Score one mono clip that is ALREADY at [`Self::sample_rate`].
    fn score(&mut self, samples: &[f32]) -> Result<AffectScore, AffectError>;
}

/// Resample `samples` from `sr` to the judge's rate, then score.
///
/// The resampler is `syrinx_qwen::speaker::resample` (64-lobe Lanczos, 0.96 passband),
/// reused rather than reimplemented because it was itself tuned against a reference: an
/// earlier 16-lobe version leaked imaging above the input Nyquist and measurably corrupted
/// a speaker embedding (`tests/real_qwen_speaker_parity.rs`, the driver-path bound). A
/// second resampler in the tree would be a second chance to make that mistake.
pub fn score_resampled(
    judge: &mut dyn AffectJudge,
    samples: &[f32],
    sr: u32,
) -> Result<AffectScore, AffectError> {
    let want = judge.sample_rate();
    if sr == want {
        judge.score(samples)
    } else {
        judge.score(&syrinx_qwen::speaker::resample(samples, sr, want))
    }
}

/// What the ONNX output should have applied to it before it is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputActivation {
    /// The graph emits raw logits; turn them into a probability simplex.
    Softmax,
    /// The graph already emits the reportable numbers.
    None,
}

/// Everything that distinguishes one ONNX judge from another. Adding a model means adding
/// one of these plus an `.onnx` file — not touching [`OnnxJudge`].
#[derive(Debug, Clone)]
pub struct OnnxJudgeSpec {
    /// Identifier for reports.
    pub name: String,
    /// Output vocabulary, in the graph's output order.
    pub labels: Vec<String>,
    /// How the numbers should be read once `activation` has been applied.
    pub kind: ScoreKind,
    /// Rate the graph's waveform input expects.
    pub sample_rate: u32,
    /// Name of the waveform input tensor, `[1, time]` f32.
    pub input: String,
    /// Name of the score output tensor, `[1, labels]` f32.
    pub output: String,
    /// Name of an optional pooled-embedding output, `[1, dim]` f32. Not used for scoring;
    /// it exists so a parity test can localise a disagreement to the trunk or the head.
    pub embedding: Option<String>,
    /// See [`OutputActivation`].
    pub activation: OutputActivation,
}

/// The spec for the adopted judge: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
/// as exported by `scripts/export-affect-onnx.py`.
///
/// The label order is `config.id2label`'s and is asserted against the graph at load time.
pub fn ravdess8_spec() -> OnnxJudgeSpec {
    OnnxJudgeSpec {
        name: "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition".to_string(),
        labels: RAVDESS8_LABELS.iter().map(|s| s.to_string()).collect(),
        kind: ScoreKind::Categorical,
        sample_rate: 16_000,
        input: "signal".to_string(),
        output: "logits".to_string(),
        embedding: Some("hidden_states".to_string()),
        activation: OutputActivation::Softmax,
    }
}

/// The 8 RAVDESS classes, in `config.id2label` order.
pub const RAVDESS8_LABELS: [&str; 8] = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
];

/// The cue-vocabulary id that each RAVDESS class corresponds to.
///
/// Only the six labels with an unambiguous counterpart in `crates/syrinx-cue/vocab.toml`
/// are mapped. `calm` and `neutral` are the judge's own; the vocab's `calm` maps to the
/// judge's `calm`, and it has no `neutral` because an absent cue *is* neutral.
///
/// Returns the judge label for a cue id, or `None` when the cue has no counterpart — which
/// is most of them. The vocabulary has 25 emotions and 8 classes cover six of them; a
/// direction check for `[sarcastic]` is simply not available from this judge, and saying so
/// is better than mapping it onto the nearest class and pretending.
pub fn ravdess_label_for_cue(cue_id: &str) -> Option<&'static str> {
    match cue_id {
        "happy" => Some("happy"),
        "sad" => Some("sad"),
        "angry" => Some("angry"),
        "calm" => Some("calm"),
        "afraid" => Some("fearful"),
        "surprised" => Some("surprised"),
        "disgusted" => Some("disgust"),
        _ => None,
    }
}

/// The 9 emotion2vec+ classes, in `tokens.txt` order.
///
/// `other` and `unknown` are the model's own escape hatches and map to no cue. They are
/// kept in the vocabulary because the graph emits them and dropping a column would
/// misalign every index after it.
pub const EMOTION2VEC9_LABELS: [&str; 9] = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
];

/// The spec for `emotion2vec/emotion2vec_plus_large`, as exported by
/// `scripts/export-emotion2vec-onnx.py`.
///
/// **The graph emits logits and this spec does NOT soften them.** funasr's own inference
/// ends in a softmax, but the model is saturated — on real speech the logit spread is
/// 19–26, so softmax returns a one-hot vector to float precision. The whole measurement is
/// "cued mean minus plain mean against seed noise", and a one-hot score makes every delta
/// 0 or ±1. `docs/LICENSES.md` already carries the rule this follows — "use the full
/// probability vector, never the argmax" — and a saturated softmax *is* an argmax.
///
/// **The exported graph accepts at most 160,079 samples (10.005 s @ 16 kHz).** The AUDIO
/// encoder's position-bias buffer is 499 frames and the export bakes it; beyond that,
/// `OnnxJudge::read` returns an `ort` error rather than a wrong number. Measured by binary
/// search, not assumed — see `scripts/export-emotion2vec-onnx.py`. Renders scored by this
/// judge are 3–6 s, so the limit is not in the way; anything longer must be chunked by the
/// caller, deliberately, because chunking changes what "the emotion of this clip" means.
///
/// The consequence is that a reading is [`ScoreKind::Dimensional`], not `Categorical`: the
/// numbers are unbounded per-class evidence, they do not sum to 1, and a rise in one class
/// does not imply a fall in another. Anything comparing these across clips must difference
/// them, never treat them as probabilities.
pub fn emotion2vec9_spec() -> OnnxJudgeSpec {
    OnnxJudgeSpec {
        name: "emotion2vec/emotion2vec_plus_large".to_string(),
        labels: EMOTION2VEC9_LABELS.iter().map(|s| s.to_string()).collect(),
        kind: ScoreKind::Dimensional,
        sample_rate: 16_000,
        input: "signal".to_string(),
        output: "logits".to_string(),
        embedding: None,
        activation: OutputActivation::None,
    }
}

/// The emotion2vec+ class each cue id corresponds to, or `None` when the judge has no
/// counterpart for it.
///
/// Seven of the nine classes are reachable from the cue vocabulary — one more than the
/// RAVDESS judge, and without RAVDESS's `calm`, which the vocab has but this model does
/// not. As with [`ravdess_label_for_cue`], a cue with no counterpart returns `None` rather
/// than being mapped onto the nearest class: "no reading available for `[sarcastic]`" is
/// the honest answer and "it looks a bit angry" is not.
pub fn emotion2vec_label_for_cue(cue_id: &str) -> Option<&'static str> {
    match cue_id {
        "happy" => Some("happy"),
        "sad" => Some("sad"),
        "angry" => Some("angry"),
        "afraid" => Some("fearful"),
        "surprised" => Some("surprised"),
        "disgusted" => Some("disgusted"),
        // The vocab has no `neutral` cue — an absent cue IS neutral — but the class is
        // reachable as a *reference* level, so it is named here for lookups that want it.
        "neutral" => Some("neutral"),
        _ => None,
    }
}

/// An [`AffectJudge`] backed by an ONNX graph, run through `ort` on the CPU.
pub struct OnnxJudge {
    session: Session,
    spec: OnnxJudgeSpec,
    out_idx: usize,
    emb_idx: Option<usize>,
}

/// A single forward pass: the reportable score plus the raw tensors behind it.
#[derive(Debug, Clone)]
pub struct Reading {
    /// The score, after [`OnnxJudgeSpec::activation`].
    pub score: AffectScore,
    /// The graph's `output` tensor, before the activation.
    pub raw: Vec<f32>,
    /// The graph's `embedding` tensor, if the spec names one.
    pub embedding: Option<Vec<f32>>,
}

impl OnnxJudge {
    /// Load `onnx_path` and check it against `spec`.
    ///
    /// Single intra-op thread and full graph optimisation, mirroring
    /// `syrinx_frontend::speech_token::SpeechTokenizer` — the tree's other `ort` session —
    /// so results are reproducible run to run.
    pub fn load(onnx_path: impl AsRef<Path>, spec: OnnxJudgeSpec) -> Result<Self, AffectError> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(onnx_path.as_ref())?;

        let inputs: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        if !inputs.iter().any(|n| *n == spec.input) {
            return Err(AffectError::Contract(format!(
                "graph has inputs {inputs:?}, spec wants {:?}",
                spec.input
            )));
        }
        let outputs: Vec<String> =
            session.outputs().iter().map(|o| o.name().to_string()).collect();
        let out_idx = outputs.iter().position(|n| *n == spec.output).ok_or_else(|| {
            AffectError::Contract(format!(
                "graph has outputs {outputs:?}, spec wants {:?}",
                spec.output
            ))
        })?;
        let emb_idx = match &spec.embedding {
            None => None,
            Some(name) => Some(outputs.iter().position(|n| n == name).ok_or_else(|| {
                AffectError::Contract(format!(
                    "graph has outputs {outputs:?}, spec wants embedding {name:?}"
                ))
            })?),
        };

        Ok(OnnxJudge { session, spec, out_idx, emb_idx })
    }

    /// The spec this judge was loaded with.
    pub fn spec(&self) -> &OnnxJudgeSpec {
        &self.spec
    }

    /// One forward pass over a mono clip already at [`OnnxJudgeSpec::sample_rate`],
    /// returning everything the graph produced.
    ///
    /// The waveform goes in **raw**: the exported graph performs the model's own
    /// zero-mean/unit-variance normalisation internally, which `scripts/export-affect-onnx.py`
    /// verifies by feeding the same clip at three gains. Scaling here would be a second,
    /// silently-wrong normalisation.
    pub fn read(&mut self, samples: &[f32]) -> Result<Reading, AffectError> {
        // The wav2vec2 conv front end strides 320 samples over a 400-sample window: below
        // that there is no frame at all and the graph fails deep inside a matmul with an
        // unreadable message. 400 samples is 25 ms.
        const MIN_SAMPLES: usize = 400;
        if samples.len() < MIN_SAMPLES {
            return Err(AffectError::TooShort { samples: samples.len(), minimum: MIN_SAMPLES });
        }

        let signal = OrtTensor::from_array((vec![1_i64, samples.len() as i64], samples.to_vec()))?;
        let outputs =
            self.session.run(ort::inputs![self.spec.input.as_str() => signal])?;

        let (shape, raw) = outputs[self.out_idx].try_extract_tensor::<f32>()?;
        let n = self.spec.labels.len();
        if raw.len() != n {
            return Err(AffectError::Contract(format!(
                "output {:?} has shape {shape:?} ({} values), spec declares {n} labels",
                self.spec.output,
                raw.len()
            )));
        }
        let raw = raw.to_vec();

        let embedding = match self.emb_idx {
            None => None,
            Some(i) => Some(outputs[i].try_extract_tensor::<f32>()?.1.to_vec()),
        };

        let values = match self.spec.activation {
            OutputActivation::None => raw.clone(),
            OutputActivation::Softmax => softmax(&raw),
        };

        Ok(Reading {
            score: AffectScore {
                labels: self.spec.labels.clone(),
                kind: self.spec.kind,
                values,
            },
            raw,
            embedding,
        })
    }
}

impl AffectJudge for OnnxJudge {
    fn name(&self) -> &str {
        &self.spec.name
    }
    fn labels(&self) -> &[String] {
        &self.spec.labels
    }
    fn kind(&self) -> ScoreKind {
        self.spec.kind
    }
    fn sample_rate(&self) -> u32 {
        self.spec.sample_rate
    }
    fn score(&mut self, samples: &[f32]) -> Result<AffectScore, AffectError> {
        Ok(self.read(samples)?.score)
    }
}

/// Numerically-stable softmax (subtract the max before exponentiating).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|v| v / sum).collect()
}

// ---------------------------------------------------------------------------
// direction: did the cue move the render toward what it named?
// ---------------------------------------------------------------------------

/// One label's before/after within a plain-vs-tagged pair.
#[derive(Debug, Clone)]
pub struct LabelDelta {
    /// The judge's label.
    pub label: String,
    /// Its value on the un-cued render.
    pub plain: f32,
    /// Its value on the cued render.
    pub tagged: f32,
}

impl LabelDelta {
    /// `tagged - plain`.
    pub fn delta(&self) -> f32 {
        self.tagged - self.plain
    }
}

/// What one plain/tagged pair looked like to the judge.
///
/// **Reported, never asserted.** See the module docs: there is no honest threshold to
/// gate on here, and inventing one would turn a screening instrument into a fake
/// certificate.
#[derive(Debug, Clone)]
pub struct DirectionReport {
    /// The cue vocabulary id that was applied to the tagged render.
    pub cue_id: String,
    /// The judge label the cue maps to, if it has one.
    pub cued_label: Option<String>,
    /// Every label, in the judge's order.
    pub deltas: Vec<LabelDelta>,
}

impl DirectionReport {
    /// Movement of the cued label, `tagged - plain`. `None` when the cue has no
    /// counterpart in this judge's vocabulary — which is a real, reportable answer, not a
    /// zero.
    pub fn cued_delta(&self) -> Option<f32> {
        let want = self.cued_label.as_deref()?;
        self.deltas.iter().find(|d| d.label == want).map(|d| d.delta())
    }

    /// The label that gained the most, and how much.
    pub fn largest_gain(&self) -> (&str, f32) {
        let mut best = 0usize;
        for i in 1..self.deltas.len() {
            if self.deltas[i].delta() > self.deltas[best].delta() {
                best = i;
            }
        }
        (self.deltas[best].label.as_str(), self.deltas[best].delta())
    }
}

/// Compare a cued render against its un-cued control.
///
/// `cue_id` is a `crates/syrinx-cue/vocab.toml` emotion id; it is mapped to the judge's
/// vocabulary with [`ravdess_label_for_cue`]. Both scores must come from the same judge.
pub fn direction(cue_id: &str, plain: &AffectScore, tagged: &AffectScore) -> DirectionReport {
    let deltas = plain
        .labels
        .iter()
        .enumerate()
        .map(|(i, label)| LabelDelta {
            label: label.clone(),
            plain: plain.values[i],
            tagged: tagged.values[i],
        })
        .collect();
    DirectionReport {
        cue_id: cue_id.to_string(),
        cued_label: ravdess_label_for_cue(cue_id).map(|s| s.to_string()),
        deltas,
    }
}
