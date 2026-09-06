//! C4.2′ — the arm-contrast decision, per ADR-0003.
//!
//! [`activation`](crate::activation) asks "did the cued renders differ from the plain
//! ones". That question has a confound the module cannot see: `assemble_text_mode`
//! prepends the instruct block as **text tokens**, so cued and plain differ in prompt
//! length and content, not only in meaning. At a fixed seed a longer prompt gives a
//! different AR trajectory whether or not the model attaches meaning to the words — a
//! model treating the instruct as pure noise would still reject the cued-vs-plain null.
//!
//! So a run carries four [`Arm`]s and the decision reads the contrasts between them:
//!
//! ```text
//! cue vs plain    an instruction of some kind changed the audio   (the weak claim)
//! sham vs plain   ANY instruction changes the audio               (the confound)
//! cue vs sham     the DELIVERY CONTENT changed the audio          (the claim we want)
//! A vs A          two halves of the plain arm                     (the calibration)
//! ```
//!
//! This module is **pure** — measurements in, verdict out — which is the point: the
//! numbers need a GPU and 26 minutes, but the decision logic is frozen-tested on every
//! board (`tests/cue_contrast_gate.rs`). A future pass that wants to "just lower the
//! margin" has to get past a test to do it.
//!
//! Everything here is additive. `evaluate_activation` and its types keep their exact
//! signatures, because `tests/cue_activation_gate.rs` is frozen against them and
//! `real_cue_activation` still certifies the fish research path.

use std::collections::BTreeMap;

/// Which condition a measurement came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Arm {
    /// No instruction at all.
    Plain,
    /// The cue's instruction.
    Cue,
    /// A delivery-neutral instruction of comparable length.
    Sham,
    /// The first half of a split plain arm (A/A calibration).
    ControlA,
    /// The second half.
    ControlB,
}

impl Arm {
    pub fn as_str(self) -> &'static str {
        match self {
            Arm::Plain => "plain",
            Arm::Cue => "cue",
            Arm::Sham => "sham",
            Arm::ControlA => "control-a",
            Arm::ControlB => "control-b",
        }
    }
}

/// One contrast between two arms of one case.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmContrast {
    pub case_id: String,
    pub backend: String,
    /// The arm being tested against `against`.
    pub arm: Arm,
    pub against: Arm,
    pub p_value: f64,
    pub effect: f64,
    /// Highest WER observed in either arm of this contrast.
    pub wer: f64,
    pub baseline_wer: f64,
}

impl ArmContrast {
    pub fn wer_delta(&self) -> f64 {
        self.wer - self.baseline_wer
    }

    /// Significant at an already-corrected alpha.
    pub fn significant_at(&self, corrected_alpha: f64) -> bool {
        self.p_value <= corrected_alpha
    }
}

/// Whether a checkpoint is expected to drop the instruction before it reaches the model.
///
/// Named for what it actually is. `qwen3-0.6b-customvoice` renders identically with and
/// without an instruction because `prompt.rs::honors_instruct` filters the string out in
/// **our** code — so this is a control on the harness, not on the model. ADR-0003 §3 A1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineControl {
    pub backend: String,
    /// The cued and plain renders must be byte-identical.
    pub renders_identical: bool,
}

/// The pre-registered decision parameters. Pre-registered means: written down before the
/// run, not chosen after seeing it.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastThresholds {
    /// Family-wise alpha, before correction.
    pub alpha: f64,
    /// Number of comparisons the run makes. Bonferroni divides `alpha` by this.
    pub comparisons: usize,
    /// Maximum A/A activations tolerated before the instrument is called broken. `None`
    /// derives it as `ceil(corrected_alpha * comparisons)`.
    pub max_aa_activations: Option<usize>,
    /// Maximum tolerated absolute WER increase, anywhere. Carried over unchanged from
    /// `Thresholds::max_wer_delta` — clause C is the one part of C4.2 that survives.
    pub max_wer_delta: f64,
    /// The pre-registered sentinel case (clause B). `None` = clause B not enabled, which
    /// ADR-0003 records in advance as an acceptable outcome.
    pub sentinel: Option<String>,
}

impl Default for ContrastThresholds {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            comparisons: 1,
            max_aa_activations: None,
            max_wer_delta: 0.5,
            sentinel: None,
        }
    }
}

impl ContrastThresholds {
    /// The Bonferroni-corrected alpha.
    ///
    /// Correction is not optional here. At 33 measurable cases and alpha=0.05, a pipeline
    /// emitting pure noise satisfies "at least one case activated" with probability
    /// 1 - 0.95^33 = 0.82 — the vacuous pass this gate exists to prevent.
    pub fn corrected_alpha(&self) -> f64 {
        self.alpha / self.comparisons.max(1) as f64
    }

    /// How many A/A activations are tolerated over `n_aa` A/A contrasts.
    ///
    /// Derived as `ceil(alpha * n_aa)` from the **nominal** alpha, because that is what the
    /// calibration is asking: at nominal size, `n_aa` tests are expected to produce
    /// `alpha * n_aa` false positives, and materially more than that means the test is not
    /// the size it claims. An earlier version used the corrected alpha times the comparison
    /// count, which is algebraically just `ceil(alpha)` — a constant 1 for any sane alpha,
    /// and therefore not a calibration at all. A surviving mutant found it.
    pub fn aa_budget(&self, n_aa: usize) -> usize {
        self.max_aa_activations.unwrap_or_else(|| (self.alpha * n_aa as f64).ceil() as usize)
    }
}

/// Why a run failed. One variant per clause, so a failure names its own clause.
#[derive(Debug, Clone, PartialEq)]
pub enum ContrastViolation {
    /// A1 — a checkpoint that must drop the instruction did not render identically.
    PipelineControlBroken { backend: String },
    /// A2 — the plain arm disagrees with itself more often than alpha allows.
    CalibrationFailed { activations: usize, budget: usize },
    /// A3 — a delivery-neutral instruction moved the audio. Every activation number in
    /// the run is uninterpretable when this fires.
    ShamActivated { case_id: String, p_value: f64, alpha: f64 },
    /// B — the pre-registered sentinel did not activate.
    SentinelSilent { case_id: String, p_value: f64, alpha: f64 },
    /// B — a sentinel was named but the run produced no contrast for it.
    SentinelNotMeasured { case_id: String },
    /// C — WER regressed beyond the veto.
    WerRegressed { case_id: String, delta: f64, limit: f64 },
    /// A cue lowered to no instruct on a backend whose caps claim to honour that kind, so
    /// cued and plain were the identical request. A lowering bug, not a null result.
    CueDroppedButHonored { case_id: String, backend: String },
}

/// What a run concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastReport {
    pub corrected_alpha: f64,
    /// Cases where the cue beat its **sham** — the claim ADR-0003 actually wants.
    pub content_activated: Vec<String>,
    /// Cases where the cue beat **plain** but not its sham. Recorded separately because
    /// the difference between these two lists is the entire point of the sham arm.
    pub perturbation_only: Vec<String>,
    /// Cases whose kind the backend cannot express, so no measurement was possible.
    /// Never reported as an activation rate of zero — that is a fabricated measurement of
    /// a channel that does not exist (ADR-0003 §5).
    pub not_applicable: Vec<String>,
    pub violations: Vec<ContrastViolation>,
}

impl ContrastReport {
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Evaluate one C4.2′ run.
///
/// `not_applicable` names cases whose kind the backend cannot express; they are counted
/// and never scored.
pub fn evaluate_contrast(
    contrasts: &[ArmContrast],
    controls: &[PipelineControl],
    not_applicable: &[String],
    th: &ContrastThresholds,
) -> ContrastReport {
    let alpha = th.corrected_alpha();
    let mut violations = Vec::new();

    // ---- clause A1: the pipeline control renders identically
    for c in controls {
        if !c.renders_identical {
            violations.push(ContrastViolation::PipelineControlBroken {
                backend: c.backend.clone(),
            });
        }
    }

    // ---- clause A2: the plain arm must agree with itself
    //
    // Checked at the NOMINAL alpha, not the corrected one. This clause asks whether the
    // test's false-positive rate matches its nominal size; running it at a corrected alpha
    // would answer a different question and would almost never fire, which is the failure
    // mode of a calibration nobody can trip.
    let n_aa = contrasts
        .iter()
        .filter(|c| c.arm == Arm::ControlA && c.against == Arm::ControlB)
        .count();
    let aa = contrasts
        .iter()
        .filter(|c| c.arm == Arm::ControlA && c.against == Arm::ControlB)
        .filter(|c| c.significant_at(th.alpha))
        .count();
    let budget = th.aa_budget(n_aa);
    if aa > budget {
        violations.push(ContrastViolation::CalibrationFailed { activations: aa, budget });
    }

    // ---- clause A3: no sham may activate
    for c in contrasts.iter().filter(|c| c.arm == Arm::Sham && c.against == Arm::Plain) {
        if c.significant_at(alpha) {
            violations.push(ContrastViolation::ShamActivated {
                case_id: c.case_id.clone(),
                p_value: c.p_value,
                alpha,
            });
        }
    }

    // ---- clause C: the WER veto, on every contrast
    for c in contrasts {
        if c.wer_delta() > th.max_wer_delta {
            violations.push(ContrastViolation::WerRegressed {
                case_id: c.case_id.clone(),
                delta: c.wer_delta(),
                limit: th.max_wer_delta,
            });
        }
    }

    // ---- the two activation lists
    let beat_sham: BTreeMap<&str, bool> = contrasts
        .iter()
        .filter(|c| c.arm == Arm::Cue && c.against == Arm::Sham)
        .map(|c| (c.case_id.as_str(), c.significant_at(alpha)))
        .collect();

    let mut content_activated = Vec::new();
    let mut perturbation_only = Vec::new();
    for c in contrasts.iter().filter(|c| c.arm == Arm::Cue && c.against == Arm::Plain) {
        if !c.significant_at(alpha) {
            continue;
        }
        // Beating plain is not enough. A cue that beats plain but not a delivery-neutral
        // string of the same length has demonstrated prompt perturbation, not steering.
        if beat_sham.get(c.case_id.as_str()).copied().unwrap_or(false) {
            content_activated.push(c.case_id.clone());
        } else {
            perturbation_only.push(c.case_id.clone());
        }
    }

    // ---- clause B: the pre-registered sentinel
    if let Some(name) = &th.sentinel {
        match contrasts
            .iter()
            .find(|c| &c.case_id == name && c.arm == Arm::Cue && c.against == Arm::Plain)
        {
            None => violations
                .push(ContrastViolation::SentinelNotMeasured { case_id: name.clone() }),
            Some(c) if !c.significant_at(alpha) => {
                violations.push(ContrastViolation::SentinelSilent {
                    case_id: name.clone(),
                    p_value: c.p_value,
                    alpha,
                })
            }
            Some(_) => {}
        }
    }

    ContrastReport {
        corrected_alpha: alpha,
        content_activated,
        perturbation_only,
        not_applicable: not_applicable.to_vec(),
        violations,
    }
}
