//! Cue **activation** harness (C4.1) and its gate thresholds (C4.2).
//!
//! ## What activation means, and what this module can and cannot certify
//!
//! A cue *activated* if it measurably changed the audio relative to the same sentence
//! without the cue. Deciding that requires **running the model**, which needs weights and a
//! GPU — so this module computes and gates the metrics but does not itself synthesize.
//! Callers supply [`Measurement`]s from a run; the harness aggregates them per kind and per
//! backend, applies the thresholds, and emits the JSON.
//!
//! That split is deliberate and is the honest line: the aggregation, the thresholds, the
//! frozen-set checksum and the regression gate are all deterministic and CI-gateable. The
//! **numbers** are only as real as the run that produced them, and a run that never
//! happened yields no measurements rather than a green.
//!
//! Per the spec, only ONNX / own-runtime models may be used to produce measurements — no
//! Python at inference.

use std::collections::BTreeMap;

/// One case in the frozen cue set.
#[derive(Debug, Clone, PartialEq)]
pub struct CueCase {
    pub id: String,
    /// `emotion` / `style` / `event` / `neutral`.
    pub kind: String,
    pub label: String,
    pub text: String,
    pub lang: String,
    pub placement: String,
}

impl CueCase {
    /// Parse one JSONL line of the frozen set.
    ///
    /// Hand-rolled rather than pulling serde_json into this crate: the schema is six flat
    /// string fields and the file is frozen, so a dependency would buy nothing.
    pub fn from_json_line(line: &str) -> Option<Self> {
        let get = |key: &str| -> Option<String> {
            let pat = format!("\"{key}\":");
            let i = line.find(&pat)? + pat.len();
            let rest = line[i..].trim_start();
            let rest = rest.strip_prefix('"')?;
            let mut out = String::new();
            let mut chars = rest.chars();
            while let Some(c) = chars.next() {
                match c {
                    '\\' => out.push(chars.next()?),
                    '"' => return Some(out),
                    _ => out.push(c),
                }
            }
            None
        };
        Some(Self {
            id: get("id")?,
            kind: get("kind")?,
            label: get("label")?,
            text: get("text")?,
            lang: get("lang")?,
            placement: get("placement")?,
        })
    }

    /// Load the frozen set from JSONL text.
    pub fn parse_set(src: &str) -> Vec<Self> {
        src.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(Self::from_json_line)
            .collect()
    }
}

/// One measured outcome from an actual synthesis run.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub case_id: String,
    pub backend: String,
    /// Did the cue measurably change the audio versus the un-cued control?
    pub activated: bool,
    /// WER of the cued render, from the native Whisper oracle.
    pub wer: f64,
    /// WER of the same sentence rendered without the cue.
    pub baseline_wer: f64,
}

impl Measurement {
    /// The cost of the cue in intelligibility. Positive means the cue made it worse.
    pub fn wer_delta(&self) -> f64 {
        self.wer - self.baseline_wer
    }
}

/// Aggregated numbers for one (backend, kind) cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub backend: String,
    pub kind: String,
    pub n: usize,
    /// Fraction of cues that measurably changed the audio.
    pub activation_rate: f64,
    /// Mean WER delta against the un-cued baseline.
    pub wer_delta: f64,
}

/// The gate thresholds (C4.2 starting values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Minimum activation for **events** on an open-vocabulary backend.
    pub min_event_activation: f64,
    /// Maximum tolerated absolute WER increase, anywhere.
    pub max_wer_delta: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { min_event_activation: 0.85, max_wer_delta: 0.5 }
    }
}

/// A threshold breach.
#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    pub backend: String,
    pub kind: String,
    pub metric: String,
    pub value: f64,
    pub limit: f64,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}: {} = {:.3} (limit {:.3})",
            self.backend, self.kind, self.metric, self.value, self.limit
        )
    }
}

/// The harness result.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationReport {
    pub cells: Vec<Cell>,
    pub violations: Vec<Violation>,
}

impl ActivationReport {
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    /// The JSON the AC calls for: per-kind / per-backend activation rate and WER delta.
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n  \"cells\": [\n");
        for (i, c) in self.cells.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"backend\": \"{}\", \"kind\": \"{}\", \"n\": {}, \
                 \"activation_rate\": {:.6}, \"wer_delta\": {:.6}}}{}\n",
                c.backend,
                c.kind,
                c.n,
                c.activation_rate,
                c.wer_delta,
                if i + 1 == self.cells.len() { "" } else { "," }
            ));
        }
        s.push_str("  ],\n  \"violations\": [\n");
        for (i, v) in self.violations.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"backend\": \"{}\", \"kind\": \"{}\", \"metric\": \"{}\", \
                 \"value\": {:.6}, \"limit\": {:.6}}}{}\n",
                v.backend,
                v.kind,
                v.metric,
                v.value,
                v.limit,
                if i + 1 == self.violations.len() { "" } else { "," }
            ));
        }
        s.push_str(&format!("  ],\n  \"passed\": {}\n}}\n", self.passed()));
        s
    }
}

/// Aggregate measurements into per-(backend, kind) cells and apply the thresholds.
///
/// `open_vocab_backends` names the backends whose event activation is gated — the event
/// threshold only makes sense where the backend can express events at all.
pub fn evaluate_activation(
    cases: &[CueCase],
    measurements: &[Measurement],
    open_vocab_backends: &[String],
    th: Thresholds,
) -> ActivationReport {
    let kind_of: BTreeMap<&str, &str> =
        cases.iter().map(|c| (c.id.as_str(), c.kind.as_str())).collect();

    // (backend, kind) -> (activated count, n, summed wer delta)
    let mut acc: BTreeMap<(String, String), (usize, usize, f64)> = BTreeMap::new();
    for m in measurements {
        let Some(kind) = kind_of.get(m.case_id.as_str()) else { continue };
        if *kind == "neutral" {
            continue; // the control group defines the baseline; it has no cue to activate
        }
        let e = acc.entry((m.backend.clone(), (*kind).to_string())).or_insert((0, 0, 0.0));
        e.0 += usize::from(m.activated);
        e.1 += 1;
        e.2 += m.wer_delta();
    }

    let mut cells = Vec::with_capacity(acc.len());
    let mut violations = Vec::new();
    for ((backend, kind), (hits, n, sum_delta)) in acc {
        let activation_rate = if n == 0 { 0.0 } else { hits as f64 / n as f64 };
        let wer_delta = if n == 0 { 0.0 } else { sum_delta / n as f64 };

        if kind == "event" && open_vocab_backends.contains(&backend)
            && activation_rate < th.min_event_activation
        {
            violations.push(Violation {
                backend: backend.clone(),
                kind: kind.clone(),
                metric: "activation_rate".into(),
                value: activation_rate,
                limit: th.min_event_activation,
            });
        }
        if wer_delta > th.max_wer_delta {
            violations.push(Violation {
                backend: backend.clone(),
                kind: kind.clone(),
                metric: "wer_delta".into(),
                value: wer_delta,
                limit: th.max_wer_delta,
            });
        }
        cells.push(Cell { backend, kind, n, activation_rate, wer_delta });
    }
    ActivationReport { cells, violations }
}
