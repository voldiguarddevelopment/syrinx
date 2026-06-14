//! syrinx-eval — the eval-harness skeleton (task T-00.03).
//!
//! This is the deterministic substrate that the real metrics (SIM-o, WER, MOS,
//! latency) plug into later. It computes nothing about audio: a [`MetricSet`] is
//! a pluggable map from each of the five fixed [`REQUIRED_KEYS`] to a closure
//! `Fn(&StubInput) -> Option<f64>`, and [`run`] drives that set over a stub input
//! to write a metrics JSON whose top-level object always carries exactly those
//! five keys — each a finite number or an explicit `null`.
//!
//! The five-key, present-or-null schema is the invariant every run upholds. A
//! metric that yields `None` is serialized as JSON `null` (its key kept, never
//! omitted); a metric set that omits a required key is a typed [`EvalError`],
//! with no partial file written.

use std::path::Path;

/// The five metric keys the metrics JSON must always carry, in schema order.
pub const REQUIRED_KEYS: [&str; 5] = ["sim_o", "wer", "mos_proxy", "ttfb_ms", "rtf"];

/// The stub synth input the harness skeleton runs over: no audio, no model — it
/// exists only so registered metrics share a typed input until real ones land.
pub struct StubInput;

impl StubInput {
    /// The single stub synth input used by the skeleton.
    pub fn stub() -> Self {
        StubInput
    }
}

/// A single pluggable metric: maps the stub input to `Some(value)` to record a
/// number, or `None` to record an explicit JSON `null`.
type Metric = Box<dyn Fn(&StubInput) -> Option<f64>>;

/// A pluggable set of metrics keyed by name. [`run`] requires the set to cover
/// every one of [`REQUIRED_KEYS`]; extra keys are never emitted.
pub struct MetricSet {
    entries: Vec<(String, Metric)>,
}

impl MetricSet {
    /// An empty set. Populate it with [`register`](MetricSet::register).
    pub fn new() -> Self {
        MetricSet {
            entries: Vec::new(),
        }
    }

    /// Register `metric` under `key`. The metric is invoked once per [`run`].
    pub fn register<F>(&mut self, key: &str, metric: F)
    where
        F: Fn(&StubInput) -> Option<f64> + 'static,
    {
        self.entries.push((key.to_string(), Box::new(metric)));
    }

    /// The default five-metric set: one stub metric per required key, each
    /// yielding a finite number.
    pub fn default_stub() -> Self {
        let mut set = MetricSet::new();
        set.register("sim_o", |_| Some(0.90));
        set.register("wer", |_| Some(0.05));
        set.register("mos_proxy", |_| Some(4.0));
        set.register("ttfb_ms", |_| Some(120.0));
        set.register("rtf", |_| Some(0.30));
        set
    }

    /// The metric registered under `key`, if any.
    fn get(&self, key: &str) -> Option<&Metric> {
        self.entries
            .iter()
            .find(|(k, _)| k.as_str() == key)
            .map(|(_, metric)| metric)
    }
}

impl Default for MetricSet {
    fn default() -> Self {
        MetricSet::new()
    }
}

/// A typed harness failure.
#[derive(Debug)]
pub enum EvalError {
    /// The metric set omitted this required key; no metrics file was written.
    MissingKey(String),
}

/// Run `metrics` over `input` and write the metrics JSON to `out_path`.
///
/// Returns [`EvalError::MissingKey`] — writing nothing — if `metrics` omits any
/// of [`REQUIRED_KEYS`]. Otherwise writes a JSON object with exactly those five
/// keys: each metric's `Some(x)` as the number `x`, each `None` as `null`.
pub fn run(input: &StubInput, metrics: &MetricSet, out_path: &Path) -> Result<(), EvalError> {
    let mut fields: Vec<String> = Vec::new();
    for key in REQUIRED_KEYS {
        let metric = match metrics.get(key) {
            Some(metric) => metric,
            None => return Err(EvalError::MissingKey(key.to_string())),
        };
        let rendered = match metric(input) {
            Some(value) => value.to_string(),
            None => "null".to_string(),
        };
        fields.push(format!("\"{key}\":{rendered}"));
    }
    let json = format!("{{{}}}", fields.join(","));
    std::fs::write(out_path, json).expect("failed to write metrics JSON");
    Ok(())
}
