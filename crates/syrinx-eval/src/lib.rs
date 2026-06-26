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

/// Real measured metrics (SIM-o / RTF / TTFB), behind the `real` feature.
#[cfg(feature = "real")]
pub mod real;

use std::collections::BTreeMap;
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
    entries: BTreeMap<String, Metric>,
}

impl MetricSet {
    /// An empty set. Populate it with [`register`](MetricSet::register).
    pub fn new() -> Self {
        MetricSet {
            entries: BTreeMap::new(),
        }
    }

    /// Register `metric` under `key`. The metric is invoked once per [`run`].
    pub fn register<F>(&mut self, key: &str, metric: F)
    where
        F: Fn(&StubInput) -> Option<f64> + 'static,
    {
        self.entries.insert(key.to_string(), Box::new(metric));
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
        self.entries.get(key)
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

// ===== Frozen-eval-set checksum mechanism (task T-00.04) =====
//
// The eval set is immutable by construction: a manifest records the SHA-256 of
// every file under the eval-set directory, keyed by relative path, and `verify`
// reports `Ok(())` iff the on-disk set is byte-for-byte and membership-identical
// to that manifest. Any drift — a tampered byte, a missing file, an extra file —
// becomes a typed error naming the path at fault.

use sha2::{Digest, Sha256};
use std::io;

/// A checksum manifest: a map from each eval-set file's relative path (with `/`
/// separators) to its lowercase-hex SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    entries: BTreeMap<String, String>,
}

impl Manifest {
    /// The `relative-path -> hex-digest` map this manifest records.
    pub fn entries(&self) -> &BTreeMap<String, String> {
        &self.entries
    }

    /// Persist the manifest to `path`, one `relpath\tdigest` line per entry in
    /// sorted path order.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut text = String::new();
        for (rel, digest) in &self.entries {
            text.push_str(rel);
            text.push('\t');
            text.push_str(digest);
            text.push('\n');
        }
        std::fs::write(path, text)
    }

    /// Load a manifest previously written by [`write`](Manifest::write).
    pub fn load(path: &Path) -> io::Result<Manifest> {
        let text = std::fs::read_to_string(path)?;
        let mut entries = BTreeMap::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let (rel, digest) = line
                .split_once('\t')
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed manifest line"))?;
            entries.insert(rel.to_string(), digest.to_string());
        }
        Ok(Manifest { entries })
    }
}

/// A typed verification failure naming the relative path at fault.
#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// A manifested file's bytes changed: its recomputed digest no longer matches.
    DigestMismatch { path: String },
    /// File membership drifted: `path` is in the manifest but missing on disk, or
    /// on disk but absent from the manifest.
    MembershipMismatch { path: String },
}

/// Walk `dir` and build a manifest mapping each file's relative path to its
/// lowercase-hex SHA-256 digest, one entry per file in the set.
pub fn build_manifest(dir: &Path) -> io::Result<Manifest> {
    let mut entries = BTreeMap::new();
    collect_digests(dir, dir, &mut entries)?;
    Ok(Manifest { entries })
}

/// Recursively hash every file under `current`, keying on the path relative to
/// `root` with `/` separators.
fn collect_digests(
    root: &Path,
    current: &Path,
    entries: &mut BTreeMap<String, String>,
) -> io::Result<()> {
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_digests(root, &path, entries)?;
        } else {
            let rel = relative_path(root, &path);
            entries.insert(rel, hash_file(&path)?);
        }
    }
    Ok(())
}

/// Recompute every digest under `dir` and check membership against `manifest`.
///
/// `Ok(())` iff the set is byte-for-byte and membership-identical to `manifest`.
pub fn verify(dir: &Path, manifest: &Manifest) -> Result<(), VerifyError> {
    let mut on_disk = BTreeMap::new();
    collect_digests(dir, dir, &mut on_disk)
        .map_err(|_| VerifyError::MembershipMismatch { path: dir.display().to_string() })?;

    // A manifested file missing on disk is a membership drift.
    for rel in manifest.entries.keys() {
        if !on_disk.contains_key(rel) {
            return Err(VerifyError::MembershipMismatch { path: rel.clone() });
        }
    }
    // A disk file absent from the manifest is a membership drift.
    for rel in on_disk.keys() {
        if !manifest.entries.contains_key(rel) {
            return Err(VerifyError::MembershipMismatch { path: rel.clone() });
        }
    }
    // Membership matches: a differing digest is a tampered file.
    for (rel, digest) in &manifest.entries {
        if on_disk.get(rel) != Some(digest) {
            return Err(VerifyError::DigestMismatch { path: rel.clone() });
        }
    }
    Ok(())
}

/// The path of `file` relative to `root`, with `/` as the separator.
fn relative_path(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).unwrap_or(file);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// The lowercase-hex SHA-256 digest of `path`'s bytes.
fn hash_file(path: &Path) -> io::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}
