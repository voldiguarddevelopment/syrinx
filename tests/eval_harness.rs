//! Frozen RED tests for T-00.03 — the `syrinx-eval` harness skeleton.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build:
//!
//!   * `run(input, metrics, out_path) -> Result<(), EvalError>` — the harness
//!     entry function that writes the metrics JSON.
//!   * `MetricSet` — the pluggable metric set: `new()` + `register(key, metric)`,
//!     and `default_stub()` for the default five-metric set.
//!   * `StubInput::stub()` — the stub synth input (no audio, no model).
//!   * `REQUIRED_KEYS` — the five fixed keys the schema must carry.
//!   * `EvalError::MissingKey(String)` — the typed error naming an omitted key.
//!
//! A registered metric is a closure `Fn(&StubInput) -> Option<f64>`: `Some(x)`
//! records a number, `None` records an explicit JSON `null`.
//!
//! RED: `syrinx-eval` is an empty crate, so none of these symbols resolve and the
//! target fails to build — every criterion is unmet. GREEN implements the harness
//! so each assertion below holds. Paths use `CARGO_TARGET_TMPDIR` (fixed at
//! compile time) so the tests never depend on the process working directory.

use std::path::{Path, PathBuf};

use serde_json::Value;
use syrinx_eval::{run, EvalError, MetricSet, StubInput, REQUIRED_KEYS};

/// The five keys the metrics JSON must always carry (criterion C1). Held locally
/// so the tests pin the literal schema independently of the crate's constant.
const KEYS: [&str; 5] = ["sim_o", "wer", "mos_proxy", "ttfb_ms", "rtf"];

/// A fresh, collision-free output path under cargo's per-suite temp dir. Any
/// stale file from a previous run is removed so existence checks are meaningful.
fn out_path(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("t0003_{name}.json"));
    let _ = std::fs::remove_file(&path);
    path
}

/// Read the metrics file the harness wrote and return its top-level object map.
fn read_object(path: &Path) -> serde_json::Map<String, Value> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("harness did not write {}: {e}", path.display()));
    let value: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("metrics file is not JSON: {e}"));
    match value {
        Value::Object(map) => map,
        other => panic!("metrics JSON top level must be an object, got {other}"),
    }
}

/// A complete five-key set with one metric (`mos_proxy`) that yields no result,
/// for the explicit-null path (criterion C3).
fn set_with_null_mos() -> MetricSet {
    let mut set = MetricSet::new();
    set.register("sim_o", |_| Some(0.91));
    set.register("wer", |_| Some(0.07));
    set.register("mos_proxy", |_| None); // plugged-in metric yields no result
    set.register("ttfb_ms", |_| Some(123.0));
    set.register("rtf", |_| Some(0.42));
    set
}

/// A four-key set that omits `omit`, for the missing-key error path (C4).
fn set_missing(omit: &str) -> MetricSet {
    let mut set = MetricSet::new();
    for key in KEYS {
        if key != omit {
            set.register(key, |_| Some(1.0));
        }
    }
    set
}

// ----- C1: the default stub run writes a JSON object with exactly the five
//           keys sim_o, wer, mos_proxy, ttfb_ms, rtf. -----

#[test]
fn test_default_run_writes_all_five_keys() {
    let path = out_path("all_five_keys");
    run(&StubInput::stub(), &MetricSet::default_stub(), &path).expect("default run must succeed");
    let object = read_object(&path);
    for key in KEYS {
        assert!(
            object.contains_key(key),
            "metrics JSON must contain key `{key}`; got keys {:?}",
            object.keys().collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_default_run_writes_no_keys_beyond_the_five() {
    let path = out_path("exactly_five_keys");
    run(&StubInput::stub(), &MetricSet::default_stub(), &path).expect("default run must succeed");
    let object = read_object(&path);
    assert_eq!(
        object.len(),
        5,
        "metrics JSON must have exactly five keys; got {:?}",
        object.keys().collect::<Vec<_>>()
    );
    for key in object.keys() {
        assert!(
            KEYS.contains(&key.as_str()),
            "metrics JSON carries an unexpected key `{key}`; allowed keys are {KEYS:?}"
        );
    }
}

#[test]
fn test_required_keys_constant_is_exactly_the_five() {
    assert_eq!(
        REQUIRED_KEYS.len(),
        5,
        "REQUIRED_KEYS must list exactly the five metric keys"
    );
    for key in KEYS {
        assert!(
            REQUIRED_KEYS.contains(&key),
            "REQUIRED_KEYS must include `{key}`; got {REQUIRED_KEYS:?}"
        );
    }
}

// ----- C2: every value the stub run records is a finite f64. -----

#[test]
fn test_default_run_values_are_all_finite_numbers() {
    let path = out_path("finite_values");
    run(&StubInput::stub(), &MetricSet::default_stub(), &path).expect("default run must succeed");
    let object = read_object(&path);
    for key in KEYS {
        let value = object
            .get(key)
            .unwrap_or_else(|| panic!("default run must record key `{key}`"));
        let number = value
            .as_f64()
            .unwrap_or_else(|| panic!("default metric `{key}` must be a number, got {value}"));
        assert!(
            number.is_finite(),
            "default metric `{key}` must be finite (not NaN/inf); got {number}"
        );
    }
}

// ----- C3: a metric that yields no result is serialized as JSON null, with its
//           key still present in the object. -----

#[test]
fn test_metric_yielding_none_is_serialized_as_null() {
    let path = out_path("none_is_null");
    run(&StubInput::stub(), &set_with_null_mos(), &path).expect("run with a null metric must succeed");
    let object = read_object(&path);
    assert!(
        object.contains_key("mos_proxy"),
        "a metric with no result must keep its key, never omit it"
    );
    assert_eq!(
        object.get("mos_proxy"),
        Some(&Value::Null),
        "a metric that yields no result must serialize as JSON null"
    );
}

#[test]
fn test_metrics_with_results_are_not_null() {
    let path = out_path("some_not_null");
    run(&StubInput::stub(), &set_with_null_mos(), &path).expect("run with a null metric must succeed");
    let object = read_object(&path);
    // Every key but the null one must carry a finite number, so `null` is specific
    // to the metric that yielded no result — not blanket-applied.
    for key in KEYS {
        if key == "mos_proxy" {
            continue;
        }
        let value = object
            .get(key)
            .unwrap_or_else(|| panic!("key `{key}` must be present"));
        let number = value
            .as_f64()
            .unwrap_or_else(|| panic!("metric `{key}` had a result, so it must be a number, got {value}"));
        assert!(number.is_finite(), "metric `{key}` must be finite; got {number}");
    }
}

#[test]
fn test_null_metric_run_still_has_exactly_five_keys() {
    let path = out_path("null_still_five");
    run(&StubInput::stub(), &set_with_null_mos(), &path).expect("run with a null metric must succeed");
    let object = read_object(&path);
    assert_eq!(
        object.len(),
        5,
        "a run with a null metric still carries all five keys; got {:?}",
        object.keys().collect::<Vec<_>>()
    );
}

// ----- C4: a metric set omitting one of the five keys yields a typed error
//           naming the missing key, and no partial JSON is written. -----

#[test]
fn test_complete_set_does_not_error() {
    let path = out_path("complete_ok");
    let result = run(&StubInput::stub(), &MetricSet::default_stub(), &path);
    assert!(
        result.is_ok(),
        "a complete five-key set must not error; got {result:?}"
    );
}

#[test]
fn test_missing_key_returns_typed_error_naming_it() {
    let path = out_path("missing_wer");
    match run(&StubInput::stub(), &set_missing("wer"), &path) {
        Err(EvalError::MissingKey(key)) => assert_eq!(
            key, "wer",
            "the missing-key error must name the omitted key `wer`; named `{key}`"
        ),
        other => panic!("omitting `wer` must return EvalError::MissingKey; got {other:?}"),
    }
}

#[test]
fn test_missing_key_error_names_the_actual_omitted_key() {
    // A different omission must name *that* key — the error is not hardwired.
    let path = out_path("missing_rtf");
    match run(&StubInput::stub(), &set_missing("rtf"), &path) {
        Err(EvalError::MissingKey(key)) => assert_eq!(
            key, "rtf",
            "the missing-key error must name the omitted key `rtf`; named `{key}`"
        ),
        other => panic!("omitting `rtf` must return EvalError::MissingKey; got {other:?}"),
    }
}

#[test]
fn test_missing_key_writes_no_partial_file() {
    let path = out_path("missing_no_write");
    let result = run(&StubInput::stub(), &set_missing("sim_o"), &path);
    assert!(result.is_err(), "an incomplete set must error; got {result:?}");
    assert!(
        !path.exists(),
        "a missing-key error must not write a partial metrics file at {}",
        path.display()
    );
}
