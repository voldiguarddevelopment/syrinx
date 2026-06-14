//! Frozen RED golden suite for T-01.01 criterion C3.
//!
//! Drives `syrinx_frontend::normalize::normalize` over the (input, expected)
//! pairs stored under the repo-root `tests/golden/normalize/`. Each case is a
//! pair of files sharing a stem: `<case>.in` holds the raw input bytes and
//! `<case>.expected` holds the exact expected output bytes. For every `.in`
//! file the test normalizes its UTF-8 contents and asserts the result equals the
//! matching `.expected` file BYTE FOR BYTE — so mutating any single expected
//! file's bytes makes that case fail (criterion C3).
//!
//! Paths resolve from `CARGO_MANIFEST_DIR` (fixed at compile time) so the suite
//! never depends on the process working directory.
//!
//! RED: `syrinx-frontend` exposes no `normalize` module yet, so the symbol does
//! not resolve and the test target fails to build. GREEN implements `normalize`
//! so every golden case reproduces its expected bytes.

use std::path::PathBuf;

use syrinx_frontend::normalize::normalize;

/// The repo-root directory holding the golden `(.in, .expected)` pairs.
fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("normalize")
}

/// Collect every `<case>.in` file under the golden dir, sorted by name for
/// determinism.
fn input_files() -> Vec<PathBuf> {
    let dir = golden_dir();
    let mut ins: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read golden dir {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("in"))
        .collect();
    ins.sort();
    ins
}

/// The golden corpus must be non-empty — an empty directory must NOT vacuously
/// pass the suite (criterion C3 requires real (input, expected) pairs).
#[test]
fn golden_corpus_is_non_empty() {
    let count = input_files().len();
    assert!(count >= 1, "expected at least one golden .in case, found {count}");
}

/// Every golden case: `normalize(<case>.in)` must equal `<case>.expected` byte
/// for byte. Mutating any expected file's bytes breaks the corresponding
/// assertion (criterion C3).
#[test]
fn golden_cases_match_expected_bytes() {
    let inputs = input_files();
    assert!(!inputs.is_empty(), "no golden cases discovered");

    for in_path in inputs {
        let expected_path = in_path.with_extension("expected");

        let input_bytes = std::fs::read(&in_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", in_path.display()));
        let input = String::from_utf8(input_bytes)
            .unwrap_or_else(|e| panic!("{} is not UTF-8: {e}", in_path.display()));

        let expected = std::fs::read(&expected_path).unwrap_or_else(|e| {
            panic!("missing expected file {}: {e}", expected_path.display())
        });

        let got = normalize(&input);
        assert_eq!(
            got.as_bytes(),
            expected.as_slice(),
            "golden case {} did not reproduce {} byte-for-byte",
            in_path.display(),
            expected_path.display(),
        );
    }
}
