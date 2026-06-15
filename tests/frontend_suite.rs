//! Frozen RED suite for T-01.11 — the aggregating golden-file harness for the
//! deterministic frontend.
//!
//! One harness drives three frontend transforms through a single dispatcher,
//! [`syrinx_frontend::render_stage`], over a directory tree of `(input, expected)`
//! golden pairs rooted at the repo-root `tests/golden/frontend_suite/`:
//!
//!   * `normalize/<case>.in` -> `normalize(input)`
//!   * `numbers/<case>.in`   -> `expand_numbers(input)`
//!   * `ssml/<case>.in`      -> `format!("{:?}", parse(input))`
//!
//! Each `<case>.in` holds raw input bytes and its sibling `<case>.expected` holds
//! the exact expected output bytes. Stage directories AND the cases inside them
//! are enumerated from disk — no case list is baked into this harness — so a
//! newly dropped `(input, expected)` pair is picked up with no edit here.
//!
//! Criteria pinned:
//!   * C1 — the suite runs golden cases covering normalize, number-expansion, and
//!     SSML, and every case reproduces its expected bytes (green run).
//!   * C2 — the goldens actually gate behaviour: a *different* (i.e. mutated)
//!     input for a case no longer reproduces that case's expected output.
//!   * C3 — enumeration is directory-driven (a newly added pair is auto-discovered)
//!     and an input with no matching `.expected` FAILS the run rather than being
//!     silently skipped.
//!
//! Paths resolve from `CARGO_MANIFEST_DIR` (fixed at compile time) so the suite is
//! independent of the process working directory; the synthetic-tree tests write
//! under `CARGO_TARGET_TMPDIR`, a per-target scratch dir Cargo provides.
//!
//! RED: `syrinx-frontend` exposes no `render_stage` yet, so the symbol does not
//! resolve and this target fails to build — every criterion is unmet. GREEN adds
//! `render_stage` so the dispatcher reproduces every golden's bytes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use syrinx_frontend::render_stage;

/// Repo-root directory holding the per-stage golden fixture sub-trees.
fn suite_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("frontend_suite")
}

/// A discovered golden case. `expected` is `None` when the sibling `.expected`
/// file is absent — discovery records the unpaired input rather than dropping it,
/// so a missing expected can be surfaced as a failure (criterion C3).
struct Case {
    stage: String,
    name: String,
    input: String,
    expected: Option<String>,
}

/// Immediate sub-directories of `root` (the per-stage fixture dirs), sorted.
fn stage_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)
        .unwrap_or_else(|e| panic!("cannot read suite root {}: {e}", root.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// Every `<case>.in` file directly under `dir`, sorted by name for determinism.
fn input_files(dir: &Path) -> Vec<PathBuf> {
    let mut ins: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read stage dir {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("in"))
        .collect();
    ins.sort();
    ins
}

/// Walk every stage dir then every `.in` inside it, pairing each with its
/// `.expected` sibling. The case list is built purely from what is on disk, so
/// adding a pair needs no edit to this function (criterion C3, auto-enumeration).
fn discover(root: &Path) -> Vec<Case> {
    let mut cases = Vec::new();
    for dir in stage_dirs(root) {
        let stage = dir
            .file_name()
            .and_then(|n| n.to_str())
            .expect("stage dir name")
            .to_string();
        for in_path in input_files(&dir) {
            let name = in_path
                .file_stem()
                .and_then(|n| n.to_str())
                .expect("case stem")
                .to_string();
            let input = fs::read_to_string(&in_path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", in_path.display()));
            let expected_path = in_path.with_extension("expected");
            let expected = if expected_path.exists() {
                Some(fs::read_to_string(&expected_path).unwrap_or_else(|e| {
                    panic!("cannot read {}: {e}", expected_path.display())
                }))
            } else {
                None
            };
            cases.push(Case { stage: stage.clone(), name, input, expected });
        }
    }
    cases
}

/// Run one case strictly: a missing `.expected` is a hard failure (panic), never
/// a skip, and the produced output must equal the expected bytes (criterion C3
/// missing-fails; criterion C1 green run). Returns the produced output.
fn run_case_strict(case: &Case) -> String {
    let expected = case.expected.as_ref().unwrap_or_else(|| {
        panic!(
            "golden input {}/{}.in has no matching .expected file",
            case.stage, case.name
        )
    });
    let got = render_stage(&case.stage, &case.input);
    assert_eq!(
        &got, expected,
        "stage {} case {} did not reproduce its expected bytes",
        case.stage, case.name
    );
    got
}

/// Group discovered cases by their stage name.
fn by_stage(cases: &[Case]) -> BTreeMap<String, Vec<&Case>> {
    let mut map: BTreeMap<String, Vec<&Case>> = BTreeMap::new();
    for c in cases {
        map.entry(c.stage.clone()).or_default().push(c);
    }
    map
}

// ----------------------------------------------------------------------------
// C1 — the aggregating green run over the real golden tree.
// ----------------------------------------------------------------------------

/// Every golden case under the suite reproduces its expected bytes through the
/// dispatcher — the suite as a whole exits clean (criterion C1).
#[test]
fn c1_all_golden_cases_match_expected() {
    let cases = discover(&suite_root());
    assert!(!cases.is_empty(), "no golden cases discovered under the suite root");
    for case in &cases {
        run_case_strict(case);
    }
}

/// The suite actually covers the three required transforms — normalize,
/// number-expansion, and SSML — each as its own stage dir with >=1 case
/// (criterion C1, the "covering normalize, number-expansion, and SSML" clause).
#[test]
fn c1_suite_covers_normalize_numbers_and_ssml() {
    let cases = discover(&suite_root());
    let grouped = by_stage(&cases);
    for required in ["normalize", "numbers", "ssml"] {
        let count = grouped.get(required).map(|v| v.len()).unwrap_or(0);
        assert!(
            count >= 1,
            "suite must cover stage `{required}` with at least one case, found {count}"
        );
    }
}

// ----------------------------------------------------------------------------
// C2 — the goldens gate behaviour: a changed input fails its paired case.
// ----------------------------------------------------------------------------

/// For each stage, two cases with *distinct* expected outputs witness that the
/// transform genuinely depends on the input: feeding one case's input where the
/// other case's expected is required does NOT reproduce that expected. So
/// mutating a golden input (to different content) breaks its case — proving the
/// goldens gate behaviour rather than passing vacuously (criterion C2).
#[test]
fn c2_changed_input_breaks_its_case() {
    let cases = discover(&suite_root());
    let grouped = by_stage(&cases);
    assert!(!grouped.is_empty(), "no stages discovered");

    for (stage, group) in &grouped {
        let mut witnessed = false;
        for a in group {
            let ea = a
                .expected
                .as_ref()
                .unwrap_or_else(|| panic!("{}/{} missing expected", a.stage, a.name));
            // Own input reproduces own expected (the case is genuinely green).
            assert_eq!(
                &render_stage(stage, &a.input),
                ea,
                "stage {stage} case {} must reproduce its own expected",
                a.name
            );
            for b in group {
                let eb = b.expected.as_ref().unwrap();
                if ea == eb {
                    continue;
                }
                // A different input (b's) must NOT reproduce a's expected — i.e.
                // replacing a's input file with other bytes fails a's case.
                assert_ne!(
                    &render_stage(stage, &b.input),
                    ea,
                    "stage {stage}: a changed input still reproduced case {}'s expected — \
                     the golden does not gate behaviour",
                    a.name
                );
                witnessed = true;
            }
        }
        assert!(
            witnessed,
            "stage {stage} needs >=2 cases with distinct expected to prove inputs gate output"
        );
    }
}

// ----------------------------------------------------------------------------
// C3 — directory-driven auto-enumeration, and missing-expected fails not skips.
// ----------------------------------------------------------------------------

/// Discovery is driven by directory contents: a newly written `(input, expected)`
/// pair is picked up with no change to this harness — adding one pair to a stage
/// dir grows the discovered count by exactly one (criterion C3, auto-enumeration).
#[test]
fn c3_new_pair_is_auto_discovered() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("frontend_suite_discover");
    let stage = tmp.join("normalize");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&stage).expect("create synthetic stage dir");

    fs::write(stage.join("a.in"), "x").unwrap();
    fs::write(stage.join("a.expected"), render_stage("normalize", "x")).unwrap();
    let before = discover(&tmp).len();
    assert_eq!(before, 1, "exactly the one seeded pair is discovered first");

    fs::write(stage.join("b.in"), "y").unwrap();
    fs::write(stage.join("b.expected"), render_stage("normalize", "y")).unwrap();
    let after = discover(&tmp).len();

    assert_eq!(
        after,
        before + 1,
        "a newly added (input, expected) pair must be auto-discovered without harness edits"
    );

    // The freshly discovered pairs still pass through the strict runner.
    for case in &discover(&tmp) {
        run_case_strict(case);
    }
}

/// An input with no matching `.expected` is still discovered (not skipped) and is
/// surfaced as a failure by the strict runner — never a silent pass (criterion C3,
/// the missing-expected clause).
#[test]
fn c3_missing_expected_fails_rather_than_skips() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("frontend_suite_missing");
    let stage = tmp.join("normalize");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&stage).expect("create synthetic stage dir");

    // Note: a lone `.in` with deliberately NO sibling `.expected`.
    fs::write(stage.join("lonely.in"), "hi").unwrap();

    let cases = discover(&tmp);
    assert_eq!(cases.len(), 1, "the unpaired input is still discovered, not dropped");
    assert!(
        cases[0].expected.is_none(),
        "an input with no matching expected must be flagged as unpaired, not silently skipped"
    );

    // The strict runner turns that unpaired input into a hard failure.
    let outcome = std::panic::catch_unwind(|| run_case_strict(&cases[0]));
    assert!(
        outcome.is_err(),
        "a missing expected file must FAIL the run, not pass silently"
    );
}
