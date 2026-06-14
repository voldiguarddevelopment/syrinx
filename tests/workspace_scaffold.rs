//! Frozen scaffold tests for T-00.01 — the eleven-crate Cargo workspace.
//!
//! These pin the three cargo-observable acceptance criteria by inspecting the
//! workspace-root manifest and the member-crate layout on disk. Paths are
//! resolved relative to `CARGO_MANIFEST_DIR` (the package root, fixed at compile
//! time) so the tests never depend on the process working directory.
//!
//! RED: the root manifest is a plain package and the `crates/` tree is absent,
//! so every assertion below fails. GREEN makes them pass by declaring the
//! workspace and creating the eleven empty member crates.

use std::path::PathBuf;

/// The eleven member crates the workspace must declare (criterion C2).
const MEMBER_CRATES: [&str; 11] = [
    "syrinx-frontend",
    "syrinx-core",
    "syrinx-lm",
    "syrinx-speaker",
    "syrinx-acoustic",
    "syrinx-vocoder",
    "syrinx-prosody",
    "syrinx-stream",
    "syrinx-serve",
    "syrinx-eval",
    "syrinx-cli",
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Drops `#` comments (a `#` to end-of-line) so assertions match real manifest
/// directives, never prose in a comment.
fn strip_comments(text: &str) -> String {
    text.lines()
        .map(|line| match line.find('#') {
            Some(i) => &line[..i],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn root_manifest() -> String {
    let path = workspace_root().join("Cargo.toml");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read root manifest {}: {e}", path.display()));
    strip_comments(&raw)
}

/// The whitespace-stripped manifest text, for robust substring matching.
fn root_manifest_compact() -> String {
    root_manifest().replace([' ', '\t'], "")
}

/// The entries of the root manifest's `members = [ ... ]` array.
fn workspace_members(manifest: &str) -> Vec<String> {
    let key = manifest
        .find("members")
        .expect("root manifest has no `members` key");
    let open = manifest[key..]
        .find('[')
        .map(|i| key + i)
        .expect("`members` declaration has no opening `[`");
    let close = manifest[open..]
        .find(']')
        .map(|i| open + i)
        .expect("`members` array has no closing `]`");
    manifest[open + 1..close]
        .split(',')
        .map(|entry| entry.trim().trim_matches('"').trim().to_string())
        .filter(|entry| !entry.is_empty())
        .collect()
}

fn crate_dir(name: &str) -> PathBuf {
    workspace_root().join("crates").join(name)
}

// ----- C1: root manifest declares a `[workspace]` with `resolver = "2"`, and
//           the members are buildable so `cargo build` exits 0. -----

#[test]
fn test_root_cargo_declares_workspace_table() {
    let manifest = root_manifest();
    assert!(
        manifest.contains("[workspace]"),
        "root Cargo.toml must contain a [workspace] table"
    );
}

#[test]
fn test_root_cargo_sets_resolver_two() {
    let compact = root_manifest_compact();
    assert!(
        compact.contains("resolver=\"2\""),
        "root Cargo.toml must set resolver = \"2\""
    );
}

#[test]
fn test_member_crates_have_buildable_target() {
    for name in MEMBER_CRATES {
        let dir = crate_dir(name);
        let has_lib = dir.join("src").join("lib.rs").is_file();
        let has_bin = dir.join("src").join("main.rs").is_file();
        assert!(
            has_lib || has_bin,
            "member crate {name} must have a buildable target (src/lib.rs or src/main.rs)"
        );
    }
}

// ----- C2: the workspace declares the eleven named member crates. -----

#[test]
fn test_workspace_lists_all_eleven_members() {
    let manifest = root_manifest();
    let members = workspace_members(&manifest);
    for name in MEMBER_CRATES {
        assert!(
            members.iter().any(|entry| entry.ends_with(name)),
            "workspace `members` must declare {name}; got {members:?}"
        );
    }
}

#[test]
fn test_all_eleven_member_crates_have_manifests() {
    for name in MEMBER_CRATES {
        let manifest_path = crate_dir(name).join("Cargo.toml");
        assert!(
            manifest_path.is_file(),
            "member crate {name} must have a Cargo.toml at {}",
            manifest_path.display()
        );
        let text = std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest_path.display()));
        let compact = text.replace([' ', '\t'], "");
        assert!(
            compact.contains(&format!("name=\"{name}\"")),
            "member crate manifest {} must declare name = \"{name}\"",
            manifest_path.display()
        );
    }
}

#[test]
fn test_workspace_has_exactly_eleven_members() {
    let manifest = root_manifest();
    let members = workspace_members(&manifest);
    assert_eq!(
        members.len(),
        11,
        "workspace must declare exactly eleven members; got {members:?}"
    );
}

// ----- C3: `cargo test` is green with zero tests defined in the member crates. -----

#[test]
fn test_member_crates_define_no_unit_tests() {
    for name in MEMBER_CRATES {
        let dir = crate_dir(name);
        assert!(
            dir.is_dir(),
            "member crate {name} directory must exist at {}",
            dir.display()
        );
        for src in ["lib.rs", "main.rs"] {
            let file = dir.join("src").join(src);
            if file.is_file() {
                let text = std::fs::read_to_string(&file)
                    .unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display()));
                assert!(
                    !text.contains("#[test]"),
                    "member crate {name} (src/{src}) must define zero tests"
                );
            }
        }
    }
}

#[test]
fn test_member_crates_have_no_integration_test_dirs() {
    for name in MEMBER_CRATES {
        let dir = crate_dir(name);
        assert!(
            dir.is_dir(),
            "member crate {name} directory must exist at {}",
            dir.display()
        );
        let tests_dir = dir.join("tests");
        assert!(
            !tests_dir.exists(),
            "member crate {name} must not define an integration tests/ dir (zero tests defined)"
        );
    }
}
