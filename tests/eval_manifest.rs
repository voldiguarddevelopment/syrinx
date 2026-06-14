//! Frozen RED tests for T-00.04 — the frozen-eval-set checksum mechanism.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build in `syrinx-eval`:
//!
//!   * `build_manifest(dir) -> io::Result<Manifest>` — walk an eval-set directory
//!     and map each file's relative path to its lowercase-hex SHA-256 digest.
//!   * `Manifest` — the checksum manifest: `entries()` exposes the
//!     `relative-path -> hex-digest` map; `write(path)` / `load(path)` persist it.
//!   * `verify(dir, &manifest) -> Result<(), VerifyError>` — recompute every
//!     digest and check membership against the manifest.
//!   * `VerifyError::DigestMismatch { path }` — a manifested file's bytes changed,
//!     naming the offending relative path.
//!   * `VerifyError::MembershipMismatch { path }` — a file is in the manifest but
//!     missing on disk, or present on disk but absent from the manifest.
//!
//! The invariant under test: `verify()` is `Ok(())` iff the set is byte-for-byte
//! and membership-identical to the manifest; any other state is a typed error
//! naming the path at fault.
//!
//! Digests are pinned to published SHA-256 test vectors so the tests assert the
//! exact algorithm and lowercase-hex encoding without recomputing it:
//!   SHA-256("abc")   = ba78…15ad
//!   SHA-256("hello") = 2cf2…9824
//!   SHA-256("")      = e3b0…b855  (the empty file)
//!
//! RED: none of these symbols resolve against the current `syrinx-eval` (which
//! only carries the T-00.03 harness), so this target fails to build — every
//! criterion is unmet. GREEN implements the manifest/verify API so each assertion
//! below holds. Eval-set directories live under `CARGO_TARGET_TMPDIR` (fixed at
//! compile time) so the tests never depend on the process working directory.

use std::path::{Path, PathBuf};

use syrinx_eval::{build_manifest, verify, Manifest, VerifyError};

/// Published SHA-256 vectors for the three eval-set files, lowercase hex.
const DIGEST_A: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"; // "abc"
const DIGEST_B: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"; // "hello"
const DIGEST_C: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"; // ""

/// The relative paths the manifest must key on. The nested `sub/c.txt` pins that
/// relative paths (not bare file names) are recorded for files in subdirectories.
const REL_A: &str = "a.txt";
const REL_B: &str = "b.txt";
const REL_C: &str = "sub/c.txt";

/// Build a fresh eval-set directory with the three known-content files, removing
/// any stale copy first so each test starts from a clean, isolated set.
fn make_set(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("t0004_set_{name}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("sub")).expect("create eval-set sub dir");
    std::fs::write(root.join("a.txt"), b"abc").expect("write a.txt");
    std::fs::write(root.join("b.txt"), b"hello").expect("write b.txt");
    std::fs::write(root.join("sub").join("c.txt"), b"").expect("write sub/c.txt");
    root
}

/// A fresh, collision-free manifest file path under cargo's per-suite temp dir.
fn manifest_path(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("t0004_{name}.manifest"));
    let _ = std::fs::remove_file(&path);
    path
}

/// Build the manifest for `dir` or panic with the I/O error.
fn build(dir: &Path) -> Manifest {
    build_manifest(dir).unwrap_or_else(|e| panic!("build_manifest failed for {}: {e}", dir.display()))
}

// ----- C1: the manifest maps each file's relative path to its lowercase-hex
//           SHA-256 digest, with exactly one entry per file in the set. -----

#[test]
fn test_manifest_has_exactly_one_entry_per_file() {
    let dir = make_set("one_per_file");
    let manifest = build(&dir);
    assert_eq!(
        manifest.entries().len(),
        3,
        "manifest must hold exactly one entry per file (3 files); got {:?}",
        manifest.entries().keys().collect::<Vec<_>>()
    );
}

#[test]
fn test_manifest_keys_are_the_relative_paths() {
    let dir = make_set("rel_paths");
    let manifest = build(&dir);
    for rel in [REL_A, REL_B, REL_C] {
        assert!(
            manifest.entries().contains_key(rel),
            "manifest must key on relative path `{rel}`; got {:?}",
            manifest.entries().keys().collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_manifest_digests_are_lowercase_hex_sha256() {
    let dir = make_set("digests");
    let manifest = build(&dir);
    let entries = manifest.entries();
    assert_eq!(
        entries.get(REL_A).map(String::as_str),
        Some(DIGEST_A),
        "`{REL_A}` (\"abc\") must map to its SHA-256 digest"
    );
    assert_eq!(
        entries.get(REL_B).map(String::as_str),
        Some(DIGEST_B),
        "`{REL_B}` (\"hello\") must map to its SHA-256 digest"
    );
    assert_eq!(
        entries.get(REL_C).map(String::as_str),
        Some(DIGEST_C),
        "`{REL_C}` (empty file) must map to the SHA-256 of the empty input"
    );
}

#[test]
fn test_written_manifest_round_trips_through_disk() {
    let dir = make_set("round_trip");
    let built = build(&dir);
    let path = manifest_path("round_trip");
    built
        .write(&path)
        .unwrap_or_else(|e| panic!("manifest write failed: {e}"));
    assert!(
        path.exists(),
        "manifest builder must write a manifest file at {}",
        path.display()
    );
    let loaded = Manifest::load(&path).unwrap_or_else(|e| panic!("manifest load failed: {e}"));
    assert_eq!(
        loaded.entries(),
        built.entries(),
        "a written manifest must round-trip its path->digest map exactly"
    );
}

// ----- C2: verify() is Ok(()) when every digest matches and membership is
//           identical to the manifest. -----

#[test]
fn test_verify_ok_on_unchanged_set() {
    let dir = make_set("verify_ok");
    let manifest = build(&dir);
    assert_eq!(
        verify(&dir, &manifest),
        Ok(()),
        "an unchanged, membership-identical set must verify Ok"
    );
}

#[test]
fn test_verify_ok_after_round_trip_through_disk() {
    // The manifest reloaded from disk must still verify the untouched set Ok, so
    // the persisted form carries the same digests the in-memory build did.
    let dir = make_set("verify_ok_reload");
    let path = manifest_path("verify_ok_reload");
    build(&dir)
        .write(&path)
        .unwrap_or_else(|e| panic!("manifest write failed: {e}"));
    let loaded = Manifest::load(&path).unwrap_or_else(|e| panic!("manifest load failed: {e}"));
    assert_eq!(
        verify(&dir, &loaded),
        Ok(()),
        "a reloaded manifest must verify the untouched set Ok"
    );
}

// ----- C3: editing any byte of a manifested file makes verify() return a
//           DigestMismatch naming that file; an unchanged set never does. -----

#[test]
fn test_tampered_byte_yields_digest_mismatch_naming_file() {
    let dir = make_set("tamper");
    let manifest = build(&dir);
    // Rewrite b.txt with one byte changed ("hello" -> "Hello"); membership is
    // unchanged, only the content drifts.
    std::fs::write(dir.join("b.txt"), b"Hello").expect("rewrite b.txt");
    match verify(&dir, &manifest) {
        Err(VerifyError::DigestMismatch { path }) => assert_eq!(
            path, REL_B,
            "the digest error must name the tampered file `{REL_B}`; named `{path}`"
        ),
        other => panic!("a tampered byte must yield DigestMismatch; got {other:?}"),
    }
}

#[test]
fn test_unchanged_set_is_not_a_digest_mismatch() {
    // The exact path C3 contrasts: a byte-identical set must NOT report a digest
    // error — verify is Ok, never DigestMismatch.
    let dir = make_set("no_false_tamper");
    let manifest = build(&dir);
    assert!(
        !matches!(verify(&dir, &manifest), Err(VerifyError::DigestMismatch { .. })),
        "an unchanged set must never report a digest mismatch"
    );
}

// ----- C4: a file in the manifest but missing on disk, or on disk but absent
//           from the manifest, yields a MembershipMismatch naming the path. -----

#[test]
fn test_missing_file_yields_membership_mismatch_naming_path() {
    let dir = make_set("missing");
    let manifest = build(&dir);
    // Remove a manifested (nested) file from disk; the directory drifts below the
    // manifest's membership.
    std::fs::remove_file(dir.join("sub").join("c.txt")).expect("remove sub/c.txt");
    match verify(&dir, &manifest) {
        Err(VerifyError::MembershipMismatch { path }) => assert_eq!(
            path, REL_C,
            "the membership error must name the missing file `{REL_C}`; named `{path}`"
        ),
        other => panic!("a missing manifested file must yield MembershipMismatch; got {other:?}"),
    }
}

#[test]
fn test_extra_file_yields_membership_mismatch_naming_path() {
    let dir = make_set("extra");
    let manifest = build(&dir);
    // Add a file not in the manifest; the directory drifts above the manifest's
    // membership.
    std::fs::write(dir.join("d.txt"), b"extra").expect("write d.txt");
    match verify(&dir, &manifest) {
        Err(VerifyError::MembershipMismatch { path }) => assert_eq!(
            path, "d.txt",
            "the membership error must name the extra file `d.txt`; named `{path}`"
        ),
        other => panic!("an unmanifested disk file must yield MembershipMismatch; got {other:?}"),
    }
}

#[test]
fn test_membership_drift_is_not_reported_as_ok() {
    // Neither drift direction may slip through as Ok.
    let missing_dir = make_set("drift_missing");
    let m1 = build(&missing_dir);
    std::fs::remove_file(missing_dir.join("a.txt")).expect("remove a.txt");
    assert!(
        verify(&missing_dir, &m1).is_err(),
        "a missing manifested file must not verify Ok"
    );

    let extra_dir = make_set("drift_extra");
    let m2 = build(&extra_dir);
    std::fs::write(extra_dir.join("e.txt"), b"e").expect("write e.txt");
    assert!(
        verify(&extra_dir, &m2).is_err(),
        "an extra unmanifested file must not verify Ok"
    );
}

#[test]
fn test_unchanged_set_is_not_a_membership_mismatch() {
    // The contrast case for C4: identical membership must NOT report a membership
    // error — verify is Ok.
    let dir = make_set("no_false_membership");
    let manifest = build(&dir);
    assert!(
        !matches!(
            verify(&dir, &manifest),
            Err(VerifyError::MembershipMismatch { .. })
        ),
        "a membership-identical set must never report a membership mismatch"
    );
}
