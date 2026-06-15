//! Frozen RED tests for T-03.01 — the editable prosody-plan data model.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build in `syrinx-prosody::plan`:
//!
//!   * `PLAN_SCHEMA_VERSION: u32` — the current prosody-plan schema version.
//!   * `ProsodyPlan` — the typed editable plan: an explicit `schema_version: u32`
//!     field plus equal-length `durations_ms: Vec<f32>` and `pitch_hz: Vec<f32>`
//!     arrays. It derives `Debug + Serialize + Deserialize` (and is comparable),
//!     and `schema_version` is a *required* serde field (no default).
//!   * `ProsodyPlan::new(n, durations_ms, pitch_hz) -> Result<ProsodyPlan, PlanError>`
//!     — the checked constructor. It validates that both arrays have length `n`
//!     and stamps `schema_version` with the current `PLAN_SCHEMA_VERSION`. A
//!     length disagreement returns `Err(PlanError::LengthMismatch)` — never a panic.
//!   * `ProsodyPlan::phoneme(i) -> Result<Phoneme, PlanError>` — bounds-checked
//!     index access returning the `(duration_ms, pitch_hz)` pair at `i`. `i == N`
//!     (one past the last) and any larger `usize` yield
//!     `Err(PlanError::IndexOutOfRange)`; `i == N-1` is `Ok`. Never panics.
//!   * `Phoneme { duration_ms: f32, pitch_hz: f32 }` — one phoneme's plan entry.
//!   * `PlanError::{LengthMismatch, IndexOutOfRange}` — the typed errors.
//!
//! Contract (list.md / DESIGN §T3.01): the plan is caller-supplied data — no
//! prediction, no defaults. The invariant `durations_ms.len() == pitch_hz.len()
//! == N` always holds, the wire format is JSON, a constructed plan round-trips
//! byte-identically, every index access is total (a `Result`, never a panic), and
//! the schema version is an explicit field that must be present on the wire.
//!
//! RED: `syrinx-prosody` exposes no `plan` module yet, so none of these symbols
//! resolve and the test target fails to build — every criterion is unmet. GREEN
//! implements the module so each assertion below holds.

use syrinx_prosody::plan::{Phoneme, PlanError, ProsodyPlan, PLAN_SCHEMA_VERSION};

/// A populated, schema-current plan with three phonemes whose `durations_ms` and
/// `pitch_hz` values are all distinct — so an index test that read the wrong slot
/// or the wrong array would observe a wrong number. Shared by several tests.
fn sample() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![100.0, 200.0, 300.0])
        .expect("equal-length arrays of length n must construct")
}

// ----------------------------------------------------------------------------
// C1 — byte-identical JSON round-trip for a plan with at least one phoneme.
// ----------------------------------------------------------------------------

/// `to_vec` → `from_slice` → `to_vec` reproduces the original bytes exactly, and
/// the deserialized value equals the original.
#[test]
fn test_json_roundtrip_byte_identical() {
    let plan = sample();

    let bytes = serde_json::to_vec(&plan).expect("ProsodyPlan must serialize to JSON");
    let back: ProsodyPlan =
        serde_json::from_slice(&bytes).expect("a full payload must deserialize");
    let rebytes = serde_json::to_vec(&back).expect("the round-tripped plan must re-serialize");

    // The re-serialized bytes are identical to the original bytes (C1).
    assert_eq!(bytes, rebytes);
    // And the arrays survived the trip intact (a non-empty, ≥1-phoneme plan).
    assert_eq!(back.durations_ms, plan.durations_ms);
    assert_eq!(back.pitch_hz, plan.pitch_hz);
}

// ----------------------------------------------------------------------------
// C2 — length agreement at N == 0 and N == 3; mismatch is a typed error.
// ----------------------------------------------------------------------------

/// A plan constructed for N phonemes has both arrays of length N, for the empty
/// boundary (N == 0) and a populated case (N == 3).
#[test]
fn test_lengths_agree_for_empty_and_three() {
    // N == 0 — the empty plan is valid.
    let empty = ProsodyPlan::new(0, vec![], vec![]).expect("N == 0 is a valid empty plan");
    assert_eq!(empty.durations_ms.len(), 0);
    assert_eq!(empty.pitch_hz.len(), 0);

    // N == 3 — both arrays agree with N.
    let three = ProsodyPlan::new(3, vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0])
        .expect("N == 3 with equal-length arrays is valid");
    assert_eq!(three.durations_ms.len(), 3);
    assert_eq!(three.pitch_hz.len(), 3);
}

/// Mismatched array lengths return `Err(PlanError::LengthMismatch)` — never a
/// panic. Each branch of the length check is exercised on both sides: the
/// all-equal case constructs `Ok`; a too-short `durations_ms`, a too-short
/// `pitch_hz`, and arrays that agree with each other but not with `n` each error.
#[test]
fn test_mismatched_lengths_error() {
    // Baseline: all three lengths agree → Ok (pins the false side of the check).
    assert!(ProsodyPlan::new(3, vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).is_ok());

    // durations_ms shorter than n, pitch_hz equal to n → LengthMismatch.
    let e1 = ProsodyPlan::new(3, vec![1.0, 2.0], vec![4.0, 5.0, 6.0])
        .expect_err("durations_ms shorter than n must be rejected");
    assert!(matches!(e1, PlanError::LengthMismatch));

    // pitch_hz shorter than n, durations_ms equal to n → LengthMismatch.
    let e2 = ProsodyPlan::new(3, vec![1.0, 2.0, 3.0], vec![4.0, 5.0])
        .expect_err("pitch_hz shorter than n must be rejected");
    assert!(matches!(e2, PlanError::LengthMismatch));

    // Both arrays agree with each other (length 2) but not with n == 3 → still a
    // mismatch: the arrays must be length N, so `n` is genuinely checked.
    let e3 = ProsodyPlan::new(3, vec![1.0, 2.0], vec![4.0, 5.0])
        .expect_err("arrays not of length n must be rejected");
    assert!(matches!(e3, PlanError::LengthMismatch));
}

// ----------------------------------------------------------------------------
// C3 — index boundary: Ok at N-1, IndexOutOfRange at N, never panics.
// ----------------------------------------------------------------------------

/// `phoneme(N-1)` is `Ok` and yields that phoneme's exact values; `phoneme(N)`
/// (one past the last) is `Err(PlanError::IndexOutOfRange)`. The first non-empty
/// index is also `Ok` so the lower side of the bound is pinned too.
#[test]
fn test_phoneme_index_boundary() {
    let plan = sample(); // N == 3

    // i == 0 — in range, reads the first slot of each array.
    let first: Phoneme = plan.phoneme(0).expect("i == 0 is in range");
    assert_eq!(first.duration_ms, 10.0);
    assert_eq!(first.pitch_hz, 100.0);

    // i == N-1 == 2 — the last valid index, reads the last slot of each array.
    let last: Phoneme = plan.phoneme(2).expect("i == N-1 must be Ok");
    assert_eq!(last.duration_ms, 30.0);
    assert_eq!(last.pitch_hz, 300.0);

    // i == N == 3 — one past the last → IndexOutOfRange (not Ok, not a panic).
    let err = plan.phoneme(3).expect_err("i == N must be IndexOutOfRange");
    assert!(matches!(err, PlanError::IndexOutOfRange));
}

/// Index access is total: an empty plan errors at i == 0, and the maximum `usize`
/// errors rather than panicking. No `usize` index ever panics.
#[test]
fn test_phoneme_never_panics_on_any_index() {
    // Empty plan: i == 0 == N is out of range.
    let empty = ProsodyPlan::new(0, vec![], vec![]).expect("empty plan constructs");
    assert!(matches!(empty.phoneme(0), Err(PlanError::IndexOutOfRange)));

    // A wildly out-of-range index returns an error instead of panicking.
    let plan = sample(); // N == 3
    assert!(matches!(plan.phoneme(usize::MAX), Err(PlanError::IndexOutOfRange)));
}

// ----------------------------------------------------------------------------
// C4 — schema_version equals the constant; missing field fails to deserialize.
// ----------------------------------------------------------------------------

/// A constructed plan stamps the current `PLAN_SCHEMA_VERSION`, and that value
/// survives a JSON round-trip on the deserialized plan.
#[test]
fn test_schema_version_matches_constant() {
    let plan = sample();
    let version: u32 = plan.schema_version;
    assert_eq!(version, PLAN_SCHEMA_VERSION);

    let bytes = serde_json::to_vec(&plan).expect("serialize");
    let back: ProsodyPlan = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(back.schema_version, PLAN_SCHEMA_VERSION);
}

/// JSON that is otherwise well-formed but omits the `schema_version` field fails
/// to deserialize — the field is required, never silently defaulted.
#[test]
fn test_missing_schema_version_fails() {
    // Sanity: WITH the field present (and the rest of the struct), it parses — so
    // the only thing missing in the failing case is `schema_version` itself.
    let with_version = format!(
        r#"{{"schema_version":{},"durations_ms":[10.0],"pitch_hz":[100.0]}}"#,
        PLAN_SCHEMA_VERSION
    );
    assert!(serde_json::from_slice::<ProsodyPlan>(with_version.as_bytes()).is_ok());

    // Same payload minus `schema_version` → deserialization error.
    let no_version = br#"{"durations_ms":[10.0],"pitch_hz":[100.0]}"#;
    assert!(serde_json::from_slice::<ProsodyPlan>(no_version).is_err());
}
