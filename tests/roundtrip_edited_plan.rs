//! Frozen RED tests for T-03.10 — round-trip an EDITED plan, end to end.
//!
//! This task does no new editing (that is T-03.09) and adds no non-JSON wire
//! format. It pins the *end-to-end serialization fidelity* of a plan that already
//! carries a T-03.09 edit: serialize → deserialize → serialize is byte-stable, the
//! decoded plan equals the in-memory edited plan (and differs from the pre-edit
//! original) under `PartialEq`, repeat decoding is deterministic, and a payload
//! missing `schema_version` is a typed deserialize error rather than a silent
//! default.
//!
//! Every criterion's property already holds for `ProsodyPlan`'s derived
//! `Serialize`/`Deserialize`/`PartialEq`; what is missing is the crate's canonical
//! JSON wire codec — the first-class, tested entry point this task introduces so
//! "the wire format is JSON" (DESIGN) is a named operation rather than an ad-hoc
//! `serde_json` call at every call site. The green phase must add to
//! `syrinx-prosody`, on top of the T-03.01 model and T-03.09 edit:
//!
//!   * `ProsodyPlan::to_json(&self) -> Vec<u8>` — the canonical JSON wire bytes,
//!     i.e. exactly `serde_json::to_vec(self)` for the always-serializable plan.
//!   * `ProsodyPlan::from_json(bytes: &[u8]) -> Result<ProsodyPlan, _>` — decode
//!     from JSON wire bytes, i.e. exactly `serde_json::from_slice(bytes)`. A
//!     missing `schema_version` field is an `Err`, never a silent default.
//!
//! Placement note for GREEN: put this codec in its OWN module
//! (`crates/syrinx-prosody/src/roundtrip.rs`, `impl ProsodyPlan`) so the
//! task-scoped mutation gate bites only the new codec and not the prior-task
//! functions in `plan.rs`.
//!
//! RED: `ProsodyPlan` exposes neither `to_json` nor `from_json` yet, so these
//! symbols do not resolve and the test target fails to build — every criterion is
//! unmet. GREEN adds the two methods so each assertion below holds.

use syrinx_prosody::plan::{PhonemeEdit, ProsodyPlan};

/// A populated, schema-current plan with three phonemes whose `durations_ms` and
/// `pitch_hz` values are all distinct — the pre-edit original. Shared by the tests.
fn sample() -> ProsodyPlan {
    ProsodyPlan::new(3, vec![10.0, 20.0, 30.0], vec![100.0, 200.0, 300.0])
        .expect("equal-length arrays of length n must construct")
}

/// The pre-edit plan with a full T-03.09 edit applied at index 1 (duration 20→99,
/// pitch 200→440) — the canonical "edited plan" exercised by the round-trip tests.
fn edited_sample() -> ProsodyPlan {
    sample()
        .edit_phoneme(
            1,
            PhonemeEdit {
                duration_ms: Some(99.0),
                pitch_hz: Some(440.0),
            },
        )
        .expect("i == 1 is in range")
}

// ----------------------------------------------------------------------------
// C1 — an edited plan round-trips byte-stably: to_json → from_json → to_json
//      yields bytes byte-identical to the first serialization.
// ----------------------------------------------------------------------------

/// Serializing the edited plan, deserializing, then re-serializing reproduces the
/// first serialization's bytes exactly. The first serialization is also pinned to
/// the exact canonical wire bytes, so a codec that drops or empties the payload (a
/// body-replacement mutant returning `Vec::new()`) is caught rather than passing a
/// vacuous "empty == empty".
#[test]
fn test_edited_plan_roundtrips_byte_stable() {
    let edited = edited_sample();

    let bytes = edited.to_json();

    // The canonical wire bytes for this edited plan are exactly these — the edit at
    // index 1 is visible (99.0 in durations, 440.0 in pitch) and `schema_version`
    // leads the object. Pinning the literal bytes kills an empty/defaulted encoder.
    assert_eq!(
        bytes,
        br#"{"schema_version":1,"durations_ms":[10.0,99.0,30.0],"pitch_hz":[100.0,440.0,300.0]}"#
            .to_vec()
    );

    // to_json → from_json → to_json is byte-identical to the first serialization.
    let back = ProsodyPlan::from_json(&bytes).expect("a full payload must deserialize");
    let rebytes = back.to_json();
    assert_eq!(bytes, rebytes);
}

// ----------------------------------------------------------------------------
// C2 — the decoded plan equals the in-memory edited plan under PartialEq, and is
//      NOT equal to the pre-edit original (a round-tripped edit is distinguished
//      from the unedited original).
// ----------------------------------------------------------------------------

/// `from_json(edited.to_json()) == edited` and `!= the pre-edit plan` — the decode
/// recovers exactly the edited value and is distinguishable from the original.
#[test]
fn test_deser_equals_edited_and_differs_from_pre_edit() {
    let base = sample();
    let edited = edited_sample();

    let deser = ProsodyPlan::from_json(&edited.to_json()).expect("deserialize the edited plan");

    // Equal to the in-memory edited plan...
    assert_eq!(deser, edited);
    // ...and NOT equal to the pre-edit original (the edit survives the round-trip).
    assert_ne!(deser, base);
}

/// The symmetric pin: round-tripping the PRE-EDIT plan recovers the original and is
/// NOT equal to the edited plan — so the distinction is real in both directions,
/// not an artifact of one fixture. Uses an index-0 edit to vary the edited side.
#[test]
fn test_pre_edit_roundtrip_distinct_from_edited() {
    let base = sample();
    let edited = base
        .edit_phoneme(
            0,
            PhonemeEdit {
                duration_ms: Some(7.0),
                pitch_hz: Some(77.0),
            },
        )
        .expect("i == 0 is in range");

    let base_back = ProsodyPlan::from_json(&base.to_json()).expect("deserialize the pre-edit plan");

    // Round-tripping the original recovers the original...
    assert_eq!(base_back, base);
    // ...and the original is distinguishable from the edited plan.
    assert_ne!(base_back, edited);
}

// ----------------------------------------------------------------------------
// C3 — repeat decoding is deterministic, and JSON missing `schema_version` fails
//      to deserialize with an error rather than silently defaulting.
// ----------------------------------------------------------------------------

/// Deserializing the edited plan's JSON twice yields two values equal under
/// `PartialEq` (the decode is deterministic), and both equal the edited plan.
#[test]
fn test_roundtrip_deterministic_and_missing_version_errors() {
    let edited = edited_sample();
    let bytes = edited.to_json();

    let a = ProsodyPlan::from_json(&bytes).expect("first decode");
    let b = ProsodyPlan::from_json(&bytes).expect("second decode");

    // Two independent decodes of the same bytes agree...
    assert_eq!(a, b);
    // ...and the deterministic value is the right value (the edited plan).
    assert_eq!(a, edited);

    // JSON that omits `schema_version` is a deserialize ERROR — never a silent
    // default. (`from_json` surfaces the typed `serde` error via `Result`.)
    let no_version = br#"{"durations_ms":[10.0],"pitch_hz":[100.0]}"#;
    assert!(ProsodyPlan::from_json(no_version).is_err());

    // Sanity / boundary: the SAME shape WITH `schema_version` present decodes `Ok`,
    // so the only difference in the failing case is the absent required field —
    // pinning both sides of the missing-field boundary.
    let with_version = br#"{"schema_version":1,"durations_ms":[10.0],"pitch_hz":[100.0]}"#;
    assert!(ProsodyPlan::from_json(with_version).is_ok());
}
