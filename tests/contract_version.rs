//! Frozen RED tests for T-01.12 — the versioned frontend→LM contract.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build in `syrinx-frontend::contract`:
//!
//!   * `SCHEMA_VERSION: u32` — the current contract schema version constant.
//!   * `FrontendOutput` — the typed hand-off struct: an explicit
//!     `schema_version: u32` field plus typed token/phoneme entries (`tokens:
//!     Vec<TokenEntry>`) and control events (`events: Vec<ControlEvent>`). It
//!     derives `Debug + PartialEq + Serialize + Deserialize`, and
//!     `FrontendOutput::new(tokens, events)` stamps `schema_version` with the
//!     current `SCHEMA_VERSION`.
//!   * `TokenEntry { text: String, phonemes: Vec<String> }` — one typed
//!     token/phoneme entry.
//!   * `ControlEvent` — typed control events (`Break { ms }`, `Text(String)`).
//!   * `from_json(&str) -> Result<FrontendOutput, ContractError>` — the *checked*
//!     deserializer that validates the schema version.
//!   * `ContractError::VersionMismatch` — the payload's `schema_version` differs
//!     from `SCHEMA_VERSION`.
//!   * `ContractError::Malformed` — the payload is unparseable or is missing the
//!     required `schema_version` field (no defaulting).
//!
//! Contract (list.md / DESIGN §T1.12): the schema version is an explicit, checked
//! field on every payload. The wire format is JSON only. A populated value
//! survives `serde_json::to_string` → `from_str` unchanged (`PartialEq`). A
//! version mismatch or an absent version field yields a typed `ContractError` —
//! never a silent accept and never a panic. There is no cross-version migration.
//!
//! RED: `syrinx-frontend` exposes no `contract` module yet, so none of these
//! symbols resolve and the test target fails to build — every criterion is unmet.
//! GREEN implements the module so each assertion below holds.

use syrinx_frontend::contract::{
    from_json, ContractError, ControlEvent, FrontendOutput, TokenEntry, SCHEMA_VERSION,
};

/// A populated, schema-current `FrontendOutput` with typed token/phoneme entries
/// and control events — the shared fixture for the round-trip and version checks.
fn sample() -> FrontendOutput {
    FrontendOutput::new(
        vec![
            TokenEntry {
                text: "hello".to_string(),
                phonemes: vec![
                    "h".to_string(),
                    "ə".to_string(),
                    "l".to_string(),
                    "oʊ".to_string(),
                ],
            },
            TokenEntry {
                text: "world".to_string(),
                phonemes: vec!["w".to_string(), "ɜː".to_string(), "l".to_string(), "d".to_string()],
            },
        ],
        vec![
            ControlEvent::Break { ms: 250 },
            ControlEvent::Text("hello world".to_string()),
        ],
    )
}

/// An integer that is guaranteed to differ from `SCHEMA_VERSION` — an "older"
/// value where possible, so the mismatch payload is provably not schema-current.
fn other_version() -> u32 {
    if SCHEMA_VERSION == 0 {
        1
    } else {
        SCHEMA_VERSION - 1
    }
}

/// C1 — `FrontendOutput` carries an explicit `schema_version: u32` set to the
/// current `SCHEMA_VERSION`, and a constructed value exposes that exact integer.
#[test]
fn test_schema_version_field_is_current_constant() {
    let out = sample();
    // The field exists, is a `u32`, and equals the version constant exactly.
    let version: u32 = out.schema_version;
    assert_eq!(version, SCHEMA_VERSION);
}

/// C2 — a populated value round-trips through JSON to an equal struct: `PartialEq`
/// holds before serialization and after deserialization.
#[test]
fn test_json_roundtrip_preserves_equality() {
    let out = sample();
    // Equality holds before the wire trip (a fresh build of the same data).
    assert_eq!(out, sample());

    let json = serde_json::to_string(&out).expect("FrontendOutput must serialize to JSON");
    let back: FrontendOutput =
        serde_json::from_str(&json).expect("a full payload must deserialize");

    // Equality holds after the wire trip — the value is unchanged.
    assert_eq!(out, back);
    // And the schema version survived intact.
    assert_eq!(back.schema_version, SCHEMA_VERSION);
}

/// C3 — `from_json` accepts a schema-current payload but rejects a payload whose
/// `schema_version` differs, returning `ContractError::VersionMismatch` (not a
/// silent accept, and — being a `Result` — not a panic). The accept side pins the
/// matching boundary; the reject side pins the mismatch boundary.
#[test]
fn test_version_mismatch_is_typed_error() {
    // Matching version → accepted, and equal to the original.
    let good = sample();
    let good_json = serde_json::to_string(&good).expect("serialize");
    let parsed = from_json(&good_json).expect("a schema-current payload must be accepted");
    assert_eq!(parsed, good);

    // Differing version → rejected with the typed VersionMismatch error.
    let bad = FrontendOutput {
        schema_version: other_version(),
        tokens: sample().tokens,
        events: sample().events,
    };
    let bad_json = serde_json::to_string(&bad).expect("serialize");
    let err = from_json(&bad_json).expect_err("a version mismatch must be rejected");
    assert_eq!(err, ContractError::VersionMismatch);
}

/// C4 — a payload missing the `schema_version` field fails deserialization with a
/// typed error rather than defaulting the version. The error must be the typed
/// `ContractError::Malformed`, and explicitly NOT a `VersionMismatch` produced by
/// silently defaulting the field to some integer.
#[test]
fn test_missing_version_field_is_typed_error() {
    // Valid JSON for the rest of the struct, but with no `schema_version` key.
    let no_version = r#"{"tokens":[],"events":[]}"#;
    let err = from_json(no_version).expect_err("a payload without schema_version must fail");
    assert_eq!(err, ContractError::Malformed);
}
