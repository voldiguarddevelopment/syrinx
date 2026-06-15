//! The versioned frontend→LM hand-off contract (T-01.12).
//!
//! [`FrontendOutput`] is the typed payload the deterministic frontend produces
//! for `syrinx-lm`: typed token/phoneme entries plus control events, stamped with
//! an explicit [`SCHEMA_VERSION`]. The wire format is JSON only. A populated value
//! survives a `serde_json` round-trip unchanged. [`from_json`] is the *checked*
//! deserializer: a payload whose `schema_version` differs from [`SCHEMA_VERSION`]
//! is rejected with [`ContractError::VersionMismatch`], and a payload missing the
//! field (or otherwise unparseable) is rejected with [`ContractError::Malformed`]
//! — never a silent accept, never a panic, and never a defaulted version. There is
//! no cross-version migration.

use serde::{Deserialize, Serialize};

/// The current contract schema version. Every [`FrontendOutput`] carries this
/// exact integer, and [`from_json`] rejects any payload that does not.
pub const SCHEMA_VERSION: u32 = 1;

/// One typed token/phoneme entry: the source token text and its phoneme sequence.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenEntry {
    /// The source token text.
    pub text: String,
    /// The token's phoneme sequence.
    pub phonemes: Vec<String>,
}

/// A typed control event interleaved with the token stream.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum ControlEvent {
    /// A timed pause of `ms` milliseconds.
    Break {
        /// Pause duration in milliseconds.
        ms: u32,
    },
    /// A literal text segment.
    Text(String),
}

/// The typed, versioned hand-off struct from the frontend to `syrinx-lm`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct FrontendOutput {
    /// The schema version this payload was produced under.
    pub schema_version: u32,
    /// Typed token/phoneme entries.
    pub tokens: Vec<TokenEntry>,
    /// Typed control events.
    pub events: Vec<ControlEvent>,
}

impl FrontendOutput {
    /// Build a payload stamped with the current [`SCHEMA_VERSION`].
    pub fn new(tokens: Vec<TokenEntry>, events: Vec<ControlEvent>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            tokens,
            events,
        }
    }
}

/// A typed failure from [`from_json`].
#[derive(Debug, PartialEq)]
pub enum ContractError {
    /// The payload parsed, but its `schema_version` differs from [`SCHEMA_VERSION`].
    VersionMismatch,
    /// The payload is unparseable or is missing the required `schema_version` field.
    Malformed,
}

/// Deserialize a JSON payload, validating its schema version against
/// [`SCHEMA_VERSION`].
pub fn from_json(s: &str) -> Result<FrontendOutput, ContractError> {
    let out: FrontendOutput = serde_json::from_str(s).map_err(|_| ContractError::Malformed)?;
    if out.schema_version == SCHEMA_VERSION {
        Ok(out)
    } else {
        Err(ContractError::VersionMismatch)
    }
}
