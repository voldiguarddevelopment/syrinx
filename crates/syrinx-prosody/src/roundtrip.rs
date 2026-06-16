//! The canonical JSON wire codec for [`ProsodyPlan`] (T-03.10).
//!
//! The plan's wire format is JSON (DESIGN). This module makes that a named,
//! first-class operation rather than an ad-hoc `serde_json` call at each call
//! site: [`ProsodyPlan::to_json`] produces the canonical wire bytes and
//! [`ProsodyPlan::from_json`] decodes them back. The plan is always
//! serializable, so encoding is infallible; decoding surfaces serde's typed
//! error via [`Result`] — in particular a payload missing the required
//! `schema_version` field is an `Err`, never a silent default.

use crate::plan::ProsodyPlan;

impl ProsodyPlan {
    /// Encode this plan to its canonical JSON wire bytes.
    ///
    /// Exactly `serde_json::to_vec(self)` for the always-serializable plan.
    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a ProsodyPlan is always serializable")
    }

    /// Decode a plan from JSON wire bytes.
    ///
    /// Exactly `serde_json::from_slice(bytes)`. A missing `schema_version`
    /// field yields an `Err`, never a silent default.
    pub fn from_json(bytes: &[u8]) -> Result<ProsodyPlan, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}
