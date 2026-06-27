//! syrinx-eval — measured CosyVoice2/3 evaluation metrics.
//!
//! The real metrics (SIM-o, RTF, TTFB, and the externally-helped WER / MOS-proxy)
//! are computed in [`real`] by running the real `Synthesizer`. The five-key,
//! present-or-null schema every metrics record upholds is named by [`REQUIRED_KEYS`]:
//! a metric that yields `None` serializes as JSON `null` (its key kept, never
//! omitted).

/// Real measured metrics (SIM-o / RTF / TTFB / WER / MOS-proxy). On by default.
#[cfg(feature = "real")]
pub mod real;

/// The five metric keys the metrics JSON always carries, in schema order.
pub const REQUIRED_KEYS: [&str; 5] = ["sim_o", "wer", "mos_proxy", "ttfb_ms", "rtf"];
