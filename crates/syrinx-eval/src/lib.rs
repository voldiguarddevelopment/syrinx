//! syrinx-eval — measured CosyVoice2/3 evaluation metrics.
//!
//! The real metrics (SIM-o, RTF, TTFB, and the externally-helped WER / MOS-proxy)
//! are computed in [`real`] by running the real `Synthesizer`. The five-key,
//! present-or-null schema every metrics record upholds is named by [`REQUIRED_KEYS`]:
//! a metric that yields `None` serializes as JSON `null` (its key kept, never
//! omitted).

/// Real measured metrics (SIM-o / RTF / TTFB / WER / MOS-proxy). On by default.
///
/// Gated on `real` because it drives the actual `Synthesizer` out of `syrinx-serve`,
/// which is only a dependency under that feature. Keep this attribute adjacent to
/// `metrics`: an item inserted between the two silently steals the gate, which is how
/// `--no-default-features` came to be broken once already.
#[cfg(feature = "real")]
pub mod metrics;

/// Cue-activation aggregation and the C4.2 gate. Deliberately NOT behind `real`: it is
/// pure aggregation over caller-supplied measurements and must stay available to the
/// model-free board.
pub mod activation;

/// C4.2' — the arm-contrast decision from ADR-0003: cue vs plain, sham vs plain, cue vs
/// sham, A vs A. Pure like `activation` and for the same reason — the numbers need a GPU,
/// the decision logic must stay on the model-free board.
pub mod contrast;

/// Qwen3-TTS evaluation: drives the serve Qwen engine and scores it with the in-tree
/// Whisper (`syrinx-stt`) rather than shelling out to Python, as C4.1 requires. Behind
/// `real` because it drives real weights.
#[cfg(feature = "real")]
pub mod qwen;

/// The affect judge: what emotion a speech-emotion model hears in a render. Behind the
/// off-by-default `affect` feature because it needs an ONNX Runtime and a downloaded
/// checkpoint; a default build must never pull either.
#[cfg(feature = "affect")]
pub mod affect;

/// Acoustic features and the permutation test that decides whether a cue activated.
/// Pure DSP and statistics, no model dependency, so it is testable without a GPU.
pub mod acoustic;

/// The five metric keys the metrics JSON always carries, in schema order.
pub const REQUIRED_KEYS: [&str; 5] = ["sim_o", "wer", "mos_proxy", "ttfb_ms", "rtf"];
