//! Variant-agnostic foundation shared by the s1 and s2 backends.
//!
//! `config` (the dual-AR + codec config schema) and `sampling` (the seeded sampler)
//! are pure Rust and always compile. `dualar` (the backend trait + driver loop),
//! `codec` (the RVQ-codec trait), and `audio` (44.1 kHz wav I/O) use Candle / `hound`
//! and so live behind the crate's `real` feature, mirroring the `syrinx-lm`
//! convention.

pub mod config;
pub mod sampling;

#[cfg(feature = "real")]
pub mod dualar;

#[cfg(feature = "real")]
pub mod codec;

#[cfg(feature = "real")]
pub mod audio;
