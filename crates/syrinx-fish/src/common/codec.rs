//! The RVQ-codec contract.
//!
//! [`RvqCodec`] is the variant-agnostic interface the modded-DAC (s1) and EVA-GAN /
//! causal-DAC (s2) decoders implement. The driver produces a `[num_codebooks, T]` code
//! matrix (see [`super::dualar::drive`]); the codec turns it into a 44.1 kHz waveform.
//! The reverse direction ([`RvqCodec::encode`]) turns reference audio into codes for
//! voice cloning (the reference `encode_audio` → `prompt_tokens` path).

use candle_core::{Result, Tensor};

use super::config::CodecConfig;

/// Codes ↔ waveform for one variant's RVQ codec. Implemented by the s1 / s2 backends.
pub trait RvqCodec {
    /// The codec geometry (codebook count/size, sample rate, hop).
    fn config(&self) -> &CodecConfig;

    /// Output sample rate (Hz) — `44100` for both Fish variants.
    fn sample_rate(&self) -> u32 {
        self.config().sample_rate
    }

    /// Decode a `[num_codebooks, T]` (or `[1, num_codebooks, T]`) code matrix to a mono
    /// waveform `[n_samples]` at [`Self::sample_rate`]. This is the reference
    /// `DAC.from_indices` → `decoder` path: `quantizer.decode(codes)` → causal conv decoder.
    fn decode(&self, codes: &Tensor) -> Result<Tensor>;

    /// Encode a mono waveform `[n_samples]` (or `[1, 1, n_samples]`) at
    /// [`Self::sample_rate`] to a `[num_codebooks, T]` code matrix — the reference
    /// `DAC.encode` (analysis conv encoder → residual VQ). Used to derive cloning codes
    /// from a reference voice.
    fn encode(&self, wav: &Tensor) -> Result<Tensor>;
}
