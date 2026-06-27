// ============================================================================
// Output watermarking (additive, OPT-IN — applied only via `synthesize_watermarked`,
// never on the default `synthesize`/CLI/server path; see the README Ethics section).
//
// Kept in its own module so it does not overlap the core pipeline. The
// watermark itself lives in the pure-Rust, model-free `crate::watermark` module
// (embed/detect on a 24 kHz mono `f32` buffer); this is only the synth-side glue
// that runs the existing `synthesize` and stamps its output. The plain
// `synthesize` is untouched, so the parity path is unaffected.
// ============================================================================

use super::*;

impl Synthesizer {
    /// Synthesize `tts_text` in the reference voice and **stamp the output with a
    /// spread-spectrum watermark** (see [`crate::watermark`]), returning the 24 kHz
    /// waveform.
    ///
    /// Identical to [`Synthesizer::synthesize`] followed by
    /// [`crate::watermark::embed_watermark`] at the default amplitude
    /// ([`crate::watermark::DEFAULT_AMP`], ≈ −48 dBFS). The `key` is the shared
    /// secret the detector needs; `payload` is a 16-bit tag (e.g. a model/version
    /// or batch id). The perturbation is `±DEFAULT_AMP` per sample — well below the
    /// perceptual threshold for speech — and [`crate::watermark::detect_watermark`]
    /// recovers `(present, confidence, payload)` from the unmodified output (and
    /// after light post-editing; see the module's honest robustness boundary).
    pub fn synthesize_watermarked(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        key: u64,
        payload: u16,
    ) -> Result<Vec<f32>, SynthError> {
        let mut wav = self.synthesize(tts_text, prompt_text, ref_wav_16k, ref_wav_24k, inputs)?;
        crate::watermark::embed_watermark(&mut wav, key, payload);
        Ok(wav)
    }
}
