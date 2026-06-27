//! Incremental (streaming) synthesis: emit audio chunk-by-chunk with low first-byte
//! latency, plus the phase-continuous per-chunk HiFT source builder it drives.

use candle_core::{DType, Tensor};

use syrinx_acoustic::cv2::{token2wav_streaming, AudioChunk};

use super::*;

impl Synthesizer {
    /// **Streaming** synthesis: same `tts_text`-in-reference-voice flow + vocoder as
    /// [`Synthesizer::synthesize`], but audio is emitted **incrementally** chunk by
    /// chunk (low first-byte latency) via `on_chunk`, instead of one final `Vec`.
    ///
    /// This drives [`token2wav_streaming`], which replicates CosyVoice2's
    /// `token2wav` streaming path: a HiFT mel/source/speech overlap cache across
    /// chunks plus a hamming cross-fade at every chunk boundary. `token_hop` is the
    /// number of finalized speech tokens per chunk (a chunk needs only
    /// `token_hop + pre_lookahead` tokens present, so the first chunk lands long
    /// before the utterance finishes).
    ///
    /// Each emitted chunk is delivered to `on_chunk` as a flat `Vec<f32>` of 24 kHz
    /// samples in order; concatenating them yields the full streamed waveform.
    ///
    /// The per-chunk HiFT source is built by [`Synthesizer::streaming_source_phase`],
    /// which carries the deterministic F0 excitation phase **continuously** across
    /// chunks (one global sinusoid, matching the non-streaming source) and overwrites
    /// each chunk's overlap region with the previous chunk's trailing source samples
    /// (CosyVoice2 `cache_source`). This is what makes the streamed waveform
    /// sample-faithful to the non-streaming path rather than only length/energy-faithful.
    /// A pinned `inputs.s_stft` is **not** used here (the streaming source is rebuilt
    /// per chunk); `inputs.z`, the LM seed/cap, and `inputs.pinned_speech_token` are
    /// honoured exactly as in `synthesize`.
    pub fn synthesize_streaming(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        token_hop: usize,
        mut on_chunk: impl FnMut(Vec<f32>) -> Result<(), SynthError>,
    ) -> Result<(), SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;

        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // Per-chunk **phase-continuous** source builder: continues the global F0
        // excitation phase across chunks and overwrites each chunk's overlap with the
        // previous chunk's trailing source (CosyVoice2 `cache_source`), so the streamed
        // waveform is sample-faithful to the non-streaming source. `&self` capture
        // keeps the vocoder + device in scope.
        let source_fn = |mel_in: &Tensor,
                         phase_in: f64,
                         overlap: usize,
                         prev_tail: Option<&Tensor>|
         -> candle_core::Result<(Tensor, Tensor, f64)> {
            self.streaming_source_phase(mel_in, phase_in, overlap, prev_tail)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        };

        let mut cb_err: Option<SynthError> = None;
        let res = token2wav_streaming(
            &self.flow,
            &self.vocoder,
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            &source_fn,
            token_hop,
            N_TIMESTEPS,
            |chunk: AudioChunk| {
                let wav: Vec<f32> = chunk
                    .wav
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())?;
                if let Err(e) = on_chunk(wav) {
                    cb_err = Some(e);
                    return Err(candle_core::Error::Msg("streaming callback aborted".into()));
                }
                Ok(())
            },
        );
        if let Some(e) = cb_err {
            return Err(e);
        }
        res.map_err(SynthError::from)
    }

    /// Build the **streaming** HiFT source for one mel chunk with *global F0-phase
    /// continuity* (the faithfulness fix). Same deterministic F0 -> single-harmonic
    /// sine -> STFT core as [`Synthesizer::deterministic_source_stft`], but:
    ///
    ///   * the instantaneous phase is **not** reset to 0 for the chunk — it continues
    ///     from `phase_in` (the global phase, in cycles, at the start of this chunk's
    ///     *new* region), so concatenated chunks form one continuous sinusoid exactly
    ///     like the non-streaming source. (Per-chunk phase reset is what drove the
    ///     streaming-vs-non-streaming sample correlation to ~0.)
    ///   * the leading `overlap_samples` (the carried mel-overlap region) are
    ///     overwritten by `prev_tail`, the previous chunk's trailing source samples,
    ///     so the overlap is sample-identical to what the vocoder already emitted
    ///     (CosyVoice2 `cache_source`), keeping the boundary cross-fade coherent.
    ///
    /// Returns `(s_stft [1,18,T], source_wave [1, T_src], phase_out)` where `phase_out`
    /// is the global phase at this chunk's end (the next chunk's `phase_in`).
    pub fn streaming_source_phase(
        &self,
        mel_in: &Tensor,
        phase_in: f64,
        overlap_samples: usize,
        prev_tail: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, f64), SynthError> {
        let f0 = self.vocoder.f0_predict(mel_in)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        let n = f0v.len() * F0_UPSAMPLE; // source samples = mel frames * 480

        // New-region excitation: continue the global cumulative phase from phase_in.
        // The overlap region [0, overlap_samples) is overwritten below, so it does not
        // advance the global phase (phase_in already accounts for it via the previous
        // chunk's integral over the same frames).
        let mut source: Vec<f32> = vec![0f32; n];
        let mut acc = phase_in;
        for s in overlap_samples.min(n)..n {
            let fhz = f0v[s / F0_UPSAMPLE] as f64;
            acc += fhz / MEL_SR as f64;
            source[s] = (SINE_AMP * (2.0 * std::f64::consts::PI * acc).sin()) as f32;
        }
        let phase_out = acc;

        // Overwrite the overlap with the previous chunk's trailing source (phase-coherent).
        if let Some(t) = prev_tail {
            let prev: Vec<f32> = t.flatten_all()?.to_vec1::<f32>()?;
            let m = overlap_samples.min(prev.len()).min(n);
            let off = prev.len() - m; // align on the tail if lengths differ
            source[..m].copy_from_slice(&prev[off..off + m]);
        }

        let s_stft = self.source_stft(&source)?;
        let src_wave = Tensor::from_vec(source, (1, n), &self.dev)?;
        Ok((s_stft, src_wave, phase_out))
    }
}
