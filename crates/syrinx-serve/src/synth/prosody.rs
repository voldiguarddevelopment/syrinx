// ============================================================================
// Prosody control hooks (additive — speech-rate + editable render plan).
//
// Both hooks live here, separate from the core pipeline. The rate knob time-scales
// the *generated mel* along its frame axis (via `syrinx_prosody::render::time_scale_mel`)
// between the flow and the vocoder: more frames => longer/slower audio, fewer =>
// shorter/faster, with each frame's spectral shape (and thus pitch) preserved. The
// render-plan hook additionally retunes the HiFT F0 (pitch) and can bin-warp the mel
// (formant shift). The non-`real`-deterministic (live z=zeros / F0 source) path is
// used, matching the functional smoke; a pinned-source parity run keeps the original
// `synthesize` untouched.
// ============================================================================

use candle_core::{DType, Device, Tensor};

use super::*;

impl Synthesizer {
    /// Synthesize `tts_text` in the reference voice at speech-rate `rate`,
    /// returning the 24 kHz waveform.
    ///
    /// `rate == 1.0` reproduces the default-speed [`Synthesizer::synthesize`]
    /// audio (up to mel-frame rounding); `rate > 1.0` is faster/shorter and
    /// `rate < 1.0` slower/longer, with pitch preserved. The rate is applied by
    /// time-scaling the generated mel before the vocoder; the speech-token
    /// sequence and prompt conditioning are unchanged, so the *content* is the
    /// same utterance, only its duration differs. A non-positive or non-finite
    /// `rate` returns [`SynthError::Candle`] describing the bad knob.
    ///
    /// This always uses the deterministic (live `z = zeros`, deterministic F0)
    /// source — the same functional path as the default smoke — so any pinned
    /// `s_stft` in `inputs` is ignored here (a pinned source is length-locked to
    /// the unscaled mel and cannot describe a rate-scaled one). `inputs` is still
    /// honoured for `pinned_speech_token`, `lm_seed`, and `max_gen_steps`.
    pub fn synthesize_with_rate(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        rate: f64,
    ) -> Result<Vec<f32>, SynthError> {
        if !(rate.is_finite() && rate > 0.0) {
            return Err(SynthError::Candle(format!(
                "speech rate must be finite and > 0, got {rate}"
            )));
        }
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: pinned or zeros (the deterministic functional fallback).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // Flow -> generated mel [1, 80, 2*Tg].
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            N_TIMESTEPS,
        )?;

        // The rate knob: time-scale the mel along its frame axis (pitch-preserving).
        let mel = time_scale_mel_tensor(&mel, rate, &self.dev)?;

        // Deterministic source from the (now rate-scaled) mel, then vocode.
        let s = self.deterministic_source_stft(&mel)?;
        let audio = self.vocoder.decode(&mel, &s)?;
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }

    /// Synthesize `tts_text` in the reference voice under an editable
    /// [`RenderPlan`](syrinx_prosody::render_plan::RenderPlan), returning the 24 kHz
    /// waveform.
    ///
    /// The plan is applied to the flow's generated mel before the vocoder:
    ///   * **duration** — the mel frame axis is time-warped by the plan's global +
    ///     per-region rate (pitch-preserving length regulation);
    ///   * **pitch** — the HiFT F0 predicted from the (warped) mel is multiplied by
    ///     the plan's per-frame F0 ratio (global + per-region semitone shift) before
    ///     the sine source is built — a formant-preserving pitch shift via the NSF
    ///     source; with `mel_envelope_shift` the mel is also bin-warped (formant
    ///     shifting, opt-in).
    ///
    /// [`RenderPlan::identity`](syrinx_prosody::render_plan::RenderPlan::identity)
    /// reproduces the default-speed [`Synthesizer::synthesize`] audio (up to mel
    /// rounding). A plan that fails validation against the generated mel length
    /// returns [`SynthError::Candle`] describing the bad knob.
    pub fn synthesize_with_plan(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        plan: &syrinx_prosody::render_plan::RenderPlan,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: pinned or zeros (the deterministic functional fallback).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // Flow -> generated mel [1, 80, 2*Tg].
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            N_TIMESTEPS,
        )?;

        // Apply the plan to the mel grid: time-warp (duration) + per-output-frame
        // F0 multiplier (pitch) + opt-in mel-bin formant shift.
        let grid = mel_tensor_to_grid(&mel)?;
        let (scaled_grid, f0_mult) = plan
            .apply(&grid)
            .map_err(|e| SynthError::Candle(format!("prosody plan apply failed: {e:?}")))?;
        let scaled_mel = grid_to_mel_tensor(&scaled_grid, &self.dev)?; // [1, 80, T_out]

        // Predict F0 from the warped mel, retune it by the per-frame ratio, build
        // the source, then vocode.
        let f0 = self.vocoder.f0_predict(&scaled_mel)?; // [1, T_out]
        let mut f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        if f0v.len() != f0_mult.len() {
            return Err(SynthError::Candle(format!(
                "prosody plan F0 length mismatch: f0={} mult={}",
                f0v.len(),
                f0_mult.len()
            )));
        }
        for (v, m) in f0v.iter_mut().zip(f0_mult.iter()) {
            *v = (*v as f64 * *m) as f32;
        }
        let s = self.f0_to_source_stft(&f0v)?;
        let audio = self.vocoder.decode(&scaled_mel, &s)?;
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }
}

/// Convert a `[1, n_mels, T]` Candle mel tensor to a `[n_mels][T]` row-major grid
/// for the pure-Rust `syrinx-prosody` transforms (keeping that crate Candle-free).
fn mel_tensor_to_grid(mel: &Tensor) -> Result<Vec<Vec<f32>>, SynthError> {
    let (b, n_mels, t_in) = mel.dims3()?;
    if b != 1 {
        return Err(SynthError::Candle(format!(
            "mel grid conversion expects batch 1, got {b}"
        )));
    }
    let m2 = mel.squeeze(0)?; // [n_mels, T]
    let flat: Vec<f32> = m2.flatten_all()?.to_vec1::<f32>()?;
    let mut grid: Vec<Vec<f32>> = Vec::with_capacity(n_mels);
    for row in 0..n_mels {
        grid.push(flat[row * t_in..(row + 1) * t_in].to_vec());
    }
    Ok(grid)
}

/// Rebuild a `[1, n_mels, T]` Candle mel tensor from a `[n_mels][T]` grid on `dev`.
fn grid_to_mel_tensor(grid: &[Vec<f32>], dev: &Device) -> Result<Tensor, SynthError> {
    let n_mels = grid.len();
    let t = if n_mels == 0 { 0 } else { grid[0].len() };
    let mut flat: Vec<f32> = Vec::with_capacity(n_mels * t);
    for row in grid {
        flat.extend_from_slice(row);
    }
    Ok(Tensor::from_vec(flat, (1, n_mels, t), dev)?)
}

/// Time-scale a `[1, n_mels, T]` Candle mel tensor by speech-rate `rate` using
/// [`syrinx_prosody::render::time_scale_mel`], returning `[1, n_mels, T_out]`.
///
/// The tensor is moved to a flat `[n_mels][T]` grid for the pure-Rust transform
/// (keeping `syrinx-prosody` Candle-free) and rebuilt on `dev`. `rate > 1` yields
/// fewer frames (shorter), `rate < 1` more (longer); the band axis is untouched.
fn time_scale_mel_tensor(
    mel: &Tensor,
    rate: f64,
    dev: &Device,
) -> Result<Tensor, SynthError> {
    let grid = mel_tensor_to_grid(mel)?; // [n_mels][T]
    let scaled = syrinx_prosody::render::time_scale_mel(&grid, rate)
        .map_err(|e| SynthError::Candle(format!("mel time-scale failed: {e:?}")))?;
    grid_to_mel_tensor(&scaled, dev)
}
