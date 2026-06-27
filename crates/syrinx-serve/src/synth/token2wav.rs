//! Flow + vocoder glue: turn the speech tokens into the 24 kHz waveform tensor
//! (the `token2wav` half), plus the isolated flow-forward timing entry point.

use candle_core::{DType, Tensor};

use syrinx_acoustic::cv2::token2wav;

use super::*;

impl Synthesizer {
    /// The flow + vocoder half: speech tokens -> mel -> audio `[1, L]`. `z` and
    /// `s_stft` come from `inputs` when pinned, else are derived (zeros `z`,
    /// deterministic F0 source).
    pub fn token2wav(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        inputs: &SynthInputs,
    ) -> Result<Tensor, SynthError> {
        // total flow length = 2 * (|prompt_token| + |speech_token|).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);

        // CFM noise z: pinned or zeros fallback.
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };

        // HiFT source STFT: pinned, or a deterministic zero-phase F0 source.
        let s_stft = match &inputs.s_stft {
            Some(s) => s.clone(),
            None => {
                // Need the generated mel to drive the F0 predictor, so compute it
                // here (forward_zero_shot), build the source, then decode with both.
                let mel = self.flow.forward_zero_shot(
                    &cond.prompt_token,
                    speech_token,
                    &cond.prompt_feat,
                    &cond.spk_embedding,
                    &z,
                    N_TIMESTEPS,
                )?; // [1, 80, 2*Tg]
                let s = self.deterministic_source_stft(&mel)?;
                let audio = self.vocoder.decode(&mel, &s)?;
                return Ok(audio);
            }
        };

        // Pinned-source path: the single-call token2wav glue.
        let audio = token2wav(
            &self.flow,
            &self.vocoder,
            &cond.prompt_token,
            speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            &s_stft,
            N_TIMESTEPS,
        )?;
        Ok(audio)
    }

    /// Run only the flow-matching CFM mel decoder (`forward_zero_shot`) — the
    /// heaviest acoustic stage — returning the generated mel `[1, 80, 2*Tg]`.
    ///
    /// Exposed so callers (e.g. the GPU speed benchmark) can time the flow ODE in
    /// isolation. The full pipeline uses this internally inside [`token2wav`].
    pub fn flow_forward(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor, SynthError> {
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            z,
            n_timesteps,
        )?;
        Ok(mel)
    }
}
