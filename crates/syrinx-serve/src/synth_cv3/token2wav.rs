//! CV3 flow + vocoder glue: the `token2wav` half (flow -> deterministic source ->
//! HiFT), the seeded CFM-noise `z` builder, and the isolated flow-forward /
//! vocode / reference-flow entry points.

use candle_core::{DType, Tensor};

use super::*;

impl Cv3Synthesizer {
    /// The flow + vocoder half: speech tokens -> mel -> audio `[1, L]`. `z` is the CFM
    /// noise; the HiFT source is the deterministic F0 source. A `Some(z)` is used as-is;
    /// a `None` falls back to an **explicit zeros** init — the degenerate mean-ODE start,
    /// retained only for callers that deliberately want it (parity). The live
    /// [`Cv3Synthesizer::synthesize`] no longer passes `None`: it fills a seeded
    /// standard-normal `z` first (see [`Cv3Synthesizer::seeded_normal_z`]).
    pub fn token2wav(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        z: Option<&Tensor>,
    ) -> Result<Tensor, SynthError> {
        // total flow length = 2 * (|prompt_token| + |speech_token|).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match z {
            Some(z) => z.clone(),
            None => Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &self.dev)?,
        };
        let mel = self.flow_forward(cond, speech_token, &z, N_TIMESTEPS)?; // [1,80,2*Tg]
        let source = self.deterministic_source(&mel)?; // [1, 1, L_src]
        self.vocode(&mel, &source) // [1, L]
    }

    /// Build the default CFM noise `z` `[1, 80, total]`: a **seeded standard-normal** init
    /// (the flow's `rand_noise` *distribution*, matched in distribution to `torch.randn`,
    /// not its byte stream). This is the live default `synthesize` / `synthesize_quality` /
    /// `synthesize_instruct` now use in place of the degenerate zeros init: a zeros `z`
    /// starts the CFM ODE at the conditional mean and integrates to a low-detail, smeared
    /// mel (the live-quality defect), whereas a true Gaussian draw is the distribution the
    /// flow was trained to denoise. Seeded (never system RNG), so a given `(tokens, seed)`
    /// is bit-reproducible. The byte-exact reference `z` is still injectable via
    /// [`Cv3SynthInputs::z`] for the deterministic parity chain.
    pub(super) fn seeded_normal_z(&self, total: usize, seed: u64) -> Result<Tensor, SynthError> {
        let nz = MEL_NUM_MELS * total;
        let mut rng = SplitMix64::new(seed ^ 0x5DEE_CE66_C0DE_F10D);
        let zv: Vec<f32> = (0..nz).map(|_| rng.next_gauss() as f32).collect();
        Ok(Tensor::from_vec(zv, (1, MEL_NUM_MELS, total), &self.dev)?)
    }

    /// Run only the CV3 flow CFM mel decoder for live conditioning: feeds `cond`'s
    /// prompt token/feat/embedding plus the generated `speech_token` + noise `z` to
    /// [`Cv3Flow::forward`], returning the generated mel `[1, 80, 2*Tg]` (prompt-mel
    /// prefix dropped). Exposed so callers can inspect the intermediate mel (e.g. the
    /// smoke test prints its shape before vocoding).
    pub fn flow_forward(
        &self,
        cond: &PromptCond,
        speech_token: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor, SynthError> {
        let mel = self.flow.forward(
            &cond.prompt_token,
            speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            z,
            n_timesteps,
        )?;
        Ok(mel)
    }

    /// Vocode a generated mel `[1, 80, T]` with a source waveform `[1, 1, L]` into 24 kHz
    /// audio `[1, L]` via [`Cv3Hift::decode`]. The decode STFTs the source internally.
    pub fn vocode(&self, mel: &Tensor, source: &Tensor) -> Result<Tensor, SynthError> {
        Ok(self.vocoder.decode(mel, source)?)
    }

    /// Run only the CV3 flow CFM mel decoder over **explicit reference** prompt/speech
    /// tokens + conditioning — the parity-test entry point.
    ///
    /// Feeds `prompt_token`, `token`, `prompt_feat`, `embedding`, `z` straight to
    /// [`Cv3Flow::forward`] and returns the generated mel `[1, 80, 2*Tg]` (the prompt-mel
    /// prefix is dropped inside `forward`). Used by the deterministic chain anchor: feed
    /// the fixture's reference `speech_token`/`prompt_feat`/`embedding` and compare the
    /// mel to the reference `mel`, verifying frontend->flow end to end without invoking
    /// the non-bit-reproducible LM sampler.
    pub fn flow_from_reference_tokens(
        &self,
        prompt_token: &Tensor,
        token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor, SynthError> {
        let mel = self
            .flow
            .forward(prompt_token, token, prompt_feat, embedding, z, n_timesteps)?;
        Ok(mel)
    }
}
