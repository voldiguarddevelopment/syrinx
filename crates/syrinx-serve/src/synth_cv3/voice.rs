//! Voice-conditioned CV3 synthesis: drive the pipeline from a cached [`Voice`] (its
//! pre-extracted embedding / prompt_feat / prompt_token / prompt_text) instead of
//! re-running the frontend on a reference clip. Additive — [`Cv3Synthesizer::synthesize`]
//! / [`Cv3Synthesizer::prompt_cond`] are byte-unchanged; this only re-tokenizes the
//! (new) tts text and reuses the cached prompt-side conditioning.

use std::borrow::Cow;

use crate::voice::Voice;

use super::*;

impl Cv3Synthesizer {
    /// Build a [`PromptCond`] from a cached [`Voice`] + a new `tts_text`, re-deriving only
    /// the text tokens (the cheap part) and reusing the voice's cached
    /// embedding / prompt_feat / prompt_token (the expensive frontend part).
    ///
    /// The text-token construction mirrors [`Cv3Synthesizer::prompt_cond`] exactly (the
    /// `<|endofprompt|>` boundary on the prompt-text segment, `tn` normalization), so a
    /// voice-conditioned synthesis is byte-identical to a ref-conditioned one whose
    /// reference produced the same cached conditioning.
    fn cond_from_voice(&mut self, tts_text: &str, voice: &Voice) -> Result<PromptCond, SynthError> {
        // --- text tokens: prompt_text(+<|endofprompt|>) ++ tts_text (mirrors prompt_cond). ---
        let tts_text = tn_normalize(tts_text);
        let tts_text = tts_text.as_ref();
        let prompt_text = if voice.prompt_text.contains(ENDOFPROMPT) {
            Cow::Borrowed(voice.prompt_text.as_str())
        } else {
            Cow::Owned(format!("{}{ENDOFPROMPT}", tn_normalize(&voice.prompt_text)))
        };
        let prompt_text_ids = self.tokenizer.encode(prompt_text.as_ref())?;
        let tts_text_ids = self.tokenizer.encode(tts_text)?;
        let prompt_text_len = prompt_text_ids.len();
        let mut text_token = prompt_text_ids;
        text_token.extend_from_slice(&tts_text_ids);

        // --- cached prompt-side conditioning (moved onto this synth's device). ---
        let prompt_token = ids_i64_to_tensor(&voice.prompt_token, &self.dev)?;
        let spk_embedding = voice.speaker_embedding.to_device(&self.dev)?;
        let prompt_feat = voice.prompt_feat.to_device(&self.dev)?;

        Ok(PromptCond {
            text_token,
            prompt_text_len,
            spk_embedding,
            prompt_token,
            prompt_feat,
        })
    }

    /// Full CV3 synthesis of `tts_text` in a cached [`Voice`] — the
    /// [`Cv3Synthesizer::synthesize`] analogue that takes a pre-extracted voice instead of
    /// a reference clip (no per-call frontend run). `inputs` may pin the generated tokens
    /// and `z` exactly as for `synthesize`.
    pub fn synthesize_with_voice(
        &mut self,
        tts_text: &str,
        voice: &Voice,
        inputs: &Cv3SynthInputs,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.cond_from_voice(tts_text, voice)?;

        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: pinned (parity), else a SEEDED standard-normal init — identical to
        // `synthesize`.
        let z = match inputs.z.as_ref() {
            Some(z) => z.clone(),
            None => {
                let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
                self.seeded_normal_z(total, inputs.lm_seed)?
            }
        };
        let audio = self.token2wav(&cond, &speech_token, Some(&z))?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Perceptual-quality CV3 synthesis of `tts_text` in a cached [`Voice`] — the
    /// [`Cv3Synthesizer::synthesize_quality`] analogue (random-phase NSF source + seeded
    /// standard-normal `z`, both reproducible from `source_seed`).
    pub fn synthesize_quality_with_voice(
        &mut self,
        tts_text: &str,
        voice: &Voice,
        source_seed: u64,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.cond_from_voice(tts_text, voice)?;
        let speech_token = self.generate_speech_token(&cond, source_seed, None)?;

        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = self.seeded_normal_z(total, source_seed)?;

        let mel = self.flow_forward(&cond, &speech_token, &z, N_TIMESTEPS)?; // [1,80,2*Tg]
        let source = self.quality_source(&mel, source_seed)?; // [1,1,L]
        let audio = self.vocode(&mel, &source)?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }
}
