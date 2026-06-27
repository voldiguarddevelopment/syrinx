//! Voice-conditioned CV2 synthesis: drive the pipeline from a cached [`Voice`] (its
//! pre-extracted embedding / prompt_feat / prompt_token / prompt_text) instead of
//! re-running the frontend on a reference clip. Additive — [`Synthesizer::synthesize`] /
//! [`Synthesizer::prompt_cond`] are byte-unchanged; this only re-tokenizes the (new) tts
//! text and reuses the cached prompt-side conditioning.

use crate::voice::Voice;

use super::*;

impl Synthesizer {
    /// Build a [`PromptCond`] from a cached [`Voice`] + a new `tts_text`, re-deriving only
    /// the text tokens and reusing the voice's cached embedding / prompt_feat /
    /// prompt_token. The text-token construction mirrors [`Synthesizer::prompt_cond`]
    /// (CosyVoice2 concatenates `prompt_text ++ tts_text`, with the `tn` hook).
    fn cond_from_voice(&mut self, tts_text: &str, voice: &Voice) -> Result<PromptCond, SynthError> {
        let prompt_text = tn_normalize(&voice.prompt_text);
        let prompt_text = prompt_text.as_ref();
        let tts_text = tn_normalize(tts_text);
        let tts_text = tts_text.as_ref();
        let prompt_text_ids = self.tokenizer.encode(prompt_text)?;
        let tts_text_ids = self.tokenizer.encode(tts_text)?;
        let prompt_text_len = prompt_text_ids.len();
        let mut text_token = prompt_text_ids;
        text_token.extend_from_slice(&tts_text_ids);

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

    /// Full CV2 synthesis of `tts_text` in a cached [`Voice`] — the
    /// [`Synthesizer::synthesize`] analogue that takes a pre-extracted voice instead of a
    /// reference clip (no per-call frontend run). `inputs` may pin the generated tokens,
    /// `z`, and the HiFT source exactly as for `synthesize`.
    pub fn synthesize_with_voice(
        &mut self,
        tts_text: &str,
        voice: &Voice,
        inputs: &SynthInputs,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.cond_from_voice(tts_text, voice)?;

        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        let audio = self.token2wav(&cond, &speech_token, inputs)?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }
}
