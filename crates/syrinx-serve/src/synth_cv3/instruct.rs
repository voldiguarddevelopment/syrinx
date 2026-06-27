//! CV3 instruct / emotion synthesis (CosyVoice3 `inference_instruct2` on the same CV3
//! weights): instruct text in the `prompt_text` role + an empty prompt speech-token
//! prefix, with the reference voice kept on the flow side.

use super::*;

impl Cv3Synthesizer {
    /// **Instruct / emotion** synthesis: speak `tts_text` in the reference voice while
    /// following the natural-language `instruct_text` (e.g. "用开心的语气说", "speak in a
    /// sad tone"), returning the 24 kHz waveform. This is CosyVoice3's `inference_instruct2`
    /// on the **same** CV3 weights — not a separate model.
    ///
    /// It differs from [`Cv3Synthesizer::synthesize`] exactly as CV3's `frontend_instruct2`
    /// does (`cosyvoice/cli/frontend.py:209`):
    ///   1. the instruct text is fed in the **`prompt_text` role**, so the LM text prompt is
    ///      `tok(instruct_text ++ <|endofprompt|>) ++ tok(tts_text)`; and
    ///   2. the LM is driven with an **empty prompt speech-token prefix** (`frontend_instruct2`
    ///      deletes `llm_prompt_speech_token`) — content + requested style come from text
    ///      alone — assembled by [`Cv3Lm::build_lm_input_instruct`] (the CV3 structural
    ///      instruct slot `[sos, instruct, text, task_id]`).
    ///
    /// The flow + vocoder run **identically** to `synthesize` (reference `prompt_token`,
    /// `prompt_feat`, speaker embedding kept), so the cloned voice is preserved and only the
    /// prosody/emotion follows the instruction. CV3 requires a trailing `<|endofprompt|>` on
    /// the instruct text (asserted by `Qwen2LM.inference`); it is appended if absent (the
    /// append is idempotent). The LM sampling seed is fixed (0) for reproducibility.
    pub fn synthesize_instruct(
        &mut self,
        tts_text: &str,
        instruct_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<Vec<f32>, SynthError> {
        // (1) instruct text takes the prompt_text role; append CV3's <|endofprompt|> marker
        //     if the caller did not. `prompt_cond` then builds the ref prompt conditioning
        //     and text_token = tok(instruct ++ <|endofprompt|>) ++ tok(tts).
        let instruct = if instruct_text.contains(ENDOFPROMPT) {
            std::borrow::Cow::Borrowed(instruct_text)
        } else {
            std::borrow::Cow::Owned(format!("{instruct_text}{ENDOFPROMPT}"))
        };
        let cond = self.prompt_cond(tts_text, instruct.as_ref(), ref_wav_16k, ref_wav_24k)?;

        // (2) LM with an EMPTY prompt speech-token prefix via the structural instruct input.
        //     text_token = instruct(prefix) ++ tts(suffix); split at prompt_text_len.
        let instruct_ids = &cond.text_token[..cond.prompt_text_len];
        let tts_ids = &cond.text_token[cond.prompt_text_len..];
        let net = tts_ids.len();
        let min_len = net * MIN_TOKEN_TEXT_RATIO;
        let max_len = net * MAX_TOKEN_TEXT_RATIO;
        let gen = self
            .lm
            .generate_instruct(instruct_ids, tts_ids, min_len, max_len, 0)?;
        let ids: Vec<i64> = gen.iter().map(|&t| t as i64).collect();
        let speech_token = ids_i64_to_tensor(&ids, &self.dev)?;

        // flow + vocoder exactly as `synthesize` (ref prompt_token/feat/spk kept; det source).
        // Use the same seeded standard-normal CFM init as `synthesize` (seed 0), not zeros.
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = self.seeded_normal_z(total, 0)?;
        let audio = self.token2wav(&cond, &speech_token, Some(&z))?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }
}
