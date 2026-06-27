//! LM speech-token generation: drive the live Qwen2 LM (pinned-PRNG sampling) to
//! turn the prompt-conditioned text tokens into a speech-token sequence.

use candle_core::Tensor;

use super::*;

impl Synthesizer {
    /// Generate the speech-token sequence with the **live** LM (pinned-PRNG sampling).
    /// Returns the ids as i64 `[1, N]` ready for the flow.
    pub fn generate_speech_token(
        &self,
        cond: &PromptCond,
        seed: u64,
        max_gen_steps: Option<usize>,
    ) -> Result<Tensor, SynthError> {
        let text_len = cond.text_token.len();
        let net = text_len.saturating_sub(cond.prompt_text_len);
        let min_len = net * MIN_TOKEN_TEXT_RATIO;
        let real_max = net * MAX_TOKEN_TEXT_RATIO;
        // Honour any caller cap (never below min_len so the EOS-mask window is intact).
        let max_len = match max_gen_steps {
            Some(cap) => cap.max(min_len + 1),
            None => real_max,
        };
        let prompt_speech_u32 = tensor_ids_u32(&cond.prompt_token)?;
        let gen = self
            .lm
            .generate(&cond.text_token, &prompt_speech_u32, min_len, max_len, seed)?;
        let ids: Vec<i64> = gen.iter().map(|&t| t as i64).collect();
        ids_i64_to_tensor(&ids, &self.dev).map_err(Into::into)
    }
}
