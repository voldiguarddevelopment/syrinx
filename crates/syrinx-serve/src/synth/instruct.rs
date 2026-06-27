// ============================================================================
// Instruct / emotion synthesis (additive — CosyVoice2 `inference_instruct2`).
//
// `synthesize_instruct` reproduces CosyVoice2's `frontend_instruct2`, which is
// the zero-shot path with exactly two changes (frontend.py):
//
//     frontend_instruct2(tts, instruct, wav) =
//         model_input = frontend_zero_shot(tts, prompt_text=instruct, wav)
//         del model_input['llm_prompt_speech_token']      # LM gets no ref tokens
//         del model_input['llm_prompt_speech_token_len']
//
// i.e. (1) the **instruct text takes the `prompt_text` role**, so the LM text is
// `tok(instruct) ++ tok(tts)`, and (2) the LM is driven with an **empty prompt
// speech-token prefix** (it must generate both content and the requested style
// from text alone). The flow side is untouched: it KEEPS the reference
// `flow_prompt_speech_token` (`cond.prompt_token`), `prompt_speech_feat`
// (`cond.prompt_feat`) and the speaker embedding, so the cloned voice is
// preserved while the prosody/emotion follows the instruction.
//
// This maps onto the existing pipeline with no change to the parity core:
//   * `prompt_cond(tts, instruct, ..)` already yields `text_token =
//     tok(instruct)++tok(tts)`, `prompt_text_len = |tok(instruct)|`, plus the ref
//     `prompt_token`/`prompt_feat`/`spk` — exactly the instruct conditioning;
//   * the empty LM prefix needs **no** `syrinx-lm` change: `Qwen2Lm::generate`
//     already accepts an empty `prompt_speech_token` and `build_lm_input` omits
//     the prompt-speech segment for it — the byte-for-byte analogue of CV2's
//     `prompt_speech_token_len == 0 -> torch.zeros(1, 0, H)` branch
//     (`cosyvoice/llm/llm.py` Qwen2LM.inference);
//   * `token2wav` runs unchanged, with the ref `prompt_token` kept.
//
// CV2's instruct prompts always carry a trailing `<|endofprompt|>` marker (every
// `inference_instruct2` example + `utils/common.py instruct_list`); the Syrinx
// tokenizer recognizes it as the same atomic special id CV2's
// `allowed_special='all'` emits, so `synthesize_instruct` appends it idempotently
// to match the prompt boundary the LM was trained on.
//
// `synthesize` / `prompt_cond` / `generate_speech_token` are left byte-unchanged
// (parity); all new code lives in this block.
// ============================================================================

use candle_core::Tensor;

use super::*;

/// The `<|endofprompt|>` boundary marker CosyVoice2 places at the end of every
/// instruct prompt (`inference_instruct2` examples + `utils/common.py`).
const ENDOFPROMPT: &str = "<|endofprompt|>";

impl Synthesizer {
    /// **Instruct / emotion** synthesis: speak `tts_text` in the reference voice
    /// while following the natural-language `instruct_text` (e.g. "用开心的语气说",
    /// "speak in a sad tone", "请用四川话说"), returning the 24 kHz waveform.
    ///
    /// This is CosyVoice2's `inference_instruct2` on the **same** 0.5B weights
    /// Syrinx already runs — not a separate model. It differs from
    /// [`Synthesizer::synthesize`] in exactly the two ways CV2's
    /// `frontend_instruct2` does:
    ///   1. the instruct text is fed in the **`prompt_text` role**, so the LM text
    ///      prompt is `tok(instruct_text) ++ tok(tts_text)`; and
    ///   2. the LM is driven with an **empty prompt speech-token prefix** (no
    ///      reference speech tokens), so it generates content + the requested
    ///      style from the text alone.
    ///
    /// The flow + vocoder run **identically** to `synthesize`: the reference
    /// `prompt_token`, `prompt_feat` and speaker embedding are kept, so the cloned
    /// voice is preserved and only the prosody/emotion follows the instruction.
    ///
    /// A trailing `<|endofprompt|>` is appended to `instruct_text` if absent (CV2's
    /// instruct prompts always carry it); passing it explicitly is also fine
    /// (the append is idempotent). `inputs` is honoured exactly as in `synthesize`
    /// (`pinned_speech_token`, `z`, `s_stft`, `lm_seed`, `max_gen_steps`).
    pub fn synthesize_instruct(
        &mut self,
        tts_text: &str,
        instruct_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
    ) -> Result<Vec<f32>, SynthError> {
        // (1) instruct text takes the prompt_text role; append CV2's endofprompt
        //     marker if the caller did not. `prompt_cond` then builds
        //     text_token = tok(instruct) ++ tok(tts) and the ref prompt cond.
        let instruct = if instruct_text.contains(ENDOFPROMPT) {
            std::borrow::Cow::Borrowed(instruct_text)
        } else {
            std::borrow::Cow::Owned(format!("{instruct_text}{ENDOFPROMPT}"))
        };
        let cond = self.prompt_cond(tts_text, instruct.as_ref(), ref_wav_16k, ref_wav_24k)?;

        // (2) LM speech tokens with an EMPTY prompt speech-token prefix (pinned or
        //     live). The flow keeps the ref prompt_token via `cond` below.
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => {
                self.generate_speech_token_no_prompt(&cond, inputs.lm_seed, inputs.max_gen_steps)?
            }
        };

        // flow + vocoder exactly as `synthesize` (ref prompt_token/feat/spk kept).
        let audio = self.token2wav(&cond, &speech_token, inputs)?; // [1, L]
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }

    /// Generate the speech-token sequence for the **instruct** path: identical to
    /// [`Synthesizer::generate_speech_token`] (same `min_len`/`max_len` ratios over
    /// `net = text_len - prompt_text_len = |tok(tts_text)|`, same seed/cap), except
    /// the LM is driven with an **empty** prompt speech-token prefix — CV2's
    /// `frontend_instruct2` deletes `llm_prompt_speech_token`. `Qwen2Lm::generate`
    /// already accepts an empty prefix (its `build_lm_input` omits the prompt-speech
    /// segment), so this reproduces CV2 with no LM change.
    fn generate_speech_token_no_prompt(
        &self,
        cond: &PromptCond,
        seed: u64,
        max_gen_steps: Option<usize>,
    ) -> Result<Tensor, SynthError> {
        let text_len = cond.text_token.len();
        let net = text_len.saturating_sub(cond.prompt_text_len);
        let min_len = net * MIN_TOKEN_TEXT_RATIO;
        let real_max = net * MAX_TOKEN_TEXT_RATIO;
        let max_len = match max_gen_steps {
            Some(cap) => cap.max(min_len + 1),
            None => real_max,
        };
        // Empty prompt-speech prefix: the instruct difference. The LM generates
        // content + style from `text_token` (= tok(instruct)++tok(tts)) alone.
        let gen = self.lm.generate(&cond.text_token, &[], min_len, max_len, seed)?;
        let ids: Vec<i64> = gen.iter().map(|&t| t as i64).collect();
        ids_i64_to_tensor(&ids, &self.dev).map_err(Into::into)
    }
}
