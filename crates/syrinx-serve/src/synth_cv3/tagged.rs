//! Inline **multi-emotion** CV3 synthesis: render `"[happy] hi there [sad] bye"` so each
//! span is spoken with its tag's emotion and the spans are concatenated into one waveform.
//!
//! This is the tensor-side glue over the model-free [`crate::emotion`] module:
//!   1. [`parse_tagged`] splits the text into [`Segment`]s (text + emotion-in-effect);
//!   2. each segment is synthesized **in the reference voice** — an emotion span via
//!      [`Cv3Synthesizer::synthesize_instruct`] (the tag's CV3 instruct string), a neutral
//!      span via the plain [`Cv3Synthesizer::synthesize`];
//!   3. the per-segment waveforms are joined with a short equal-power cross-fade
//!      ([`concat_crossfade`]) so an emotion change does not click at the seam.
//!
//! Additive: the existing `synthesize*` paths are byte-unchanged. The instruct string for
//! each tag comes from the supplied [`EmotionRegistry`] (its active [`InstructLang`]); the
//! `<|endofprompt|>` marker is appended by `synthesize_instruct`, so the registry strings
//! stay clean.
//!
//! ## Honest limits
//! The emotion is only as good as CV3's instruct-following — a single instruct string
//! steers prosody, it is not a separate emotion model. The boundary cross-fade smooths the
//! *audio* seam (no click) but does not blend *prosody* across an emotion change: pitch /
//! energy can still step at a boundary, because each span is a fully independent CV3 run.

use crate::emotion::{concat_crossfade, parse_tagged, EmotionRegistry, DEFAULT_XFADE_SAMPLES};
use crate::voice::Voice;

use super::*;

impl Cv3Synthesizer {
    /// Synthesize inline-tagged text into one waveform: parse `tagged_text` into emotion
    /// segments, render each in the reference voice (emotion spans via `synthesize_instruct`
    /// with the tag's instruct string, neutral spans via `synthesize`), and concatenate with
    /// an equal-power cross-fade.
    ///
    /// `prompt_text` is the reference transcript (used for neutral spans, exactly as
    /// `synthesize`); `ref_wav_16k` / `ref_wav_24k` are the resampled reference waveforms;
    /// `registry` supplies the `tag -> instruct` mapping and the active instruct language;
    /// `inputs` is forwarded to the neutral-span `synthesize` (the LM seed / step cap).
    ///
    /// Returns the 24 kHz waveform. An empty / all-whitespace `tagged_text` (no segments)
    /// returns an empty `Vec` rather than erroring. Unknown tags are spoken neutrally (the
    /// parser logs a warning).
    pub fn synthesize_tagged(
        &mut self,
        tagged_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        registry: &EmotionRegistry,
        inputs: &Cv3SynthInputs,
    ) -> Result<Vec<f32>, SynthError> {
        let segments = parse_tagged(tagged_text, registry);
        let mut waves: Vec<Vec<f32>> = Vec::with_capacity(segments.len());
        for seg in &segments {
            let wav = match seg.emotion.as_deref().and_then(|tag| registry.instruct(tag)) {
                // An emotion span: speak the segment text following the tag's instruct.
                Some(instruct) => {
                    self.synthesize_instruct(&seg.text, instruct, ref_wav_16k, ref_wav_24k)?
                }
                // A neutral span (no tag, or an unknown tag): plain synthesis.
                None => {
                    self.synthesize(&seg.text, prompt_text, ref_wav_16k, ref_wav_24k, inputs)?
                }
            };
            waves.push(wav);
        }
        Ok(concat_crossfade(&waves, DEFAULT_XFADE_SAMPLES))
    }

    /// Inline-tagged synthesis from a cached [`Voice`] (no per-call frontend run) — the
    /// [`Cv3Synthesizer::synthesize_tagged`] analogue of
    /// [`Cv3Synthesizer::synthesize_with_voice`].
    ///
    /// Each segment reuses the voice's cached prompt conditioning (speaker embedding /
    /// `prompt_feat` / `prompt_token`): neutral spans go through `synthesize_with_voice`,
    /// emotion spans through an instruct path that keeps the cached flow-side voice but puts
    /// the tag's instruct string in the LM prompt-text role with an empty prompt
    /// speech-token prefix (the same `inference_instruct2` shape as `synthesize_instruct`,
    /// driven from the voice cache instead of a reference clip).
    pub fn synthesize_tagged_with_voice(
        &mut self,
        tagged_text: &str,
        voice: &Voice,
        registry: &EmotionRegistry,
        inputs: &Cv3SynthInputs,
    ) -> Result<Vec<f32>, SynthError> {
        let segments = parse_tagged(tagged_text, registry);
        let mut waves: Vec<Vec<f32>> = Vec::with_capacity(segments.len());
        for seg in &segments {
            let wav = match seg.emotion.as_deref().and_then(|tag| registry.instruct(tag)) {
                Some(instruct) => self.synthesize_instruct_with_voice(&seg.text, instruct, voice)?,
                None => self.synthesize_with_voice(&seg.text, voice, inputs)?,
            };
            waves.push(wav);
        }
        Ok(concat_crossfade(&waves, DEFAULT_XFADE_SAMPLES))
    }

    /// Instruct synthesis (`inference_instruct2`) driven from a cached [`Voice`]: the
    /// [`Cv3Synthesizer::synthesize_instruct`] analogue that reuses the voice's cached
    /// prompt-side conditioning instead of re-running the frontend on a reference clip.
    ///
    /// Mirrors `synthesize_instruct` exactly — the instruct text takes the prompt-text role
    /// (`tok(instruct ++ <|endofprompt|>) ++ tok(tts)`), the LM runs with an empty prompt
    /// speech-token prefix via [`Cv3Lm::generate_instruct`], and the flow + vocoder run on
    /// the cached `prompt_token` / `prompt_feat` / speaker embedding — only the prompt
    /// conditioning source differs (voice cache vs live frontend). The CFM noise `z` is the
    /// same seeded standard-normal as `synthesize_instruct` (seed 0).
    fn synthesize_instruct_with_voice(
        &mut self,
        tts_text: &str,
        instruct_text: &str,
        voice: &Voice,
    ) -> Result<Vec<f32>, SynthError> {
        // Instruct text takes the prompt-text role; append CV3's <|endofprompt|> if absent.
        let instruct = if instruct_text.contains(ENDOFPROMPT) {
            std::borrow::Cow::Borrowed(instruct_text)
        } else {
            std::borrow::Cow::Owned(format!("{instruct_text}{ENDOFPROMPT}"))
        };
        let cond = self.cond_from_voice_role(instruct.as_ref(), tts_text, voice)?;

        // LM with an EMPTY prompt speech-token prefix (the structural instruct input);
        // text_token = instruct(prefix) ++ tts(suffix), split at prompt_text_len.
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

        // flow + vocoder exactly as `synthesize_instruct` (seeded standard-normal z, seed 0).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = self.seeded_normal_z(total, 0)?;
        let audio = self.token2wav(&cond, &speech_token, Some(&z))?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Build a [`PromptCond`] from a cached [`Voice`] whose **prompt-text role** is an
    /// arbitrary `role_text` (an instruct string), rather than the voice's own transcript.
    ///
    /// Identical in shape to the voice module's `cond_from_voice` (cached
    /// embedding / `prompt_feat` / `prompt_token`, `tok(role ++ <|endofprompt|>) ++
    /// tok(tts)` text tokens) — only the text in the prompt-text role differs, which is
    /// exactly the instruct/zero-shot distinction. `role_text` is passed already carrying
    /// `<|endofprompt|>`, so it is tokenized verbatim (no tn mangling of the marker).
    fn cond_from_voice_role(
        &mut self,
        role_text: &str,
        tts_text: &str,
        voice: &Voice,
    ) -> Result<PromptCond, SynthError> {
        let tts_text = tn_normalize(tts_text);
        let prompt_text_ids = self.tokenizer.encode(role_text)?;
        let tts_text_ids = self.tokenizer.encode(tts_text.as_ref())?;
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
}
