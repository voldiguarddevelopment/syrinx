//! CV3 LM body glue: loaders + footprint pass-throughs, the CV3 step-0 input assembly
//! (`build_lm_input` / `build_lm_input_instruct`), and the forward / teacher-forced /
//! cached-step logit paths.
//!
//! Split out verbatim from the original single-file CV3 port. `step_logits_cached` is
//! `pub(super)` (driven by the `generate` loops); everything else keeps its original
//! visibility.

use super::{Cv3Lm, DECODER_OUT, ENDOFPROMPT_ID, SOS, TASK_ID};
use crate::cv2::{Footprint, KvCache, Qwen2Lm};
use candle_core::{Device, Result, Tensor};

impl Cv3Lm {
    /// Load the converted fp32 CV3 checkpoint (`llm_fp32.safetensors`) onto `dev`.
    ///
    /// Every weight is normalised to f32 by [`Qwen2Lm::load`] — the parity build. The CV3
    /// checkpoint's key layout matches CV2's (`llm.model.model.layers.N.*`,
    /// `llm.model.model.{embed_tokens,norm}.weight`, `speech_embedding.weight`,
    /// `llm_decoder.weight`), so the shared body loads it directly. The untied
    /// `llm.model.lm_head.weight` is present but never on the speech path.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        Ok(Self {
            body: Qwen2Lm::load(path, dev)?,
        })
    }

    /// Load the same CV3 `llm_fp32.safetensors`, but quantize the LM to **int4** (GGML
    /// `Q4_0` big linears + per-row int4 embeddings) for a ~4× smaller LM footprint — the
    /// README size goal, the CV3 analogue of [`Qwen2Lm::load_quantized`].
    ///
    /// CV3's speech LM is the *same* Qwen2-0.5B backbone + speech head as CV2, so the int4
    /// scheme is reused verbatim through the shared body: every layer's `q/k/v/o_proj` and
    /// `gate/up/down_proj` plus the `llm_decoder` head become `Q4_0` `QMatMul`s, and the
    /// embedding tables (`embed_tokens` / `speech_embedding`) become per-row int4
    /// dequant-on-gather tables. RMSNorm weights and the q/k/v biases stay f32. (CV3 has no
    /// separate `llm_embedding` table — `sos`/`task_id` are rows of `speech_embedding` — so
    /// that name is simply absent; nothing else changes.)
    ///
    /// ★ CV3 `lm_head` is **tied to `embed_tokens`**: in this checkpoint
    /// `llm.model.lm_head.weight` is byte-identical to `llm.model.model.embed_tokens.weight`
    /// (verified on the model box), i.e. the *shared text embedding*, not CV2's distinct
    /// ~520 MB dead-weight matrix. The shared body's loader drops the `lm_head`-named copy,
    /// but that is **lossless here**: the very same weights are retained — and int4-quantized
    /// — under `embed_tokens` (the table the speech path actually gathers from via
    /// `text_embed`). So the shared text embedding is preserved in the footprint; only the
    /// redundant duplicate of it is not resident twice. The CV3 speech forward
    /// (`forward_hidden` → `llm_decoder`) never calls `lm_head`, so dropping the duplicate
    /// changes no output.
    ///
    /// int4 trades accuracy for size; the forward is otherwise the identical code path, so
    /// quantized logits track but do not equal the fp32 logits. ⚠️ Like CV2's, this int4
    /// path is an **opt-in size win, not a speed win** — the per-row int4 embedding is
    /// dequantized on every gather (a load-time-dominant cost), so inference stalls vs the
    /// fp32 [`Cv3Lm::load`]; choose it for footprint, not latency.
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        Ok(Self {
            body: Qwen2Lm::load_quantized(path, dev)?,
        })
    }

    /// Realized weight footprint (int4 quant + per-row int4 embeds + dense f32) of this
    /// loaded CV3 LM — a thin pass-through to the shared body's [`Qwen2Lm::footprint`]. In
    /// the fp32 [`Cv3Lm::load`] build every weight is dense (`quant_bytes == 0`).
    pub fn footprint(&self) -> Footprint {
        self.body.footprint()
    }

    /// Per-tensor `(name, bytes)` of every retained **dense** weight, largest first — the
    /// instrumentation that surfaces whatever (if anything) is left un-quantized. A
    /// pass-through to [`Qwen2Lm::dense_breakdown`]; in the int4 build this should be only
    /// the tiny RMSNorm weights + attention biases (the tied `lm_head` duplicate is gone).
    pub fn dense_breakdown(&self) -> Vec<(String, usize)> {
        self.body.dense_breakdown()
    }

    /// Assemble the step-0 LM input exactly as CosyVoice3 `Qwen2LM.inference`:
    /// `cat[ sos_emb, embed_tokens(text_token), task_id_emb, speech_embedding(prompt_speech_token) ]`
    /// along the sequence axis → `[1, T0, 896]`.
    ///
    /// `text_token` is the concatenation of the prompt text and the synthesis text token
    /// ids (the reference concatenates them before embedding). `sos_emb` / `task_id_emb`
    /// are `speech_embedding.weight[sos]` / `[task_id]` — the CV3-specific source. The
    /// prompt-speech segment is omitted when `prompt_speech_token` is empty. Every part is
    /// a pure embedding lookup, so this reproduces the reference `input_embeds` bit-for-bit.
    pub fn build_lm_input(&self, text_token: &[u32], prompt_speech_token: &[u32]) -> Result<Tensor> {
        let sos = self.body.speech_embed(&[SOS])?; // [1,1,H]  speech_embedding[sos]
        let task = self.body.speech_embed(&[TASK_ID])?; // [1,1,H]  speech_embedding[task_id]
        let text = self.body.text_embed(text_token)?; // [1,Tt,H] embed_tokens(text_token)
        let mut parts: Vec<Tensor> = vec![sos, text, task];
        if !prompt_speech_token.is_empty() {
            parts.push(self.body.speech_embed(prompt_speech_token)?); // [1,Tp,H]
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Assemble the **instruct** step-0 LM input, mirroring CosyVoice3's structural
    /// instruct slot in `CosyVoice3LM`/`Qwen2LM.prepare_lm_input_target`
    /// (`cosyvoice/llm/llm.py:343`): the unistream training form places the instruct
    /// tokens *between* `sos` and the synthesis text —
    /// `cat[ sos_emb, embed_tokens(instruct_token), embed_tokens(text_token), task_id_emb ]`
    /// → `[1, T0, 896]`. As in CV2's `inference_instruct2`, the LM is driven with an
    /// **empty prompt speech-token prefix** (the reference speech tokens are dropped on
    /// the LM side — `frontend_instruct2` deletes `llm_prompt_speech_token`), so there is
    /// no trailing speech segment here.
    ///
    /// Note the equivalence to the runtime path: CosyVoice3's `CosyVoice3Model.tts` calls
    /// the *inherited* `Qwen2LM.inference` (it does **not** pass `instruct_token`), which
    /// builds `cat[sos, embed_tokens(prompt_text ++ tts_text), task_id, prompt_speech_emb]`
    /// with the instruct text in the `prompt_text` role. Because `embed_tokens` is a
    /// per-token lookup, `embed(instruct) ++ embed(text)` equals `embed(instruct ++ text)`,
    /// so this structural assembly is **byte-identical** to the runtime construction when
    /// the prompt-speech prefix is empty. `instruct_token` must carry the trailing
    /// `<|endofprompt|>` id (151646) that CV3 asserts on the instruct/prompt text.
    ///
    /// Additive — [`Cv3Lm::build_lm_input`] is left byte-unchanged.
    pub fn build_lm_input_instruct(
        &self,
        instruct_token: &[u32],
        text_token: &[u32],
    ) -> Result<Tensor> {
        // Direct-caller safety: CV3 requires the `<|endofprompt|>` marker on the instruct
        // text (`Qwen2LM.inference` asserts it). The higher-level `synthesize_instruct`
        // appends it before tokenizing; a direct caller that forgets it would silently
        // build a malformed prompt — catch that here (debug builds; never affects release
        // or the parity fixtures, which carry the marker).
        debug_assert!(
            instruct_token.contains(&ENDOFPROMPT_ID),
            "build_lm_input_instruct: instruct_token must carry the <|endofprompt|> id \
             ({ENDOFPROMPT_ID}) that CV3 asserts on the instruct/prompt text"
        );
        let sos = self.body.speech_embed(&[SOS])?; // [1,1,H]  speech_embedding[sos]
        let task = self.body.speech_embed(&[TASK_ID])?; // [1,1,H]  speech_embedding[task_id]
        let instruct = self.body.text_embed(instruct_token)?; // [1,Ti,H]
        let text = self.body.text_embed(text_token)?; // [1,Tt,H]
        Tensor::cat(&[&sos, &instruct, &text, &task], 1)
    }

    /// Full (uncached) CV3 LM forward over a precomputed embedding sequence `[1, T, 896]`:
    /// the shared Qwen2 body (full causal attention) → bias-free `llm_decoder` →
    /// logits `[1, T, 6761]`. This is the teacher-forced / parity path — one forward over
    /// the whole sequence, no sampling.
    pub fn forward_logits(&self, embeds: &Tensor) -> Result<Tensor> {
        let h = self.body.forward_hidden(embeds)?; // [1, T, 896]
        // CV3 `llm_decoder` is bias-free (`Linear(896 -> 6761, bias=False)`).
        self.body.head_linear(&h, "llm_decoder.weight", None)
    }

    /// Teacher-forced per-position logits: given the full teacher-forced embedding sequence
    /// `embeds` `[1, T, 896]` (= step-0 input embeds with the generated speech tokens'
    /// embeddings appended), return the `n` per-step last-position logits as `[n, 6761]`,
    /// where row `k` is the logit at absolute position `t0 - 1 + k` (the model's prediction
    /// of generated token `k`). This is the deterministic, RNG-free correctness signal:
    /// it matches the incremental decode to fp rounding without invoking the sampler.
    pub fn teacher_forced_logits(&self, embeds: &Tensor, t0: usize, n: usize) -> Result<Tensor> {
        let logits = self.forward_logits(embeds)?; // [1, T, 6761]
        logits.narrow(1, t0 - 1, n)?.reshape((n, DECODER_OUT))
    }

    /// Last-position `llm_decoder` logits `[6761]` for `embeds` `[1, t_new, 896]` fed into
    /// the **KV-cached** body, advancing `cache` by `t_new`. Used by [`Cv3Lm::generate`]:
    /// numerically the cached analogue of a full recompute at the same positions.
    pub(super) fn step_logits_cached(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let h = self.body.forward_hidden_cached(embeds, cache)?; // [1, t_new, 896]
        let t = h.dim(1)?;
        let last = h.narrow(1, t - 1, 1)?; // [1, 1, 896]
        let logits = self.body.head_linear(&last, "llm_decoder.weight", None)?;
        logits.reshape((DECODER_OUT,))
    }
}
