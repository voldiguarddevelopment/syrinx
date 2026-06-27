//! Embedding lookups, step-0 prompt assembly, per-step logits and the autoregressive
//! speech-token generation loops (CosyVoice2 `Qwen2LM.inference`).
//!
//! Split out verbatim from the original single-file `real` port. All methods here are
//! inherent on [`super::Qwen2Lm`]; the embedding gather (`embed_rows`) reaches into the
//! shared [`super::QEmbed`] store, and `generate`/`generate_full_recompute` drive the
//! `super::sampling` primitives.

use super::sampling::{log_softmax_vec, ras_sampling, SplitMix64};
use super::{EmbedScheme, KvCache, Qwen2Lm, SOS, SPEECH_VOCAB, STOP_TOKENS, TASK_ID};
use candle_core::{DType, Result, Tensor};

impl Qwen2Lm {
    /// Gather rows `ids` from a `[V, HIDDEN]` embedding table, returning `[1, n, HIDDEN]`.
    ///
    /// `ids` are u32 token ids; this is a plain row lookup (the `nn.Embedding` op).
    ///
    /// In the int8 quantized build the table lives in `qembed` as per-row int8: we
    /// `index_select` only the needed rows of the u8 store + their per-row scales, then
    /// **dequantize just those rows** to f32 (`(q-128)*scale`) — the full f32 table is
    /// never reconstructed. In the fp32 build the table is a plain f32 tensor in `w`.
    fn embed_rows(&self, table: &str, ids: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.dev)?;
        if let Some(qe) = self.qembed.get(table) {
            match qe.scheme {
                EmbedScheme::Int8 => {
                    let q = qe.q.index_select(&idx, 0)?.to_dtype(DType::F32)?; // [n, HIDDEN]
                    let s = qe.scale.index_select(&idx, 0)?; // [n, 1]
                    let rows = (q - 128.0)?.broadcast_mul(&s)?; // [n, HIDDEN], dequantized
                    return rows.unsqueeze(0); // [1, n, HIDDEN]
                }
                EmbedScheme::Int4 => {
                    // Gather the packed nibble-rows + per-row scales, then unpack +
                    // dequantize on the host (only `n` rows, so this is cheap): each byte
                    // holds two weights — low nibble is element 2j, high nibble 2j+1.
                    let n = ids.len();
                    let h = qe.h;
                    let hp = h / 2;
                    let packed: Vec<u8> = qe
                        .q
                        .index_select(&idx, 0)?
                        .flatten_all()?
                        .to_vec1()?; // [n*hp]
                    let scales: Vec<f32> = qe
                        .scale
                        .index_select(&idx, 0)?
                        .flatten_all()?
                        .to_vec1()?; // [n]
                    let mut out = vec![0f32; n * h];
                    for r in 0..n {
                        let s = scales[r];
                        for j in 0..hp {
                            let byte = packed[r * hp + j];
                            let lo = (byte & 0x0F) as i32 - 8; // element 2j
                            let hi = (byte >> 4) as i32 - 8; // element 2j+1
                            out[r * h + 2 * j] = lo as f32 * s;
                            out[r * h + 2 * j + 1] = hi as f32 * s;
                        }
                    }
                    let rows = Tensor::from_vec(out, (n, h), &self.dev)?;
                    return rows.unsqueeze(0); // [1, n, HIDDEN]
                }
            }
        }
        let w = self.g(table)?; // [V, HIDDEN]
        let rows = w.index_select(&idx, 0)?; // [n, HIDDEN]
        let rows = rows.to_dtype(DType::F32)?;
        rows.unsqueeze(0) // [1, n, HIDDEN]
    }

    /// The Qwen2 text embedding for `text_token` ids (`embed_tokens`), `[1, n, HIDDEN]`.
    pub fn text_embed(&self, text_token: &[u32]) -> Result<Tensor> {
        self.embed_rows("llm.model.model.embed_tokens.weight", text_token)
    }

    /// One `llm_embedding` row (`sos`=0 / `task_id`=1), shaped `[1, 1, HIDDEN]`.
    fn llm_embed_row(&self, id: u32) -> Result<Tensor> {
        self.embed_rows("llm_embedding.weight", &[id])
    }

    /// `speech_embedding` rows for the given speech-token ids, `[1, n, HIDDEN]`.
    pub fn speech_embed(&self, speech_token: &[u32]) -> Result<Tensor> {
        self.embed_rows("speech_embedding.weight", speech_token)
    }

    /// Assemble the step-0 LM input exactly as `Qwen2LM.inference`:
    /// `[sos_emb, text_emb(text_token), task_id_emb, prompt_speech_emb]` -> `[1, T0, HIDDEN]`.
    ///
    /// `text_token` here is already the concatenation of `prompt_text` and `text`
    /// (the reference concatenates them before embedding). `prompt_speech_token` may be
    /// empty, in which case the prompt-speech segment is omitted.
    pub fn build_lm_input(&self, text_token: &[u32], prompt_speech_token: &[u32]) -> Result<Tensor> {
        let sos = self.llm_embed_row(SOS)?; // [1,1,H]
        let task = self.llm_embed_row(TASK_ID)?; // [1,1,H]
        let text = self.text_embed(text_token)?; // [1,Tt,H]
        let mut parts: Vec<Tensor> = vec![sos, text, task];
        if !prompt_speech_token.is_empty() {
            parts.push(self.speech_embed(prompt_speech_token)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Last-position raw `llm_decoder` logits for an input embedding sequence
    /// `[1, T, HIDDEN]`, returning `[V]` (`V = SPEECH_VOCAB = 6564`).
    ///
    /// Recomputes the full transformer each call (O(n²) but logit-identical to a KV
    /// cache — same positions, same causal mask), which is what we want for parity.
    pub fn step_logits(&self, embeds: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(embeds)?; // [1, T, V]
        let t = logits.dim(1)?;
        logits.narrow(1, t - 1, 1)?.reshape((SPEECH_VOCAB,))
    }

    /// Last-position raw `llm_decoder` logits for `embeds` `[1, t_new, HIDDEN]` fed into
    /// the **cached** path, returning `[V]`. Advances `cache` by `t_new`. With the cache
    /// at length `L`, this is the logit-identical O(t_new) analogue of `step_logits` over
    /// an `L+t_new` recompute (same positions, same causal visibility).
    pub fn step_logits_cached(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let logits = self.forward_logits_cached(embeds, cache)?; // [1, t_new, V]
        let t = logits.dim(1)?;
        logits.narrow(1, t - 1, 1)?.reshape((SPEECH_VOCAB,))
    }

    /// Autoregressively generate speech tokens, mirroring `Qwen2LM.inference`, using a
    /// **KV cache** so each step is O(n) instead of O(n²).
    ///
    /// Prefill: assemble `build_lm_input` and run it once through the cached forward,
    /// populating every layer's K/V and yielding the step-0 last-position logits. Then
    /// per step: `log_softmax` -> `ras_sampling` (with `seed`-pinned multinomial draws)
    /// -> stop if the chosen id is a stop token, else append its `speech_embedding` row
    /// and feed **only that one token** through the cached forward (cache grows by 1).
    /// EOS is masked while `step < min_len`. Returns the generated token ids (stop token
    /// excluded), matching the reference's `out_tokens`. Because the cache carries the
    /// full history, generation may run to the real `max_len` with no practical cap.
    pub fn generate(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let lm_input0 = self.build_lm_input(text_token, prompt_speech_token)?;
        let mut cache = KvCache::new();
        let mut rng = SplitMix64::new(seed);
        let mut out: Vec<u32> = Vec::new();
        // Prefill the assembled prompt once; `logits` is the step-0 last-position logit.
        let mut logits = self.step_logits_cached(&lm_input0, &mut cache)?;
        for i in 0..max_len {
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            if STOP_TOKENS.contains(&top) {
                break;
            }
            out.push(top);
            // Feed only the newly sampled token; the cache supplies all prior context.
            let row = self.speech_embed(&[top])?; // [1,1,H]
            logits = self.step_logits_cached(&row, &mut cache)?;
        }
        Ok(out)
    }

    /// Reference O(n²) full-recompute generation — the pre-cache algorithm, kept as the
    /// correctness oracle for the cached `generate`. Identical sampling, stop conditions,
    /// pinned PRNG and `min_len` EOS masking; the *only* difference from `generate` is
    /// that each step re-runs the whole sequence (`step_logits`) instead of using a cache.
    /// A fixed seed must yield the exact same token vector as `generate`.
    pub fn generate_full_recompute(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let mut embeds = self.build_lm_input(text_token, prompt_speech_token)?;
        let mut rng = SplitMix64::new(seed);
        let mut out: Vec<u32> = Vec::new();
        for i in 0..max_len {
            let logits = self.step_logits(&embeds)?;
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            if STOP_TOKENS.contains(&top) {
                break;
            }
            out.push(top);
            let row = self.speech_embed(&[top])?; // [1,1,H]
            embeds = Tensor::cat(&[&embeds, &row], 1)?;
        }
        Ok(out)
    }

    /// Teacher-forced per-step logits: given the reference's chosen token sequence,
    /// rebuild the full embedding sequence and return every step's last-position logits
    /// as `[N, V]` (step `k`'s logits at row `k`). This proves the AR forward reproduces
    /// the reference logit-for-logit independent of the (stochastic) sampler, and is the
    /// real correctness signal for the generation loop.
    pub fn teacher_forced_logits(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        gen_tokens: &[u32],
    ) -> Result<Tensor> {
        let lm_input0 = self.build_lm_input(text_token, prompt_speech_token)?;
        let t0 = lm_input0.dim(1)?;
        // Append speech embeddings for all but the last generated token: step k's
        // last-position logit lives at absolute position (t0 - 1 + k).
        let embeds = if gen_tokens.len() > 1 {
            let tail = self.speech_embed(&gen_tokens[..gen_tokens.len() - 1])?;
            Tensor::cat(&[&lm_input0, &tail], 1)?
        } else {
            lm_input0
        };
        let logits = self.forward_logits(&embeds)?; // [1, T, V]
        let n = gen_tokens.len();
        // rows [t0-1 .. t0-1+n) are the n per-step last-position logits.
        logits.narrow(1, t0 - 1, n)?.reshape((n, SPEECH_VOCAB))
    }
}
