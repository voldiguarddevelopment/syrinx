//! Real **CosyVoice3** LM forward via Candle — the first CV3 component port, and the
//! anchor of the CV3 module structure in `syrinx-lm`.
//!
//! CV3's speech LM is a Qwen2-0.5B backbone with a speech-token output head, exactly the
//! same backbone shape as CosyVoice2: 24 decoder layers, hidden 896, GQA 14 query / 2 KV
//! heads (head_dim 64, q/k/v carry bias, o_proj does not), SwiGLU MLP (intermediate
//! 4864), RoPE θ=1e6, RMSNorm eps 1e-6, and **sliding-window attention disabled**
//! (`use_sliding_window=false` in the checkpoint config, so it is plain full causal
//! attention — identical to CV2's port, which also never windows). This port therefore
//! **reuses [`Qwen2Lm`] unchanged as the body** ([`Qwen2Lm::forward_hidden`] /
//! `forward_hidden_cached`, `text_embed`, `speech_embed`, `head_linear`, `KvCache`) and
//! adds only the CV3-specific pieces around it.
//!
//! What is CV3-specific (the delta from CV2):
//!   * **`sos` / `task_id` embeddings come from `speech_embedding.weight`** — rows `[sos]`
//!     and `[task_id]` of the *speech* table, not a separate `llm_embedding` (CV3 has no
//!     `llm_embedding`; CV2 did). This is the key architectural difference.
//!   * **`llm_decoder` is `Linear(896 → 6761, bias=False)`** — bias-free, width 6761 (this
//!     checkpoint extends the speech vocab with 200 control ids: `speech_token_size`(6561)
//!     .. 6760), vs CV2's biased `Linear(896 → 6564)`.
//!   * **Stop set = `6561..=6760`** (the 200 control ids), vs CV2's three.
//!
//! Constants below are the verified values from the reference dump metadata
//! (`/root/parity-cv3/lm/ref.safetensors`): `sos=6561`, `task_id=6563`,
//! `speech_token_size=6561`, `fill=6564`, `decoder_out=6761`.
//!
//! Gated behind the `real` cargo feature; the parity test (`tests/real_cv3_lm_parity.rs`)
//! skips cleanly when the on-box weights/fixture are absent.

use crate::real::{KvCache, Qwen2Lm};
use candle_core::{Device, Result, Tensor};

/// `sos` row index into `speech_embedding` (CV3: the start-of-sequence embedding is a
/// speech-table row, not a separate `llm_embedding` row).
const SOS: u32 = 6561;
/// `task_id` row index into `speech_embedding`.
const TASK_ID: u32 = 6563;
/// `speech_token_size` — also the `eos` index masked while `step < min_len`, and the low
/// bound of the stop set (every id `>= SPEECH_TOKEN_SIZE` is a decode-stop / control id).
const SPEECH_TOKEN_SIZE: u32 = 6561;
/// `llm_decoder` output width (`decoder_out`). The full speech+control logit vector.
pub const DECODER_OUT: usize = 6761;

/// The real CosyVoice3 LM: the shared Qwen2-0.5B body plus the CV3 embedding assembly and
/// bias-free `llm_decoder` head.
pub struct Cv3Lm {
    /// The Qwen2-0.5B backbone + weight store, reused verbatim from the CV2 port. Holds
    /// `embed_tokens`, `speech_embedding`, the 24 layers, the final norm, and the CV3
    /// `llm_decoder.weight` (applied here via [`Qwen2Lm::head_linear`], bias-free).
    body: Qwen2Lm,
}

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
        let sos = self.body.speech_embed(&[SOS])?; // [1,1,H]  speech_embedding[sos]
        let task = self.body.speech_embed(&[TASK_ID])?; // [1,1,H]  speech_embedding[task_id]
        let instruct = self.body.text_embed(instruct_token)?; // [1,Ti,H]
        let text = self.body.text_embed(text_token)?; // [1,Tt,H]
        Tensor::cat(&[&sos, &instruct, &text, &task], 1)
    }

    /// Autoregressively generate CV3 speech tokens for the **instruct / emotion** path:
    /// the [`Cv3Lm::generate`] AR loop seeded by [`Cv3Lm::build_lm_input_instruct`]
    /// (empty prompt-speech prefix) instead of [`Cv3Lm::build_lm_input`]. Same KV-cached
    /// `ras_sampling` decode + 200-control-id stop set + `min_len` EOS mask. Additive —
    /// [`Cv3Lm::generate`] is byte-unchanged.
    pub fn generate_instruct(
        &self,
        instruct_token: &[u32],
        text_token: &[u32],
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let lm_input0 = self.build_lm_input_instruct(instruct_token, text_token)?;
        let mut cache = KvCache::new();
        let mut rng = SplitMix64::new(seed);
        let mut out: Vec<u32> = Vec::new();
        let mut logits = self.step_logits_cached(&lm_input0, &mut cache)?;
        for i in 0..max_len {
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            if top >= SPEECH_TOKEN_SIZE {
                break;
            }
            out.push(top);
            let row = self.body.speech_embed(&[top])?; // [1,1,H]
            logits = self.step_logits_cached(&row, &mut cache)?;
        }
        Ok(out)
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
    fn step_logits_cached(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let h = self.body.forward_hidden_cached(embeds, cache)?; // [1, t_new, 896]
        let t = h.dim(1)?;
        let last = h.narrow(1, t - 1, 1)?; // [1, 1, 896]
        let logits = self.body.head_linear(&last, "llm_decoder.weight", None)?;
        logits.reshape((DECODER_OUT,))
    }

    /// Autoregressively generate CV3 speech tokens, mirroring `Qwen2LM.inference`, using a
    /// **KV cache** (each step O(n)). Prefill `build_lm_input` once to seed the cache and
    /// the step-0 logits, then per step: `log_softmax` → `ras_sampling` (pinned
    /// `top_p=0.8, top_k=25, win_size=10, tau_r=0.1`, seed-pinned multinomial draws) → stop
    /// when the chosen id is a control id (`>= speech_token_size`), else append its
    /// `speech_embedding` row and feed only that token through the cached body. EOS
    /// (index `speech_token_size`) is masked while `step < min_len`. Returns the generated
    /// ids (stop token excluded), matching the reference `gen_tokens` under the same seed.
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
        let mut logits = self.step_logits_cached(&lm_input0, &mut cache)?;
        for i in 0..max_len {
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            // CV3 stop set = the 200 control ids `speech_token_size..=6760`; every id at or
            // above `speech_token_size` ends decoding.
            if top >= SPEECH_TOKEN_SIZE {
                break;
            }
            out.push(top);
            let row = self.body.speech_embed(&[top])?; // [1,1,H]
            logits = self.step_logits_cached(&row, &mut cache)?;
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Deterministic sampler — a focused mirror of the CV2 (`real.rs`) sampler, kept
// local so the CV2 module stays byte-for-byte unchanged (additive-only port).
// The PRNG + nucleus + repetition-aware logic are the same pinned algorithm; the
// only CV3 specialisation lives in `Cv3Lm::generate`'s stop check (200 control
// ids) — `ras_sampling`'s EOS index (`speech_token_size` = 6561) is identical to
// CV2's, so the sampler body matches the reference `ras_sampling` exactly.
// -----------------------------------------------------------------------------

/// `log_softmax` over a 1-D logit vector `[V]`, returned as a host `Vec<f32>` (f64 accum).
fn log_softmax_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let v: Vec<f32> = logits.to_vec1()?;
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &x in &v {
        sum += ((x - m) as f64).exp();
    }
    let lse = m as f64 + sum.ln();
    Ok(v.iter().map(|&x| (x as f64 - lse) as f32).collect())
}

/// Deterministic SplitMix64 PRNG — pins the otherwise-stochastic multinomial draws so a
/// `generate` run is bit-reproducible from a seed. `next_f64` yields a uniform in `[0,1)`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Inverse-CDF sample one index from `probs` (need not be normalised) on a single uniform
/// draw — the deterministic analogue of `torch.multinomial(probs, 1)`.
fn multinomial1(probs: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f64 = probs.iter().map(|&p| p as f64).sum();
    let u = rng.next_f64() * total;
    let mut acc = 0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p as f64;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// `nucleus_sampling`: take the leading tokens (stable descending by probability) while
/// `cum_prob < top_p` AND `count < top_k`, then multinomial-sample one. `logp` is a
/// log-probability vector; probabilities are `exp(logp)`.
fn nucleus_sampling(logp: &[f32], top_p: f32, top_k: usize, rng: &mut SplitMix64) -> u32 {
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut cum = 0f32;
    let mut cand_idx: Vec<u32> = Vec::new();
    let mut cand_prob: Vec<f32> = Vec::new();
    for &i in &order {
        if cum < top_p && cand_prob.len() < top_k {
            cum += probs[i];
            cand_prob.push(probs[i]);
            cand_idx.push(i as u32);
        } else {
            break;
        }
    }
    let pick = multinomial1(&cand_prob, rng);
    cand_idx[pick]
}

/// `random_sampling`: full-softmax multinomial over the whole vocab (the `ras` fallback).
fn random_sampling(logp: &[f32], rng: &mut SplitMix64) -> u32 {
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    multinomial1(&probs, rng) as u32
}

/// `ras_sampling` (Repetition-Aware Sampling): nucleus-sample a candidate; if it repeated
/// `>= win_size * tau_r` times in the last `win_size` decoded tokens, mask it and fall back
/// to `random_sampling`. EOS (`speech_token_size`) is `-inf`-masked first when `ignore_eos`.
/// Pinned `top_p=0.8, top_k=25, win_size=10, tau_r=0.1` (matches the dump metadata).
fn ras_sampling(logp: &[f32], decoded: &[u32], ignore_eos: bool, rng: &mut SplitMix64) -> u32 {
    const TOP_P: f32 = 0.8;
    const TOP_K: usize = 25;
    const WIN: usize = 10;
    const TAU_R: f32 = 0.1;
    let mut logp = logp.to_vec();
    if ignore_eos {
        // CV3's decode-stop is the WHOLE control range `SPEECH_TOKEN_SIZE..DECODER_OUT`
        // (6561..=6760), not just the single EOS. While `step < min_len`, NONE of those
        // control ids may be chosen — otherwise an adjacent control id (e.g. 6562) gets
        // sampled and trips `Cv3Lm::generate`'s `top >= SPEECH_TOKEN_SIZE` stop *inside*
        // the min_len window, ending decoding at step 0 (the live-loop "0 tokens" bug).
        // CV2 masked only EOS because its stop set was 3 ids it happened never to hit
        // early; CV3's wider stop set needs the wider mask to actually enforce min_len.
        for s in logp
            .iter_mut()
            .take(DECODER_OUT)
            .skip(SPEECH_TOKEN_SIZE as usize)
        {
            *s = f32::NEG_INFINITY;
        }
    }
    let top = nucleus_sampling(&logp, TOP_P, TOP_K, rng);
    let start = decoded.len().saturating_sub(WIN);
    let rep = decoded[start..].iter().filter(|&&t| t == top).count();
    if (rep as f32) >= WIN as f32 * TAU_R {
        logp[top as usize] = f32::NEG_INFINITY;
        return random_sampling(&logp, rng);
    }
    top
}
