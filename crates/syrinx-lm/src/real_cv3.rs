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

use crate::real::{Footprint, KvCache, Qwen2Lm};
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
/// `<|endofprompt|>` text-token id — the boundary marker CV3 asserts on the instruct /
/// prompt text segment (`Qwen2LM.inference`). Used by [`Cv3Lm::build_lm_input_instruct`]
/// as a direct-caller safety check.
const ENDOFPROMPT_ID: u32 = 151646;

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
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng).token;
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
        let t0 = lm_input0.dim(1)?;
        let debug = gen_debug_enabled();
        let mut cache = KvCache::new();
        let mut rng = SplitMix64::new(seed);
        let mut out: Vec<u32> = Vec::new();
        // Under `SYRINX_CV3_GEN_DEBUG`, retain every step's cached-decode logit vector so we
        // can replay the realized token sequence through the uncached single-forward path and
        // measure per-step cached-vs-single divergence; and the per-step repetition-aware
        // fallback flag (`true` == the nucleus pick was replaced by a `random_sampling` draw).
        // Both empty (no allocation) otherwise.
        let mut dbg_cached_logits: Vec<Vec<f32>> = Vec::new();
        let mut dbg_ras_triggers: Vec<bool> = Vec::new();
        let mut logits = self.step_logits_cached(&lm_input0, &mut cache)?;
        for i in 0..max_len {
            if debug {
                dbg_cached_logits.push(logits.to_vec1()?);
            }
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let outcome = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            if debug {
                dbg_ras_triggers.push(outcome.triggered);
            }
            let top = outcome.token;
            // CV3 stop set = the 200 control ids `speech_token_size..=6760`; every id at or
            // above `speech_token_size` ends decoding.
            if top >= SPEECH_TOKEN_SIZE {
                break;
            }
            out.push(top);
            let row = self.body.speech_embed(&[top])?; // [1,1,H]
            logits = self.step_logits_cached(&row, &mut cache)?;
        }
        if debug {
            self.gen_debug_report(&lm_input0, t0, &out, &dbg_cached_logits, &dbg_ras_triggers)?;
        }
        Ok(out)
    }

    /// Env-gated (`SYRINX_CV3_GEN_DEBUG`) diagnostic for the live AR loop — pure
    /// instrumentation, never alters the returned token sequence. It answers the two
    /// questions that separate the degeneracy suspects:
    ///
    /// **(1) Is the KV-cached incremental decode faithful?** The realized token sequence is
    /// replayed through the *uncached single-forward* path ([`Cv3Lm::teacher_forced_logits`])
    /// and, per step, the single-forward logits are compared to the logits the cached decode
    /// *actually used* to sample: per-step argmax agreement, top-5 overlap, and
    /// `max|Δlogit|`. The CV2 body these reuse is proven bit-identical (multi-token) /
    /// argmax-exact (single-token) in `tests/real_lm_kvcache.rs`, so the expectation is
    /// `max|Δlogit| <= ~5e-5` with **zero argmax flips**. Flips that appear early and *grow*
    /// with step ⇒ cached decode drift (suspect 1: RoPE/mask/append offset). All-agree ⇒
    /// suspect 1 is refuted on this run and the cause is sampling/logit-shape, not the cache.
    ///
    /// **(2) What shape is the degeneracy?** The id histogram, unique count (vs the
    /// reference's ~80/102), and longest consecutive run distinguish a *collapse* (one id or
    /// a short cycle repeating ⇒ sampler / repetition-aware bias, suspect 3) from
    /// *varied-but-wrong* output (⇒ logit content).
    ///
    /// **(3) How often does repetition-aware sampling (RAS) fall back to `random_sampling`?**
    /// `ras_triggers[i]` is `true` when step `i`'s nucleus pick was discarded for a
    /// full-distribution draw. Reports the trigger RATE and tags whether the **stopping**
    /// step (if any) was a RAS fallback draw vs a plain nucleus draw — a high rate and/or a
    /// RAS-fallback stop right after `min_len` is the signature of the RAS divergence.
    fn gen_debug_report(
        &self,
        lm_input0: &Tensor,
        t0: usize,
        out: &[u32],
        cached_logits: &[Vec<f32>],
        ras_triggers: &[bool],
    ) -> Result<()> {
        let n = out.len();
        eprintln!("== SYRINX_CV3_GEN_DEBUG ==  T0={t0}  generated n={n}");

        // (3) Repetition-aware-sampling fallback rate + stop-step path. `ras_triggers` has one
        // entry per sampling step (n produced tokens, plus one more if a control id stopped
        // the loop). A high rate + a RAS-fallback stop just past `min_len` is the RAS-divergence
        // signature; the reference's RAS fires rarely.
        let total_steps = ras_triggers.len();
        let ras_count = ras_triggers.iter().filter(|&&t| t).count();
        let stopped = total_steps > n; // the loop broke on a control id at step `n`
        let rate = if total_steps > 0 {
            100.0 * ras_count as f64 / total_steps as f64
        } else {
            0.0
        };
        eprintln!(
            "RAS fallback: {ras_count}/{total_steps} steps ({rate:.1}%) drew from random_sampling \
             (reference fires rarely)"
        );
        if stopped {
            let via = if ras_triggers[n] { "RAS random_sampling fallback" } else { "plain nucleus draw" };
            eprintln!("STOP at step {n} (control id) came from: {via}");
        } else {
            eprintln!("no stop: ran to max_len ({total_steps} steps)");
        }
        // Per-step trigger flags ('R' = fallback fired, '.' = plain nucleus), aligned with ids.
        let flags: String = ras_triggers.iter().map(|&t| if t { 'R' } else { '.' }).collect();
        eprintln!("RAS per-step flags = {flags}");

        if n == 0 {
            eprintln!(
                "DEGENERACY: ZERO tokens generated — the step-0 sample was already a control \
                 id (>= {SPEECH_TOKEN_SIZE}). Check the `min_len` EOS/control mask (suspect 4) \
                 and the step-0 logits."
            );
            return Ok(());
        }

        // (2) Token-sequence shape: unique count, frequency histogram, longest run.
        let unique: std::collections::BTreeSet<u32> = out.iter().copied().collect();
        let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for &t in out {
            *counts.entry(t).or_default() += 1;
        }
        let mut hist: Vec<(u32, usize)> = counts.into_iter().collect();
        hist.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut longest_run = 1usize;
        let mut cur = 1usize;
        for w in out.windows(2) {
            if w[0] == w[1] {
                cur += 1;
                longest_run = longest_run.max(cur);
            } else {
                cur = 1;
            }
        }
        eprintln!("ids = {out:?}");
        eprintln!(
            "unique={}/{n}  longest_run={longest_run}  (reference seed-0: ~80 unique / 102 total)",
            unique.len()
        );
        eprintln!("top-8 ids by frequency (id, count): {:?}", &hist[..hist.len().min(8)]);

        // (1) Replay the realized sequence through the uncached single-forward path and
        // compare per step to the cached logits actually used.
        let embeds = if n > 1 {
            let tail = self.body.speech_embed(&out[..n - 1])?; // [1, n-1, H]
            Tensor::cat(&[lm_input0, &tail], 1)?
        } else {
            lm_input0.clone()
        };
        let single = self.teacher_forced_logits(&embeds, t0, n)?; // [n, DECODER_OUT]
        let mut max_abs = 0f32;
        let mut max_abs_step = 0usize;
        let mut argmax_flips = 0usize;
        let mut first_flip: Option<usize> = None;
        eprintln!(
            "step :  cached_argmax  single_argmax   chosen   max|Δlogit|  top5_overlap"
        );
        for k in 0..n {
            let cl = &cached_logits[k];
            let sl: Vec<f32> = single.narrow(0, k, 1)?.reshape((DECODER_OUT,))?.to_vec1()?;
            let ca = argmax_f32(cl);
            let sa = argmax_f32(&sl);
            let d = cl
                .iter()
                .zip(sl.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if d > max_abs {
                max_abs = d;
                max_abs_step = k;
            }
            if ca != sa {
                argmax_flips += 1;
                first_flip.get_or_insert(k);
            }
            let overlap = topk_overlap(cl, &sl, 5);
            // First 8 and last 8 steps in detail (enough to see whether divergence grows).
            if k < 8 || k + 8 >= n {
                eprintln!(
                    "{k:>4} :  {ca:>12}  {sa:>12}  {:>7}   {d:>10.3e}   {overlap}/5",
                    out[k]
                );
            }
        }
        eprintln!(
            "cached-vs-single-forward:  max|Δlogit|={max_abs:.3e} (step {max_abs_step})  \
             argmax_flips={argmax_flips}/{n}  first_flip={first_flip:?}"
        );
        eprintln!("INTERPRET:");
        eprintln!(
            "  * max|Δlogit| <= ~5e-5 AND argmax_flips==0  => KV-cache decode FAITHFUL \
             (suspect 1 REFUTED on this run)."
        );
        eprintln!(
            "      then read the histogram above: collapse to a few ids / large longest_run \
             => sampler or repetition-aware bias (suspect 3);"
        );
        eprintln!(
            "      varied ids but low SIM-o => the logit CONTENT is wrong under free-run \
             conditioning, not the decode."
        );
        eprintln!(
            "  * argmax_flips small-and-EARLY then GROWING (first_flip near 0) => KV-cache \
             DRIFT (suspect 1: rotary offset / causal-mask offset / K-V append)."
        );
        Ok(())
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

/// Whether the env-gated live-AR generation diagnostic is on. Enabled by setting
/// `SYRINX_CV3_GEN_DEBUG` to any non-empty value other than `0`. Read once per `generate`.
fn gen_debug_enabled() -> bool {
    std::env::var("SYRINX_CV3_GEN_DEBUG")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Index of the maximum element of a logit vector (first on ties) — the argmax used by the
/// `SYRINX_CV3_GEN_DEBUG` cached-vs-single comparison.
fn argmax_f32(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// The `k` highest-scoring indices of `v` (descending by value, index tiebreak).
fn topk_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| {
        v[b]
            .partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// How many of the top-`k` indices `a` and `b` share — a coarse "do they agree on the
/// decision surface" signal for the `SYRINX_CV3_GEN_DEBUG` per-step comparison.
fn topk_overlap(a: &[f32], b: &[f32], k: usize) -> usize {
    let sa: std::collections::BTreeSet<usize> = topk_indices(a, k).into_iter().collect();
    topk_indices(b, k).into_iter().filter(|i| sa.contains(i)).count()
}

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

/// The result of one [`ras_sampling`] draw: the chosen `token`, plus whether the
/// repetition-aware guard `triggered` (nucleus pick discarded → `random_sampling` fallback).
/// `triggered` is read only by the `SYRINX_CV3_GEN_DEBUG` instrumentation; it never changes
/// the returned token, so the live path is byte-identical with the diagnostic off.
struct RasOutcome {
    token: u32,
    triggered: bool,
}

/// `ras_sampling` (Repetition-Aware Sampling), an exact port of
/// `cosyvoice/utils/common.py:ras_sampling` (:138): nucleus-sample a candidate; if it
/// repeated `>= win_size * tau_r` (= 1.0) times in the last `win_size` decoded tokens, fall
/// back to `random_sampling`. The control range (`speech_token_size..DECODER_OUT`) is
/// `-inf`-masked first when `ignore_eos` (CV3's wider-than-CV2 stop set; the masking analogue
/// of the reference's eos rejection loop while `step < min_len`). Pinned
/// `top_p=0.8, top_k=25, win_size=10, tau_r=0.1` (the dump metadata).
fn ras_sampling(logp: &[f32], decoded: &[u32], ignore_eos: bool, rng: &mut SplitMix64) -> RasOutcome {
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
    // `decoded[-win_size:] == top` count, threshold `>= win_size * tau_r` (= 1.0): a single
    // repeat in the last `WIN` decoded tokens trips the guard. Exact match to the reference.
    let start = decoded.len().saturating_sub(WIN);
    let rep = decoded[start..].iter().filter(|&&t| t == top).count();
    if (rep as f32) >= WIN as f32 * TAU_R {
        // BUGFIX (CV3 live-AR degeneracy): the reference `random_sampling`
        // (cosyvoice/utils/common.py:165) is the PLAIN full distribution —
        // `weighted_scores.softmax(dim=0).multinomial(1)` — and crucially does **NOT** mask
        // the repeated pick `top`. The previous code set `logp[top] = -inf` before the
        // fallback, FORCING a different (off-distribution) token on every repeat; speech
        // tokens naturally repeat within a 10-step window, so this fired often and pushed the
        // free-run trajectory off-track and into early stops. Sample the same (only
        // `min_len`-masked while `ignore_eos`) distribution the reference does, WITHOUT
        // removing `top`.
        return RasOutcome { token: random_sampling(&logp, rng), triggered: true };
    }
    RasOutcome { token: top, triggered: false }
}

/// Test-only seam exposing the **production** sampler primitives so the root integration
/// test can validate them without copying the algorithm. Hidden from the public API/docs;
/// the live forward path never touches it. Used by `tests/real_cv3_multinomial.rs` to prove
/// [`multinomial1`] is an unbiased inverse-CDF draw (suspect 3) — a biased draw would skew
/// toward high-probability ids and could explain a collapsed token sequence.
#[doc(hidden)]
pub mod testkit {
    use super::{multinomial1, SplitMix64};

    /// Draw `n` indices from `probs` (need not be normalised) using the exact production
    /// [`super::multinomial1`] + [`super::SplitMix64`] PRNG seeded by `seed`.
    pub fn multinomial1_draws(probs: &[f32], seed: u64, n: usize) -> Vec<usize> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| multinomial1(probs, &mut rng)).collect()
    }

    /// `n` raw uniforms in `[0,1)` from the production [`super::SplitMix64`] — lets the test
    /// confirm the PRNG feeding `multinomial1` is itself uniform (no low-bit clumping).
    pub fn uniform_draws(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| rng.next_f64()).collect()
    }

    /// `n` draws from the production [`super::ras_sampling`], each as `(token, triggered)`
    /// where `triggered` is `true` when the repetition-aware guard fell back to
    /// `random_sampling`. Lets `tests/real_cv3_ras.rs` prove the fallback samples the PLAIN
    /// full distribution and — the regression lock — that it does NOT mask the repeated pick.
    pub fn ras_draws(
        logp: &[f32],
        decoded: &[u32],
        ignore_eos: bool,
        seed: u64,
        n: usize,
    ) -> Vec<(u32, bool)> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                let o = super::ras_sampling(logp, decoded, ignore_eos, &mut rng);
                (o.token, o.triggered)
            })
            .collect()
    }
}
