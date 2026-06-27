//! The CV3 autoregressive speech-token generation loops (KV-cached `Qwen2LM.inference`):
//! the standard [`Cv3Lm::generate`] and the instruct-seeded [`Cv3Lm::generate_instruct`].
//!
//! Split out verbatim from the original single-file CV3 port. Both drive the
//! `super::sampling` primitives and the shared body's cached step logits; `generate` also
//! invokes the env-gated `super::gendebug` diagnostic.

use super::gendebug::gen_debug_enabled;
use super::sampling::{log_softmax_vec, ras_sampling, SplitMix64};
use super::{Cv3Lm, SPEECH_TOKEN_SIZE};
use crate::real::KvCache;
use candle_core::Result;

impl Cv3Lm {
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
}
