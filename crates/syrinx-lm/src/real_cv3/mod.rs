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
//!
//! Module layout (a pure structural split — all logic is byte-preserved from the original
//! single-file CV3 port): this `mod.rs` owns the [`Cv3Lm`] struct + the CV3 constants;
//! `lm` owns the loaders/footprint + input assembly + forward/teacher-forced/step logits;
//! `generate` owns the autoregressive loops; `gendebug` owns the `SYRINX_CV3_GEN_DEBUG`
//! diagnostic; `sampling` owns the PRNG + nucleus/ras/random samplers and the `testkit`
//! test seam (re-exported here at the original `real_cv3::testkit` path).

use crate::real::Qwen2Lm;

mod gendebug;
mod generate;
mod lm;
mod sampling;

#[doc(hidden)]
pub use sampling::testkit;

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
