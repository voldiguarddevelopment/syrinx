//! Chunked-causal streaming for CV3: the [`Cv3StreamCfg`] knobs, the additive DiT chunk
//! mask, and the streaming flow forward ([`Cv3Flow::forward_zero_shot_streaming`]). Moved
//! verbatim from `real_cv3.rs`.

use super::*;

/// Chunked-causal streaming configuration for [`Cv3Flow::forward_zero_shot_streaming`].
///
/// CosyVoice3 streams the **same** `CausalMaskedDiffWithDiT` weights under a chunk
/// attention mask in its DiT estimator (it does *not* swap in a different architecture):
/// a mel frame in chunk `c` may attend only to chunks `[c - num_left, c]`, never the
/// future, so finalized frames stay bit-stable as later chunks arrive. Unlike CV2 there is
/// **no conformer encoder** to mask — the CV3 front-end (`input_embedding ->
/// pre_lookahead -> repeat_interleave`) is convolution-only, so the DiT estimator is the
/// sole attention stage that needs a chunk mask. Defaults come from [`Cv3StreamCfg::cosyvoice3`].
#[derive(Clone, Copy, Debug)]
pub struct Cv3StreamCfg {
    /// DiT-estimator chunk size, in **mel frames** (the estimator runs at mel rate, which
    /// is `token_mel_ratio = 2`× the token rate).
    pub est_chunk: usize,
    /// Number of left chunks a frame may additionally attend to (besides its own chunk).
    /// `usize::MAX` ⇒ all left chunks (CosyVoice3's `num_decoding_left_chunks = -1`).
    pub num_left: usize,
}

impl Cv3StreamCfg {
    /// The CosyVoice3-0.5B defaults, read straight from the box reference
    /// (`cosyvoice/flow/DiT/dit.py`):
    ///
    /// ```text
    ///   DiT.static_chunk_size = 50            # the estimator chunk, IN MEL FRAMES
    ///   DiT.forward(streaming=True):
    ///     add_optional_chunk_mask(x, mask, ..., static_chunk_size=50, num_left=-1)
    /// ```
    ///
    /// * `est_chunk = 50` — `DiT.static_chunk_size`. The DiT runs on the **mel** sequence
    ///   (length `2*(Tp+Tg)`); CV3 sets its streaming chunk to 50 mel frames (= 25 tokens
    ///   × `token_mel_ratio`), matching the LM/flow `token_hop_len = 25`.
    /// * `num_left = usize::MAX` — at the call site the DiT passes `num_decoding_left_chunks
    ///   = -1` (all left chunks) regardless of the stored `2`, exactly as CV2's runtime does.
    pub fn cosyvoice3() -> Self {
        Self { est_chunk: 50, num_left: usize::MAX }
    }
}

impl Cv3Flow {
    /// Chunked-causal **streaming** counterpart of [`Self::forward`].
    ///
    /// Identical conditioning, weights, noise, and CFM ODE as `forward`, but the DiT
    /// estimator runs under a chunk-causal attention mask built from `cfg`: a finalized
    /// mel frame never attends to the future, so re-running on a grown token prefix leaves
    /// already-finalized frames **bit-stable** — the property that makes streaming
    /// sample-faithful (the CV3 analogue of CV2's `forward_zero_shot_streaming`; see
    /// `syrinx-acoustic/docs/STREAMING.md`). With `cfg.est_chunk` set huge this reduces to
    /// the unmasked path; with [`Cv3StreamCfg::cosyvoice3`] it matches CosyVoice3's
    /// `flow.inference(streaming=True)` / `DiT.forward(streaming=True)`.
    ///
    /// CV3 has no conformer encoder, so — unlike CV2 — the only masked stage is the DiT
    /// estimator (the `input_embedding -> pre_lookahead -> repeat_interleave` front-end is
    /// convolution-only and carries its own causal/lookahead padding). `forward` (the
    /// non-streaming batch path) is byte-unchanged.
    ///
    /// Returns the generated mel `[1, 80, 2*Tg]`, same shape/semantics as `forward`.
    pub fn forward_zero_shot_streaming(
        &self,
        prompt_token: &Tensor,
        token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
        cfg: Cv3StreamCfg,
    ) -> Result<Tensor> {
        let spk = self.spk_proj(embedding)?; // [1,80]
        let tok_cat = Tensor::cat(&[prompt_token, token], 1)?; // [1, Tp+Tg]
        let mu = self.token_to_mu(&tok_cat)?; // [1,80, 2*(Tp+Tg)]

        let total = mu.dim(2)?;
        let mel_len1 = prompt_feat.dim(1)?; // 2*Tp
        let mel_len2 = total - mel_len1; // 2*Tg

        let prompt_ct = prompt_feat.transpose(1, 2)?.contiguous()?; // [1,80,mel_len1]
        let cond_tail = Tensor::zeros((1, MEL, mel_len2), DType::F32, &self.dev)?;
        let cond = Tensor::cat(&[&prompt_ct, &cond_tail], 2)?; // [1,80,total]

        // Estimator mask: built at the mel length `total` with chunk `est_chunk`. A frame
        // in chunk `c` attends only to chunks `[0, c]` (num_left = all-left), never future.
        let m_est = add_optional_chunk_mask(total, cfg.est_chunk, cfg.num_left, &self.dev)?;

        let mel_full = self.cfm_solve_masked(&mu, &spk, &cond, z, n_timesteps, Some(&m_est))?;
        mel_full.narrow(2, mel_len1, mel_len2) // drop prompt-mel prefix
    }
}

/// Build CosyVoice3's additive chunked-causal attention mask `[1, 1, t, t]` for the DiT
/// estimator — the Rust analogue of `add_optional_chunk_mask` +
/// `subsequent_chunk_mask` (`cosyvoice/utils/mask.py`) as invoked by
/// `DiT.forward(streaming=True)`.
///
/// Position `i` lives in chunk `c = i / chunk_size`; it may attend to position `j`
/// (chunk `cj = j / chunk_size`) iff `c - num_left <= cj <= c` — its own chunk and up to
/// `num_left` chunks of left context, **never the future**. Allowed entries are `0.0`;
/// disallowed are `f32::NEG_INFINITY`, so adding this to pre-softmax scores zeros out the
/// forbidden positions. Every row always includes its own chunk (`cj = c`, which contains
/// `i`), so no row is fully masked and the softmax never sees an all-`-inf` row (no NaN) —
/// matching the reference's "force set to true" all-false guard.
///
/// `num_left == usize::MAX` (saturating) ⇒ all left chunks (CosyVoice3's
/// `num_decoding_left_chunks = -1`, which is what the DiT passes at runtime). A
/// `chunk_size` larger than `t` ⇒ a single chunk ⇒ an all-zeros mask (no masking), and
/// `chunk_size == 0` is treated as no masking too — so the non-streaming path passes `None`.
fn add_optional_chunk_mask(
    t: usize,
    chunk_size: usize,
    num_left: usize,
    dev: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    if chunk_size > 0 {
        for i in 0..t {
            let ci = i / chunk_size;
            let start_chunk = ci.saturating_sub(num_left); // num_left==MAX ⇒ 0 (all left)
            let row = i * t;
            for j in 0..t {
                let cj = j / chunk_size;
                if cj < start_chunk || cj > ci {
                    data[row + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), dev)
}
