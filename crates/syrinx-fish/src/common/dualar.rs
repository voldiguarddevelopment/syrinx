//! The dual-AR contract + driver.
//!
//! [`DualArBackend`] is the trait the s1 and s2 backends implement — it abstracts over
//! everything that differs between the variants (Llama vs Qwen3 slow backbone,
//! per-codebook tables vs shared-embedding+RoPE fast head, modded-DAC vs causal-DAC) so
//! the autoregressive driver [`drive`] is written **once**.
//!
//! The loop mirrors the reference `generate` / `decode_one_token_ar`:
//!
//! 1. **prefill** the encoded prompt `[1 + num_codebooks, T_prompt]` into the slow KV
//!    cache, getting the slow step for the last prompt position.
//! 2. Repeatedly: sample the **semantic** token from the slow logits (constrained +
//!    RAS); stop on the `<|im_end|>` id; otherwise derive codebook-0 from the semantic
//!    token ([`DualArBackend::first_code`]) and run the **fast** AR to expand the frame
//!    to all `num_codebooks` residual codes ([`DualArBackend::fast_expand`]); append the
//!    frame; feed the full `[1 + num_codebooks]` frame back into the slow AR.
//! 3. Return the `[num_codebooks, T]` code matrix (ready for [`super::codec::RvqCodec`]).
//!
//! ## Notes for the backend authors (s1 / s2 waves)
//!
//! * **`prefill` and `slow_step` are `&mut self`** because they mutate the slow KV cache;
//!   **`fast_expand` is `&self`** — the fast AR's KV cache is tiny (one slot per codebook)
//!   and MUST be rebuilt per frame, so allocate it locally inside `fast_expand`. Do not
//!   stash fast-AR state on `self`.
//! * **The driver owns sampling policy.** `slow_step` / `prefill` return raw
//!   `semantic_logits` over the **full slow vocab** (the driver applies the semantic
//!   constraint + RAS). `fast_expand` is handed the same [`super::sampling::Sampler`] and
//!   must call [`super::sampling::Sampler::sample_codebook`] for each residual draw so the
//!   PRNG stream stays shared and deterministic — do not sample with your own RNG.
//! * **`hidden` is whatever the backend feeds its own fast head** (already projected via
//!   `fast_project_in` if the checkpoint has it). The driver never inspects it.
//! * **`fast_expand` returns the full frame** `Vec<u32>` of length `num_codebooks`, with
//!   index 0 == `first_code` (the deterministic semantic-derived codebook-0) and indices
//!   `1..num_codebooks` the 9 fast-AR residuals (the reference `range(1, num_codebooks)`
//!   loop). The driver appends it verbatim as one column of the `[num_codebooks, T]` matrix.

use candle_core::{Device, Result, Tensor};

use super::config::FishConfig;
use super::sampling::{Sampler, SamplingParams, SemanticConstraint};

/// One slow-AR step's outputs: the raw semantic logits (driver samples them) and the
/// hidden state the fast AR expands.
pub struct SlowStep {
    /// Raw slow-AR logits over the **full slow vocab**, shape `[vocab_size]`, f32. The
    /// driver applies the semantic constraint + RAS; the backend must NOT pre-mask.
    pub semantic_logits: Tensor,
    /// The per-frame hidden state fed into [`DualArBackend::fast_expand`] (backend-owned
    /// shape, already `fast_project_in`-projected if applicable).
    pub hidden: Tensor,
}

/// One **batched** slow-AR step's outputs across N samples: raw semantic logits `[N,
/// vocab]` (the driver samples each row with that sample's own PRNG) and the per-sample
/// fast-head hidden `[N, 1, fast_dim]`.
pub struct SlowStepBatch {
    /// Raw slow-AR logits over the full slow vocab for every sample, `[N, vocab_size]`, f32.
    pub logits: Tensor,
    /// The per-frame hidden state per sample, `[N, 1, fast_dim]`, fed to
    /// [`DualArBackend::fast_expand_batch`].
    pub hidden: Tensor,
}

/// The contract the s1 / s2 backends implement so the shared [`drive`] loop can run them.
pub trait DualArBackend {
    /// The resolved config for this backend (drives the driver's geometry + constraint).
    fn config(&self) -> &FishConfig;

    /// The device the code matrix is assembled on (typically the backbone's device).
    fn device(&self) -> Device;

    /// Allocate / clear the slow KV cache for a fresh utterance up to `max_seq_len`.
    fn reset(&mut self, max_seq_len: usize) -> Result<()>;

    /// Prefill the encoded prompt `[1 + num_codebooks, T_prompt]` into the slow KV cache
    /// and return the slow step for the **last** prompt position (so the driver can sample
    /// the first frame's semantic token). Row 0 is the slow-vocab token id; rows
    /// `1..=num_codebooks` are the prompt's RVQ codes (0 where a position has no audio).
    fn prefill(&mut self, prompt: &Tensor) -> Result<SlowStep>;

    /// One slow-AR step on the previous **full** frame `[1 + num_codebooks]` at absolute
    /// position `pos`, advancing the slow KV cache by one.
    fn slow_step(&mut self, frame: &[u32], pos: usize) -> Result<SlowStep>;

    /// Map a sampled semantic token id to codebook-0 (s1: `clamp(tok - semantic_begin, 0,
    /// residual_size - 1)`; s2 owns its own mapping). Deterministic, no sampling.
    fn first_code(&self, semantic_token: u32) -> u32;

    /// Run the fast AR for one frame: prime a fresh fast KV cache with `hidden`, seed it
    /// with `first_code`, then sample the remaining residual codes via `sampler`. Returns
    /// the full frame `[num_codebooks]` (index 0 == `first_code`).
    fn fast_expand(&self, hidden: &Tensor, first_code: u32, sampler: &mut Sampler)
        -> Result<Vec<u32>>;

    /// Whether `semantic_token` is the end-of-utterance stop (`<|im_end|>`).
    fn is_stop(&self, semantic_token: u32) -> bool {
        semantic_token == self.config().stop_token_id
    }

    // --- Batched generation (corpus-render speedup) -------------------------------------
    // The 5B forward is memory-bandwidth-bound at batch=1, so rendering N short samples
    // together gives ~Nx throughput. The default impls error; backends that support it
    // (s2-pro) override all three. The single-sample methods above are untouched.

    /// Prefill N (variable-length, left-padded) prompts into a shared `[N, ...]` slow cache,
    /// returning the slow step for every sample's last real prompt position.
    fn prefill_batch(&mut self, _prompts: &[Tensor]) -> Result<SlowStepBatch> {
        Err(candle_core::Error::Msg(
            "batched generation (prefill_batch) is not implemented for this backend".into(),
        ))
    }

    /// One batched slow step: feed each sample's previous full frame `[1 + num_codebooks]`
    /// at physical position `pos`, advancing the shared cache by one.
    fn slow_step_batch(&mut self, _frames: &[Vec<u32>], _pos: usize) -> Result<SlowStepBatch> {
        Err(candle_core::Error::Msg(
            "batched generation (slow_step_batch) is not implemented for this backend".into(),
        ))
    }

    /// Batched fast AR over N frames. `hidden` is `[N, 1, fast_dim]`; `first_codes[i]` seeds
    /// sample `i`'s codebook-0; `active[i]` marks samples still generating (a finished sample
    /// draws no residuals so its PRNG does not advance); `samplers[i]` draws sample `i`'s
    /// residuals. Returns N frames `[num_codebooks]` (index 0 == `first_codes[i]`).
    fn fast_expand_batch(
        &self,
        _hidden: &Tensor,
        _first_codes: &[u32],
        _active: &[bool],
        _samplers: &mut [Sampler],
    ) -> Result<Vec<Vec<u32>>> {
        Err(candle_core::Error::Msg(
            "batched generation (fast_expand_batch) is not implemented for this backend".into(),
        ))
    }
}

/// Knobs for one [`drive`] run.
pub struct DriveParams {
    /// Hard cap on frames generated after the prompt (the reference `max_new_tokens`).
    pub max_new_frames: usize,
    /// PRNG seed (bit-reproducible runs).
    pub seed: u64,
    /// Sampling knobs (temperature / top-p / top-k / repetition penalty / RAS).
    pub sampling: SamplingParams,
}

impl Default for DriveParams {
    fn default() -> Self {
        Self {
            max_new_frames: 2048,
            seed: 42,
            sampling: SamplingParams::default(),
        }
    }
}

/// Drive the dual-AR loop to completion, returning the `[num_codebooks, T]` code matrix
/// (u32) on the backend's device. `T` is the number of frames generated before the stop
/// token (or `max_new_frames`). An immediate stop yields a `[num_codebooks, 0]` matrix.
pub fn drive<B: DualArBackend + ?Sized>(
    backend: &mut B,
    prompt: &Tensor,
    params: &DriveParams,
) -> Result<Tensor> {
    let cfg = backend.config().clone();
    let n_cb = cfg.codec.num_codebooks;
    let constraint = SemanticConstraint {
        begin: cfg.semantic_begin_id,
        end: cfg.semantic_end_id,
        stop: cfg.stop_token_id,
    };

    backend.reset(cfg.slow.max_seq_len)?;
    let mut sampler = Sampler::new(params.seed, params.sampling.clone());

    let prompt_len = prompt.dim(1)?;

    // Per-frame residual codes, plus the rolling RAS window of recent semantic tokens.
    let mut frames: Vec<Vec<u32>> = Vec::new();
    let mut window: Vec<u32> = Vec::new();
    let ras_win = params.sampling.ras_win_size.max(1);

    // Prefill → the slow step for the last prompt position (the first frame to sample).
    let mut step = backend.prefill(prompt)?;
    let mut pos = prompt_len;

    loop {
        let logits: Vec<f32> = step.semantic_logits.to_vec1()?;
        let semantic = sampler.sample_semantic(&logits, &window, &constraint);
        if backend.is_stop(semantic) {
            break;
        }

        let first = backend.first_code(semantic);
        let frame = backend.fast_expand(&step.hidden, first, &mut sampler)?;
        debug_assert_eq!(frame.len(), n_cb, "fast_expand must return num_codebooks codes");
        frames.push(frame.clone());

        // Roll the RAS window of semantic tokens.
        window.push(semantic);
        if window.len() > ras_win {
            window.remove(0);
        }

        if frames.len() >= params.max_new_frames {
            break;
        }
        if pos + 1 >= cfg.slow.max_seq_len {
            break;
        }

        // Feed the full frame `[semantic, code0..code(n-1)]` back into the slow AR.
        let mut full = Vec::with_capacity(1 + n_cb);
        full.push(semantic);
        full.extend_from_slice(&frame);
        step = backend.slow_step(&full, pos)?;
        pos += 1;
    }

    // Assemble `[num_codebooks, T]`: matrix[c][t] = frames[t][c].
    let t = frames.len();
    let mut flat = vec![0u32; n_cb * t];
    for (ti, frame) in frames.iter().enumerate() {
        for (ci, &code) in frame.iter().enumerate() {
            flat[ci * t + ti] = code;
        }
    }
    Tensor::from_vec(flat, (n_cb, t), &backend.device())
}

/// Drive the dual-AR loop for **N samples at once** (the corpus-render speedup), returning
/// one `[num_codebooks, T_i]` code matrix per sample — each trimmed to its OWN stop length,
/// not the padded length. `prompts[i]` is sample `i`'s encoded prompt `[1 + num_codebooks,
/// T_i]` (lengths may differ; the backend left-pads internally).
///
/// Per-sample reproducibility: sample `i` gets its own [`Sampler`] seeded `params.seed + i`
/// with independent RAS history, and within a frame its draws happen in the SAME order as
/// the single-sample [`drive`] (the two semantic draws, then — only if not stopped — the
/// nine fast residual draws). So sample `i` reproduces a batch=1 [`drive`] run seeded
/// `params.seed + i`, provided the backend's left-pad/mask/positions make padding invisible.
///
/// A finished sample is **frozen**: it stops being sampled and contributing frames, is fed a
/// pad frame to keep the batch rectangular, and its rows are discarded — but it stays in the
/// batch until ALL samples stop or `max_new_frames` is reached.
pub fn drive_batch<B: DualArBackend + ?Sized>(
    backend: &mut B,
    prompts: &[Tensor],
    params: &DriveParams,
) -> Result<Vec<Tensor>> {
    let cfg = backend.config().clone();
    let n_cb = cfg.codec.num_codebooks;
    let constraint = SemanticConstraint {
        begin: cfg.semantic_begin_id,
        end: cfg.semantic_end_id,
        stop: cfg.stop_token_id,
    };

    backend.reset(cfg.slow.max_seq_len)?;
    let n = prompts.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // One independent sampler per sample (seed + i), mirroring the single-sample stream but
    // with per-sample reproducibility; each carries its own rolling RAS window.
    let mut samplers: Vec<Sampler> = (0..n)
        .map(|i| Sampler::new(params.seed + i as u64, params.sampling.clone()))
        .collect();
    let mut frames: Vec<Vec<Vec<u32>>> = vec![Vec::new(); n];
    let mut windows: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut finished = vec![false; n];
    let ras_win = params.sampling.ras_win_size.max(1);

    // Physical cache length after prefill == max prompt length (everything is left-padded
    // to it); the next generated frame sits at that physical position for all samples.
    let mut tmax = 0usize;
    for p in prompts {
        tmax = tmax.max(p.dim(1)?);
    }

    let mut step = backend.prefill_batch(prompts)?;
    let mut pos = tmax;
    let mut n_generated = 0usize;

    loop {
        // 1. Sample each active sample's semantic token (its own sampler + RAS window).
        let logits_rows: Vec<Vec<f32>> = step.logits.to_vec2()?;
        let mut semantics = vec![0u32; n];
        let mut first_codes = vec![0u32; n];
        for i in 0..n {
            if finished[i] {
                continue;
            }
            let s = samplers[i].sample_semantic(&logits_rows[i], &windows[i], &constraint);
            if backend.is_stop(s) {
                finished[i] = true; // frozen from here: no frame, no fast draws
                continue;
            }
            semantics[i] = s;
            first_codes[i] = backend.first_code(s);
        }
        if finished.iter().all(|&f| f) {
            break;
        }

        // 2. Batched fast AR over all N (finished samples draw nothing — see `active`).
        let active: Vec<bool> = finished.iter().map(|&f| !f).collect();
        let batch_frames =
            backend.fast_expand_batch(&step.hidden, &first_codes, &active, &mut samplers)?;

        // 3. Append each active sample's frame + roll its RAS window.
        for i in 0..n {
            if finished[i] {
                continue;
            }
            debug_assert_eq!(batch_frames[i].len(), n_cb);
            frames[i].push(batch_frames[i].clone());
            windows[i].push(semantics[i]);
            if windows[i].len() > ras_win {
                windows[i].remove(0);
            }
        }

        n_generated += 1;
        if n_generated >= params.max_new_frames {
            break;
        }
        if pos + 1 >= cfg.slow.max_seq_len {
            break;
        }

        // 4. Build each sample's full frame `[semantic, code0..]` (pad frame for finished)
        //    and advance the slow AR one step over the whole batch.
        let mut full: Vec<Vec<u32>> = Vec::with_capacity(n);
        for i in 0..n {
            if finished[i] {
                full.push(vec![0u32; 1 + n_cb]); // inert pad frame (row discarded)
            } else {
                let mut f = Vec::with_capacity(1 + n_cb);
                f.push(semantics[i]);
                f.extend_from_slice(&batch_frames[i]);
                full.push(f);
            }
        }
        step = backend.slow_step_batch(&full, pos)?;
        pos += 1;
    }

    // Assemble one `[num_codebooks, T_i]` matrix per sample, trimmed to its own stop length.
    let dev = backend.device();
    let mut out = Vec::with_capacity(n);
    for sample in &frames {
        let t = sample.len();
        let mut flat = vec![0u32; n_cb * t];
        for (ti, frame) in sample.iter().enumerate() {
            for (ci, &code) in frame.iter().enumerate() {
                flat[ci * t + ti] = code;
            }
        }
        out.push(Tensor::from_vec(flat, (n_cb, t), &dev)?);
    }
    Ok(out)
}
