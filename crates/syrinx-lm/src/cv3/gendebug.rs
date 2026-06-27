//! The env-gated (`SYRINX_CV3_GEN_DEBUG`) live-AR generation diagnostic for [`Cv3Lm`] —
//! pure instrumentation, never alters the returned token sequence.
//!
//! Split out verbatim from the original single-file CV3 port: [`Cv3Lm::gen_debug_report`]
//! plus its `gen_debug_enabled` gate and the `argmax_f32` / `topk_indices` / `topk_overlap`
//! helpers. `gen_debug_report` + `gen_debug_enabled` are `pub(super)` (driven by the
//! `generate` loop); the comparison helpers stay private.

use super::{Cv3Lm, DECODER_OUT, SPEECH_TOKEN_SIZE};
use candle_core::{Result, Tensor};

impl Cv3Lm {
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
    pub(super) fn gen_debug_report(
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

/// Whether the env-gated live-AR generation diagnostic is on. Enabled by setting
/// `SYRINX_CV3_GEN_DEBUG` to any non-empty value other than `0`. Read once per `generate`.
pub(super) fn gen_debug_enabled() -> bool {
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
