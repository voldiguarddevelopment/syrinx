//! Deterministic sampling: the SplitMix64 PRNG, `log_softmax`, and the
//! nucleus / repetition-aware (`ras`) / random samplers.
//!
//! Split out verbatim from the original single-file `real` port. The entry points the
//! generation loops call (`log_softmax_vec`, `ras_sampling`, and the `SplitMix64` PRNG)
//! are `pub(super)`; `multinomial1`/`nucleus_sampling`/`random_sampling` are only used
//! within this module and stay private.

use super::{SPEECH_TOKEN_SIZE, SPEECH_VOCAB};
use candle_core::{Result, Tensor};

/// `log_softmax` over a 1-D logit vector `[V]`, returned as a host `Vec<f32>`.
pub(super) fn log_softmax_vec(logits: &Tensor) -> Result<Vec<f32>> {
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
/// `generate` run is bit-reproducible from a seed (the reference pins torch's RNG; we
/// pin ours). `next_f64` yields a uniform in `[0, 1)`.
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
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
        // 53-bit mantissa uniform in [0,1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Sample one index from a categorical distribution given by `probs` (need not be
/// normalised) using inverse-CDF on a single uniform draw — the deterministic analogue
/// of `torch.multinomial(probs, 1)`.
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

/// `nucleus_sampling`: softmax(logp) is `exp(logp)`; sort descending (stable), take the
/// leading tokens while `cum_prob < top_p` AND `count < top_k`, then sample one of those
/// by `multinomial`. Returns the chosen vocab id. `logp` is a log-probability vector.
fn nucleus_sampling(logp: &[f32], top_p: f32, top_k: usize, rng: &mut SplitMix64) -> u32 {
    // probabilities = exp(log_softmax)
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    // stable descending sort by probability (ties keep ascending index, like torch stable)
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

/// `random_sampling`: full-softmax multinomial over the whole vocab (used by `ras` after
/// it masks a repeated token).
fn random_sampling(logp: &[f32], rng: &mut SplitMix64) -> u32 {
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    multinomial1(&probs, rng) as u32
}

/// `ras_sampling` (Repetition-Aware Sampling): nucleus-sample a candidate; if it has
/// repeated `>= win_size * tau_r` times in the last `win_size` decoded tokens, fall back
/// to `random_sampling` over the **plain full distribution** (the reference does NOT mask
/// the repeated id — masking it forces an off-distribution token on every natural repeat,
/// which collapses generation; this was a real bug, the same one fixed on the CV3 path).
/// EOS (`speech_token_size`) is `-inf`-masked first when `ignore_eos`. Mirrors
/// `cosyvoice.utils.common.ras_sampling` with the pinned `top_p=0.8, top_k=25, win_size=10, tau_r=0.1`.
pub(super) fn ras_sampling(logp: &[f32], decoded: &[u32], ignore_eos: bool, rng: &mut SplitMix64) -> u32 {
    const TOP_P: f32 = 0.8;
    const TOP_K: usize = 25;
    const WIN: usize = 10;
    const TAU_R: f32 = 0.1;
    let mut logp = logp.to_vec();
    if ignore_eos {
        // Mask the FULL stop set (`STOP_TOKENS` = 6561..6564), not just the EOS id —
        // otherwise a min_len-window draw of an adjacent control id (6562/6563) trips
        // the stop early. This is the analogue of CV3's full-control-range min_len mask
        // (`real_cv3::ras_sampling`); the previous EOS-only mask was a latent early-stop.
        for s in logp[SPEECH_TOKEN_SIZE as usize..SPEECH_VOCAB].iter_mut() {
            *s = f32::NEG_INFINITY;
        }
    }
    let top = nucleus_sampling(&logp, TOP_P, TOP_K, rng);
    let start = decoded.len().saturating_sub(WIN);
    let rep = decoded[start..].iter().filter(|&&t| t == top).count();
    if (rep as f32) >= WIN as f32 * TAU_R {
        // Resample from the full distribution — do NOT mask `top` (matches the reference).
        return random_sampling(&logp, rng);
    }
    top
}
