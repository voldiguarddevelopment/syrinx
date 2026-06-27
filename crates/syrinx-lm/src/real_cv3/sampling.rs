// -----------------------------------------------------------------------------
// Deterministic sampler — a focused mirror of the CV2 (`real`) sampler, kept
// local so the CV2 module stays byte-for-byte unchanged (additive-only port).
// The PRNG + nucleus + repetition-aware logic are the same pinned algorithm; the
// only CV3 specialisation lives in `Cv3Lm::generate`'s stop check (200 control
// ids) — `ras_sampling`'s EOS index (`speech_token_size` = 6561) is identical to
// CV2's, so the sampler body matches the reference `ras_sampling` exactly.
//
// Split out verbatim from the original single-file CV3 port. The entry points the
// `generate` loops use (`log_softmax_vec`, `ras_sampling`, the `SplitMix64` PRNG and
// the `RasOutcome` it returns) are `pub(super)`; the nucleus/random/multinomial
// primitives are private. The `testkit` test seam is re-exported at the original
// `real_cv3::testkit` path from `mod.rs`.
// -----------------------------------------------------------------------------

use super::{DECODER_OUT, SPEECH_TOKEN_SIZE};
use candle_core::{Result, Tensor};

/// `log_softmax` over a 1-D logit vector `[V]`, returned as a host `Vec<f32>` (f64 accum).
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
/// `generate` run is bit-reproducible from a seed. `next_f64` yields a uniform in `[0,1)`.
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
pub(super) struct RasOutcome {
    pub(super) token: u32,
    pub(super) triggered: bool,
}

/// `ras_sampling` (Repetition-Aware Sampling), an exact port of
/// `cosyvoice/utils/common.py:ras_sampling` (:138): nucleus-sample a candidate; if it
/// repeated `>= win_size * tau_r` (= 1.0) times in the last `win_size` decoded tokens, fall
/// back to `random_sampling`. The control range (`speech_token_size..DECODER_OUT`) is
/// `-inf`-masked first when `ignore_eos` (CV3's wider-than-CV2 stop set; the masking analogue
/// of the reference's eos rejection loop while `step < min_len`). Pinned
/// `top_p=0.8, top_k=25, win_size=10, tau_r=0.1` (the dump metadata).
pub(super) fn ras_sampling(logp: &[f32], decoded: &[u32], ignore_eos: bool, rng: &mut SplitMix64) -> RasOutcome {
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
