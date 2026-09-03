//! Seeded sampling for the Qwen3-TTS dual-AR driver.
//!
//! Pure Rust (no Candle): [`crate::model`] pulls its logit `Tensor` to a host `Vec<f32>`
//! and hands it here, exactly as the Fish port's `common/sampling.rs` does.
//!
//! ## The pipeline is HuggingFace's, in HuggingFace's order
//!
//! The reference never hand-rolls a sampler: `Qwen3TTSTalkerForConditionalGeneration`
//! and `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` are both
//! `GenerationMixin` subclasses, so every draw goes through
//! `transformers.generation.utils._get_logits_processor` → `LogitsProcessorList`. The
//! order that list is built in is load-bearing and is reproduced here:
//!
//! 1. [`apply_repetition_penalty`] — `RepetitionPenaltyLogitsProcessor`, over the ids
//!    generated **so far in this call** (the talker is driven with `inputs_embeds` and no
//!    `input_ids`, so HF seeds `input_ids` empty and the penalty never sees the prompt).
//! 2. [`block_ids`] as the min-new-tokens guard — `MinNewTokensLengthLogitsProcessor`
//!    with the reference's `min_new_tokens = 2`, masking the codec EOS.
//! 3. [`block_ids`] again for `suppress_tokens` — `SuppressTokensLogitsProcessor`.
//! 4. [`Sampler::sample`], which is the three warpers in HF's order: **temperature,
//!    then top-k, then top-p**, then `softmax` + `multinomial`.
//!
//! The temperature-first ordering matters and is *not* what the Fish port does (Fish cuts
//! the nucleus on raw logits and divides by the temperature afterwards). Applying top-p
//! before temperature changes which tokens survive whenever `temperature != 1`, so the two
//! conventions are not interchangeable.
//!
//! Two HF details are reproduced exactly rather than approximated, because both change the
//! support at the boundary:
//!
//! * `TopKLogitsWarper` removes `scores < kth_largest` — **strictly** less, so a tie at the
//!   threshold keeps more than `k` candidates.
//! * `TopPLogitsWarper` sorts **ascending** and removes while `cumsum(softmax) <= 1 - top_p`,
//!   always keeping the single largest (`min_tokens_to_keep = 1`, since the reference runs
//!   `num_beams == 1`). That is not the same cut as "keep the shortest descending prefix
//!   whose mass reaches `top_p`" when probabilities tie at the edge.

/// Deterministic SplitMix64 PRNG — pins the otherwise-stochastic multinomial draws so a
/// generation run is bit-reproducible from a seed. `next_f64` yields a uniform in `[0, 1)`.
/// Identical algorithm to the Fish port's sampler, so the two crates' seeds behave alike.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the PRNG.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Warper knobs for one autoregressive head.
///
/// Defaults come from the published `generation_config.json`, which is what the reference
/// actually runs: `Qwen3TTSModel._merge_generate_kwargs` prefers that file over its own
/// hard-coded fallbacks, and every shipped checkpoint (0.6B/1.7B x Base/CustomVoice/
/// VoiceDesign) carries the same values.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature. Applied FIRST, before top-k/top-p (HF warper order).
    pub temperature: f32,
    /// Nucleus cutoff. `>= 1.0` disables the warper entirely — HF only installs
    /// `TopPLogitsWarper` when `top_p < 1.0`, and the reference default is exactly `1.0`.
    pub top_p: f32,
    /// Top-k cap. `0` disables the warper (HF installs it only when `top_k != 0`).
    pub top_k: usize,
    /// Repetition penalty over previously drawn ids. `1.0` = off.
    pub repetition_penalty: f32,
}

impl SamplingParams {
    /// The talker (slow AR, code group 0) defaults: `generation_config.json`'s
    /// `temperature` / `top_p` / `top_k` / `repetition_penalty`.
    pub fn talker() -> Self {
        Self {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.05,
        }
    }

    /// The code predictor (fast AR, groups `1..num_code_groups`) defaults: the
    /// `subtalker_*` entries of `generation_config.json`. There is deliberately **no**
    /// repetition penalty here — `generate()` forwards only `subtalker_dosample`,
    /// `subtalker_top_k`, `subtalker_top_p` and `subtalker_temperature` to
    /// `code_predictor.generate`, so the penalty falls back to the code predictor's own
    /// config value, which is `1.0`.
    pub fn code_predictor() -> Self {
        Self {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::talker()
    }
}

/// `RepetitionPenaltyLogitsProcessor`: divide the positive logits of ids in `generated` by
/// `penalty`, multiply the negative ones by it. No-op at `1.0`, and an id repeated in
/// `generated` is still penalised only once (the penalty is idempotent per id under HF's
/// gather/scatter, which this loop matches by touching each slot from its original value).
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    // HF gathers the ORIGINAL scores, scales them, and scatters back — so a duplicate id
    // in the history is penalised once, not once per occurrence. Deduplicate to match.
    let mut done = vec![false; logits.len()];
    for &id in generated {
        let i = id as usize;
        if i >= logits.len() || done[i] {
            continue;
        }
        done[i] = true;
        let l = logits[i];
        logits[i] = if l < 0.0 { l * penalty } else { l / penalty };
    }
}

/// Mask `ids` to `-inf`. This is both `SuppressTokensLogitsProcessor` (the talker's
/// `suppress_tokens` block) and `MinNewTokensLengthLogitsProcessor` (masking the codec EOS
/// until `min_new_tokens` frames exist) — HF implements them with the same
/// `torch.where(mask, -inf, scores)`.
pub fn block_ids(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        let i = id as usize;
        if i < logits.len() {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// `TemperatureLogitsWarper`: `scores / temperature`. HF installs it only when
/// `temperature != 1.0`; dividing by 1.0 is the identity, so the guard is cosmetic here.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

/// `TopKLogitsWarper`: mask everything strictly below the `k`-th largest score.
///
/// `k == 0` disables the warper (HF's `top_k != 0` guard). `k >= len` is a no-op, since the
/// `k`-th largest is then the minimum and nothing is strictly below it.
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut sorted: Vec<f32> = logits.to_vec();
    // Descending; NaN is not expected from a logit head, and `total_cmp` keeps the sort
    // total regardless.
    sorted.sort_by(|a, b| b.total_cmp(a));
    let kth = sorted[k - 1];
    for l in logits.iter_mut() {
        if *l < kth {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// `TopPLogitsWarper` with `min_tokens_to_keep = 1`: sort ascending, take the cumulative
/// softmax, and mask every token whose cumulative mass is `<= 1 - top_p`, never masking the
/// largest.
///
/// `top_p >= 1.0` disables the warper (HF's `top_p < 1.0` guard), which is what the shipped
/// `generation_config.json` selects.
pub fn top_p_filter(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 || logits.len() < 2 {
        return;
    }
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Ascending, index as the tiebreak so the order is total and reproducible.
    order.sort_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(a.cmp(&b)));

    let max = logits[order[n - 1]];
    if !max.is_finite() {
        return;
    }
    let exps: Vec<f64> = order.iter().map(|&i| ((logits[i] - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        return;
    }

    let cutoff = 1.0 - top_p as f64;
    let mut cum = 0f64;
    // `min_tokens_to_keep = 1`: the last (largest) entry is never masked.
    for rank in 0..n - 1 {
        cum += exps[rank] / sum;
        if cum <= cutoff {
            logits[order[rank]] = f32::NEG_INFINITY;
        }
    }
}

/// Softmax `logits` (already warped) into a probability vector. `-inf` entries become 0.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let exps: Vec<f64> = logits.iter().map(|&l| ((l - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// The seeded sampler. One instance drives a whole utterance: the talker draw and the 15
/// code-predictor draws of every frame share the stream, so a `(seed, weights, prompt)`
/// triple reproduces bit-for-bit.
pub struct Sampler {
    rng: SplitMix64,
}

impl Sampler {
    /// New sampler from a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
        }
    }

    /// Apply the three warpers in HF's order (temperature → top-k → top-p) to `logits`
    /// **in place**, then draw one id.
    ///
    /// `logits` is mutated so the caller can inspect the realised support in a test; the
    /// repetition penalty, the EOS min-length guard and `suppress_tokens` are the caller's
    /// job, because they run BEFORE the warpers and only the caller knows the history.
    pub fn sample(&mut self, logits: &mut [f32], p: &SamplingParams) -> u32 {
        apply_temperature(logits, p.temperature);
        top_k_filter(logits, p.top_k);
        top_p_filter(logits, p.top_p);
        let probs = softmax(logits);
        self.multinomial(&probs)
    }

    /// Inverse-CDF draw from `probs` on a single uniform — the deterministic analogue of
    /// `torch.multinomial(probs, 1)`.
    ///
    /// PARITY: this samples the same categorical distribution as torch's multinomial; only
    /// the RNG stream differs, and a stochastic sampler is never bit-exact across PRNGs
    /// anyway. Distribution-level agreement is the thing to confirm on-box.
    pub fn multinomial(&mut self, probs: &[f32]) -> u32 {
        let total: f64 = probs.iter().map(|&p| p as f64).sum();
        if total <= 0.0 {
            return 0;
        }
        let u = self.rng.next_f64() * total;
        let mut acc = 0f64;
        for (i, &p) in probs.iter().enumerate() {
            acc += p as f64;
            if u < acc {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A logit vector with a clear, hand-checkable ordering: id 4 > 3 > 2 > 1 > 0.
    fn ramp() -> Vec<f32> {
        vec![0.0, 1.0, 2.0, 3.0, 4.0]
    }

    fn draw_many(seed: u64, logits: &[f32], p: &SamplingParams, n: usize) -> Vec<u32> {
        let mut s = Sampler::new(seed);
        (0..n)
            .map(|_| {
                let mut w = logits.to_vec();
                s.sample(&mut w, p)
            })
            .collect()
    }

    /// The defaults must be the published `generation_config.json`, not HF's library
    /// fallbacks — the reference prefers the file, and the two differ in `max_new_tokens`.
    #[test]
    fn defaults_are_the_published_generation_config() {
        let t = SamplingParams::talker();
        assert_eq!(t.temperature, 0.9);
        assert_eq!(t.top_p, 1.0);
        assert_eq!(t.top_k, 50);
        assert_eq!(t.repetition_penalty, 1.05);
        // The subtalker gets the same warpers but NO repetition penalty.
        let c = SamplingParams::code_predictor();
        assert_eq!(c.temperature, 0.9);
        assert_eq!(c.top_p, 1.0);
        assert_eq!(c.top_k, 50);
        assert_eq!(c.repetition_penalty, 1.0);
        assert_eq!(SamplingParams::default(), t);
    }

    /// A fixed seed must reproduce a fixed sequence, and a different seed must not.
    #[test]
    fn a_fixed_seed_reproduces_a_fixed_sequence() {
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
        };
        let a = draw_many(0, &ramp(), &p, 12);
        let b = draw_many(0, &ramp(), &p, 12);
        assert_eq!(a, b, "same seed must replay identically");
        // Pinned so a change to the PRNG, the draw, or the warper order fails loudly.
        assert_eq!(a, vec![4, 4, 1, 4, 2, 3, 3, 4, 3, 4, 4, 4]);
        let c = draw_many(1, &ramp(), &p, 12);
        assert_ne!(a, c, "a different seed must give a different stream");
    }

    /// Top-k must actually restrict the support, on both sides of the cut.
    #[test]
    fn top_k_restricts_the_support() {
        let mut l = ramp();
        top_k_filter(&mut l, 2);
        // ids 3 and 4 survive; 0..=2 are masked.
        assert_eq!(l[4], 4.0);
        assert_eq!(l[3], 3.0);
        assert!(l[2].is_infinite() && l[2] < 0.0);
        assert!(l[1].is_infinite() && l[1] < 0.0);
        assert!(l[0].is_infinite() && l[0] < 0.0);

        // k = 1 leaves only the argmax; sampling can only ever return it.
        let p = SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 1, repetition_penalty: 1.0 };
        assert!(draw_many(7, &ramp(), &p, 40).iter().all(|&i| i == 4));

        // k = 2 admits exactly {3, 4} and, over enough draws, actually reaches both.
        let p2 = SamplingParams { top_k: 2, ..p.clone() };
        let got = draw_many(7, &ramp(), &p2, 200);
        assert!(got.iter().all(|&i| i == 3 || i == 4), "support leaked past k=2");
        assert!(got.contains(&3) && got.contains(&4), "k=2 collapsed to one id");

        // k = 0 is OFF (HF only installs the warper when top_k != 0), so ids below the
        // top-2 must reappear — the boundary just past "restricting".
        let p0 = SamplingParams { top_k: 0, ..p.clone() };
        assert!(draw_many(7, &ramp(), &p0, 200).iter().any(|&i| i < 3));

        // k >= len is a no-op: the k-th largest is the minimum, nothing is strictly below.
        let mut l = ramp();
        top_k_filter(&mut l, 5);
        assert_eq!(l, ramp());
    }

    /// Top-k keeps ties at the threshold — HF masks `scores < kth`, strictly.
    #[test]
    fn top_k_keeps_ties_at_the_threshold() {
        let mut l = vec![5.0f32, 5.0, 5.0, 1.0];
        top_k_filter(&mut l, 2);
        // All three 5.0s survive even though k == 2, because none is strictly below 5.0.
        assert_eq!(&l[0..3], &[5.0, 5.0, 5.0]);
        assert!(l[3].is_infinite() && l[3] < 0.0);
    }

    /// Top-p must restrict the support, and `>= 1.0` must be a no-op (the shipped default).
    #[test]
    fn top_p_restricts_the_support() {
        // softmax(ramp) ~= [0.0117, 0.0317, 0.0861, 0.2341, 0.6364]; ascending cumsum is
        // [0.0117, 0.0434, 0.1295, 0.3636, 1.0]. With top_p = 0.9 the cutoff is 0.1, so
        // ids 0 and 1 go (0.0117 <= 0.1, 0.0434 <= 0.1) and id 2 stays (0.1295 > 0.1).
        let mut l = ramp();
        top_p_filter(&mut l, 0.9);
        assert!(l[0].is_infinite() && l[1].is_infinite());
        assert_eq!(&l[2..], &[2.0, 3.0, 4.0]);

        // Just past that boundary: top_p = 0.85 lifts the cutoff to 0.15, which now also
        // swallows id 2 (0.1295 <= 0.15). Both sides of the same comparison.
        let mut l = ramp();
        top_p_filter(&mut l, 0.85);
        assert!(l[2].is_infinite() && l[2] < 0.0);
        assert_eq!(&l[3..], &[3.0, 4.0]);

        // 1.0 is OFF — HF installs the warper only when top_p < 1.0.
        let mut l = ramp();
        top_p_filter(&mut l, 1.0);
        assert_eq!(l, ramp());

        // And the restriction is visible through the sampler, not just the filter.
        let p = SamplingParams { temperature: 1.0, top_p: 0.9, top_k: 0, repetition_penalty: 1.0 };
        assert!(draw_many(3, &ramp(), &p, 300).iter().all(|&i| i >= 2));
        let off = SamplingParams { top_p: 1.0, ..p };
        assert!(draw_many(3, &ramp(), &off, 300).iter().any(|&i| i < 2));
    }

    /// A tiny top-p must still keep exactly one token (`min_tokens_to_keep = 1`).
    #[test]
    fn top_p_always_keeps_the_argmax() {
        let mut l = ramp();
        top_p_filter(&mut l, 1e-6);
        assert_eq!(l[4], 4.0);
        assert!(l[..4].iter().all(|v| v.is_infinite() && *v < 0.0));
    }

    /// Temperature is applied BEFORE the nucleus cut (HF warper order). A temperature that
    /// sharpens the distribution must therefore let MORE tokens survive the same top_p —
    /// the observable consequence of the ordering, and what a Fish-style
    /// nucleus-then-temperature sampler would get backwards.
    #[test]
    fn temperature_runs_before_the_nucleus_cut() {
        let p_cold = SamplingParams { temperature: 0.5, top_p: 0.9, top_k: 0, repetition_penalty: 1.0 };
        let mut cold = ramp();
        let mut s = Sampler::new(0);
        s.sample(&mut cold, &p_cold);

        let p_hot = SamplingParams { temperature: 2.0, ..p_cold.clone() };
        let mut hot = ramp();
        s.sample(&mut hot, &p_hot);

        let survivors = |v: &[f32]| v.iter().filter(|x| x.is_finite()).count();
        // Sharper (t < 1) concentrates mass on the top ids, so the low tail's cumulative
        // mass stays under the 1 - top_p cutoff for longer and MORE of it is masked; a
        // flatter distribution (t > 1) crosses the cutoff sooner and keeps more.
        assert!(
            survivors(&hot) > survivors(&cold),
            "hot {} vs cold {}",
            survivors(&hot),
            survivors(&cold)
        );
        // The load-bearing part: had the nucleus been cut before the temperature — the
        // Fish port's order — the cut would have run on the same raw logits both times and
        // the two counts would be equal.
        let mut raw = ramp();
        top_p_filter(&mut raw, 0.9);
        let raw_survivors = survivors(&raw);
        assert!(
            survivors(&cold) != raw_survivors || survivors(&hot) != raw_survivors,
            "temperature had no effect on the nucleus cut"
        );
    }

    /// Very low temperature is effectively greedy — the property `model.rs`'s loop tests
    /// lean on to get a deterministic run out of synthetic weights.
    #[test]
    fn low_temperature_is_greedy() {
        let p = SamplingParams { temperature: 0.01, top_p: 1.0, top_k: 0, repetition_penalty: 1.0 };
        assert!(draw_many(11, &ramp(), &p, 50).iter().all(|&i| i == 4));
    }

    /// The HF penalty convention: positive logits are DIVIDED, negative ones MULTIPLIED,
    /// so both move toward -inf. `1.0` is a no-op on both signs.
    #[test]
    fn repetition_penalty_follows_the_hf_sign_convention() {
        let mut l = vec![2.0f32, -2.0, 5.0];
        apply_repetition_penalty(&mut l, &[0, 1], 2.0);
        assert_eq!(l[0], 1.0, "positive logit divided");
        assert_eq!(l[1], -4.0, "negative logit multiplied");
        assert_eq!(l[2], 5.0, "untouched id unchanged");

        let mut l = vec![2.0f32, -2.0, 5.0];
        apply_repetition_penalty(&mut l, &[0, 1, 2], 1.0);
        assert_eq!(l, vec![2.0, -2.0, 5.0], "penalty 1.0 must be a no-op");
    }

    /// A repeated id is penalised once, not once per occurrence (HF gathers the original
    /// scores and scatters back, so duplicates collapse).
    #[test]
    fn repetition_penalty_is_applied_at_most_once_per_id() {
        let mut l = vec![8.0f32];
        apply_repetition_penalty(&mut l, &[0, 0, 0], 2.0);
        assert_eq!(l[0], 4.0);
    }

    /// Blocked ids are unreachable — the mechanism behind both `suppress_tokens` and the
    /// min-new-tokens EOS guard.
    #[test]
    fn blocked_ids_are_never_drawn() {
        let mut l = ramp();
        block_ids(&mut l, &[3, 4, 99]); // 99 is out of range and must be ignored
        assert!(l[3].is_infinite() && l[4].is_infinite());
        assert_eq!(&l[0..3], &[0.0, 1.0, 2.0]);

        let p = SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0 };
        let mut s = Sampler::new(5);
        for _ in 0..200 {
            let mut w = ramp();
            block_ids(&mut w, &[3, 4]);
            assert!(s.sample(&mut w, &p) < 3);
        }
    }

    /// An all-masked vector must not panic or index out of bounds.
    #[test]
    fn a_fully_masked_vector_degrades_safely() {
        let mut l = vec![f32::NEG_INFINITY; 4];
        let mut s = Sampler::new(0);
        assert_eq!(s.sample(&mut l, &SamplingParams::talker()), 0);
    }
}
