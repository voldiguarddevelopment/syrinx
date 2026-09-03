//! `syrinx-qwen` — the sampling stack, which is pure Rust and therefore fully decidable
//! off-box.
//!
//! **What is certified here:** that `syrinx_qwen::sampling` reproduces HuggingFace's
//! `LogitsProcessorList` semantics exactly — the warper ORDER (temperature → top-k →
//! top-p), the two boundary conventions that decide the support (`TopKLogitsWarper` masks
//! `scores < kth`, **strictly**; `TopPLogitsWarper` masks while `cumsum <= 1 - top_p` and
//! always keeps the argmax), the repetition-penalty sign convention, the masking
//! primitives behind `suppress_tokens` and `min_new_tokens`, and the seeded PRNG.
//!
//! Both sides of every comparison are pinned: a warper that is *off* must let the tail
//! back in, and a warper that is *on* must cut exactly where HF cuts. Getting these
//! backwards does not crash and does not sound obviously wrong — it silently changes the
//! distribution the model draws from, which is why they are worth a gate.
//!
//! **What is not:** that this matches `torch.multinomial` bit-for-bit. It cannot — a
//! stochastic sampler is never bit-exact across PRNGs, which `sampling.rs` marks with a
//! `// PARITY:` comment. Distribution-level agreement against the reference needs the
//! model and is listed in `docs/backends/QWEN_PORT_STATUS.md`.
//!
//! Model-free by construction: no weights, no Candle, no GPU. It must never SKIP.

use syrinx_qwen::sampling::{
    apply_repetition_penalty, apply_temperature, block_ids, top_k_filter, top_p_filter,
    SamplingParams, Sampler, SplitMix64,
};

/// Four logits, strictly ordered, so every rank is unambiguous: id 3 > 2 > 1 > 0.
fn ladder() -> Vec<f32> {
    vec![-1.0, 0.0, 1.0, 2.0]
}

/// Four EQUAL logits — softmax is exactly `[0.25; 4]`, which makes the top-p cutoff land
/// on a representable boundary and separates `<=` from `<`.
fn flat() -> Vec<f32> {
    vec![0.0; 4]
}

fn survivors(v: &[f32]) -> usize {
    v.iter().filter(|x| x.is_finite()).count()
}

fn masked(v: &[f32], i: usize) -> bool {
    v[i] == f32::NEG_INFINITY
}

fn off() -> SamplingParams {
    SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0 }
}

fn draws(seed: u64, logits: &[f32], p: &SamplingParams, n: usize) -> Vec<u32> {
    let mut s = Sampler::new(seed);
    (0..n)
        .map(|_| {
            let mut w = logits.to_vec();
            s.sample(&mut w, p)
        })
        .collect()
}

// ------------------------------------------------------------------------------- PRNG

/// SplitMix64, against values derived from the published algorithm rather than from this
/// implementation's own output. A wrong multiplier or a wrong shift changes every draw.
#[test]
fn splitmix64_matches_the_reference_algorithm() {
    let mut r = SplitMix64::new(0);
    for want in [
        0.8833108082136426_f64,
        0.43152799704850997,
        0.026433771592597743,
        0.9708819781538285,
    ] {
        assert_eq!(r.next_f64(), want);
    }
    let mut r = SplitMix64::new(42);
    for want in [
        0.7415648787718233_f64,
        0.1599103928769201,
        0.27860113025513866,
        0.34419071652363753,
    ] {
        assert_eq!(r.next_f64(), want);
    }
}

/// The draw is a half-open uniform: `0.0` is reachable in principle, `1.0` never is, and
/// nothing leaves the interval.
#[test]
fn splitmix64_stays_in_the_half_open_unit_interval() {
    let mut r = SplitMix64::new(7);
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for _ in 0..20_000 {
        let u = r.next_f64();
        assert!((0.0..1.0).contains(&u), "{u} outside [0, 1)");
        lo = lo.min(u);
        hi = hi.max(u);
    }
    // Not a constant, and it actually reaches both ends of the interval.
    assert!(lo < 0.001, "never sampled near 0 (min {lo})");
    assert!(hi > 0.999, "never sampled near 1 (max {hi})");
}

/// A seed reproduces a run; a different seed does not. This is what makes a corpus render
/// re-runnable.
#[test]
fn a_seed_pins_the_whole_stream_and_a_different_seed_does_not() {
    let p = off();
    let a = draws(0, &ladder(), &p, 32);
    assert_eq!(a, draws(0, &ladder(), &p, 32), "the same seed must replay");
    assert_ne!(a, draws(1, &ladder(), &p, 32), "a different seed must diverge");
    // One sampler drives a whole utterance, so consecutive draws must NOT repeat the
    // first one — an un-advanced PRNG would look deterministic and be broken.
    assert!(a.windows(2).any(|w| w[0] != w[1]), "the stream never advances");
}

// -------------------------------------------------------------------------- temperature

/// `TemperatureLogitsWarper` is a plain divide, applied to negative logits too. `1.0` is
/// the identity (HF does not even install the warper).
#[test]
fn temperature_divides_and_is_the_identity_at_one() {
    let mut l = vec![-4.0f32, 0.0, 2.0];
    apply_temperature(&mut l, 2.0);
    assert_eq!(l, vec![-2.0, 0.0, 1.0], "sharper values, sign preserved");

    let mut l = vec![-4.0f32, 0.0, 2.0];
    apply_temperature(&mut l, 0.5);
    assert_eq!(l, vec![-8.0, 0.0, 4.0]);

    let mut l = vec![-4.0f32, 0.0, 2.0];
    apply_temperature(&mut l, 1.0);
    assert_eq!(l, vec![-4.0, 0.0, 2.0], "1.0 must be a no-op");

    // Just off the identity: 1.0 is a boundary, not a range.
    let mut l = vec![2.0f32];
    apply_temperature(&mut l, 1.0001);
    assert!(l[0] < 2.0, "any temperature above 1 must flatten");
}

/// The ordering claim, made observable: HF warps temperature FIRST, so the temperature
/// changes *which tokens survive* the nucleus cut. A `nucleus-then-temperature` sampler
/// (what the sibling Fish port does) would cut the same set at every temperature.
#[test]
fn temperature_is_applied_before_the_nucleus_cut() {
    // softmax([0,0,0,4]) puts 94.8 % on id 3; at temperature 4 that falls to 47.5 %, so
    // the low tail crosses the 1 - top_p cutoff sooner and one more id survives.
    let base = vec![0.0f32, 0.0, 0.0, 4.0];
    let cold = SamplingParams { temperature: 1.0, top_p: 0.5, top_k: 0, repetition_penalty: 1.0 };
    let hot = SamplingParams { temperature: 4.0, ..cold.clone() };

    let mut a = base.clone();
    Sampler::new(0).sample(&mut a, &cold);
    let mut b = base.clone();
    Sampler::new(0).sample(&mut b, &hot);

    assert_eq!(survivors(&a), 1, "at t = 1 only the argmax clears the cut");
    assert_eq!(survivors(&b), 2, "at t = 4 the flatter tail keeps one more");
    assert!(survivors(&b) > survivors(&a));

    // The control: cutting the nucleus on the RAW logits gives the cold answer whatever
    // the temperature, which is precisely the bug this ordering avoids.
    let mut raw = base.clone();
    top_p_filter(&mut raw, 0.5);
    assert_eq!(survivors(&raw), survivors(&a));
    assert_ne!(survivors(&raw), survivors(&b));
}

// ------------------------------------------------------------------------------- top-k

/// `k` is a floor on the support, not a ceiling: HF masks `scores < kth_largest`, so ties
/// at the threshold ALL survive and more than `k` candidates can remain.
#[test]
fn top_k_masks_strictly_below_the_kth_largest() {
    // No ties: exactly k survive.
    let mut l = ladder();
    top_k_filter(&mut l, 2);
    assert_eq!(survivors(&l), 2);
    assert_eq!(&l[2..], &[1.0, 2.0]);
    assert!(masked(&l, 0) && masked(&l, 1));

    // Ties at the threshold: THREE survive at k = 2, because none of the 5.0s is strictly
    // below 5.0. A `<=` implementation would leave two.
    let mut l = vec![5.0f32, 5.0, 5.0, 1.0];
    top_k_filter(&mut l, 2);
    assert_eq!(survivors(&l), 3, "ties at the cut are kept");
    assert!(masked(&l, 3));

    // …and one rank further down the same vector the tie is no longer at the threshold.
    let mut l = vec![5.0f32, 5.0, 5.0, 1.0];
    top_k_filter(&mut l, 4);
    assert_eq!(survivors(&l), 4, "k == len is a no-op");
}

/// The two disabling boundaries, each checked immediately on both sides.
#[test]
fn top_k_is_disabled_at_zero_and_at_full_length() {
    // k = 0 is OFF (HF installs the warper only when top_k != 0): the tail survives.
    let mut l = ladder();
    top_k_filter(&mut l, 0);
    assert_eq!(l, ladder());
    // k = 1 is the smallest ON value: only the argmax.
    let mut l = ladder();
    top_k_filter(&mut l, 1);
    assert_eq!(survivors(&l), 1);
    assert_eq!(l[3], 2.0);

    // k = len is a no-op (the kth largest is the minimum; nothing is strictly below it).
    let mut l = ladder();
    top_k_filter(&mut l, 4);
    assert_eq!(l, ladder());
    // k = len - 1 is the largest ON value: it drops exactly the minimum.
    let mut l = ladder();
    top_k_filter(&mut l, 3);
    assert_eq!(survivors(&l), 3);
    assert!(masked(&l, 0));
    // k beyond the length is still a no-op, not a panic.
    let mut l = ladder();
    top_k_filter(&mut l, 99);
    assert_eq!(l, ladder());
}

/// Through the sampler, not just the filter: the realised support obeys `k` in both
/// directions.
#[test]
fn top_k_bounds_what_the_sampler_can_draw() {
    let k1 = SamplingParams { top_k: 1, ..off() };
    assert!(draws(3, &ladder(), &k1, 64).iter().all(|&i| i == 3), "k = 1 is greedy");

    let k2 = SamplingParams { top_k: 2, ..off() };
    let got = draws(3, &ladder(), &k2, 400);
    assert!(got.iter().all(|&i| i >= 2), "support leaked past k = 2");
    assert!(got.contains(&2) && got.contains(&3), "k = 2 collapsed to one id");

    // Off again: the ids k = 2 excluded must come back.
    assert!(draws(3, &ladder(), &off(), 400).iter().any(|&i| i < 2));
}

// ------------------------------------------------------------------------------- top-p

/// `TopPLogitsWarper` masks while the ascending cumulative mass is `<= 1 - top_p`. With
/// four equal logits every probability is exactly 0.25, so `top_p = 0.75` puts the cutoff
/// exactly on the first entry's mass — the one input that tells `<=` apart from `<`.
#[test]
fn top_p_masks_while_the_cumulative_mass_is_at_most_the_cutoff() {
    let mut l = flat();
    top_p_filter(&mut l, 0.75); // cutoff = 0.25, and cum after the first entry is 0.25
    assert!(masked(&l, 0), "an entry whose mass EQUALS the cutoff is masked");
    assert_eq!(survivors(&l), 3);

    // A hair the other way: cutoff drops just below 0.25 and the same entry survives.
    let mut l = flat();
    top_p_filter(&mut l, 0.76);
    assert_eq!(survivors(&l), 4, "cum 0.25 > cutoff 0.24 keeps everything");

    // Further in: cutoff 0.5 swallows the first two, cutoff 0.75 the first three.
    let mut l = flat();
    top_p_filter(&mut l, 0.5);
    assert_eq!(survivors(&l), 2);
    let mut l = flat();
    top_p_filter(&mut l, 0.25);
    assert_eq!(survivors(&l), 1);
}

/// `min_tokens_to_keep = 1`: however small `top_p` is, the largest logit is never masked —
/// the loop stops one short of the top rank.
#[test]
fn top_p_always_keeps_the_argmax() {
    let mut l = ladder();
    top_p_filter(&mut l, 1e-9);
    assert_eq!(survivors(&l), 1);
    assert_eq!(l[3], 2.0, "the argmax survives an arbitrarily tight nucleus");

    // Even when everything ties, exactly one survives.
    let mut l = flat();
    top_p_filter(&mut l, 1e-9);
    assert_eq!(survivors(&l), 1);
    assert!(!masked(&l, 3), "the tiebreak keeps the LAST of the tied maxima");

    // The extreme: `top_p = 0` puts the cutoff at 1.0, which the FULL cumulative mass
    // reaches exactly. Only the "stop one rank short" rule keeps the vector from being
    // masked to nothing — with four equal logits the cumsum is exactly 0.25/0.5/0.75/1.0,
    // so this is an exact comparison and not a floating-point near-miss.
    let mut l = flat();
    top_p_filter(&mut l, 0.0);
    assert_eq!(survivors(&l), 1, "the top rank is never even considered for masking");
    assert!(!masked(&l, 3));
}

/// `top_p >= 1.0` disables the warper — the value the shipped `generation_config.json`
/// actually carries, so the "on" path must not be reachable by accident.
#[test]
fn top_p_is_disabled_at_one_and_above() {
    for p in [1.0f32, 1.5, 2.0] {
        let mut l = ladder();
        top_p_filter(&mut l, p);
        assert_eq!(l, ladder(), "top_p = {p} must be a no-op");
    }
    // At exactly 1.0 the cutoff would be 0.0, which masks nothing on a well-conditioned
    // vector — so `>= 1.0` and `> 1.0` would look identical there. `-1000` underflows to
    // probability 0.0, whose cumulative mass IS `<= 0.0`, and only the early return keeps
    // it unmasked. This is the input that separates the two guards.
    let underflowing = vec![-1000.0f32, 0.0];
    let mut l = underflowing.clone();
    top_p_filter(&mut l, 1.0);
    assert_eq!(l, underflowing, "top_p = 1.0 must return before touching anything");
    // Just below 1.0 the warper is live. `[0, 30]` puts e^-30 ~= 9.4e-14 on the tail, so
    // a cutoff of 1e-6 reaches it while `top_p = 1.0` leaves it alone — the same input on
    // both sides of the one comparison that installs the warper at all.
    let tiny_tail = vec![0.0f32, 30.0];
    let mut on = tiny_tail.clone();
    top_p_filter(&mut on, 0.999999);
    assert_eq!(survivors(&on), 1, "top_p just below 1.0 must still cut");
    let mut disabled = tiny_tail.clone();
    top_p_filter(&mut disabled, 1.0);
    assert_eq!(disabled, tiny_tail, "the same input is untouched at exactly 1.0");

    // Degenerate inputs must not panic: a single logit has no tail to cut, and an
    // all-masked vector has no finite maximum to normalise against.
    let mut one = vec![3.0f32];
    top_p_filter(&mut one, 0.1);
    assert_eq!(one, vec![3.0]);
    let mut dead = vec![f32::NEG_INFINITY; 3];
    top_p_filter(&mut dead, 0.1);
    assert_eq!(survivors(&dead), 0);
}

// -------------------------------------------------------------- repetition penalty

/// HF's sign convention: a positive logit is DIVIDED and a negative one MULTIPLIED, so
/// both move toward `-inf`. Zero is on the positive branch and stays zero.
#[test]
fn the_repetition_penalty_pushes_both_signs_downward() {
    let mut l = vec![4.0f32, -4.0, 0.0, 9.0];
    apply_repetition_penalty(&mut l, &[0, 1, 2], 2.0);
    assert_eq!(l[0], 2.0, "positive divided");
    assert_eq!(l[1], -8.0, "negative multiplied");
    assert_eq!(l[2], 0.0, "zero is unchanged by either branch");
    assert_eq!(l[3], 9.0, "an id not in the history is untouched");
    assert!(l[0] < 4.0 && l[1] < -4.0, "both signs move toward -inf");
}

/// `1.0` is a no-op on both signs — and only exactly `1.0`.
#[test]
fn a_penalty_of_one_changes_nothing() {
    let mut l = vec![4.0f32, -4.0];
    apply_repetition_penalty(&mut l, &[0, 1], 1.0);
    assert_eq!(l, vec![4.0, -4.0]);

    let mut l = vec![4.0f32, -4.0];
    apply_repetition_penalty(&mut l, &[0, 1], 1.05);
    assert!(l[0] < 4.0 && l[1] < -4.0, "the shipped 1.05 is not a no-op");
}

/// HF gathers the original scores and scatters back, so a repeated id is penalised once —
/// not compounded per occurrence.
#[test]
fn a_repeated_id_is_penalised_exactly_once() {
    let mut once = vec![8.0f32];
    apply_repetition_penalty(&mut once, &[0], 2.0);
    let mut thrice = vec![8.0f32];
    apply_repetition_penalty(&mut thrice, &[0, 0, 0], 2.0);
    assert_eq!(thrice, once, "duplicates must collapse");
    assert_eq!(thrice[0], 4.0);

    // An empty history is a no-op, and an out-of-range id is ignored rather than a panic.
    let mut l = vec![8.0f32, 1.0];
    apply_repetition_penalty(&mut l, &[], 2.0);
    assert_eq!(l, vec![8.0, 1.0]);
    apply_repetition_penalty(&mut l, &[2, 99, 1], 2.0);
    assert_eq!(l, vec![8.0, 0.5], "only the in-range id moved");
}

// ------------------------------------------------------- suppress / min-new-tokens mask

/// `block_ids` is the one primitive behind both `SuppressTokensLogitsProcessor` and
/// `MinNewTokensLengthLogitsProcessor`. In range it masks; one past the end it is a no-op.
#[test]
fn block_ids_masks_in_range_and_ignores_out_of_range() {
    let mut l = ladder();
    block_ids(&mut l, &[3]);
    assert!(masked(&l, 3), "the last valid index is in range");
    assert_eq!(survivors(&l), 3);

    let mut l = ladder();
    block_ids(&mut l, &[4, 99]);
    assert_eq!(l, ladder(), "one past the end must be ignored, not panic");

    let mut l = ladder();
    block_ids(&mut l, &[]);
    assert_eq!(l, ladder());
}

/// The processor list the dual-AR loop composes, in HF's order: repetition penalty, then
/// the min-new-tokens EOS guard, then `suppress_tokens`, then the warpers. Both sides of
/// the guard: below the minimum the EOS is unreachable, at the minimum it is reachable.
#[test]
fn the_eos_guard_and_the_suppress_mask_bound_what_the_talker_can_draw() {
    // A 6-id codec vocabulary: 5 is the EOS, 4 is a suppressed control id.
    const EOS: u32 = 5;
    const SUPPRESS: [u32; 1] = [4];
    // Both control ids are the most likely draws, so only the masks can keep them out.
    let logits = vec![0.0f32, 0.0, 0.0, 0.0, 9.0, 9.0];
    let params = off();

    let step = |frames: usize, min_new: usize, s: &mut Sampler| -> u32 {
        let mut l = logits.clone();
        apply_repetition_penalty(&mut l, &[], params.repetition_penalty);
        if frames < min_new {
            block_ids(&mut l, &[EOS]);
        }
        block_ids(&mut l, &SUPPRESS);
        s.sample(&mut l, &params)
    };

    // Below the minimum (the reference pins min_new_tokens = 2): EOS is unreachable and
    // so is the suppressed id, so every draw is a real code.
    let mut s = Sampler::new(11);
    for frames in [0usize, 1] {
        for _ in 0..200 {
            let id = step(frames, 2, &mut s);
            assert!(id < 4, "frame {frames}: drew a masked control id {id}");
        }
    }

    // At the minimum the guard lifts: the EOS becomes reachable — and, with 9.0 against
    // 0.0, overwhelmingly likely — while the suppressed id stays masked forever.
    let mut s = Sampler::new(11);
    let got: Vec<u32> = (0..200).map(|_| step(2, 2, &mut s)).collect();
    assert!(got.contains(&EOS), "the EOS never became reachable at the minimum");
    assert!(!got.contains(&4), "a suppressed id was drawn");
}

// -------------------------------------------------------------------------- multinomial

/// The inverse-CDF draw must reproduce the categorical distribution, not merely stay in
/// range. Deterministic given the seed, so the bounds are assertions and not flakes.
#[test]
fn the_multinomial_follows_the_probability_vector() {
    let mut s = Sampler::new(2024);
    let probs = [0.25f32, 0.75];
    let n = 8_000;
    let ones = (0..n).filter(|_| s.multinomial(&probs) == 1).count();
    let share = ones as f64 / n as f64;
    assert!((share - 0.75).abs() < 0.02, "P(id = 1) came out {share}, expected ~0.75");

    // A point mass is drawn exactly, on either side.
    let mut s = Sampler::new(1);
    assert!((0..64).all(|_| s.multinomial(&[0.0, 1.0]) == 1));
    let mut s = Sampler::new(1);
    assert!((0..64).all(|_| s.multinomial(&[1.0, 0.0]) == 0));

    // An unnormalised vector is renormalised by its own total rather than assumed to sum
    // to one, so the same shape at a different scale gives the same distribution.
    let mut a = Sampler::new(9);
    let mut b = Sampler::new(9);
    let scaled: Vec<u32> = (0..500).map(|_| b.multinomial(&[2.0, 6.0])).collect();
    let unit: Vec<u32> = (0..500).map(|_| a.multinomial(&[0.25, 0.75])).collect();
    assert_eq!(scaled, unit);
}

/// A degenerate distribution must degrade safely rather than panic or index past the end —
/// it is reachable whenever every candidate has been masked.
#[test]
fn an_all_masked_distribution_degrades_safely() {
    let mut s = Sampler::new(0);
    assert_eq!(s.multinomial(&[0.0, 0.0, 0.0]), 0, "zero total falls back to id 0");
    assert_eq!(s.multinomial(&[]), 0, "an empty vector must not index");

    let mut l = vec![f32::NEG_INFINITY; 4];
    let mut s = Sampler::new(0);
    assert_eq!(s.sample(&mut l, &SamplingParams::talker()), 0);
}

// ----------------------------------------------------------------------------- defaults

/// The shipped `generation_config.json` values. The talker carries a repetition penalty
/// and the code predictor deliberately does not — `generate()` forwards only the
/// `subtalker_*` knobs, so the penalty falls back to the predictor's own `1.0`.
#[test]
fn the_defaults_are_the_published_generation_config() {
    let t = SamplingParams::talker();
    assert_eq!(t.temperature, 0.9);
    assert_eq!(t.top_p, 1.0);
    assert_eq!(t.top_k, 50);
    assert_eq!(t.repetition_penalty, 1.05);

    let c = SamplingParams::code_predictor();
    assert_eq!(c.temperature, t.temperature);
    assert_eq!(c.top_p, t.top_p);
    assert_eq!(c.top_k, t.top_k);
    assert_eq!(c.repetition_penalty, 1.0);
    assert_ne!(c.repetition_penalty, t.repetition_penalty, "the two heads differ here");

    assert_eq!(SamplingParams::default(), t, "the default head is the talker");

    // `top_p = 1.0` means the nucleus warper is OFF on both heads as shipped, and
    // `top_k = 50` means the top-k warper is ON. Pinned because flipping either default
    // silently changes every render.
    let mut l = ladder();
    top_p_filter(&mut l, t.top_p);
    assert_eq!(l, ladder(), "the shipped top_p installs no warper");
    assert!(t.top_k > 0, "the shipped top_k installs a warper");
}
