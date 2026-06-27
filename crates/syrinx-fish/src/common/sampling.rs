//! Seeded sampling for the dual-AR driver — temperature / top-p / top-k nucleus, an
//! optional repetition penalty, and Fish's **Repetition-Aware Sampling (RAS)** for the
//! semantic token, all on a deterministic `SplitMix64` PRNG (the CV-port sampler idiom,
//! so a run is bit-reproducible from a seed).
//!
//! Two draw paths, matching the reference `decode_one_token_ar`:
//!   * [`Sampler::sample_semantic`] — the **slow** AR token. Logits are first constrained
//!     to the semantic range + the stop id (the reference `semantic_logit_bias`), then RAS
//!     applies: draw a normal `(temperature, top_p)` token AND a high-`(temp, top_p)`
//!     token; if the normal token is a semantic id already seen in the recent window, use
//!     the high-temp token instead.
//!   * [`Sampler::sample_codebook`] — each **fast** AR residual code. Plain
//!     `(temperature, top_p, top_k)` nucleus over the residual codebook (no constraint),
//!     with the optional repetition penalty applied first.
//!
//! Pure Rust (no Candle): the backend converts its logit `Tensor` to a host `Vec<f32>`
//! and hands it here.

/// Deterministic SplitMix64 PRNG — pins the otherwise-stochastic multinomial draws so a
/// `drive` run is bit-reproducible from a seed. `next_f64` yields a uniform in `[0, 1)`.
/// Identical algorithm to the CV-port sampler.
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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Sampling knobs. Defaults are the Fish reference values (`generate_long` CLI defaults +
/// the RAS constants `RAS_WIN_SIZE` / `RAS_HIGH_TEMP` / `RAS_HIGH_TOP_P`).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature for the normal draw.
    pub temperature: f32,
    /// Nucleus (top-p) cumulative-probability cutoff.
    pub top_p: f32,
    /// Top-k cap on the candidate set (`0` = unbounded).
    pub top_k: usize,
    /// Repetition penalty (`1.0` = off) applied to logits of recently emitted ids.
    pub repetition_penalty: f32,
    /// RAS look-back window (frames) for the semantic-token repeat check.
    pub ras_win_size: usize,
    /// RAS fallback temperature (used when the normal semantic token repeats).
    pub ras_high_temp: f32,
    /// RAS fallback top-p.
    pub ras_high_top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // PARITY: these are the reference `generate_long` defaults (top_p 0.9, top_k 30,
        // temperature 1.0, repetition_penalty 1.1) + RAS constants (win 10, high_temp 1.0,
        // high_top_p 0.9). Confirm the production preset on-box.
        Self {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 30,
            repetition_penalty: 1.1,
            ras_win_size: 10,
            ras_high_temp: 1.0,
            ras_high_top_p: 0.9,
        }
    }
}

/// The slow-AR semantic constraint: only the contiguous semantic id range
/// `[begin, end]` plus the stop id may be drawn (the reference `semantic_logit_bias`,
/// which masks everything else to `-inf`).
#[derive(Debug, Clone, Copy)]
pub struct SemanticConstraint {
    /// First allowed semantic id.
    pub begin: u32,
    /// Last allowed semantic id (inclusive).
    pub end: u32,
    /// The `<|im_end|>` stop id (also allowed).
    pub stop: u32,
}

impl SemanticConstraint {
    fn is_semantic(&self, id: u32) -> bool {
        id >= self.begin && id <= self.end
    }
    /// Mask `logits` in place to `-inf` outside `[begin, end] ∪ {stop}`.
    fn apply(&self, logits: &mut [f32]) {
        for (i, l) in logits.iter_mut().enumerate() {
            let id = i as u32;
            if !(self.is_semantic(id) || id == self.stop) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

/// Seeded sampler bundling the PRNG and the sampling knobs. One instance drives a whole
/// utterance (slow + fast draws share the stream, matching the reference).
pub struct Sampler {
    rng: SplitMix64,
    params: SamplingParams,
}

impl Sampler {
    /// New sampler from a seed and knobs.
    pub fn new(seed: u64, params: SamplingParams) -> Self {
        Self {
            rng: SplitMix64::new(seed),
            params,
        }
    }

    /// The configured knobs.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Sample one **fast-AR residual code** from raw `logits` (full residual codebook),
    /// applying the repetition penalty over `recent` first, then `(temperature, top_p,
    /// top_k)` nucleus. No semantic constraint (the reference draws fast codebooks freely).
    pub fn sample_codebook(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        let mut work = logits.to_vec();
        apply_repetition_penalty(&mut work, recent, self.params.repetition_penalty);
        let probs = logits_to_probs(&work, self.params.temperature, self.params.top_p, self.params.top_k);
        multinomial1(&probs, &mut self.rng) as u32
    }

    /// Sample one **slow-AR semantic token** from raw `logits`. Applies `constraint`
    /// (semantic range + stop), then Fish RAS: a normal draw plus a high-temp draw, using
    /// the high-temp token iff the normal token is a semantic id present in `window` (the
    /// recently emitted semantic tokens).
    pub fn sample_semantic(
        &mut self,
        logits: &[f32],
        window: &[u32],
        constraint: &SemanticConstraint,
    ) -> u32 {
        let mut constrained = logits.to_vec();
        constraint.apply(&mut constrained);

        let probs_normal =
            logits_to_probs(&constrained, self.params.temperature, self.params.top_p, self.params.top_k);
        let normal = multinomial1(&probs_normal, &mut self.rng) as u32;

        let probs_high =
            logits_to_probs(&constrained, self.params.ras_high_temp, self.params.ras_high_top_p, self.params.top_k);
        let high = multinomial1(&probs_high, &mut self.rng) as u32;

        // RAS fallback: only when the normal token is semantic AND already in the window.
        let in_window = window.iter().any(|&t| t == normal);
        if in_window && constraint.is_semantic(normal) {
            high
        } else {
            normal
        }
    }
}

/// Penalize the logits of ids in `recent`: divide positive logits by `penalty`, multiply
/// negative logits by `penalty` (the HF `repetition_penalty` convention). No-op at `1.0`.
fn apply_repetition_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    for &id in recent {
        let i = id as usize;
        if i < logits.len() {
            let l = logits[i];
            logits[i] = if l > 0.0 { l / penalty } else { l * penalty };
        }
    }
}

/// Convert `logits` to a probability vector via the reference nucleus recipe: sort
/// descending, cumulative-softmax, drop tokens past `top_p` or beyond `top_k` (always
/// keeping the top-1), divide surviving logits by `temperature`, softmax. Returns probs
/// aligned to the original index order (dropped ids have probability 0).
fn logits_to_probs(logits: &[f32], temperature: f32, top_p: f32, top_k: usize) -> Vec<f32> {
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // Cumulative softmax over the sorted logits (for the top-p cutoff).
    let max = order.first().map(|&i| logits[i]).unwrap_or(0.0);
    let exps: Vec<f64> = order
        .iter()
        .map(|&i| ((logits[i] - max) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();

    let k_cap = if top_k == 0 { n } else { top_k };
    let mut keep = vec![false; n];
    let mut cum = 0f64;
    for (rank, &i) in order.iter().enumerate() {
        // Always keep the top-1; otherwise stop once cum-prob exceeds top_p or rank hits k.
        if rank == 0 {
            keep[i] = true;
        } else if cum <= top_p as f64 && rank < k_cap {
            keep[i] = true;
        } else {
            break;
        }
        cum += exps[rank] / sum;
    }

    // Temperature-scaled softmax over the kept ids only.
    let t = temperature.max(1e-5);
    let mut kept_max = f32::NEG_INFINITY;
    for i in 0..n {
        if keep[i] {
            kept_max = kept_max.max(logits[i] / t);
        }
    }
    let mut probs = vec![0f32; n];
    let mut psum = 0f64;
    for i in 0..n {
        if keep[i] {
            let e = ((logits[i] / t - kept_max) as f64).exp();
            probs[i] = e as f32;
            psum += e;
        }
    }
    if psum > 0.0 {
        for p in probs.iter_mut() {
            *p = (*p as f64 / psum) as f32;
        }
    }
    probs
}

/// Inverse-CDF sample one index from `probs` (need not be normalised) on a single uniform
/// draw — the deterministic analogue of `torch.multinomial(probs, 1)`.
///
/// PARITY: the reference `multinomial_sample_one_no_sync` uses a Gumbel-max draw
/// (`argmax(probs / -log(rand))`). This inverse-CDF draw samples the **same categorical
/// distribution**; only the RNG stream differs (a stochastic sampler is never bit-exact
/// across PRNGs anyway). Confirm distribution-level agreement on-box.
fn multinomial1(probs: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f64 = probs.iter().map(|&p| p as f64).sum();
    if total <= 0.0 {
        return 0;
    }
    let u = rng.next_f64() * total;
    let mut acc = 0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p as f64;
        if u < acc {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}
