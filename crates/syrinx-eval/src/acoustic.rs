//! Acoustic features and the two-sample test that decides whether a cue **activated**.
//!
//! ## The problem this module exists to solve
//!
//! [`crate::activation`] consumes [`Measurement`](crate::activation::Measurement)s whose
//! `activated` flag answers "did the cue measurably change the audio versus the un-cued
//! control?". Answering that honestly is harder than it looks, and getting it wrong in
//! either direction produces a number that certifies nothing:
//!
//! * **Comparing one cued render against one un-cued render is meaningless.** The dual-AR
//!   loop samples, so two renders of the *same* text with different seeds already differ in
//!   every sample. Any "did the waveform change" test on a single pair answers `true`
//!   always, which would report 100 % activation for a backend that ignores cues entirely.
//! * **Bit-comparison at a fixed seed is equally meaningless in the other direction.**
//!   Prepending a cue changes the prompt, so the token stream diverges whether or not the
//!   model attaches any meaning to the cue. That also answers `true` always.
//!
//! So the question is not "is there a difference" but "is the difference **larger than this
//! model's own run-to-run variation**". That requires several renders per condition and a
//! statistical test, which is what this module implements.
//!
//! ## The test
//!
//! Render the cued text `n` times and the un-cued text `n` times, each with a different
//! [`DriveParams::seed`](https://docs.rs) — the loop is bit-reproducible per seed, so the
//! spread across seeds *is* the model's sampling noise, measured rather than assumed.
//! Summarize each render as a [`Features`] vector, then run an **exact two-sample
//! permutation test** on the distance between condition means.
//!
//! Permutation is the right family here: it is distribution-free (no normality assumption
//! about F0 or pause structure, which are visibly non-normal), exact at these sample sizes
//! rather than asymptotic, and it needs no variance model beyond the data in front of it.
//! With `n = 4` there are `C(8,4)/2 = 35` distinct labelings, so the smallest attainable
//! p-value is `1/35 ≈ 0.029` — below the conventional `α = 0.05`, which is the minimum `n`
//! at which the test can reject at all. [`min_n_for_alpha`] enforces that, because a test
//! that *cannot* reject silently reports zero activation for a working backend.
//!
//! Standardization is deliberately computed over the **pooled** set of all `2n` renders,
//! ignoring the condition labels. Standardizing per-condition would leak the labeling into
//! the statistic and break the exactness of the permutation null.
//!
//! ## What this does and does not claim
//!
//! It measures that the cue moved the acoustics beyond sampling noise. It does **not**
//! claim the audio sounds like the cue asked for: "does `[sad]` sound sad" is a perceptual
//! judgement, and per `CLAUDE.md` those are out of scope for any automated gate here. A
//! backend that reacts to `[sad]` by shouting would pass this test, and should — it did
//! react. Direction and appropriateness are a human listening call, deliberately not
//! automated into a green.

use std::f64::consts::PI;

/// Names of the [`Features`] dimensions, in vector order.
pub const FEATURE_NAMES: [&str; 11] = [
    "duration_s",
    "rms_mean",
    "rms_std",
    "pause_ratio",
    "pause_rate_hz",
    "f0_mean",
    "f0_std",
    "voiced_ratio",
    "centroid_mean",
    "centroid_std",
    "spectral_tilt",
];

/// Number of feature dimensions.
pub const N_FEATURES: usize = FEATURE_NAMES.len();

/// Analysis frame length in seconds. 46 ms at 44.1 kHz is 2048 samples — long enough that
/// the autocorrelation still resolves a 60 Hz f0 (lag 735) inside the reliable first half
/// of the lag axis, short enough that prosodic change is not averaged away.
const FRAME_SECONDS: f64 = 0.046;

/// Hop as a fraction of the frame (50 % overlap).
const HOP_FRACTION: f64 = 0.5;

/// f0 search range in Hz. Wide enough for a low male voice through an excited high female
/// one; anything outside it is far likelier to be a period-doubling artifact than a voice.
const F0_MIN_HZ: f64 = 60.0;
const F0_MAX_HZ: f64 = 400.0;

/// Normalized-autocorrelation peak below which a frame is called unvoiced. 0.3 is the
/// conventional voicing floor for ACF trackers; below it the "period" is usually noise.
const VOICING_THRESHOLD: f64 = 0.3;

/// A frame counts as silence when its RMS is below this fraction of the utterance's peak
/// frame RMS. Relative rather than absolute because overall level varies per render, and
/// an absolute floor would score a quiet render as one long pause.
const SILENCE_REL: f64 = 0.08;

/// Boundary between the "low" and "high" band of the spectral tilt measure, in Hz.
const TILT_SPLIT_HZ: f64 = 1000.0;

/// A fixed-length acoustic summary of one rendered utterance.
///
/// The dimensions are chosen to span the axes cues actually move: **rate and pausing**
/// (`duration_s`, `pause_ratio`, `pause_rate_hz`), **loudness and its dynamics**
/// (`rms_mean`, `rms_std`), **pitch** (`f0_mean`, `f0_std`, `voiced_ratio`) and **timbre**
/// (`centroid_mean`, `centroid_std`, `spectral_tilt`). A whispered line moves tilt and
/// voicing; a shouted one moves RMS and f0; `[sigh]` or `[breath]` moves pause structure
/// and duration. No single scalar covers that, which is why the test below is multivariate.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub v: [f64; N_FEATURES],
}

impl Features {
    /// Look up one dimension by its name in [`FEATURE_NAMES`].
    pub fn get(&self, name: &str) -> Option<f64> {
        FEATURE_NAMES.iter().position(|n| *n == name).map(|i| self.v[i])
    }
}

/// Extract the acoustic summary of one render.
///
/// `samples` is mono; `sample_rate` is its rate in Hz (Fish s2-pro emits 44.1 kHz — the
/// 2048-sample codec hop at the 21.5 Hz frame rate — so nothing here may assume 24 kHz).
/// An empty or all-silent input yields an all-zero vector rather than a NaN: a render that
/// produced nothing is a legitimate observation the test should see, not an error.
pub fn features(samples: &[f32], sample_rate: u32) -> Features {
    let sr = f64::from(sample_rate.max(1));
    let mut v = [0.0f64; N_FEATURES];
    if samples.is_empty() {
        return Features { v };
    }

    let duration_s = samples.len() as f64 / sr;
    v[0] = duration_s;

    let frame_len = next_pow2(((FRAME_SECONDS * sr) as usize).max(64));
    let hop = ((frame_len as f64) * HOP_FRACTION) as usize;
    if samples.len() < frame_len {
        // Too short to frame. Duration and a whole-signal RMS are still honest; the
        // frame-derived dimensions stay zero.
        v[1] = rms(samples);
        return Features { v };
    }

    let n_frames = (samples.len() - frame_len) / hop + 1;
    let window = hann(frame_len);

    let mut rms_per_frame = Vec::with_capacity(n_frames);
    let mut f0_per_frame = Vec::with_capacity(n_frames);
    let mut centroid_per_frame = Vec::with_capacity(n_frames);
    let mut tilt_lo = 0.0f64;
    let mut tilt_hi = 0.0f64;

    // Transform length is twice the frame, with the frame in the first half and zeros
    // after. The zero-padding is what makes the inverse transform below a *linear*
    // autocorrelation: an unpadded transform gives the circular one, whose wraparound
    // biases exactly the long lags that carry the low-f0 answer.
    let fft_len = frame_len * 2;
    let mut re = vec![0.0f64; fft_len];
    let mut im = vec![0.0f64; fft_len];
    let mut power = vec![0.0f64; fft_len];

    for f in 0..n_frames {
        let frame = &samples[f * hop..f * hop + frame_len];
        rms_per_frame.push(rms(frame));

        re[..frame_len]
            .iter_mut()
            .zip(frame.iter().zip(window.iter()))
            .for_each(|(dst, (x, w))| *dst = f64::from(*x) * w);
        re[frame_len..].fill(0.0);
        im.fill(0.0);
        fft(&mut re, &mut im, false);

        for i in 0..fft_len {
            power[i] = re[i] * re[i] + im[i] * im[i];
        }

        // Spectral centroid and the low/high band split, over the non-redundant half.
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..fft_len / 2 {
            let hz = i as f64 * sr / fft_len as f64;
            let p = power[i];
            num += hz * p;
            den += p;
            if hz < TILT_SPLIT_HZ {
                tilt_lo += p;
            } else {
                tilt_hi += p;
            }
        }
        centroid_per_frame.push(if den > 0.0 { num / den } else { 0.0 });

        // Autocorrelation as the inverse transform of the power spectrum (Wiener-
        // Khinchin). Reusing the spectrum already computed costs one extra transform
        // instead of an O(lags x frame) time-domain loop.
        let mut ar = power.clone();
        let mut ai = vec![0.0f64; fft_len];
        fft(&mut ar, &mut ai, true);
        f0_per_frame.push(acf_f0(&ar, sr, frame_len));
    }

    let peak = rms_per_frame.iter().copied().fold(0.0f64, f64::max);
    let floor = peak * SILENCE_REL;

    v[1] = mean(&rms_per_frame);
    v[2] = std_dev(&rms_per_frame);

    let silent: Vec<bool> = rms_per_frame.iter().map(|r| *r <= floor).collect();
    let n_silent = silent.iter().filter(|s| **s).count();
    v[3] = n_silent as f64 / n_frames as f64;
    // Pause *runs* per second, not raw silent frames: one long pause and many scattered
    // short ones have very different prosodic meaning but the same pause_ratio.
    let runs = silent.iter().enumerate().filter(|(i, s)| **s && (*i == 0 || !silent[i - 1])).count();
    v[4] = runs as f64 / duration_s.max(1e-9);

    let voiced: Vec<f64> = f0_per_frame.iter().copied().filter(|f| *f > 0.0).collect();
    v[5] = mean(&voiced);
    v[6] = std_dev(&voiced);
    v[7] = voiced.len() as f64 / n_frames as f64;

    v[8] = mean(&centroid_per_frame);
    v[9] = std_dev(&centroid_per_frame);
    // Log ratio so a doubling up and a halving down are symmetric distances.
    v[10] = ((tilt_hi + 1e-12) / (tilt_lo + 1e-12)).ln();

    Features { v }
}

/// Outcome of the activation test for one case.
#[derive(Debug, Clone, PartialEq)]
pub struct TestOutcome {
    /// Did the cue move the acoustics beyond this model's own sampling noise?
    pub activated: bool,
    /// Exact permutation p-value.
    pub p_value: f64,
    /// Observed between-condition distance, in pooled standard deviations.
    pub effect: f64,
    /// Number of distinct labelings enumerated (the resolution of `p_value`).
    pub n_permutations: usize,
}

/// The smallest per-condition `n` whose exact permutation test can reject at `alpha`.
///
/// The test enumerates `C(2n, n) / 2` distinct labelings, so the smallest attainable
/// p-value is its reciprocal. Below this `n` the test can never reject and would report a
/// working backend as 0 % activated — a false red that looks exactly like a real one.
pub fn min_n_for_alpha(alpha: f64) -> usize {
    for n in 1..=12 {
        if 1.0 / (n_splits(n) as f64) <= alpha {
            return n;
        }
    }
    usize::MAX
}

/// Exact two-sample permutation test on the distance between condition means.
///
/// Returns `None` when the groups are unequal, empty, or too small to reject at `alpha` —
/// all three are caller errors that must surface rather than silently become `false`.
pub fn activation_test(cued: &[Features], uncued: &[Features], alpha: f64) -> Option<TestOutcome> {
    let n = cued.len();
    if n == 0 || uncued.len() != n || n < min_n_for_alpha(alpha) {
        return None;
    }

    // Pool both conditions, then standardize each dimension over the pool. Label-agnostic
    // by construction, which is what keeps the permutation null exact.
    let mut pool: Vec<[f64; N_FEATURES]> = Vec::with_capacity(2 * n);
    pool.extend(cued.iter().map(|f| f.v));
    pool.extend(uncued.iter().map(|f| f.v));

    let mut scale = [0.0f64; N_FEATURES];
    for d in 0..N_FEATURES {
        let col: Vec<f64> = pool.iter().map(|r| r[d]).collect();
        scale[d] = std_dev(&col);
    }
    for row in pool.iter_mut() {
        for d in 0..N_FEATURES {
            // A dimension with no spread across the pool carries no information and is
            // dropped, rather than exploding to infinity on a near-zero denominator.
            row[d] = if scale[d] > 1e-12 { row[d] / scale[d] } else { 0.0 };
        }
    }

    let observed = split_distance(&pool, &first_half(n));
    let mut n_perm = 0usize;
    let mut at_least = 0usize;
    for combo in combinations(2 * n, n) {
        // Each split and its complement give the same distance; canonicalize on splits
        // containing index 0 so every labeling is counted exactly once.
        if !combo.contains(&0) {
            continue;
        }
        n_perm += 1;
        if split_distance(&pool, &combo) >= observed - 1e-12 {
            at_least += 1;
        }
    }

    let p_value = at_least as f64 / n_perm as f64;
    Some(TestOutcome {
        activated: p_value <= alpha,
        p_value,
        effect: observed,
        n_permutations: n_perm,
    })
}

/// `C(2n, n) / 2` — the number of distinct labelings the exact test enumerates.
fn n_splits(n: usize) -> usize {
    let mut c: u128 = 1;
    for i in 0..n {
        c = c * (2 * n - i) as u128 / (i + 1) as u128;
    }
    (c / 2) as usize
}

fn first_half(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// Euclidean distance between the mean of `group` and the mean of its complement.
fn split_distance(pool: &[[f64; N_FEATURES]], group: &[usize]) -> f64 {
    let mut a = [0.0f64; N_FEATURES];
    let mut b = [0.0f64; N_FEATURES];
    let mut na = 0.0;
    let mut nb = 0.0;
    for (i, row) in pool.iter().enumerate() {
        if group.contains(&i) {
            for d in 0..N_FEATURES {
                a[d] += row[d];
            }
            na += 1.0;
        } else {
            for d in 0..N_FEATURES {
                b[d] += row[d];
            }
            nb += 1.0;
        }
    }
    let mut sum = 0.0;
    for d in 0..N_FEATURES {
        let diff = a[d] / na - b[d] / nb;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// All `k`-subsets of `0..n`, in lexicographic order.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

// ---- small numeric helpers --------------------------------------------------------

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    (s / x.len() as f64).sqrt()
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

fn std_dev(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()).collect()
}

/// Pick the f0 of one frame from its autocorrelation, or 0.0 if the frame is unvoiced.
fn acf_f0(acf: &[f64], sr: f64, frame_len: usize) -> f64 {
    let zero = acf[0];
    if zero <= 0.0 {
        return 0.0;
    }
    let min_lag = (sr / F0_MAX_HZ) as usize;
    let max_lag = ((sr / F0_MIN_HZ) as usize).min(frame_len - 1);
    if min_lag >= max_lag {
        return 0.0;
    }
    let mut best = min_lag;
    let mut best_v = f64::MIN;
    for lag in min_lag..max_lag {
        if acf[lag] > best_v {
            best_v = acf[lag];
            best = lag;
        }
    }
    if best_v / zero < VOICING_THRESHOLD {
        return 0.0;
    }
    sr / best as f64
}

/// In-place iterative radix-2 FFT. `inverse` divides by `n` on the way out, so a forward
/// transform followed by an inverse one is the identity.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(im.len(), n);

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                    re[i + k + len / 2] * ci + im[i + k + len / 2] * cr,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        for i in 0..n {
            re[i] /= n as f64;
            im[i] /= n as f64;
        }
    }
}
