//! A real, **pure-Rust, training-free spread-spectrum audio watermark** for the
//! 24 kHz mono `f32` output of the [`Synthesizer`](crate::synth::Synthesizer).
//!
//! # What this is
//!
//! A direct-sequence spread-spectrum (DSSS / CDMA-style) watermark. A 16-bit
//! payload is spread across the whole waveform by a key-derived pseudo-random
//! `±1` chip sequence and added at a low, fixed amplitude (default
//! [`DEFAULT_AMP`] ≈ `4e-3`, about −48 dBFS). The waveform is essentially
//! unchanged: the per-sample perturbation magnitude is **exactly** the embed
//! amplitude (one chip per sample), so `max |Δ| = amp` and `rms(Δ) = amp`.
//!
//! Detection correlates the candidate audio against the same key-derived chips.
//! The watermark adds **coherently** (every block carries the same chips) while
//! the host audio is incoherent with the key, so a matched-filter correlation
//! recovers the payload with a processing gain of roughly `√(N/16)` (number of
//! chips per payload bit). The reported confidence is a **z-score**: the
//! correlation expressed in units of the host-noise standard deviation. A clean
//! or wrong-key signal scores ≈ 1; a present watermark scores well above
//! [`PRESENT_Z_THRESHOLD`].
//!
//! # Honest robustness boundary
//!
//! This is a *modest, well-characterized* watermark, not an adversarial one.
//!
//! It **survives** (because it is a low-amplitude, redundant, correlation-recovered
//! signal, and detection is block-folded + amplitude/gain-normalized):
//!   * lossless / high-bitrate re-encoding (16-bit PCM round-trip, high-rate codecs),
//!   * a small linear gain change (the z-score is scale-invariant),
//!   * light additive noise / dither down to a modest SNR,
//!   * cropping / trimming (a key-block sync search re-aligns the chip phase).
//!
//! It does **NOT** survive (and we do not claim it does):
//!   * aggressive lossy compression (low-bitrate MP3/Opus, which reshapes the very
//!     low-level spectral content the watermark lives in),
//!   * time-stretch / pitch-shift / resampling to a different rate (these warp the
//!     chip timing; only integer-sample crops are handled),
//!   * deliberate adversarial removal (denoise-then-re-add, spectral subtraction,
//!     or an attacker who knows the scheme).
//!
//! Those threat models need a **learned**, perceptually-masked watermark such as
//! AudioSeal/WavMark. This module is the training-free baseline: detectable on the
//! unmodified output and robust to *light* post-editing, stated plainly.

/// Period of the pseudo-random chip sequence, in samples. The chip sequence
/// repeats every `BLOCK` samples, which (a) lets the detector fold many blocks
/// together for processing gain and (b) makes a crop recoverable by searching the
/// `BLOCK` possible phase offsets. Must be a multiple of [`PAYLOAD_BITS`].
pub const BLOCK: usize = 1024;

/// Payload width in bits (`u16`).
pub const PAYLOAD_BITS: usize = 16;

/// Default embed amplitude: the magnitude of the `±amp` chip added to every
/// sample. ≈ `4e-3` is about −48 dBFS — below the perceptual threshold for
/// speech of typical level yet well above the quantization/noise floor of 16-bit
/// PCM, giving a healthy detection margin over a few seconds of audio.
pub const DEFAULT_AMP: f32 = 4.0e-3;

/// Detection threshold on the reported z-score. A clean or wrong-key signal
/// scores ≈ 1 (the null distribution is ~`N(0,1)` per bit); a present watermark
/// over a few seconds scores well above this. 3.0 sits ~11σ above the null mean,
/// so the false-positive rate is negligible while a real watermark clears it
/// comfortably.
pub const PRESENT_Z_THRESHOLD: f64 = 3.0;

/// The outcome of [`detect_watermark`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WmResult {
    /// Whether a watermark for the given key is judged present
    /// (`confidence >= PRESENT_Z_THRESHOLD`).
    pub present: bool,
    /// Detection confidence as an RMS z-score across the 16 payload bits: the
    /// correlation peak in units of the host-noise standard deviation. ≈ 1 means
    /// absent (consistent with chance); ≫ 1 means present.
    pub confidence: f64,
    /// The recovered 16-bit payload (only meaningful when `present`).
    pub payload: u16,
}

/// SplitMix64 — a small, fast, well-distributed seeded PRNG. Deterministic from
/// `state`; used to derive the `±1` chip sequence from the watermark key. (Not a
/// CSPRNG, and not meant to be — the key is a shared secret for detection, not a
/// cryptographic guarantee.)
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive the deterministic `±1` chip sequence (length [`BLOCK`]) from `key`.
///
/// The same sequence is produced for embed and detect, so the key is the shared
/// secret that ties the two together. A different key yields an (almost surely)
/// uncorrelated sequence, which is what makes a wrong-key detection fail.
fn chips_for_key(key: u64) -> [f32; BLOCK] {
    // Mix the key with a domain-separation constant so `key = 0` is not a
    // degenerate all-from-zero seed.
    let mut state = key ^ 0xD1B5_4A32_D192_ED03;
    let mut chips = [0f32; BLOCK];
    for c in chips.iter_mut() {
        // Use the top bit of each draw for an unbiased ±1.
        *c = if (splitmix64(&mut state) >> 63) == 1 {
            1.0
        } else {
            -1.0
        };
    }
    chips
}

/// Map payload bit `b` (0-based, LSB first) to its `±1` sign.
#[inline]
fn bit_sign(payload: u16, b: usize) -> f32 {
    if (payload >> b) & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}

/// Embed a [`PAYLOAD_BITS`]-bit `payload` into `audio` in place at the default
/// amplitude ([`DEFAULT_AMP`]), keyed by `key`.
///
/// The waveform is perturbed by exactly `±DEFAULT_AMP` per sample (spread-spectrum
/// chips), so `max |Δ| = rms(Δ) = DEFAULT_AMP`. See [`embed_watermark_with_amp`]
/// to choose the amplitude (e.g. to trade imperceptibility against robustness).
pub fn embed_watermark(audio: &mut [f32], key: u64, payload: u16) {
    embed_watermark_with_amp(audio, key, payload, DEFAULT_AMP);
}

/// Embed `payload` into `audio` in place at an explicit `amp`, keyed by `key`.
///
/// Sample `p` is perturbed by `amp · sign(payload bit b) · chip[p mod BLOCK]`,
/// where `b = (p mod BLOCK) mod PAYLOAD_BITS`. The result is clamped to `[-1, 1]`
/// so the watermark can never push a sample out of range.
pub fn embed_watermark_with_amp(audio: &mut [f32], key: u64, payload: u16, amp: f32) {
    if audio.is_empty() || amp == 0.0 {
        return;
    }
    let chips = chips_for_key(key);
    for (p, s) in audio.iter_mut().enumerate() {
        let r = p % BLOCK;
        let b = r % PAYLOAD_BITS;
        let delta = amp * bit_sign(payload, b) * chips[r];
        *s = (*s + delta).clamp(-1.0, 1.0);
    }
}

/// Detect a watermark for `key` in `audio`, recovering the payload.
///
/// Returns `present = confidence >= PRESENT_Z_THRESHOLD`. The detector folds the
/// signal over the chip period [`BLOCK`], then searches the `BLOCK` possible chip
/// phase offsets (so an integer-sample crop is re-aligned) and reports the best
/// match. The confidence is an RMS z-score across the payload bits — a clean or
/// wrong-key signal scores ≈ 1, a present watermark scores well above threshold.
///
/// Returns `present = false`, `confidence = 0`, `payload = 0` for an empty or
/// silent (zero-energy) input.
pub fn detect_watermark(audio: &[f32], key: u64) -> Option<WmResult> {
    if audio.is_empty() {
        return Some(WmResult {
            present: false,
            confidence: 0.0,
            payload: 0,
        });
    }

    // Host-noise scale estimate (RMS). The watermark contribution is tiny, so this
    // is ~the host level; it normalizes the correlation into a z-score and makes
    // the statistic invariant to a linear gain change.
    let sumsq: f64 = audio.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let sigma = (sumsq / audio.len() as f64).sqrt();
    if sigma <= 0.0 {
        return Some(WmResult {
            present: false,
            confidence: 0.0,
            payload: 0,
        });
    }

    let chips = chips_for_key(key);

    // Fold the signal onto the BLOCK residues: fold[j] = Σ audio[p] over p ≡ j,
    // cnt[j] = count of such p. Folding aligned blocks is the matched filter for a
    // block-periodic chip sequence and gives the processing gain.
    let mut fold = [0f64; BLOCK];
    let mut cnt = [0u32; BLOCK];
    for (p, &x) in audio.iter().enumerate() {
        let j = p % BLOCK;
        fold[j] += x as f64;
        cnt[j] += 1;
    }

    // Search every chip phase offset (handles an integer-sample crop) and keep the
    // strongest. At offset `s`, fold-bin `j` carries original residue (j+s) % BLOCK.
    let mut best_score = f64::NEG_INFINITY;
    let mut best_payload: u16 = 0;
    for s in 0..BLOCK {
        let mut raw = [0f64; PAYLOAD_BITS];
        let mut nb = [0u64; PAYLOAD_BITS];
        for j in 0..BLOCK {
            let orig = (j + s) % BLOCK;
            let b = orig % PAYLOAD_BITS;
            raw[b] += fold[j] * chips[orig] as f64;
            nb[b] += cnt[j] as u64;
        }
        // Per-bit z-score: correlation / (σ · √Nb). Under the host-only null each
        // is ~N(0,1); the watermark biases it to ±(amp·√Nb / σ).
        let mut sumz2 = 0.0f64;
        let mut payload: u16 = 0;
        for b in 0..PAYLOAD_BITS {
            let z = if nb[b] > 0 {
                raw[b] / (sigma * (nb[b] as f64).sqrt())
            } else {
                0.0
            };
            sumz2 += z * z;
            if z > 0.0 {
                payload |= 1 << b;
            }
        }
        let score = (sumz2 / PAYLOAD_BITS as f64).sqrt();
        if score > best_score {
            best_score = score;
            best_payload = payload;
        }
    }

    Some(WmResult {
        present: best_score >= PRESENT_Z_THRESHOLD,
        confidence: best_score,
        payload: best_payload,
    })
}
