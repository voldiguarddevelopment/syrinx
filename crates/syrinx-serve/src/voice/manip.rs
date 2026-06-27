//! Embedding-space **voice manipulation**: blend, interpolate (LERP/SLERP), voice
//! arithmetic and a lightweight attribute hook — all operating on the CAM++ speaker
//! embedding (the timbre identity) and all returning a new [`Voice`].
//!
//! Every op works on the embedding only and carries a chosen **base voice**'s
//! `prompt_feat` / `prompt_token` / `prompt_text` (those are clip-tied sequences that
//! cannot be averaged across clips — see the module-level docs). The CAM++ embedding is a
//! **cosine**-compared vector, so operands are L2-normalized and the result is
//! re-normalized to the unit sphere.

use candle_core::{Device, Tensor};

use super::Voice;
use crate::synth::SynthError;

/// L2-normalize a flat embedding vector. A zero vector is returned unchanged (no NaNs).
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm <= f64::MIN_POSITIVE {
        return v.to_vec();
    }
    v.iter().map(|&x| (x as f64 / norm) as f32).collect()
}

/// Cosine similarity of two equal-length embedding vectors (0 if either is degenerate).
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na <= f64::MIN_POSITIVE || nb <= f64::MIN_POSITIVE {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Rebuild a `[1, n]` f32 embedding tensor from a flat vector on `dev`.
fn emb_tensor(v: Vec<f32>, dev: &Device) -> Result<Tensor, SynthError> {
    let n = v.len();
    Ok(Tensor::from_vec(v, (1, n), dev)?)
}

impl Voice {
    /// Clone `self` but swap in a new speaker embedding (flat, length 192), keeping the
    /// base clip's `prompt_feat` / `prompt_token` / `prompt_text`. The shared tail of every
    /// manipulation op. The new embedding is placed on the base embedding's device.
    fn with_embedding(&self, emb: Vec<f32>, name: String) -> Result<Voice, SynthError> {
        let dev = self.speaker_embedding.device().clone();
        Ok(Voice {
            name,
            speaker_embedding: emb_tensor(emb, &dev)?,
            prompt_feat: self.prompt_feat.clone(),
            prompt_token: self.prompt_token.clone(),
            prompt_text: self.prompt_text.clone(),
            source: None,
        })
    }

    /// **Blend** several voices by a weighted, L2-normalized average of their speaker
    /// embeddings (the result re-normalized to the unit sphere).
    ///
    /// Each operand embedding is L2-normalized first (CAM++ is cosine-compared, so raw
    /// magnitudes are not comparable), the weights are normalized to sum to 1, the
    /// normalized embeddings are summed by those weights, and the result is L2-normalized.
    /// Negative weights are allowed (a "push away from" term); the only error is an empty
    /// slice or weights that sum to ~0.
    ///
    /// The result **carries the first voice's** `prompt_feat` / `prompt_token` /
    /// `prompt_text` (the clip-tied conditioning — see the module docs) and is named
    /// `"blend"`. Use [`Voice::with_name`] / [`Voice::with_source`] to relabel.
    pub fn blend(voices: &[(&Voice, f32)]) -> Result<Voice, SynthError> {
        let (base, _) = *voices.first().ok_or_else(|| {
            SynthError::Candle("Voice::blend: needs at least one voice".to_string())
        })?;
        let wsum: f32 = voices.iter().map(|&(_, w)| w).sum();
        if wsum.abs() <= f32::EPSILON {
            return Err(SynthError::Candle(
                "Voice::blend: weights sum to ~0 (cannot normalize)".to_string(),
            ));
        }
        let dim = base.embedding_vec()?.len();
        let mut acc = vec![0f64; dim];
        for &(v, w) in voices {
            let e = l2_normalize(&v.embedding_vec()?);
            if e.len() != dim {
                return Err(SynthError::Candle(format!(
                    "Voice::blend: embedding dim mismatch ({} vs {dim})",
                    e.len()
                )));
            }
            let wn = (w / wsum) as f64;
            for (a, &x) in acc.iter_mut().zip(&e) {
                *a += wn * x as f64;
            }
        }
        let mixed: Vec<f32> = acc.iter().map(|&x| x as f32).collect();
        base.with_embedding(l2_normalize(&mixed), "blend".to_string())
    }

    /// **Linear interpolation** (LERP) between two voices' embeddings at `t ∈ [0, 1]`:
    /// `normalize((1−t)·â + t·b̂)` where `â`, `b̂` are the L2-normalized endpoints.
    ///
    /// `t = 0` returns `a`'s (normalized) timbre, `t = 1` returns `b`'s. The result carries
    /// **`a`'s** clip-tied conditioning. For perceptually-even motion along the unit sphere
    /// prefer [`Voice::slerp`].
    pub fn interpolate(a: &Voice, b: &Voice, t: f32) -> Result<Voice, SynthError> {
        let ea = l2_normalize(&a.embedding_vec()?);
        let eb = l2_normalize(&b.embedding_vec()?);
        if ea.len() != eb.len() {
            return Err(SynthError::Candle(
                "Voice::interpolate: embedding dim mismatch".to_string(),
            ));
        }
        let t = t as f64;
        let mixed: Vec<f32> = ea
            .iter()
            .zip(&eb)
            .map(|(&x, &y)| ((1.0 - t) * x as f64 + t * y as f64) as f32)
            .collect();
        a.with_embedding(l2_normalize(&mixed), "interpolate".to_string())
    }

    /// **Spherical** interpolation (SLERP) between two voices' embeddings at `t ∈ [0, 1]`:
    /// great-circle motion on the unit sphere between the L2-normalized endpoints, which
    /// keeps a constant rate of timbre change (unlike [`Voice::interpolate`]'s chord).
    ///
    /// Falls back to LERP when the endpoints are nearly (anti)parallel (`sin ω ≈ 0`), where
    /// the SLERP formula is numerically unstable. `t = 0`/`t = 1` return the (normalized)
    /// endpoints exactly. Carries **`a`'s** clip-tied conditioning.
    pub fn slerp(a: &Voice, b: &Voice, t: f32) -> Result<Voice, SynthError> {
        let ea = l2_normalize(&a.embedding_vec()?);
        let eb = l2_normalize(&b.embedding_vec()?);
        if ea.len() != eb.len() {
            return Err(SynthError::Candle(
                "Voice::slerp: embedding dim mismatch".to_string(),
            ));
        }
        let t = t as f64;
        let cos = cosine(&ea, &eb);
        let omega = cos.acos();
        let sin_omega = omega.sin();
        let mixed: Vec<f32> = if sin_omega.abs() < 1e-6 {
            // (anti)parallel endpoints: SLERP is ill-conditioned — fall back to LERP.
            ea.iter()
                .zip(&eb)
                .map(|(&x, &y)| ((1.0 - t) * x as f64 + t * y as f64) as f32)
                .collect()
        } else {
            let wa = ((1.0 - t) * omega).sin() / sin_omega;
            let wb = (t * omega).sin() / sin_omega;
            ea.iter()
                .zip(&eb)
                .map(|(&x, &y)| (wa * x as f64 + wb * y as f64) as f32)
                .collect()
        };
        a.with_embedding(l2_normalize(&mixed), "slerp".to_string())
    }

    /// A lightweight, **honestly-scoped** attribute hook: nudge the embedding along a
    /// deterministic, name-derived axis by `amount`, then re-normalize.
    ///
    /// The axis is a fixed pseudo-random unit vector derived **only** from `axis` (a stable
    /// hash → seeded SplitMix64 → Gaussian draw → normalize), so `with_attribute("warmth",
    /// 0.2)` is reproducible and orthogonal-ish across distinct names. It perturbs the
    /// timbre in a stable, repeatable direction and can be inverted by negating `amount`.
    ///
    /// ## Honesty caveat
    /// This is NOT a trained, semantic attribute axis. The mapping from a name like
    /// `"warmth"` / `"brightness"` to a perceptual effect is **not learned** — CosyVoice
    /// exposes no disentangled attribute dimensions, and the CAM++ space has no labeled
    /// axes. Treat this as a reproducible timbre *perturbation* knob (useful for
    /// A/B nudging and data augmentation), not as fine, named attribute control. Keep
    /// `amount` modest (≈±0.3); large values walk far from any real speaker.
    pub fn with_attribute(&self, axis: &str, amount: f32) -> Result<Voice, SynthError> {
        let base = l2_normalize(&self.embedding_vec()?);
        let dir = name_axis(axis, base.len());
        let nudged: Vec<f32> = base
            .iter()
            .zip(&dir)
            .map(|(&x, &d)| (x as f64 + amount as f64 * d as f64) as f32)
            .collect();
        self.with_embedding(l2_normalize(&nudged), format!("{}@{axis}", self.name))
    }
}

/// A small **voice-arithmetic** builder — timbre algebra on speaker embeddings:
/// `VoiceArithmetic::base(v).add(v2, w).sub(v3, w).build()`.
///
/// Every operand embedding is L2-normalized, weighted, and accumulated onto the base
/// (also L2-normalized); [`build`](VoiceArithmetic::build) L2-normalizes the result. `add`
/// pushes toward an operand's timbre, `sub` pushes away (it is exactly `add` with a negated
/// weight). The built voice carries the **base voice's** clip-tied conditioning (see the
/// module docs) and is named `"arithmetic"`.
pub struct VoiceArithmetic<'a> {
    base: &'a Voice,
    /// Running accumulator in embedding space (starts as the normalized base embedding).
    acc: Vec<f64>,
    /// First error encountered while chaining (surfaced at [`build`](Self::build)).
    err: Option<SynthError>,
}

impl<'a> VoiceArithmetic<'a> {
    /// Start a voice-arithmetic chain from a base voice (its normalized embedding seeds the
    /// accumulator; its clip-tied conditioning is what the result will carry).
    pub fn base(v: &'a Voice) -> Self {
        let (acc, err) = match v.embedding_vec() {
            Ok(e) => (l2_normalize(&e).iter().map(|&x| x as f64).collect(), None),
            Err(e) => (Vec::new(), Some(e)),
        };
        VoiceArithmetic { base: v, acc, err }
    }

    /// Add `w · normalize(v.embedding)` to the accumulator (push toward `v`'s timbre).
    pub fn add(self, v: &Voice, w: f32) -> Self {
        self.combine(v, w)
    }

    /// Subtract `w · normalize(v.embedding)` from the accumulator (push away from `v`'s
    /// timbre) — exactly [`add`](Self::add) with `-w`.
    pub fn sub(self, v: &Voice, w: f32) -> Self {
        self.combine(v, -w)
    }

    fn combine(mut self, v: &Voice, w: f32) -> Self {
        if self.err.is_some() {
            return self;
        }
        match v.embedding_vec() {
            Ok(e) => {
                let e = l2_normalize(&e);
                if e.len() != self.acc.len() {
                    self.err = Some(SynthError::Candle(
                        "VoiceArithmetic: embedding dim mismatch".to_string(),
                    ));
                    return self;
                }
                for (a, &x) in self.acc.iter_mut().zip(&e) {
                    *a += w as f64 * x as f64;
                }
            }
            Err(e) => self.err = Some(e),
        }
        self
    }

    /// Finish the chain: L2-normalize the accumulated embedding and return a new [`Voice`]
    /// carrying the base voice's clip-tied conditioning. Surfaces the first error hit while
    /// chaining (e.g. a dim mismatch).
    pub fn build(self) -> Result<Voice, SynthError> {
        if let Some(e) = self.err {
            return Err(e);
        }
        let mixed: Vec<f32> = self.acc.iter().map(|&x| x as f32).collect();
        self.base
            .with_embedding(l2_normalize(&mixed), "arithmetic".to_string())
    }
}

/// Derive a deterministic pseudo-random **unit** axis of length `dim` from a name, via a
/// stable FNV-1a hash → SplitMix64 → Box–Muller Gaussian draw → L2-normalize. Same name ⇒
/// same axis; different names ⇒ near-orthogonal axes (random Gaussian vectors in high-dim
/// are near-orthogonal w.h.p.). Used by [`Voice::with_attribute`].
fn name_axis(name: &str, dim: usize) -> Vec<f32> {
    // FNV-1a 64-bit of the name → seed.
    let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        seed ^= *b as u64;
        seed = seed.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut rng = AxisRng { state: seed | 1 };
    let raw: Vec<f32> = (0..dim).map(|_| rng.next_gauss() as f32).collect();
    l2_normalize(&raw)
}

/// Minimal seeded SplitMix64 + Box–Muller, local to the axis derivation (never system RNG,
/// so a name → axis mapping is bit-reproducible across runs/platforms).
struct AxisRng {
    state: u64,
}

impl AxisRng {
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

    fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
