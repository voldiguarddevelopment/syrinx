//! CV3 HiFT source builders + the perceptual-quality synthesis path.
//!
//! Holds the deterministic (zero-phase, single-harmonic smoke) source, the
//! random-phase NSF (`SourceModuleHnNSF` / SineGen) source, the `synthesize_quality`
//! capstone, and the model-free `det_source_from_f0` / `quality_source_from_f0` test
//! seams the methods wrap. CV3's `Cv3Hift::decode` STFTs the source **waveform**
//! `[1,1,L]` internally (unlike CV2's pre-STFT'd `s_stft`).

use candle_core::Tensor;

use super::*;

impl Cv3Synthesizer {
    /// Build a **deterministic, zero-phase, single-harmonic** HiFT source **waveform**
    /// `[1, 1, L]` from a generated mel, for the functional (non-parity) path.
    ///
    /// Mirrors the deterministic core of CV3's `m_source` excitation: the CV3 HiFT F0
    /// predictor -> nearest-upsample by 480 -> single sine `sine_amp * sin(2π·cumsum(f0/sr))`.
    /// `Cv3Hift::decode` STFTs this waveform internally. The model's `SourceModuleHnNSF`
    /// SineGen additionally draws a random initial phase + Gaussian noise and applies a
    /// learned `l_linear(9->1)+tanh` harmonic merge (none bit-portable across RNGs), so
    /// this is a faithful smoke source, not a parity source.
    pub fn deterministic_source(&self, mel: &Tensor) -> Result<Tensor, SynthError> {
        let f0 = self.vocoder.f0_predict(mel)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        let source = det_source_from_f0(&f0v);
        let n = source.len();
        Ok(Tensor::from_vec(source, (1, 1, n), &self.dev)?)
    }

    /// Build the **random-phase NSF** HiFT source **waveform** `[1, 1, L]` from a generated
    /// mel — the faithful CV3 `SourceModuleHnNSF` excitation for the perceptual-quality
    /// path, the exact analogue of CV2's `Synthesizer::random_phase_source_stft` but
    /// emitting the *waveform* (CV3's `Cv3Hift::decode` STFTs it internally), not a
    /// pre-STFT'd `s_stft`.
    ///
    /// CV3's HiFT (`CausalHiFTGenerator`, 24 kHz) instantiates `SourceModuleHnNSF` with
    /// `SineGen2(causal=True)` (`sinegen_type='2'` when `sampling_rate != 22050`). Its
    /// steady-state output equals the per-sample cumulative-phase ramp used here; every
    /// perceptually load-bearing element is reproduced (and seeded for reproducibility):
    ///   * **`NB_HARMONICS + 1 = 9` sines** — fundamental + 8 overtones, the `i`-th at
    ///     instantaneous phase `(i+1)·θ(t)` with `θ(t) = 2π·cumsum(f0_up/sr)`;
    ///   * **random initial phase per harmonic** — `φ_0 = 0` (fundamental), `φ_i ∈ (-π, π]`
    ///     (CV3 `SineGen2`'s `rand_ini` with `rand_ini[:,0]=0`);
    ///   * **voiced/unvoiced mask** `uv = f0 > 10 Hz` (gates the sines to 0 when unvoiced);
    ///   * **additive Gaussian noise** `noise_amp = uv·noise_std + (1-uv)·sine_amp/3`
    ///     (`σ=0.003` breath when voiced, `sine_amp/3 ≈ 0.033` broadband when unvoiced);
    ///   * **learned merge** — the 9 components fused by the checkpoint's
    ///     `m_source.l_linear` `Linear(9→1)` + `tanh` ([`Cv3Hift::source_merge_linear`]).
    ///
    /// All randomness comes from a **seeded** SplitMix64 (never system RNG), so a given
    /// `(mel, seed)` is bit-reproducible. This is a perceptual-quality source, not a
    /// bit-parity source — the model's torch RNG draw order is not portable. The smoke
    /// [`Cv3Synthesizer::deterministic_source`] is left byte-unchanged.
    pub fn quality_source(&self, mel: &Tensor, seed: u64) -> Result<Tensor, SynthError> {
        let f0 = self.vocoder.f0_predict(mel)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        let (merge_w, merge_b) = self.vocoder.source_merge_linear()?; // ([9], b)
        if merge_w.len() != NB_HARMONICS + 1 {
            return Err(SynthError::Candle(format!(
                "m_source.l_linear expects {} weights, got {}",
                NB_HARMONICS + 1,
                merge_w.len()
            )));
        }
        let source = quality_source_from_f0(&f0v, &merge_w, merge_b, seed);
        let n = source.len();
        Ok(Tensor::from_vec(source, (1, 1, n), &self.dev)?)
    }

    /// Full CV3 synthesis with the **random-phase NSF source** (perceptual-quality path),
    /// returning the 24 kHz waveform. Same pipeline as [`Cv3Synthesizer::synthesize`] but
    /// the HiFT excitation is [`Cv3Synthesizer::quality_source`] (seeded by `source_seed`)
    /// instead of the buzzy single-harmonic [`Cv3Synthesizer::deterministic_source`], and
    /// the CFM noise `z` is a **seeded standard-normal** init (the flow's `rand_noise`
    /// *distribution*) rather than the degenerate zeros init — both reproducible from
    /// `source_seed`, which also seeds live LM sampling so a `(text, ref, source_seed)`
    /// triple is fully reproducible.
    pub fn synthesize_quality(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        source_seed: u64,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = self.generate_speech_token(&cond, source_seed, None)?;

        // CFM noise z: a SEEDED standard-normal init (torch.randn's distribution, not its
        // byte stream). Now the same default `synthesize` uses (see `seeded_normal_z`).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = self.seeded_normal_z(total, source_seed)?;

        let mel = self.flow_forward(&cond, &speech_token, &z, N_TIMESTEPS)?; // [1,80,2*Tg]
        let source = self.quality_source(&mel, source_seed)?; // [1,1,L]
        let audio = self.vocode(&mel, &source)?; // [1, L]
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }
}

// ---- HiFT source-excitation math (pure; the `*_source` methods call these) --------
//
// Extracted verbatim from `Cv3Synthesizer::deterministic_source` / `quality_source` so
// the f0 → source math is testable **model-free** (the methods only add the `f0_predict`
// + merge-weight fetch + tensor wrap around these). The computation is byte-unchanged.
// `#[doc(hidden)] pub` is the test seam — not part of the documented surface.

/// Deterministic, zero-phase, single-harmonic HiFT source from an f0 frame vector:
/// nearest-upsample f0 by [`F0_UPSAMPLE`], then `sine_amp·sin(2π·cumsum(f0_up/sr))`.
#[doc(hidden)]
pub fn det_source_from_f0(f0v: &[f32]) -> Vec<f32> {
    let n = f0v.len() * F0_UPSAMPLE;
    let mut source: Vec<f32> = Vec::with_capacity(n);
    let mut acc = 0.0f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    for &v in f0v {
        let fhz = v as f64;
        for _ in 0..F0_UPSAMPLE {
            acc += fhz / MEL_SR as f64;
            source.push((SINE_AMP * (two_pi * acc).sin()) as f32);
        }
    }
    source
}

/// Random-phase NSF (`SourceModuleHnNSF`) source from an f0 frame vector + the learned
/// `m_source.l_linear` merge `(merge_w, merge_b)`: `merge_w.len()` harmonics with seeded
/// random initial phase (φ_0 = 0), a voiced/unvoiced mask, additive Gaussian breath, and
/// the learned `linear + tanh` merge. All randomness is the **seeded** [`SplitMix64`], so
/// a `(f0v, merge, seed)` triple is bit-reproducible.
#[doc(hidden)]
pub fn quality_source_from_f0(f0v: &[f32], merge_w: &[f32], merge_b: f32, seed: u64) -> Vec<f32> {
    let n = f0v.len() * F0_UPSAMPLE; // source samples = mel frames * 480
    let mut rng = SplitMix64::new(seed);

    // Random initial phase per harmonic: φ_0 = 0 (fundamental), φ_i ∈ (-π, π].
    let mut phi = vec![0f64; merge_w.len()];
    for p in phi.iter_mut().skip(1) {
        *p = (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI;
    }

    // Single-channel merged excitation, sample by sample.
    let mut source: Vec<f32> = Vec::with_capacity(n);
    let mut base_phase = 0.0f64; // θ(t) = 2π·cumsum(f0_up/sr), built incrementally
    let two_pi = 2.0 * std::f64::consts::PI;
    for s in 0..n {
        let fhz = f0v[s / F0_UPSAMPLE] as f64; // nearest-upsample of f0 by 480
        base_phase += two_pi * (fhz / MEL_SR as f64);
        let uv = if fhz > NSF_VOICED_THRESHOLD { 1.0 } else { 0.0 };
        let noise_amp = uv * NSF_NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
        // harmonic components -> learned linear(9->1) + tanh merge.
        let mut acc = merge_b as f64;
        for (i, &w) in merge_w.iter().enumerate() {
            let h = (i + 1) as f64; // harmonic multiplier (fundamental = 1)
            let sine = SINE_AMP * (h * base_phase + phi[i]).sin();
            let noise = noise_amp * rng.next_gauss();
            let comp = sine * uv + noise; // SineGen: sine_waves * uv + noise
            acc += w as f64 * comp;
        }
        source.push(acc.tanh() as f32);
    }
    source
}
