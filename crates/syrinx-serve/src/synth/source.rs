//! HiFT source builders + the perceptual-quality synthesis path.
//!
//! Holds the deterministic (zero-phase, single-harmonic smoke) source, the
//! random-phase NSF (`SourceModuleHnNSF` / SineGen) source, the shared `source_stft`
//! glue, the `synthesize_quality` capstone that uses the random-phase source, and the
//! seeded `SplitMix64` PRNG that is the source builders' only randomness.

use candle_core::Tensor;

use super::*;

impl Synthesizer {
    /// Build a **deterministic, zero-phase, noise-free** HiFT source STFT `[1, 18, T]`
    /// from a generated mel, for the functional (non-parity) path. Mirrors the real
    /// source branch's deterministic core: F0 predictor -> nearest-upsample by 480 ->
    /// single-harmonic sine `sine_amp * sin(2π·cumsum(f0/sr))` -> STFT (n_fft=16,
    /// hop=4, hann window) -> `cat([real, imag])`. The model's SineGen adds a random
    /// initial phase + Gaussian noise (intentionally not ported), so this is a
    /// faithful *smoke* source — finite, non-silent, correctly shaped — not a parity
    /// source.
    pub fn deterministic_source_stft(&self, mel: &Tensor) -> Result<Tensor, SynthError> {
        let f0 = self.vocoder.f0_predict(mel)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        self.f0_to_source_stft(&f0v)
    }

    /// Build the deterministic zero-phase HiFT source STFT `[1, 18, T]` from an
    /// explicit per-mel-frame F0 vector `f0v` (Hz). This is the F0→source half of
    /// [`Synthesizer::deterministic_source_stft`], factored out so a prosody plan
    /// can retune the F0 (pitch control) before the source is built: nearest
    /// upsample by 480 → single-harmonic `sine_amp·sin(2π·cumsum(f0/sr))` → STFT.
    pub(super) fn f0_to_source_stft(&self, f0v: &[f32]) -> Result<Tensor, SynthError> {
        // Upsample F0 by nearest (Upsample(scale_factor=480)) to the source rate.
        let mut f0_up: Vec<f64> = Vec::with_capacity(f0v.len() * F0_UPSAMPLE);
        for &v in f0v {
            for _ in 0..F0_UPSAMPLE {
                f0_up.push(v as f64);
            }
        }
        // Zero-phase instantaneous phase: phase[t] = 2π · cumsum(f0/sr).
        let mut source: Vec<f32> = Vec::with_capacity(f0_up.len());
        let mut acc = 0.0f64;
        for &fhz in &f0_up {
            acc += fhz / MEL_SR as f64;
            let s = SINE_AMP * (2.0 * std::f64::consts::PI * acc).sin();
            source.push(s as f32);
        }
        self.source_stft(&source)
    }

    /// Build the **random-phase NSF** HiFT source STFT `[1, 18, T]` from a generated
    /// mel — a faithful port of CosyVoice2's `SourceModuleHnNSF` (`SineGen`) source,
    /// for the perceptual-quality path. Unlike [`deterministic_source_stft`] (a
    /// single zero-phase harmonic, no noise), this reproduces every stochastic and
    /// multi-harmonic element of the real excitation:
    ///
    ///   * **`NB_HARMONICS + 1 = 9` sines** — fundamental + 8 overtones, the `i`-th
    ///     at instantaneous phase `(i+1)·θ(t)` where `θ(t) = 2π·cumsum(f0_up/sr)` is
    ///     the shared zero-phase ramp (the same per-sample cumulative phase the
    ///     deterministic source uses);
    ///   * **random initial phase per harmonic** — a constant offset `φ_i ∈ (-π, π]`
    ///     added to each overtone (the fundamental keeps `φ_0 = 0`, exactly as
    ///     `SineGen`'s `phase_vec[:,0,:] = 0`), decorrelating the overtones so the
    ///     source is no longer a single rigidly phase-locked buzz;
    ///   * **voiced/unvoiced mask** `uv = f0 > NSF_VOICED_THRESHOLD (10 Hz)` — the
    ///     harmonic sines are gated to zero in unvoiced frames;
    ///   * **additive Gaussian noise** at `noise_amp = uv·noise_std + (1-uv)·sine_amp/3`
    ///     — quiet (`σ=0.003`) breath in voiced frames, and `sine_amp/3 ≈ 0.033`
    ///     broadband noise replacing the (masked-out) sines in unvoiced frames;
    ///   * **learned harmonic merge** — the 9 components are fused by the checkpoint's
    ///     `m_source.l_linear` `Linear(9→1)` + `tanh`, the real `SourceModuleHnNSF`
    ///     merge, yielding the single-channel excitation that is then STFT'd.
    ///
    /// All randomness comes from a **seeded** SplitMix64 stream keyed by `seed`
    /// (never system RNG), so a given `(mel, seed)` is bit-reproducible across runs —
    /// the A/B harness can pin a seed and the result is stable.
    ///
    /// Honesty note: the 24 kHz config technically instantiates `SineGen2`, whose
    /// `_f02sine` accumulates phase at *frame* rate and linearly interpolates back up
    /// (an anti-aliasing detail); its steady-state output equals the per-sample
    /// `cumsum` ramp used here, and the perceptually load-bearing elements (the
    /// random per-harmonic phase, the 9 harmonics, the noise, the uv mask, and the
    /// learned merge) are identical between the two. This is a perceptual-quality
    /// source, not a bit-parity source — the model's RNG draw order is not portable.
    pub fn random_phase_source_stft(
        &self,
        mel: &Tensor,
        seed: u64,
    ) -> Result<Tensor, SynthError> {
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

        let n = f0v.len() * F0_UPSAMPLE; // source samples = mel frames * 480
        let mut rng = SplitMix64::new(seed);

        // Random initial phase per harmonic: φ_0 = 0 (fundamental), φ_i ∈ (-π, π].
        let n_comp = NB_HARMONICS + 1;
        let mut phi = vec![0f64; n_comp];
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
            // 9 harmonic components -> learned linear(9->1) + tanh merge.
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
        self.source_stft(&source)
    }

    /// STFT a 1-D source waveform into `[1, 18, T]` (n_fft=16, hop=4, periodic hann,
    /// `center=True` reflect padding) matching the HiFT `_stft`, real ++ imag.
    pub(super) fn source_stft(&self, source: &[f32]) -> Result<Tensor, SynthError> {
        let n_fft = HIFT_N_FFT;
        let hop = HIFT_HOP;
        let bins = HIFT_BINS;
        // periodic hann window (get_window("hann", n_fft, fftbins=True)).
        let window: Vec<f64> = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
            .collect();
        // center=True: reflect-pad n_fft/2 each side.
        let pad = n_fft / 2;
        let padded = reflect_pad(source, pad);
        let n = padded.len();
        if n < n_fft {
            return Err(SynthError::Candle("source too short for STFT".into()));
        }
        let n_frames = (n - n_fft) / hop + 1;
        // real and imag channels: [n_fft+2] = [bins (re) ++ bins (im)] over T frames.
        let mut data = vec![0f32; (2 * bins) * n_frames];
        for t in 0..n_frames {
            let start = t * hop;
            for k in 0..bins {
                let mut re = 0f64;
                let mut im = 0f64;
                let w = -2.0 * std::f64::consts::PI * k as f64 / n_fft as f64;
                for j in 0..n_fft {
                    let x = padded[start + j] as f64 * window[j];
                    let ang = w * j as f64;
                    re += x * ang.cos();
                    im += x * ang.sin();
                }
                data[k * n_frames + t] = re as f32;
                data[(bins + k) * n_frames + t] = im as f32;
            }
        }
        let t = Tensor::from_vec(data, (1, 2 * bins, n_frames), &self.dev)?;
        Ok(t)
    }
}

// ============================================================================
// Perceptual-quality synthesis (additive — random-phase NSF source).
//
// `synthesize_quality` is byte-for-byte `synthesize`'s flow + vocoder, except the
// HiFT excitation is built by `random_phase_source_stft` (CosyVoice2's real
// `SourceModuleHnNSF`: 9 harmonics + random per-harmonic phase + Gaussian noise +
// uv mask + learned merge) instead of the deterministic zero-phase smoke source.
// The deterministic source is buzzy (a single rigidly phase-locked harmonic, no
// noise) and is the main reason the functional UTMOS came out low (~2.03); the
// random-phase source restores the natural harmonic decorrelation + breath the
// HiFT filter expects, which should raise MOS. Kept in its own `impl` block so the
// parity-default `synthesize` and the existing smoke tests are untouched. A pinned
// `inputs.s_stft` is ignored here (this path *is* the source choice); `inputs` is
// still honoured for `pinned_speech_token`, `z`, `lm_seed`, and `max_gen_steps`.
// ============================================================================
impl Synthesizer {
    /// Full synthesis with the **random-phase NSF source** (perceptual-quality
    /// path), returning the 24 kHz waveform. Same pipeline as
    /// [`Synthesizer::synthesize`] but the HiFT source is
    /// [`Synthesizer::random_phase_source_stft`] (seeded by `source_seed`) instead
    /// of the deterministic zero-phase source.
    ///
    /// `source_seed` makes the (otherwise stochastic) source reproducible: the same
    /// `source_seed` + same tokens yields the same waveform. `inputs.z` is honoured
    /// (pinned or zeros fallback) exactly as in `synthesize`; any `inputs.s_stft` is
    /// ignored (this method builds the source itself).
    pub fn synthesize_quality(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &SynthInputs,
        source_seed: u64,
    ) -> Result<Vec<f32>, SynthError> {
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;
        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // CFM noise z: pinned, else a SEEDED standard-normal init — the model's CFM
        // `rand_noise` (torch.randn's *distribution*, not its byte stream), reproducible
        // from `source_seed`. Far more natural than the zeros fallback, which gives the
        // degenerate mean trajectory of the flow ODE (a likely MOS limiter on the
        // deterministic path). `synthesize` keeps zeros (parity/smoke default).
        let total = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z = match &inputs.z {
            Some(z) => z.clone(),
            None => {
                let n = MEL_NUM_MELS * total;
                let mut rng = SplitMix64::new(source_seed ^ 0x5DEE_CE66_C0DE_F10D);
                let zv: Vec<f32> = (0..n).map(|_| rng.next_gauss() as f32).collect();
                Tensor::from_vec(zv, (1, MEL_NUM_MELS, total), &self.dev)?
            }
        };

        // Flow -> generated mel, random-phase NSF source from it, then vocode.
        let mel = self.flow.forward_zero_shot(
            &cond.prompt_token,
            &speech_token,
            &cond.prompt_feat,
            &cond.spk_embedding,
            &z,
            N_TIMESTEPS,
        )?;
        let s = self.random_phase_source_stft(&mel, source_seed)?;
        let audio = self.vocoder.decode(&mel, &s)?;
        let wav: Vec<f32> = audio.flatten_all()?.to_vec1::<f32>()?;
        Ok(wav)
    }
}

/// Minimal seeded **SplitMix64** PRNG — the source builder's *only* randomness, so
/// the random-phase source is reproducible from its `seed` (never system RNG).
///
/// SplitMix64 is the standard fast seeding generator (the one Rust's `StdRng` and
/// xoshiro use to expand a seed); a single stream here drives both the per-harmonic
/// initial phases (uniform) and the additive Gaussian noise (Box–Muller).
struct SplitMix64 {
    state: u64,
    /// Cached second Box–Muller normal (generated in pairs).
    spare: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    /// Next raw 64-bit value (the canonical SplitMix64 mix).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)` (top 53 bits, the standard construction).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal `N(0, 1)` via Box–Muller, generating two at a time and
    /// caching the spare (matches `torch.randn`'s distribution, not its byte stream).
    fn next_gauss(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        // Guard u1 away from 0 so ln is finite.
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// Reflect-pad a 1-D signal by `pad` each side (torch reflect: boundary not repeated).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![0.0; 2 * pad];
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    for k in 0..pad {
        out.push(x[(pad - k).min(n - 1)]);
    }
    out.extend_from_slice(x);
    for k in 0..pad {
        let idx = n.saturating_sub(2).saturating_sub(k);
        out.push(x[idx]);
    }
    out
}
