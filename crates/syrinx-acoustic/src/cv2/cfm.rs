//! The CV2 `CausalConditionalCFM` Euler/CFG solver variants (with frozen noise, with a
//! prompt `cond`, and the masked/streaming form). The estimator they integrate lives in
//! [`super::estimator`]. Moved verbatim from `real.rs`.

use super::*;

impl Flow {
    // ============================ CFM / ESTIMATOR ============================

    /// CausalConditionalCFM.solve_euler with CFG. `mu`: [1,80,L], `spk`: [1,80].
    /// Noise z is the frozen design buffer; here we read it from the reference so
    /// the ODE is bit-reproducible (the buffer is a fixed seed-0 randn, baked in).
    pub fn cfm_solve_with_noise(&self, mu: &Tensor, spk: &Tensor, z0: &Tensor, n_timesteps: usize) -> Result<Tensor> {
        let l = mu.dim(2)?;
        // t_span (cosine schedule)
        let mut tvals = vec![0f32; n_timesteps + 1];
        for (i, s) in tvals.iter_mut().enumerate() {
            let lin = i as f32 / n_timesteps as f32;
            *s = 1.0 - (lin * 0.5 * std::f32::consts::PI).cos();
        }
        let mut x = z0.clone(); // [1,80,L]
        let mut t = tvals[0];
        let cfg = 0.7f64; // inference_cfg_rate
        for step in 1..=n_timesteps {
            // CFG batch of 2: index 0 carries mu/spks/cond, index 1 is zeros.
            // x_in[:] = x  -> both rows are x
            let x_in = Tensor::cat(&[&x, &x], 0)?; // [2,80,L]
            let mu0 = mu.clone();
            let mu1 = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
            let mu_in = Tensor::cat(&[&mu0, &mu1], 0)?; // [2,80,L]
            let spk0 = spk.clone();
            let spk1 = Tensor::zeros((1, MEL), DType::F32, &self.dev)?;
            let spk_in = Tensor::cat(&[&spk0, &spk1], 0)?; // [2,80]
            let cond_in = Tensor::zeros((2, MEL, l), DType::F32, &self.dev)?; // no prompt
            let t_in = Tensor::from_vec(vec![t, t], (2,), &self.dev)?;
            let dphi = self.estimator(&x_in, &mu_in, &t_in, &spk_in, &cond_in)?; // [2,80,L]
            let real = dphi.narrow(0, 0, 1)?;
            let uncond = dphi.narrow(0, 1, 1)?;
            // (1+cfg)*real - cfg*uncond
            let dphi_dt = ((real * (1.0 + cfg))? - (uncond * cfg)?)?;
            let dt = tvals[step] - t;
            x = (x + (dphi_dt * dt as f64)?)?;
            t = tvals[step];
        }
        Ok(x)
    }

    /// CausalConditionalCFM.solve_euler with CFG **and a non-trivial `cond`** — the
    /// zero-shot prompt path. Identical to [`Self::cfm_solve_with_noise`] except the
    /// conditioning signal is the caller-supplied `cond` `[1, 80, L]` (the prompt mel
    /// prepended, zeros after) rather than all-zeros. The unconditioned CFG branch
    /// (index 1) keeps `cond = 0`, mirroring `solve_euler`'s `cond_in[0] = cond`.
    pub fn cfm_solve_with_cond(
        &self,
        mu: &Tensor,
        spk: &Tensor,
        cond: &Tensor,
        z0: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        // Parity default: the unmasked (full-context) estimator. Thin pass-through so
        // existing callers + frozen parity tests keep byte-identical behavior.
        self.cfm_solve_with_cond_masked(mu, spk, cond, z0, n_timesteps, None)
    }

    /// [`Self::cfm_solve_with_cond`] with an optional chunked-causal estimator mask.
    ///
    /// `mask`, if given, is the additive `[1,1,L,L]` mask (built at the mel length `L`)
    /// threaded into every estimator call so the CFM U-Net's self-attention is
    /// chunk-causal; `None` reproduces the full-context parity path exactly.
    pub fn cfm_solve_with_cond_masked(
        &self,
        mu: &Tensor,
        spk: &Tensor,
        cond: &Tensor,
        z0: &Tensor,
        n_timesteps: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let l = mu.dim(2)?;
        let mut tvals = vec![0f32; n_timesteps + 1];
        for (i, s) in tvals.iter_mut().enumerate() {
            let lin = i as f32 / n_timesteps as f32;
            *s = 1.0 - (lin * 0.5 * std::f32::consts::PI).cos();
        }
        let mut x = z0.clone(); // [1,80,L]
        let mut t = tvals[0];
        let cfg = 0.7f64; // inference_cfg_rate
        let cond_zero = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        for step in 1..=n_timesteps {
            let x_in = Tensor::cat(&[&x, &x], 0)?; // [2,80,L]
            let mu0 = mu.clone();
            let mu1 = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
            let mu_in = Tensor::cat(&[&mu0, &mu1], 0)?; // [2,80,L]
            let spk0 = spk.clone();
            let spk1 = Tensor::zeros((1, MEL), DType::F32, &self.dev)?;
            let spk_in = Tensor::cat(&[&spk0, &spk1], 0)?; // [2,80]
            // cond[0] = prompt cond, cond[1] = zeros (the CFG-dropped branch).
            let cond_in = Tensor::cat(&[cond, &cond_zero], 0)?; // [2,80,L]
            let t_in = Tensor::from_vec(vec![t, t], (2,), &self.dev)?;
            let dphi = self.estimator_masked(&x_in, &mu_in, &t_in, &spk_in, &cond_in, mask)?; // [2,80,L]
            let real = dphi.narrow(0, 0, 1)?;
            let uncond = dphi.narrow(0, 1, 1)?;
            let dphi_dt = ((real * (1.0 + cfg))? - (uncond * cfg)?)?;
            let dt = tvals[step] - t;
            x = (x + (dphi_dt * dt as f64)?)?;
            t = tvals[step];
        }
        Ok(x)
    }

    /// Convenience: solve using the design noise buffer reconstructed via the
    /// reference fixture is preferred; this variant uses a provided z explicitly.
    pub fn cfm_solve(&self, mu: &Tensor, spk: &Tensor, n_timesteps: usize) -> Result<Tensor> {
        // Without the frozen randn buffer we cannot reproduce z; callers that need
        // bit-parity must pass the reference noise via cfm_solve_with_noise.
        let l = mu.dim(2)?;
        let z = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        self.cfm_solve_with_noise(mu, spk, &z, n_timesteps)
    }
}
