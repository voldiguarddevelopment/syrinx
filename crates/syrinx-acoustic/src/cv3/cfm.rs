//! The CV3 CFM Euler/CFG solver (reuses the CV2 wrapper structure around the DiT
//! estimator in [`super::dit`]). Moved verbatim from `real_cv3.rs`.

use super::*;

impl Cv3Flow {
    // ============================ CFM SOLVE ============================

    /// CV3 CFM Euler solve with CFG (reuses the CV2 wrapper structure around the DiT).
    ///
    /// `mu,cond: [1,80,L]`, `spk: [1,80]`, `z0: [1,80,L]` (frozen noise consumed
    /// verbatim). 10 cosine-schedule steps, CFG batch-of-2 (`idx0` carries
    /// mu/spk/cond, `idx1` is zeros), `cfg_rate=0.7`. Returns `mel_full [1,80,L]`.
    pub fn cfm_solve(
        &self,
        mu: &Tensor,
        spk: &Tensor,
        cond: &Tensor,
        z0: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        // Parity default: the unmasked (full-context) estimator, byte-identical for the
        // batch path + the frozen parity test.
        self.cfm_solve_masked(mu, spk, cond, z0, n_timesteps, None)
    }

    /// [`Self::cfm_solve`] with an optional chunked-causal estimator mask threaded into
    /// every Euler step's DiT call. `mask` is the additive `[1,1,L,L]` chunk mask built at
    /// the mel length `L`; `None` reproduces the full-context parity path exactly.
    pub fn cfm_solve_masked(
        &self,
        mu: &Tensor,
        spk: &Tensor,
        cond: &Tensor,
        z0: &Tensor,
        n_timesteps: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let l = mu.dim(2)?;
        // t_span: cosine schedule 1 - cos(linspace(0,1,n+1) * 0.5*pi).
        let mut tvals = vec![0f32; n_timesteps + 1];
        for (i, s) in tvals.iter_mut().enumerate() {
            let lin = i as f32 / n_timesteps as f32;
            *s = 1.0 - (lin * 0.5 * std::f32::consts::PI).cos();
        }
        let mut x = z0.clone(); // [1,80,L]
        let mut t = tvals[0];
        let zero_mu = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        let zero_spk = Tensor::zeros((1, MEL), DType::F32, &self.dev)?;
        let zero_cond = Tensor::zeros((1, MEL, l), DType::F32, &self.dev)?;
        for step in 1..=n_timesteps {
            let x_in = Tensor::cat(&[&x, &x], 0)?; // [2,80,L]
            let mu_in = Tensor::cat(&[mu, &zero_mu], 0)?; // [2,80,L]
            let spk_in = Tensor::cat(&[spk, &zero_spk], 0)?; // [2,80]
            let cond_in = Tensor::cat(&[cond, &zero_cond], 0)?; // [2,80,L]
            let t_in = Tensor::from_vec(vec![t, t], (2,), &self.dev)?;
            let dphi = self.estimator_masked(&x_in, &mu_in, &t_in, &spk_in, &cond_in, mask)?; // [2,80,L]
            let real = dphi.narrow(0, 0, 1)?;
            let uncond = dphi.narrow(0, 1, 1)?;
            let dphi_dt = ((real * (1.0 + CFG_RATE))? - (uncond * CFG_RATE)?)?;
            let dt = tvals[step] - t;
            x = (x + (dphi_dt * dt as f64)?)?;
            t = tvals[step];
        }
        Ok(x)
    }
}
