//! The float64 `f0_predictor` and the `m_source` harmonic-merge weights of the CV3 HiFT
//! vocoder. Split out of `real_cv3.rs` unchanged; the methods extend [`super::Cv3Hift`].

use candle_core::{DType, Result, Tensor, D};

use super::Cv3Hift;

impl Cv3Hift {
    /// `CausalConvRNNF0Predictor` forward in **float64**: condnet[0] is a
    /// right-causal `CausalConv1d(k=4)`, condnet[2,4,6,8] are left-causal
    /// `CausalConv1d(k=3)`, each followed by ELU; then `|Linear(512->1)|`.
    /// `mel` is `[1,80,T]`; returns `[1,T]` (cast to f32 at the end).
    pub fn f0_predict(&self, mel: &Tensor) -> Result<Tensor> {
        let dt = DType::F64;
        let mut x = mel.to_dtype(dt)?;
        // condnet[0]: CausalConv1d(80,512,k=4,causal_type='right'), then ELU.
        x = self.causal_conv(
            &x,
            "f0_predictor.condnet.0.weight",
            "f0_predictor.condnet.0.bias",
            1,
            false, // right
            dt,
        )?;
        x = elu(&x)?;
        // condnet[2,4,6,8]: CausalConv1d(512,512,k=3,causal_type='left'), then ELU.
        for layer in [2usize, 4, 6, 8] {
            x = self.causal_conv(
                &x,
                &format!("f0_predictor.condnet.{layer}.weight"),
                &format!("f0_predictor.condnet.{layer}.bias"),
                1,
                true, // left
                dt,
            )?;
            x = elu(&x)?;
        }
        // transpose [B,T,512], Linear(512->1), abs, squeeze.
        let x = x.transpose(1, 2)?.contiguous()?;
        let w = self.raw_t("f0_predictor.classifier.weight", dt)?; // [1,512]
        let b = self.raw_t("f0_predictor.classifier.bias", dt)?; // [1]
        let y = x.broadcast_matmul(&w.t()?)?.broadcast_add(&b)?; // [B,T,1]
        y.squeeze(D::Minus1)?.abs()?.to_dtype(DType::F32)
    }

    /// The `SourceModuleHnNSF.l_linear` harmonic-merge weights: a learned
    /// `Linear(nb_harmonics+1 -> 1)` that fuses the per-harmonic sine excitations
    /// `[.., 9]` (fundamental + 8 overtones) into the single-channel NSF source, then
    /// `tanh`. CV3's `m_source.l_linear` is a plain (non-`weight_norm`) `nn.Linear`, so
    /// it is fetched verbatim (no fold). Returns `(weight[9], bias)` so the perceptual
    /// **quality** source builder can reproduce CV3's `m_source` merge exactly. The
    /// deterministic single-harmonic smoke source does not use these. Additive — the
    /// existing decode/f0 paths are byte-unchanged.
    pub fn source_merge_linear(&self) -> Result<(Vec<f32>, f32)> {
        let w = self.raw_t("m_source.l_linear.weight", DType::F32)?; // [1, 9]
        let b = self.raw_t("m_source.l_linear.bias", DType::F32)?; // [1]
        let w: Vec<f32> = w.flatten_all()?.to_vec1::<f32>()?;
        let b: f32 = b.flatten_all()?.to_vec1::<f32>()?[0];
        Ok((w, b))
    }
}

/// ELU (alpha=1): `x` for `x>0`, else `exp(x)-1`.
fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg_in = x.neg()?.relu()?.neg()?; // min(x,0)
    let neg = neg_in.exp()?.affine(1.0, -1.0)?;
    pos.add(&neg)
}
