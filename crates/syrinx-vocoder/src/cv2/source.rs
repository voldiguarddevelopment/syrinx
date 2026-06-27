//! The source/F0 branch of the HiFT vocoder: the deterministic F0 predictor, the
//! `m_source` harmonic-merge weights, and the source-downsample stage helpers. Split
//! out of `real.rs` unchanged; the methods extend [`super::HiftVocoder`].

use candle_core::{Result, Tensor, D};

use super::resblock::elu;
use super::HiftVocoder;

impl HiftVocoder {
    /// Deterministic [`ConvRNNF0Predictor`] forward: 5×(Conv1d k3 pad1 + ELU)
    /// condnet, then `|Linear(512 -> 1)|`. `mel` is `[1, 80, T]`; returns `[1, T]`.
    pub fn f0_predict(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = mel.clone();
        for i in 0..5 {
            let layer = i * 2; // condnet has ELU between convs (indices 1,3,5,7,9)
            x = self.conv1d(
                &x,
                &format!("f0_predictor.condnet.{layer}.weight"),
                &format!("f0_predictor.condnet.{layer}.bias"),
                1,
                1,
                1,
            )?;
            x = elu(&x)?;
        }
        // transpose to [B, T, C], Linear(512 -> 1), abs, squeeze.
        let x = x.transpose(1, 2)?.contiguous()?; // [B, T, 512]
        let w = self.g("f0_predictor.classifier.weight")?; // [1, 512]
        let b = self.g("f0_predictor.classifier.bias")?; // [1]
        let y = x.broadcast_matmul(&w.t()?)?.broadcast_add(&b)?; // [B, T, 1]
        y.squeeze(D::Minus1)?.abs()
    }

    /// The `SourceModuleHnNSF.l_linear` harmonic-merge weights: a learned
    /// `Linear(nb_harmonics+1 -> 1)` that fuses the per-harmonic sine excitations
    /// `[.., 9]` (fundamental + 8 overtones) into the single-channel NSF source the
    /// vocoder consumes, followed by `tanh`. Returns `(weight[9], bias)` so the
    /// random-phase source builder can reproduce CosyVoice2's `m_source` merge
    /// exactly. The deterministic single-harmonic smoke source does not use these.
    pub fn source_merge_linear(&self) -> Result<(Vec<f32>, f32)> {
        let w = self.g("m_source.l_linear.weight")?; // [1, 9]
        let b = self.g("m_source.l_linear.bias")?; // [1]
        let w: Vec<f32> = w.flatten_all()?.to_vec1::<f32>()?;
        let b: f32 = b.flatten_all()?.to_vec1::<f32>()?[0];
        Ok((w, b))
    }
}

// ---- free helpers -----------------------------------------------------------

/// `(padding, stride)` for the `source_downs[i]` conv (downsamples the source STFT
/// to the channel/time resolution of upsample stage `i`).
pub(super) fn source_down_pad_stride(i: usize) -> (usize, usize) {
    // downsample_cum_rates reversed = [15, 3, 1]; conv is (u*2, u, pad=u//2) for
    // u>1, else (1, 1, 0).
    match i {
        0 => (7, 15),
        1 => (1, 3),
        2 => (0, 1),
        _ => unreachable!(),
    }
}

/// `source_resblock_kernel_sizes = [7, 7, 11]`.
pub(super) fn source_resblock_kernel(i: usize) -> usize {
    [7, 7, 11][i]
}
