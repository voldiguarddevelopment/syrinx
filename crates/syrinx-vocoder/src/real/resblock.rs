//! Snake activation, the HiFi-GAN ResBlock, the activation primitives and the upsample
//! padding helpers. Split out of `real.rs` unchanged; the methods extend
//! [`super::HiftVocoder`].

use candle_core::{Result, Tensor, D};

use super::{HiftVocoder, RESBLOCK_DILATIONS, SNAKE_EPS, UPSAMPLE_RATES};

impl HiftVocoder {
    /// Snake activation: `x + (1/(alpha+eps)) * sin(x*alpha)^2`, channel-wise alpha
    /// (`alpha_logscale=False`). `x` is `[B, C, T]`, `alpha` is `[C]`.
    fn snake(&self, x: &Tensor, alpha_name: &str) -> Result<Tensor> {
        let a = self.g(alpha_name)?; // [C]
        let alpha = a.reshape((1, a.dim(0)?, 1))?;
        let xa = x.broadcast_mul(&alpha)?;
        let s = xa.sin()?.sqr()?;
        let inv = alpha.affine(1.0, SNAKE_EPS)?.recip()?;
        x.add(&s.broadcast_mul(&inv)?)
    }

    /// One HiFi-GAN [`ResBlock`]: for each of the 3 dilations, Snake -> dilated
    /// conv -> Snake -> conv, added back to the running residual.
    pub(super) fn resblock(&self, x: &Tensor, prefix: &str, kernel: usize) -> Result<Tensor> {
        let mut x = x.clone();
        for (idx, &dil) in RESBLOCK_DILATIONS.iter().enumerate() {
            let pad1 = (kernel * dil - dil) / 2; // get_padding(k, dil)
            let pad2 = (kernel - 1) / 2; // get_padding(k, 1)
            let xt = self.snake(&x, &format!("{prefix}.activations1.{idx}.alpha"))?;
            let xt = self.conv1d(
                &xt,
                &format!("{prefix}.convs1.{idx}.weight"),
                &format!("{prefix}.convs1.{idx}.bias"),
                pad1,
                1,
                dil,
            )?;
            let xt = self.snake(&xt, &format!("{prefix}.activations2.{idx}.alpha"))?;
            let xt = self.conv1d(
                &xt,
                &format!("{prefix}.convs2.{idx}.weight"),
                &format!("{prefix}.convs2.{idx}.bias"),
                pad2,
                1,
                1,
            )?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

// ---- free helpers -----------------------------------------------------------

/// `(padding, stride)` for upsample stage `i` (`padding=(k-u)//2`).
pub(super) fn ups_pad_stride(i: usize) -> (usize, usize) {
    let kernels = [16usize, 11, 7];
    let u = UPSAMPLE_RATES[i];
    ((kernels[i] - u) / 2, u)
}

/// LeakyReLU: `x` where `x>=0`, `slope*x` otherwise, as `relu(x) - slope*relu(-x)`.
pub(super) fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu()?; // max(x, 0)
    let neg = x.neg()?.relu()?.affine(slope, 0.0)?; // slope * max(-x, 0)
    pos.sub(&neg)
}

/// ELU: `x` where `x>0`, `exp(x)-1` otherwise (alpha=1).
pub(super) fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?; // x for x>0, else 0
    // negative branch: min(x,0) -> exp(min)-1 ; for x>0 exp(0)-1 = 0.
    let neg_in = x.neg()?.relu()?.neg()?; // min(x, 0)
    let neg = neg_in.exp()?.affine(1.0, -1.0)?;
    pos.add(&neg)
}

/// ReflectionPad1d((1, 0)): prepend a reflection of the second sample on the left.
/// torch reflects across the boundary, so the prepended value is `x[..., 1]`.
pub(super) fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let left = x.narrow(D::Minus1, 1, 1)?; // reflect across the boundary: index 1
    Tensor::cat(&[&left, x], D::Minus1)
}
