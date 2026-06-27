//! The causal upsample / source-fusion / Snake-ResBlock stack and the `decode` forward of
//! the CV3 HiFT vocoder. Split out of `real_cv3.rs` unchanged; the methods extend
//! [`super::Cv3Hift`].

use candle_core::{DType, Result, Tensor, D};

use super::stft::istft;
use super::{
    Cv3Hift, AUDIO_LIMIT, CONV_PRE_LOOK_RIGHT, LRELU_SLOPE, N_BINS, RESBLOCK_DILATIONS, SNAKE_EPS,
    UPSAMPLE_RATES,
};

impl Cv3Hift {
    /// `CausalConv1dUpsample`: nearest-neighbour upsample by `stride`, then a
    /// left-causal conv (pad `k-1` on the left). Output length == input*stride.
    fn ups(&self, x: &Tensor, wname: &str, bname: &str, stride: usize, dtype: DType) -> Result<Tensor> {
        let up = upsample_nearest1d(x, stride)?;
        self.causal_conv(&up, wname, bname, 1, true, dtype)
    }

    /// `CausalConv1dDownSample`: pad `stride-1` on the left, then a strided conv
    /// (`kernel = stride*2`, no dilation). Downsamples the source STFT to a stage.
    fn source_down(&self, x: &Tensor, wname: &str, bname: &str, stride: usize, dtype: DType) -> Result<Tensor> {
        let w = self.weight(wname, dtype)?; // [out,18,k]
        let b = self.raw_t(bname, dtype)?;
        let xp = x.pad_with_zeros(2, stride - 1, 0)?;
        let y = xp.conv1d(&w, 0, stride, 1, 1)?;
        let b = b.reshape((1, b.dim(0)?, 1))?;
        y.broadcast_add(&b)
    }

    /// Snake activation `x + (1/(alpha+eps)) * sin(x*alpha)^2`, channel-wise alpha
    /// (`alpha_logscale=False`). `x` is `[B,C,T]`, alpha `[C]`.
    fn snake(&self, x: &Tensor, alpha_name: &str, dtype: DType) -> Result<Tensor> {
        let a = self.raw_t(alpha_name, dtype)?; // [C]
        let alpha = a.reshape((1, a.dim(0)?, 1))?;
        let xa = x.broadcast_mul(&alpha)?;
        let s = xa.sin()?.sqr()?;
        let inv = alpha.affine(1.0, SNAKE_EPS)?.recip()?;
        x.add(&s.broadcast_mul(&inv)?)
    }

    /// Causal [`ResBlock`] (`causal=True`): for each dilation, Snake -> left-causal
    /// dilated conv -> Snake -> left-causal conv (dilation 1), added to the residual.
    fn resblock(&self, x: &Tensor, prefix: &str, dtype: DType) -> Result<Tensor> {
        let mut x = x.clone();
        for (idx, &dil) in RESBLOCK_DILATIONS.iter().enumerate() {
            let xt = self.snake(&x, &format!("{prefix}.activations1.{idx}.alpha"), dtype)?;
            let xt = self.causal_conv(
                &xt,
                &format!("{prefix}.convs1.{idx}.weight"),
                &format!("{prefix}.convs1.{idx}.bias"),
                dil,
                true,
                dtype,
            )?;
            let xt = self.snake(&xt, &format!("{prefix}.activations2.{idx}.alpha"), dtype)?;
            let xt = self.causal_conv(
                &xt,
                &format!("{prefix}.convs2.{idx}.weight"),
                &format!("{prefix}.convs2.{idx}.bias"),
                1,
                true,
                dtype,
            )?;
            x = (x + xt)?;
        }
        Ok(x)
    }

    /// Causal `decode(mel, source)`. `mel` is `[1,80,T]`; `source` is the dumped
    /// SineGen waveform `[1,1,L]`. Computes `s_stft = _stft(source)` then the
    /// causal upsample/fusion/resblock stack + iSTFT head. Returns `[1, L]`.
    pub fn decode(&self, mel: &Tensor, source: &Tensor) -> Result<Tensor> {
        let dt = DType::F32;
        let s_stft = self.stft(source)?; // [1,18,TT]

        // conv_pre: CausalConv1d(k=5, causal_type='right').
        let mut x = self.causal_conv(mel, "conv_pre.weight", "conv_pre.bias", 1, false, dt)?;
        let _ = CONV_PRE_LOOK_RIGHT; // pad amount = k-1 = look_right; documented above.

        let num_upsamples = UPSAMPLE_RATES.len();
        for i in 0..num_upsamples {
            x = leaky_relu(&x, LRELU_SLOPE)?;
            x = self.ups(
                &x,
                &format!("ups.{i}.weight"),
                &format!("ups.{i}.bias"),
                UPSAMPLE_RATES[i],
                dt,
            )?;

            if i == num_upsamples - 1 {
                x = reflection_pad_left1(&x)?;
            }

            // Source-branch fusion.
            let (sd_stride, _) = source_down_params(i);
            let si = self.source_down(
                &s_stft,
                &format!("source_downs.{i}.weight"),
                &format!("source_downs.{i}.bias"),
                sd_stride,
                dt,
            )?;
            let si = self.resblock(&si, &format!("source_resblocks.{i}"), dt)?;
            x = (x + si)?;

            // Mean over the 3 kernel ResBlocks at this stage.
            let mut acc: Option<Tensor> = None;
            for j in 0..3 {
                let rb = self.resblock(&x, &format!("resblocks.{}", i * 3 + j), dt)?;
                acc = Some(match acc {
                    None => rb,
                    Some(a) => (a + rb)?,
                });
            }
            x = (acc.unwrap() / 3.0)?;
        }

        // Pre-conv_post activation is F.leaky_relu(x) with torch's default 0.01.
        x = leaky_relu(&x, 0.01)?;
        // conv_post: CausalConv1d(k=7, causal_type='left').
        x = self.causal_conv(&x, "conv_post.weight", "conv_post.bias", 1, true, dt)?;

        // Head: magnitude = exp(x[:9]) (clipped 1e2), phase = sin(x[9:]).
        let magnitude = x.narrow(1, 0, N_BINS)?.exp()?.clamp(0.0, 1e2)?;
        let phase = x.narrow(1, N_BINS, N_BINS)?.sin()?;
        let real = magnitude.mul(&phase.cos()?)?;
        let imag = magnitude.mul(&phase.sin()?)?;

        let wav = istft(&real, &imag, &self.dev)?;
        wav.clamp(-AUDIO_LIMIT, AUDIO_LIMIT)
    }
}

// ---- free helpers -----------------------------------------------------------

/// `source_downs[i]` conv stride. `downsample_cum_rates[::-1] = [15, 3, 1]`; the
/// `u>1` stages are `CausalConv1dDownSample(stride=u, kernel=2u)`, the `u==1` stage
/// is a plain `CausalConv1d(k=1)` (stride 1, handled by `source_down` with `stride=1`
/// -> left-pad 0). Returns `(stride, _)`.
fn source_down_params(i: usize) -> (usize, ()) {
    match i {
        0 => (15, ()),
        1 => (3, ()),
        2 => (1, ()),
        _ => unreachable!(),
    }
}

/// Nearest-neighbour upsample of `[B,C,T]` by integer `scale` -> `[B,C,T*scale]`
/// (each time-step repeated `scale` times — `nn.Upsample(mode='nearest')` for an
/// integer `scale_factor`).
fn upsample_nearest1d(x: &Tensor, scale: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    x.unsqueeze(D::Minus1)? // [B,C,T,1]
        .broadcast_as((b, c, t, scale))?
        .contiguous()?
        .reshape((b, c, t * scale))
}

/// LeakyReLU as `relu(x) - slope*relu(-x)`.
fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = x.neg()?.relu()?.affine(slope, 0.0)?;
    pos.sub(&neg)
}

/// `ReflectionPad1d((1,0))`: prepend a reflection of the second sample (`x[...,1]`).
fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let left = x.narrow(D::Minus1, 1, 1)?;
    Tensor::cat(&[&left, x], D::Minus1)
}
