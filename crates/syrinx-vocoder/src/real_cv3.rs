//! Real **CosyVoice3** HiFT vocoder forward via Candle (CPU) — the
//! `CausalHiFTGenerator` mel -> 24 kHz waveform path, parity-verified against the
//! dumped CV3 reference. Gated behind the `real` cargo feature + on-disk weights,
//! mirroring [`crate::real`] (the CV2 port).
//!
//! ## What CV3 changes vs CV2 ([`crate::real::HiftVocoder`])
//! The architecture is the same HiFTNet (NSF + iSTFTNet) skeleton — `conv_pre`,
//! 3 upsample stages with source/F0 STFT fusion, Snake ResBlocks, a `conv_post`
//! magnitude/phase head and an iSTFT — but two deltas matter:
//!
//! 1. **Causal convolutions.** Every conv is a `CausalConv1d`
//!    (`cosyvoice/transformer/convolution.py`), which pads *entirely on one side*
//!    rather than symmetrically. For kernel `k`, dilation `d`, stride 1 the total
//!    pad is `causal_padding = (k*d - d) - ... = d*(k-1)` (the source formula
//!    `int((k*d-d)/2)*2 + (k+1)%2` reduces to exactly `d*(k-1)` for every kernel we
//!    use). It is placed on the **left** (`causal_type='left'`) for all convs
//!    except `conv_pre` and `f0_predictor.condnet[0]`, which look ahead and pad on
//!    the **right** (`causal_type='right'`, `conv_pre_look_right=4`). Output length
//!    equals input length (`assert x.shape[2] == input_timestep`).
//!    The upsample stages are **nearest-neighbour upsample + a left-causal conv**
//!    (`CausalConv1dUpsample`), not the CV2 `ConvTranspose1d`.
//!
//! 2. **float64 f0_predictor.** `CausalConvRNNF0Predictor` is run in float64
//!    (`self.f0_predictor.to(torch.float64)`; "precision is crucial for causal
//!    inference"), so the whole condnet + classifier is computed in f64 here.
//!
//! ## Weight loading
//! Unlike the CV2 checkpoint (folded offline), `hift_fp32.safetensors` here still
//! carries the `weight_norm` parametrization (`...parametrizations.weight.original0`
//! = g `[out,1,1]`, `original1` = v `[out,in,k]`). This loader folds them on fetch:
//! `weight = g * v / ||v||` with the norm over all dims but `0` — and folds in the
//! **requested dtype** so the f64 f0 path matches torch's `to(float64)` recompute.
//! Plain (non-parametrized) weights — `source_downs.*`, `classifier`,
//! `m_source.l_linear` — are used as-is.
//!
//! ## Determinism / inputs
//! As in CV2, the source/F0 branch's `SineGen` is stochastic (seeded Gaussian +
//! random phase) and not reproducible across torch/Candle RNGs. The reference dumps
//! the SineGen **`source`** waveform `[1,1,97920]`; [`Cv3Hift::decode`] consumes it
//! verbatim, computes its STFT (`_stft`), and reproduces the deterministic decode.

use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;

const N_FFT: usize = 16;
const HOP: usize = 4;
const N_BINS: usize = N_FFT / 2 + 1; // 9 (onesided)
const AUDIO_LIMIT: f64 = 0.99;
const LRELU_SLOPE: f64 = 0.1;
const SNAKE_EPS: f64 = 1e-9;
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
/// ResBlock dilations (`resblock_dilation_sizes = [[1,3,5], ...]`).
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
/// `conv_pre_look_right`: `conv_pre` is `CausalConv1d(k=look_right+1, type='right')`.
const CONV_PRE_LOOK_RIGHT: usize = 4;

/// The real CosyVoice3 (`CausalHiFTGenerator`) HiFT vocoder, loaded from the
/// `weight_norm`-parametrized fp32 safetensors checkpoint.
pub struct Cv3Hift {
    /// Raw safetensors tensors (kept f32; folded/cast per-fetch).
    raw: HashMap<String, Tensor>,
    dev: Device,
}

impl Cv3Hift {
    /// Load `hift_fp32.safetensors` (still `weight_norm`-parametrized) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = candle_core::safetensors::load(path, &dev)?;
        Ok(Self { raw, dev })
    }

    /// Fetch a plain (non-parametrized) tensor cast to `dtype`.
    fn raw_t(&self, name: &str, dtype: DType) -> Result<Tensor> {
        self.raw
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor: {name}")))?
            .to_dtype(dtype)
    }

    /// Resolve a conv `weight` in `dtype`, folding `weight_norm` if parametrized.
    ///
    /// `name` is the logical `"<prefix>.weight"`. If
    /// `"<prefix>.parametrizations.weight.original0/1"` exist they are folded:
    /// `weight = g * v / ||v||_2`, the norm taken over every dim except `0`
    /// (torch `weight_norm(dim=0)`). Done in `dtype` to mirror torch's
    /// `module.to(float64)` recompute on the f0 path. Otherwise the plain tensor.
    fn weight(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let g_key = format!("{base}.parametrizations.weight.original0");
        let v_key = format!("{base}.parametrizations.weight.original1");
        match (self.raw.get(&g_key), self.raw.get(&v_key)) {
            (Some(g), Some(v)) => {
                let g = g.to_dtype(dtype)?; // [out,1,1]
                let v = v.to_dtype(dtype)?; // [out,in,k]
                // ||v||_2 over (in, k), keepdim -> [out,1,1].
                let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
                v.broadcast_mul(&g)?.broadcast_div(&norm)
            }
            _ => self.raw_t(name, dtype),
        }
    }

    /// `CausalConv1d` (stride 1): pad `dilation*(k-1)` entirely on one side
    /// (`left` true = `causal_type='left'`, false = `'right'`), conv with no
    /// internal padding. Output length equals input length.
    fn causal_conv(
        &self,
        x: &Tensor,
        wname: &str,
        bname: &str,
        dilation: usize,
        left: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let w = self.weight(wname, dtype)?; // [out,in,k]
        let b = self.raw_t(bname, dtype)?; // [out]
        let k = w.dim(2)?;
        let pad = dilation * (k - 1);
        let xp = if left {
            x.pad_with_zeros(2, pad, 0)?
        } else {
            x.pad_with_zeros(2, 0, pad)?
        };
        let y = xp.conv1d(&w, 0, 1, dilation, 1)?;
        let b = b.reshape((1, b.dim(0)?, 1))?;
        y.broadcast_add(&b)
    }

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

    /// `_stft(source)` -> `s_stft` `[1, 18, TT]` (real then imag, `n_fft+2` chans).
    ///
    /// `torch.stft(x, n_fft=16, hop=4, win=16, window=hann, center=True,
    /// pad_mode='reflect', return_complex=True)` then `view_as_real`, concatenated
    /// over the channel dim. `source` is `[1,1,L]` or `[1,L]`.
    pub fn stft(&self, source: &Tensor) -> Result<Tensor> {
        let x = if source.dims().len() == 3 {
            source.squeeze(1)?
        } else {
            source.clone()
        };
        let l = x.dim(D::Minus1)?;
        let sig: Vec<f32> = x.flatten_all()?.to_vec1::<f32>()?;

        // center=True: reflect-pad n_fft/2 on each side (no edge repeat).
        let pad = N_FFT / 2;
        let mut padded = vec![0f64; l + 2 * pad];
        for n in 0..pad {
            padded[pad - 1 - n] = sig[n + 1] as f64; // left reflect: x[1..=pad]
            padded[pad + l + n] = sig[l - 2 - n] as f64; // right reflect: x[l-2..]
        }
        for n in 0..l {
            padded[pad + n] = sig[n] as f64;
        }

        let frames = (padded.len() - N_FFT) / HOP + 1;
        // Hann periodic window.
        let window: Vec<f64> = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
            .collect();

        let mut re = vec![0f32; N_BINS * frames];
        let mut im = vec![0f32; N_BINS * frames];
        for t in 0..frames {
            let start = t * HOP;
            for k in 0..N_BINS {
                let mut sr = 0f64;
                let mut si = 0f64;
                for n in 0..N_FFT {
                    let v = window[n] * padded[start + n];
                    let theta = 2.0 * std::f64::consts::PI * (k * n) as f64 / N_FFT as f64;
                    sr += v * theta.cos();
                    si -= v * theta.sin();
                }
                re[k * frames + t] = sr as f32;
                im[k * frames + t] = si as f32;
            }
        }
        let re = Tensor::from_vec(re, (1, N_BINS, frames), &self.dev)?;
        let im = Tensor::from_vec(im, (1, N_BINS, frames), &self.dev)?;
        Tensor::cat(&[&re, &im], 1)
    }

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

/// ELU (alpha=1): `x` for `x>0`, else `exp(x)-1`.
fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg_in = x.neg()?.relu()?.neg()?; // min(x,0)
    let neg = neg_in.exp()?.affine(1.0, -1.0)?;
    pos.add(&neg)
}

/// `ReflectionPad1d((1,0))`: prepend a reflection of the second sample (`x[...,1]`).
fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let left = x.narrow(D::Minus1, 1, 1)?;
    Tensor::cat(&[&left, x], D::Minus1)
}

/// iSTFT reproducing `torch.istft` (center=True, onesided, hann window) for
/// `n_fft=16`, `hop=4`. `real`/`imag` are `[1, 9, T]`; returns `[1, L]`.
fn istft(real: &Tensor, imag: &Tensor, dev: &Device) -> Result<Tensor> {
    let (b, bins, frames) = real.dims3()?;
    debug_assert_eq!(b, 1);
    debug_assert_eq!(bins, N_BINS);

    let window: Vec<f32> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos())
        .collect();

    // Inverse-DFT basis folding in Hermitian symmetry (bins 0 & N/2 weight 1, else 2).
    let mut cos_w = vec![0f32; N_FFT * N_BINS];
    let mut sin_w = vec![0f32; N_FFT * N_BINS];
    for n in 0..N_FFT {
        for k in 0..N_BINS {
            let mult = if k == 0 || k == N_FFT / 2 { 1.0 } else { 2.0 };
            let theta = 2.0 * std::f32::consts::PI * (k * n) as f32 / N_FFT as f32;
            cos_w[n * N_BINS + k] = mult * theta.cos() / N_FFT as f32;
            sin_w[n * N_BINS + k] = mult * theta.sin() / N_FFT as f32;
        }
    }
    let cos_mat = Tensor::from_vec(cos_w, (N_FFT, N_BINS), dev)?;
    let sin_mat = Tensor::from_vec(sin_w, (N_FFT, N_BINS), dev)?;

    let re = real.squeeze(0)?; // [9, T]
    let im = imag.squeeze(0)?;
    let frames_time = cos_mat.matmul(&re)?.sub(&sin_mat.matmul(&im)?)?; // [16, T]

    let win_t = Tensor::from_vec(window.clone(), (N_FFT, 1), dev)?;
    let frames_win = frames_time.broadcast_mul(&win_t)?; // [16, T]

    let out_len = HOP * (frames - 1) + N_FFT;
    let frames_win = frames_win.to_vec2::<f32>()?; // [16][T]
    let mut ytmp = vec![0f64; out_len];
    let mut wsum = vec![0f64; out_len];
    let win_sq: Vec<f64> = window.iter().map(|&w| (w as f64) * (w as f64)).collect();
    for t in 0..frames {
        let start = t * HOP;
        for n in 0..N_FFT {
            ytmp[start + n] += frames_win[n][t] as f64;
            wsum[start + n] += win_sq[n];
        }
    }
    for i in 0..out_len {
        if wsum[i] > 1e-11 {
            ytmp[i] /= wsum[i];
        }
    }
    let pad = N_FFT / 2;
    let trimmed: Vec<f32> = ytmp[pad..out_len - pad].iter().map(|&v| v as f32).collect();
    let len = trimmed.len();
    Tensor::from_vec(trimmed, (1, len), dev)
}
