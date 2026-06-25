//! Real CosyVoice2 **HiFT vocoder** forward via Candle (CPU fp32) — the mel ->
//! 24 kHz waveform synthesis behind the toy-reference parity.
//!
//! This ports the non-causal [`HiFTGenerator`] `decode()` path from
//! `cosyvoice/hifigan/generator.py` (HiFTNet = Neural Source Filter + iSTFTNet):
//! a HiFi-GAN transposed-conv upsampling stack with Snake-activated residual
//! blocks, a source/F0 STFT branch fused in at every stage, and an iSTFT head.
//! Gated behind the `real` cargo feature + on-disk weights, mirroring syrinx-lm.
//!
//! ## Architecture (from `cosyvoice2.yaml` `hift:` + `hift.pt`)
//! - `in_channels=80`, `base_channels=512`, `sampling_rate=24000`
//! - `conv_pre`: Conv1d(80 -> 512, k7, pad3)
//! - 3 upsample stages, `upsample_rates=[8,5,3]`, kernels `[16,11,7]`:
//!   leaky_relu(0.1) -> ConvTranspose1d -> (last stage: ReflectionPad1d(1,0)) ->
//!   add `source_resblocks[i](source_downs[i](s_stft))` -> mean of 3 ResBlocks
//! - `conv_post`: leaky_relu -> Conv1d(ch -> n_fft+2 = 18, k7, pad3)
//! - head: `magnitude = exp(x[:9])`, `phase = sin(x[9:])`, then iSTFT
//!   (n_fft=16, hop=4, hann window) and clamp to `±audio_limit (0.99)`.
//!
//! ## Weight loading
//! The PyTorch checkpoint stores every conv under the `weight_norm`
//! parametrization (`...parametrizations.weight.original0/1`). The reference
//! dumper folds those into plain `.weight` tensors offline (too large/awkward to
//! vendor), so this loader sees ordinary `Conv1d`/`ConvTranspose1d` weights.
//!
//! ## Determinism / inputs
//! The source/F0 path (`SineGen2` + additive noise) is *stochastic* in the real
//! model (random initial phase and Gaussian noise), so a pure mel->waveform map
//! is not reproducible across torch/Candle RNGs. The honest deterministic unit is
//! `decode(mel, s)`: every learned weight and the entire iSTFT live here. The
//! reference therefore captures the source STFT `s_stft` (`[1, 18, T]`) as a fixed
//! input, and this port reproduces `decode` exactly. The F0 predictor (fully
//! deterministic) is also ported and checked separately.

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

const N_FFT: usize = 16;
const HOP: usize = 4;
const N_BINS: usize = N_FFT / 2 + 1; // 9 (onesided)
const AUDIO_LIMIT: f64 = 0.99;
const LRELU_SLOPE: f64 = 0.1;
const SNAKE_EPS: f64 = 1e-9;
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];

/// The real HiFT vocoder, loaded from a folded fp32 safetensors checkpoint.
pub struct HiftVocoder {
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl HiftVocoder {
    /// Load the folded fp32 checkpoint (`hift_fp32.safetensors`) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self { w, dev })
    }

    fn g(&self, name: &str) -> Result<Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
            .cloned()
    }

    /// Plain `Conv1d`: `[B, Cin, T]` -> `[B, Cout, T']` with the given padding,
    /// stride and dilation, adding the bias broadcast over time.
    fn conv1d(
        &self,
        x: &Tensor,
        wname: &str,
        bname: &str,
        pad: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let weight = self.g(wname)?; // [Cout, Cin, K]
        let bias = self.g(bname)?; // [Cout]
        let y = x.conv1d(&weight, pad, stride, dilation, 1)?;
        let b = bias.reshape((1, bias.dim(0)?, 1))?;
        y.broadcast_add(&b)
    }

    /// `ConvTranspose1d`: weight layout `[Cin, Cout, K]` (torch convention, which
    /// candle's `conv_transpose1d` shares), with `output_padding=0`.
    fn conv_transpose1d(
        &self,
        x: &Tensor,
        wname: &str,
        bname: &str,
        pad: usize,
        stride: usize,
    ) -> Result<Tensor> {
        let weight = self.g(wname)?; // [Cin, Cout, K]
        let bias = self.g(bname)?; // [Cout]
        let y = x.conv_transpose1d(&weight, pad, 0, stride, 1, 1)?;
        let b = bias.reshape((1, bias.dim(0)?, 1))?;
        y.broadcast_add(&b)
    }

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
    fn resblock(&self, x: &Tensor, prefix: &str, kernel: usize) -> Result<Tensor> {
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

    /// Run the deterministic `decode(mel, s_stft)` path.
    ///
    /// `mel` is `[1, 80, T]`; `s_stft` is the source STFT `[1, 18, T_src]`
    /// (real then imag, n_fft+2 channels). Returns the waveform `[1, L]`.
    pub fn decode(&self, mel: &Tensor, s_stft: &Tensor) -> Result<Tensor> {
        let mut x = self.conv1d(mel, "conv_pre.weight", "conv_pre.bias", 3, 1, 1)?;

        let num_upsamples = UPSAMPLE_RATES.len();
        for i in 0..num_upsamples {
            x = leaky_relu(&x, LRELU_SLOPE)?;
            let (pad, stride) = ups_pad_stride(i);
            x = self.conv_transpose1d(
                &x,
                &format!("ups.{i}.weight"),
                &format!("ups.{i}.bias"),
                pad,
                stride,
            )?;

            if i == num_upsamples - 1 {
                x = reflection_pad_left1(&x)?;
            }

            // Source-branch fusion.
            let (sd_pad, sd_stride) = source_down_pad_stride(i);
            let si = self.conv1d(
                s_stft,
                &format!("source_downs.{i}.weight"),
                &format!("source_downs.{i}.bias"),
                sd_pad,
                sd_stride,
                1,
            )?;
            let src_kernel = source_resblock_kernel(i);
            let si = self.resblock(&si, &format!("source_resblocks.{i}"), src_kernel)?;
            x = (x + si)?;

            // Mean over the 3 kernel ResBlocks at this stage.
            let mut acc: Option<Tensor> = None;
            for (j, &k) in RESBLOCK_KERNELS.iter().enumerate() {
                let rb = self.resblock(&x, &format!("resblocks.{}", i * 3 + j), k)?;
                acc = Some(match acc {
                    None => rb,
                    Some(a) => (a + rb)?,
                });
            }
            x = (acc.unwrap() / RESBLOCK_KERNELS.len() as f64)?;
        }

        // NOTE: the pre-`conv_post` activation is `F.leaky_relu(x)` with no slope
        // argument in the reference, i.e. torch's *default* 0.01 — not the model's
        // `lrelu_slope` (0.1) used inside the upsample loop.
        x = leaky_relu(&x, 0.01)?;
        x = self.conv1d(&x, "conv_post.weight", "conv_post.bias", 3, 1, 1)?;

        // Head: magnitude = exp(x[:9]) (clipped at 1e2), phase = sin(x[9:]).
        // magnitude = exp(x[:9]), clipped to a 1e2 max (exp output is always >= 0).
        let magnitude = x.narrow(1, 0, N_BINS)?.exp()?.clamp(0.0, 1e2)?;
        let phase = x.narrow(1, N_BINS, N_BINS)?.sin()?;
        let real = magnitude.mul(&phase.cos()?)?;
        let imag = magnitude.mul(&phase.sin()?)?;

        let wav = istft(&real, &imag, &self.dev)?;
        wav.clamp(-AUDIO_LIMIT, AUDIO_LIMIT)
    }

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
}

/// `(padding, stride)` for upsample stage `i` (`padding=(k-u)//2`).
fn ups_pad_stride(i: usize) -> (usize, usize) {
    let kernels = [16usize, 11, 7];
    let u = UPSAMPLE_RATES[i];
    ((kernels[i] - u) / 2, u)
}

/// `(padding, stride)` for the `source_downs[i]` conv (downsamples the source STFT
/// to the channel/time resolution of upsample stage `i`).
fn source_down_pad_stride(i: usize) -> (usize, usize) {
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
fn source_resblock_kernel(i: usize) -> usize {
    [7, 7, 11][i]
}

/// LeakyReLU: `x` where `x>=0`, `slope*x` otherwise, as `relu(x) - slope*relu(-x)`.
fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu()?; // max(x, 0)
    let neg = x.neg()?.relu()?.affine(slope, 0.0)?; // slope * max(-x, 0)
    pos.sub(&neg)
}

/// ELU: `x` where `x>0`, `exp(x)-1` otherwise (alpha=1).
fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?; // x for x>0, else 0
    // negative branch: min(x,0) -> exp(min)-1 ; for x>0 exp(0)-1 = 0.
    let neg_in = x.neg()?.relu()?.neg()?; // min(x, 0)
    let neg = neg_in.exp()?.affine(1.0, -1.0)?;
    pos.add(&neg)
}

/// ReflectionPad1d((1, 0)): prepend a reflection of the second sample on the left.
/// torch reflects across the boundary, so the prepended value is `x[..., 1]`.
fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let left = x.narrow(D::Minus1, 1, 1)?; // reflect across the boundary: index 1
    Tensor::cat(&[&left, x], D::Minus1)
}

/// iSTFT reproducing `torch.istft` (center=True, onesided, normalized=False, hann
/// window) for `n_fft=16`, `hop=4`. `real`/`imag` are `[1, 9, T]`.
///
/// Per frame: reconstruct the full length-16 spectrum by Hermitian symmetry,
/// inverse real DFT (`x[n] = (1/N) Σ_k Re{X[k] e^{j2πkn/N}}`), window, overlap-add,
/// then divide by the overlap-added window² envelope and trim `n_fft/2` each side.
fn istft(real: &Tensor, imag: &Tensor, dev: &Device) -> Result<Tensor> {
    let (b, bins, frames) = real.dims3()?;
    debug_assert_eq!(b, 1);
    debug_assert_eq!(bins, N_BINS);

    // Hann window (fftbins / periodic): w[n] = 0.5 - 0.5 cos(2π n / N).
    let window: Vec<f32> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos())
        .collect();

    // Inverse-DFT basis from the onesided bins to the N_FFT time samples, folding
    // in Hermitian symmetry. For time index n:
    //   x[n] = (1/N) [ Re X[0]
    //                + 2 Σ_{k=1}^{N/2-1} (Re X[k] cos θ - Im X[k] sin θ)
    //                + Re X[N/2] cos(π n) ]
    // where θ = 2π k n / N. Bins 0 and N/2 are real (no symmetric partner), the
    // interior bins each count twice via their conjugate at N-k.
    // Build cos/sin matrices [N_FFT, N_BINS] with the symmetry weights folded in.
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

    // real/imag: [9, T] (drop batch). frame[n, t] = Σ_k cos*Re - sin*Im.
    let re = real.squeeze(0)?; // [9, T]
    let im = imag.squeeze(0)?; // [9, T]
    // frames_time[n, t] = (cos_mat @ re - sin_mat @ im)[n, t]  -> [N_FFT, T]
    let frames_time = cos_mat.matmul(&re)?.sub(&sin_mat.matmul(&im)?)?; // [16, T]

    // Apply window per time-sample-in-frame.
    let win_t = Tensor::from_vec(window.clone(), (N_FFT, 1), dev)?;
    let frames_win = frames_time.broadcast_mul(&win_t)?; // [16, T]

    // Overlap-add into the full (pre-trim) signal, plus the window² envelope.
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
