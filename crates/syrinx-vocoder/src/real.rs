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

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

/// Smallest weight (in elements) worth quantizing in [`HiftVocoder::load_quantized`].
/// Below this the Q4_0 per-block scale overhead dominates and the tiny weight is left
/// f32 (keeps small/sensitive params — the head `conv_post`-adjacent tensors — exact).
const QUANT_MIN_ELEMS: usize = 4096;

const N_FFT: usize = 16;
const HOP: usize = 4;
const N_BINS: usize = N_FFT / 2 + 1; // 9 (onesided)
const AUDIO_LIMIT: f64 = 0.99;
const LRELU_SLOPE: f64 = 0.1;
const SNAKE_EPS: f64 = 1e-9;
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];

/// A `Q4_0`-quantized weight stored for **dequant-on-fetch**: the forward is unchanged
/// (it asks [`HiftVocoder::g`] for an f32 tensor and gets one back), only the *resident*
/// storage is the int4 `QTensor`. The original logical shape (conv kernels are 3-D
/// `[out,in,k]`) is kept so `g` can restore it after dequantizing the 2-D block store.
struct QStore {
    qt: QTensor,
    shape: Vec<usize>,
}

/// The real HiFT vocoder, loaded from a folded fp32 safetensors checkpoint.
///
/// Two precisions share one struct + one forward, like the LM/flow:
///   * **fp32 (default, parity)** — [`HiftVocoder::load`], every weight in `w` as f32,
///     byte-unchanged.
///   * **int4 (`load_quantized`)** — the large weight *matrices* (conv kernels reshaped
///     to `[out, in*k]` and any linear) are stored Q4_0 in `qw` and dequantized on fetch;
///     biases, Snake `alpha`s and other 1-D / small params stay f32 in `w`.
pub struct HiftVocoder {
    w: HashMap<String, Tensor>,
    /// Q4_0 dequant-on-fetch weights (empty in the fp32 build).
    qw: HashMap<String, QStore>,
    /// Sum of the `QTensor` storage bytes realized by quantization (0 in the fp32 build).
    quant_bytes: usize,
    dev: Device,
}

/// Realized weight footprint of a loaded [`HiftVocoder`], split into the int4-quantized
/// and retained dense (f32) parts.
#[derive(Debug, Clone, Copy)]
pub struct HiftFootprint {
    /// Bytes held by the `Q4_0` quantized weights (0 in the fp32 build).
    pub quant_bytes: usize,
    /// Bytes held by the retained dense f32 weights (biases, alphas, small/odd kernels).
    pub dense_bytes: usize,
    /// Number of weights quantized to int4.
    pub n_quantized: usize,
}

impl HiftFootprint {
    /// Total realized weight bytes (`quant + dense`).
    pub fn total_bytes(&self) -> usize {
        self.quant_bytes + self.dense_bytes
    }
    /// Total realized weight footprint in mebibytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes() as f64 / (1024.0 * 1024.0)
    }
}

impl HiftVocoder {
    /// Load the folded fp32 checkpoint (`hift_fp32.safetensors`) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self { w, qw: HashMap::new(), quant_bytes: 0, dev })
    }

    /// Load the same checkpoint but quantize the large weight **matrices** to GGML `Q4_0`
    /// for a smaller vocoder footprint (the README size goal; goal #3 of the quant pass).
    ///
    /// Quantized (dequant-on-fetch, [`QStore`]): every weight whose flattened
    /// `[out, in*k]` matrix has a 32-aligned inner dim and ≥ [`QUANT_MIN_ELEMS`] elements —
    /// i.e. the HiFi-GAN upsample `ConvTranspose1d` kernels and the Snake `ResBlock`
    /// `Conv1d` kernels (the footprint bulk). They are reshaped to a 2-D block store, int4
    /// quantized, and reconstructed to their original `[out,in,k]` shape on lookup, so the
    /// `decode`/`f0_predict` math is byte-for-byte the fp32 code path on the dequantized
    /// weight.
    ///
    /// Kept dense (f32): all biases, the Snake channel `alpha`s, and any small or
    /// non-32-aligned kernel (e.g. `conv_pre` `[512,80,7]`, `in*k = 560`).
    ///
    /// NOTE: this **does** quantize convolution kernels (not just pure linears) — the only
    /// way the vocoder's conv-dominated weights yield a real size win. int4 on conv
    /// kernels trades quality for size; the on-box SIM-o eval is the arbiter, and the fp32
    /// [`load`](Self::load) path stays available for full quality.
    pub fn load_quantized(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        let mut w = HashMap::with_capacity(raw.len());
        let mut qw = HashMap::new();
        let mut quant_bytes = 0usize;
        for (k, v) in raw {
            let vf = v.to_dtype(DType::F32)?;
            if let Some(qs) = maybe_quantize_weight(&vf)? {
                quant_bytes += qs.qt.storage_size_in_bytes();
                qw.insert(k, qs);
            } else {
                w.insert(k, vf);
            }
        }
        Ok(Self { w, qw, quant_bytes, dev })
    }

    /// Realized weight footprint (quantized + dense) of this loaded vocoder.
    pub fn footprint(&self) -> HiftFootprint {
        let dense_bytes: usize = self
            .w
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        HiftFootprint {
            quant_bytes: self.quant_bytes,
            dense_bytes,
            n_quantized: self.qw.len(),
        }
    }

    fn g(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.w.get(name) {
            return Ok(t.clone());
        }
        if let Some(q) = self.qw.get(name) {
            // Dequantize the 2-D block store and restore the original logical shape.
            return q.qt.dequantize(&self.dev)?.reshape(q.shape.clone());
        }
        Err(candle_core::Error::Msg(format!("missing weight: {name}")))
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

/// Decide whether a weight `vf` is a large block-aligned matrix worth quantizing to
/// `Q4_0` (dequant-on-fetch), returning its [`QStore`] if so, else `None` (keep f32).
///
/// A conv kernel `[out,in,k]` (or a 2-D linear `[out,in]`) flattens to `[out, in*k]`;
/// it is quantized when `in*k` is a multiple of the 32-element `Q4_0` block and the
/// tensor has at least [`QUANT_MIN_ELEMS`] elements. 1-D weights (biases, Snake `alpha`s)
/// and small/odd kernels stay dense.
fn maybe_quantize_weight(vf: &Tensor) -> Result<Option<QStore>> {
    let dims = vf.dims().to_vec();
    if dims.len() < 2 || vf.elem_count() < QUANT_MIN_ELEMS {
        return Ok(None);
    }
    let out = dims[0];
    let inner: usize = dims[1..].iter().product(); // in * k (conv) or in (linear)
    if inner % GgmlDType::Q4_0.block_size() != 0 {
        return Ok(None);
    }
    let mat = vf.reshape((out, inner))?; // 2-D block store
    let qt = QTensor::quantize(&mat, GgmlDType::Q4_0)?;
    Ok(Some(QStore { qt, shape: dims }))
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
    // Guard `frames == 0` — `frames - 1` would underflow `usize` into a giant allocation.
    if frames == 0 {
        return Err(candle_core::Error::Msg("istft: zero-frame input (empty mel)".to_string()));
    }
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
