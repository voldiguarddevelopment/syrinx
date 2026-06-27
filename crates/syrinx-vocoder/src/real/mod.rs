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
//!
//! ## Module layout
//! Pure structural split of the original single `real.rs` (logic byte-unchanged):
//!   * this `mod.rs` — the [`HiftVocoder`] struct, loaders, conv primitives, `decode`;
//!   * [`source`] — the F0 predictor + `m_source` harmonic-merge weights;
//!   * [`resblock`] — Snake activation, the HiFi-GAN ResBlock, upsample helpers;
//!   * [`istft`] — the iSTFT-via-inverse-DFT head;
//!   * [`quant`] — the int4 `Q4_0` quantizer.

use candle_core::{safetensors, DType, Device, Result, Tensor};
use std::collections::HashMap;

mod istft;
mod quant;
mod resblock;
mod source;

use istft::istft;
use quant::{maybe_quantize_weight, QStore};
use resblock::{leaky_relu, reflection_pad_left1, ups_pad_stride};
use source::{source_down_pad_stride, source_resblock_kernel};

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
    /// `[out, in*k]` matrix has a 32-aligned inner dim and ≥ [`quant::QUANT_MIN_ELEMS`]
    /// elements — i.e. the HiFi-GAN upsample `ConvTranspose1d` kernels and the Snake
    /// `ResBlock` `Conv1d` kernels (the footprint bulk). They are reshaped to a 2-D block
    /// store, int4 quantized, and reconstructed to their original `[out,in,k]` shape on
    /// lookup, so the `decode`/`f0_predict` math is byte-for-byte the fp32 code path on the
    /// dequantized weight.
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
}
