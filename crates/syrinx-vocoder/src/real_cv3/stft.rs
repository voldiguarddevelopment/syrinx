//! The forward source `_stft` and the iSTFT head of the CV3 HiFT vocoder. Split out of
//! `real_cv3.rs` unchanged; the `stft` method extends [`super::Cv3Hift`].

use candle_core::{Device, Result, Tensor, D};

use super::{Cv3Hift, HOP, N_BINS, N_FFT};

impl Cv3Hift {
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
        // Defense: `center=True` reflect padding reads `sig[1..=pad]` and `sig[l-1-pad..]`,
        // so it needs at least `n_fft` samples. A shorter `source` means the upstream mel
        // is empty/degenerate (e.g. the LM AR loop produced no speech tokens) — fail with a
        // clear error instead of indexing out of bounds inside the reflect-pad loop.
        if l < N_FFT {
            return Err(candle_core::Error::Msg(format!(
                "Cv3Hift::stft: source length {l} < n_fft {N_FFT}; the source/mel is empty or \
                 degenerate (did the speech-token generation produce zero tokens?)"
            )));
        }
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
}

/// iSTFT reproducing `torch.istft` (center=True, onesided, hann window) for
/// `n_fft=16`, `hop=4`. `real`/`imag` are `[1, 9, T]`; returns `[1, L]`.
pub(super) fn istft(real: &Tensor, imag: &Tensor, dev: &Device) -> Result<Tensor> {
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

    // Guard `frames == 0` — `frames - 1` would underflow `usize` into a giant allocation
    // (the same guard as the CV2 istft; this twin was missed in the first pass).
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
