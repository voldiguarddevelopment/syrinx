//! The iSTFT head: `torch.istft` reproduced via an explicit inverse real-DFT +
//! overlap-add. Split out of `real.rs` unchanged.

use candle_core::{Device, Result, Tensor};

use super::{HOP, N_BINS, N_FFT};

/// iSTFT reproducing `torch.istft` (center=True, onesided, normalized=False, hann
/// window) for `n_fft=16`, `hop=4`. `real`/`imag` are `[1, 9, T]`.
///
/// Per frame: reconstruct the full length-16 spectrum by Hermitian symmetry,
/// inverse real DFT (`x[n] = (1/N) Σ_k Re{X[k] e^{j2πkn/N}}`), window, overlap-add,
/// then divide by the overlap-added window² envelope and trim `n_fft/2` each side.
pub(super) fn istft(real: &Tensor, imag: &Tensor, dev: &Device) -> Result<Tensor> {
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
