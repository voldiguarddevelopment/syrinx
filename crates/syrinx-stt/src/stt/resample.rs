//! Band-limited mono resampling to 16 kHz for the Whisper front end.
//!
//! A Lanczos-windowed sinc (radius 16 lobes); when down-sampling the cutoff drops
//! to the output Nyquist so the kernel doubles as the anti-alias filter. This is a
//! compact, dependency-free port of `syrinx-serve`'s `wavio::resample` — kept
//! local so `syrinx-stt` does not pull the whole TTS synth stack just for one
//! resampler. Identity (`in == out`) is a copy.

/// Resample `input` from `in_sr` to `out_sr` (mono `f32`).
pub fn resample(input: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    if input.is_empty() || in_sr == 0 || out_sr == 0 {
        return Vec::new();
    }
    if in_sr == out_sr {
        return input.to_vec();
    }

    let ratio = out_sr as f64 / in_sr as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;

    let a = 16.0_f64;
    let cutoff = if out_sr < in_sr {
        out_sr as f64 / in_sr as f64
    } else {
        1.0
    };
    let half = (a / cutoff).ceil() as isize;
    let n = input.len() as isize;

    let mut out = Vec::with_capacity(out_len);
    for o in 0..out_len {
        let center = o as f64 / ratio;
        let i0 = center.floor() as isize;
        let mut acc = 0.0_f64;
        let mut wsum = 0.0_f64;
        for k in (i0 - half + 1)..=(i0 + half) {
            if k < 0 || k >= n {
                continue;
            }
            let x = (center - k as f64) * cutoff;
            let w = lanczos(x, a) * cutoff;
            acc += input[k as usize] as f64 * w;
            wsum += w;
        }
        let v = if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 };
        out.push(v as f32);
    }
    out
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let p = std::f64::consts::PI * x;
        p.sin() / p
    }
}

fn lanczos(x: f64, a: f64) -> f64 {
    if x.abs() < a {
        sinc(x) * sinc(x / a)
    } else {
        0.0
    }
}
