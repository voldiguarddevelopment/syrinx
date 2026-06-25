//! Real audio-feature parity — the kaldi fbank (CAM++ input) and the flow-decoder
//! prompt mel, ported in `syrinx-frontend::feat`, checked against a CosyVoice2
//! Python reference dump.
//!
//! Gated on the `real` feature AND an env var pointing at the reference
//! safetensors. The fixture carries the *exact* resampled waveforms the Python
//! pipeline used (16 kHz for fbank, 24 kHz for the mel) plus the reference
//! features, so the only thing under test here is the feature math — resampling
//! and decode are out of scope. Skips cleanly when the fixture is absent, like the
//! other device-bound parity tests.
//!
//!   SYRINX_FEAT_REF=/root/parity/frontend/feat_ref.safetensors \
//!     cargo test --features real --test real_feat_parity -- --nocapture
//!
//! Reference keys (see the dumper):
//!   wav16_a [N16]    fbank_a    [T,80]    (kaldi fbank, 16 kHz)
//!   wav24_a [N24]    promptmel_a[80,T']   (matcha prompt mel, 24 kHz)
//!   ...and the `_b` variants for a second clip.
//!   povey_window [400], fbank_frame0_windowed [400], fbank_frame0_power [257]
//!     -- staged intermediates to localize a mismatch.

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use syrinx_frontend::feat;

/// Max absolute difference between a flat Rust `[T][D]` grid and a reference
/// tensor of matching shape `[T, D]`.
fn max_abs_diff_grid(rust: &[Vec<f32>], reference: &Tensor) -> f32 {
    let dims = reference.dims();
    assert_eq!(dims.len(), 2, "reference must be 2-D");
    assert_eq!(rust.len(), dims[0], "row count mismatch (got {}, ref {})", rust.len(), dims[0]);
    let refv: Vec<f32> = reference
        .flatten_all()
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    let cols = dims[1];
    let mut worst = 0.0f32;
    for (r, row) in rust.iter().enumerate() {
        assert_eq!(row.len(), cols, "col count mismatch at row {r}");
        for (c, &v) in row.iter().enumerate() {
            let d = (v - refv[r * cols + c]).abs();
            if d > worst {
                worst = d;
            }
        }
    }
    worst
}

fn load_ref(path: &str) -> HashMap<String, Tensor> {
    candle_core::safetensors::load(path, &Device::Cpu).expect("load feat reference fixture")
}

fn wav(reference: &HashMap<String, Tensor>, key: &str) -> Vec<f32> {
    reference
        .get(key)
        .unwrap_or_else(|| panic!("reference missing {key}"))
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn ref_path() -> Option<String> {
    match std::env::var("SYRINX_FEAT_REF").ok() {
        Some(p) if Path::new(&p).exists() => Some(p),
        _ => {
            eprintln!(
                "SKIP real_feat parity: set SYRINX_FEAT_REF to the on-disk feature \
                 reference (feat_ref.safetensors)"
            );
            None
        }
    }
}

#[test]
fn real_kaldi_fbank_matches_reference_within_1e_3() {
    let Some(path) = ref_path() else { return };
    let r = load_ref(&path);

    // C1 (staged): the povey window itself must match — isolates a window bug from
    // the rest of the pipeline before checking the full feature.
    {
        let expected = wav(&r, "povey_window");
        // re-derive via the module's public fbank on a unit impulse? The window is
        // private; instead we reconstruct it the same way and compare to the dump,
        // proving the convention (hann^0.85, non-periodic) is what the reference used.
        let n = expected.len();
        let win: Vec<f32> = (0..n)
            .map(|i| {
                let h = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
                h.powf(0.85)
            })
            .collect();
        let d = win
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("povey window  max-abs-diff = {d:.3e}");
        assert!(d < 1e-5, "povey window diverges ({d:.3e}) — window convention is wrong");
    }

    // C2: full fbank on clip A within 1e-3 of the kaldi reference.
    let wav16_a = wav(&r, "wav16_a");
    let fb_a = feat::kaldi_fbank(&wav16_a, 16000.0, 80);
    let ref_fb_a = r.get("fbank_a").expect("reference has fbank_a [T,80]");
    assert_eq!(fb_a.len(), ref_fb_a.dims()[0], "fbank A frame count mismatch");
    assert_eq!(ref_fb_a.dims()[1], 80, "fbank must be 80-dim");
    let d_a = max_abs_diff_grid(&fb_a, ref_fb_a);
    eprintln!("fbank A   frames={} max-abs-diff = {d_a:.3e}", fb_a.len());
    assert!(d_a < 1e-3, "fbank A max-abs diff {d_a:.3e} exceeds the 1e-3 parity tolerance");

    // C3: a second, distinct clip B also matches within 1e-3 (guards against a
    // length-specific or clip-specific fluke).
    if let Some(ref_fb_b) = r.get("fbank_b") {
        let wav16_b = wav(&r, "wav16_b");
        let fb_b = feat::kaldi_fbank(&wav16_b, 16000.0, 80);
        let d_b = max_abs_diff_grid(&fb_b, ref_fb_b);
        eprintln!("fbank B   frames={} max-abs-diff = {d_b:.3e}", fb_b.len());
        assert!(d_b < 1e-3, "fbank B max-abs diff {d_b:.3e} exceeds 1e-3");
    }
}

#[test]
fn real_prompt_mel_matches_reference_within_1e_3() {
    let Some(path) = ref_path() else { return };
    let r = load_ref(&path);

    // C1: prompt mel on clip A within 1e-3 of the matcha reference. Output is
    // mel-major [80, T'] to match the Python `.transpose`-free dump layout.
    let wav24_a = wav(&r, "wav24_a");
    let mel_a = feat::prompt_mel(&wav24_a, 1920, 80, 24000.0, 480, 1920, 0.0, 8000.0);
    let ref_mel_a = r.get("promptmel_a").expect("reference has promptmel_a [80,T']");
    assert_eq!(mel_a.len(), 80, "prompt mel must be 80 mel bins (mel-major)");
    assert_eq!(ref_mel_a.dims()[0], 80, "reference prompt mel must be 80-major");
    let d_a = max_abs_diff_grid(&mel_a, ref_mel_a);
    eprintln!("prompt-mel A   T'={} max-abs-diff = {d_a:.3e}", mel_a[0].len());
    assert!(d_a < 1e-3, "prompt-mel A max-abs diff {d_a:.3e} exceeds the 1e-3 parity tolerance");

    // C2: distinct clip B also within 1e-3.
    if let Some(ref_mel_b) = r.get("promptmel_b") {
        let wav24_b = wav(&r, "wav24_b");
        let mel_b = feat::prompt_mel(&wav24_b, 1920, 80, 24000.0, 480, 1920, 0.0, 8000.0);
        let d_b = max_abs_diff_grid(&mel_b, ref_mel_b);
        eprintln!("prompt-mel B   T'={} max-abs-diff = {d_b:.3e}", mel_b[0].len());
        assert!(d_b < 1e-3, "prompt-mel B max-abs diff {d_b:.3e} exceeds 1e-3");
    }
}
