//! Numeric parity for the **Fish Audio `s2-pro`** port against a Python reference dump.
//!
//! Loads the real s2 model and compares at least one component against a fixed reference
//! safetensors produced ON BOX by `scripts/gen-fish-ref.py` (a FIXED prompt + seed). Two
//! anchors, each checked only when its fixture keys are present (so a partial dump still
//! gives a clean signal, and an empty/unrecognized dump SKIPS rather than FAILS):
//!
//!   1. **slow-AR step-0 logits** — feed the reference prompt (`prompt_ids`, row-0 slow-
//!      vocab token ids) through the Qwen3-4B dual-AR slow backbone's prefill and compare
//!      the raw semantic logits at the last prompt position to `slow_logits_step0`.
//!   2. **codec decode** — decode the fixed `codec_codes` `[num_codebooks, T]` grid through
//!      the EVA-GAN / causal-DAC codec and compare the waveform to `codec_wav`.
//!
//! Fixture keys (see `scripts/gen-fish-ref.py`):
//!   * `prompt_ids`        I64 `[T_prompt]`            — row-0 token ids of the fixed prompt
//!   * `slow_logits_step0` F32 `[vocab_size]`          — reference slow logits, last position
//!   * `codec_codes`       I64 `[num_codebooks, T]`    — a fixed RVQ code grid
//!   * `codec_wav`         F32 `[N]`                    — Python codec decode of `codec_codes`
//!
//! Gated on `real` + `SYRINX_FISH_S2_DIR` (checkpoint) + `SYRINX_FISH_S2_REF` (the dump);
//! skips cleanly when either is absent:
//!
//!   SYRINX_FISH_S2_DIR=/root/models/s2-pro \
//!   SYRINX_FISH_S2_REF=/root/parity-fish/s2/ref.safetensors \
//!   cargo test --features real --test real_fish_s2_parity -- --nocapture
//!
//! PARITY: the tolerances below are best-effort fp32 floors and are UNVERIFIED until the
//! test runs on box against a real dump — tighten/loosen them once the box reports. The s2
//! codec is the riskiest module (see `s2::codec`), so `codec_wav` is the key anchor.

#![cfg(feature = "real")]

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use syrinx_fish::common::codec::RvqCodec;
use syrinx_fish::common::dualar::DualArBackend;
use syrinx_fish::s2::S2Pro;

// PARITY: fp32 accumulation floors — confirm/adjust on box against a real dump.
const SLOW_LOGITS_TOL: f32 = 2e-2;
const CODEC_WAV_TOL: f32 = 1e-3;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

/// Max absolute element-wise difference between two same-shape f32 tensors.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn f32_tensor(r: &HashMap<String, Tensor>, k: &str) -> Tensor {
    r.get(k)
        .unwrap_or_else(|| panic!("reference missing `{k}`"))
        .to_dtype(DType::F32)
        .unwrap()
}

#[test]
fn real_fish_s2_parity() {
    let (dir, ref_path) = match (env_path("SYRINX_FISH_S2_DIR"), env_path("SYRINX_FISH_S2_REF")) {
        (Some(d), Some(r)) => (d, r),
        _ => {
            eprintln!(
                "SKIP real_fish_s2_parity: set SYRINX_FISH_S2_DIR (checkpoint) and \
                 SYRINX_FISH_S2_REF (the gen-fish-ref.py safetensors dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let mut model = S2Pro::load(&dir, dev.clone()).expect("load s2-pro");
    let r = candle_core::safetensors::load(&ref_path, &dev).expect("load s2 parity ref");

    let n_cb = <S2Pro as RvqCodec>::config(&model).num_codebooks;
    let mut anchors = 0usize;

    // ---- (1) slow-AR step-0 logits ----
    if r.contains_key("prompt_ids") && r.contains_key("slow_logits_step0") {
        let ids: Vec<u32> = r["prompt_ids"]
            .flatten_all()
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let t = ids.len();
        assert!(t > 0, "prompt_ids is empty");
        // Build the encoded prompt `[1 + num_codebooks, T]`: row 0 = ids, code rows = 0.
        let mut flat = vec![0u32; (1 + n_cb) * t];
        flat[..t].copy_from_slice(&ids);
        let prompt = Tensor::from_vec(flat, (1 + n_cb, t), &dev).unwrap();

        let step = model.prefill(&prompt).expect("s2 prefill");
        let our = step.semantic_logits.flatten_all().unwrap();
        let reference = f32_tensor(&r, "slow_logits_step0").flatten_all().unwrap();
        assert_eq!(
            our.dims(),
            reference.dims(),
            "slow logits shape: ours={:?} ref={:?}",
            our.dims(),
            reference.dims()
        );
        let d = max_abs_diff(&our, &reference);
        eprintln!("s2 slow-AR step0 logits max-abs-diff = {d:.3e} (tol {SLOW_LOGITS_TOL:.1e})");
        assert!(
            d < SLOW_LOGITS_TOL,
            "s2 slow-AR logits diff {d:.3e} exceeds {SLOW_LOGITS_TOL:.1e}"
        );
        anchors += 1;
    }

    // ---- (2) codec decode ----
    if r.contains_key("codec_codes") && r.contains_key("codec_wav") {
        let codes = r["codec_codes"].to_dtype(DType::U32).unwrap();
        let our_wav = model.decode_codes(&codes).expect("s2 codec decode").flatten_all().unwrap();
        let ref_wav = f32_tensor(&r, "codec_wav").flatten_all().unwrap();
        assert_eq!(
            our_wav.dims(),
            ref_wav.dims(),
            "codec decode length: ours={:?} ref={:?}",
            our_wav.dims(),
            ref_wav.dims()
        );
        let d = max_abs_diff(&our_wav, &ref_wav);
        eprintln!("s2 codec decode max-abs-diff = {d:.3e} (tol {CODEC_WAV_TOL:.1e})");
        assert!(
            d < CODEC_WAV_TOL,
            "s2 codec decode diff {d:.3e} exceeds {CODEC_WAV_TOL:.1e}"
        );
        anchors += 1;
    }

    if anchors == 0 {
        eprintln!(
            "SKIP real_fish_s2_parity: the dump at {ref_path} carries none of the recognized \
             keys (prompt_ids+slow_logits_step0 and/or codec_codes+codec_wav); regenerate it \
             with scripts/gen-fish-ref.py"
        );
        return;
    }
    eprintln!("PASS: s2 parity — {anchors} anchor(s) within tolerance.");
}
