//! Real CosyVoice2 speaker-encoder (CAM++ / campplus) parity — the GPU / real-weights
//! track, buildable on the model box. Mirrors the LM parity recipe.
//!
//! Gated on the `real` feature AND env vars pointing at the fp32 weight dump (exported
//! from `campplus.onnx`) + the Python reference dump (seeded fbank in, 192-d embedding
//! out). Both are too large to vendor — they live on the model box. Skips cleanly when
//! absent, like the device-bound task recipe.
//!
//!   SYRINX_SPK_WEIGHTS=/root/parity/speaker/campplus_weights.safetensors \
//!   SYRINX_SPK_REF=/root/parity/speaker/ref.safetensors \
//!   cargo test -p syrinx-speaker --features real -- --nocapture

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use syrinx_speaker::campplus::CamPlus;

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

/// Cosine similarity between two flat `[1, D]` (or `[D]`) embeddings.
fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap();
    let b = b.flatten_all().unwrap();
    let dot = (&a * &b).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    let na = a.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
    let nb = b.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
    dot / (na * nb)
}

#[test]
fn real_campplus_forward_matches_reference_within_1e_3() {
    let (weights, reference) = match (
        std::env::var("SYRINX_SPK_WEIGHTS").ok(),
        std::env::var("SYRINX_SPK_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => (w, r),
        _ => {
            eprintln!(
                "SKIP real_campplus parity: set SYRINX_SPK_WEIGHTS + SYRINX_SPK_REF to the \
                 on-disk fp32 fixtures (campplus weights + reference dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    // Default fp32 compute path reaches ~1.3e-5 against the onnxruntime fp32 reference —
    // well under the 1e-3 bar; `load_with_dtype(.., F64)` is available for an even more
    // precise accumulation but is not needed here.
    let enc = CamPlus::load(&weights, dev.clone()).expect("load campplus weights");

    let r = candle_core::safetensors::load(&reference, &dev).expect("load reference fixture");
    let fbank = r
        .get("fbank")
        .expect("reference has fbank [1,T,80]")
        .to_dtype(DType::F32)
        .unwrap();
    let expected = r
        .get("embedding")
        .expect("reference has embedding [1,192]")
        .to_dtype(DType::F32)
        .unwrap();

    let got = enc.forward(&fbank).expect("campplus forward");

    // C1: emitted embedding dimensionality equals the reference (192-d x-vector).
    assert_eq!(got.dims(), expected.dims(), "embedding shape mismatch");
    assert_eq!(got.dim(1).unwrap(), 192, "x-vector must be 192-d");

    // C2: max-abs parity within 1e-3 (a 2e-3 perturbation would exceed it).
    let d = max_abs_diff(&got, &expected);
    let cos = cosine(&got, &expected);
    eprintln!("campplus  max-abs-diff = {d:.3e}   cosine = {cos:.6}");
    assert!(d < 1e-3, "campplus max-abs diff {d:.3e} exceeds the 1e-3 parity tolerance");

    // C3: cosine similarity to the reference embedding >= 0.9999.
    assert!(cos >= 0.9999, "campplus cosine {cos:.6} below the 0.9999 parity bar");

    // C4: a second, distinct reference clip must yield an embedding whose cosine
    // *ordering* matches the reference's — self-similarity (A·A) strictly above the
    // cross-clip similarity (A·B). A swapped/degenerate embedding would invert this.
    if let (Some(fb), Some(eb)) = (r.get("fbank_b"), r.get("embedding_b")) {
        let fb = fb.to_dtype(DType::F32).unwrap();
        let eb = eb.to_dtype(DType::F32).unwrap();
        let got_b = enc.forward(&fb).expect("campplus forward (clip B)");

        // clip B is itself a parity match.
        let db = max_abs_diff(&got_b, &eb);
        assert!(db < 1e-3, "clip-B max-abs diff {db:.3e} exceeds 1e-3");

        // ordering: ours(A·A)=1.0 ~ ref(A·A); ours(A·B) tracks ref(A·B) and is < self.
        let ref_ab = cosine(&expected, &eb);
        let our_ab = cosine(&got, &got_b);
        let self_aa = cosine(&got, &got); // 1.0 by construction
        eprintln!("C4 ordering: ref(A,B)={ref_ab:.4}  ours(A,B)={our_ab:.4}  self={self_aa:.4}");
        assert!(self_aa > our_ab, "self-similarity must exceed cross-clip similarity");
        assert!(
            (our_ab - ref_ab).abs() < 1e-3,
            "cross-clip cosine {our_ab:.4} diverges from reference {ref_ab:.4}"
        );
    }
}
