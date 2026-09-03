//! Pins the range clamp in the Fish s2-pro EVA-GAN codec's RVQ decode.
//!
//! ## The bug this test exists to prevent
//!
//! The codec's two code tables are **different heights**: the semantic codebook has 4096
//! entries, every one of the nine residual codebooks has 1024
//! (`quantizer.semantic_quantizer.quantizers.0.codebook.weight` is `[4096, 8]`,
//! `quantizer.quantizer.quantizers.{0..8}.codebook.weight` are `[1024, 8]`, read from the
//! shipped `codec.pth`). `decode` used to clamp residual codes with
//! `CodecConfig::residual_size`, which is the **fast AR head's logit width** (4096, taken
//! from `fast_embeddings.weight`) — a sampling bound, not a table height. A residual code
//! in `1024..=4095` is therefore representable by the sampler, and used to walk straight
//! past the end of a 1024-row table into `index_select`, where the reference implementation
//! (`DownsampleResidualVectorQuantize.decode`) simply clamps. Verified against the pre-fix
//! code: candle rejects it rather than returning garbage —
//! `decode row 1 code 1024: index-select invalid index 1024 with dim size 1024` — so the
//! symptom was a hard-failed render, not silent corruption.
//!
//! The clamp now lives in `decode_codebook` and is taken from each table's own height, so
//! it cannot go stale when a config field is repurposed. This test pins that behaviour at
//! both tables and on both sides of both boundaries.
//!
//! ## Why it loads the codec alone
//!
//! Only the decode side of the codec is needed (~2 GB in f32), versus ~10 GB on GPU or
//! ~19 GB on the CPU parity path to go through `S2Pro`. That is what keeps this test
//! ordinarily runnable rather than a special occasion.
//!
//! Gated on `SYRINX_FISH_S2_DIR`; skips cleanly off-box.
//!
//!   SYRINX_FISH_S2_DIR=/data/models/s2-pro \
//!     cargo test --features real --release --test real_fish_s2_codec_clamp -- --nocapture

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};

use syrinx_fish::common::config::FishConfig;
use syrinx_fish::s2::codec::EvaGanDac;
use syrinx_fish::s2::load::{load_codec, CodecParts};

/// Height of every residual codebook in the shipped checkpoint.
const RESIDUAL_TABLE: u32 = 1024;
/// Height of the semantic codebook in the shipped checkpoint.
const SEMANTIC_TABLE: u32 = 4096;
/// Frames per probe. Small on purpose: the clamp is per-element, so length buys nothing.
const FRAMES: usize = 8;

fn decode_with(codec: &EvaGanDac, n_cb: usize, row: usize, code: u32) -> Vec<f32> {
    // A matrix of zeros with one row held at `code`. Row 0 is the semantic codebook; rows
    // 1.. are the residual ones.
    let mut host = vec![0u32; n_cb * FRAMES];
    for t in 0..FRAMES {
        host[row * FRAMES + t] = code;
    }
    let codes = Tensor::from_vec(host, (n_cb, FRAMES), &Device::Cpu).expect("codes tensor");
    codec
        .decode(&codes)
        .unwrap_or_else(|e| panic!("decode row {row} code {code}: {e}"))
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "decoded lengths differ");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn residual_codes_clamp_at_the_real_table_height() {
    let Some(dir) = std::env::var("SYRINX_FISH_S2_DIR")
        .ok()
        .filter(|d| std::path::Path::new(d).exists())
    else {
        eprintln!(
            "SKIP real_fish_s2_codec_clamp: set SYRINX_FISH_S2_DIR to the s2-pro checkpoint dir"
        );
        return;
    };

    let path = format!("{dir}/codec.pth");
    let w = load_codec(&path, Device::Cpu, DType::F32, CodecParts::Decode).expect("load codec");

    let mut cfg = FishConfig::s2_pro().codec;
    // Reproduce the PRODUCTION configuration, which is the only one the bug appears in.
    // The static default is `residual_size: 1024` — correct, and confirmed against the
    // checkpoint by this very test — but `S2Pro::load` overwrites it with
    // `fast_embeddings.weight.dim(0)` = 4096, the fast AR head's logit width. Testing the
    // static default would quietly exercise a configuration nothing ever runs.
    assert_eq!(cfg.residual_size, 1024, "static default should still be the true table height");
    cfg.residual_size = 4096;

    let n_cb = cfg.num_codebooks;
    let codec = EvaGanDac::new(w, cfg);
    assert!(n_cb >= 2, "need a semantic row plus at least one residual row");

    // ---- residual row: the boundary is the table height, not the config field ----
    let last_valid = decode_with(&codec, n_cb, 1, RESIDUAL_TABLE - 1); // 1023
    let one_past = decode_with(&codec, n_cb, 1, RESIDUAL_TABLE); // 1024 -> clamps
    let far_past = decode_with(&codec, n_cb, 1, SEMANTIC_TABLE - 1); // 4095 -> clamps

    // Below the boundary nothing is clamped, so neighbouring codes must still differ. This
    // is the half that catches a clamp set too LOW (or a table read at the wrong width).
    let below = decode_with(&codec, n_cb, 1, RESIDUAL_TABLE - 2); // 1022
    assert!(
        max_abs_diff(&below, &last_valid) > 1e-6,
        "codes 1022 and 1023 are both valid rows and must decode differently"
    );

    // At and above the boundary everything collapses onto the last row. Before the fix
    // these two calls indexed out of bounds instead.
    assert_eq!(
        max_abs_diff(&last_valid, &one_past),
        0.0,
        "residual code 1024 must clamp to 1023 exactly, as the reference does"
    );
    assert_eq!(
        max_abs_diff(&last_valid, &far_past),
        0.0,
        "residual code 4095 — representable by the 4096-wide fast head — must clamp to 1023"
    );

    // ---- semantic row: same rule, different height ----
    let sem_last = decode_with(&codec, n_cb, 0, SEMANTIC_TABLE - 1); // 4095
    let sem_below = decode_with(&codec, n_cb, 0, SEMANTIC_TABLE - 2); // 4094
    let sem_past = decode_with(&codec, n_cb, 0, SEMANTIC_TABLE); // 4096 -> clamps
    assert!(
        max_abs_diff(&sem_below, &sem_last) > 1e-6,
        "semantic codes 4094 and 4095 are both valid rows and must decode differently"
    );
    assert_eq!(
        max_abs_diff(&sem_last, &sem_past),
        0.0,
        "semantic code 4096 must clamp to 4095"
    );

    // The semantic table is genuinely taller than the residual one: a code valid there and
    // invalid here is exactly the case the old shared ceiling got wrong.
    assert!(
        max_abs_diff(&sem_last, &decode_with(&codec, n_cb, 0, RESIDUAL_TABLE)) > 1e-6,
        "semantic code 1024 is a valid row and must NOT be clamped to the residual height"
    );

    eprintln!(
        "[codec-clamp] residual tables clamp at {RESIDUAL_TABLE}, semantic at {SEMANTIC_TABLE}; \
         {n_cb} codebooks checked"
    );
}
