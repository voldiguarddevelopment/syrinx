//! Realized **CosyVoice3 quantized footprint** report — the README ~270 MB size-goal
//! track for the CV3 component ports (the CV3 twin of `tests/real_quant_footprint.rs`).
//!
//! Loads every CV3 sub-model in its int4-quantized variant and prints the realized
//! per-component + grand-total weight footprint vs its fp32 load:
//!
//!   * LM ([`Cv3Lm::load_quantized`]): int4 (Q4_0) big linears + int4 per-row
//!     dequant-on-gather embedding tables + f32 norms/biases. CV3's `lm_head` is *tied* to
//!     `embed_tokens` (byte-identical), so the shared text embedding is retained — and
//!     quantized — as the `embed_tokens` table; only its redundant `lm_head` duplicate is
//!     dropped. Also prints the LM **dense breakdown** (everything left un-quantized).
//!   * Flow ([`Cv3Flow::load_quantized`]): Q4_0 the DiT `linear()` weights (22 blocks'
//!     attention + FF + AdaLN modulation + `input_embed.proj`/`proj_out`/time-MLP) + f32
//!     conv/embed.
//!   * HiFT ([`Cv3Hift::load_quantized`]): Q4_0 the large decode conv kernels
//!     (weight_norm-folded) + f32 biases/Snake/conv_pre/source_downs and the float64
//!     f0_predictor (never quantized).
//!   * CAM++ speaker ([`CamPlus::load_quantized`], shared with CV2): Q4_0 conv/linear
//!     kernels + f32 biases/BN-stats.
//!
//! This is a *measurement*, not a parity assertion (int4 trades quality for size — and it is
//! an opt-in size, NOT speed, win: the dequant-on-fetch stalls inference. The quality cost is
//! measured by the on-box SIM-o eval, not here). It asserts only that each quantized build is
//! strictly smaller than its fp32 load — the size win is real — and prints the numbers the
//! maintainer compares against the fp32 budgets and the 270 MB README target.
//!
//! Gated on the `real` feature AND the on-disk fp32 CV3 checkpoints (too large to vendor —
//! they live on the GPU box); skips cleanly when absent, per the device-bound recipe.
//!
//!   SYRINX_CV3_LM_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/llm_fp32.safetensors \
//!   SYRINX_CV3_FLOW_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/flow_fp32.safetensors \
//!   SYRINX_CV3_HIFT_WEIGHTS=/root/models/Fun-CosyVoice3-0.5B-2512/hift_fp32.safetensors \
//!   SYRINX_CV3_SPK_WEIGHTS=/root/models/CosyVoice2-0.5B/campplus_weights.safetensors \
//!   cargo test --features real --test real_cv3_quant_footprint -- --nocapture

#![cfg(feature = "real")]

use candle_core::Device;
use std::path::Path;

/// README ~270 MB 4-bit target for the whole CV3 model.
const TARGET_MB: f64 = 270.0;

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Resolve an env-var weight path that exists on disk, else `None` (skip that component).
fn weights(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|p| Path::new(p).exists())
}

#[test]
fn cv3_quantized_footprint_report() {
    let dev = Device::Cpu;
    let mut total_q = 0usize;
    let mut any = false;

    // --- LM: int4 linears + int4 embeds (lm_head tied to embed_tokens) ------
    if let Some(w) = weights("SYRINX_CV3_LM_WEIGHTS") {
        use syrinx_lm::cv3::Cv3Lm;
        let f = Cv3Lm::load(&w, dev.clone()).expect("load fp32 CV3 LM");
        let q = Cv3Lm::load_quantized(&w, dev.clone()).expect("load quantized CV3 LM");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "LM      fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
             (int4 linears {:.1} MB [{} wts] + int4 embeds {:.1} MB + dense {:.1} MB)",
            mb(fp.total_bytes()),
            mb(qp.total_bytes()),
            mb(qp.quant_bytes),
            qp.n_quantized,
            mb(qp.embed_bytes),
            mb(qp.dense_bytes),
        );
        // Instrumentation: per-tensor breakdown of everything NOT quantized. In the int4
        // build this should be only tiny norms + biases (the tied lm_head duplicate is gone);
        // any large entry is an un-quantized matmul to investigate.
        eprintln!("  LM dense (un-quantized) tensors, largest first:");
        for (name, bytes) in q.dense_breakdown().into_iter().take(8) {
            eprintln!("    {:>9.3} MB  {name}", mb(bytes));
        }
        assert!(qp.total_bytes() < fp.total_bytes(), "quant CV3 LM not smaller than fp32");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP CV3 LM footprint: set SYRINX_CV3_LM_WEIGHTS to llm_fp32.safetensors");
    }

    // --- Flow: Q4_0 DiT linears ---------------------------------------------
    if let Some(w) = weights("SYRINX_CV3_FLOW_WEIGHTS") {
        use syrinx_acoustic::cv3::Cv3Flow;
        let f = Cv3Flow::load(&w, dev.clone()).expect("load fp32 CV3 flow");
        let q = Cv3Flow::load_quantized(&w, dev.clone()).expect("load quantized CV3 flow");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "Flow    fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
             (int4 linears {:.1} MB [{} wts] + dense {:.1} MB)",
            mb(fp.total_bytes()),
            mb(qp.total_bytes()),
            mb(qp.quant_bytes),
            qp.n_quantized,
            mb(qp.dense_bytes),
        );
        assert!(qp.total_bytes() < fp.total_bytes(), "quant CV3 flow not smaller than fp32");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP CV3 flow footprint: set SYRINX_CV3_FLOW_WEIGHTS to flow_fp32.safetensors");
    }

    // --- HiFT vocoder: Q4_0 dequant-on-fetch conv kernels -------------------
    if let Some(w) = weights("SYRINX_CV3_HIFT_WEIGHTS") {
        use syrinx_vocoder::cv3::Cv3Hift;
        let f = Cv3Hift::load(&w, dev.clone()).expect("load fp32 CV3 hift");
        let q = Cv3Hift::load_quantized(&w, dev.clone()).expect("load quantized CV3 hift");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "HiFT    fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
             (int4 kernels {:.1} MB [{} wts] + dense {:.1} MB)",
            mb(fp.total_bytes()),
            mb(qp.total_bytes()),
            mb(qp.quant_bytes),
            qp.n_quantized,
            mb(qp.dense_bytes),
        );
        assert!(qp.total_bytes() < fp.total_bytes(), "quant CV3 hift not smaller than fp32");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP CV3 hift footprint: set SYRINX_CV3_HIFT_WEIGHTS to hift_fp32.safetensors");
    }

    // --- CAM++ speaker (shared with CV2): Q4_0 conv/linear kernels ----------
    if let Some(w) = weights("SYRINX_CV3_SPK_WEIGHTS") {
        use syrinx_speaker::campplus::CamPlus;
        let f = CamPlus::load(&w, dev.clone()).expect("load fp32 speaker");
        let q = CamPlus::load_quantized(&w, dev.clone()).expect("load quantized speaker");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "Speaker fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
             (int4 kernels {:.1} MB [{} wts] + dense {:.1} MB)",
            mb(fp.total_bytes()),
            mb(qp.total_bytes()),
            mb(qp.quant_bytes),
            qp.n_quantized,
            mb(qp.dense_bytes),
        );
        assert!(qp.total_bytes() < fp.total_bytes(), "quant speaker not smaller than fp32");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP speaker footprint: set SYRINX_CV3_SPK_WEIGHTS to campplus_weights.safetensors");
    }

    if any {
        eprintln!(
            "TOTAL CV3 quantized (present components) = {:.1} MB   (README 4-bit target ~{:.0} MB)",
            mb(total_q),
            TARGET_MB,
        );
    } else {
        eprintln!(
            "SKIP real_cv3_quant_footprint: no weights present (set SYRINX_CV3_LM_WEIGHTS / \
             SYRINX_CV3_FLOW_WEIGHTS / SYRINX_CV3_HIFT_WEIGHTS / SYRINX_CV3_SPK_WEIGHTS)"
        );
    }
}
