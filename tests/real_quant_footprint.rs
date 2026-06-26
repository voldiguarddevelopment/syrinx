//! Realized **quantized footprint** report — the README ~270 MB size-goal track.
//!
//! Loads every sub-model in its quantized variant and prints the realized per-component
//! + grand-total weight footprint vs the README ~270 MB target:
//!
//!   * LM ([`Qwen2Lm::load_quantized`]): int4 (Q4_0) big linears + int4 per-row
//!     dequant-on-gather embedding tables + f32 norms/biases, with the unused Qwen2
//!     `lm_head` (the ~520 MB dense remainder) dropped. Also prints the LM **dense
//!     breakdown** (per-tensor name + bytes of everything left un-quantized) — the
//!     instrumentation that surfaced the `lm_head` remainder.
//!   * Flow ([`Flow::load_quantized`]): Q4_0 `linear()` weights + f32 conv/norm/embed.
//!   * HiFT vocoder (`HiftVocoder::load_quantized`): Q4_0 dequant-on-fetch conv/linear
//!     kernels + f32 biases/alphas.
//!   * CAM++ speaker (`CamPlus::load_quantized`): Q4_0 dequant-on-fetch conv/linear
//!     kernels + f32 biases/BN-stats.
//!
//! This is a *measurement*, not a parity assertion (int4 trades quality for size; the
//! quality cost is measured by the on-box SIM-o eval, not here). It asserts only that
//! each quantized build is strictly smaller than its fp32 load — the size win is real —
//! and prints the numbers the maintainer compares against the fp32 budgets and the
//! 270 MB README target.
//!
//! Gated on the `real` feature AND the on-disk fp32 checkpoints (too large to vendor —
//! they live on the GPU box); skips cleanly when absent, per the device-bound recipe.
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_FLOW_WEIGHTS=/root/models/CosyVoice2-0.5B/flow_fp32.safetensors \
//!   SYRINX_HIFT_WEIGHTS=/root/parity/vocoder/hift_fp32.safetensors \
//!   SYRINX_SPK_WEIGHTS=/root/parity/speaker/campplus_weights.safetensors \
//!   cargo test --features real --test real_quant_footprint -- --nocapture

#![cfg(feature = "real")]

use candle_core::Device;
use std::path::Path;

/// fp32 reference footprints from the README budget (MB).
const FP32_LM_MB: f64 = 2449.0;
const FP32_FLOW_MB: f64 = 429.0;
/// README ~270 MB 4-bit target for the whole model.
const TARGET_MB: f64 = 270.0;

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Resolve an env-var weight path that exists on disk, else `None` (skip that component).
fn weights(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|p| Path::new(p).exists())
}

#[test]
fn quantized_footprint_report() {
    let dev = Device::Cpu;
    let mut total_q = 0usize;
    let mut any = false;

    // --- LM: int4 linears + int4 embeds, lm_head dropped --------------------
    if let Some(w) = weights("SYRINX_LM_WEIGHTS") {
        use syrinx_lm::real::Qwen2Lm;
        let f = Qwen2Lm::load(&w, dev.clone()).expect("load fp32 LM");
        let q = Qwen2Lm::load_quantized(&w, dev.clone()).expect("load quantized LM");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "LM      fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
             (int4 linears {:.1} MB [{} wts] + int4 embeds {:.1} MB + dense {:.1} MB)   \
             README fp32 LM = {:.0} MB",
            mb(fp.total_bytes()),
            mb(qp.total_bytes()),
            mb(qp.quant_bytes),
            qp.n_quantized,
            mb(qp.embed_bytes),
            mb(qp.dense_bytes),
            FP32_LM_MB,
        );
        // Instrumentation: per-tensor breakdown of everything NOT quantized. In the
        // quantized build this should now be only tiny norms + biases (the ~520 MB
        // lm_head is dropped); any large entry is an un-quantized matmul to investigate.
        eprintln!("  LM dense (un-quantized) tensors, largest first:");
        for (name, bytes) in q.dense_breakdown().into_iter().take(8) {
            eprintln!("    {:>9.3} MB  {name}", mb(bytes));
        }
        assert!(qp.total_bytes() < fp.total_bytes(), "quant LM not smaller than fp32");
        assert!(qp.total_mb() < FP32_LM_MB, "quant LM not below fp32 budget");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP LM footprint: set SYRINX_LM_WEIGHTS to llm_fp32.safetensors");
    }

    // --- Flow: Q4_0 linears -------------------------------------------------
    if let Some(w) = weights("SYRINX_FLOW_WEIGHTS") {
        use syrinx_acoustic::real::Flow;
        let f = Flow::load(&w, dev.clone()).expect("load fp32 flow");
        let q = Flow::load_quantized(&w, dev.clone()).expect("load quantized flow");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "Flow    fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
             (int4 linears {:.1} MB [{} wts] + dense {:.1} MB)   README fp32 flow = {:.0} MB",
            mb(fp.total_bytes()),
            mb(qp.total_bytes()),
            mb(qp.quant_bytes),
            qp.n_quantized,
            mb(qp.dense_bytes),
            FP32_FLOW_MB,
        );
        assert!(qp.total_bytes() < fp.total_bytes(), "quant flow not smaller than fp32");
        assert!(qp.total_mb() < FP32_FLOW_MB, "quant flow not below fp32 budget");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP flow footprint: set SYRINX_FLOW_WEIGHTS to flow_fp32.safetensors");
    }

    // --- HiFT vocoder: Q4_0 dequant-on-fetch conv/linear kernels -------------
    if let Some(w) = weights("SYRINX_HIFT_WEIGHTS") {
        use syrinx_vocoder::real::HiftVocoder;
        let f = HiftVocoder::load(&w, dev.clone()).expect("load fp32 hift");
        let q = HiftVocoder::load_quantized(&w, dev.clone()).expect("load quantized hift");
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
        assert!(qp.total_bytes() < fp.total_bytes(), "quant hift not smaller than fp32");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP hift footprint: set SYRINX_HIFT_WEIGHTS to hift_fp32.safetensors");
    }

    // --- CAM++ speaker: Q4_0 dequant-on-fetch conv/linear kernels ------------
    if let Some(w) = weights("SYRINX_SPK_WEIGHTS") {
        use syrinx_speaker::real::CamPlus;
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
        eprintln!("SKIP speaker footprint: set SYRINX_SPK_WEIGHTS to campplus_weights.safetensors");
    }

    if any {
        eprintln!(
            "TOTAL quantized (present components) = {:.1} MB   (README 4-bit target ~{:.0} MB)",
            mb(total_q),
            TARGET_MB,
        );
    } else {
        eprintln!(
            "SKIP real_quant_footprint: no weights present (set SYRINX_LM_WEIGHTS / \
             SYRINX_FLOW_WEIGHTS / SYRINX_HIFT_WEIGHTS / SYRINX_SPK_WEIGHTS)"
        );
    }
}
