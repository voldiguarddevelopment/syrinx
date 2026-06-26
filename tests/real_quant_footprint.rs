//! Realized **quantized footprint** report — the README ~270 MB size-goal track.
//!
//! Loads the LM and the flow in their quantized variants and prints the realized
//! per-component + total weight footprint:
//!
//!   * LM ([`Qwen2Lm::load_quantized`]): int4 (Q4_0) big linears + int8 per-row
//!     dequant-on-gather embedding tables + f32 norms/biases.
//!   * Flow ([`Flow::load_quantized`]): Q4_0 `linear()` weights + f32 conv/norm/embed.
//!
//! This is a *measurement*, not a parity assertion (int4/int8 trade quality for size;
//! the quality cost is measured by the on-box SIM-o eval, not here). It asserts only
//! that each quantized build is strictly smaller than its fp32 load — the size win is
//! real — and prints the numbers the maintainer compares against the 2449 + 429 MB
//! fp32 budgets and the 270 MB README target.
//!
//! Gated on the `real` feature AND the on-disk fp32 checkpoints (too large to vendor —
//! they live on the GPU box); skips cleanly when absent, per the device-bound recipe.
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_FLOW_WEIGHTS=/root/models/CosyVoice2-0.5B/flow_fp32.safetensors \
//!   cargo test --features real --test real_quant_footprint -- --nocapture

#![cfg(feature = "real")]

use candle_core::Device;
use std::path::Path;

/// fp32 reference footprints from the README budget (MB).
const FP32_LM_MB: f64 = 2449.0;
const FP32_FLOW_MB: f64 = 429.0;

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[test]
fn quantized_footprint_report() {
    let dev = Device::Cpu;
    let mut total_q = 0usize;
    let mut any = false;

    // --- LM: int4 linears + int8 embeds -------------------------------------
    if let Some(w) = std::env::var("SYRINX_LM_WEIGHTS").ok().filter(|p| Path::new(p).exists()) {
        use syrinx_lm::real::Qwen2Lm;
        let f = Qwen2Lm::load(&w, dev.clone()).expect("load fp32 LM");
        let q = Qwen2Lm::load_quantized(&w, dev.clone()).expect("load quantized LM");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "LM    fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
             (int4 linears {:.1} MB [{} wts] + int8 embeds {:.1} MB + dense {:.1} MB)   \
             README fp32 LM = {:.0} MB",
            mb(fp.total_bytes()),
            mb(qp.total_bytes()),
            mb(qp.quant_bytes),
            qp.n_quantized,
            mb(qp.embed_bytes),
            mb(qp.dense_bytes),
            FP32_LM_MB,
        );
        assert!(qp.total_bytes() < fp.total_bytes(), "quant LM not smaller than fp32");
        assert!(qp.total_mb() < FP32_LM_MB, "quant LM not below fp32 budget");
        total_q += qp.total_bytes();
        any = true;
    } else {
        eprintln!("SKIP LM footprint: set SYRINX_LM_WEIGHTS to llm_fp32.safetensors");
    }

    // --- Flow: Q4_0 linears -------------------------------------------------
    if let Some(w) = std::env::var("SYRINX_FLOW_WEIGHTS").ok().filter(|p| Path::new(p).exists()) {
        use syrinx_acoustic::real::Flow;
        let f = Flow::load(&w, dev.clone()).expect("load fp32 flow");
        let q = Flow::load_quantized(&w, dev.clone()).expect("load quantized flow");
        let fp = f.footprint();
        let qp = q.footprint();
        eprintln!(
            "Flow  fp32 = {:>8.1} MB   quant = {:>8.1} MB  \
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

    if any {
        eprintln!(
            "TOTAL quantized LM+flow = {:.1} MB  (fp32 LM+flow budget = {:.0} MB; README target ~270 MB at 4-bit)",
            mb(total_q),
            FP32_LM_MB + FP32_FLOW_MB,
        );
    } else {
        eprintln!("SKIP real_quant_footprint: no weights present (set SYRINX_LM_WEIGHTS / SYRINX_FLOW_WEIGHTS)");
    }
}
