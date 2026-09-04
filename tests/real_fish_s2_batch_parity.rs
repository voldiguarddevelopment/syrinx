//! Does Fish s2-pro's BATCHED generation path agree with the verified batch=1 path?
//!
//! `drive_batch` (commit 44cb560) renders N samples per forward, because the 5B forward is
//! memory-bandwidth-bound at batch=1 and batching gives roughly Nx throughput. Its own
//! commit message says "GPU correctness to be Whisper-verified on-box; uncertain
//! masking/position math marked `// PARITY:`" — that verification never happened, and the
//! corpus-render path (`syrinx synth --batch-size N`) uses it. So the fast path is
//! currently unproven while the slow path it replaces is not.
//!
//! ## The contract being checked
//!
//! `drive_batch` gives slot `i` its own `Sampler::new(params.seed + i)`, deliberately so
//! that a batched slot reproduces the single-sample stream. That makes the check exact
//! rather than statistical:
//!
//!   `drive_batch(prompts, seed = S)[i]`  ==  `drive(prompts[i], seed = S + i)`
//!
//! Integer code matrices, compared for equality. No tolerance, because there is nothing
//! here that may legitimately differ — same weights, same device, same sampler stream.
//!
//! ## Why the prompts have different lengths
//!
//! This is the whole point of the test. Batching left-pads every prompt to the longest,
//! then relies on per-sample RoPE positions (`c - pad_i`, so padding stays
//! position-invisible) and a per-sample causal+pad mask. All of that is exactly what is
//! marked `// PARITY:`, and ALL of it is inert when every prompt is the same length —
//! equal-length prompts would produce a green that proves almost nothing. The texts below
//! are chosen to tokenize to clearly different lengths so every slot but the longest is
//! genuinely padded.
//!
//! ## What was measured, and what it means for batched renders
//!
//! On CPU/f32 the batched prefill is **bit-exact** against the single-sample path, so the
//! left-pad, the per-sample RoPE offsets and the per-sample mask are all correct.
//!
//! The SAMPLED codes are a different story and are deliberately NOT asserted here. On GPU
//! bf16 the batched and unbatched GEMMs differ in their last bits (0.125-0.1875 on logits
//! of order 18 — a few ulps), and temperature sampling amplifies one flipped bit into a
//! different draw and then a completely different utterance. Measured at batch=2: slot 0
//! stopped at 24 frames versus 20, and slot 1 diverged from frame 3 onward. That is
//! chaotic amplification of legitimate rounding, not a defect — but the practical
//! consequence is real and worth stating: **a `--batch-size N` corpus render does not
//! reproduce a batch=1 render of the same text and seed.** Both are valid; they are not
//! the same audio. Anything that needs reproducibility must fix the batch width too.
//!
//! One aside worth keeping: shifting every position in a sample by a constant is
//! invisible to RoPE, which is relative. An early negative control that dropped the
//! per-sample offset entirely (`c` instead of `c - pad_i`) changed nothing at all, because
//! it shifts a sample uniformly. Breaking the mask is the control that bites.
//!
//! ## Running it
//!
//! Deliberately in NO group and opt-in on `SYRINX_FISH_BATCH_PARITY=1`, following the
//! `real_cue_activation` precedent: it loads s2-pro (~10 GB on GPU, ~19 GB on the CPU
//! parity path) and generates several times over, which is far too heavy for a routine
//! board.
//!
//! ```text
//! source scripts/test-all.env
//! SYRINX_FISH_BATCH_PARITY=1 MEMMAX=24G scripts/run-isolated.sh \
//!   cargo test --features "real cuda" --release --test real_fish_s2_batch_parity -- --nocapture
//! ```

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};

use syrinx_fish::common::dualar::DualArBackend;
use syrinx_fish::s2::S2Pro;

/// Different token lengths on purpose — see the module docs. Short/medium/long.
const TEXTS: [&str; 3] = [
    "Hello there.",
    "The quick brown fox jumps over the lazy dog.",
    "She had walked the same road every morning for thirty years, and still it surprised her.",
];

/// Bound on the batched-vs-single prefill logit difference, derived from measurement.
///
/// On CPU/f32 the batched path is **bit-exact** against the single-sample path: measured
/// 0.00000 on all three slots (pad widths 15, 8 and 0), against logits of order 18-22.
/// Breaking the left-pad mask moves it to **0.23633 / 0.14311 / 0.00000** — the last slot
/// is the longest prompt, which has no padding and correctly does not move.
///
/// 1e-3 therefore sits ~240x below a real fault and infinitely above the noise. For
/// contrast, the same comparison on GPU bf16 gives 0.125-0.1875 clean and 0.25 broken,
/// which is why this test refuses to run there — see the device note below.
const TOL_LOGITS: f32 = 1e-3;

fn codes_to_host(t: &Tensor) -> Vec<u32> {
    t.flatten_all().expect("flatten").to_vec1::<u32>().expect("to_vec1")
}

#[test]
fn fish_s2_batched_generation_matches_single_sample() {
    if std::env::var("SYRINX_FISH_BATCH_PARITY").ok().as_deref() != Some("1") {
        eprintln!(
            "SKIP real_fish_s2_batch_parity: set SYRINX_FISH_BATCH_PARITY=1 to run it \
             (loads s2-pro and generates {} times; opt-in like real_cue_activation)",
            TEXTS.len() + 1
        );
        return;
    }
    let Some(dir) = std::env::var("SYRINX_FISH_S2_DIR")
        .ok()
        .filter(|d| std::path::Path::new(d).exists())
    else {
        eprintln!("SKIP real_fish_s2_batch_parity: set SYRINX_FISH_S2_DIR to the s2-pro checkpoint");
        return;
    };

    // CPU/f32, and SYRINX_FISH_DEVICE is deliberately ignored. Measured on GPU bf16, the
    // ordinary batched-vs-single difference is 0.125-0.1875 on logits of order 18, while
    // BREAKING the left-pad mask moves it only to 0.25 — signal and noise are the same
    // order, so at bf16 this comparison cannot tell a real fault from GEMM accumulation.
    // In f32 the noise collapses and the same fault stands out. Costs ~19 GB (the CPU
    // parity path upcasts the bf16 weights), which is why this test is opt-in.
    let dev = Device::Cpu;
    eprintln!("[batch-parity] device=Cpu (f32; SYRINX_FISH_DEVICE ignored on purpose)");

    let mut model = S2Pro::load(&dir, dev.clone()).expect("load s2-pro");

    // How many of TEXTS to batch. Adjustable because the batch width a card can hold is a
    // property of the card, not of the maths being checked: s2-pro is ~9.7 GB in bf16, so
    // a 12 GB device has little room left for N sets of activations. Two distinct lengths
    // are the minimum that exercises left-padding at all, so the floor is 2.
    let n = std::env::var("SYRINX_FISH_BATCH_N")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(TEXTS.len())
        .clamp(2, TEXTS.len());
    let prompts: Vec<Tensor> =
        TEXTS[..n].iter().map(|t| model.build_prompt(t).expect("build prompt")).collect();
    eprintln!("[batch-parity] batching {n} of {} prompts", TEXTS.len());
    for (i, p) in prompts.iter().enumerate() {
        eprintln!("[batch-parity] prompt {i}: {} positions ({:?})", p.dim(1).unwrap_or(0), TEXTS[i]);
    }
    // If these were equal the test would be vacuous, so assert the premise rather than
    // trusting the texts to stay different if someone edits them.
    let lens: Vec<usize> = prompts.iter().map(|p| p.dim(1).unwrap_or(0)).collect();
    assert!(
        lens.iter().collect::<std::collections::HashSet<_>>().len() == lens.len(),
        "prompts must have DIFFERENT lengths or the left-pad path is never exercised: {lens:?}"
    );

    // ---- 1. The DETERMINISTIC comparison: prefill logits, batched vs single ----------
    //
    // This is the assertion that means something. Sampled codes cannot distinguish a real
    // masking/position bug from chaotic amplification: the batched and unbatched GEMMs
    // take different shapes, so their last bits may legitimately differ, and under
    // temperature sampling one flipped bit changes a draw and everything after it. Raw
    // prefill logits have no PRNG anywhere near them, so they isolate the maths — which
    // is exactly what the `// PARITY:` markers cover (left-pad, per-sample RoPE positions
    // `c - pad_i`, the per-sample causal+pad mask).
    // Reset between every prefill: the KV cache carries its batch width, so a single
    // prefill straight after a batched one fails with "shape mismatch on dim 0, 2 <> 1".
    let max_seq = model.config().slow.max_seq_len;
    model.reset(max_seq).expect("reset");
    let batch_step = model.prefill_batch(&prompts).expect("prefill_batch");
    let batch_logits: Vec<Tensor> =
        (0..prompts.len()).map(|i| batch_step.logits.get(i).expect("row")).collect();

    let mut logit_diffs = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        model.reset(max_seq).expect("reset");
        let single = model.prefill(prompt).expect("prefill");
        let want = single.semantic_logits.flatten_all().expect("flatten");
        let got = batch_logits[i].flatten_all().expect("flatten");
        let d = (got.to_dtype(candle_core::DType::F32).unwrap()
            - want.to_dtype(candle_core::DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        let scale = want
            .to_dtype(candle_core::DType::F32).unwrap()
            .abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        eprintln!(
            "[batch-parity] slot {i}: prefill logits max abs diff {d:.5} (reference max abs {scale:.2})"
        );
        logit_diffs.push(d);
    }

    // The gate.
    let worst = logit_diffs.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        worst <= TOL_LOGITS,
        "batched prefill logits differ from the single-sample path by {worst} (tolerance \
         {TOL_LOGITS}). This is the batched masking/position math itself — no sampling \
         enters a prefill — so suspect the left-pad, the per-sample RoPE offset `c - pad_i`, \
         or the per-sample causal+pad mask, all marked // PARITY: in slow_ar.rs. \
         Per-slot: {logit_diffs:?}"
    );
}
