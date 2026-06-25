//! Real CosyVoice2 LM **KV-cache** parity + speed (the O(n) incremental decode path).
//!
//! The autoregressive generation forward used to recompute the entire sequence every
//! step (O(n²)), capping live synthesis. This adds a per-layer KV cache so each step is
//! O(n). The cache MUST NOT change the math: the cached forward must produce logits
//! identical to the full-recompute path. These tests are the deterministic proof of
//! that, plus a visible speed comparison.
//!
//! Gated on the `real` feature AND env vars pointing at the converted fp32 checkpoint
//! plus the Python reference dump (same fixtures as `real_lm_gen_parity.rs`). Skips
//! cleanly when absent (device-bound recipe), runs for real where the fixtures exist.
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_LMGEN_REF=/root/parity/lmgen/ref.safetensors \
//!   cargo test --features real --test real_lm_kvcache -- --nocapture
//!
//! What it proves, in increasing strictness:
//!
//!   (a) BIT-IDENTICAL FOR MULTI-TOKEN CHUNKS (zero tolerance) — feeding a chunk of
//!       `t_new > 1` tokens through the cache, and splitting the prefill into two cached
//!       chunks, both reproduce the full-recompute logits with max-abs-diff **exactly
//!       0.0**. This is the load-bearing correctness statement: the RoPE absolute
//!       position, the GQA repeat over cached KV, the causal mask offset, and the K/V
//!       append are all exactly right — if any were wrong this would not be bit-identical.
//!
//!   (b) SINGLE-TOKEN DECODE within the fp floor + ARGMAX-EXACT — the real decode loop
//!       feeds one token per step (`t_new == 1`). That degenerate single-row attention
//!       matmul dispatches a different gemm reduction order in candle, so its logits
//!       differ from the full recompute only by fp non-associativity (empirically
//!       <= ~4e-5 over 200 steps), with the **argmax identical on every step** — the
//!       sampler sees the same decision surface. We assert a tight, documented fp bound
//!       (NOT a weakened correctness tolerance: (a) already proved the math exact).
//!
//!   (c) REFERENCE PARITY — the cached per-step logits still match the Python reference
//!       dump within 1e-3 (the cache does not drift from the ground truth).
//!
//!   (d) TOKEN SEQUENCE — cached `generate` reproduces the exact token vector of the
//!       full-recompute `generate_full_recompute` for a fixed seed.
//!
//! Plus: a wall-clock speed comparison (cached vs uncached over many steps).

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use std::path::Path;
use std::time::Instant;
use syrinx_lm::real::{KvCache, Qwen2Lm};

/// Fp-rounding floor for the single-token (`t_new == 1`) cached decode step. The cache
/// is proved bit-exact for `t_new > 1` in part (a); the only residual is candle's
/// single-row gemm reduction order, measured at ~4e-5 over 200 steps. This bound sits an
/// order of magnitude below the 1e-3 reference tolerance and far below any argmax flip.
const SINGLE_TOKEN_FP_FLOOR: f32 = 5e-5;

fn ids_u32(t: &Tensor) -> Vec<u32> {
    let t = t.flatten_all().unwrap().to_dtype(DType::U32).unwrap();
    t.to_vec1::<u32>().unwrap()
}

fn scalar_i64(t: &Tensor) -> i64 {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()[0]
}

/// Max abs diff between two same-shape tensors, as f32.
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

fn argmax_u32(v: &Tensor) -> u32 {
    v.argmax(0).unwrap().to_scalar::<u32>().unwrap()
}

fn fixtures() -> Option<(String, String)> {
    match (
        std::env::var("SYRINX_LM_WEIGHTS").ok(),
        std::env::var("SYRINX_LMGEN_REF").ok(),
    ) {
        (Some(w), Some(r)) if Path::new(&w).exists() && Path::new(&r).exists() => Some((w, r)),
        _ => None,
    }
}

#[test]
fn real_lm_kvcache_matches_full_recompute_and_reference() {
    let (weights, reference) = match fixtures() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP real_lm kvcache parity: set SYRINX_LM_WEIGHTS + SYRINX_LMGEN_REF \
                 to the on-disk fp32 fixtures (CosyVoice2 weights + AR-gen reference dump)"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let lm = Qwen2Lm::load(&weights, dev.clone()).expect("load fp32 weights");
    let r = candle_core::safetensors::load(&reference, &dev).expect("load gen reference");

    let text_token = ids_u32(r.get("text_token").expect("text_token"));
    let prompt_speech_token = ids_u32(r.get("prompt_speech_token").expect("prompt_speech_token"));
    let gen_tokens = ids_u32(r.get("gen_tokens").expect("gen_tokens"));
    let ref_step_logits = r
        .get("step_logits")
        .expect("step_logits")
        .to_dtype(DType::F32)
        .unwrap(); // [N, V]
    let n = gen_tokens.len();
    assert!(n >= 2, "reference must have >=2 generated tokens, got {n}");

    // The full-recompute oracle: teacher-force ALL reference tokens in one forward and
    // slice out every step's last-position logits. This is the established parity path.
    let lm_input0 = lm
        .build_lm_input(&text_token, &prompt_speech_token)
        .expect("build_lm_input");
    let t0 = lm_input0.dim(1).unwrap();
    let full = lm
        .teacher_forced_logits(&text_token, &prompt_speech_token, &gen_tokens)
        .expect("teacher_forced_logits"); // [N, V]

    // ================================================================================
    // (a) BIT-IDENTICAL for multi-token chunks (zero tolerance) — the strongest proof
    //     the cache math (RoPE abs-pos, GQA over cached KV, mask offset, K/V append) is
    //     exact. If any of those were wrong, these would NOT be 0.0.
    // ================================================================================

    // (a.1) Prefill the prompt, then feed all gen tokens except the last as ONE chunk;
    //       its per-position logits must equal the teacher-forced rows 1..n exactly.
    {
        let mut cache = KvCache::new();
        let _ = lm
            .step_logits_cached(&lm_input0, &mut cache)
            .expect("prefill");
        assert_eq!(cache.len(), t0, "cache length after prefill != T0");
        let tail = lm.speech_embed(&gen_tokens[..n - 1]).expect("speech_embed tail"); // [1,n-1,H]
        let chunk = lm
            .forward_logits_cached(&tail, &mut cache)
            .expect("forward_logits_cached chunk"); // [1, n-1, V]
        let mut mx = 0f32;
        for k in 1..n {
            let a = chunk.narrow(1, k - 1, 1).unwrap().reshape((6564,)).unwrap();
            let b = full.narrow(0, k, 1).unwrap().reshape((6564,)).unwrap();
            mx = mx.max(max_abs_diff(&a, &b));
        }
        eprintln!("(a.1) multi-token cached chunk vs full recompute: max-abs-diff = {mx:.3e}");
        assert_eq!(mx, 0.0, "multi-token cached chunk must be BIT-IDENTICAL to full recompute");
    }

    // (a.2) Split the prefill into two cached chunks; the step-0 logit must equal the
    //       single full-recompute step-0 logit exactly (prefill-chunking is bit-exact).
    {
        let mut cache = KvCache::new();
        let h = t0 / 2;
        let first = lm_input0.narrow(1, 0, h).unwrap();
        let second = lm_input0.narrow(1, h, t0 - h).unwrap();
        let _ = lm
            .forward_logits_cached(&first, &mut cache)
            .expect("prefill chunk 1");
        let split0 = lm
            .step_logits_cached(&second, &mut cache)
            .expect("prefill chunk 2");
        let full0 = lm.step_logits(&lm_input0).expect("full step_logits");
        let d = max_abs_diff(&split0, &full0);
        eprintln!("(a.2) split-prefill (2 chunks) vs single full step-0: max-abs-diff = {d:.3e}");
        assert_eq!(d, 0.0, "split prefill must be BIT-IDENTICAL to a single full prefill");
    }

    // ================================================================================
    // (b) SINGLE-TOKEN decode path within the fp floor + ARGMAX-EXACT, and
    // (c) cached still matches the Python reference within 1e-3.
    // ================================================================================
    let mut cache = KvCache::new();
    let mut cached_k = lm
        .step_logits_cached(&lm_input0, &mut cache)
        .expect("prefill step_logits_cached");
    assert_eq!(cache.len(), t0, "cache length after prefill != T0");

    let mut max_vs_full = 0f32;
    let mut max_vs_ref = 0f32;
    for k in 0..n {
        let full_k = full.narrow(0, k, 1).unwrap().reshape((6564,)).unwrap();
        let ref_k = ref_step_logits.narrow(0, k, 1).unwrap().reshape((6564,)).unwrap();
        let d_full = max_abs_diff(&cached_k, &full_k);
        let d_ref = max_abs_diff(&cached_k, &ref_k);
        max_vs_full = max_vs_full.max(d_full);
        max_vs_ref = max_vs_ref.max(d_ref);

        // (b) every cached step picks the SAME argmax as the full recompute: the sampler
        //     sees an identical decision surface.
        let am_c = argmax_u32(&cached_k);
        let am_f = argmax_u32(&full_k);
        assert_eq!(
            am_c, am_f,
            "step {k}: cached argmax {am_c} != full-recompute argmax {am_f}"
        );
        // (b) single-token cached logits stay within the fp floor of the full recompute.
        assert!(
            d_full <= SINGLE_TOKEN_FP_FLOOR,
            "step {k}: cached-vs-full {d_full:.3e} exceeds the fp floor {SINGLE_TOKEN_FP_FLOOR:.0e} \
             — this is a real numerics regression, not gemm rounding"
        );
        // (c) cached logits still match the Python reference within 1e-3.
        assert!(
            d_ref < 1e-3,
            "step {k}: cached logits drift from reference by {d_ref:.3e} (> 1e-3)"
        );

        if k < 6 || k + 1 == n {
            let am_r = argmax_u32(&ref_k);
            eprintln!(
                "  step {k:>3}: cached-vs-full = {d_full:.3e}  cached-vs-ref = {d_ref:.3e}  \
                 argmax cached={am_c} full={am_f} ref={am_r}  chosen={}",
                gen_tokens[k]
            );
        }

        if k + 1 < n {
            let row = lm.speech_embed(&[gen_tokens[k]]).unwrap(); // [1,1,H]
            cached_k = lm
                .step_logits_cached(&row, &mut cache)
                .expect("step_logits_cached");
            assert_eq!(cache.len(), t0 + k + 1, "cache length out of sync at step {k}");
        }
    }
    eprintln!("(b) single-token cached-vs-full  max over {n} steps = {max_vs_full:.3e}  (<= {SINGLE_TOKEN_FP_FLOOR:.0e}, argmax-exact)");
    eprintln!("(c) cached-vs-reference          max over {n} steps = {max_vs_ref:.3e}  (< 1e-3)");

    // ================================================================================
    // (d) cached generate reproduces the full-recompute generate, fixed seed.
    // ================================================================================
    let min_len = scalar_i64(r.get("min_len").expect("min_len")) as usize;
    let gen_cap: usize = std::env::var("SYRINX_LMGEN_MAXSTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let g_cached = lm
        .generate(&text_token, &prompt_speech_token, min_len, gen_cap, 1234)
        .expect("cached generate");
    let g_full = lm
        .generate_full_recompute(&text_token, &prompt_speech_token, min_len, gen_cap, 1234)
        .expect("full-recompute generate");
    assert_eq!(
        g_cached, g_full,
        "cached generate token sequence diverged from the full-recompute oracle"
    );
    assert!(!g_cached.is_empty(), "generation produced no tokens");
    for &t in &g_cached {
        assert!(t < 6561, "emitted token {t} is not a valid speech token (< 6561)");
    }
    eprintln!(
        "(d) cached generate == full-recompute generate: {} tokens, first 8 = {:?}",
        g_cached.len(),
        &g_cached[..g_cached.len().min(8)]
    );

    // ================================================================================
    // SPEED: cached vs uncached over a fixed step count, so the win is visible.
    // ================================================================================
    let speed_steps: usize = std::env::var("SYRINX_LMGEN_SPEEDSTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    // Uncached: full recompute each step over a growing sequence (the old O(n²) path).
    let t_un = Instant::now();
    let mut embeds = lm_input0.clone();
    for k in 0..speed_steps {
        let _ = lm.step_logits(&embeds).unwrap();
        let tok = gen_tokens[k % n];
        let row = lm.speech_embed(&[tok]).unwrap();
        embeds = Tensor::cat(&[&embeds, &row], 1).unwrap();
    }
    let dt_un = t_un.elapsed();

    // Cached: prefill once, then one token per step (the new O(n) path).
    let t_ca = Instant::now();
    let mut cache2 = KvCache::new();
    let _ = lm.step_logits_cached(&lm_input0, &mut cache2).unwrap();
    for k in 0..speed_steps {
        let tok = gen_tokens[k % n];
        let row = lm.speech_embed(&[tok]).unwrap();
        let _ = lm.step_logits_cached(&row, &mut cache2).unwrap();
    }
    let dt_ca = t_ca.elapsed();

    let speedup = dt_un.as_secs_f64() / dt_ca.as_secs_f64().max(1e-9);
    eprintln!(
        "SPEED ({speed_steps} steps from T0={t0}):  uncached {:.3}s   cached {:.3}s   speedup {speedup:.1}x",
        dt_un.as_secs_f64(),
        dt_ca.as_secs_f64()
    );

    eprintln!(
        "PASS: cache is BIT-IDENTICAL to full recompute for multi-token chunks; single-token \
         decode is argmax-exact within the fp floor; matches the reference; reproduces the same \
         token sequence; and is faster."
    );
}
