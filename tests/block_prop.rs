//! T-02.02b — the SwiGLU transformer block, frozen RED tests.
//!
//! Two new `syrinx-lm` entry points, exact transcriptions of `reference.py`
//! §5.3 (SwiGLU FFN) and §5.1 (pre-RMSNorm residual block) = PARITY.md §5.3/§5.1:
//!
//!   * `swiglu_ffn(x[T,dim], w1, w3, w2) -> [T,dim]` with
//!       gate = silu(linear(x, w1)) , up = linear(x, w3) , out = linear(gate ⊙ up, w2)
//!     where `w1`=gate `[ffn_hidden,dim]`, `w3`=up `[ffn_hidden,dim]`,
//!     `w2`=down `[dim,ffn_hidden]` and `⊙` is the elementwise Hadamard product.
//!   * `block(h[T,dim], attn_norm_w, ffn_norm_w, attn_weights, ffn_weights, positions)`
//!     applies the §5.1 order — `h = h + attention(rmsnorm(h, attn_norm_w))` then
//!     `h = h + swiglu_ffn(rmsnorm(h, ffn_norm_w))` — each residual adding the
//!     sub-block output to the PRE-norm `h`. `attn_weights` is the `(wq,wk,wv,wo)`
//!     tuple §5.2 consumes; `ffn_weights` is the `(w1,w3,w2)` tuple §5.3 consumes.
//!
//! These tests pin the four gateable properties (no end-to-end golden — exact
//! numeric values are covered by T-02.02c):
//!
//!   * C1 — FFN shape `[T,dim]`, and the gate/up roles: swapping `w1`↔`w3` (so
//!     silu wraps the up instead of the gate) changes the output.
//!   * C2 — SwiGLU elementwise structure: replacing `gate ⊙ up` with `gate + up`
//!     changes the output, and zeroing `w3` (the up path) drives the output to
//!     all-zeros.
//!   * C3 — pre-norm residual identity: with both sub-block weight sets zeroed the
//!     block is the identity (`block(h) == h` to ±1e-6), which holds only because
//!     each residual adds to the un-normed input `h`.
//!   * C4 — the residual targets the PRE-norm tensor, not the normed one: for an
//!     input whose RMS ≠ 1 the zeroed-weights block returns `h`, not the (clearly
//!     different) normed value a residual-on-normed order would yield.
//!
//! RED: `syrinx-lm` exposes no `swiglu_ffn`/`block`, so this target fails to
//! build. GREEN adds them as the minimal §5.3/§5.1 transcription. To keep the
//! task-scoped mutation gate honest, GREEN should COMPOSE both fns purely from
//! the already-verified `syrinx-core` primitives (`silu`/`linear`/`mul`/`add`/
//! `rmsnorm`) and the verified `attention` — so the new code introduces no bare
//! arithmetic operators of its own — and place them in their OWN source file so
//! `crates/syrinx-lm/src/lib.rs` (which holds T-02.02a's `attention`) stays
//! byte-identical and out of the mutation target set. This file is frozen at
//! red-pass; do not edit it in GREEN.

use syrinx_core::{add, linear, mul, rmsnorm, silu, weights, Tensor};
use syrinx_lm::{block, swiglu_ffn};

// LM config (PARITY.md §3 / REFERENCE.md): dim=128, ffn_hidden=256.
const DIM: usize = 128;
const FFN_HIDDEN: usize = 256;
// RMSNorm eps (PARITY.md "Global": eps = 1e-5).
const EPS: f32 = 1e-5;

/// `impl == independent reference` (both accumulate through the identical
/// `syrinx-core` ops, so the gap is essentially float noise; single-op tol 1e-4).
const PARITY_TOL: f32 = 1e-4;
/// Zeroed-weights residual identity: the add-of-zero is bit-exact, so 0 ≤ 1e-6.
const IDENTITY_TOL: f32 = 1e-6;
/// Zeroing the up path makes the output bit-exactly zero; allow a hair of slack.
const ZERO_TOL: f32 = 1e-12;
/// A deliberately-wrong variant (swapped roles, `+` instead of `⊙`, residual on
/// the normed tensor) must diverge by at least this — two orders above PARITY_TOL.
const FAIL_GAP: f32 = 1e-2;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared vectors must be equal length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// A deterministic `[rows, cols]` tensor from the T-02.01c name-seeded stream
/// (`weights` returns f32 in [-0.02, 0.02)), scaled to a healthy magnitude so the
/// SiLU nonlinearity is well into its curved region and the wrong-variant gaps
/// land comfortably above `FAIL_GAP`.
fn tensor2d(name: &str, rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = weights(name, rows * cols).iter().map(|w| w * scale).collect();
    Tensor::new(data, vec![rows, cols])
}

/// An all-zeros `[rows, cols]` tensor.
fn zeros(rows: usize, cols: usize) -> Tensor {
    Tensor::new(vec![0.0f32; rows * cols], vec![rows, cols])
}

/// An all-ones RMSNorm weight `[d]` (a pass-through scale, so the norm only does
/// its mean-square renormalization).
fn ones(d: usize) -> Tensor {
    Tensor::new(vec![1.0f32; d], vec![d])
}

/// Independent transcription of `reference.py` §5.3, parameterised by `hadamard`
/// so the C2 test can construct the wrong `gate + up` variant. `gate =
/// silu(linear(x, w1))`, `up = linear(x, w3)`, `out = linear(gate ⊙/+ up, w2)`.
fn ref_swiglu(x: &Tensor, w1: &Tensor, w3: &Tensor, w2: &Tensor, hadamard: bool) -> Tensor {
    let gate = silu(&linear(x, w1, None));
    let up = linear(x, w3, None);
    let combined = if hadamard {
        mul(&gate, &up)
    } else {
        add(&gate, &up)
    }
    .unwrap();
    linear(&combined, w2, None)
}

// ----- C1: FFN output shape + gate/up roles ----------------------------------

#[test]
fn test_swiglu_output_shape_is_t_by_dim() {
    let t = 3;
    let x = tensor2d("ffn.x", t, DIM, 50.0);
    let w1 = tensor2d("ffn.w1", FFN_HIDDEN, DIM, 50.0);
    let w3 = tensor2d("ffn.w3", FFN_HIDDEN, DIM, 50.0);
    let w2 = tensor2d("ffn.w2", DIM, FFN_HIDDEN, 50.0);

    let out = swiglu_ffn(&x, &w1, &w3, &w2);

    assert_eq!(out.shape(), &[t, DIM], "swiglu_ffn must return shape [T, dim]");
}

#[test]
fn test_swiglu_gate_up_roles_and_swap() {
    let t = 3;
    let x = tensor2d("ffn.x", t, DIM, 50.0);
    let w1 = tensor2d("ffn.w1", FFN_HIDDEN, DIM, 50.0);
    let w3 = tensor2d("ffn.w3", FFN_HIDDEN, DIM, 50.0);
    let w2 = tensor2d("ffn.w2", DIM, FFN_HIDDEN, 50.0);

    let out = swiglu_ffn(&x, &w1, &w3, &w2);

    // The impl computes exactly silu(x·w1) ⊙ (x·w3) then down-projects with w2:
    // w1 is the gate (wrapped by silu), w3 is the up.
    let reference = ref_swiglu(&x, &w1, &w3, &w2, true);
    assert!(
        max_abs_diff(out.data(), reference.data()) <= PARITY_TOL,
        "swiglu_ffn does not match silu(x·w1) ⊙ (x·w3) projected by w2"
    );

    // Swapping w1 and w3 makes silu wrap the up instead of the gate. SiLU is
    // nonlinear, so this changes the output — pinning that w1 is the gate.
    let swapped = swiglu_ffn(&x, &w3, &w1, &w2);
    assert!(
        max_abs_diff(out.data(), swapped.data()) > FAIL_GAP,
        "swapping w1↔w3 (silu on the up path) must change the FFN output"
    );
}

// ----- C2: SwiGLU elementwise structure (Hadamard, up-path drives output) -----

#[test]
fn test_swiglu_hadamard_not_sum() {
    let t = 3;
    let x = tensor2d("ffn.x", t, DIM, 50.0);
    let w1 = tensor2d("ffn.w1", FFN_HIDDEN, DIM, 50.0);
    let w3 = tensor2d("ffn.w3", FFN_HIDDEN, DIM, 50.0);
    let w2 = tensor2d("ffn.w2", DIM, FFN_HIDDEN, 50.0);

    let out = swiglu_ffn(&x, &w1, &w3, &w2);
    let hadamard = ref_swiglu(&x, &w1, &w3, &w2, true);
    let sum = ref_swiglu(&x, &w1, &w3, &w2, false);

    // The impl uses the elementwise Hadamard product gate ⊙ up.
    assert!(
        max_abs_diff(out.data(), hadamard.data()) <= PARITY_TOL,
        "swiglu_ffn must combine gate and up with the elementwise product"
    );
    // Replacing ⊙ with + (gate + up) changes the output.
    assert!(
        max_abs_diff(out.data(), sum.data()) > FAIL_GAP,
        "using gate + up instead of gate ⊙ up must change the FFN output"
    );
    // Non-vacuous: the two combiners genuinely differ for this input.
    assert!(
        max_abs_diff(hadamard.data(), sum.data()) > FAIL_GAP,
        "the ⊙ and + references must differ for this input"
    );
}

#[test]
fn test_swiglu_zero_up_drives_output_zero() {
    let t = 3;
    let x = tensor2d("ffn.x", t, DIM, 50.0);
    let w1 = tensor2d("ffn.w1", FFN_HIDDEN, DIM, 50.0);
    let w3 = tensor2d("ffn.w3", FFN_HIDDEN, DIM, 50.0);
    let w2 = tensor2d("ffn.w2", DIM, FFN_HIDDEN, 50.0);

    // Zeroing the up path (w3) makes up = 0, so gate ⊙ up = 0, so the down
    // projection of an all-zero hidden is all-zero. This holds for the Hadamard
    // product but NOT for gate + up (which would leave the silu(gate) term).
    let w3_zero = zeros(FFN_HIDDEN, DIM);
    let out = swiglu_ffn(&x, &w1, &w3_zero, &w2);
    assert!(
        out.data().iter().all(|&v| v.abs() <= ZERO_TOL),
        "zeroing w3 (the up path) must drive the FFN output to all zeros"
    );

    // Non-vacuous: with a real up path the output is not all-zeros.
    let nonzero = swiglu_ffn(&x, &w1, &w3, &w2);
    assert!(
        nonzero.data().iter().any(|&v| v.abs() > FAIL_GAP),
        "with a non-zero up path the FFN output must be non-trivial"
    );
}

// ----- C3: pre-RMSNorm residual identity under zeroed sub-block weights -------

#[test]
fn test_block_zeroed_weights_is_identity() {
    let t = 3;
    // RMS ≠ 1 so rmsnorm(h) ≠ h: the identity can only hold if the residual adds
    // to the PRE-norm h (see C4).
    let h = tensor2d("blk.h", t, DIM, 50.0);
    let attn_norm_w = ones(DIM);
    let ffn_norm_w = ones(DIM);
    let positions = [0usize, 1, 2];

    // Zero every sub-block weight: attention(rmsnorm(h)) = 0 (v = 0 -> ctx = 0),
    // and swiglu_ffn(rmsnorm(h)) = 0 (up = 0 -> gate ⊙ up = 0). Each residual then
    // adds 0 to the pre-norm h, so the block is the identity.
    let zq = zeros(DIM, DIM);
    let zk = zeros(DIM, DIM);
    let zv = zeros(DIM, DIM);
    let zo = zeros(DIM, DIM);
    let zw1 = zeros(FFN_HIDDEN, DIM);
    let zw3 = zeros(FFN_HIDDEN, DIM);
    let zw2 = zeros(DIM, FFN_HIDDEN);

    let out = block(
        &h,
        &attn_norm_w,
        &ffn_norm_w,
        (&zq, &zk, &zv, &zo),
        (&zw1, &zw3, &zw2),
        &positions,
    );

    assert!(
        max_abs_diff(out.data(), h.data()) <= IDENTITY_TOL,
        "with both sub-block weight sets zeroed the block must be the identity"
    );

    // Non-vacuous: with real sub-block weights the block is NOT the identity, so
    // the identity above is a real consequence of zeroing, not always-true.
    let wq = tensor2d("blk.wq", DIM, DIM, 50.0);
    let wk = tensor2d("blk.wk", DIM, DIM, 50.0);
    let wv = tensor2d("blk.wv", DIM, DIM, 50.0);
    let wo = tensor2d("blk.wo", DIM, DIM, 50.0);
    let w1 = tensor2d("blk.w1", FFN_HIDDEN, DIM, 50.0);
    let w3 = tensor2d("blk.w3", FFN_HIDDEN, DIM, 50.0);
    let w2 = tensor2d("blk.w2", DIM, FFN_HIDDEN, 50.0);
    let active = block(
        &h,
        &attn_norm_w,
        &ffn_norm_w,
        (&wq, &wk, &wv, &wo),
        (&w1, &w3, &w2),
        &positions,
    );
    assert!(
        max_abs_diff(active.data(), h.data()) > FAIL_GAP,
        "with non-zero sub-block weights the block must change h"
    );
}

// ----- C4: the residual targets the PRE-norm tensor, not the normed one -------

#[test]
fn test_block_residual_targets_prenorm_not_normed() {
    let t = 3;
    let h = tensor2d("blk.h", t, DIM, 50.0);
    let attn_norm_w = ones(DIM);
    let ffn_norm_w = ones(DIM);
    let positions = [0usize, 1, 2];

    let zq = zeros(DIM, DIM);
    let zk = zeros(DIM, DIM);
    let zv = zeros(DIM, DIM);
    let zo = zeros(DIM, DIM);
    let zw1 = zeros(FFN_HIDDEN, DIM);
    let zw3 = zeros(FFN_HIDDEN, DIM);
    let zw2 = zeros(DIM, FFN_HIDDEN);

    let out = block(
        &h,
        &attn_norm_w,
        &ffn_norm_w,
        (&zq, &zk, &zv, &zo),
        (&zw1, &zw3, &zw2),
        &positions,
    );

    // The wrong order `h = rmsnorm(h) + sub_block(...)` (residual on the NORMED
    // value) would, with zeroed sub-blocks, return rmsnorm(rmsnorm(h, attn_norm),
    // ffn_norm) — the doubly-normed tensor, not h.
    let normed_once = rmsnorm(&h, &attn_norm_w, EPS);
    let normed_twice = rmsnorm(&normed_once, &ffn_norm_w, EPS);

    // The correct §5.1 order adds to the pre-norm h, so the block returns h.
    assert!(
        max_abs_diff(out.data(), h.data()) <= IDENTITY_TOL,
        "the zeroed-weights block must return the pre-norm h"
    );
    // Non-vacuous: h is genuinely un-normalized, so the normed value differs.
    assert!(
        max_abs_diff(normed_twice.data(), h.data()) > FAIL_GAP,
        "input h must not already be normalized (else the test is vacuous)"
    );
    // Therefore the impl adds to the PRE-norm tensor: it returns h, not the
    // residual-on-normed value.
    assert!(
        max_abs_diff(out.data(), normed_twice.data()) > FAIL_GAP,
        "the block must add the sub-block output to the pre-norm h, not the normed tensor"
    );
}
