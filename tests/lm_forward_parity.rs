//! T-02.02c-b — the full LM forward logits, frozen RED parity tests.
//!
//! `syrinx-lm` exposes `forward(token_ids) -> logits[T, vocab]`: the end-to-end
//! `reference.py` §5 forward, composed from the T-02.02c-a-verified stage —
//!
//!   h = embed(token_ids, tok_embeddings)        # [T, dim]
//!   for L in 0..4: h = block(h, L, positions)    # the 4 transformer blocks
//!   h = rmsnorm(h, norm.weight, eps=1e-5)        # final pre-head norm
//!   logits = linear(h, output.weight)            # [T, vocab=512], untied, no bias
//!
//! with `positions = [0..T-1]`, every weight drawn by its literal §3 name
//! (PARITY.md §3), pinned at 1e-3 max-abs against the full-forward golden
//! `lm_forward.json` on the fixed input `[1,5,9,2,0]`.
//!
//! ────────────────────────────────────────────────────────────────────────────
//! WHICH CONTROLS ARE VISIBLE AT THE 1e-3 LOGIT TOLERANCE (and which are NOT)
//! ────────────────────────────────────────────────────────────────────────────
//! The dominant embed/norm/head path IS visible at 1e-3, so this file pins it
//! with detectable negative controls. Each divergence below was measured by a
//! pure-Python port of `reference.py` §2/§4/§5 reproducing `lm_forward.json` to
//! ~5e-10, against `max|lm_forward| = 6.1e-3`:
//!
//!   * UNTIED HEAD (C3): using `tok_embeddings` as the output projection instead
//!     of the separate `output.weight` moves the logits by ~8.7e-3 (> 1e-3). The
//!     lm_head is the final, dominant projection, so this control IS visible.
//!   * FINAL-NORM PRESENCE (C4): omitting `rmsnorm(norm.weight)` before the head
//!     moves the logits by ~9.4e-3 (> 1e-3).
//!   * FINAL-NORM POSITION (C4): applying that norm AFTER `output.weight` (over
//!     the [5,512] logits, which `syrinx_core::rmsnorm` folds into 20 rows of
//!     `dim=128`) instead of before moves the logits by ~2.6e-2 (> 1e-3).
//!
//! What is DELIBERATELY NOT a control here (per the task's out-of-scope note and
//! the project memory): the BLOCK COUNT. Each of the 4 blocks contributes only
//! ~2e-4 to the logits — below the 1e-3 tol — so a 3-vs-4-block discriminator is
//! numerically unsatisfiable at logit scale (the exact wall the parent T-02.02c
//! hit). Block correctness is pinned at activation scale (×50) by the frozen
//! `lm_stage_parity.rs` / `block_prop.rs` of T-02.02c-a; this task pins only the
//! dominant embed/norm/head path and the end-to-end numeric parity. No assertion
//! in this file claims a divergence I have not measured to hold.
//!
//! RED: `syrinx-lm` exposes no `forward`, so this target fails to build. GREEN
//! adds it as the minimal §5 composition over the verified stage — every
//! operator it introduces must be killed by the parity gate below. This file is
//! frozen at red-pass; do not edit it in GREEN.

use syrinx_core::{linear, rmsnorm, weights, Tensor};
use syrinx_lm::{forward, embed_tokens, transformer_block};

// LM config (PARITY.md §3 / REFERENCE.md): vocab=512, dim=128, n_layers=4.
const VOCAB: usize = 512;
const DIM: usize = 128;
const N_LAYERS: usize = 4;
// RMSNorm epsilon (PARITY.md "Global": eps = 1e-5).
const EPS: f32 = 1e-5;

// The fixed parity input (golden `input.token_ids`) and its positions [0..T-1].
const TOKENS: [usize; 5] = [1, 5, 9, 2, 0];
const POSITIONS: [usize; 5] = [0, 1, 2, 3, 4];
const T: usize = 5;

// The full LM forward is pinned at 1e-3 max-abs (PARITY.md §5: tol 1e-3 for the
// full forward). The golden also carries its own `tol` field (1e-3).
const TOL: f32 = 1e-3;

// ----- golden plumbing --------------------------------------------------------

fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/golden/parity/{}", env!("CARGO_MANIFEST_DIR"), name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// A flat JSON number array → `Vec<f32>` (the golden `data` field).
fn flat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("data is an array")
        .iter()
        .map(|x| x.as_f64().expect("data cell is a number") as f32)
        .collect()
}

/// A JSON integer array → `Vec<usize>` (the golden `shape` field).
fn ints(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("integer array")
        .iter()
        .map(|x| x.as_u64().expect("integer cell") as usize)
        .collect()
}

fn tol(g: &serde_json::Value) -> f32 {
    g["tol"].as_f64().expect("tol is a number") as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared vectors must be equal length");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ----- named-weight helpers (PARITY.md §3 literal tensor names) ---------------

/// The `[rows, cols]` weight matrix named `name` from the T-02.01c name-seeded
/// stream (`[out,in]` orientation, as `linear` expects).
fn mat(name: &str, rows: usize, cols: usize) -> Tensor {
    Tensor::new(weights(name, rows * cols), vec![rows, cols])
}

/// The `[d]` weight vector named `name` (an RMSNorm weight).
fn vecw(name: &str, d: usize) -> Tensor {
    Tensor::new(weights(name, d), vec![d])
}

/// The hidden state after `embed -> 4 blocks` for `[1,5,9,2,0]`, built from the
/// T-02.02c-a-verified stage (`embed_tokens` + `transformer_block`). This is the
/// exact tensor the final `rmsnorm(norm.weight)` then `output.weight` head turns
/// into the golden logits — the isolated input for the C3/C4 head-and-norm
/// controls, so they do not depend on the `forward` impl under test.
fn hidden_after_blocks() -> Tensor {
    let mut h = embed_tokens(&TOKENS);
    for l in 0..N_LAYERS {
        h = transformer_block(&h, l, &POSITIONS);
    }
    h
}

// =====================================================================
// C1 — `forward` runs embed -> 4 blocks -> final norm -> head and returns
//      logits of shape [T, vocab]; for [1,5,9,2,0] that is [5, 512].
// =====================================================================

#[test]
fn test_forward_output_shape() {
    let out = forward(&TOKENS);
    assert_eq!(out.shape(), &[T, VOCAB], "forward([1,5,9,2,0]) logits must be [5,512]");
    assert_eq!(out.data().len(), T * VOCAB, "logits hold exactly 5*512 elements");

    // The leading axis tracks the token count, not a hard-coded 5: a 3-token
    // input yields [3, 512]. This pins logits.shape == [token_ids.len(), vocab].
    let three = forward(&[3, 3, 3]);
    assert_eq!(three.shape(), &[3, VOCAB], "forward of 3 tokens must be [3,512]");
}

// =====================================================================
// C2 — full-forward numeric parity against lm_forward.json within 1e-3,
//      rejected by a 2e-3 single-logit perturbation.
// =====================================================================

#[test]
fn test_forward_parity_and_perturbation() {
    let g = load("lm_forward.json");
    let want = flat(&g["data"]);
    let want_shape = ints(&g["shape"]);
    assert_eq!(want_shape, vec![T, VOCAB], "golden lm_forward shape is [5,512]");
    assert_eq!(want.len(), T * VOCAB, "golden holds all 5*512 logits");
    assert!(tol(&g) <= TOL, "golden tol must be <= 1e-3");
    assert!((tol(&g) - TOL).abs() <= 1e-12, "golden tol is exactly 1e-3");

    // forward([1,5,9,2,0]) matches every one of the 5*512 logits within 1e-3.
    let out = forward(&TOKENS);
    assert_eq!(out.shape(), want_shape.as_slice(), "logits shape must equal the golden's");
    assert!(
        max_abs_diff(out.data(), &want) <= tol(&g),
        "forward([1,5,9,2,0]) must match lm_forward.json within 1e-3 max-abs"
    );

    // Non-vacuity: the golden carries a real signal (max|logit| ~6.1e-3), so the
    // 1e-3 gate is not trivially satisfied by an all-zero output.
    assert!(
        want.iter().map(|v| v.abs()).fold(0.0f32, f32::max) > TOL,
        "the golden logits are non-trivial (max-abs exceeds the 1e-3 tol)"
    );
    let zeros = vec![0.0f32; T * VOCAB];
    assert!(
        max_abs_diff(&zeros, &want) > TOL,
        "an all-zero logit tensor must diverge from the golden"
    );

    // Boundary: perturbing ANY single logit by 2e-3 pushes max-abs past 1e-3 and
    // fails the assertion — the parity check is element-wise sensitive, both at
    // the head (index 0) and the tail (last index) of the flattened logits.
    let mut head = out.data().to_vec();
    head[0] += 2e-3;
    assert!(
        max_abs_diff(&head, &want) > tol(&g),
        "a 2e-3 perturbation of the first logit must fail the 1e-3 parity gate"
    );
    let mut tail = out.data().to_vec();
    let last = tail.len() - 1;
    tail[last] += 2e-3;
    assert!(
        max_abs_diff(&tail, &want) > tol(&g),
        "a 2e-3 perturbation of the last logit must fail the 1e-3 parity gate"
    );

    // And non-perpetual: the un-perturbed forward output still passes the SAME
    // gate, so the failures above are caused by the perturbation, not a check
    // that can never pass.
    assert!(
        max_abs_diff(out.data(), &want) <= tol(&g),
        "the un-perturbed forward output must pass the 1e-3 parity gate"
    );
}

// =====================================================================
// C3 — the lm_head is UNTIED: `output.weight`, not `tok_embeddings`.
//      The head is the dominant final projection, so the control IS visible.
// =====================================================================

#[test]
fn test_untied_head_control() {
    let g = load("lm_forward.json");
    let want = flat(&g["data"]);

    let h = hidden_after_blocks();
    let norm_w = vecw("norm.weight", DIM);
    let normed = rmsnorm(&h, &norm_w, EPS);

    // CORRECT — the separate, untied `output.weight` head reproduces the golden
    // within 1e-3, exactly as `forward` does.
    let out_w = mat("output.weight", VOCAB, DIM);
    let correct = linear(&normed, &out_w, None);
    assert_eq!(correct.shape(), &[T, VOCAB], "untied-head logits must be [5,512]");
    assert!(
        max_abs_diff(correct.data(), &want) <= tol(&g),
        "the untied output.weight head must reproduce lm_forward within 1e-3"
    );
    // Tie to the impl under test: `forward` reproduces the same golden, so it is
    // applying this untied head, not a tied one.
    let fwd = forward(&TOKENS);
    assert!(
        max_abs_diff(fwd.data(), &want) <= tol(&g),
        "forward must reproduce lm_forward via the untied head"
    );

    // WRONG — tying the head (using `tok_embeddings` [512,128] as the output
    // projection instead of `output.weight`) moves the logits by ~8.7e-3, well
    // past 1e-3, and fails. The two tensors are distinct name-seeded streams, so
    // this is a genuine substitution on the dominant path.
    let tied = mat("tok_embeddings", VOCAB, DIM);
    let wrong = linear(&normed, &tied, None);
    assert_eq!(wrong.shape(), &[T, VOCAB], "the tied-head control still yields [5,512]");
    assert!(
        max_abs_diff(wrong.data(), &want) > TOL,
        "a tied head (tok_embeddings as output proj) must diverge from lm_forward"
    );
}

// =====================================================================
// C4 — the final `rmsnorm(norm.weight)` is PRESENT and BEFORE the head.
//      Omitting it, or applying it after the head, each diverges > 1e-3.
// =====================================================================

#[test]
fn test_final_norm_presence_and_position() {
    let g = load("lm_forward.json");
    let want = flat(&g["data"]);

    let h = hidden_after_blocks();
    let norm_w = vecw("norm.weight", DIM);
    let out_w = mat("output.weight", VOCAB, DIM);

    // CORRECT — norm THEN head reproduces the golden within 1e-3, as `forward`
    // does. (Tie to the impl: `forward` matches the same golden.)
    let correct = linear(&rmsnorm(&h, &norm_w, EPS), &out_w, None);
    assert!(
        max_abs_diff(correct.data(), &want) <= tol(&g),
        "rmsnorm(norm.weight) THEN output.weight must reproduce lm_forward within 1e-3"
    );
    let fwd = forward(&TOKENS);
    assert!(
        max_abs_diff(fwd.data(), &want) <= tol(&g),
        "forward must reproduce lm_forward with norm-before-head ordering"
    );

    // WRONG #1 — OMITTING the final norm (project the un-normed hidden state
    // straight through the head) moves the logits by ~9.4e-3 (> 1e-3) and fails.
    let no_norm = linear(&h, &out_w, None);
    assert_eq!(no_norm.shape(), &[T, VOCAB], "the omit-norm control still yields [5,512]");
    assert!(
        max_abs_diff(no_norm.data(), &want) > TOL,
        "omitting the final rmsnorm(norm.weight) must diverge from lm_forward"
    );

    // WRONG #2 — applying the final norm AFTER the head instead of before. The
    // head sees the un-normed hidden state; the norm then folds over the [5,512]
    // logits (as 20 rows of dim=128). This moves the logits by ~2.6e-2 (> 1e-3)
    // and fails — pinning that the norm precedes the head.
    let norm_after = rmsnorm(&no_norm, &norm_w, EPS);
    assert_eq!(norm_after.shape(), &[T, VOCAB], "the norm-after control still yields [5,512]");
    assert!(
        max_abs_diff(norm_after.data(), &want) > TOL,
        "applying rmsnorm(norm.weight) AFTER the head must diverge from lm_forward"
    );
}
