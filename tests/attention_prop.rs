//! T-02.02a — multi-head causal self-attention, frozen RED tests.
//!
//! `syrinx_lm::attention(x[T,dim], wq, wk, wv, wo, positions) -> [T,dim]` is the
//! causal self-attention sub-block, an exact transcription of `reference.py`
//! §5.2 (= PARITY.md §5.2): bias-free Q/K/V projections, a contiguous split of
//! each `[T,dim]` into `n_heads=4` head slices of `head_dim=32` (head `hi` owns
//! columns `hi*head_dim..(hi+1)*head_dim`), RoPE on Q and K only, score scaling
//! of `1/sqrt(head_dim)` applied to the Q·K dot product *before* the additive
//! causal mask, softmax over keys, the value-weighted context, and the `wo`
//! output projection. These tests pin the four gateable properties:
//!
//!   * C1 — shape: the output is `[T, dim]` for `T=4`; and a dense end-to-end
//!     parity check against an independent in-test reference pins the whole
//!     projection→split→RoPE→combine→`wo` pipeline numerically.
//!   * C2 — causality: recomputing attention after arbitrarily changing input
//!     rows at positions `> i` leaves output row `i` unchanged to ±1e-5 for every
//!     `i`. Dropping the mask add leaks a future key, which moves row `i` and
//!     fails the assertion. A control asserts the perturbation *does* move a
//!     future output row (so the test is not vacuous).
//!   * C3 — scaling: the score scale is exactly `1/sqrt(head_dim)`, applied
//!     before the mask add. Pinned against a hand-controlled single-key-gap input
//!     by comparing the impl to a reference computed with the correct scale and
//!     asserting it diverges from references computed with `1/head_dim` and
//!     `1/sqrt(dim)`.
//!   * C4 — head split: the per-head split is contiguous and head-local. Pinned
//!     by comparing the impl to a contiguous reference and asserting it diverges
//!     from an interleaved-split reference (head `hi` reads columns
//!     `{hi, hi+n_heads, ...}` instead of a contiguous block).
//!
//! RED: `syrinx-lm` exposes no `attention`, so this target fails to build. GREEN
//! adds it as the minimal transcription of `reference.py` §5.2. This file is
//! frozen at red-pass; do not edit it in GREEN.

use syrinx_core::{linear, rope, weights, Tensor};
use syrinx_lm::attention;

// LM config (PARITY.md §3): head_dim = dim / n_heads = 32.
const DIM: usize = 128;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 32;
const THETA: f32 = 10000.0;

/// Tolerance for `impl == independent reference` (both accumulate in f64 then
/// round to f32; single-op golden tol is 1e-4).
const PARITY_TOL: f32 = 1e-4;
/// C2's causal bound (the unchanged row is in fact bit-identical, so 0 ≤ 1e-5).
const CAUSAL_TOL: f32 = 1e-5;
/// A deliberately-wrong variant (wrong scale, interleaved split, leaked future)
/// must diverge by at least this — two orders of magnitude above `PARITY_TOL`.
const FAIL_GAP: f32 = 1e-2;

fn correct_scale() -> f64 {
    1.0 / (HEAD_DIM as f64).sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared vectors must be equal length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn row(t: &Tensor, i: usize) -> &[f32] {
    &t.data()[i * DIM..(i + 1) * DIM]
}

/// A deterministic `[rows, cols]` tensor from the T-02.01c name-seeded stream,
/// scaled so the values land in a healthy magnitude range.
fn tensor2d(name: &str, rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = weights(name, rows * cols).iter().map(|w| w * scale).collect();
    Tensor::new(data, vec![rows, cols])
}

/// The `[dim, dim]` identity, so a projection is `y = x` and `wo` is a pass-
/// through — lets the controlled-input tests reason about Q/K/V == x directly.
fn identity() -> Tensor {
    let mut d = vec![0.0f32; DIM * DIM];
    for i in 0..DIM {
        d[i * DIM + i] = 1.0;
    }
    Tensor::new(d, vec![DIM, DIM])
}

/// Columns of `[T,dim]` that head `hi` reads. Contiguous (the reference §5.2):
/// `hi*head_dim + d`. Interleaved (the wrong alternative C4 pins against):
/// `d*n_heads + hi`. Both are permutations of `0..dim`.
fn head_cols(hi: usize, interleaved: bool) -> Vec<usize> {
    (0..HEAD_DIM)
        .map(|d| if interleaved { d * N_HEADS + hi } else { hi * HEAD_DIM + d })
        .collect()
}

fn gather_head(flat: &[f32], cols: &[usize], t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * HEAD_DIM];
    for i in 0..t {
        for (d, &c) in cols.iter().enumerate() {
            out[i * HEAD_DIM + d] = flat[i * DIM + c];
        }
    }
    out
}

/// Independent transcription of `reference.py` §5.2, parameterised by `scale`
/// and `interleaved` so the tests can construct deliberately-wrong variants.
/// Bias-free Q/K/V via `syrinx_core::linear`; per-head RoPE on Q/K only (not V);
/// scores = `(q·k) * scale` for `j <= i` and `-inf` for `j > i` (the additive
/// causal mask, applied *after* the scale); softmax over keys; value-weighted
/// context scattered back to the head's columns; `wo` projection.
fn reference_attention(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    positions: &[usize],
    scale: f64,
    interleaved: bool,
) -> Tensor {
    let t = x.shape()[0];
    let q = linear(x, wq, None);
    let k = linear(x, wk, None);
    let v = linear(x, wv, None);
    let mut ctx = vec![0.0f32; t * DIM];
    for hi in 0..N_HEADS {
        let cols = head_cols(hi, interleaved);
        let qh = gather_head(q.data(), &cols, t);
        let kh = gather_head(k.data(), &cols, t);
        let vh = gather_head(v.data(), &cols, t);
        // RoPE applies to Q and K only — V is gathered raw.
        let qh = rope(&Tensor::new(qh, vec![t, 1, HEAD_DIM]), positions, THETA);
        let kh = rope(&Tensor::new(kh, vec![t, 1, HEAD_DIM]), positions, THETA);
        let qh = qh.data();
        let kh = kh.data();
        for i in 0..t {
            // scores over keys j; future keys (j > i) take the -inf mask.
            let mut scores = vec![f64::NEG_INFINITY; t];
            for j in 0..=i {
                let mut dot = 0.0f64;
                for d in 0..HEAD_DIM {
                    dot += qh[i * HEAD_DIM + d] as f64 * kh[j * HEAD_DIM + d] as f64;
                }
                scores[j] = dot * scale; // scale BEFORE the (implicit) mask add
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut s = 0.0f64;
            let mut e = vec![0.0f64; t];
            for j in 0..t {
                e[j] = (scores[j] - m).exp();
                s += e[j];
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f64;
                for j in 0..t {
                    acc += (e[j] / s) * vh[j * HEAD_DIM + d] as f64;
                }
                ctx[i * DIM + cols[d]] = acc as f32;
            }
        }
    }
    linear(&Tensor::new(ctx, vec![t, DIM]), wo, None)
}

// ----- C1: output shape + dense end-to-end parity ----------------------------

#[test]
fn test_output_shape_is_t_by_dim() {
    let t = 4;
    let x = tensor2d("c1.x", t, DIM, 50.0);
    let wq = tensor2d("c1.wq", DIM, DIM, 1.0);
    let wk = tensor2d("c1.wk", DIM, DIM, 1.0);
    let wv = tensor2d("c1.wv", DIM, DIM, 1.0);
    let wo = tensor2d("c1.wo", DIM, DIM, 1.0);
    let positions = [0usize, 1, 2, 3];

    let out = attention(&x, &wq, &wk, &wv, &wo, &positions);

    assert_eq!(out.shape(), &[t, DIM], "attention must return shape [T, dim]");
}

#[test]
fn test_attention_matches_reference_dense() {
    let t = 4;
    let x = tensor2d("c1.x", t, DIM, 50.0);
    let wq = tensor2d("c1.wq", DIM, DIM, 1.0);
    let wk = tensor2d("c1.wk", DIM, DIM, 1.0);
    let wv = tensor2d("c1.wv", DIM, DIM, 1.0);
    let wo = tensor2d("c1.wo", DIM, DIM, 1.0);
    let positions = [0usize, 1, 2, 3];

    let out = attention(&x, &wq, &wk, &wv, &wo, &positions);
    let reference =
        reference_attention(&x, &wq, &wk, &wv, &wo, &positions, correct_scale(), false);

    assert_eq!(out.shape(), &[t, DIM]);
    let d = max_abs_diff(out.data(), reference.data());
    assert!(
        d <= PARITY_TOL,
        "dense attention diverges from the §5.2 reference by {d} (tol {PARITY_TOL})"
    );
}

// ----- C2: causal independence of future positions ---------------------------

#[test]
fn test_causal_future_independence() {
    let t = 4;
    let x = tensor2d("c2.x", t, DIM, 50.0);
    let wq = tensor2d("c2.wq", DIM, DIM, 1.0);
    let wk = tensor2d("c2.wk", DIM, DIM, 1.0);
    let wv = tensor2d("c2.wv", DIM, DIM, 1.0);
    let wo = tensor2d("c2.wo", DIM, DIM, 1.0);
    let positions = [0usize, 1, 2, 3];

    let base = attention(&x, &wq, &wk, &wv, &wo, &positions);

    // For every position i with a future, hugely perturb all rows j > i and
    // recompute. Output row i must be unchanged (it depends only on keys/values
    // at positions <= i); without the causal mask a future key would leak in.
    for i in 0..t - 1 {
        let mut pd = x.data().to_vec();
        for j in (i + 1)..t {
            for c in 0..DIM {
                pd[j * DIM + c] += 100.0;
            }
        }
        let xp = Tensor::new(pd, vec![t, DIM]);
        let out = attention(&xp, &wq, &wk, &wv, &wo, &positions);

        let unchanged = max_abs_diff(row(&base, i), row(&out, i));
        assert!(
            unchanged <= CAUSAL_TOL,
            "row {i} changed by {unchanged} after perturbing future rows; \
             the causal mask is not being applied"
        );

        // Control: the perturbation must actually move a future output row, so
        // the unchanged-row assertion above is not vacuously true (and this is
        // exactly the signal a mask-removed implementation would also move
        // row i).
        let moved = max_abs_diff(row(&base, t - 1), row(&out, t - 1));
        assert!(
            moved > FAIL_GAP,
            "perturbing future rows must change the future output row {} (moved {moved})",
            t - 1
        );
    }
}

// ----- C3: score scaling is exactly 1/sqrt(head_dim), before the mask --------

#[test]
fn test_scale_is_inv_sqrt_head_dim() {
    // T=2, identity weights (Q=K=V=x, out=ctx). Query row 1 sees keys {0,1}.
    // x[0] = 0  -> score(1,0) = 0 (and v0 = 0). x[1] has head-0 norm^2 = 17, so
    // score(1,1) = 17 (RoPE preserves the per-head norm). The raw score gap of
    // 17 is large enough that the softmax — and thus the value-weighted output —
    // is sharply sensitive to the scale, distinguishing 1/sqrt(32) from both
    // wrong alternatives.
    let t = 2;
    let mut xd = vec![0.0f32; t * DIM];
    xd[DIM] = (17.0f32).sqrt(); // x[1][0]
    let x = Tensor::new(xd, vec![t, DIM]);
    let id = identity();
    let positions = [0usize, 1];

    let out = attention(&x, &id, &id, &id, &id, &positions);

    let correct =
        reference_attention(&x, &id, &id, &id, &id, &positions, correct_scale(), false);
    let wrong_head_dim =
        reference_attention(&x, &id, &id, &id, &id, &positions, 1.0 / HEAD_DIM as f64, false);
    let wrong_dim = reference_attention(
        &x,
        &id,
        &id,
        &id,
        &id,
        &positions,
        1.0 / (DIM as f64).sqrt(),
        false,
    );

    // The impl uses exactly 1/sqrt(head_dim) applied before the mask add.
    assert!(
        max_abs_diff(out.data(), correct.data()) <= PARITY_TOL,
        "attention does not use scale 1/sqrt(head_dim) before the mask add"
    );
    // 1/head_dim is the wrong scale and must change the output.
    assert!(
        max_abs_diff(out.data(), wrong_head_dim.data()) > FAIL_GAP,
        "a 1/head_dim scale must diverge from the impl"
    );
    // 1/sqrt(dim) is the wrong scale and must change the output.
    assert!(
        max_abs_diff(out.data(), wrong_dim.data()) > FAIL_GAP,
        "a 1/sqrt(dim) scale must diverge from the impl"
    );
    // Sanity: the wrong-scale references really do differ from the correct one
    // (so the divergence assertions above are meaningful, not a degenerate input).
    assert!(max_abs_diff(correct.data(), wrong_head_dim.data()) > FAIL_GAP);
    assert!(max_abs_diff(correct.data(), wrong_dim.data()) > FAIL_GAP);
}

// ----- C4: per-head split is contiguous and head-local -----------------------

#[test]
fn test_head_split_is_contiguous() {
    // T=2, positions=[0,0] so RoPE is the identity and ONLY the head split can
    // change the result. Identity weights -> Q=K=V=x, out=ctx. x[0] = 0, and
    // x[1] has equal mass c in columns 1 and 2. Under the contiguous split both
    // land in head 0 (norm^2 = 2c^2); under an interleaved (stride n_heads)
    // split they land in heads 1 and 2 (norm^2 = c^2 each). The differing
    // per-head norms drive different softmax weights, so the output differs.
    let t = 2;
    let c = (10.0f32).sqrt();
    let mut xd = vec![0.0f32; t * DIM];
    xd[DIM + 1] = c; // x[1][1]
    xd[DIM + 2] = c; // x[1][2]
    let x = Tensor::new(xd, vec![t, DIM]);
    let id = identity();
    let positions = [0usize, 0];

    let out = attention(&x, &id, &id, &id, &id, &positions);

    let contiguous =
        reference_attention(&x, &id, &id, &id, &id, &positions, correct_scale(), false);
    let interleaved =
        reference_attention(&x, &id, &id, &id, &id, &positions, correct_scale(), true);

    // The impl uses a contiguous head_dim split.
    assert!(
        max_abs_diff(out.data(), contiguous.data()) <= PARITY_TOL,
        "attention does not use a contiguous head_dim column split"
    );
    // An interleaved split reads different columns per head and must diverge.
    assert!(
        max_abs_diff(out.data(), interleaved.data()) > FAIL_GAP,
        "an interleaved head split must diverge from the impl"
    );
    // Sanity: contiguous and interleaved genuinely differ for this input.
    assert!(
        max_abs_diff(contiguous.data(), interleaved.data()) > FAIL_GAP,
        "contiguous and interleaved splits must differ for this input"
    );
}
