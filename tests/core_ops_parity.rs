//! T-02.01b — the seven neural primitives the LM composes, frozen RED tests.
//!
//! These pin every op in `reference.py` §4 that `core_arith_parity.rs` did not
//! already cover (matmul/add/mul live there). Each criterion:
//!   * C1 — `linear` (with bias), `rmsnorm` (eps 1e-5 inside the sqrt) and
//!     `softmax` (last-axis, max-subtracted) match their goldens to <= 1e-4
//!     max-abs, and a one-element corruption of any of the three fails its case.
//!   * C2 — `silu` (`v*sigmoid(v)`), `rope` (interleaved `(2i,2i+1)` pairs,
//!     theta 10000, positions `[0,1]`) and `embed` (row copy by id) match their
//!     goldens to <= 1e-4 with the produced shape equal to the golden `shape`.
//!   * C3 — `causal_mask(3)` is `0.0` for `j <= i` and `-inf` for `j > i` at
//!     every `(i,j)`, matching `causal_mask.json` (where `-inf` is the string
//!     `"-inf"`); the exhaustive cell check rejects a flipped inequality.
//!   * C4 — golden-free properties: softmax rows sum to `1 ± 1e-6` and stay
//!     non-negative (and finite for a large-magnitude row, pinning the
//!     max-subtract); rmsnorm RMS over the last axis is `1 ± 1e-3` with all-ones
//!     `w` (and an all-zeros row stays finite, pinning eps INSIDE the sqrt);
//!     rope preserves each rotated pair's L2 norm to `± 1e-5` and is the identity
//!     at `pos == 0`; `silu(0) == 0` and silu is strictly increasing across the
//!     pinned positive sample points.
//!
//! RED: `syrinx-core` exposes none of `linear`/`rmsnorm`/`softmax`/`silu`/`rope`/
//! `embed`/`causal_mask`, so this target fails to build. GREEN adds them as a
//! direct transcription of `reference.py` §4. This file is frozen at red-pass;
//! do not edit it in GREEN.

use syrinx_core::{causal_mask, embed, linear, rmsnorm, rope, silu, softmax, Tensor};

// ----- golden plumbing --------------------------------------------------------

fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/golden/parity/{}", env!("CARGO_MANIFEST_DIR"), name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Flatten a nested `[[..], [..]]` JSON matrix into row-major data + `[rows, cols]`.
fn flatten_2d(v: &serde_json::Value) -> (Vec<f32>, Vec<usize>) {
    let rows = v.as_array().expect("matrix is an array of rows");
    let nrows = rows.len();
    let ncols = rows[0].as_array().expect("row is an array").len();
    let mut data = Vec::with_capacity(nrows * ncols);
    for row in rows {
        for x in row.as_array().expect("row is an array") {
            data.push(x.as_f64().expect("cell is a number") as f32);
        }
    }
    (data, vec![nrows, ncols])
}

/// Recursively flatten any nested JSON number array into row-major `f32`.
fn flatten_leaves(v: &serde_json::Value) -> Vec<f32> {
    match v {
        serde_json::Value::Array(a) => a.iter().flat_map(flatten_leaves).collect(),
        serde_json::Value::Number(n) => vec![n.as_f64().expect("number") as f32],
        other => panic!("unexpected non-number leaf: {other:?}"),
    }
}

/// A flat JSON number array → `Vec<f32>` (the golden `data` field).
fn flat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("data is an array")
        .iter()
        .map(|x| x.as_f64().expect("data cell is a number") as f32)
        .collect()
}

/// A JSON integer array → `Vec<usize>` (`shape`, `positions`, `ids`).
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
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// =====================================================================
// C1 — linear / rmsnorm / softmax golden parity + corruption sensitivity
// =====================================================================

#[test]
fn test_linear_parity() {
    let g = load("linear.json");
    let (x_data, x_shape) = flatten_2d(&g["input"]["x"]);
    let (w_data, w_shape) = flatten_2d(&g["input"]["W"]); // [out, in]
    let b_data = flat(&g["input"]["b"]);
    let x = Tensor::new(x_data, x_shape);
    let w = Tensor::new(w_data, w_shape.clone());
    let b = Tensor::new(b_data, vec![w_shape[0]]);

    let out = linear(&x, &w, Some(&b));

    // Shape equals the golden `shape` [*, out] = [2, 4].
    assert_eq!(out.shape(), ints(&g["shape"]).as_slice());
    assert_eq!(out.data().len(), out.shape().iter().product::<usize>());

    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "linear max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_rmsnorm_parity() {
    let g = load("rmsnorm.json");
    let (x_data, x_shape) = flatten_2d(&g["input"]["x"]);
    let w_data = flat(&g["input"]["w"]);
    let eps = g["input"]["eps"].as_f64().expect("eps is a number") as f32;
    let x = Tensor::new(x_data, x_shape.clone());
    let w = Tensor::new(w_data, vec![x_shape[1]]);

    let out = rmsnorm(&x, &w, eps);

    assert_eq!(out.shape(), ints(&g["shape"]).as_slice());
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "rmsnorm max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_softmax_parity() {
    let g = load("softmax.json");
    let (x_data, x_shape) = flatten_2d(&g["input"]["x"]);
    let x = Tensor::new(x_data, x_shape);

    let out = softmax(&x);

    assert_eq!(out.shape(), ints(&g["shape"]).as_slice());
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "softmax max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_c1_goldens_reject_one_element_corruption() {
    // linear
    {
        let g = load("linear.json");
        let (x_data, x_shape) = flatten_2d(&g["input"]["x"]);
        let (w_data, w_shape) = flatten_2d(&g["input"]["W"]);
        let b = Tensor::new(flat(&g["input"]["b"]), vec![w_shape[0]]);
        let out = linear(
            &Tensor::new(x_data, x_shape),
            &Tensor::new(w_data, w_shape),
            Some(&b),
        );
        let t = tol(&g);
        let mut golden = flat(&g["data"]);
        assert!(max_abs_diff(out.data(), &golden) <= t, "linear truth within tol");
        golden[0] += 1.0;
        let d = max_abs_diff(out.data(), &golden);
        assert!(d > t, "one-element linear corruption (diff {d}) should exceed tol {t}");
    }
    // rmsnorm
    {
        let g = load("rmsnorm.json");
        let (x_data, x_shape) = flatten_2d(&g["input"]["x"]);
        let w = Tensor::new(flat(&g["input"]["w"]), vec![x_shape[1]]);
        let eps = g["input"]["eps"].as_f64().unwrap() as f32;
        let out = rmsnorm(&Tensor::new(x_data, x_shape), &w, eps);
        let t = tol(&g);
        let mut golden = flat(&g["data"]);
        assert!(max_abs_diff(out.data(), &golden) <= t, "rmsnorm truth within tol");
        golden[0] += 1.0;
        let d = max_abs_diff(out.data(), &golden);
        assert!(d > t, "one-element rmsnorm corruption (diff {d}) should exceed tol {t}");
    }
    // softmax
    {
        let g = load("softmax.json");
        let (x_data, x_shape) = flatten_2d(&g["input"]["x"]);
        let out = softmax(&Tensor::new(x_data, x_shape));
        let t = tol(&g);
        let mut golden = flat(&g["data"]);
        assert!(max_abs_diff(out.data(), &golden) <= t, "softmax truth within tol");
        golden[0] += 1.0;
        let d = max_abs_diff(out.data(), &golden);
        assert!(d > t, "one-element softmax corruption (diff {d}) should exceed tol {t}");
    }
}

// =====================================================================
// C2 — silu / rope / embed golden parity + exact output shape
// =====================================================================

#[test]
fn test_silu_parity() {
    let g = load("silu.json");
    let (x_data, x_shape) = flatten_2d(&g["input"]["x"]);
    let x = Tensor::new(x_data, x_shape);

    let out = silu(&x);

    assert_eq!(out.shape(), ints(&g["shape"]).as_slice(), "silu shape == golden [1,5]");
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "silu max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_rope_parity() {
    let g = load("rope.json");
    // x is [T, n_heads, head_dim] = [2, 1, 4]; derive the shape from the nesting.
    let xv = &g["input"]["x"];
    let t_dim = xv.as_array().unwrap().len();
    let h_dim = xv[0].as_array().unwrap().len();
    let hd = xv[0][0].as_array().unwrap().len();
    let x = Tensor::new(flatten_leaves(xv), vec![t_dim, h_dim, hd]);
    let positions = ints(&g["input"]["positions"]);
    let theta = g["input"]["theta"].as_f64().unwrap() as f32;

    let out = rope(&x, &positions, theta);

    assert_eq!(out.shape(), ints(&g["shape"]).as_slice(), "rope shape == golden [2,1,4]");
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "rope max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_embed_parity() {
    let g = load("embed.json");
    let ids = ints(&g["input"]["ids"]);
    let (tab_data, tab_shape) = flatten_2d(&g["input"]["table"]); // [V, d]
    let table = Tensor::new(tab_data, tab_shape);

    let out = embed(&ids, &table);

    assert_eq!(out.shape(), ints(&g["shape"]).as_slice(), "embed shape == golden [4,3]");
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "embed max-abs diff {d} exceeds tol {t}");
}

// =====================================================================
// C3 — causal_mask exact 0.0 / -inf pattern (rejects a flipped inequality)
// =====================================================================

#[test]
fn test_causal_mask_pattern() {
    let g = load("causal_mask.json");
    let t = g["input"]["T"].as_u64().unwrap() as usize;
    assert_eq!(t, 3, "golden pins T=3");

    let out = causal_mask(t);
    assert_eq!(out.shape(), ints(&g["shape"]).as_slice(), "mask shape == golden [3,3]");

    // Parse the golden grid, decoding the string "-inf" to NEG_INFINITY.
    let rows = g["data"].as_array().expect("data is a grid");
    for (i, row) in rows.iter().enumerate() {
        let cells = row.as_array().expect("row is an array");
        for (j, cell) in cells.iter().enumerate() {
            let got = out.data()[i * t + j];
            let expect_neg_inf = cell.as_str() == Some("-inf");
            if expect_neg_inf {
                // j > i must be additive -inf.
                assert!(j > i, "golden -inf only above the diagonal at ({i},{j})");
                assert!(
                    got.is_infinite() && got.is_sign_negative(),
                    "mask[{i}][{j}] must be -inf (j > i), got {got}"
                );
            } else {
                // j <= i (incl. the diagonal j == i) must be exactly 0.0.
                assert!(j <= i, "golden 0.0 only on/below the diagonal at ({i},{j})");
                let want = cell.as_f64().expect("numeric 0.0 cell") as f32;
                assert_eq!(want, 0.0, "golden on/below diagonal is 0.0");
                assert_eq!(got, 0.0, "mask[{i}][{j}] must be 0.0 (j <= i), got {got}");
            }
        }
    }

    // Spell out the cells that distinguish `j <= i` from a flipped `j < i`:
    // the diagonal must be 0.0 (a `j < i` mask would make it -inf), and the
    // strictly-lower entry must be 0.0 while the strictly-upper one is -inf.
    assert_eq!(out.data()[0 * t + 0], 0.0, "diagonal (0,0) is 0.0");
    assert_eq!(out.data()[1 * t + 1], 0.0, "diagonal (1,1) is 0.0");
    assert_eq!(out.data()[1 * t + 0], 0.0, "below-diagonal (1,0) is 0.0");
    assert!(
        out.data()[0 * t + 1].is_infinite() && out.data()[0 * t + 1].is_sign_negative(),
        "above-diagonal (0,1) is -inf"
    );
}

// =====================================================================
// C4 — golden-free algebraic properties
// =====================================================================

#[test]
fn test_softmax_rows_sum_to_one_and_nonneg() {
    // Two ordinary rows plus one large-magnitude row. The big value makes the
    // max-subtract load-bearing: without it exp() overflows to inf and the row
    // becomes NaN, so this row pins `exp(x - max)` against `exp(x + max)`.
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, -1.0, 0.0, 0.5, 1000.0, 1.0, 2.0],
        vec![3, 3],
    );
    let out = softmax(&x);
    let d = out.data();
    for r in 0..3 {
        let row = &d[r * 3..r * 3 + 3];
        for &v in row {
            assert!(v.is_finite(), "softmax entry must be finite, got {v}");
            assert!(v >= 0.0, "softmax entry must be >= 0, got {v}");
        }
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() <= 1e-6, "row {r} sums to {s}, expected 1.0 ± 1e-6");
    }
}

#[test]
fn test_rmsnorm_unit_rms_and_zero_row_finite() {
    let eps = 1e-5f32;
    let w = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);

    // All-ones weight => the output RMS over the last axis is 1.0 ± 1e-3.
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0], vec![2, 4]);
    let out = rmsnorm(&x, &w, eps);
    let d = out.data();
    for r in 0..2 {
        let row = &d[r * 4..r * 4 + 4];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rms = ms.sqrt();
        assert!((rms - 1.0).abs() <= 1e-3, "row {r} RMS {rms} not within 1e-3 of 1.0");
    }

    // An all-zeros row must stay finite: with eps INSIDE the sqrt the scale is
    // 1/sqrt(0 + eps); flipping to 1/sqrt(0 - eps) yields sqrt of a negative =>
    // NaN propagates into the output. So this pins eps placement.
    let z = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![1, 4]);
    let zout = rmsnorm(&z, &w, eps);
    for &v in zout.data() {
        assert!(v.is_finite(), "rmsnorm of a zero row must be finite, got {v}");
    }
}

#[test]
fn test_rope_preserves_pair_norm_and_identity_at_pos_zero() {
    let theta = 10000.0f32;

    // Norm preservation: a single token at a non-zero position; each rotated
    // pair keeps its L2 norm to ± 1e-5. x = [[[3, 4, 1, 2]]], shape [1,1,4].
    let x = Tensor::new(vec![3.0, 4.0, 1.0, 2.0], vec![1, 1, 4]);
    let out = rope(&x, &[2], theta);
    let d = out.data();
    let pair0_in = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt();
    let pair1_in = (1.0f32 * 1.0 + 2.0 * 2.0).sqrt();
    let pair0_out = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let pair1_out = (d[2] * d[2] + d[3] * d[3]).sqrt();
    assert!((pair0_out - pair0_in).abs() <= 1e-5, "pair0 norm changed: {pair0_in} -> {pair0_out}");
    assert!((pair1_out - pair1_in).abs() <= 1e-5, "pair1 norm changed: {pair1_in} -> {pair1_out}");

    // Identity at pos == 0: cos(0)=1, sin(0)=0, so the output equals the input.
    let x0 = Tensor::new(vec![0.3, -0.7, 0.5, -0.5], vec![1, 1, 4]);
    let out0 = rope(&x0, &[0], theta);
    let diff = max_abs_diff(out0.data(), x0.data());
    assert!(diff <= 1e-6, "rope at pos 0 must be the identity, max-abs diff {diff}");
}

#[test]
fn test_silu_zero_and_monotone_on_positive_samples() {
    // silu(0) = 0 * sigmoid(0) = 0 exactly.
    let zero = silu(&Tensor::new(vec![0.0], vec![1, 1]));
    assert_eq!(zero.data()[0], 0.0, "silu(0.0) must equal 0.0");

    // silu is strictly increasing across these pinned positive samples.
    let xs = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![1, 5]);
    let ys = silu(&xs);
    let d = ys.data();
    for k in 1..d.len() {
        assert!(
            d[k] > d[k - 1],
            "silu must be increasing: silu(sample {k})={} !> silu(sample {})={}",
            d[k],
            k - 1,
            d[k - 1]
        );
    }
}
