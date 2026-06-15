//! T-02.01a — core tensors and arithmetic, frozen RED tests.
//!
//! These pin the arithmetic floor every other LM op stands on:
//!   * C1 — the `Tensor { data, shape }` contract (declared dims; `data.len()`
//!     equals the product of `shape`).
//!   * C2 — `matmul` numerical parity against `tests/golden/parity/matmul.json`
//!     (max-abs <= tol) and an exact `[A.rows, B.cols]` output shape.
//!   * C3 — `add`/`mul` elementwise parity against `add.json`/`mul.json`, with a
//!     one-element golden corruption proving the comparison is sensitive.
//!   * C4 — inner-dim mismatch in `matmul` and shape disagreement in `add`/`mul`
//!     return a typed `ShapeError` naming the two mismatched dims (never a panic
//!     or a truncated result), while the matching-shape call on the same path is
//!     `Ok`.
//!
//! RED: `syrinx-core` exposes none of these items, so this target fails to build.
//! Do not implement here — GREEN adds `Tensor`/`matmul`/`add`/`mul`/`ShapeError`
//! to `syrinx-core`. This file is frozen at red-pass; do not edit it in GREEN.

use syrinx_core::{add, matmul, mul, ShapeError, Tensor};

// ----- golden plumbing --------------------------------------------------------

fn load(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/golden/parity/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
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

/// A flat JSON number array → `Vec<f32>` (the golden `data` field).
fn flat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("data is an array")
        .iter()
        .map(|x| x.as_f64().expect("data cell is a number") as f32)
        .collect()
}

/// A JSON integer array → `Vec<usize>` (the golden `shape` field).
fn dims(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("shape is an array")
        .iter()
        .map(|x| x.as_u64().expect("shape cell is an integer") as usize)
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

// ----- C1: the Tensor contract -----------------------------------------------

#[test]
fn test_tensor_shape_and_data_contract() {
    // Constructor accepts flat row-major data and a shape.
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::new(data.clone(), vec![2, 3]);

    // `shape()` returns the declared dims exactly.
    assert_eq!(t.shape(), &[2usize, 3usize]);

    // `data()` preserves the flat row-major contents in order.
    assert_eq!(t.data(), data.as_slice());

    // `data().len()` equals the product of `shape`.
    let prod: usize = t.shape().iter().product();
    assert_eq!(prod, 6);
    assert_eq!(t.data().len(), prod);

    // A second, differently-shaped tensor pins that `shape()` echoes *its* dims,
    // not a constant.
    let v = Tensor::new(vec![7.0, 8.0, 9.0, 10.0], vec![4]);
    assert_eq!(v.shape(), &[4usize]);
    assert_eq!(v.data().len(), v.shape().iter().product::<usize>());
}

// ----- C2: matmul parity + output shape --------------------------------------

#[test]
fn test_matmul_parity() {
    let g = load("matmul.json");
    let (a_data, a_shape) = flatten_2d(&g["input"]["A"]);
    let (b_data, b_shape) = flatten_2d(&g["input"]["B"]);
    let a = Tensor::new(a_data, a_shape.clone());
    let b = Tensor::new(b_data, b_shape.clone());

    let out = matmul(&a, &b).expect("matmul on matching inner dims is Ok");

    // Output shape is exactly [m, n] = [A.rows, B.cols] ...
    let expected_shape = vec![a_shape[0], b_shape[1]];
    assert_eq!(out.shape(), expected_shape.as_slice());
    // ... and equals the golden `shape` exactly.
    assert_eq!(out.shape(), dims(&g["shape"]).as_slice());

    // data.len() == prod(shape) invariant holds on the produced tensor.
    assert_eq!(out.data().len(), out.shape().iter().product::<usize>());

    // Max-abs elementwise difference to the golden `data` is within tol (1e-4).
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "matmul max-abs diff {d} exceeds tol {t}");
}

// ----- C3: add/mul parity + corruption sensitivity ---------------------------

#[test]
fn test_add_parity() {
    let g = load("add.json");
    let (a_data, a_shape) = flatten_2d(&g["input"]["A"]);
    let (b_data, b_shape) = flatten_2d(&g["input"]["B"]);
    let a = Tensor::new(a_data, a_shape);
    let b = Tensor::new(b_data, b_shape);

    let out = add(&a, &b).expect("add on equal shapes is Ok");

    assert_eq!(out.shape(), dims(&g["shape"]).as_slice());
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "add max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_mul_parity() {
    let g = load("mul.json");
    let (a_data, a_shape) = flatten_2d(&g["input"]["A"]);
    let (b_data, b_shape) = flatten_2d(&g["input"]["B"]);
    let a = Tensor::new(a_data, a_shape);
    let b = Tensor::new(b_data, b_shape);

    let out = mul(&a, &b).expect("mul on equal shapes is Ok");

    assert_eq!(out.shape(), dims(&g["shape"]).as_slice());
    let t = tol(&g);
    let d = max_abs_diff(out.data(), &flat(&g["data"]));
    assert!(d <= t, "mul max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_add_corruption_fails() {
    let g = load("add.json");
    let (a_data, a_shape) = flatten_2d(&g["input"]["A"]);
    let (b_data, b_shape) = flatten_2d(&g["input"]["B"]);
    let out = add(
        &Tensor::new(a_data, a_shape),
        &Tensor::new(b_data, b_shape),
    )
    .expect("add on equal shapes is Ok");

    // Sanity: the true golden is within tol.
    let t = tol(&g);
    let mut golden = flat(&g["data"]);
    assert!(max_abs_diff(out.data(), &golden) <= t);

    // A one-element corruption of the golden pushes the case past tol.
    golden[0] += 1.0;
    let d = max_abs_diff(out.data(), &golden);
    assert!(d > t, "one-element add corruption (diff {d}) should exceed tol {t}");
}

#[test]
fn test_mul_corruption_fails() {
    let g = load("mul.json");
    let (a_data, a_shape) = flatten_2d(&g["input"]["A"]);
    let (b_data, b_shape) = flatten_2d(&g["input"]["B"]);
    let out = mul(
        &Tensor::new(a_data, a_shape),
        &Tensor::new(b_data, b_shape),
    )
    .expect("mul on equal shapes is Ok");

    let t = tol(&g);
    let mut golden = flat(&g["data"]);
    assert!(max_abs_diff(out.data(), &golden) <= t);

    golden[0] += 1.0;
    let d = max_abs_diff(out.data(), &golden);
    assert!(d > t, "one-element mul corruption (diff {d}) should exceed tol {t}");
}

// ----- C4: typed ShapeError on mismatch, Ok on the matching path -------------

#[test]
fn test_matmul_inner_dim_mismatch() {
    // A[m=2, k=3] x B[p=4, n=5] — inner dims disagree (k=3 != p=4).
    let a = Tensor::new(vec![0.0; 6], vec![2, 3]);
    let b = Tensor::new(vec![0.0; 20], vec![4, 5]);
    match matmul(&a, &b) {
        Err(ShapeError::MatmulInner { k, p }) => {
            assert_eq!(k, 3, "error must name A's inner dim k");
            assert_eq!(p, 4, "error must name B's inner dim p");
        }
        other => panic!("expected Err(MatmulInner {{ k: 3, p: 4 }}), got {other:?}"),
    }

    // The matching-shape call on the same path is Ok with shape [m, n] = [2, 4].
    let a_ok = Tensor::new(vec![0.0; 6], vec![2, 3]);
    let b_ok = Tensor::new(vec![0.0; 12], vec![3, 4]);
    let out = matmul(&a_ok, &b_ok).expect("matmul on matching inner dims is Ok");
    assert_eq!(out.shape(), &[2usize, 4usize]);
}

#[test]
fn test_add_shape_mismatch() {
    // [3,3] vs [2,2] — unequal shapes.
    let a = Tensor::new(vec![0.0; 9], vec![3, 3]);
    let b = Tensor::new(vec![0.0; 4], vec![2, 2]);
    match add(&a, &b) {
        Err(ShapeError::ElementwiseMismatch { lhs, rhs }) => {
            assert_eq!(lhs, vec![3, 3], "error must name the lhs shape");
            assert_eq!(rhs, vec![2, 2], "error must name the rhs shape");
        }
        other => panic!("expected Err(ElementwiseMismatch), got {other:?}"),
    }

    // The matching-shape call on the same path is Ok.
    let out = add(
        &Tensor::new(vec![1.0; 4], vec![2, 2]),
        &Tensor::new(vec![2.0; 4], vec![2, 2]),
    )
    .expect("add on equal shapes is Ok");
    assert_eq!(out.shape(), &[2usize, 2usize]);
}

#[test]
fn test_mul_shape_mismatch() {
    let a = Tensor::new(vec![0.0; 9], vec![3, 3]);
    let b = Tensor::new(vec![0.0; 4], vec![2, 2]);
    match mul(&a, &b) {
        Err(ShapeError::ElementwiseMismatch { lhs, rhs }) => {
            assert_eq!(lhs, vec![3, 3], "error must name the lhs shape");
            assert_eq!(rhs, vec![2, 2], "error must name the rhs shape");
        }
        other => panic!("expected Err(ElementwiseMismatch), got {other:?}"),
    }

    let out = mul(
        &Tensor::new(vec![1.0; 4], vec![2, 2]),
        &Tensor::new(vec![2.0; 4], vec![2, 2]),
    )
    .expect("mul on equal shapes is Ok");
    assert_eq!(out.shape(), &[2usize, 2usize]);
}
