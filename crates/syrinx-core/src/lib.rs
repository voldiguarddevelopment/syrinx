//! syrinx-core — tensor-ops glue, weight loading, device mgmt.
//!
//! T-02.01a: the host-memory f32 reference arithmetic floor every LM op stands
//! on — the `Tensor { data, shape }` contract plus `matmul`/`add`/`mul` and the
//! typed `ShapeError` they return on a dimension disagreement. Single host f32
//! path only: no SIMD, no BLAS, no device/quantization concerns. matmul sums
//! over the shared inner dim row-major (`reference.py` §4.1); add/mul are
//! elementwise.

/// A dense, row-major f32 tensor: a flat `data` buffer plus its declared `shape`.
///
/// The contract `data.len() == prod(shape)` holds for any tensor built from
/// consistent inputs and is preserved by every op in this module.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Build a tensor from flat row-major `data` and a `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// The declared dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The flat row-major contents.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

/// A typed dimension disagreement, returned instead of panicking or truncating.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeError {
    /// `matmul` inner dims disagree: `A[m, k]` × `B[p, n]` with `k != p`.
    MatmulInner { k: usize, p: usize },
    /// elementwise `add`/`mul` on tensors of unequal shape.
    ElementwiseMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
}

/// Row-major matrix multiply: `A[m, k]` × `B[p, n]` → `[m, n]`, summing over the
/// shared inner dim. Returns `ShapeError::MatmulInner` when `k != p`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, ShapeError> {
    let m = a.shape[0];
    let k = a.shape[1];
    let p = b.shape[0];
    let n = b.shape[1];
    if k != p {
        return Err(ShapeError::MatmulInner { k, p });
    }
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for x in 0..k {
                sum += a.data[i * k + x] * b.data[x * n + j];
            }
            data[i * n + j] = sum;
        }
    }
    Ok(Tensor::new(data, vec![m, n]))
}

/// Elementwise sum of two equal-shaped tensors. Returns
/// `ShapeError::ElementwiseMismatch` when the shapes disagree.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, ShapeError> {
    if a.shape != b.shape {
        return Err(ShapeError::ElementwiseMismatch {
            lhs: a.shape.clone(),
            rhs: b.shape.clone(),
        });
    }
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();
    Ok(Tensor::new(data, a.shape.clone()))
}

/// Elementwise product of two equal-shaped tensors. Returns
/// `ShapeError::ElementwiseMismatch` when the shapes disagree.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, ShapeError> {
    if a.shape != b.shape {
        return Err(ShapeError::ElementwiseMismatch {
            lhs: a.shape.clone(),
            rhs: b.shape.clone(),
        });
    }
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect();
    Ok(Tensor::new(data, a.shape.clone()))
}
