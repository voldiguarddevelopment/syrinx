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

// =====================================================================
// T-02.01b — the seven neural primitives the LM composes.
// Direct transcription of `reference.py` §4 (= REFERENCE.md / PARITY.md §4).
// All accumulation is in f64, rounded to f32 per output cell (within tol).
// =====================================================================

/// `linear(x[*, in], W[out, in], b[out]?) -> [*, out]` (PARITY.md §4.3).
///
/// `W` is row-major `[out, in]` (PyTorch `nn.Linear.weight`): row `o` holds output
/// neuron `o`'s weights. `y[i][o] = sum_k x[i][k] * W[o][k] (+ b[o])`. The leading
/// dims of `x` are preserved and the last dim becomes `out`.
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let in_dim = w.shape[1];
    let out_dim = w.shape[0];
    let rows = x.data.len() / in_dim;
    let mut data = vec![0.0f32; rows * out_dim];
    for i in 0..rows {
        for o in 0..out_dim {
            let mut sum = 0.0f64;
            for k in 0..in_dim {
                sum += x.data[i * in_dim + k] as f64 * w.data[o * in_dim + k] as f64;
            }
            if let Some(bias) = b {
                sum += bias.data[o] as f64;
            }
            data[i * out_dim + o] = sum as f32;
        }
    }
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = out_dim;
    Tensor::new(data, shape)
}

/// `rmsnorm(x[*, d], w[d], eps) -> [*, d]` (PARITY.md §4.4).
///
/// Per last-axis row: `ms = mean(x^2)`, `inv = 1/sqrt(ms + eps)` (eps INSIDE the
/// sqrt), `y[k] = x[k] * inv * w[k]`.
pub fn rmsnorm(x: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    let d = w.shape[0];
    let rows = x.data.len() / d;
    let mut data = vec![0.0f32; x.data.len()];
    for i in 0..rows {
        let mut ms = 0.0f64;
        for k in 0..d {
            let v = x.data[i * d + k] as f64;
            ms += v * v;
        }
        ms /= d as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for k in 0..d {
            data[i * d + k] = ((x.data[i * d + k] as f64 * inv) * w.data[k] as f64) as f32;
        }
    }
    Tensor::new(data, x.shape.clone())
}

/// `softmax(x) -> same shape`, over the last axis (PARITY.md §4.5).
///
/// Per last-axis row: subtract the row max for stability, exponentiate, normalize.
pub fn softmax(x: &Tensor) -> Tensor {
    let d = *x.shape.last().unwrap();
    let rows = x.data.len() / d;
    let mut data = vec![0.0f32; x.data.len()];
    for i in 0..rows {
        let row = &x.data[i * d..i * d + d];
        let mut m = row[0];
        for &v in row {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f64;
        let mut e = vec![0.0f64; d];
        for k in 0..d {
            let ek = ((row[k] - m) as f64).exp();
            e[k] = ek;
            s += ek;
        }
        for k in 0..d {
            data[i * d + k] = (e[k] / s) as f32;
        }
    }
    Tensor::new(data, x.shape.clone())
}

/// `silu(v) = v * sigmoid(v)`, elementwise, `sigmoid(v) = 1/(1 + exp(-v))`
/// (PARITY.md §4.6).
pub fn silu(x: &Tensor) -> Tensor {
    let data = x
        .data
        .iter()
        .map(|&v| {
            let sig = 1.0f64 / (1.0 + (-(v as f64)).exp());
            (v as f64 * sig) as f32
        })
        .collect();
    Tensor::new(data, x.shape.clone())
}

/// `rope(x[T, n_heads, head_dim], positions[T], theta) -> same shape`
/// (PARITY.md §4.7).
///
/// Interleaved pairing: dims `(2i, 2i+1)` rotate together with
/// `inv_freq[i] = theta^(-(2i)/head_dim)` and `angle = pos * inv_freq[i]`. At
/// `pos == 0` (cos=1, sin=0) the rotation is the identity.
pub fn rope(x: &Tensor, positions: &[usize], theta: f32) -> Tensor {
    let t_dim = x.shape[0];
    let n_heads = x.shape[1];
    let head_dim = x.shape[2];
    let half = head_dim / 2;
    let mut data = x.data.clone();
    for (t, &position) in positions.iter().enumerate().take(t_dim) {
        let pos = position as f64;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let inv_freq = (theta as f64).powf(-((2 * i) as f64) / head_dim as f64);
                let angle = pos * inv_freq;
                let c = angle.cos();
                let s = angle.sin();
                let a = x.data[base + 2 * i] as f64;
                let bb = x.data[base + 2 * i + 1] as f64;
                data[base + 2 * i] = (a * c - bb * s) as f32;
                data[base + 2 * i + 1] = (a * s + bb * c) as f32;
            }
        }
    }
    Tensor::new(data, x.shape.clone())
}

/// `embed(ids, table[V, d]) -> [ids.len(), d]` (PARITY.md §4.8): row `i` is a copy
/// of `table[ids[i]]`.
pub fn embed(ids: &[usize], table: &Tensor) -> Tensor {
    let d = table.shape[1];
    let mut data = vec![0.0f32; ids.len() * d];
    for (i, &id) in ids.iter().enumerate() {
        for c in 0..d {
            data[i * d + c] = table.data[id * d + c];
        }
    }
    Tensor::new(data, vec![ids.len(), d])
}

/// `causal_mask(T) -> [T, T]` additive mask (PARITY.md §4.9):
/// `mask[i][j] = 0.0` when `j <= i`, else `-inf`.
pub fn causal_mask(t: usize) -> Tensor {
    let mut data = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            data[i * t + j] = if j <= i { 0.0 } else { f32::NEG_INFINITY };
        }
    }
    Tensor::new(data, vec![t, t])
}
