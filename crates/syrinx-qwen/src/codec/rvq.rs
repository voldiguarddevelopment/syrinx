//! The residual vector quantiser of the Qwen3-TTS 12 Hz tokenizer.
//!
//! The codec is a **Mimi**-family RVQ (`MimiEuclideanCodebook` in the reference), split
//! into a 1-layer *semantic* quantiser (`rvq_first`) and a 15-layer *acoustic* one
//! (`rvq_rest`) — 16 groups total, matching the talker's `num_code_groups`.
//!
//! ## The codebook is stored as an EMA accumulator, not as centroids
//!
//! Each layer ships `embedding_sum [size, dim]` and `cluster_usage [size]`. The usable
//! codebook is
//!
//! ```text
//!   embed = embedding_sum / cluster_usage.clamp(min = 1e-5)[:, None]
//! ```
//!
//! taken verbatim from the reference's `MimiEuclideanCodebook.embed` property. This is
//! not cosmetic: `cluster_usage` averages ~0.15 and ranges 0.02..0.58 across rows, so
//! using `embedding_sum` directly scales every centroid by a *different* factor of
//! roughly 2x-50x. The result would still decode to audio-shaped output, which is
//! exactly why it needs stating rather than discovering by ear.

use candle_core::{DType, Result, Tensor, D};

use crate::nn::Weights;

/// Reference epsilon from `MimiEuclideanCodebook.__init__`.
const CLUSTER_USAGE_EPS: f64 = 1e-5;

/// One RVQ stack (`rvq_first` = semantic, `rvq_rest` = acoustic).
pub struct Rvq {
    prefix: String,
    /// Reconstructed centroids per layer, `[codebook_size, codebook_dim]`.
    codebooks: Vec<Tensor>,
}

impl Rvq {
    /// Reconstruct every layer's codebook from its EMA buffers.
    pub fn load(w: &Weights, prefix: &str, n_layers: usize) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("{prefix}.vq.layers.{l}._codebook");
            let sum = w.g(&format!("{p}.embedding_sum"))?.to_dtype(DType::F32)?;
            let usage = w.g(&format!("{p}.cluster_usage"))?.to_dtype(DType::F32)?;
            let usage = usage.clamp(CLUSTER_USAGE_EPS, f64::INFINITY)?.unsqueeze(1)?;
            codebooks.push(sum.broadcast_div(&usage)?);
        }
        Ok(Self { prefix: prefix.to_string(), codebooks })
    }

    pub fn n_layers(&self) -> usize {
        self.codebooks.len()
    }

    /// The reconstructed centroids for layer `l` (for tests / inspection).
    pub fn codebook(&self, l: usize) -> Option<&Tensor> {
        self.codebooks.get(l)
    }

    /// Sum the per-layer centroids selected by `codes` (`[n_layers][t]`), then apply the
    /// stack's `output_proj`.
    ///
    /// RVQ decoding is additive: layer 0 gives a coarse vector and each later layer adds
    /// its residual, so the latent is the sum over layers of `codebook[l][codes[l][t]]`.
    pub fn decode(&self, w: &Weights, codes: &[Vec<u32>], dt: DType) -> Result<Tensor> {
        if codes.len() != self.codebooks.len() {
            return Err(candle_core::Error::Msg(format!(
                "{}: got {} code rows, expected {}",
                self.prefix,
                codes.len(),
                self.codebooks.len()
            )));
        }
        let t = codes.first().map(|r| r.len()).unwrap_or(0);
        let mut acc: Option<Tensor> = None;
        for (l, row) in codes.iter().enumerate() {
            if row.len() != t {
                return Err(candle_core::Error::Msg(format!(
                    "{}: layer {l} has {} codes, expected {t}",
                    self.prefix,
                    row.len()
                )));
            }
            let cb = &self.codebooks[l];
            let size = cb.dim(0)?;
            if let Some(&bad) = row.iter().find(|&&c| (c as usize) >= size) {
                return Err(candle_core::Error::Msg(format!(
                    "{}: layer {l} code {bad} out of range (codebook size {size})",
                    self.prefix
                )));
            }
            let idx = Tensor::from_vec(row.clone(), (t,), cb.device())?;
            let sel = cb.index_select(&idx, 0)?; // [t, dim]
            acc = Some(match acc {
                None => sel,
                Some(a) => (a + sel)?,
            });
        }
        let z = acc.ok_or_else(|| candle_core::Error::Msg("empty RVQ".into()))?;
        let z = z.to_dtype(dt)?.unsqueeze(0)?; // [1, t, dim]
        // `output_proj` is a kernel-1 conv: [out, in, 1] -> a matmul over channels.
        let ow = w.g(&format!("{}.output_proj.weight", self.prefix))?;
        let ow = ow.squeeze(D::Minus1)?; // [out, in]
        let y = z.broadcast_matmul(&ow.t()?)?; // [1, t, out]
        y.transpose(1, 2)?.contiguous() // [1, out, t]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    /// A 2-layer RVQ with hand-chosen EMA buffers, so the expected latent is exact.
    fn fixture() -> (Weights, Rvq) {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // layer 0: embedding_sum rows [[2,4],[6,8]], usage [2,4]
        //          -> embed [[1,2],[1.5,2]]
        map.insert(
            "q.vq.layers.0._codebook.embedding_sum".into(),
            Tensor::from_vec(vec![2f32, 4., 6., 8.], (2, 2), &dev).unwrap(),
        );
        map.insert(
            "q.vq.layers.0._codebook.cluster_usage".into(),
            Tensor::from_vec(vec![2f32, 4.], (2,), &dev).unwrap(),
        );
        // layer 1: embedding_sum [[1,1],[10,10]], usage [1,5] -> embed [[1,1],[2,2]]
        map.insert(
            "q.vq.layers.1._codebook.embedding_sum".into(),
            Tensor::from_vec(vec![1f32, 1., 10., 10.], (2, 2), &dev).unwrap(),
        );
        map.insert(
            "q.vq.layers.1._codebook.cluster_usage".into(),
            Tensor::from_vec(vec![1f32, 5.], (2,), &dev).unwrap(),
        );
        // identity output_proj [2,2,1]
        map.insert(
            "q.output_proj.weight".into(),
            Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2, 1), &dev).unwrap(),
        );
        let w = Weights { map, dev, dt: DType::F32 };
        let rvq = Rvq::load(&w, "q", 2).unwrap();
        (w, rvq)
    }

    /// The codebook must be `embedding_sum / cluster_usage`, NOT `embedding_sum`.
    #[test]
    fn codebook_is_the_ema_quotient() {
        let (_w, rvq) = fixture();
        let cb: Vec<f32> = rvq.codebook(0).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(cb, vec![1.0, 2.0, 1.5, 2.0]);
        // the raw accumulator would be [2,4,6,8] — a different vector per row
        assert_ne!(cb, vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// Decoding sums the selected centroid across layers.
    #[test]
    fn decode_sums_layers_additively() {
        let (w, rvq) = fixture();
        // t=2. layer0 picks rows [0,1] -> [1,2],[1.5,2]; layer1 picks [1,0] -> [2,2],[1,1]
        let codes = vec![vec![0u32, 1], vec![1u32, 0]];
        let y = rvq.decode(&w, &codes, DType::F32).unwrap();
        assert_eq!(y.dims(), &[1, 2, 2]); // [1, channels, t]
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        // channel-major: ch0 = [1+2, 1.5+1] = [3, 2.5]; ch1 = [2+2, 2+1] = [4, 3]
        assert_eq!(v, vec![3.0, 2.5, 4.0, 3.0]);
    }

    #[test]
    fn rejects_out_of_range_codes_and_ragged_rows() {
        let (w, rvq) = fixture();
        let e = rvq.decode(&w, &[vec![0u32], vec![9u32]], DType::F32).unwrap_err();
        assert!(e.to_string().contains("out of range"), "{e}");
        let e = rvq.decode(&w, &[vec![0u32, 1], vec![0u32]], DType::F32).unwrap_err();
        assert!(e.to_string().contains("expected"), "{e}");
    }
}
