//! The CV3 front-end (token -> mu): the speaker projection, the `Embedding(6561,80)`
//! lookup, the `PreLookaheadLayer`, and the `repeat_interleave` that builds `mu`. Moved
//! verbatim from `real_cv3.rs`.

use super::*;

impl Cv3Flow {
    // ======================== FRONT-END (token -> mu) ========================

    /// xvec projection: L2-normalize over dim 1, then affine 192 -> 80. `[1,192]->[1,80]`.
    pub fn spk_proj(&self, embedding: &Tensor) -> Result<Tensor> {
        debug_assert_eq!(embedding.dim(1).unwrap_or(SPK_DIM), SPK_DIM);
        let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normed = embedding.broadcast_div(&norm)?;
        self.linear(&normed, "spk_embed_affine_layer.weight", Some("spk_embed_affine_layer.bias"))
    }

    /// Token -> input embedding `[1, T, 80]` (single-utterance: mask all-ones, plain
    /// lookup of `Embedding(6561, 80)` after `clamp(min=0)`).
    pub fn input_embedding(&self, token: &Tensor) -> Result<Tensor> {
        let table = self.g("input_embedding.weight")?; // [6561, 80]
        let (b, t) = token.dims2()?;
        // clamp(min=0): tokens are already non-negative ids; clamp for fidelity.
        let idx = token
            .to_dtype(DType::I64)?
            .clamp(0i64, (VOCAB - 1) as i64)?
            .reshape((b * t,))?;
        let emb = table.index_select(&idx, 0)?; // [b*t, 80]
        emb.reshape((b, t, MEL))
    }

    /// `PreLookaheadLayer` (inference, no streaming context). Input/out `[1, T, 80]`.
    ///
    /// transpose -> pad RIGHT `pre_lookahead_len` (=3) -> conv1 (k=4, 80->1024) ->
    /// leaky_relu(0.01) -> pad LEFT (k2-1)=2 -> conv2 (k=3, 1024->80) -> transpose ->
    /// + residual.
    pub fn pre_lookahead(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?; // [1, 80, T]
        let padded = pad_time(&xt, 0, PRE_LOOKAHEAD)?; // pad right by 3
        let c1 = conv1d(
            &padded,
            &self.g("pre_lookahead_layer.conv1.weight")?,
            Some(&self.g("pre_lookahead_layer.conv1.bias")?),
            1,
            1,
        )?;
        let c1 = leaky_relu(&c1, 0.01)?;
        let padded2 = pad_time(&c1, 2, 0)?; // pad left by k2-1 = 2
        let c2 = conv1d(
            &padded2,
            &self.g("pre_lookahead_layer.conv2.weight")?,
            Some(&self.g("pre_lookahead_layer.conv2.bias")?),
            1,
            1,
        )?;
        let out = c2.transpose(1, 2)?.contiguous()?; // [1, T, 80]
        out + x
    }

    /// Full token -> mu front-end: `input_embedding -> pre_lookahead ->
    /// repeat_interleave(2, time) -> transpose`. Returns `mu` `[1, 80, 2T]`.
    pub fn token_to_mu(&self, token: &Tensor) -> Result<Tensor> {
        let emb = self.input_embedding(token)?; // [1, T, 80]
        let h_pre = self.pre_lookahead(&emb)?; // [1, T, 80]
        let h = repeat_interleave_time(&h_pre, TOKEN_MEL_RATIO)?; // [1, 2T, 80]
        h.transpose(1, 2)?.contiguous() // mu [1, 80, 2T]
    }
}

/// Repeat each time step `factor` times along dim 1: `[B,T,C] -> [B,factor*T,C]`.
/// (`torch.repeat_interleave(factor, dim=1)`: `[f0,f0,f1,f1,...]`.)
fn repeat_interleave_time(x: &Tensor, factor: usize) -> Result<Tensor> {
    let t = x.dim(1)?;
    let mut idx: Vec<u32> = Vec::with_capacity(t * factor);
    for p in 0..t {
        for _ in 0..factor {
            idx.push(p as u32);
        }
    }
    let idx = Tensor::from_vec(idx, (t * factor,), x.device())?;
    x.index_select(&idx, 1)
}

fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = (x - &pos)?; // == min(x,0)
    pos + (neg * slope)?
}
